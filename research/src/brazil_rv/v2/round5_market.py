"""Bounded public market archives and their decision-time information sets."""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END

SAO_PAULO = ZoneInfo("America/Sao_Paulo")
NEW_YORK = ZoneInfo("America/New_York")
FIRST_DATE = date(2010, 1, 4)
TREASURY_FIELDS = {
    "BC_3MONTH": "us_treasury_3m",
    "BC_2YEAR": "us_treasury_2y",
    "BC_5YEAR": "us_treasury_5y",
    "BC_10YEAR": "us_treasury_10y",
}
OBSERVATION_SCHEMA = {
    "series": pl.String,
    "reference_date": pl.Date,
    "available_at": pl.Datetime("us", "UTC"),
    "value": pl.Float64,
    "source_file": pl.String,
}


def first_available_decision(available_at: datetime, sessions: list[date]) -> int:
    """First 15:45 decision >= availability; never add a second session lag."""
    if available_at.tzinfo is None:
        raise ValueError("availability requires a source timezone")
    local = available_at.astimezone(SAO_PAULO)
    position = int(np.searchsorted(sessions, local.date()))
    if (
        position < len(sessions)
        and sessions[position] == local.date()
        and local.time().replace(tzinfo=None) > time(15, 45)
    ):
        position += 1
    return position


def decision_snapshots(
    observations: pl.DataFrame, sessions: list[date]
) -> pl.DataFrame:
    """Known market levels, with source age; an archive holiday is not a new print.

    Each event becomes visible at its first eligible decision. Later delivery of
    an older reference date cannot replace a more recent known market level.
    Untimed observations are excluded, rather than assigned an invented lag.
    """
    output = []
    for (series,), frame in observations.partition_by("series", as_dict=True).items():
        events = []
        for row in frame.filter(pl.col("available_at").is_not_null()).iter_rows(
            named=True
        ):
            position = first_available_decision(row["available_at"], sessions)
            if position < len(sessions):
                events.append((position, row["available_at"], row))
        events.sort(key=lambda event: (event[0], event[1], event[2]["reference_date"]))
        cursor = 0
        current = None
        known_since = 0
        for index, session in enumerate(sessions):
            while cursor < len(events) and events[cursor][0] <= index:
                position, _, row = events[cursor]
                if (
                    current is None
                    or row["reference_date"] >= current["reference_date"]
                ):
                    current = row
                    known_since = position
                cursor += 1
            if current is not None:
                output.append(
                    {
                        "date": session,
                        "series": series,
                        "value": current["value"],
                        "reference_date": current["reference_date"],
                        "available_at": current["available_at"],
                        "age_sessions": index - known_since,
                    }
                )
    return pl.DataFrame(
        output,
        schema={
            "date": pl.Date,
            "series": pl.String,
            "value": pl.Float64,
            "reference_date": pl.Date,
            "available_at": pl.Datetime("us", "UTC"),
            "age_sessions": pl.Int32,
        },
    ).sort("series", "date")


def _get(path: Path, url: str) -> dict:
    """Cache successful raw responses without modifying an existing source."""
    record_path = path.with_suffix(path.suffix + ".source.json")
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["url"] != url or record["sha256"] != sha256_file(path):
            raise ValueError(f"cached source identity differs: {path}")
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 Brazil-RV research"}
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type")
    if not payload:
        raise ValueError(f"empty source response: {url}")
    with path.open("xb") as output:
        output.write(payload)
    record = {
        "path": str(path.resolve()),
        "url": url,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "content_type": content_type,
        "bytes": len(payload),
        "sha256": sha256_file(path),
    }
    write_json_atomic(record_path, record)
    return record


def parse_ptax(payload: bytes, source: str) -> list[dict]:
    rows = []
    for item in json.loads(payload)["value"]:
        stamp = datetime.fromisoformat(item["dataHoraCotacao"]).replace(
            tzinfo=SAO_PAULO
        )
        # The source can report only a rounded minute. Its upper bound still
        # admits a 15:44 source timestamp at the 15:45 decision.
        available = stamp + timedelta(minutes=1)
        rows.append(
            {
                "series": "ptax_brl_per_usd",
                "reference_date": stamp.date(),
                "available_at": available.astimezone(UTC),
                "value": float(item["cotacaoVenda"]),
                "source_file": source,
            }
        )
    return rows


def parse_treasury(payload: bytes, source: str) -> list[dict]:
    rows = []
    for entry in ET.fromstring(payload).findall("{*}entry"):
        properties = entry.find("{*}content/{*}properties")
        if properties is None:
            continue
        fields = {child.tag.split("}")[-1]: child.text for child in properties}
        reference = date.fromisoformat(fields["NEW_DATE"][:10])
        # Treasury identifies the underlying quote snapshot as near 15:30 NY.
        # Every such snapshot is after the Brazil 15:45 decision in this era.
        available = datetime.combine(reference, time(15, 30), NEW_YORK)
        for field, series in TREASURY_FIELDS.items():
            value = fields.get(field)
            if value is not None:
                rows.append(
                    {
                        "series": series,
                        "reference_date": reference,
                        "available_at": available.astimezone(UTC),
                        "value": float(value) / 100.0,
                        "source_file": source,
                    }
                )
    return rows


def parse_fred(payload: bytes, source: str, series: str) -> list[dict]:
    rows = []
    for item in csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))):
        if "observation_date" not in item or series not in item:
            raise ValueError("FRED source is not the requested historical CSV")
        value = item[series]
        if not value or value == ".":
            continue
        reference = date.fromisoformat(item["observation_date"])
        # Brent's precise assessment timestamp remains a source-audit question.
        # Retrieve it now without pretending a guessed time proves availability.
        available = (
            datetime.combine(reference, time(16, 15), NEW_YORK).astimezone(UTC)
            if series == "VIXCLS"
            else None
        )
        rows.append(
            {
                "series": "vix_close" if series == "VIXCLS" else "brent_spot",
                "reference_date": reference,
                "available_at": available,
                "value": float(value),
                "source_file": source,
            }
        )
    return rows


def acquire(root: Path, *, workers: int = 4) -> dict:
    raw = root / "raw"
    jobs = []
    for year in range(FIRST_DATE.year, DEVELOPMENT_END.year + 1):
        start = max(FIRST_DATE, date(year, 1, 1))
        end = min(DEVELOPMENT_END, date(year, 12, 31))
        ptax = (
            "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
            "CotacaoDolarPeriodo(dataInicial=@dataInicial,"
            "dataFinalCotacao=@dataFinalCotacao)?"
            + urllib.parse.urlencode(
                {
                    "@dataInicial": start.strftime("'%m-%d-%Y'"),
                    "@dataFinalCotacao": end.strftime("'%m-%d-%Y'"),
                    "$format": "json",
                }
            )
        )
        treasury = (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xml?data=daily_treasury_yield_curve"
            f"&field_tdr_date_value={year}"
        )
        jobs.extend(
            [
                ("ptax", raw / f"ptax_{year}.json", ptax),
                ("treasury", raw / f"treasury_{year}.xml", treasury),
            ]
        )
    for series in ("VIXCLS", "DCOILBRENTEU"):
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv?"
            + urllib.parse.urlencode(
                {
                    "id": series,
                    "cosd": FIRST_DATE.isoformat(),
                    "coed": DEVELOPMENT_END.isoformat(),
                }
            )
        )
        jobs.append((series, raw / f"{series}.csv", url))

    def download(job: tuple[str, Path, str]) -> tuple[str, Path, dict]:
        kind, path, url = job
        try:
            return kind, path, {"status": "retrieved", **_get(path, url)}
        except (OSError, ValueError) as error:
            return (
                kind,
                path,
                {"status": "unavailable", "url": url, "error": str(error)},
            )

    records = []
    observations = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for kind, path, record in executor.map(download, jobs):
            records.append(record)
            if record["status"] != "retrieved":
                continue
            if kind == "ptax":
                parsed = parse_ptax(path.read_bytes(), str(path))
            elif kind == "treasury":
                parsed = parse_treasury(path.read_bytes(), str(path))
            else:
                parsed = parse_fred(path.read_bytes(), str(path), kind)
            # Treasury's annual source can contain 12/31/2024, beyond the B3
            # endpoint; bound it before any feature/output consumer sees a row.
            if any(row["reference_date"].year > 2024 for row in parsed):
                raise PermissionError(
                    "date-bounded source returned held-out observations"
                )
            observations.extend(
                row
                for row in parsed
                if FIRST_DATE <= row["reference_date"] <= DEVELOPMENT_END
            )
    frame = pl.DataFrame(observations, schema=OBSERVATION_SCHEMA).sort(
        "series", "reference_date", "available_at"
    )
    target = root / "market_observations.parquet"
    if target.exists():
        raise FileExistsError(target)
    frame.write_parquet(target)
    coverage = frame.group_by("series").agg(
        pl.len().alias("observations"),
        pl.col("reference_date").min().alias("first_reference_date"),
        pl.col("reference_date").max().alias("last_reference_date"),
        pl.col("available_at").is_not_null().sum().alias("timed_observations"),
    )
    result = {
        "schema": "BRAZIL_RV_ROUND5_MARKET_SOURCES_V1",
        "status": "acquired_pending_joined_availability_proofs",
        "sources": records,
        "coverage": json.loads(coverage.write_json()),
        "data": {
            "path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        },
        "brent_availability": "source-semantics pending: daily Refinitiv spot close; precise fixing/publication not established; EIA archive upload lag is not applied",
        "ptax_availability": "actual recorded dataHoraCotacao in historical America/Sao_Paulo plus minute-resolution upper bound; pre-July-2011 closes are not assumed to be lunchtime fixings",
        "treasury_availability": "underlying indicative quote snapshot near15:30 America/New_York; first subsequent B3 15:45 decision",
        "vix_availability": "previous completed US index close; no archive posting lag",
        "evidence": [
            "https://www.bcb.gov.br/pre/normativos/circ/2011/pdf/circ_3537_v1_O.pdf",
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?field_tdr_date_value=2024&type=daily_treasury_yield_curve",
            "https://www.cboe.com/tradable_products/vix/vix_historical_data",
            "https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp",
        ],
    }
    write_json_atomic(root / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = acquire(args.root, workers=args.workers)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"]}))


if __name__ == "__main__":
    main()
