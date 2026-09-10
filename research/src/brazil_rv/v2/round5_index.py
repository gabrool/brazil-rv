"""Published index snapshots and a dated preview-minus-last-effective proxy."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import numpy as np
import polars as pl
import fastexcel

from ..preprocessing.index_rebalance import parse_composition
from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END
from .round5_market import SAO_PAULO, first_available_decision

PROXY = "preview_minus_last_effective_weight_proxy_contains_price_drift"
PORTFOLIO_SCHEMA = {
    "disclosure_date": pl.Date,
    "effective_date": pl.Date,
    "stage": pl.String,
    "available_at": pl.Datetime("us", "UTC"),
    "index": pl.String,
    "ticker": pl.String,
    "weight_fraction": pl.Float64,
    "quantity": pl.Float64,
    "source_file": pl.String,
}


def linked_assets(page: str, page_url: str) -> set[str]:
    """Resolve only actual announcement links, including mail safe-link wrappers."""
    output = set()
    for link in re.findall(r'href=["\']([^"\']+)["\']', page, re.I):
        link = urljoin(page_url, html.unescape(link))
        wrapped = parse_qs(urlparse(link).query).get("url")
        if wrapped:
            link = wrapped[0]
        output.add(unquote(link).lower())
    return output


def publication_date(page: str) -> date:
    values = re.findall(r"<small[^>]*>\s*(\d{2}/\d{2}/\d{4})\s*</small>", page)
    if len(set(values)) != 1:
        raise ValueError("announcement has no unambiguous publication date")
    return datetime.strptime(values[0], "%d/%m/%Y").date()


def parse_bdi_tables(pages: list[str]) -> tuple[dict, dict]:
    """Parse only the complete IBOV/IBXX/SMLL sections, preserving phase."""
    tables, opening_dates = {}, {}
    phase = active = None
    for number, text in enumerate(pages, 1):
        if "Composição das Carteiras de Índices" in text:
            phase = "effective"
        if "Prévia das Carteiras Teóricas de Índices" in text:
            phase = "preview"
        if phase:
            for value in re.findall(
                r"abertura dos negócios do dia\s+(\d{2}/\d{2}/\d{4})", text
            ):
                opening_dates.setdefault(phase, set()).add(value)
        for line in text.splitlines():
            line = line.strip()
            if re.fullmatch(r"[A-Z][A-Z0-9 -]{1,16}", line) and line != "BDI":
                active = {
                    "IBOVESPA": "IBOV",
                    "IBXX": "IBXX",
                    "IBRX": "IBXX",
                    "SMLL": "SMLL",
                }.get(line)
                if phase and active:
                    tables.setdefault((phase, active), {})
            if line.startswith("Participação total:"):
                active = None
            if not (phase and active):
                continue
            match = re.fullmatch(
                r"([A-Z0-9]{4}\d{1,2})\s+.+\s+([\d.,]+)\s+([\d.,]+)", line
            )
            if match:
                ticker, quantity, weight = match.groups()
                table = tables[phase, active]
                if ticker in table:
                    raise ValueError(
                        f"duplicate BDI index constituent: {phase}/{active}/{ticker}"
                    )
                table[ticker] = {
                    "weight_fraction": float(weight.replace(",", ".")) / 100,
                    "quantity": float(quantity.replace(",", "").replace(".", "")),
                    "page": number,
                }
    return tables, {key: sorted(values) for key, values in opening_dates.items()}


def _verify(source: dict) -> Path:
    path = Path(source["path"])
    if sha256_file(path) != source["sha256"]:
        raise ValueError(f"source hash mismatch: {path}")
    return path


def normalize_events(events: list[dict]) -> tuple[pl.DataFrame, list[dict]]:
    """Source dates and exact BDI weights establish the first decision bound.

    Current BDI tables are not used to claim drift-adjusted weights. Publication
    dates are verified from official announcement HTML. HTTP file timestamps
    never establish first availability. The BDI-derived opening bound requires
    all three complete weight vectors to match the linked composition workbook.
    """
    rows, audit = [], []
    for event in sorted(events, key=lambda row: row["disclosure_date"]):
        day = date.fromisoformat(event["disclosure_date"])
        if day > DEVELOPMENT_END:
            raise PermissionError("index source event enters held-out history")
        page = _verify(event["page"]).read_text("utf8")
        if publication_date(page) != day:
            raise ValueError("announcement date differs from registered event")
        links = linked_assets(page, event["page"]["url"])
        candidates = []
        for source in event["assets"]:
            if unquote(source["url"]).lower() not in links:
                raise ValueError("composition source is not linked by its announcement")
            path = _verify(source)
            try:
                portfolios = {item.index: item for item in parse_composition(path)}
            except (ValueError, KeyError, fastexcel.SheetNotFoundError):
                continue  # Entries/exits attachments are not full portfolios.
            candidates.append((source, portfolios))
        if len(candidates) != 1:
            raise ValueError(f"expected one complete composition attachment: {day}")
        source, portfolios = candidates[0]
        for identity_evidence in event.get("identity_publication_evidence", []):
            _verify(identity_evidence["source"])
            if date.fromisoformat(identity_evidence["publication_date"]) >= day:
                raise ValueError(
                    "identity announcement does not precede the source day"
                )
        bdi_path = _verify(event["bdi"])
        text_path = bdi_path.with_suffix(".pages.json")
        extracted = json.loads(text_path.read_text("utf8"))
        pages = [item["text"] if isinstance(item, dict) else item for item in extracted]
        tables, opening = parse_bdi_tables(pages)
        phase = "effective" if event["stage"] == "effective" else "preview"
        differences = {}
        for index, portfolio in portfolios.items():
            table = tables.get((phase, index), {})
            differences[index] = sum(
                abs(
                    portfolio.weights.get(ticker, -1)
                    - table.get(ticker, {}).get("weight_fraction", -2)
                )
                > 1e-10
                for ticker in set(portfolio.weights) | set(table)
            )
        exact = not any(differences.values())
        opening_days = [
            datetime.strptime(value, "%d/%m/%Y").date()
            for value in opening.get(phase, [])
        ]
        at_open = exact and len(set(opening_days)) == 1 and opening_days[0] <= day
        # A pre-effective opening footnote is not permission to backdate the
        # release. Date-only evidence gets a safe, decision-equivalent day end.
        available = datetime.combine(
            day, time(10) if at_open else time(23, 59, 59), SAO_PAULO
        ).astimezone(UTC)
        for index, portfolio in portfolios.items():
            for ticker, weight in portfolio.weights.items():
                rows.append(
                    {
                        "disclosure_date": day,
                        "effective_date": date.fromisoformat(event["effective_date"]),
                        "stage": event["stage"],
                        "available_at": available,
                        "index": index,
                        "ticker": ticker,
                        "weight_fraction": weight,
                        "quantity": portfolio.quantities[ticker],
                        "source_file": source["path"],
                    }
                )
        audit.append(
            {
                **event,
                "composition_source": source,
                "bdi_weight_differences": differences,
                "bdi_opening_dates": opening,
                "bdi_text_sha256": sha256_file(text_path),
                "available_at": available.isoformat(),
                "availability_evidence": "dated_bdi_exact_weights_disclosed_for_open"
                if at_open
                else "official_announcement_date_only_end_of_day_bound",
                "source_revision_share": None,
            }
        )
    return pl.DataFrame(rows, schema=PORTFOLIO_SCHEMA), audit


def _identity_at(cash: pl.DataFrame, day: date) -> dict[str, str]:
    """Latest observed assignment per ticker, using no later source row."""
    past = cash.filter(pl.col("source_trade_date") <= day)
    latest = past.group_by("ticker").agg(pl.col("source_trade_date").max())
    dated = past.join(latest, on=["ticker", "source_trade_date"])
    unique = dated.group_by("ticker").agg(pl.col("isin").unique())
    return {
        row["ticker"]: row["isin"][0]
        for row in unique.iter_rows(named=True)
        if len(row["isin"]) == 1
    }


def pressure_panel(
    portfolios: pl.DataFrame,
    cash: pl.DataFrame,
    sessions: list[date],
    isins: tuple[str, ...],
    active: np.ndarray,
    prior_adv20: np.ndarray,
    *,
    future_sessions: list[date] = (),
) -> tuple[pl.DataFrame, list[dict]]:
    """Two native fields on active dated ISINs, no post-effective signal.

    Every snapshot binds its own ticker identities as of publication. Ticker
    renames with the same ISIN combine correctly. An unmapped constituent masks
    its entire event: a complete three-index aggregate cannot assign that
    weight to a security safely. Unknown history is not zero membership.
    """
    all_sessions = sorted(set(sessions) | set(future_sessions))
    identities, effective, events, identity_audit = {}, {}, [], []
    groups = portfolios.partition_by(["disclosure_date", "stage"], as_dict=True)
    for (day, stage), group in sorted(groups.items()):
        effective_day = group.item(0, "effective_date")
        available = group.item(0, "available_at")
        identities.setdefault(day, _identity_at(cash, day))
        previous_identity = _identity_at(cash, day - timedelta(days=1))
        weights, unmapped = {}, []
        for row in group.iter_rows(named=True):
            isin = identities[day].get(row["ticker"])
            if isin is None:
                unmapped.append({"index": row["index"], "ticker": row["ticker"]})
            else:
                weights[row["index"], isin] = (
                    weights.get((row["index"], isin), 0) + row["weight_fraction"]
                )
        identity_audit.append(
            {
                "date": day.isoformat(),
                "stage": stage,
                "unmapped": unmapped,
                "same_day_assignment_changes": {
                    ticker: identities[day].get(ticker)
                    for ticker in group["ticker"].unique()
                    if identities[day].get(ticker) != previous_identity.get(ticker)
                },
            }
        )
        if stage == "effective":
            effective[effective_day] = (available, weights, unmapped)
            continue
        previous_days = [
            value
            for value, record in effective.items()
            if value < effective_day and value <= day and record[0] <= available
        ]
        prior = effective[max(previous_days)] if previous_days else None
        known = prior is not None and not unmapped and not prior[2]
        delta = {}
        if known:
            for index, isin in set(weights) | set(prior[1]):
                delta[isin] = (
                    delta.get(isin, 0)
                    + weights.get((index, isin), 0)
                    - prior[1].get((index, isin), 0)
                )
        events.append(
            (
                first_available_decision(available, sessions),
                effective_day,
                delta,
                known,
                day,
            )
        )
    rows = []
    for start, effective_day, delta, known, source_day in events:
        if not known:
            continue
        if effective_day not in all_sessions:
            raise ValueError(
                "announced effective date requires a verified session calendar"
            )
        end = int(np.searchsorted(sessions, effective_day))
        next_starts = [
            s for s, e, _, _, d in events if e == effective_day and d > source_day
        ]
        if next_starts:
            end = min(end, min(next_starts))
        for day_index in range(start, min(end, len(sessions))):
            remaining = int(
                np.searchsorted(all_sessions, effective_day)
                - np.searchsorted(all_sessions, sessions[day_index])
            )
            for name_index in np.flatnonzero(active[day_index]):
                adv = prior_adv20[day_index, name_index]
                if not np.isfinite(adv) or adv <= 0:
                    continue
                isin = isins[name_index]
                rows.append(
                    {
                        "date": sessions[day_index],
                        "isin": isin,
                        "index_pressure": 100
                        * delta.get(isin, 0)
                        * remaining
                        / (adv / 1e6),
                        "index_event_age": float(day_index - start),
                    }
                )
    panel = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "isin": pl.String,
            "index_pressure": pl.Float64,
            "index_event_age": pl.Float64,
        },
    )
    return panel.with_columns(
        pl.col("index_event_age").alias("index_pressure_age_sessions"),
        pl.col("index_event_age").alias("index_event_age_age_sessions"),
    ), identity_audit


def build_features(
    snapshots_manifest: Path,
    store_root: Path,
    cash_path: Path,
    calendar_path: Path,
    output: Path,
) -> dict:
    from .build_store import _prior_adv20
    from .contract import FINETUNE_START, PRETRAIN_END
    from .research_rounds import _git_identity
    from .store import open_store_for_samples

    code = _git_identity()
    if output.exists():
        raise FileExistsError(output)
    snapshots = json.loads(snapshots_manifest.read_text("utf8"))
    portfolios = pl.read_parquet(_verify(snapshots["output"]))
    manifest = json.loads((store_root / "manifest.json").read_text("utf8"))
    if manifest["axes"]["date_end"] > DEVELOPMENT_END.isoformat():
        raise PermissionError("index consumer store extends into held-out history")
    dates = np.load(store_root / "date_index.npy", allow_pickle=False)
    indices = np.arange(len(dates))
    samples = indices[
        (dates <= np.datetime64(PRETRAIN_END))
        | (dates >= np.datetime64(FINETUNE_START))
    ]
    store, access = open_store_for_samples(
        store_root,
        samples,
        purpose="training",
        history_lookbacks=60,
        history_end_offsets=0,
    )
    try:
        active = store.read("active", indices)
        adv = _prior_adv20(
            store.read("volume_brl", indices), store.read("activity_valid", indices)
        )
        isins = store.isins
    finally:
        store.close()
    cash = pl.read_parquet(cash_path, columns=["source_trade_date", "isin", "ticker"])
    if cash["source_trade_date"].max() > DEVELOPMENT_END:
        raise PermissionError("cash identity source extends into held-out history")
    calendar = json.loads(calendar_path.read_text("utf8"))
    _verify(calendar["source"])
    calendar_known = date.fromisoformat(calendar["publication_date"])
    future_sessions = [date.fromisoformat(value) for value in calendar["sessions"]]
    needed = portfolios.filter(pl.col("effective_date") > DEVELOPMENT_END)
    if not needed.is_empty() and calendar_known >= needed["disclosure_date"].min():
        raise ValueError(
            "future calendar was not known before its first consumer preview"
        )
    sessions = dates.astype(object).tolist()
    panel, identity_audit = pressure_panel(
        portfolios, cash, sessions, isins, active, adv, future_sessions=future_sessions
    )
    # Actual joined-output future mutation, not just a pure-function fixture.
    cutoff = date(2024, 4, 16)
    mutation = portfolios.with_columns(
        pl.when(pl.col("disclosure_date") > cutoff)
        .then(pl.col("weight_fraction") * 7)
        .otherwise(pl.col("weight_fraction"))
        .alias("weight_fraction")
    )
    bounded_cash = cash.filter(pl.col("source_trade_date") <= cutoff)
    mutated, _ = pressure_panel(
        mutation,
        bounded_cash,
        sessions,
        isins,
        active,
        adv,
        future_sessions=future_sessions,
    )
    announced = {
        (event["disclosure_date"], ticker)
        for event in snapshots["events"]
        for evidence in event.get("identity_publication_evidence", [])
        for ticker in evidence["tickers"]
    }
    if any(
        (item["date"], ticker) not in announced
        for item in identity_audit
        for ticker in item["same_day_assignment_changes"]
    ):
        raise ValueError("same-day identity change lacks prior published announcement")
    causal = panel.filter(pl.col("date") <= cutoff).equals(
        mutated.filter(pl.col("date") <= cutoff)
    )
    if not causal:
        raise ValueError("future mutation changed an earlier index consumer value")
    # Move the first preview to just after its decision and prove exactly that
    # day's rows disappear, with the next day's same economic value retained.
    previews = portfolios.filter(pl.col("stage") != "effective")
    first = previews["disclosure_date"].min()
    first_rows = previews.filter(pl.col("disclosure_date") == first)
    first_position = first_available_decision(
        first_rows.item(0, "available_at"), sessions
    )
    first_decision_day = sessions[first_position]
    delayed_time = datetime.combine(
        first_decision_day, time(15, 46), SAO_PAULO
    ).astimezone(UTC)
    delayed = portfolios.with_columns(
        pl.when(pl.col("disclosure_date") == first)
        .then(delayed_time)
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    later, _ = pressure_panel(
        delayed, cash, sessions, isins, active, adv, future_sessions=future_sessions
    )
    first_check = (
        panel.filter(pl.col("date") == first_decision_day).height > 0
        and later.filter(pl.col("date") == first_decision_day).is_empty()
    )
    tomorrow = sessions[first_position + 1]
    compare = ["date", "isin", "index_pressure"]
    first_check &= (
        panel.filter(pl.col("date") == tomorrow)
        .select(compare)
        .equals(later.filter(pl.col("date") == tomorrow).select(compare))
    )
    if not first_check:
        raise ValueError("first-available index decision mutation failed")
    output.mkdir(parents=True)
    path = output / "index_features.parquet"
    panel.write_parquet(path)
    data = {"path": str(path), "sha256": sha256_file(path), "rows": panel.height}
    proof = {
        "causality_passed": causal,
        "first_available_decision_passed": first_check,
        "data_sha256": data["sha256"],
        "future_mutation_cutoff": cutoff.isoformat(),
        "first_decision_mutation": first_decision_day.isoformat(),
        "future_price_payload_read": False,
        "access": access.payload(),
    }
    write_json_atomic(output / "availability_proof.json", proof)
    result = {
        "schema": "BRAZIL_RV_ROUND5_INDEX_FEATURES_V1",
        "code_identity": code,
        "proxy": PROXY,
        "status": "admitted",
        "data": data,
        "feature_names": ["index_pressure", "index_event_age"],
        "snapshots": {
            "path": str(snapshots_manifest),
            "sha256": sha256_file(snapshots_manifest),
        },
        "base_store": {
            "path": str(store_root / "manifest.json"),
            "sha256": sha256_file(store_root / "manifest.json"),
        },
        "cash_identity": {"path": str(cash_path), "sha256": sha256_file(cash_path)},
        "calendar": {"path": str(calendar_path), "sha256": sha256_file(calendar_path)},
        "identity_audit": identity_audit,
        "coverage": json.loads(
            panel.group_by(pl.col("date").dt.year().alias("year"))
            .agg(
                pl.len(),
                pl.col("date").n_unique().alias("sessions"),
                pl.col("isin").n_unique().alias("securities"),
            )
            .write_json()
        ),
        "availability_proof": {
            "path": str(output / "availability_proof.json"),
            "sha256": sha256_file(output / "availability_proof.json"),
        },
    }
    write_json_atomic(output / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["normalize", "build"], nargs="?", default="normalize"
    )
    parser.add_argument("--events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots-manifest", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--cash", type=Path)
    parser.add_argument("--calendar", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        if any(
            value is None
            for value in (args.snapshots_manifest, args.store, args.cash, args.calendar)
        ):
            parser.error("build requires snapshots-manifest, store, cash and calendar")
        print(
            json.dumps(
                build_features(
                    args.snapshots_manifest,
                    args.store,
                    args.cash,
                    args.calendar,
                    args.output,
                )["coverage"]
            )
        )
        return
    if args.events is None:
        parser.error("normalize requires events")
    from .research_rounds import _git_identity

    code = _git_identity()
    if args.output.exists():
        raise FileExistsError(args.output)
    events = json.loads(args.events.read_text("utf8"))
    frame, audit = normalize_events(events)
    args.output.mkdir(parents=True)
    path = args.output / "index_portfolios.parquet"
    frame.write_parquet(path)
    write_json_atomic(
        args.output / "manifest.json",
        {
            "schema": "BRAZIL_RV_ROUND5_INDEX_SNAPSHOTS_V1",
            "code_identity": code,
            "proxy": PROXY,
            "events_source": {
                "path": str(args.events),
                "sha256": sha256_file(args.events),
            },
            "output": {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": frame.height,
            },
            "events": audit,
        },
    )


if __name__ == "__main__":
    main()
