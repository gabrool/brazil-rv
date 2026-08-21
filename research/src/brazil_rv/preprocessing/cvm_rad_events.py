from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import html
import json
import math
import re
import shutil
import tempfile
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .contract import DECISION_TIMES

CONTRACT_VERSION = "CVM_RAD_INTRADAY_EVENT_STATE_V1"
RAW_CONTRACT_VERSION = "CVM_RAD_EXACT_DELIVERY_RESPONSES_V1"
RAD_PAGE = "https://www.rad.cvm.gov.br/ENETWeb/frmConsultaExternaCVM.aspx"
RAD_LIST_ENDPOINT = f"{RAD_PAGE}/ListarDocumentos"
RAD_TIMEZONE = "America/Sao_Paulo"
QUERY_GROUPS = {
    "structured": ("EST_3", "EST_4"),
    "material_fact": ("IPE_4_-1_-1",),
    "market_communication": ("IPE_6_-1_-1",),
    "shareholder_notice": ("IPE_3_-1_-1",),
}
CATEGORY_GROUPS = {
    "ITR - Informações Trimestrais": "itr_dfp",
    "DFP - Demonstrações Financeiras Padronizadas": "itr_dfp",
    "Fato Relevante": "material_fact",
    "Comunicado ao Mercado": "market_communication",
    "Aviso aos Acionistas": "corporate_action",
}
EVENT_GROUPS = (
    "itr_dfp",
    "material_fact",
    "market_communication",
    "corporate_action",
)
RECENT_SESSION_COUNT = 5
AGE_CLIP_SESSIONS = 20
DECISIONS_PER_SESSION = len(DECISION_TIMES)
FEATURES = (
    "event_itr_dfp_recent_5s",
    "event_material_fact_recent_5s",
    "event_market_communication_recent_5s",
    "event_corporate_action_recent_5s",
    "event_latest_log_trading_minutes_20s",
)
_TAG = re.compile(r"<[^>]+>")
_DELIVERY = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})")
_REFERENCE = re.compile(r"(\d{2}/\d{2}/\d{4})")
_ENET_SEQUENCE = re.compile(r"NumeroSequencialDocumento=(\d+)")
_IPE_SEQUENCE = re.compile(r"OpenDownloadDocumentos\('(\d+)'")
_IPE_PROTOCOL = re.compile(r"NumeroProtocoloEntrega=(\d+)")
_CNPJ = re.compile(r"\D")
_ALLOWED_SECURITIES = {"Ações Ordinárias", "Ações Preferenciais", "Units"}


class CaptchaRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class RadEvent:
    cvm_code: str
    company_name: str
    category: str
    event_group: str
    subtype: str
    subject: str
    reference_date: date | None
    delivery_timestamp: datetime
    status: str
    version: str
    presentation: str
    sequence_id: str
    protocol_id: str | None
    source_file: str
    cnpj: str | None = None
    available_date: date | None = None
    decision_idx: int | None = None


@dataclass(frozen=True)
class FcaSecurity:
    cnpj: str
    ticker: str
    known_date: date
    effective_start: date
    effective_end: date | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_field(value: str) -> str:
    value = html.unescape(_TAG.sub("", value)).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return "" if value == "-" else value.removesuffix(" -").strip()


def _digits(value: str) -> str:
    return _CNPJ.sub("", value)


def _parse_iso_date(value: str) -> date | None:
    value = value.strip()
    return date.fromisoformat(value) if value else None


def _parse_reference(value: str) -> date | None:
    match = _REFERENCE.search(_clean_field(value))
    return datetime.strptime(match.group(1), "%d/%m/%Y").date() if match else None


def parse_rad_data(data: str, source_file: str) -> list[RadEvent]:
    events: list[RadEvent] = []
    for raw_row in data.split("$&&*"):
        if not raw_row.strip():
            continue
        fields = raw_row.split("$&")
        if len(fields) < 11:
            raise ValueError(f"RAD row has {len(fields)} fields in {source_file}")
        category = _clean_field(fields[2])
        try:
            event_group = CATEGORY_GROUPS[category]
        except KeyError as error:
            raise ValueError(f"Unexpected RAD category {category!r}") from error
        delivery_match = _DELIVERY.search(_clean_field(fields[6]))
        if delivery_match is None:
            raise ValueError(
                f"RAD delivery timestamp lacks minute precision: {fields[6]}"
            )
        delivered = datetime.strptime(
            " ".join(delivery_match.groups()), "%d/%m/%Y %H:%M"
        )
        action = fields[10]
        sequence = _ENET_SEQUENCE.search(action) or _IPE_SEQUENCE.search(action)
        if sequence is None:
            raise ValueError(f"RAD row lacks a document sequence key in {source_file}")
        protocol = _IPE_PROTOCOL.search(action)
        cvm_code = _digits(fields[0])
        if len(cvm_code) != 6:
            raise ValueError(f"Invalid formatted CVM code {fields[0]!r}")
        subject = _clean_field(fields[11] if len(fields) > 11 else fields[4])
        events.append(
            RadEvent(
                cvm_code=cvm_code,
                company_name=_clean_field(fields[1]),
                category=category,
                event_group=event_group,
                subtype=_clean_field(fields[3]),
                subject=subject,
                reference_date=_parse_reference(fields[5]),
                delivery_timestamp=delivered,
                status=_clean_field(fields[7]),
                version=_clean_field(fields[8]),
                presentation=_clean_field(fields[9]),
                sequence_id=sequence.group(1),
                protocol_id=protocol.group(1) if protocol else None,
                source_file=source_file,
            )
        )
    return events


def _rad_payload(
    start: date, end: date, categories: tuple[str, ...]
) -> dict[str, object]:
    return {
        "dataDe": start.strftime("%d/%m/%Y"),
        "dataAte": end.strftime("%d/%m/%Y"),
        "empresa": "",
        "setorAtividade": "-1",
        "categoriaEmissor": "-1",
        "situacaoEmissor": "-1",
        "tipoParticipante": "-1",
        "dataReferencia": "",
        "categoria": ",".join(categories),
        "periodo": "2",
        "horaIni": "",
        "horaFim": "",
        "palavraChave": "",
        "ultimaDtRef": "false",
        "tipoEmpresa": "0",
        "token": "",
        "versaoCaptcha": "",
    }


def _query_periods(year: int, query_group: str) -> tuple[tuple[str, date, date], ...]:
    if query_group != "market_communication":
        return (("annual", date(year, 1, 1), date(year, 12, 31)),)
    return (
        ("q1", date(year, 1, 1), date(year, 3, 31)),
        ("q2", date(year, 4, 1), date(year, 6, 30)),
        ("q3", date(year, 7, 1), date(year, 9, 30)),
        ("q4", date(year, 10, 1), date(year, 12, 31)),
    )


def _request(url: str, body: bytes | None = None, *, timeout: int = 180) -> bytes:
    headers = {"User-Agent": "Brazil-RV historical research acquisition/1.0"}
    if body is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(
        url, data=body, headers=headers, method="POST" if body else "GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _captcha_setting(page: bytes) -> str:
    text = page.decode("utf-8", errors="replace")
    tag = re.search(r"<input[^>]+id=[\"']hdnHabilitaCaptcha[\"'][^>]*>", text)
    if tag is None:
        raise ValueError("RAD page omitted hdnHabilitaCaptcha")
    value = re.search(r"value=[\"']([NS])[\"']", tag.group(0))
    if value is None:
        raise ValueError("RAD page CAPTCHA setting is malformed")
    return value.group(1)


def _write_acquisition_manifest(output_dir: Path, manifest: dict[str, object]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def acquire_rad_history(
    output_dir: Path,
    *,
    years: tuple[int, ...] = tuple(range(2019, 2025)),
    pause_seconds: float = 0.25,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    page = _request(RAD_PAGE)
    if _captcha_setting(page) != "N":
        raise CaptchaRequired(
            "RAD public page requires CAPTCHA; acquisition not started"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, object] = {
        "contract_version": RAW_CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "endpoint": RAD_LIST_ENDPOINT,
        "page_sha256": hashlib.sha256(page).hexdigest(),
        "captcha_bypass": False,
        "years": list(years),
        "query_groups": {key: list(value) for key, value in QUERY_GROUPS.items()},
        "market_communication_partition": "calendar_quarter",
        "responses": [],
    }
    _write_acquisition_manifest(output_dir, manifest)
    try:
        # Structured timestamps unblock the separate fundamentals sidecar, so
        # finish each category family across all years before moving to the next.
        for query_group, categories in QUERY_GROUPS.items():
            for year in years:
                for period, start, end in _query_periods(year, query_group):
                    payload = _rad_payload(start, end, categories)
                    request_bytes = json.dumps(payload, ensure_ascii=False).encode(
                        "utf-8"
                    )
                    raw = _request(RAD_LIST_ENDPOINT, request_bytes)
                    file_name = f"rad_{year}_{query_group}_{period}.json"
                    response_path = output_dir / file_name
                    response_path.write_bytes(raw)
                    response = json.loads(raw)
                    result = response.get("d")
                    if not isinstance(result, dict):
                        raise ValueError(
                            f"Malformed RAD response for {year}/{query_group}/{period}"
                        )
                    entry = {
                        "year": year,
                        "query_group": query_group,
                        "period": period,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "categories": list(categories),
                        "request": payload,
                        "response_file": file_name,
                        "response_sha256": _sha256(response_path),
                        "response_bytes": len(raw),
                        "row_count": len(
                            parse_rad_data(str(result.get("dados", "")), file_name)
                        ),
                        "solicitar_captcha": result.get("SolicitarCaptcha"),
                    }
                    manifest["responses"].append(entry)  # type: ignore[union-attr]
                    _write_acquisition_manifest(output_dir, manifest)
                    if result.get("SolicitarCaptcha") == "S":
                        raise CaptchaRequired(
                            f"RAD requested CAPTCHA at {year}/{query_group}/{period}; "
                            "stopped without bypass"
                        )
                    if result.get("temErro") or result.get("expirouSessao"):
                        raise RuntimeError(
                            f"RAD query failed at {year}/{query_group}/{period}: "
                            f"{result.get('msgErro')}"
                        )
                    time.sleep(pause_seconds)
    except BaseException as error:
        manifest["status"] = (
            "captcha_required" if isinstance(error, CaptchaRequired) else "failed"
        )
        manifest["error"] = str(error)
        _write_acquisition_manifest(output_dir, manifest)
        raise
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["response_count"] = len(manifest["responses"])
    manifest["row_count"] = sum(
        int(item["row_count"])
        for item in manifest["responses"]  # type: ignore[union-attr]
    )
    _write_acquisition_manifest(output_dir, manifest)
    return manifest


def seal_rad_history(raw_dir: Path) -> dict[str, object]:
    """Seal the predeclared core category scope after an interrupted expansion."""
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != RAW_CONTRACT_VERSION:
        raise ValueError("Unknown RAD acquisition contract")
    years = tuple(int(value) for value in manifest["years"])
    entries = [
        entry
        for entry in manifest["responses"]
        if entry.get("query_group") in QUERY_GROUPS
    ]
    expected = {
        (year, group, period)
        for group in QUERY_GROUPS
        for year in years
        for period, _, _ in _query_periods(year, group)
    }
    observed = {
        (int(entry["year"]), str(entry["query_group"]), str(entry["period"]))
        for entry in entries
    }
    if observed != expected or len(entries) != len(expected):
        raise ValueError(
            f"Core RAD acquisition is incomplete: {len(observed)}/{len(expected)}"
        )
    _parse_response_entries(raw_dir, entries)
    manifest["query_groups"] = {key: list(value) for key, value in QUERY_GROUPS.items()}
    manifest["responses"] = entries
    manifest["status"] = "complete"
    manifest.pop("error", None)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["scope_note"] = (
        "Core candidate sealed after structured filings, material facts, market "
        "communications, and shareholder notices. Optional offer/OPA/provent "
        "expansion was stopped before candidate construction."
    )
    manifest["response_count"] = len(entries)
    manifest["row_count"] = sum(int(entry["row_count"]) for entry in entries)
    _write_acquisition_manifest(raw_dir, manifest)
    return manifest


def _parse_response_entries(
    raw_dir: Path, entries: list[dict[str, object]]
) -> list[RadEvent]:
    events: list[RadEvent] = []
    for entry in entries:
        path = raw_dir / str(entry["response_file"])
        if _sha256(path) != str(entry["response_sha256"]):
            raise ValueError(f"RAD raw response hash changed: {path}")
        result = json.loads(path.read_bytes()).get("d")
        if result.get("SolicitarCaptcha") != "N" or result.get("temErro"):
            raise ValueError(f"RAD raw response is not usable: {path}")
        parsed = parse_rad_data(str(result.get("dados", "")), path.name)
        if len(parsed) != entry["row_count"]:
            raise ValueError(f"RAD row count changed while parsing {path}")
        events.extend(parsed)
    return events


def _deduplicate_events(events: list[RadEvent]) -> list[RadEvent]:
    by_key: dict[tuple[str, str], RadEvent] = {}
    for event in events:
        system = "ENET" if event.category.startswith(("ITR", "DFP")) else "IPE"
        key = (system, event.sequence_id)
        previous = by_key.get(key)
        if previous is not None and previous != event:
            raise ValueError(f"Conflicting duplicate RAD document key: {key}")
        by_key[key] = event
    return sorted(
        by_key.values(), key=lambda value: (value.delivery_timestamp, value.sequence_id)
    )


def load_rad_history(raw_dir: Path) -> tuple[list[RadEvent], dict[str, object]]:
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract_version") != RAW_CONTRACT_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("RAD acquisition is not a complete exact-response snapshot")
    events = _parse_response_entries(raw_dir, manifest["responses"])
    return _deduplicate_events(events), manifest


def materialize_structured_exact(raw_dir: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract_version") != RAW_CONTRACT_VERSION:
        raise ValueError("Unknown RAD acquisition contract")
    entries = [
        entry
        for entry in manifest["responses"]
        if entry.get("query_group") == "structured"
    ]
    expected_years = {int(value) for value in manifest["years"]}
    observed_years = {int(entry["year"]) for entry in entries}
    if observed_years != expected_years or len(entries) != len(expected_years):
        raise ValueError(
            f"Structured RAD query family is incomplete: {sorted(observed_years)}"
        )
    events = _deduplicate_events(_parse_response_entries(raw_dir, entries))
    if any(event.event_group != "itr_dfp" for event in events):
        raise ValueError("Structured RAD extraction contains a non-ITR/DFP event")
    rows = [
        {
            "ID_DOC": event.sequence_id,
            "CD_CVM": event.cvm_code,
            "DT_REFER": event.reference_date,
            "VERSAO": event.version,
            "CATEG_DOC": "ITR" if event.category.startswith("ITR") else "DFP",
            "delivery_timestamp": event.delivery_timestamp,
            "status": event.status,
            "company_name": event.company_name,
            "source_file": event.source_file,
        }
        for event in events
    ]
    frame = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("DT_REFER").cast(pl.Date),
        pl.col("delivery_timestamp").cast(pl.Datetime("us")),
    )
    if frame.group_by("CATEG_DOC", "ID_DOC").len().filter(pl.col("len") > 1).height:
        raise ValueError("Structured RAD document keys are not unique")
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "rad_structured_exact.parquet"
    frame.write_parquet(output_path, compression="zstd", statistics=True)
    output_manifest: dict[str, object] = {
        "contract_version": "CVM_RAD_STRUCTURED_EXACT_DELIVERY_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "query_group_complete": True,
        "parent_acquisition_status_at_build": manifest.get("status"),
        "years": sorted(expected_years),
        "timezone": RAD_TIMEZONE,
        "key_contract": (
            "Exact join on normalized ID_DOC/sequence_id plus CD_CVM, DT_REFER, "
            "VERSAO, and ITR/DFP category; reject missing or conflicting matches"
        ),
        "source_responses": [
            {
                "path": str((raw_dir / str(entry["response_file"])).resolve()),
                "sha256": entry["response_sha256"],
                "row_count": entry["row_count"],
            }
            for entry in entries
        ],
        "row_count": frame.height,
        "itr_rows": frame.filter(pl.col("CATEG_DOC") == "ITR").height,
        "dfp_rows": frame.filter(pl.col("CATEG_DOC") == "DFP").height,
        "first_delivery_timestamp": str(frame.get_column("delivery_timestamp").min()),
        "last_delivery_timestamp": str(frame.get_column("delivery_timestamp").max()),
        "output_file": output_path.name,
        "output_sha256": _sha256(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return output_manifest


def _zip_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [
        name for name in archive.namelist() if name.lower().endswith(suffix.lower())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} member, got {matches}")
    return matches[0]


def _read_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with archive.open(member) as raw:
        text = (line.decode("latin-1") for line in raw)
        return list(csv.DictReader(text, delimiter=";"))


def load_fca_identity(
    archives: list[Path],
) -> tuple[dict[str, list[tuple[date, str]]], list[FcaSecurity], dict[str, int]]:
    code_history: dict[str, set[tuple[date, str]]] = defaultdict(set)
    securities: set[FcaSecurity] = set()
    missing_security_document = 0
    excluded_security_rows = 0
    for path in archives:
        year = int(re.search(r"(\d{4})", path.stem).group(1))  # type: ignore[union-attr]
        with zipfile.ZipFile(path) as archive:
            header_rows = _read_csv(
                archive, _zip_member(archive, f"fca_cia_aberta_{year}.csv")
            )
            documents: dict[str, tuple[date, str, str]] = {}
            for row in header_rows:
                received = _parse_iso_date(row["DT_RECEB"])
                cnpj = _digits(row["CNPJ_CIA"])
                code = _digits(row["CD_CVM"]).zfill(6)
                if received is None or len(cnpj) != 14 or len(code) != 6:
                    continue
                documents[row["ID_DOC"]] = (received, cnpj, code)
                code_history[code].add((received, cnpj))
            value_rows = _read_csv(
                archive,
                _zip_member(archive, f"fca_cia_aberta_valor_mobiliario_{year}.csv"),
            )
            for row in value_rows:
                ticker = row["Codigo_Negociacao"].strip().upper()
                if (
                    row["Mercado"].strip() != "Bolsa"
                    or row["Valor_Mobiliario"].strip() not in _ALLOWED_SECURITIES
                    or not ticker
                ):
                    excluded_security_rows += 1
                    continue
                document = documents.get(row["ID_Documento"])
                if document is None:
                    missing_security_document += 1
                    continue
                known_date, document_cnpj, _ = document
                cnpj = _digits(row["CNPJ_Companhia"])
                if cnpj != document_cnpj:
                    raise ValueError(f"FCA document CNPJ mismatch in {path}")
                listing_start = _parse_iso_date(row["Data_Inicio_Listagem"])
                listing_end = _parse_iso_date(row["Data_Fim_Listagem"])
                start = _parse_iso_date(row["Data_Inicio_Negociacao"]) or listing_start
                if start is None:
                    excluded_security_rows += 1
                    continue
                # Exact same-day COTAHIST observation proves the ticker traded.
                # Prefer the listing interval when it is explicitly still open;
                # some FCA filings repeat a one-day negotiation end for a ticker
                # that remained listed for years (for example CEAB3).
                end = (
                    None
                    if listing_start is not None and listing_end is None
                    else _parse_iso_date(row["Data_Fim_Negociacao"]) or listing_end
                )
                securities.add(FcaSecurity(cnpj, ticker, known_date, start, end))
    normalized_history = {key: sorted(value) for key, value in code_history.items()}
    audit = {
        "cvm_codes": len(normalized_history),
        "fca_cash_security_records": len(securities),
        "missing_security_document_rows": missing_security_document,
        "excluded_security_rows": excluded_security_rows,
    }
    return (
        normalized_history,
        sorted(securities, key=lambda value: (value.ticker, value.known_date)),
        audit,
    )


def map_event_cnpjs(
    events: list[RadEvent], code_history: dict[str, list[tuple[date, str]]]
) -> tuple[list[RadEvent], int]:
    mapped: list[RadEvent] = []
    unmapped = 0
    for event in events:
        candidates = [
            item
            for item in code_history.get(event.cvm_code, ())
            if item[0] <= event.delivery_timestamp.date()
        ]
        if not candidates:
            mapped.append(event)
            unmapped += 1
            continue
        latest = max(item[0] for item in candidates)
        cnpjs = {cnpj for known, cnpj in candidates if known == latest}
        if len(cnpjs) != 1:
            mapped.append(event)
            unmapped += 1
            continue
        mapped.append(replace(event, cnpj=cnpjs.pop()))
    return mapped, unmapped


def first_available_decision(
    delivered: datetime, sessions: list[date]
) -> tuple[date, int] | None:
    position = bisect.bisect_left(sessions, delivered.date())
    if position == len(sessions):
        return None
    if sessions[position] == delivered.date():
        decision = bisect.bisect_right(DECISION_TIMES, delivered.time())
        if decision < DECISIONS_PER_SESSION:
            return sessions[position], decision
        position += 1
        if position == len(sessions):
            return None
    return sessions[position], 0


def assign_event_availability(
    events: list[RadEvent], sessions: list[date]
) -> list[RadEvent]:
    output: list[RadEvent] = []
    for event in events:
        available = first_available_decision(event.delivery_timestamp, sessions)
        output.append(
            event
            if available is None
            else replace(event, available_date=available[0], decision_idx=available[1])
        )
    return output


def _active_cnpj(records: list[FcaSecurity], current_date: date) -> str | None:
    active = [
        record
        for record in records
        if record.known_date <= current_date
        and record.effective_start <= current_date
        and (record.effective_end is None or current_date <= record.effective_end)
    ]
    if not active:
        return None
    latest = max(record.known_date for record in active)
    cnpjs = {record.cnpj for record in active if record.known_date == latest}
    return cnpjs.pop() if len(cnpjs) == 1 else None


def build_session_issuer_map(
    cotahist_dir: Path,
    feature_store: Path,
    securities: list[FcaSecurity],
    *,
    through: date,
) -> tuple[pl.DataFrame, list[date], dict[str, int]]:
    dates = pl.read_parquet(feature_store / "date_index.parquet").filter(
        pl.col("trade_date") <= through
    )
    equities = pl.read_parquet(feature_store / "equity_index.parquet").select(
        "equity_slot", "security_id"
    )
    canonical = set(equities.get_column("security_id").to_list())
    paths = sorted(cotahist_dir.glob("year=*/ticker_observations_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No COTAHIST ticker observations under {cotahist_dir}")
    quotes = (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("trade_date") <= through)
            & pl.col("security_id").is_in(canonical)
            & ~pl.col("security_id_is_fallback")
            & (pl.col("market_type") == 10)
            & (pl.col("bdi_code") == "02")
        )
        .select("trade_date", "security_id", "ticker")
        .unique()
        .collect()
    )
    sessions = sorted(quotes.get_column("trade_date").unique().to_list())
    by_ticker: dict[str, list[FcaSecurity]] = defaultdict(list)
    for record in securities:
        by_ticker[record.ticker].append(record)
    candidates: dict[tuple[date, str], set[str]] = defaultdict(set)
    for trade_date, security_id, ticker in quotes.iter_rows():
        cnpj = _active_cnpj(by_ticker.get(ticker, []), trade_date)
        if cnpj is not None:
            candidates[(trade_date, security_id)].add(cnpj)
    memberships = np.load(feature_store / "equity_membership.npy", mmap_mode="r")
    rows: list[tuple[date, int, str, str]] = []
    ambiguous = 0
    for date_idx, trade_date in dates.iter_rows():
        for equity_slot, security_id in equities.iter_rows():
            if not memberships[int(date_idx), int(equity_slot)]:
                continue
            cnpjs = candidates.get((trade_date, security_id), set())
            if len(cnpjs) == 1:
                rows.append(
                    (trade_date, int(equity_slot), security_id, next(iter(cnpjs)))
                )
            elif len(cnpjs) > 1:
                ambiguous += 1
    frame = pl.DataFrame(
        rows,
        schema={
            "trade_date": pl.Date,
            "equity_slot": pl.Int16,
            "security_id": pl.String,
            "cnpj": pl.String,
        },
        orient="row",
    )
    active_memberships = sum(
        int(memberships[int(value)].sum()) for value in dates["date_idx"]
    )
    audit = {
        "active_membership_security_days": active_memberships,
        "mapped_security_days": frame.height,
        "unmapped_security_days": active_memberships - frame.height - ambiguous,
        "ambiguous_security_days": ambiguous,
        "mapped_securities": frame.get_column("security_id").n_unique()
        if frame.height
        else 0,
    }
    return frame, sessions, audit


def _event_timelines(
    events: list[RadEvent], session_positions: dict[date, int]
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    timelines: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for event in events:
        if (
            event.cnpj is None
            or event.available_date is None
            or event.decision_idx is None
        ):
            continue
        session_idx = session_positions.get(event.available_date)
        if session_idx is not None:
            timelines[event.cnpj][event.event_group].append(
                (session_idx, event.decision_idx)
            )
    for by_group in timelines.values():
        for group, values in by_group.items():
            by_group[group] = sorted(set(values))
    return timelines


def state_rows_for_session(
    session_idx: int,
    trade_date: date,
    issuer_rows: list[tuple[str, str]],
    timelines: dict[str, dict[str, list[tuple[int, int]]]],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    age_denominator = math.log1p(AGE_CLIP_SESSIONS * DECISIONS_PER_SESSION * 5)
    for security_id, cnpj in issuer_rows:
        by_group = timelines.get(cnpj, {})
        all_events = sorted({value for values in by_group.values() for value in values})
        for decision_idx in range(DECISIONS_PER_SESSION):
            coordinate = (session_idx, decision_idx)
            values: list[float] = []
            for group in EVENT_GROUPS:
                group_events = by_group.get(group, [])
                position = bisect.bisect_right(group_events, coordinate)
                recent = (
                    position > 0
                    and session_idx - group_events[position - 1][0]
                    < RECENT_SESSION_COUNT
                )
                values.append(float(recent))
            any_position = bisect.bisect_right(all_events, coordinate)
            if any_position:
                previous_session, previous_decision = all_events[any_position - 1]
                steps = (
                    (session_idx - previous_session) * DECISIONS_PER_SESSION
                    + decision_idx
                    - previous_decision
                )
                age = min(math.log1p(steps * 5) / age_denominator, 1.0)
                age_mask = True
            else:
                age = 0.0
                age_mask = False
            rows.append(
                (
                    trade_date,
                    decision_idx,
                    security_id,
                    *values,
                    age,
                    True,
                    True,
                    True,
                    True,
                    age_mask,
                )
            )
    return rows


def _state_frame(rows: list[tuple[object, ...]]) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "available_date": pl.Date,
        "decision_idx": pl.Int8,
        "security_id": pl.String,
        **{feature: pl.Float32 for feature in FEATURES},
        **{f"{feature}_mask": pl.Boolean for feature in FEATURES},
    }
    return pl.DataFrame(rows, schema=schema, orient="row")


def _events_frame(events: list[RadEvent]) -> pl.DataFrame:
    rows = [
        {
            **asdict(event),
            "delivery_timestamp": event.delivery_timestamp,
            "reference_date": event.reference_date,
            "available_date": event.available_date,
        }
        for event in events
    ]
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("reference_date").cast(pl.Date),
        pl.col("delivery_timestamp").cast(pl.Datetime("us")),
        pl.col("available_date").cast(pl.Date),
        pl.col("decision_idx").cast(pl.Int8),
    )


def build_event_layer(
    raw_dir: Path,
    fca_dir: Path,
    cotahist_dir: Path,
    feature_store: Path,
    output_dir: Path,
    *,
    through: date = date(2024, 12, 31),
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    events, raw_manifest = load_rad_history(raw_dir)
    fca_archives = sorted(fca_dir.glob("fca_cia_aberta_20*.zip"))
    code_history, securities, fca_audit = load_fca_identity(fca_archives)
    events, unmapped_event_cnpjs = map_event_cnpjs(events, code_history)
    issuer_map, sessions, identity_audit = build_session_issuer_map(
        cotahist_dir, feature_store, securities, through=through
    )
    events = assign_event_availability(events, sessions)
    session_positions = {value: index for index, value in enumerate(sessions)}
    timelines = _event_timelines(events, session_positions)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        events_path = temporary / "rad_events_exact.parquet"
        _events_frame(events).write_parquet(
            events_path, compression="zstd", statistics=True
        )
        state_path = temporary / "cvm_rad_event_state.parquet"
        writer = None
        row_count = 0
        valid_counts = {feature: 0 for feature in FEATURES}
        try:
            import pyarrow.parquet as pq

            grouped = {
                key: [
                    (security_id, cnpj)
                    for security_id, cnpj in group.select(
                        "security_id", "cnpj"
                    ).iter_rows()
                ]
                for key, group in issuer_map.partition_by(
                    "trade_date", as_dict=True
                ).items()
            }
            for key in sorted(grouped):
                trade_date = key[0]
                session_idx = session_positions[trade_date]
                rows = state_rows_for_session(
                    session_idx, trade_date, grouped[key], timelines
                )
                if not rows:
                    continue
                frame = _state_frame(rows)
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(
                        state_path, table.schema, compression="zstd"
                    )
                writer.write_table(table)
                row_count += frame.height
                for feature in FEATURES:
                    valid_counts[feature] += int(
                        frame.get_column(f"{feature}_mask").sum()
                    )
        finally:
            if writer is not None:
                writer.close()
        if row_count == 0:
            raise ValueError("CVM RAD event state contains no rows")
        raw_manifest_path = raw_dir / "manifest.json"
        manifest: dict[str, object] = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "features": list(FEATURES),
            "availability_rule": (
                "RAD Data Entrega is interpreted in America/Sao_Paulo; availability is "
                "the first canonical five-minute decision strictly after receipt, or "
                "decision 0 of the next observed B3 session after 14:45/closed days"
            ),
            "identity_rule": (
                "CD_CVM maps to CNPJ through historically received FCA; exact FCA Bolsa "
                "cash ticker and its negotiation bounds map on each date to exact "
                "non-fallback COTAHIST security_id; active canonical share classes are "
                "broadcast separately and no ticker-prefix inference is permitted"
            ),
            "state_rule": (
                "Four observed-zero recent flags cover the latest five B3 sessions. "
                "Age uses only prior available events and log1p decision-grid trading "
                "minutes, clipped at 20 sessions; its mask is false before the first event."
            ),
            "omitted_features": {
                "pending_event": "prohibited because future event knowledge leaks",
                "event_anchored_drift": (
                    "omitted because the source layer has no causal intraday price input "
                    "and the incumbent raw sequence already contains post-event drift"
                ),
            },
            "through": through.isoformat(),
            "raw_snapshot": str(raw_dir.resolve()),
            "raw_manifest_sha256": _sha256(raw_manifest_path),
            "raw_response_count": raw_manifest["response_count"],
            "raw_row_count": raw_manifest["row_count"],
            "fca_sources": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in fca_archives
            ],
            "cotahist_source": str(cotahist_dir.resolve()),
            "feature_store": str(feature_store.resolve()),
            "event_rows": len(events),
            "event_cnpj_unmapped": unmapped_event_cnpjs,
            "events_with_model_availability": sum(
                event.available_date is not None for event in events
            ),
            "events_with_timeline": sum(
                event.cnpj is not None and event.available_date in session_positions
                for event in events
            ),
            "state_rows": row_count,
            "feature_valid_rows": valid_counts,
            "first_state_date": str(issuer_map.get_column("trade_date").min()),
            "last_state_date": str(issuer_map.get_column("trade_date").max()),
            "fca_audit": fca_audit,
            "identity_audit": identity_audit,
            "events_file": events_path.name,
            "events_sha256": _sha256(events_path),
            "state_file": state_path.name,
            "state_sha256": _sha256(state_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        if temporary.exists() and temporary.parent == output_dir.parent:
            shutil.rmtree(temporary)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire and build exact CVM RAD event state"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--out", type=Path, required=True)
    acquire.add_argument(
        "--years", type=int, nargs="*", default=list(range(2019, 2025))
    )
    seal = subparsers.add_parser("seal")
    seal.add_argument("--raw-dir", type=Path, required=True)
    structured = subparsers.add_parser("structured")
    structured.add_argument("--raw-dir", type=Path, required=True)
    structured.add_argument("--out", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--fca-dir", type=Path, required=True)
    build.add_argument("--cotahist-dir", type=Path, required=True)
    build.add_argument("--feature-store", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--through", type=date.fromisoformat, default=date(2024, 12, 31))
    args = parser.parse_args()
    if args.command == "acquire":
        manifest = acquire_rad_history(args.out, years=tuple(args.years))
        print(f"Captured {manifest['row_count']:,} exact RAD rows in {args.out}")
    elif args.command == "seal":
        manifest = seal_rad_history(args.raw_dir)
        print(f"Sealed {manifest['row_count']:,} exact RAD rows in {args.raw_dir}")
    elif args.command == "structured":
        manifest = materialize_structured_exact(args.raw_dir, args.out)
        print(f"Wrote {manifest['row_count']:,} exact structured rows to {args.out}")
    else:
        manifest = build_event_layer(
            args.raw_dir,
            args.fca_dir,
            args.cotahist_dir,
            args.feature_store,
            args.out,
            through=args.through,
        )
        print(f"Wrote {manifest['state_rows']:,} event-state rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
