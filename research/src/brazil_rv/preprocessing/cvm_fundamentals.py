from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import unicodedata
import zipfile
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path

import polars as pl

from .contract import DECISION_TIMES

CONTRACT_VERSION = "CVM_STRUCTURED_FUNDAMENTALS_PIT_INTRADAY_V1"
ACCOUNT_CODES = {
    "assets": ("BPA", "1"),
    "equity": ("BPP", "2.03"),
    "revenue": ("DRE", "3.01"),
    "net_income": ("DRE", "3.11"),
    "operating_cash_flow": ("DFC", "6.01"),
}
VALUE_FEATURES = (
    "fund_net_margin_ttm",
    "fund_roa_ttm",
    "fund_leverage",
    "fund_sales_growth_yoy",
    "fund_asset_growth_yoy",
    "fund_accruals_assets_ttm",
)
FEATURES = (
    *VALUE_FEATURES,
    "fund_filing_age",
    "fund_financial_sector",
    "fund_consolidated_basis",
)
FINANCIAL_INCOMPARABLE = {
    "fund_net_margin_ttm",
    "fund_sales_growth_yoy",
    "fund_accruals_assets_ttm",
}
FLOW_METRICS = ("revenue", "net_income", "operating_cash_flow")
SCALE_MULTIPLIERS = {"UNIDADE": 1.0, "MIL": 1_000.0}


@dataclass(frozen=True)
class AccountValue:
    value_brl: float
    period_start: date | None = None
    period_end: date | None = None


@dataclass
class FilingDocument:
    cvm_code: str
    cnpj: str
    category: str
    reference_date: date
    version: int
    sequence_id: str
    receipt_date: date
    available_date: date
    decision_idx: int
    receipt_order: datetime
    values: dict[str, dict[str, AccountValue]] = field(
        default_factory=lambda: {"con": {}, "ind": {}}
    )


@dataclass(frozen=True)
class FcaSecurity:
    ticker: str
    start: date | None
    end: date | None


@dataclass(frozen=True)
class FcaDocument:
    cvm_code: str
    cnpj: str
    reference_date: date
    version: int
    sequence_id: str
    available_date: date
    sector: str | None
    securities: tuple[FcaSecurity, ...]


@dataclass(frozen=True)
class FactorState:
    available_date: date
    decision_idx: int
    filing_date: date
    reference_date: date
    consolidated: bool
    values: dict[str, tuple[float, bool]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digits(value: object) -> str:
    return "".join(character for character in str(value) if character.isdigit())


def _cvm_code(value: object) -> str:
    return _digits(value).zfill(6)


def _version(value: object) -> int:
    return int(_digits(value))


def _iso_date(value: object) -> date:
    return date.fromisoformat(str(value))


def _optional_date(value: object) -> date | None:
    if value is None or not str(value).strip():
        return None
    return _iso_date(value)


def _read_member(archive: zipfile.ZipFile, member: str) -> pl.DataFrame:
    text = archive.read(member).decode("latin1").encode("utf-8")
    return pl.read_csv(
        io.BytesIO(text),
        separator=";",
        quote_char=None,
        infer_schema_length=0,
    )


def _next_session(value: date, sessions: Sequence[date]) -> date | None:
    index = bisect_right(sessions, value)
    return None if index == len(sessions) else sessions[index]


def _receipt_availability(
    receipt: datetime, sessions: Sequence[date]
) -> tuple[date, int] | None:
    receipt_date = receipt.date()
    if receipt_date in sessions:
        decision_idx = bisect_right(DECISION_TIMES, receipt.time())
        if decision_idx < len(DECISION_TIMES):
            return receipt_date, decision_idx
    next_session = _next_session(receipt_date, sessions)
    return None if next_session is None else (next_session, 0)


def _scale_value(value: object, moeda: object, escala: object) -> float | None:
    if str(moeda).strip() != "REAL":
        return None
    multiplier = SCALE_MULTIPLIERS.get(str(escala).strip())
    if multiplier is None:
        return None
    return float(str(value)) * multiplier


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _normalize_text(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value).lower()
        if not unicodedata.combining(character)
    )


def _is_financial_sector(sector: str | None) -> tuple[bool, bool]:
    if not sector:
        return False, False
    normalized = _normalize_text(sector)
    keywords = (
        "banco",
        "bancos",
        "financeir",
        "segurador",
        "corretor",
        "credito",
        "previdencia",
        "securitiz",
        "arrendamento",
        "bolsas de valores",
    )
    return any(keyword in normalized for keyword in keywords), True


def _prior_year(value: date) -> date:
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return value.replace(year=value.year - 1, day=28)


def _quarterly_values(
    reports: Iterable[FilingDocument], basis: str, metric: str
) -> dict[date, float]:
    cumulative: dict[tuple[date, int], AccountValue] = {}
    for report in reports:
        account = report.values[basis].get(metric)
        if (
            account is None
            or account.period_start is None
            or account.period_end is None
        ):
            continue
        months = (
            (account.period_end.year - account.period_start.year) * 12
            + account.period_end.month
            - account.period_start.month
            + 1
        )
        if months not in (3, 6, 9, 12):
            continue
        cumulative[(account.period_start, months // 3)] = account
    quarters: dict[date, float] = {}
    for (period_start, quarter), account in cumulative.items():
        if quarter == 1:
            quarters[account.period_end] = account.value_brl
            continue
        previous = cumulative.get((period_start, quarter - 1))
        if previous is not None:
            quarters[account.period_end] = account.value_brl - previous.value_brl
    return quarters


def _ttm_values(quarters: dict[date, float]) -> dict[date, float]:
    ordered = sorted(quarters)
    output: dict[date, float] = {}
    for index in range(3, len(ordered)):
        endpoints = ordered[index - 3 : index + 1]
        gaps = [
            (later - earlier).days for earlier, later in zip(endpoints, endpoints[1:])
        ]
        if all(60 <= gap <= 120 for gap in gaps):
            output[endpoints[-1]] = sum(quarters[endpoint] for endpoint in endpoints)
    return output


def _compute_basis(
    reports: Iterable[FilingDocument], basis: str
) -> tuple[date, dict[str, tuple[float, bool]]] | None:
    reports = list(reports)
    stock_reports = [
        report
        for report in reports
        if "assets" in report.values[basis] and "equity" in report.values[basis]
    ]
    if not stock_reports:
        return None
    latest = max(stock_reports, key=lambda report: report.reference_date)
    reference_date = latest.reference_date
    assets = latest.values[basis]["assets"].value_brl
    equity = latest.values[basis]["equity"].value_brl
    if not math.isfinite(assets) or assets <= 0 or not math.isfinite(equity):
        return None
    prior_reference = _prior_year(reference_date)
    prior_report = next(
        (
            report
            for report in stock_reports
            if report.reference_date == prior_reference
        ),
        None,
    )
    prior_assets = (
        None if prior_report is None else prior_report.values[basis]["assets"].value_brl
    )
    average_assets = (
        (assets + prior_assets) / 2
        if prior_assets is not None and prior_assets > 0
        else None
    )

    ttm = {
        metric: _ttm_values(_quarterly_values(reports, basis, metric))
        for metric in FLOW_METRICS
    }
    revenue = ttm["revenue"].get(reference_date)
    net_income = ttm["net_income"].get(reference_date)
    if revenue is None or net_income is None or not math.isfinite(revenue + net_income):
        return None
    prior_revenue = ttm["revenue"].get(prior_reference)
    operating_cash = ttm["operating_cash_flow"].get(reference_date)

    values: dict[str, tuple[float, bool]] = {}
    margin_valid = abs(revenue) > 1e-12
    values["fund_net_margin_ttm"] = (
        _clip(net_income / revenue, -2.0, 2.0) if margin_valid else 0.0,
        margin_valid,
    )
    roa_valid = average_assets is not None and average_assets > 0
    values["fund_roa_ttm"] = (
        _clip(net_income / average_assets, -1.0, 1.0) if roa_valid else 0.0,
        roa_valid,
    )
    values["fund_leverage"] = (_clip(1.0 - equity / assets, -1.0, 2.0), True)
    sales_growth_valid = prior_revenue is not None and revenue > 0 and prior_revenue > 0
    values["fund_sales_growth_yoy"] = (
        _clip(math.log(revenue / prior_revenue), -2.0, 2.0)
        if sales_growth_valid
        else 0.0,
        sales_growth_valid,
    )
    asset_growth_valid = prior_assets is not None and prior_assets > 0
    values["fund_asset_growth_yoy"] = (
        _clip(math.log(assets / prior_assets), -1.5, 1.5)
        if asset_growth_valid
        else 0.0,
        asset_growth_valid,
    )
    accrual_valid = operating_cash is not None and roa_valid
    values["fund_accruals_assets_ttm"] = (
        _clip((net_income - operating_cash) / average_assets, -1.0, 1.0)
        if accrual_valid
        else 0.0,
        accrual_valid,
    )
    return reference_date, values


def _compute_factor_state(
    reports: Iterable[FilingDocument],
    available_date: date,
    decision_idx: int,
    filing_date: date,
) -> FactorState | None:
    reports = list(reports)
    consolidated = _compute_basis(reports, "con")
    selected = consolidated or _compute_basis(reports, "ind")
    if selected is None:
        return None
    reference_date, values = selected
    return FactorState(
        available_date=available_date,
        decision_idx=decision_idx,
        filing_date=filing_date,
        reference_date=reference_date,
        consolidated=consolidated is not None,
        values=values,
    )


def build_factor_events(
    documents: Sequence[FilingDocument],
) -> dict[str, list[FactorState]]:
    by_issuer: dict[str, list[FilingDocument]] = defaultdict(list)
    for document in documents:
        by_issuer[document.cvm_code].append(document)
    output: dict[str, list[FactorState]] = {}
    for issuer, events in by_issuer.items():
        ledger: dict[tuple[str, date], FilingDocument] = {}
        states: list[FactorState] = []
        coordinates = sorted(
            {(event.available_date, event.decision_idx) for event in events}
        )
        events_by_coordinate: dict[tuple[date, int], list[FilingDocument]] = (
            defaultdict(list)
        )
        for event in events:
            events_by_coordinate[(event.available_date, event.decision_idx)].append(
                event
            )
        for available_date, decision_idx in coordinates:
            changed = False
            changed_events: list[FilingDocument] = []
            for event in sorted(
                events_by_coordinate[(available_date, decision_idx)],
                key=lambda value: (
                    value.receipt_order,
                    value.version,
                    value.sequence_id,
                ),
            ):
                key = (event.category, event.reference_date)
                current = ledger.get(key)
                if current is None or event.version > current.version:
                    ledger[key] = event
                    changed = True
                    changed_events.append(event)
            if changed:
                filing_date = max(
                    changed_events, key=lambda value: value.receipt_order
                ).receipt_date
                state = _compute_factor_state(
                    ledger.values(), available_date, decision_idx, filing_date
                )
                if state is not None:
                    states.append(state)
        if states:
            output[issuer] = states
    return output


def _load_rad_events(path: Path | None) -> dict[tuple[str, str, date, int], dict]:
    if path is None:
        return {}
    frame = pl.read_parquet(path)
    required = {
        "ID_DOC",
        "CD_CVM",
        "DT_REFER",
        "VERSAO",
        "CATEG_DOC",
        "delivery_timestamp",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"RAD event table is missing columns: {missing}")
    output: dict[tuple[str, str, date, int], dict] = {}
    for row in frame.iter_rows(named=True):
        key = (
            _digits(row["ID_DOC"]),
            _cvm_code(row["CD_CVM"]),
            row["DT_REFER"],
            _version(row["VERSAO"]),
        )
        if key in output:
            raise ValueError(f"RAD event table contains duplicate document key {key}")
        output[key] = row
    return output


def _document_availability(
    *,
    sequence_id: str,
    cvm_code: str,
    category: str,
    reference_date: date,
    version: int,
    receipt_date: date,
    sessions: Sequence[date],
    rad_events: dict[tuple[str, str, date, int], dict],
) -> tuple[date | None, int, datetime, bool]:
    if rad_events:
        key = (sequence_id, cvm_code, reference_date, version)
        event = rad_events.get(key)
        if event is None:
            raise ValueError(f"Structured filing has no exact RAD receipt match: {key}")
        if str(event["CATEG_DOC"]).upper() != category:
            raise ValueError(f"RAD category differs from structured filing for {key}")
        receipt_order = event["delivery_timestamp"]
        if not isinstance(receipt_order, datetime):
            raise ValueError("RAD delivery_timestamp must have Datetime dtype")
        availability = _receipt_availability(receipt_order, sessions)
        if availability is None:
            return None, 0, receipt_order, True
        available_date, decision_idx = availability
        return available_date, decision_idx, receipt_order, True
    available = _next_session(receipt_date, sessions)
    return available, 0, datetime.combine(receipt_date, time.max), False


def _filing_headers(
    raw_dir: Path,
    sessions: Sequence[date],
    issuer_codes: set[str],
    rad_events: dict[tuple[str, str, date, int], dict],
) -> tuple[dict[tuple[str, str, date, int], FilingDocument], dict[str, object]]:
    documents: dict[tuple[str, str, date, int], FilingDocument] = {}
    audit_counts = {
        "header_rows": 0,
        "exact_rad_receipts": 0,
        "date_fallback_receipts": 0,
        "post_horizon_receipts_skipped": 0,
    }
    unmatched_rad_documents: list[dict[str, object]] = []
    for category in ("itr", "dfp"):
        for year in range(2019, 2025):
            path = raw_dir / f"{category}_cia_aberta_{year}.zip"
            with zipfile.ZipFile(path) as archive:
                frame = _read_member(archive, f"{category}_cia_aberta_{year}.csv")
            for row in frame.iter_rows(named=True):
                cvm_code = _cvm_code(row["CD_CVM"])
                if cvm_code not in issuer_codes:
                    continue
                cnpj = _digits(row["CNPJ_CIA"])
                reference_date = _iso_date(row["DT_REFER"])
                version = _version(row["VERSAO"])
                sequence_id = _digits(row["ID_DOC"])
                receipt_date = _iso_date(row["DT_RECEB"])
                if receipt_date > sessions[-1]:
                    audit_counts["post_horizon_receipts_skipped"] += 1
                    continue
                rad_key = (sequence_id, cvm_code, reference_date, version)
                if rad_events and rad_key not in rad_events:
                    unmatched_rad_documents.append(
                        {
                            "ID_DOC": sequence_id,
                            "CD_CVM": cvm_code,
                            "DT_REFER": reference_date.isoformat(),
                            "VERSAO": version,
                            "CATEG_DOC": category.upper(),
                            "DT_RECEB": receipt_date.isoformat(),
                        }
                    )
                    continue
                available, decision_idx, order, exact = _document_availability(
                    sequence_id=sequence_id,
                    cvm_code=cvm_code,
                    category=category.upper(),
                    reference_date=reference_date,
                    version=version,
                    receipt_date=receipt_date,
                    sessions=sessions,
                    rad_events=rad_events,
                )
                if available is None:
                    continue
                key = (category.upper(), cnpj, reference_date, version)
                if key in documents:
                    raise ValueError(f"Duplicate structured filing header {key}")
                documents[key] = FilingDocument(
                    cvm_code=cvm_code,
                    cnpj=cnpj,
                    category=category.upper(),
                    reference_date=reference_date,
                    version=version,
                    sequence_id=sequence_id,
                    receipt_date=receipt_date,
                    available_date=available,
                    decision_idx=decision_idx,
                    receipt_order=order,
                )
                audit_counts["header_rows"] += 1
                audit_counts[
                    "exact_rad_receipts" if exact else "date_fallback_receipts"
                ] += 1
    return documents, {
        **audit_counts,
        "rad_unmatched_receipt_count": len(unmatched_rad_documents),
        "rad_unmatched_documents": unmatched_rad_documents,
    }


def _attach_statement_values(
    raw_dir: Path,
    documents: dict[tuple[str, str, date, int], FilingDocument],
) -> dict[str, int]:
    audit = {
        "accepted_account_rows": 0,
        "non_real_or_unknown_scale_rows": 0,
    }
    specifications = (
        ("BPA", "assets", "1"),
        ("BPP", "equity", "2.03"),
        ("DRE", "revenue", "3.01"),
        ("DRE", "net_income", "3.11"),
        ("DFC_MD", "operating_cash_flow", "6.01"),
        ("DFC_MI", "operating_cash_flow", "6.01"),
    )
    for category in ("itr", "dfp"):
        for year in range(2019, 2025):
            path = raw_dir / f"{category}_cia_aberta_{year}.zip"
            with zipfile.ZipFile(path) as archive:
                for statement, metric, account_code in specifications:
                    for basis in ("con", "ind"):
                        member = f"{category}_cia_aberta_{statement}_{basis}_{year}.csv"
                        frame = _read_member(archive, member).filter(
                            (pl.col("ORDEM_EXERC") == "ÚLTIMO")
                            & (pl.col("CD_CONTA") == account_code)
                            & (pl.col("ST_CONTA_FIXA") == "S")
                        )
                        for row in frame.iter_rows(named=True):
                            key = (
                                category.upper(),
                                _digits(row["CNPJ_CIA"]),
                                _iso_date(row["DT_REFER"]),
                                _version(row["VERSAO"]),
                            )
                            document = documents.get(key)
                            if document is None:
                                continue
                            if _cvm_code(row["CD_CVM"]) != document.cvm_code:
                                raise ValueError(f"Statement CD_CVM differs for {key}")
                            value = _scale_value(
                                row["VL_CONTA"], row["MOEDA"], row["ESCALA_MOEDA"]
                            )
                            if value is None:
                                audit["non_real_or_unknown_scale_rows"] += 1
                                continue
                            period_end = _iso_date(row["DT_FIM_EXERC"])
                            period_start = (
                                _iso_date(row["DT_INI_EXERC"])
                                if metric in FLOW_METRICS
                                else None
                            )
                            account = AccountValue(value, period_start, period_end)
                            previous = document.values[basis].get(metric)
                            if previous is not None:
                                if (
                                    metric in FLOW_METRICS
                                    and previous.period_end == account.period_end
                                    and previous.period_start is not None
                                    and account.period_start is not None
                                    and previous.period_start != account.period_start
                                ):
                                    # ITR DRE commonly publishes both YTD and the
                                    # current standalone quarter as ORDEM_EXERC=ÚLTIMO.
                                    # Keep the earliest start so quarter derivation
                                    # always begins from the cumulative observation.
                                    if account.period_start < previous.period_start:
                                        document.values[basis][metric] = account
                                    continue
                                same = (
                                    math.isclose(
                                        previous.value_brl,
                                        account.value_brl,
                                        rel_tol=1e-12,
                                        abs_tol=1e-6,
                                    )
                                    and previous.period_start == account.period_start
                                    and previous.period_end == account.period_end
                                )
                                if not same:
                                    raise ValueError(
                                        f"Conflicting fixed account {metric} for {key}/{basis}"
                                    )
                                continue
                            document.values[basis][metric] = account
                            audit["accepted_account_rows"] += 1
    return audit


def load_filing_documents(
    raw_dir: Path,
    sessions: Sequence[date],
    issuer_codes: set[str],
    *,
    rad_events_path: Path | None = None,
) -> tuple[list[FilingDocument], dict[str, object]]:
    rad_events = _load_rad_events(rad_events_path)
    documents, audit = _filing_headers(raw_dir, sessions, issuer_codes, rad_events)
    audit.update(_attach_statement_values(raw_dir, documents))
    usable = [
        document
        for document in documents.values()
        if document.values["con"] or document.values["ind"]
    ]
    audit["usable_documents"] = len(usable)
    return usable, audit


def load_fca_documents(
    raw_dir: Path,
    sessions: Sequence[date],
    target_tickers: set[str],
) -> tuple[list[FcaDocument], dict[str, object]]:
    all_documents: list[FcaDocument] = []
    matching_issuers: set[str] = set()
    for year in range(2019, 2025):
        path = raw_dir / f"fca_cia_aberta_{year}.zip"
        with zipfile.ZipFile(path) as archive:
            headers = _read_member(archive, f"fca_cia_aberta_{year}.csv")
            general = _read_member(archive, f"fca_cia_aberta_geral_{year}.csv")
            securities = _read_member(
                archive, f"fca_cia_aberta_valor_mobiliario_{year}.csv"
            )
        header_by_key = {
            (
                _digits(row["CNPJ_CIA"]),
                _iso_date(row["DT_REFER"]),
                _version(row["VERSAO"]),
                _digits(row["ID_DOC"]),
            ): row
            for row in headers.iter_rows(named=True)
        }
        general_by_key: dict[tuple[str, date, int, str], dict] = {}
        for row in general.iter_rows(named=True):
            key = (
                _digits(row["CNPJ_Companhia"]),
                _iso_date(row["Data_Referencia"]),
                _version(row["Versao"]),
                _digits(row["ID_Documento"]),
            )
            if key in general_by_key:
                raise ValueError(f"Duplicate FCA general row {key}")
            general_by_key[key] = row
        securities_by_key: dict[tuple[str, date, int, str], list[FcaSecurity]] = (
            defaultdict(list)
        )
        for row in securities.iter_rows(named=True):
            security_type = str(row["Valor_Mobiliario"] or "")
            if not (
                security_type.startswith("Ações Ordinárias")
                or security_type.startswith("Ações Preferenciais")
                or security_type == "Units"
            ):
                continue
            if str(row["Mercado"] or "") != "Bolsa":
                continue
            ticker = str(row["Codigo_Negociacao"] or "").strip().upper()
            if not ticker:
                continue
            key = (
                _digits(row["CNPJ_Companhia"]),
                _iso_date(row["Data_Referencia"]),
                _version(row["Versao"]),
                _digits(row["ID_Documento"]),
            )
            securities_by_key[key].append(
                FcaSecurity(
                    ticker=ticker,
                    start=_optional_date(row["Data_Inicio_Negociacao"]),
                    end=_optional_date(row["Data_Fim_Negociacao"]),
                )
            )
            if ticker in target_tickers:
                header = header_by_key.get(key)
                if header is not None:
                    matching_issuers.add(_cvm_code(header["CD_CVM"]))
        for key, row in general_by_key.items():
            header = header_by_key.get(key)
            if header is None:
                continue
            available = _next_session(_iso_date(header["DT_RECEB"]), sessions)
            if available is None:
                continue
            all_documents.append(
                FcaDocument(
                    cvm_code=_cvm_code(row["Codigo_CVM"]),
                    cnpj=key[0],
                    reference_date=key[1],
                    version=key[2],
                    sequence_id=key[3],
                    available_date=available,
                    sector=(
                        str(row["Setor_Atividade"]).strip()
                        if row["Setor_Atividade"]
                        else None
                    ),
                    securities=tuple(securities_by_key.get(key, [])),
                )
            )
    selected = [
        document for document in all_documents if document.cvm_code in matching_issuers
    ]
    audit: dict[str, object] = {
        "fca_document_count": len(selected),
        "mapped_issuer_count": len(matching_issuers),
        "target_ticker_count": len(target_tickers),
        "fca_matched_target_tickers": sorted(
            {
                security.ticker
                for document in selected
                for security in document.securities
                if security.ticker in target_tickers
            }
        ),
    }
    return selected, audit


def _latest_by_coordinate[T](
    values: Sequence[T],
    coordinates: Sequence[tuple[date, int]],
    value: tuple[date, int],
) -> T | None:
    index = bisect_right(coordinates, value) - 1
    return None if index < 0 else values[index]


def _ticker_histories(
    cotahist: pl.DataFrame,
) -> dict[str, tuple[list[date], list[str]]]:
    duplicate = (
        cotahist.group_by("trade_date", "ticker")
        .agg(pl.col("security_id").n_unique().alias("identities"))
        .filter(pl.col("identities") > 1)
    )
    if duplicate.height:
        raise ValueError(
            "A cash ticker maps to multiple permanent identities on one date"
        )
    histories: dict[str, tuple[list[date], list[str]]] = {}
    for ticker, group in cotahist.sort("ticker", "trade_date").group_by(
        "ticker", maintain_order=True
    ):
        ticker_value = ticker[0] if isinstance(ticker, tuple) else ticker
        histories[str(ticker_value)] = (
            group.get_column("trade_date").to_list(),
            group.get_column("security_id").to_list(),
        )
    return histories


def _security_for_ticker(
    ticker: FcaSecurity,
    current_date: date,
    histories: dict[str, tuple[list[date], list[str]]],
) -> str | None:
    if ticker.start is not None and current_date < ticker.start:
        return None
    if ticker.end is not None and current_date > ticker.end:
        return None
    history = histories.get(ticker.ticker)
    if history is None:
        return None
    dates, security_ids = history
    index = bisect_right(dates, current_date) - 1
    if index < 0 or (ticker.start is not None and dates[index] < ticker.start):
        return None
    return security_ids[index]


def _latest_fca_documents(
    documents: Sequence[FcaDocument], sessions: Sequence[date]
) -> dict[date, dict[str, FcaDocument]]:
    events: dict[date, list[FcaDocument]] = defaultdict(list)
    for document in documents:
        events[document.available_date].append(document)
    ledger: dict[str, dict[date, FcaDocument]] = defaultdict(dict)
    output: dict[date, dict[str, FcaDocument]] = {}
    for current_date in sessions:
        for document in events.get(current_date, []):
            current = ledger[document.cvm_code].get(document.reference_date)
            if current is None or document.version > current.version:
                ledger[document.cvm_code][document.reference_date] = document
        output[current_date] = {
            issuer: max(by_reference.values(), key=lambda value: value.reference_date)
            for issuer, by_reference in ledger.items()
        }
    return output


def _intraday_frame_chunks(
    factor_events: dict[str, list[FactorState]],
    fca_documents: Sequence[FcaDocument],
    cotahist: pl.DataFrame,
    sessions: Sequence[date],
    *,
    available_start: date,
    available_end: date,
) -> Iterator[pl.DataFrame]:
    if available_start > available_end:
        raise ValueError("available_start cannot be after available_end")
    sessions = sorted(set(sessions))
    selected_dates = [
        value for value in sessions if available_start <= value <= available_end
    ]
    if not selected_dates:
        raise ValueError("No market sessions lie inside the output interval")
    required_cotahist = {"trade_date", "ticker", "security_id"}
    if not required_cotahist.issubset(cotahist.columns):
        raise ValueError("COTAHIST identity frame is missing required columns")
    histories = _ticker_histories(cotahist)
    fca_by_date = _latest_fca_documents(fca_documents, sessions)
    state_coordinates = {
        issuer: [(state.available_date, state.decision_idx) for state in states]
        for issuer, states in factor_events.items()
    }
    for current_date in selected_dates:
        assigned: dict[str, str] = {}
        mapped_issuers: list[tuple[str, list[str], bool, bool]] = []
        for issuer, states in factor_events.items():
            fca = fca_by_date[current_date].get(issuer)
            if fca is None:
                continue
            securities = {
                security_id
                for ticker in fca.securities
                if (
                    security_id := _security_for_ticker(ticker, current_date, histories)
                )
                is not None
            }
            if not securities:
                continue
            financial, sector_known = _is_financial_sector(fca.sector)
            for security_id in sorted(securities):
                previous_issuer = assigned.get(security_id)
                if previous_issuer is not None and previous_issuer != issuer:
                    raise ValueError(
                        f"{security_id} maps to multiple FCA issuers on {current_date}"
                    )
                assigned[security_id] = issuer
            mapped_issuers.append((issuer, sorted(securities), financial, sector_known))

        rows: list[dict[str, object]] = []
        for decision_idx in range(len(DECISION_TIMES)):
            coordinate = (current_date, decision_idx)
            for issuer, securities, financial, sector_known in mapped_issuers:
                state = _latest_by_coordinate(
                    factor_events[issuer], state_coordinates[issuer], coordinate
                )
                if state is None:
                    continue
                age_days = max((current_date - state.filing_date).days, 0)
                filing_age = min(math.log1p(age_days) / math.log1p(365), 1.0)
                for security_id in securities:
                    row: dict[str, object] = {
                        "available_date": current_date,
                        "decision_idx": decision_idx,
                        "source_receipt_date": state.filing_date,
                        "security_id": security_id,
                    }
                    for feature in VALUE_FEATURES:
                        value, valid = state.values[feature]
                        if feature in FINANCIAL_INCOMPARABLE:
                            valid = valid and sector_known and not financial
                        row[feature] = value if valid else 0.0
                        row[f"{feature}_mask"] = valid
                    row["fund_filing_age"] = filing_age
                    row["fund_filing_age_mask"] = True
                    row["fund_financial_sector"] = (
                        float(financial) if sector_known else 0.0
                    )
                    row["fund_financial_sector_mask"] = sector_known
                    row["fund_consolidated_basis"] = float(state.consolidated)
                    row["fund_consolidated_basis_mask"] = True
                    rows.append(row)
        if rows:
            yield (
                pl.DataFrame(rows, infer_schema_length=None)
                .with_columns(
                    pl.col("available_date", "source_receipt_date").cast(pl.Date),
                    pl.col("decision_idx").cast(pl.Int8),
                    *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
                    *[
                        pl.col(f"{feature}_mask").cast(pl.Boolean)
                        for feature in FEATURES
                    ],
                )
                .select(
                    "available_date",
                    "decision_idx",
                    "source_receipt_date",
                    "security_id",
                    *[
                        column
                        for feature in FEATURES
                        for column in (feature, f"{feature}_mask")
                    ],
                )
                .sort("decision_idx", "security_id")
            )


def build_intraday_frame(
    factor_events: dict[str, list[FactorState]],
    fca_documents: Sequence[FcaDocument],
    cotahist: pl.DataFrame,
    sessions: Sequence[date],
    *,
    available_start: date,
    available_end: date,
) -> tuple[pl.DataFrame, dict[str, object]]:
    chunks = list(
        _intraday_frame_chunks(
            factor_events,
            fca_documents,
            cotahist,
            sessions,
            available_start=available_start,
            available_end=available_end,
        )
    )
    if not chunks:
        raise ValueError("No fundamental states map to canonical securities")
    frame = pl.concat(chunks)
    duplicate = (
        frame.group_by("available_date", "decision_idx", "security_id")
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicate.height:
        raise ValueError(
            "Intraday fundamentals contain duplicate coordinate/security keys"
        )
    audit: dict[str, object] = {
        "output_row_count": frame.height,
        "output_date_count": frame.get_column("available_date").n_unique(),
        "output_security_count": frame.get_column("security_id").n_unique(),
        "output_decision_count": frame.get_column("decision_idx").n_unique(),
        "feature_valid_rows": {
            feature: int(frame.get_column(f"{feature}_mask").sum())
            for feature in FEATURES
        },
    }
    return frame, audit


def _cotahist_files(cotahist_dir: Path, end: date) -> list[Path]:
    files = [
        path
        for path in sorted(cotahist_dir.glob("year=*/equities_daily_*.parquet"))
        if int(path.parent.name.removeprefix("year=")) <= end.year
    ]
    if not files:
        raise FileNotFoundError(f"No parsed COTAHIST files under {cotahist_dir}")
    return files


def _input_archives(raw_dir: Path) -> list[Path]:
    paths = [
        raw_dir / f"{category}_cia_aberta_{year}.zip"
        for year in range(2019, 2025)
        for category in ("itr", "dfp", "fca")
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return paths


def build_artifact(
    raw_dir: Path,
    cotahist_dir: Path,
    security_index: Path,
    output_dir: Path,
    *,
    available_start: date,
    available_end: date,
    rad_events_path: Path | None = None,
) -> dict[str, object]:
    raw_dir = raw_dir.resolve()
    cotahist_dir = cotahist_dir.resolve()
    security_index = security_index.resolve()
    output_dir = output_dir.resolve()
    rad_events_path = None if rad_events_path is None else rad_events_path.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    archives = _input_archives(raw_dir)
    cotahist_files = _cotahist_files(cotahist_dir, available_end)
    security_ids = (
        pl.read_parquet(security_index).get_column("security_id").unique().to_list()
    )
    if not security_ids or any(not value.startswith("ISIN:") for value in security_ids):
        raise ValueError("security index must contain permanent ISIN identities")
    cotahist = (
        pl.scan_parquet(cotahist_files)
        .filter(
            pl.col("security_id").is_in(security_ids),
            pl.col("trade_date") <= available_end,
        )
        .select("trade_date", "ticker", "security_id")
        .collect()
    )
    target_tickers = set(cotahist.get_column("ticker").to_list())
    sessions = (
        pl.scan_parquet(cotahist_files)
        .filter(pl.col("trade_date") <= available_end)
        .select(pl.col("trade_date").unique())
        .collect()
        .get_column("trade_date")
        .unique()
        .sort()
        .to_list()
    )
    fca_documents, fca_audit = load_fca_documents(raw_dir, sessions, target_tickers)
    issuer_codes = {document.cvm_code for document in fca_documents}
    filing_documents, filing_audit = load_filing_documents(
        raw_dir,
        sessions,
        issuer_codes,
        rad_events_path=rad_events_path,
    )
    factor_events = build_factor_events(filing_documents)
    frame, output_audit = build_intraday_frame(
        factor_events,
        fca_documents,
        cotahist,
        sessions,
        available_start=available_start,
        available_end=available_end,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        data_path = temporary / "cvm_fundamentals_intraday.parquet"
        frame.write_parquet(data_path, compression="zstd", statistics=True)
        manifest: dict[str, object] = {
            "contract_version": CONTRACT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "source_archives": [
                {"path": str(path), "sha256": _sha256(path)} for path in archives
            ],
            "cotahist_identity_files": [
                {"path": str(path), "sha256": _sha256(path)} for path in cotahist_files
            ],
            "security_index": {
                "path": str(security_index),
                "sha256": _sha256(security_index),
                "security_count": len(security_ids),
            },
            "rad_events": (
                {"path": str(rad_events_path), "sha256": _sha256(rad_events_path)}
                if rad_events_path is not None
                else None
            ),
            "availability_rule": (
                "Exact RAD Data Entrega is interpreted in America/Sao_Paulo and becomes "
                "available at the first canonical decision strictly after receipt. An "
                "exact 10:15 receipt therefore starts at decision 1, while an exact "
                "14:45 receipt starts at decision 0 of the next observed B3 session. "
                "Without RAD, date-only DT_RECEB starts at decision 0 of the next "
                "observed session. FCA date-only receipts also use next-session open."
            ),
            "rad_unmatched_rule": (
                "A structured version absent from exact public RAD history is dropped "
                "and recorded in rad_unmatched_documents; no minute timestamp is inferred "
                "and no sibling version's timestamp is borrowed."
            ),
            "version_rule": (
                "For each issuer/category/reference date, use the highest document "
                "version whose receipt is available; later revisions update state only "
                "from their own availability."
            ),
            "identity_rule": (
                "Latest available FCA version and effective cash ticker, mapped through "
                "same-or-prior COTAHIST ticker observation to permanent security_id; "
                "issuer values broadcast to every mapped canonical active share class."
            ),
            "statement_rule": (
                "ORDEM_EXERC=ÚLTIMO, ST_CONTA_FIXA=S, fixed CD_CONTA codes only; "
                "consolidated core preferred and individual core used only as fallback."
            ),
            "scale_rule": (
                "MOEDA=REAL only; ESCALA_MOEDA UNIDADE=1 and MIL=1000 before ratios."
            ),
            "flow_rule": (
                "ITR/DFP cumulative DRE and DFC values are differenced within the same "
                "fiscal-period start into standalone quarters, then four consecutive "
                "quarters form TTM. Current values never enter prior periods."
            ),
            "valuation_omission": (
                "B/M and E/P omitted: issuer market cap is not auditable from total "
                "capital shares plus one class price for units and multiple preferred "
                "classes; no contaminated valuation denominator is substituted."
            ),
            "financial_rule": (
                "PIT FCA sector flag is explicit; net margin, sales growth, and accrual "
                "comparisons are masked for financial issuers."
            ),
            "account_codes": {
                metric: {"statement": statement, "CD_CONTA": code}
                for metric, (statement, code) in ACCOUNT_CODES.items()
            },
            "features": list(FEATURES),
            "available_start": available_start.isoformat(),
            "available_end": available_end.isoformat(),
            "calendar_session_count": len(sessions),
            "calendar_first_session": sessions[0].isoformat(),
            "calendar_last_session": sessions[-1].isoformat(),
            "factor_issuer_count": len(factor_events),
            **fca_audit,
            **filing_audit,
            **output_audit,
            "first_available_date": str(frame.get_column("available_date").min()),
            "last_available_date": str(frame.get_column("available_date").max()),
            "output_file": data_path.name,
            "output_sha256": _sha256(data_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists() and temporary.parent == output_dir.parent:
            shutil.rmtree(temporary)
        raise
    return manifest


def _parse_date_argument(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build point-in-time CVM structured-fundamental features"
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--cotahist-dir", type=Path, required=True)
    parser.add_argument("--security-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--available-start", type=_parse_date_argument, required=True)
    parser.add_argument("--available-end", type=_parse_date_argument, required=True)
    parser.add_argument("--rad-events", type=Path)
    args = parser.parse_args()
    manifest = build_artifact(
        args.raw_dir,
        args.cotahist_dir,
        args.security_index,
        args.output_dir,
        available_start=args.available_start,
        available_end=args.available_end,
        rad_events_path=args.rad_events,
    )
    print(
        f"Wrote {manifest['output_row_count']:,} PIT fundamental rows to "
        f"{args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
