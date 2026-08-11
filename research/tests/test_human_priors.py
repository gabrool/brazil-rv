from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from collections.abc import Callable
from datetime import UTC, date, datetime
from html import escape
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing.human_priors import (
    CLASSIFICATION_API_BASE,
    MARKET_CAP_API_BASE,
    MARKET_CAP_OMISSION_REASON,
    SCHEMA_VERSION,
    SELECTED_GROUPING_POLICY,
    UNITS_PAGE_URL,
    FeatureInputs,
    HttpPayload,
    acquire_official_sources,
    active_security_days,
    add_self_excluded_peer_counts,
    build_human_priors,
    classification_peer_audit,
    ingest_market_cap_directory,
    issuer_peer_audit,
    mapping_exception_audit,
    market_cap_audit,
    market_cap_manual_instructions,
    market_cap_month_coverage,
    normalize_unit_components,
    parse_classification_xlsx,
    parse_args,
    parse_market_cap_csv,
    parse_units_html,
    reconcile_market_cap,
    reconcile_security_metadata,
    select_peer_policy,
    strictly_lagged_market_cap_index,
    unit_overlap_audit,
    validate_reviewed_aliases,
    validate_raw_sources,
)


def _xlsx_bytes(rows: list[list[str]]) -> bytes:
    cells = []
    for row_number, row in enumerate(rows, start=1):
        values = []
        for column_number, value in enumerate(row, start=1):
            number = column_number
            letters = ""
            while number:
                number, remainder = divmod(number - 1, 26)
                letters = chr(ord("A") + remainder) + letters
            values.append(
                f'<c r="{letters}{row_number}" t="inlineStr"><is><t>'
                f"{escape(value)}</t></is></c>"
            )
        cells.append(f'<row r="{row_number}">{"".join(values)}</row>')
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main"><sheetData>'
        f"{''.join(cells)}</sheetData></worksheet>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        "</Types>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/'
        'spreadsheetml/2006/main"><sheets/></workbook>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


def _classification_xlsx() -> bytes:
    return _xlsx_bytes(
        [
            ["SETOR", "SUBSETOR", "SEGMENTO", "EMISSOR", "", ""],
            ["", "", "", "NOME DE PREGÃO", "CÓDIGO", "SEGMENTO DE NEGOCIAÇÃO"],
            ["Financeiro", "Intermediários", "Bancos", "BANCO X", "BANC", "N2"],
            ["Petróleo", "Petróleo e Gás", "Exploração", "ENERGIA Y", "ENRG", "NM"],
        ]
    )


def _market_cap_csv(
    reference_dates: tuple[date, ...] = (date(2026, 7, 31), date(2026, 6, 30)),
    issuer_names: tuple[str, ...] | None = None,
    missing_values: set[tuple[str, date]] | None = None,
) -> bytes:
    issuer_names = issuer_names or (
        "BANCO X",
        "ENERGIA Y",
        *(f"COMPANY {index:02d}" for index in range(1, 20)),
    )
    missing_values = missing_values or set()
    header = [
        "IBOV",
        "IBRX100",
        "Empresa",
        *(f"Valor (R$) em {value:%d/%m/%Y}" for value in reference_dates),
        "Var (%)",
    ]
    rows = ["20260804" + "|" * (len(header) - 1), "|".join(header)]
    for issuer_index, issuer_name in enumerate(issuer_names, start=1):
        values = [
            (
                ""
                if (issuer_name, reference_date) in missing_values
                else f"{1000 + issuer_index * 10 + offset:.2f}"
            )
            for offset, reference_date in enumerate(reference_dates)
        ]
        rows.append("|".join(("", "", issuer_name, *values, "0.00")))
    rows.append(
        "|".join(
            (
                "",
                "",
                f"Total Geral ({len(issuer_names)})",
                *("0.00" for _ in reference_dates),
                "0.00",
            )
        )
    )
    return ("\n".join(rows) + "\n").encode()


def _units_html() -> bytes:
    return (
        "<html><body><table>"
        "<tr><th>Nome</th><th>Código</th><th>Composição</th></tr>"
        "<tr><td>BANCO X</td><td>BANC11</td>"
        "<td>1 ação ON + 2 ações PN</td></tr>"
        "</table></body></html>"
    ).encode()


def _classification_frame() -> pl.DataFrame:
    retrieved = datetime(2026, 8, 11, tzinfo=UTC)
    return pl.DataFrame(
        {
            "issuer_id": ["B3_ISSUER_CODE:BANC", "B3_ISSUER_CODE:ENRG"],
            "issuer_b3_code": ["BANC", "ENRG"],
            "issuer_name": ["BANCO X", "ENERGIA Y"],
            "sector": ["Financeiro", "Petróleo"],
            "subsector": ["Intermediários", "Petróleo e Gás"],
            "economic_segment": ["Bancos", "Exploração"],
            "trading_segment": ["N2", "NM"],
            "classification_snapshot_date": [date(2026, 8, 11)] * 2,
            "source_file": ["classification.xlsx"] * 2,
            "source_url": ["https://b3.example/classification"] * 2,
            "retrieved_at_utc": [retrieved] * 2,
        }
    )


def _accepted_security_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    assignments = pl.DataFrame(
        {
            "security_id": ["SEC_ON", "SEC_PN", "SEC_FUZZY"],
            "isin": ["BRBANCACNOR0", "BRBANCACNPR9", "BRFUZZACNOR0"],
            "latest_ticker": ["BANC3", "BANC4", "FUZZ3"],
        }
    )
    ticker_history = pl.DataFrame(
        {
            "security_id": ["SEC_ON", "SEC_PN", "SEC_FUZZY"],
            "ticker": ["BANC3", "BANC4", "FUZZ3"],
            "valid_from": [date(2020, 1, 1)] * 3,
            "valid_to": [date(2099, 12, 31)] * 3,
            "isin": ["BRBANCACNOR0", "BRBANCACNPR9", "BRFUZZACNOR0"],
            "issuer_short_name": ["BANCO X", "BANCO X", "BANCO XX"],
            "security_spec": ["ON", "PN", "ON"],
        }
    )
    return assignments, ticker_history


def _reviewed_alias(
    *,
    alias_id: str = "ALIAS-001",
    source_key: str = "SEC_FUZZY",
    normalized_source_name: str = "",
    issuer_id: str = "B3_ISSUER_CODE:BANC",
    relationship_type: str = "NAME_CONTINUITY",
    effective_from: date | None = None,
    effective_to: date | None = None,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "alias_id": [alias_id],
            "source_type": ["security_classification"],
            "source_key": [source_key],
            "normalized_source_name": [normalized_source_name],
            "issuer_id": [issuer_id],
            "relationship_type": [relationship_type],
            "effective_from": [effective_from],
            "effective_to": [effective_to],
            "reviewed_by": ["reviewer@example.com"],
            "reviewed_at_utc": ["2026-08-11T20:00:00+00:00"],
            "review_reference": ["review-ticket-001"],
        },
        schema_overrides={
            "effective_from": pl.Date,
            "effective_to": pl.Date,
        },
    )


def test_parse_b3_classification_workbook() -> None:
    parsed = parse_classification_xlsx(
        _classification_xlsx(),
        date(2026, 8, 11),
        source_file="classification.xlsx",
    )

    assert parsed.select("issuer_id", "sector", "subsector").to_dicts() == [
        {
            "issuer_id": "B3_ISSUER_CODE:BANC",
            "sector": "Financeiro",
            "subsector": "Intermediários",
        },
        {
            "issuer_id": "B3_ISSUER_CODE:ENRG",
            "sector": "Petróleo",
            "subsector": "Petróleo e Gás",
        },
    ]


def test_parse_monthly_market_cap_long_form() -> None:
    parsed = parse_market_cap_csv(_market_cap_csv(), source_file="market.csv")

    assert parsed.height == 42
    banco = parsed.filter(pl.col("issuer_name") == "BANCO X")
    assert banco["reference_date"].to_list() == [
        date(2026, 6, 30),
        date(2026, 7, 31),
    ]
    assert banco["reference_month"].to_list() == ["2026-06", "2026-07"]
    assert banco["market_cap_brl"].to_list() == [1011.0, 1010.0]


def test_market_cap_export_rejects_twenty_company_truncation() -> None:
    with pytest.raises(ValueError, match="pagination-truncated"):
        parse_market_cap_csv(
            _market_cap_csv(issuer_names=tuple(f"ISSUER {i}" for i in range(20)))
        )


def test_market_cap_export_rejects_footer_count_mismatch() -> None:
    payload = _market_cap_csv().replace(b"Total Geral (21)", b"Total Geral (344)")
    with pytest.raises(ValueError, match="completeness footer"):
        parse_market_cap_csv(payload)


def test_parse_unit_composition_into_component_rows() -> None:
    parsed = parse_units_html(_units_html(), date(2026, 8, 11))

    assert parsed.select(
        "unit_ticker", "component_share_class", "component_quantity"
    ).to_dicts() == [
        {
            "unit_ticker": "BANC11",
            "component_share_class": "ON",
            "component_quantity": 1,
        },
        {
            "unit_ticker": "BANC11",
            "component_share_class": "PN",
            "component_quantity": 2,
        },
    ]


def test_deterministic_reconciliation_groups_share_classes_and_rejects_fuzzy() -> None:
    assignments, ticker_history = _accepted_security_inputs()
    metadata, review, exceptions = reconcile_security_metadata(
        assignments, ticker_history, _classification_frame()
    )

    mapped = metadata.filter(pl.col("mapping_status") == "MAPPED")
    assert mapped["security_id"].to_list() == ["SEC_ON", "SEC_PN"]
    assert mapped["issuer_id"].n_unique() == 1
    assert set(mapped["share_class"].to_list()) == {"ON", "PN"}
    fuzzy = review.filter(pl.col("security_id") == "SEC_FUZZY").row(named=True)
    assert fuzzy["mapping_status"] == "UNRESOLVED"
    assert fuzzy["matching_method"] == "UNRESOLVED"
    assert "B3_ISSUER_CODE:BANC" in fuzzy["fuzzy_candidates"]
    fuzzy_exceptions = exceptions.filter(pl.col("source_key") == "SEC_FUZZY")
    assert fuzzy_exceptions["exception_id"].n_unique() == 1
    assert fuzzy_exceptions["candidate_rank"].sort().to_list() == [1, 2]
    exception_audit = mapping_exception_audit(fuzzy_exceptions)
    assert exception_audit["mapping_exception_count"] == 1
    assert exception_audit["mapping_exception_candidate_row_count"] == 2


def test_reviewed_alias_maps_exact_source_key_with_full_provenance() -> None:
    assignments, ticker_history = _accepted_security_inputs()
    metadata, _, exceptions = reconcile_security_metadata(
        assignments,
        ticker_history,
        _classification_frame(),
        _reviewed_alias(),
    )

    mapped = metadata.filter(pl.col("security_id") == "SEC_FUZZY").row(named=True)
    assert mapped["mapping_status"] == "MAPPED"
    assert mapped["issuer_id"] == "B3_ISSUER_CODE:BANC"
    assert mapped["matching_method"] == "REVIEWED_ALIAS_NAME_CONTINUITY"
    assert mapped["reviewed_alias_alias_id"] == "ALIAS-001"
    assert mapped["reviewed_alias_reviewed_by"] == "reviewer@example.com"
    assert mapped["reviewed_alias_review_reference"] == "review-ticket-001"
    assert exceptions.filter(pl.col("source_key") == "SEC_FUZZY").is_empty()


def test_reviewed_alias_respects_effective_dates() -> None:
    assignments, ticker_history = _accepted_security_inputs()
    aliases = _reviewed_alias(
        source_key="",
        normalized_source_name="BANCOXX",
        effective_from=date(2027, 1, 1),
    )
    current, _, _ = reconcile_security_metadata(
        assignments, ticker_history, _classification_frame(), aliases
    )
    assert (
        current.filter(pl.col("security_id") == "SEC_FUZZY")["mapping_status"].item()
        == "UNRESOLVED"
    )

    future_classification = _classification_frame().with_columns(
        pl.lit(date(2027, 1, 2)).alias("classification_snapshot_date")
    )
    future, _, _ = reconcile_security_metadata(
        assignments, ticker_history, future_classification, aliases
    )
    assert (
        future.filter(pl.col("security_id") == "SEC_FUZZY")["matching_method"].item()
        == "REVIEWED_ALIAS_NAME_CONTINUITY"
    )


def test_reviewed_alias_validation_rejects_conflicting_overlaps() -> None:
    aliases = pl.concat(
        [
            _reviewed_alias(
                alias_id="ALIAS-001",
                source_key="",
                normalized_source_name="BANCOXX",
                effective_to=date(2026, 12, 31),
            ),
            _reviewed_alias(
                alias_id="ALIAS-002",
                source_key="",
                normalized_source_name="BANCOXX",
                issuer_id="B3_ISSUER_CODE:ENRG",
                effective_from=date(2026, 6, 1),
            ),
        ]
    )

    with pytest.raises(ValueError, match="conflicting, or overlapping"):
        validate_reviewed_aliases(aliases, _classification_frame())


def test_merger_successor_is_unresolved_without_explicit_reviewed_alias() -> None:
    assignments, ticker_history = _accepted_security_inputs()
    ticker_history = ticker_history.with_columns(
        pl.when(pl.col("security_id") == "SEC_FUZZY")
        .then(pl.lit("BANCO X SUCCESSOR"))
        .otherwise(pl.col("issuer_short_name"))
        .alias("issuer_short_name")
    )
    unresolved, _, _ = reconcile_security_metadata(
        assignments, ticker_history, _classification_frame()
    )
    assert (
        unresolved.filter(pl.col("security_id") == "SEC_FUZZY")["mapping_status"].item()
        == "UNRESOLVED"
    )

    explicit, _, _ = reconcile_security_metadata(
        assignments,
        ticker_history,
        _classification_frame(),
        _reviewed_alias(relationship_type="MERGER_SUCCESSOR"),
    )
    assert (
        explicit.filter(pl.col("security_id") == "SEC_FUZZY")["matching_method"].item()
        == "REVIEWED_ALIAS_MERGER_SUCCESSOR"
    )


def test_security_reviewed_alias_does_not_leak_into_market_cap_matching() -> None:
    assignments, ticker_history = _accepted_security_inputs()
    classification = _classification_frame()
    aliases = _reviewed_alias()
    metadata, _, _ = reconcile_security_metadata(
        assignments, ticker_history, classification, aliases
    )
    market_cap = pl.DataFrame(
        {
            "reference_date": [date(2026, 8, 11)],
            "reference_month": ["2026-08"],
            "issuer_name": ["BANCOXX"],
            "market_cap_brl": [100.0],
            "source_file": ["market.csv"],
            "source_url": ["https://b3.example/market"],
            "retrieved_at_utc": [datetime(2026, 8, 12, tzinfo=UTC)],
        }
    )

    normalized, exceptions, _ = reconcile_market_cap(
        market_cap, classification, metadata, aliases
    )

    assert normalized.is_empty()
    assert exceptions["exception_id"].n_unique() == 1
    assert exceptions["source_type"].unique().to_list() == ["market_cap"]
    assert exceptions["reason"].unique().to_list() == ["NO_EXACT_ISSUER_NAME"]


def test_self_excluded_peer_counts_respect_pit_membership_and_readiness() -> None:
    dates = [date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5)]
    security_ids = ["SEC_ON", "SEC_PN", "SEC_OTHER"]
    membership = np.array(
        [[True, True, True], [True, True, True], [True, False, True]], dtype=bool
    )
    readiness = np.array(
        [[True, True, True], [True, False, True], [True, True, True]], dtype=bool
    )
    active, eligible = active_security_days(
        dates, security_ids, membership, readiness, minimum_active=3
    )
    metadata = pl.DataFrame(
        {
            "security_id": security_ids,
            "issuer_id": ["ISSUER_X", "ISSUER_X", "ISSUER_Y"],
            "sector": ["S", "S", "S"],
            "subsector": ["SS", "SS", "OTHER"],
            "economic_segment": ["E", "E", "OTHER"],
        }
    )
    peers = add_self_excluded_peer_counts(active, metadata)

    assert eligible.tolist() == [True, False, False]
    assert peers.height == 3
    first = peers.filter(pl.col("security_id") == "SEC_ON").row(named=True)
    assert first["same_issuer_peer_count"] == 1
    assert first["same_sector_peer_count"] == 2
    assert first["same_subsector_peer_count"] == 1
    assert first["same_economic_segment_peer_count"] == 1


def _single_day_peer_counts(metadata: pl.DataFrame) -> pl.DataFrame:
    security_ids = metadata["security_id"].to_list()
    active = pl.DataFrame(
        {
            "date_idx": [0] * len(security_ids),
            "trade_date": [date(2026, 8, 11)] * len(security_ids),
            "equity_slot": range(len(security_ids)),
            "security_id": security_ids,
        }
    )
    return add_self_excluded_peer_counts(active, metadata)


def test_identical_subsector_labels_under_different_sectors_are_not_peers() -> None:
    metadata = pl.DataFrame(
        {
            "security_id": ["A1", "A2", "B1", "B2"],
            "issuer_id": ["IA1", "IA2", "IB1", "IB2"],
            "sector": ["Sector A", "Sector A", "Sector B", "Sector B"],
            "subsector": ["Shared", "Shared", "Shared", "Shared"],
            "economic_segment": ["A1", "A2", "B1", "B2"],
        }
    )
    peers = _single_day_peer_counts(metadata)

    assert peers["same_subsector_peer_count"].to_list() == [1, 1, 1, 1]
    assert peers["subsector_group_key"].n_unique() == 2


def test_identical_economic_segments_under_different_subsectors_are_not_peers() -> None:
    metadata = pl.DataFrame(
        {
            "security_id": ["A1", "A2", "B1", "B2"],
            "issuer_id": ["IA1", "IA2", "IB1", "IB2"],
            "sector": ["Sector"] * 4,
            "subsector": ["Parent A", "Parent A", "Parent B", "Parent B"],
            "economic_segment": ["Shared", "Shared", "Shared", "Shared"],
        }
    )
    peers = _single_day_peer_counts(metadata)

    assert peers["same_economic_segment_peer_count"].to_list() == [1, 1, 1, 1]
    assert peers["economic_segment_group_key"].n_unique() == 2


def test_selected_peer_policy_uses_dated_active_counts_and_sector_fallback() -> None:
    metadata = pl.DataFrame(
        {
            "security_id": ["A1", "A2", "A3", "A4"],
            "issuer_id": ["IA1", "IA2", "IA3", "IA4"],
            "sector": ["Sector"] * 4,
            "subsector": ["Shared", "Shared", "Shared", "Other"],
            "economic_segment": ["EA1", "EA2", "EA3", "EA4"],
        }
    )
    active = pl.DataFrame(
        {
            "date_idx": [0, 0, 0, 0, 1, 1, 1],
            "trade_date": [date(2026, 8, 10)] * 4 + [date(2026, 8, 11)] * 3,
            "equity_slot": [0, 1, 2, 3, 0, 1, 3],
            "security_id": ["A1", "A2", "A3", "A4", "A1", "A2", "A4"],
        }
    )
    selected = select_peer_policy(add_self_excluded_peer_counts(active, metadata))
    first = selected.filter(
        (pl.col("security_id") == "A1") & (pl.col("date_idx") == 0)
    ).row(named=True)
    second = selected.filter(
        (pl.col("security_id") == "A1") & (pl.col("date_idx") == 1)
    ).row(named=True)

    assert first["same_subsector_peer_count"] == 2
    assert first["selected_peer_relation"] == "SUBSECTOR"
    assert first["selected_other_active_peer_count"] == 2
    assert first["sector_fallback_used"] is False
    assert first["selected_peer_group_key"] == '["Sector","Shared"]'
    assert second["same_subsector_peer_count"] == 1
    assert second["same_sector_peer_count"] == 2
    assert second["selected_peer_relation"] == "SECTOR"
    assert second["selected_other_active_peer_count"] == 2
    assert second["sector_fallback_used"] is True
    assert second["selected_peer_group_key"] == '["Sector"]'


def test_selected_subsector_keys_remain_parent_qualified() -> None:
    metadata = pl.DataFrame(
        {
            "security_id": ["A1", "A2", "A3", "B1", "B2", "B3"],
            "issuer_id": ["IA1", "IA2", "IA3", "IB1", "IB2", "IB3"],
            "sector": ["Sector A"] * 3 + ["Sector B"] * 3,
            "subsector": ["Shared"] * 6,
            "economic_segment": ["EA1", "EA2", "EA3", "EB1", "EB2", "EB3"],
        }
    )
    selected = select_peer_policy(_single_day_peer_counts(metadata))

    assert selected["selected_peer_relation"].unique().to_list() == ["SUBSECTOR"]
    assert selected["selected_other_active_peer_count"].unique().to_list() == [2]
    assert set(selected["selected_peer_group_key"].unique().to_list()) == {
        '["Sector A","Shared"]',
        '["Sector B","Shared"]',
    }


def test_candidate_policy_coverage_uses_hierarchical_subsector_groups() -> None:
    metadata = pl.DataFrame(
        {
            "security_id": ["A1", "A2", "B1", "B2"],
            "issuer_id": ["IA1", "IA2", "IB1", "IB2"],
            "sector": ["Sector A", "Sector A", "Sector B", "Sector B"],
            "subsector": ["Shared", "Other A", "Shared", "Other B"],
            "economic_segment": ["EA1", "EA2", "EB1", "EB2"],
        }
    )
    peers = _single_day_peer_counts(metadata)
    audit, group_sizes = classification_peer_audit(
        peers,
        metadata,
        metadata.select("sector", "subsector", "economic_segment"),
    )

    policies = audit["candidate_policy_coverage"]
    assert policies["subsector_only"]["percentage_with_at_least"]["1"] == 0.0
    assert (
        policies["subsector_if_at_least_one_other_else_sector"][
            "percentage_with_at_least"
        ]["1"]
        == 100.0
    )
    assert (
        policies["subsector_if_at_least_two_others_else_sector"][
            "percentage_with_at_least"
        ]["1"]
        == 100.0
    )
    shared = group_sizes.filter(
        (pl.col("level") == "subsector") & (pl.col("group") == "Shared")
    )
    assert shared.height == 2
    assert shared["group_key"].n_unique() == 2
    assert audit["taxonomy_label_collisions"]["subsector"] == [
        {
            "label": "Shared",
            "hierarchical_groups": [
                {
                    "group_key": '["Sector A","Shared"]',
                    "sector": "Sector A",
                    "subsector": "Shared",
                },
                {
                    "group_key": '["Sector B","Shared"]',
                    "sector": "Sector B",
                    "subsector": "Shared",
                },
            ],
        }
    ]


def test_strict_market_cap_lag_excludes_same_day() -> None:
    references = [date(2026, 5, 31), date(2026, 6, 30), date(2026, 7, 31)]

    assert strictly_lagged_market_cap_index(references, date(2026, 5, 31)) is None
    assert strictly_lagged_market_cap_index(references, date(2026, 6, 30)) == 0
    assert strictly_lagged_market_cap_index(references, date(2026, 7, 1)) == 1


def test_market_cap_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        parse_args(["build", "--allow-incomplete-market-cap", "--omit-market-cap"])
    parsed = parse_args(["build", "--omit-market-cap"])
    assert parsed.omit_market_cap is True
    assert parsed.allow_incomplete_market_cap is False
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_human_priors(
            allow_incomplete_market_cap=True,
            omit_market_cap=True,
        )


def test_market_cap_instructions_describe_supported_paths_only(tmp_path: Path) -> None:
    instructions = market_cap_manual_instructions(tmp_path)

    assert "latest two reference months" in instructions
    assert "already possess them or obtain them directly from B3" in instructions
    assert "Unsupported request parameters" in instructions
    assert "unofficial mirrors" in instructions
    assert "CVM" in instructions
    assert "Receita Federal" in instructions
    assert "third-party substitutions" in instructions
    assert "--omit-market-cap" in instructions
    assert MARKET_CAP_OMISSION_REASON.startswith(
        "Market cap was explicitly omitted because"
    )


def test_market_cap_coverage_uses_calendar_months_not_calendar_end_dates() -> None:
    coverage = market_cap_month_coverage(
        [date(2021, 1, 29), date(2021, 2, 26), date(2021, 3, 30)],
        as_of=date(2021, 4, 15),
        required_start_month="2021-01",
    )

    assert coverage["available_reference_months"] == [
        "2021-01",
        "2021-02",
        "2021-03",
    ]
    assert coverage["missing_reference_months"] == []
    assert coverage["market_cap_history_complete"] is True


def test_market_cap_coverage_reports_a_genuinely_missing_calendar_month() -> None:
    coverage = market_cap_month_coverage(
        [date(2021, 1, 29), date(2021, 3, 30)],
        as_of=date(2021, 4, 1),
        required_start_month="2021-01",
    )

    assert coverage["missing_reference_months"] == ["2021-02"]
    assert coverage["market_cap_history_complete"] is False


def test_batch_manual_ingestion_validates_every_file_before_cache_mutation(
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "Bolsa_Valores_Mensal_2021-01.csv").write_bytes(
        _market_cap_csv((date(2021, 1, 29),))
    )
    (downloads / "Bolsa_Valores_Mensal_2021-02.csv").write_bytes(
        _market_cap_csv((date(2021, 2, 26),))
    )
    malformed = downloads / "unexpected.csv"
    malformed.write_bytes(b"not an official export")
    raw_dir = tmp_path / "raw"

    with pytest.raises(ValueError, match="unexpected CSV filenames"):
        ingest_market_cap_directory(downloads, raw_dir)
    assert not (raw_dir / "market_cap").exists()

    malformed.unlink()
    sources = ingest_market_cap_directory(
        downloads,
        raw_dir,
        retrieved_at=datetime(2021, 3, 2, tzinfo=UTC),
    )

    assert len(sources) == 2
    assert [source.reference_dates for source in sources] == [
        (date(2021, 1, 29),),
        (date(2021, 2, 26),),
    ]


def test_same_issuer_audit_reports_share_class_groups_and_single_member_periods() -> (
    None
):
    dates = [date(2026, 8, 3), date(2026, 8, 4), date(2026, 8, 5)]
    security_ids = ["SEC_ON", "SEC_PN", "SEC_OTHER"]
    membership = np.array(
        [[True, False, True], [True, True, True], [False, True, True]], dtype=bool
    )
    readiness = np.ones_like(membership)
    active, eligible = active_security_days(
        dates, security_ids, membership, readiness, minimum_active=2
    )
    metadata = pl.DataFrame(
        {
            "security_id": security_ids,
            "ticker": ["BANC3", "BANC4", "OTHR3"],
            "issuer_id": ["ISSUER_X", "ISSUER_X", "ISSUER_Y"],
            "issuer_name": ["BANCO X", "BANCO X", "OTHER"],
            "share_class": ["ON", "PN", "ON"],
            "sector": ["S", "S", "S"],
            "subsector": ["SS", "SS", "OTHER"],
            "economic_segment": ["E", "E", "OTHER"],
        }
    )
    peers = add_self_excluded_peer_counts(active, metadata)

    audit, groups = issuer_peer_audit(
        peers, metadata, dates, security_ids, eligible, membership
    )

    assert audit["issuers_with_multiple_accepted_securities"] == 1
    assert audit[
        "security_day_coverage_with_at_least_one_active_same_issuer_peer_percent"
    ] == pytest.approx(200 / 7)
    group = groups.row(named=True)
    assert group["security_ids"] == "SEC_ON | SEC_PN"
    assert group["share_classes"] == "ON | PN"
    assert group["single_pit_member_period_count"] == 2


def test_market_cap_audit_joins_only_strictly_prior_months() -> None:
    peers = pl.DataFrame(
        {
            "trade_date": [
                date(2026, 5, 31),
                date(2026, 6, 30),
                date(2026, 7, 1),
            ],
            "security_id": ["SEC_ON"] * 3,
            "issuer_id": ["ISSUER_X"] * 3,
        }
    )
    market_cap = pl.DataFrame(
        {
            "reference_date": [date(2026, 5, 31), date(2026, 6, 30)],
            "issuer_id": ["ISSUER_X"] * 2,
            "issuer_name": ["BANCO X"] * 2,
            "market_cap_brl": [100.0, 110.0],
            "matching_method": ["EXACT"] * 2,
            "source_file": ["market.csv"] * 2,
            "source_url": ["https://b3.example/market"] * 2,
            "retrieved_at_utc": [datetime(2026, 8, 1, tzinfo=UTC)] * 2,
        }
    )
    metadata = pl.DataFrame(
        {
            "security_id": ["SEC_ON"],
            "issuer_id": ["ISSUER_X"],
            "issuer_name": ["BANCO X"],
        }
    )
    mapping_stats = {
        "source_row_count": 2,
        "source_distinct_issuer_names": 1,
        "mapped_distinct_issuer_names": 1,
        "issuer_name_mapping_fraction": 1.0,
        "total_normalized_market_cap_row_count": 2,
        "total_normalized_market_cap_issuer_count": 1,
        "duplicate_issuer_month_groups": 0,
        "conflicting_issuer_month_groups": 0,
    }

    audit, coverage = market_cap_audit(
        peers, market_cap, metadata, mapping_stats, as_of=date(2026, 8, 1)
    )

    assert audit["eligible_security_day_count"] == 3
    assert audit["strictly_lagged_joined_security_day_count"] == 2
    assert audit["strictly_lagged_security_day_coverage_fraction"] == pytest.approx(
        2 / 3
    )
    assert audit["staleness_days_distribution"]["minimum"] == 1.0
    assert coverage.row(named=True)["strictly_lagged_joined_security_day_count"] == 2


def test_issuer_discontinuity_uses_calendar_month_keys() -> None:
    peers = pl.DataFrame(
        schema={
            "trade_date": pl.Date,
            "security_id": pl.String,
            "issuer_id": pl.String,
        }
    )
    market_cap = pl.DataFrame(
        {
            "reference_date": [date(2026, 1, 30), date(2026, 3, 30)],
            "reference_month": ["2026-01", "2026-03"],
            "issuer_id": ["ISSUER_X"] * 2,
            "issuer_name": ["BANCO X"] * 2,
            "market_cap_brl": [100.0, 110.0],
            "matching_method": ["EXACT"] * 2,
            "source_file": ["market.csv"] * 2,
            "source_url": ["https://b3.example/market"] * 2,
            "retrieved_at_utc": [datetime(2026, 4, 1, tzinfo=UTC)] * 2,
        }
    )
    metadata = pl.DataFrame(
        {
            "security_id": ["SEC_ON"],
            "issuer_id": ["ISSUER_X"],
            "issuer_name": ["BANCO X"],
        }
    )
    mapping_stats = {
        "source_row_count": 2,
        "source_distinct_issuer_names": 1,
        "mapped_distinct_issuer_names": 1,
        "issuer_name_mapping_fraction": 1.0,
        "total_normalized_market_cap_row_count": 2,
        "total_normalized_market_cap_issuer_count": 1,
        "duplicate_issuer_month_groups": 0,
        "conflicting_issuer_month_groups": 0,
    }

    audit, _ = market_cap_audit(
        peers,
        market_cap,
        metadata,
        mapping_stats,
        as_of=date(2026, 4, 1),
        source_reference_dates=market_cap["reference_date"].to_list(),
    )

    assert audit["issuer_discontinuities"] == [
        {
            "issuer_id": "ISSUER_X",
            "issuer_name": "BANCO X",
            "missing_months_between_first_and_last": ["2026-02"],
        }
    ]


def test_unit_component_normalization_preserves_missing_universe_component() -> None:
    classification = _classification_frame().head(1)
    units = parse_units_html(_units_html(), date(2026, 8, 11))
    security_metadata = pl.DataFrame(
        {
            "security_id": ["SEC_UNIT", "SEC_ON"],
            "ticker": ["BANC11", "BANC3"],
            "issuer_id": ["B3_ISSUER_CODE:BANC"] * 2,
            "issuer_name": ["BANCO X"] * 2,
            "cotahist_issuer_name": ["BANCO X"] * 2,
            "share_class": ["UNIT", "ON"],
        }
    )

    components, exceptions = normalize_unit_components(
        units, classification, security_metadata
    )
    on = components.filter(pl.col("component_share_class") == "ON").row(named=True)
    pn = components.filter(pl.col("component_share_class") == "PN").row(named=True)
    assert on["component_security_id"] == "SEC_ON"
    assert on["mapping_status"] == "MAPPED"
    assert pn["component_security_id"] is None
    assert pn["mapping_status"] == "COMPONENT_NOT_IN_ACCEPTED_UNIVERSE"
    assert exceptions.is_empty()


def _normalized_unit_components(
    *,
    pn_security_id: str | None = "SEC_PN",
    pn_status: str = "MAPPED",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "unit_security_id": ["SEC_UNIT", "SEC_UNIT"],
            "unit_ticker": ["BANC11", "BANC11"],
            "component_issuer_id": ["ISSUER_X", "ISSUER_X"],
            "component_share_class": ["ON", "PN"],
            "component_security_id": ["SEC_ON", pn_security_id],
            "component_quantity": [1, 2],
            "mapping_status": ["MAPPED", pn_status],
            "unit_snapshot_date": [date(2026, 8, 11)] * 2,
            "source_file": ["units.html"] * 2,
            "source_url": ["https://b3.example/units"] * 2,
        }
    )


def _unit_parity(
    membership: np.ndarray,
    readiness: np.ndarray,
    *,
    components: pl.DataFrame | None = None,
) -> tuple[dict[str, object], pl.DataFrame, pl.DataFrame]:
    return unit_overlap_audit(
        _normalized_unit_components() if components is None else components,
        ["SEC_UNIT", "SEC_ON", "SEC_PN"],
        np.ones(membership.shape[0], dtype=bool),
        membership,
        readiness,
    )


def test_unit_parity_tracks_all_available_components_at_unit_level() -> None:
    audit, component_rows, unit_rows = _unit_parity(
        np.ones((2, 3), dtype=bool),
        np.ones((2, 3), dtype=bool),
    )

    assert component_rows.height == 2
    assert audit["mapped_component_security_rows"] == 2
    row = unit_rows.row(named=True)
    assert row["required_component_count"] == 2
    assert row["mapped_component_count"] == 2


def test_unit_parity_rejects_one_absent_component() -> None:
    _, _, unit_rows = _unit_parity(
        np.ones((2, 3), dtype=bool),
        np.ones((2, 3), dtype=bool),
        components=_normalized_unit_components(
            pn_security_id=None,
            pn_status="COMPONENT_NOT_IN_ACCEPTED_UNIVERSE",
        ),
    )

    row = unit_rows.row(named=True)
    assert row["mapped_component_count"] == 1
    assert row["exact_parity_possible"] is False
    assert row["exact_parity_limitation"] == "MISSING_OR_AMBIGUOUS_COMPONENTS"
    assert (
        "PN:COMPONENT_NOT_IN_ACCEPTED_UNIVERSE"
        in row["missing_or_ambiguous_components"]
    )


def test_unit_parity_rejects_components_never_simultaneously_pit_members() -> None:
    membership = np.array(
        [[True, True, False], [True, False, True]],
        dtype=bool,
    )
    _, _, unit_rows = _unit_parity(
        membership,
        np.ones_like(membership),
    )

    row = unit_rows.row(named=True)
    assert row["pit_all_component_overlap_date_count"] == 0
    assert row["exact_parity_possible"] is False
    assert row["exact_parity_limitation"] == "NO_SIMULTANEOUS_PIT_MEMBERSHIP"


def test_unit_parity_rejects_pit_overlap_without_all_component_readiness() -> None:
    membership = np.ones((2, 3), dtype=bool)
    readiness = np.array(
        [[True, True, False], [True, True, False]],
        dtype=bool,
    )
    _, _, unit_rows = _unit_parity(membership, readiness)

    row = unit_rows.row(named=True)
    assert row["pit_all_component_overlap_date_count"] == 2
    assert row["m1_ready_all_component_overlap_date_count"] == 0
    assert row["exact_parity_possible"] is False
    assert row["exact_parity_limitation"] == "NO_SIMULTANEOUS_M1_READINESS"


def test_multi_component_unit_parity_requires_one_fully_active_overlap() -> None:
    membership = np.array(
        [[True, True, True], [True, True, False], [True, False, True]],
        dtype=bool,
    )
    readiness = np.array(
        [[True, True, True], [True, True, True], [True, True, True]],
        dtype=bool,
    )
    audit, _, unit_rows = _unit_parity(membership, readiness)

    row = unit_rows.row(named=True)
    assert row["pit_all_component_overlap_date_count"] == 1
    assert row["m1_ready_all_component_overlap_date_count"] == 3
    assert row["fully_active_all_component_overlap_date_count"] == 1
    assert row["exact_parity_possible"] is True
    assert row["exact_parity_limitation"] == ""
    assert audit["units_with_exact_parity_possible"] == 1


def test_acquisition_is_cached_and_network_free_on_repeat(tmp_path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> HttpPayload:
        calls.append(url)
        if url.startswith(CLASSIFICATION_API_BASE):
            return HttpPayload(_classification_xlsx(), "application/octet-stream", url)
        if url.startswith(MARKET_CAP_API_BASE):
            return HttpPayload(base64.b64encode(_market_cap_csv()), "text/plain", url)
        assert url == UNITS_PAGE_URL
        return HttpPayload(_units_html(), "text/html", url)

    first = acquire_official_sources(
        tmp_path, now=datetime(2026, 8, 10, 14, 30, tzinfo=UTC), fetch=fetch
    )
    refreshed = acquire_official_sources(
        tmp_path, now=datetime(2026, 8, 11, 14, 30, tzinfo=UTC), fetch=fetch
    )
    second = acquire_official_sources(
        tmp_path, now=datetime(2026, 8, 11, 18, 0, tzinfo=UTC), fetch=fetch
    )

    assert len(calls) == 6
    assert {kind: source.sha256 for kind, source in first.items()} == {
        kind: source.sha256 for kind, source in second.items()
    }
    assert all(
        source.retrieved_at_utc.date() == date(2026, 8, 11)
        for source in refreshed.values()
    )
    assert all(
        len(list((tmp_path / kind).glob(f"*.{suffix}"))) == 1
        for kind, suffix in (
            ("classification", "xlsx"),
            ("market_cap", "csv"),
            ("units", "html"),
        )
    )
    assert (tmp_path / "manifest.json").is_file()


@pytest.mark.parametrize(
    ("parser", "payload"),
    [
        (
            lambda data: parse_classification_xlsx(data, date(2026, 8, 11)),
            b"<html>error</html>",
        ),
        (parse_market_cap_csv, b"unexpected|columns\n1|2\n"),
        (
            lambda data: parse_units_html(data, date(2026, 8, 11)),
            b"<html><table></table></html>",
        ),
    ],
)
def test_malformed_responses_and_schema_drift_fail_clearly(parser, payload) -> None:
    with pytest.raises(ValueError):
        parser(payload)


def _classification_xlsx_for_issuers(issuer_names: tuple[str, ...]) -> bytes:
    rows = [
        ["SETOR", "SUBSETOR", "SEGMENTO", "EMISSOR", "", ""],
        ["", "", "", "NOME DE PREGÃO", "CÓDIGO", "SEGMENTO DE NEGOCIAÇÃO"],
    ]
    for index, issuer_name in enumerate(issuer_names):
        rows.append(
            [
                "Financeiro" if index == 0 else "",
                "Intermediários" if index == 0 else "",
                "Bancos" if index == 0 else "",
                issuer_name,
                "BANC" if index == 0 else f"C{index:03d}",
                "NM",
            ]
        )
    return _xlsx_bytes(rows)


def _populate_raw_fixture(
    raw_dir: Path,
    *,
    issuer_names: tuple[str, ...],
    reference_dates: tuple[date, ...],
    market_cap_payload: bytes | None = None,
    classification_issuer_names: tuple[str, ...] | None = None,
) -> None:
    classification = _classification_xlsx_for_issuers(
        issuer_names
        if classification_issuer_names is None
        else classification_issuer_names
    )
    market_cap = (
        _market_cap_csv(reference_dates, issuer_names)
        if market_cap_payload is None
        else market_cap_payload
    )

    def fetch(url: str) -> HttpPayload:
        if url.startswith(CLASSIFICATION_API_BASE):
            return HttpPayload(classification, "application/octet-stream", url)
        if url.startswith(MARKET_CAP_API_BASE):
            return HttpPayload(base64.b64encode(market_cap), "text/plain", url)
        assert url == UNITS_PAGE_URL
        return HttpPayload(_units_html(), "text/html", url)

    acquire_official_sources(
        raw_dir,
        now=datetime(2020, 8, 15, 12, 0, tzinfo=UTC),
        fetch=fetch,
    )


def _canonical_build_fixture(
    root: Path,
) -> tuple[
    FeatureInputs,
    tuple[str, ...],
    tuple[str, ...],
    Callable[[Path], pl.DataFrame],
]:
    universe_dir = root / "universe"
    assignments_dir = root / "assignments"
    cotahist_dir = root / "cotahist"
    feature_store = root / "feature_store"
    pointers = root / "pointers"
    for directory in (
        universe_dir,
        assignments_dir,
        cotahist_dir,
        feature_store,
        pointers,
    ):
        directory.mkdir(parents=True)

    issuer_names = ("BANCO X", *(f"COMPANY {index:02d}" for index in range(1, 28)))
    security_ids = tuple(f"SEC{index:03d}" for index in range(30))
    security_issuer_names = (
        "BANCO X",
        "BANCO X",
        "BANCO X",
        *(f"COMPANY {index:02d}" for index in range(1, 28)),
    )
    tickers = (
        "BANC11",
        "BANC3",
        "BANC4",
        *(f"C{index:03d}3" for index in range(1, 28)),
    )
    share_classes = ("UNIT", "ON", "PN", *("ON" for _ in range(27)))
    isins = tuple(f"BRFIXTURE{index:04d}" for index in range(30))
    assignments = pl.DataFrame(
        {
            "security_id": security_ids,
            "isin": isins,
            "latest_ticker": tickers,
        }
    )
    assignments.write_parquet(
        assignments_dir / "xp_accepted_source_assignments_v1.parquet"
    )
    (assignments_dir / "decision_manifest.json").write_text(
        json.dumps({"fixture": True}), encoding="utf-8"
    )
    ticker_history = pl.DataFrame(
        {
            "security_id": security_ids,
            "ticker": tickers,
            "valid_from": [date(2020, 1, 1)] * 30,
            "valid_to": [date(2099, 12, 31)] * 30,
            "isin": isins,
            "issuer_short_name": security_issuer_names,
            "security_spec": share_classes,
        }
    )
    ticker_history.write_parquet(universe_dir / "ticker_history.parquet")
    (universe_dir / "manifest.json").write_text(
        json.dumps({"fixture": True}), encoding="utf-8"
    )
    (cotahist_dir / "parse_audit.json").write_text(
        json.dumps({"fixture": True}), encoding="utf-8"
    )

    dates = [date(2020, 8, 3), date(2020, 8, 4), date(2020, 8, 5)]
    pl.DataFrame({"date_idx": range(len(dates)), "trade_date": dates}).write_parquet(
        feature_store / "date_index.parquet"
    )
    pl.DataFrame({"equity_slot": range(30), "security_id": security_ids}).write_parquet(
        feature_store / "equity_index.parquet"
    )
    np.save(feature_store / "equity_membership.npy", np.ones((3, 30), dtype=bool))
    np.save(feature_store / "equity_data_ready.npy", np.ones((3, 30), dtype=bool))
    (feature_store / "manifest.json").write_text(
        json.dumps(
            {
                "eligible_date_count": 3,
                "canonical_inputs": {
                    "point_in_time_universe": {
                        "resolved_path": str(universe_dir.resolve())
                    },
                    "accepted_xp_assignments": {
                        "resolved_path": str(assignments_dir.resolve())
                    },
                    "parsed_cotahist": {"resolved_path": str(cotahist_dir.resolve())},
                },
            }
        ),
        encoding="utf-8",
    )

    pointer_values = {
        "universe": universe_dir,
        "assignments": assignments_dir,
        "cotahist": cotahist_dir,
        "feature_store": feature_store,
    }
    pointer_paths = {name: pointers / f"{name}_path.txt" for name in pointer_values}
    for name, path in pointer_paths.items():
        path.write_text(str(pointer_values[name].resolve()), encoding="utf-8")
    inputs = FeatureInputs(
        universe_dir=universe_dir,
        assignments_dir=assignments_dir,
        cotahist_dir=cotahist_dir,
        feature_store=feature_store,
        universe_pointer=pointer_paths["universe"],
        assignments_pointer=pointer_paths["assignments"],
        cotahist_pointer=pointer_paths["cotahist"],
        feature_store_pointer=pointer_paths["feature_store"],
    )
    return inputs, issuer_names, security_ids, lambda _: assignments


def test_build_human_priors_end_to_end_and_failure_semantics(tmp_path: Path) -> None:
    inputs, issuer_names, _, assignments_loader = _canonical_build_fixture(
        tmp_path / "canonical"
    )
    raw_complete = tmp_path / "raw_complete"
    _populate_raw_fixture(
        raw_complete,
        issuer_names=issuer_names,
        reference_dates=(date(2020, 7, 31), date(2020, 6, 30)),
    )
    output_base = tmp_path / "outputs"
    pointer = tmp_path / "human_priors_path.txt"
    output = build_human_priors(
        raw_dir=raw_complete,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 13, 0, tzinfo=UTC),
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )

    required = {
        "manifest.json",
        "security_metadata.parquet",
        "peer_group_index.parquet",
        "peer_policy_security_days.parquet",
        "issuer_market_cap_monthly.parquet",
        "unit_components.parquet",
        "metadata_audit.json",
        "metadata_audit.md",
        "sector_subsector_group_sizes.csv",
        "security_classification_review.csv",
        "issuer_peer_groups.csv",
        "unit_overlap_audit.csv",
        "unit_parity_coverage.csv",
        "market_cap_coverage.csv",
        "mapping_exceptions.csv",
    }
    assert {path.name for path in output.iterdir()} == required
    assert pointer.read_text(encoding="utf-8") == str(output.resolve())
    assert not list(output_base.glob("*.partial"))

    security_metadata = pl.read_parquet(output / "security_metadata.parquet")
    banco = security_metadata.filter(
        pl.col("ticker").is_in(["BANC11", "BANC3", "BANC4"])
    )
    assert banco["issuer_id"].n_unique() == 1
    parity = pl.read_csv(output / "unit_parity_coverage.csv").row(named=True)
    assert parity["required_component_count"] == 2
    assert parity["mapped_component_count"] == 2
    assert parity["exact_parity_possible"] is True
    assert (
        "exact_parity_possible"
        not in pl.read_csv(output / "unit_overlap_audit.csv").columns
    )

    audit = json.loads((output / "metadata_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert audit["build_mode"] == "complete"
    assert audit["market_cap_mode"] == "required"
    assert audit["market_cap_requested"] is True
    assert audit["market_cap_source_data_ready"] is True
    assert audit["market_cap_outputs_emitted"] is True
    assert audit["raw_market_cap_history_complete"] is True
    assert audit["usable_market_cap_history_complete"] is True
    assert audit["market_cap_data_ready"] is True
    assert audit["eligible_for_downstream_market_cap_features"] is True
    assert audit["grouping_policy_selected"] == SELECTED_GROUPING_POLICY
    assert audit["selected_peer_policy"]["minimum_other_active_subsector_peers"] == 2
    assert audit["selected_peer_policy"]["economic_segment_peers_enabled"] is False
    assert audit["selected_peer_policy"]["unit_parity_features_enabled"] is False
    assert audit["market_cap"]["raw_missing_reference_months"] == []
    assert audit["market_cap"]["usable_missing_reference_months"] == []
    assert audit["market_cap"]["total_normalized_market_cap_row_count"] == 56
    assert (
        audit["market_cap"]["accepted_universe_normalized_market_cap_row_count"] == 56
    )
    assert audit["market_cap"]["mapped_accepted_universe_issuer_count"] == 28
    assert audit["units"]["units_with_exact_parity_possible"] == 1
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["repository_commit"] == "fixture-commit"
    assert manifest["market_cap_mode"] == "required"
    assert manifest["market_cap_requested"] is True
    assert manifest["market_cap_data_ready"] is True
    assert manifest["canonical_pointer_published"] is True
    assert manifest["grouping_policy_selected"] == SELECTED_GROUPING_POLICY
    assert manifest["peer_policy_output_contract"]["security_day_key"] == [
        "date_idx",
        "equity_slot",
    ]
    assert manifest["peer_policy_output_contract"]["security_day_count"] == 90
    peer_policy = pl.read_parquet(output / "peer_policy_security_days.parquet")
    assert peer_policy.height == 90
    assert peer_policy["selected_peer_relation"].unique().to_list() == ["SUBSECTOR"]
    assert peer_policy["selected_other_active_peer_count"].unique().to_list() == [29]
    assert peer_policy["sector_fallback_used"].unique().to_list() == [False]
    assert peer_policy["same_sector_peer_count"].unique().to_list() == [29]
    assert peer_policy["same_subsector_peer_count"].unique().to_list() == [29]
    assert "economic_segment" not in " ".join(peer_policy.columns)
    assert "unit" not in " ".join(peer_policy.columns)
    group_index = pl.read_parquet(output / "peer_group_index.parquet")
    assert group_index["peer_relation"].to_list() == ["SECTOR", "SUBSECTOR"]
    assert group_index["peer_group_id"].to_list() == [0, 1]
    assert manifest["total_normalized_market_cap_row_count"] == 56
    assert manifest["accepted_universe_normalized_market_cap_row_count"] == 56
    assert manifest["mapped_accepted_universe_issuer_count"] == 28
    assert (
        manifest["usable_market_cap_definition"]
        == audit["market_cap"]["usable_market_cap_definition"]
    )
    assert (
        manifest["implementation_sha256"]
        == hashlib.sha256(
            Path(build_human_priors.__code__.co_filename).read_bytes()
        ).hexdigest()
    )
    assert all(
        entry["artifact_sha256"] for entry in manifest["canonical_inputs"].values()
    )
    assert {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in manifest["output_sha256"]
    } == manifest["output_sha256"]
    assert set(manifest["artifact_status"].values()) == {"complete"}

    raw_validation = validate_raw_sources(
        raw_complete,
        as_of=date(2020, 8, 15),
        inputs=inputs,
        assignments_loader=assignments_loader,
    )
    assert raw_validation["build_mode_if_built_now"] == "complete"
    assert raw_validation["eligible_for_strict_build"] is True
    assert raw_validation["classification"]["issuer_count"] == 28
    assert raw_validation["market_cap"]["raw_distinct_reference_month_count"] == 2
    assert raw_validation["market_cap"]["usable_distinct_reference_month_count"] == 2
    assert raw_validation["market_cap"]["market_cap_data_ready"] is True
    assert raw_validation["market_cap"]["total_normalized_market_cap_row_count"] == 56
    assert (
        raw_validation["market_cap"][
            "accepted_universe_normalized_market_cap_row_count"
        ]
        == 56
    )
    assert raw_validation["market_cap"]["source_issuer_count"] == 28
    assert raw_validation["market_cap"]["mapped_accepted_universe_issuer_count"] == 28
    assert raw_validation["units"] == {"unit_count": 1, "component_count": 2}
    assert all(
        source["source_url"] and source["sha256"]
        for source in raw_validation["sources"]
    )

    prior_pointer = pointer.read_text(encoding="utf-8")

    def forced_failure() -> tuple[str, list[str]]:
        raise RuntimeError("forced publication failure")

    with pytest.raises(RuntimeError, match="forced publication failure"):
        build_human_priors(
            raw_dir=raw_complete,
            output_base=output_base,
            pointer=pointer,
            created_at=datetime(2020, 8, 15, 14, 0, tzinfo=UTC),
            inputs=inputs,
            assignments_loader=assignments_loader,
            repository_state=forced_failure,
        )
    assert pointer.read_text(encoding="utf-8") == prior_pointer
    assert not list(output_base.glob("*.partial"))

    raw_incomplete = tmp_path / "raw_incomplete"
    _populate_raw_fixture(
        raw_incomplete,
        issuer_names=issuer_names,
        reference_dates=(date(2020, 7, 31),),
    )
    with pytest.raises(ValueError, match="missing raw calendar months: 2020-06"):
        build_human_priors(
            raw_dir=raw_incomplete,
            output_base=output_base,
            pointer=pointer,
            created_at=datetime(2020, 8, 15, 15, 0, tzinfo=UTC),
            inputs=inputs,
            assignments_loader=assignments_loader,
            repository_state=lambda: ("fixture-commit", []),
        )
    assert pointer.read_text(encoding="utf-8") == prior_pointer
    assert not list(output_base.glob("*.partial"))

    diagnostic = build_human_priors(
        raw_dir=raw_incomplete,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 16, 0, tzinfo=UTC),
        allow_incomplete_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    diagnostic_audit = json.loads(
        (diagnostic / "metadata_audit.json").read_text(encoding="utf-8")
    )
    diagnostic_manifest = json.loads(
        (diagnostic / "manifest.json").read_text(encoding="utf-8")
    )
    assert diagnostic_audit["build_mode"] == "diagnostic_market_cap_not_ready"
    assert diagnostic_audit["market_cap_mode"] == "diagnostic"
    assert diagnostic_audit["market_cap_requested"] is True
    assert diagnostic_audit["raw_market_cap_history_complete"] is False
    assert diagnostic_audit["usable_market_cap_history_complete"] is False
    assert diagnostic_audit["market_cap_data_ready"] is False
    assert diagnostic_audit["eligible_for_downstream_market_cap_features"] is False
    assert diagnostic_audit["market_cap_outputs_emitted"] is True
    assert diagnostic_manifest["market_cap_requested"] is True
    assert diagnostic_manifest["canonical_pointer_published"] is False
    assert set(diagnostic_manifest["artifact_status"].values()) == {
        "diagnostic_market_cap_not_ready"
    }
    assert "DIAGNOSTIC ONLY" in (diagnostic / "metadata_audit.md").read_text(
        encoding="utf-8"
    )
    assert pointer.read_text(encoding="utf-8") == prior_pointer
    assert not list(output_base.glob("*.partial"))

    omitted = build_human_priors(
        raw_dir=raw_incomplete,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 17, 0, tzinfo=UTC),
        omit_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    omitted_files = {path.name for path in omitted.iterdir()}
    assert omitted_files == required - {
        "issuer_market_cap_monthly.parquet",
        "market_cap_coverage.csv",
    }
    assert pointer.read_text(encoding="utf-8") == str(omitted.resolve())
    omitted_audit = json.loads(
        (omitted / "metadata_audit.json").read_text(encoding="utf-8")
    )
    omitted_manifest = json.loads(
        (omitted / "manifest.json").read_text(encoding="utf-8")
    )
    assert omitted_audit["build_mode"] == "complete_market_cap_omitted"
    assert omitted_audit["market_cap_mode"] == "omitted"
    assert omitted_audit["market_cap_requested"] is False
    assert omitted_audit["market_cap_source_data_ready"] is False
    assert omitted_audit["market_cap_data_ready"] is False
    assert omitted_audit["eligible_for_downstream_market_cap_features"] is False
    assert omitted_audit["market_cap_outputs_emitted"] is False
    assert omitted_audit["market_cap_omission_reason"] == MARKET_CAP_OMISSION_REASON
    assert omitted_audit["grouping_policy_selected"] == SELECTED_GROUPING_POLICY
    assert omitted_audit["market_cap"]["market_cap_data_ready"] is False
    assert omitted_manifest["market_cap_mode"] == "omitted"
    assert omitted_manifest["market_cap_requested"] is False
    assert omitted_manifest["market_cap_data_ready"] is False
    assert omitted_manifest["eligible_for_downstream_market_cap_features"] is False
    assert omitted_manifest["market_cap_outputs_emitted"] is False
    assert omitted_manifest["canonical_pointer_published"] is True
    assert omitted_manifest["normalized_schemas"]["issuer_market_cap_monthly"] is None
    assert set(omitted_manifest["artifact_status"].values()) == {
        "complete_market_cap_omitted"
    }
    assert "issuer_market_cap_monthly.parquet" not in omitted_manifest["output_sha256"]
    assert "market_cap_coverage.csv" not in omitted_manifest["output_sha256"]
    assert "COMPLETE — MARKET CAP OMITTED" in (omitted / "metadata_audit.md").read_text(
        encoding="utf-8"
    )
    omitted_metadata = pl.read_parquet(omitted / "security_metadata.parquet")
    assert omitted_metadata["equity_slot"].to_list() == list(range(30))
    assert omitted_metadata["issuer_id"].null_count() == 0
    assert omitted_metadata["sector_group_key"].null_count() == 0
    assert omitted_metadata["subsector_group_key"].null_count() == 0
    assert omitted_metadata["sector_peer_group_id"].null_count() == 0
    assert omitted_metadata["subsector_peer_group_id"].null_count() == 0
    assert pl.read_parquet(omitted / "peer_policy_security_days.parquet").height == 90

    deterministic_pointer = tmp_path / "deterministic_human_priors_path.txt"
    deterministic = build_human_priors(
        raw_dir=raw_incomplete,
        output_base=tmp_path / "deterministic_outputs",
        pointer=deterministic_pointer,
        created_at=datetime(2020, 8, 15, 17, 0, tzinfo=UTC),
        omit_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    deterministic_manifest = json.loads(
        (deterministic / "manifest.json").read_text(encoding="utf-8")
    )
    assert deterministic_manifest["output_sha256"] == omitted_manifest["output_sha256"]
    assert deterministic_pointer.read_text(encoding="utf-8") == str(
        deterministic.resolve()
    )

    omitted_complete_source = build_human_priors(
        raw_dir=raw_complete,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 18, 0, tzinfo=UTC),
        omit_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    omitted_complete_audit = json.loads(
        (omitted_complete_source / "metadata_audit.json").read_text(encoding="utf-8")
    )
    assert omitted_complete_audit["market_cap_source_data_ready"] is True
    assert omitted_complete_audit["market_cap_data_ready"] is False
    assert (
        omitted_complete_audit["eligible_for_downstream_market_cap_features"] is False
    )
    assert not (omitted_complete_source / "issuer_market_cap_monthly.parquet").exists()
    assert not (omitted_complete_source / "market_cap_coverage.csv").exists()
    assert pointer.read_text(encoding="utf-8") == str(omitted_complete_source.resolve())
    assert not list(output_base.glob("*.partial"))


def test_complete_build_removes_new_artifact_when_pointer_publication_fails(
    tmp_path: Path,
) -> None:
    inputs, issuer_names, _, assignments_loader = _canonical_build_fixture(
        tmp_path / "canonical"
    )
    raw_dir = tmp_path / "raw"
    _populate_raw_fixture(
        raw_dir,
        issuer_names=issuer_names,
        reference_dates=(date(2020, 7, 31), date(2020, 6, 30)),
    )
    output_base = tmp_path / "outputs"
    existing_output = output_base / "preexisting_complete_artifact"
    existing_output.mkdir(parents=True)
    sentinel = existing_output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    pointer = tmp_path / "human_priors_path.txt"
    old_pointer = str(existing_output.resolve())
    pointer.write_text(old_pointer, encoding="utf-8")
    expected_output = output_base / "human_priors_complete_20200815T130000000000Z"

    def fail_pointer_publication(actual_pointer: Path, target: Path) -> None:
        assert actual_pointer == pointer
        assert target == expected_output
        raise RuntimeError("forced pointer publication failure")

    with pytest.raises(RuntimeError, match="forced pointer publication failure"):
        build_human_priors(
            raw_dir=raw_dir,
            output_base=output_base,
            pointer=pointer,
            created_at=datetime(2020, 8, 15, 13, 0, tzinfo=UTC),
            inputs=inputs,
            assignments_loader=assignments_loader,
            repository_state=lambda: ("fixture-commit", []),
            pointer_publisher=fail_pointer_publication,
        )

    assert pointer.read_text(encoding="utf-8") == old_pointer
    assert not expected_output.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert {path.name for path in output_base.iterdir()} == {existing_output.name}
    assert not list(output_base.glob("*.partial"))


def test_build_rejects_raw_complete_but_empty_normalized_market_cap(
    tmp_path: Path,
) -> None:
    inputs, issuer_names, _, assignments_loader = _canonical_build_fixture(
        tmp_path / "canonical"
    )
    reference_dates = (date(2020, 7, 31), date(2020, 6, 30))
    unmatched_issuers = tuple(f"UNMATCHED {index:02d}" for index in range(1, 29))
    raw_dir = tmp_path / "raw"
    _populate_raw_fixture(
        raw_dir,
        issuer_names=issuer_names,
        reference_dates=reference_dates,
        market_cap_payload=_market_cap_csv(reference_dates, unmatched_issuers),
    )
    output_base = tmp_path / "outputs"
    pointer = tmp_path / "human_priors_path.txt"
    pointer.write_text("prior", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="accepted-universe normalized market-cap data is empty",
    ):
        build_human_priors(
            raw_dir=raw_dir,
            output_base=output_base,
            pointer=pointer,
            created_at=datetime(2020, 8, 15, 13, 0, tzinfo=UTC),
            inputs=inputs,
            assignments_loader=assignments_loader,
            repository_state=lambda: ("fixture-commit", []),
        )
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))

    diagnostic = build_human_priors(
        raw_dir=raw_dir,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 14, 0, tzinfo=UTC),
        allow_incomplete_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    audit = json.loads((diagnostic / "metadata_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((diagnostic / "manifest.json").read_text(encoding="utf-8"))
    assert audit["raw_market_cap_history_complete"] is True
    assert audit["usable_market_cap_history_complete"] is False
    assert audit["market_cap_data_ready"] is False
    assert audit["eligible_for_downstream_market_cap_features"] is False
    assert audit["market_cap"]["total_normalized_market_cap_row_count"] == 0
    assert audit["market_cap"]["accepted_universe_normalized_market_cap_row_count"] == 0
    assert audit["market_cap"]["mapped_accepted_universe_issuer_count"] == 0
    assert audit["market_cap"]["usable_missing_reference_months"] == [
        "2020-06",
        "2020-07",
    ]
    assert pl.read_parquet(diagnostic / "issuer_market_cap_monthly.parquet").is_empty()
    assert manifest["canonical_pointer_published"] is False
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))


def test_build_rejects_market_cap_only_for_issuers_outside_accepted_universe(
    tmp_path: Path,
) -> None:
    inputs, issuer_names, _, assignments_loader = _canonical_build_fixture(
        tmp_path / "canonical"
    )
    reference_dates = (date(2020, 7, 31), date(2020, 6, 30))
    outside_issuers = tuple(f"OUTSIDE {index:02d}" for index in range(1, 29))
    raw_dir = tmp_path / "raw"
    _populate_raw_fixture(
        raw_dir,
        issuer_names=issuer_names,
        reference_dates=reference_dates,
        market_cap_payload=_market_cap_csv(reference_dates, outside_issuers),
        classification_issuer_names=(*issuer_names, *outside_issuers),
    )
    raw_validation = validate_raw_sources(
        raw_dir,
        as_of=date(2020, 8, 15),
        inputs=inputs,
        assignments_loader=assignments_loader,
    )
    assert raw_validation["market_cap"]["raw_market_cap_history_complete"] is True
    assert raw_validation["market_cap"]["total_normalized_market_cap_row_count"] == 56
    assert (
        raw_validation["market_cap"][
            "accepted_universe_normalized_market_cap_row_count"
        ]
        == 0
    )
    assert raw_validation["market_cap"]["mapped_accepted_universe_issuer_count"] == 0
    assert raw_validation["market_cap"]["usable_market_cap_history_complete"] is False
    assert raw_validation["market_cap"]["market_cap_data_ready"] is False

    output_base = tmp_path / "outputs"
    pointer = tmp_path / "human_priors_path.txt"
    pointer.write_text("prior", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="no normalized market-cap issuer maps to the accepted model universe",
    ):
        build_human_priors(
            raw_dir=raw_dir,
            output_base=output_base,
            pointer=pointer,
            created_at=datetime(2020, 8, 15, 13, 0, tzinfo=UTC),
            inputs=inputs,
            assignments_loader=assignments_loader,
            repository_state=lambda: ("fixture-commit", []),
        )
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))

    diagnostic = build_human_priors(
        raw_dir=raw_dir,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 14, 0, tzinfo=UTC),
        allow_incomplete_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    audit = json.loads((diagnostic / "metadata_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((diagnostic / "manifest.json").read_text(encoding="utf-8"))
    assert audit["raw_market_cap_history_complete"] is True
    assert audit["usable_market_cap_history_complete"] is False
    assert audit["market_cap_data_ready"] is False
    assert audit["eligible_for_downstream_market_cap_features"] is False
    assert audit["market_cap"]["total_normalized_market_cap_row_count"] == 56
    assert audit["market_cap"]["accepted_universe_normalized_market_cap_row_count"] == 0
    assert audit["market_cap"]["mapped_accepted_universe_issuer_count"] == 0
    assert (
        pl.read_parquet(diagnostic / "issuer_market_cap_monthly.parquet").height == 56
    )
    assert manifest["canonical_pointer_published"] is False
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))


def test_build_rejects_raw_month_missing_after_normalization(tmp_path: Path) -> None:
    inputs, issuer_names, _, assignments_loader = _canonical_build_fixture(
        tmp_path / "canonical"
    )
    reference_dates = (date(2020, 7, 31), date(2020, 6, 30))
    unmatched_issuers = tuple(f"UNMATCHED {index:02d}" for index in range(1, 22))
    all_market_issuers = (*issuer_names, *unmatched_issuers)
    missing_values = {(issuer_name, date(2020, 6, 30)) for issuer_name in issuer_names}
    raw_dir = tmp_path / "raw"
    _populate_raw_fixture(
        raw_dir,
        issuer_names=issuer_names,
        reference_dates=reference_dates,
        market_cap_payload=_market_cap_csv(
            reference_dates,
            all_market_issuers,
            missing_values,
        ),
    )
    output_base = tmp_path / "outputs"
    pointer = tmp_path / "human_priors_path.txt"
    pointer.write_text("prior", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="missing usable normalized calendar months: 2020-06",
    ):
        build_human_priors(
            raw_dir=raw_dir,
            output_base=output_base,
            pointer=pointer,
            created_at=datetime(2020, 8, 15, 13, 0, tzinfo=UTC),
            inputs=inputs,
            assignments_loader=assignments_loader,
            repository_state=lambda: ("fixture-commit", []),
        )
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))

    diagnostic = build_human_priors(
        raw_dir=raw_dir,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 14, 0, tzinfo=UTC),
        allow_incomplete_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    audit = json.loads((diagnostic / "metadata_audit.json").read_text(encoding="utf-8"))
    assert audit["raw_market_cap_history_complete"] is True
    assert audit["usable_market_cap_history_complete"] is False
    assert audit["market_cap_data_ready"] is False
    assert audit["market_cap"]["raw_missing_reference_months"] == []
    assert audit["market_cap"]["usable_missing_reference_months"] == ["2020-06"]
    assert audit["market_cap"][
        "accepted_universe_normalized_market_cap_row_count"
    ] == len(issuer_names)
    assert audit["market_cap"]["total_normalized_market_cap_row_count"] == len(
        issuer_names
    )
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))


def test_build_rejects_conflicts_with_complete_raw_and_usable_months(
    tmp_path: Path,
) -> None:
    inputs, issuer_names, _, assignments_loader = _canonical_build_fixture(
        tmp_path / "canonical"
    )
    reference_dates = (date(2020, 7, 31), date(2020, 6, 30))
    raw_dir = tmp_path / "raw"
    _populate_raw_fixture(
        raw_dir,
        issuer_names=issuer_names,
        reference_dates=reference_dates,
    )
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    conflicting_payload = _market_cap_csv(reference_dates, issuer_names).replace(
        b"BANCO X|1010.00", b"BANCO X|9999.00"
    )
    (downloads / "Bolsa_Valores_Mensal_2020-07.csv").write_bytes(conflicting_payload)
    ingest_market_cap_directory(downloads, raw_dir)
    output_base = tmp_path / "outputs"
    pointer = tmp_path / "human_priors_path.txt"
    pointer.write_text("prior", encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting issuer-month groups: 1"):
        build_human_priors(
            raw_dir=raw_dir,
            output_base=output_base,
            pointer=pointer,
            created_at=datetime(2020, 8, 15, 13, 0, tzinfo=UTC),
            inputs=inputs,
            assignments_loader=assignments_loader,
            repository_state=lambda: ("fixture-commit", []),
        )
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))

    diagnostic = build_human_priors(
        raw_dir=raw_dir,
        output_base=output_base,
        pointer=pointer,
        created_at=datetime(2020, 8, 15, 14, 0, tzinfo=UTC),
        allow_incomplete_market_cap=True,
        inputs=inputs,
        assignments_loader=assignments_loader,
        repository_state=lambda: ("fixture-commit", []),
    )
    audit = json.loads((diagnostic / "metadata_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((diagnostic / "manifest.json").read_text(encoding="utf-8"))
    assert audit["raw_market_cap_history_complete"] is True
    assert audit["usable_market_cap_history_complete"] is True
    assert audit["market_cap_data_ready"] is False
    assert audit["eligible_for_downstream_market_cap_features"] is False
    assert audit["market_cap"]["raw_missing_reference_months"] == []
    assert audit["market_cap"]["usable_missing_reference_months"] == []
    assert audit["market_cap"]["conflicting_issuer_month_groups"] == 1
    assert manifest["conflicting_issuer_month_groups"] == 1
    assert manifest["canonical_pointer_published"] is False
    assert pointer.read_text(encoding="utf-8") == "prior"
    assert not list(output_base.glob("*.partial"))
