from __future__ import annotations

import base64
import io
import zipfile
from datetime import UTC, date, datetime
from html import escape

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing.human_priors import (
    CLASSIFICATION_API_BASE,
    MARKET_CAP_API_BASE,
    UNITS_PAGE_URL,
    HttpPayload,
    acquire_official_sources,
    active_security_days,
    add_self_excluded_peer_counts,
    issuer_peer_audit,
    market_cap_audit,
    normalize_unit_components,
    parse_classification_xlsx,
    parse_market_cap_csv,
    parse_units_html,
    reconcile_security_metadata,
    strictly_lagged_market_cap_index,
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


def _market_cap_csv() -> bytes:
    return (
        "20260804|||||\n"
        "IBOV|IBRX100|Empresa|Valor (R$) em 31/07/2026|"
        "Valor (R$) em 30/06/2026|Var (%)\n"
        "X|X|BANCO X|1.234,50|1200,00|2,88\n"
        "X|X|ENERGIA Y|987,65|900,00|9,74\n"
    ).encode()


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

    assert parsed.height == 4
    banco = parsed.filter(pl.col("issuer_name") == "BANCO X")
    assert banco["reference_date"].to_list() == [
        date(2026, 6, 30),
        date(2026, 7, 31),
    ]
    assert banco["market_cap_brl"].to_list() == [1200.0, 1234.5]


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
    assert exceptions.filter(pl.col("source_key") == "SEC_FUZZY").height > 0


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


def test_strict_market_cap_lag_excludes_same_day() -> None:
    references = [date(2026, 5, 31), date(2026, 6, 30), date(2026, 7, 31)]

    assert strictly_lagged_market_cap_index(references, date(2026, 5, 31)) is None
    assert strictly_lagged_market_cap_index(references, date(2026, 6, 30)) == 0
    assert strictly_lagged_market_cap_index(references, date(2026, 7, 1)) == 1


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
        "normalized_row_count": 2,
        "normalized_issuer_count": 1,
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
