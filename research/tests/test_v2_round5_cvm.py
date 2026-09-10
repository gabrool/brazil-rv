from copy import deepcopy
from datetime import date, datetime
import io
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2 import round5_cvm
from brazil_rv.v2.round5_cvm import (
    assign_account,
    available_session,
    build_identity,
    event_features,
    fiscal_quarters,
    fetch,
    fundamental_features,
    fundamental_state,
    table_rows,
    trailing_twelve_months,
)


def test_minute_receipt_enters_first_available_decision():
    sessions = [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)]
    assert available_session(datetime(2024, 1, 5, 15, 44), sessions) == 0
    assert available_session(datetime(2024, 1, 5, 15, 45), sessions) == 1
    assert available_session(date(2024, 1, 5), sessions) == 1
    assert available_session(datetime(2024, 1, 6, 12), sessions) == 1


def test_float_snapshot_date_is_separate_from_filing_year_and_receipt(monkeypatch):
    sessions = [date(2024, 5, d) for d in (6, 7)]
    document = {
        "id": "10",
        "cnpj": "12345678000100",
        "cvm_code": "001234",
        "reference": date(2024, 1, 1),
        "version": 2,
        "receipt": sessions[0],
    }
    monkeypatch.setattr(round5_cvm, "filing_headers", lambda *args: [document])
    monkeypatch.setattr(
        round5_cvm, "annual_paths", lambda *args: [Path("fre_2024.zip")]
    )
    monkeypatch.setattr(
        round5_cvm,
        "read_csv_member",
        lambda *args: [
            {
                "ID_Documento": "10",
                "Quantidade_Acoes_Ordinarias_Circulacao": "100",
                "Data_Ultima_Assembleia": "2024-04-30",
            }
        ],
    )
    result = round5_cvm.public_float_observations(
        Path("unused"),
        [{"id": "10", "group": "cadastre", "receipt": datetime(2024, 5, 6, 15, 44)}],
        sessions,
    ).row(0, named=True)
    assert result["date"] == sessions[0]
    assert result["snapshot_date"] == date(2024, 4, 30)
    assert result["reference"] == date(2024, 1, 1)
    assert result["cvm_code"] == "001234"


def test_event_source_mutation_changes_first_available_decision_only():
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5, 8)]
    earlier = {
        "cvm_code": "1",
        "kind": "ITR",
        "group": "structured",
        "reference": date(2023, 9, 30),
        "receipt": datetime(2024, 1, 2, 13),
        "subject": "",
    }
    future = {
        "cvm_code": "1",
        "kind": "Fato Relevante",
        "group": "material_fact",
        "reference": date(2024, 1, 1),
        "receipt": datetime(2024, 1, 4, 15, 44),
        "subject": "",
    }
    base = event_features([earlier], sessions)
    changed = event_features([earlier, future], sessions)
    assert base.head(2).equals(changed.head(2))
    assert base["material_fact_count_20"][2] == 0
    assert changed["material_fact_count_20"][2] == 1
    assert changed["sessions_since_material_fact"][2] == 0


def test_identity_uses_known_cadastre_and_prior_ticker_observation():
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    document = {
        "cnpj": "123",
        "id": "1",
        "cvm_code": "1",
        "reference": date(2023, 1, 1),
        "version": 1,
        "receipt": date(2024, 1, 2),
        "available_index": 1,
        "sector": "Industry",
        "securities": [
            {
                "ticker": "ABC3",
                "class": "ON",
                "preferred_class": "",
                "unit_composition": "",
                "start": date(2000, 1, 1),
                "end": date.max,
            }
        ],
    }
    observations = pl.DataFrame(
        {"trade_date": sessions, "ticker": ["ABC3"] * 4, "isin": ["ISIN1"] * 4}
    )
    result = build_identity([document], observations, sessions, ["ISIN1"])
    assert result["date"].min() == sessions[1]
    changed = deepcopy(document)
    changed["sector"] = "Changed"
    assert (
        build_identity([changed], observations, sessions, ["ISIN1"])
        .filter(pl.col("date") < sessions[1])
        .is_empty()
    )
    assert result["isin"].unique().to_list() == ["ISIN1"]


def report(reference, cumulative, version=1):
    d = {
        "id": str(reference),
        "reference": reference,
        "version": version,
        "accounts": {},
    }
    for code, value in (
        ("1", 100),
        ("2.03", 40),
        ("3.01", cumulative * 10),
        ("3.03", cumulative * 2),
        ("3.11", cumulative),
        ("6.01", cumulative),
    ):
        assign_account(
            d,
            "con",
            code,
            value,
            date(reference.year, 1, 1) if code.startswith(("3", "6")) else None,
            reference,
            "Revenue" if code == "3.01" else "Account",
        )
    return d


def test_q4_requires_received_nine_month_and_annual_versions():
    dates = [
        date(2023, 3, 31),
        date(2023, 6, 30),
        date(2023, 9, 30),
        date(2023, 12, 31),
    ]
    documents = [report(d, v) for d, v in zip(dates, (10, 30, 60, 100))]
    ledger = {d["reference"]: d for d in documents[:3]}
    before = fiscal_quarters(ledger, "con", "net_income")
    assert dates[-1] not in before
    ledger[dates[-1]] = documents[-1]
    assert fiscal_quarters(ledger, "con", "net_income")[dates[-1]] == 40
    assert trailing_twelve_months(ledger, "con", "net_income", dates[-1])[0] == 100
    del ledger[dates[2]]
    assert dates[-1] not in fiscal_quarters(ledger, "con", "net_income")
    assert trailing_twelve_months(ledger, "con", "net_income", dates[-1])[0] == 100


def test_unavailable_latest_version_cannot_borrow_later_contents():
    d = report(date(2023, 12, 31), 100)
    available = fundamental_state({("DFP", d["reference"]): d})
    assert available["liabilities_to_assets"] == pytest.approx(0.6)
    missing = {**d, "version": 2, "accounts": {}}
    assert (
        fundamental_state({("DFP", d["reference"]): missing})["liabilities_to_assets"]
        is None
    )
    assert d["version"] == 1


def test_nonfinancial_ratios_are_not_forced_onto_insurers():
    documents = [
        report(date(2023, m, day), cumulative)
        for m, day, cumulative in ((3, 31, 10), (6, 30, 30), (9, 30, 60), (12, 31, 100))
    ]
    ledger = {d["reference"]: d for d in documents}
    industrial = fundamental_state(ledger, "Industrial")
    insurer = fundamental_state(ledger, "Seguros")
    assert industrial["gross_profitability"] == pytest.approx(2.0)
    assert insurer["gross_profitability"] is None
    assert insurer["liabilities_to_assets"] == industrial["liabilities_to_assets"]


def test_viewer_keeps_dates_blank_values_and_literal_numeric_units():
    payload = b"<table><tr><th>Conta</th><th>Descricao</th><th>01/01/2023 a 31/12/2023</th></tr><tr><td>3.01</td><td>Revenue</td><td>1.234,50</td></tr><tr><td>3.03</td><td>Gross</td><td></td></tr></table>"
    rows = table_rows(payload)
    assert rows[0][2] == "01/01/2023 a 31/12/2023"
    assert rows[1][2] == "1.234,50"
    assert rows[2][2] == ""


def test_public_viewer_url_preserves_document_and_encodes_issuer_name():
    class Opener:
        def open(self, request, timeout):
            assert (
                request.full_url
                == "https://example.test/view?ID=123&Empresa=Companhia%20A%C3%A7%C3%A3o"
            )
            return io.BytesIO(b"source")

    assert (
        fetch(
            "https://example.test/view?ID=123&Empresa=Companhia Ação", opener=Opener()
        )
        == b"source"
    )


def family_fixture():
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    document = report(date(2023, 12, 31), 100)
    document.update(
        id="1",
        cnpj="123",
        cvm_code="000001",
        kind="DFP",
        receipt=date(2024, 1, 2),
        shares={"ON": 100.0, "PN": 0.0},
    )
    rad = [
        {
            "id": "1",
            "group": "structured",
            "cvm_code": "000001",
            "reference": document["reference"],
            "version": "1",
            "receipt": datetime(2024, 1, 2, 15, 44),
        }
    ]
    identity = pl.DataFrame(
        {
            "date": sessions,
            "isin": ["ISIN1"] * 4,
            "cnpj": ["123"] * 4,
            "cvm_code": ["000001"] * 4,
            "sector": ["Industrial"] * 4,
            "class": ["ON"] * 4,
            "identity_known_date": [sessions[0]] * 4,
        }
    )
    market = {
        "columns": {"ISIN1": 0},
        "close": np.array([[10.0], [20.0], [30.0], [40.0]]),
        "observed": np.ones((4, 1), dtype=bool),
        "barrier_prefix": np.zeros((5, 1), dtype=int),
    }
    return sessions, document, rad, identity, market


def test_joined_family_future_filing_changes_first_receipt_and_keeps_real_age():
    sessions, document, rad, identity, market = family_fixture()
    base, _ = fundamental_features(
        [deepcopy(document)], rad, identity, sessions, market
    )
    revised = deepcopy(document)
    revised.update(
        id="2", version=2, receipt=date(2024, 1, 4), shares={"ON": 200.0, "PN": 0.0}
    )
    changed_rad = rad + [
        {**rad[0], "id": "2", "version": "2", "receipt": datetime(2024, 1, 4, 15, 44)}
    ]
    changed, _ = fundamental_features(
        [deepcopy(document), revised], changed_rad, identity, sessions, market
    )
    assert base.head(2).equals(changed.head(2))
    assert base["log_market_cap"][2] == pytest.approx(np.log(2000))
    assert changed["log_market_cap"][2] == pytest.approx(np.log(4000))
    assert base["log_market_cap_age_sessions"][2] == 2
    assert changed["log_market_cap_age_sessions"][2] == 0
    assert changed["log_market_cap_age_sessions"][3] == 1


def test_joined_valuation_excludes_current_close_and_future_unit_boundaries():
    sessions, document, rad, identity, market = family_fixture()
    base, _ = fundamental_features(
        [deepcopy(document)], rad, identity, sessions, market
    )
    changed_market = deepcopy(market)
    changed_market["close"][2, 0] = 300
    changed, _ = fundamental_features(
        [deepcopy(document)], rad, identity, sessions, changed_market
    )
    assert base.head(3).equals(changed.head(3))
    assert changed["log_market_cap"][3] == pytest.approx(np.log(30000))
    changed_market["barrier_prefix"][3:, 0] = 1
    boundary, _ = fundamental_features(
        [deepcopy(document)], rad, identity, sessions, changed_market
    )
    assert base.head(3).equals(boundary.head(3))
    assert boundary["log_market_cap"][3] is None


def test_issuer_market_cap_never_prices_other_classes_at_one_class_close():
    sessions, document, rad, identity, market = family_fixture()
    document["shares"]["PN"] = 50
    result, _ = fundamental_features([document], rad, identity, sessions, market)
    assert result["log_market_cap"].null_count() == len(sessions)
    assert result["liabilities_to_assets"].null_count() == 0


def test_known_fre_capital_change_masks_prior_count_only_after_publication():
    sessions, document, rad, identity, market = family_fixture()
    change = {"cnpj": "123", "effective": date(2024, 1, 2), "available_index": 2}
    result, _ = fundamental_features(
        [document], rad, identity, sessions, market, [change]
    )
    assert result["log_market_cap"][1] is not None
    assert result["log_market_cap"][2] is None


def test_early_exact_legal_name_bridge_excludes_new_same_brand_security():
    sessions, _, _, _, _ = family_fixture()
    document = {
        "cnpj": "15912764000120",
        "cvm_code": "023000",
        "id": "10",
        "reference": date(2023, 1, 1),
        "version": 1,
        "receipt": sessions[0],
        "available_index": 1,
        "legal_name": "SMILES S.A.",
        "sector": "Services",
        "securities": [
            {
                "ticker": "",
                "class": "ON",
                "preferred_class": "",
                "unit_composition": "",
                "start": date(2010, 1, 1),
                "end": date.max,
            }
        ],
    }
    observations = pl.DataFrame(
        {
            "trade_date": [sessions[0], sessions[2]],
            "isin": ["OLD", "NEW"],
            "ticker": ["OLD3", "NEW3"],
            "issuer_short_name": ["SMILES", "SMILES"],
            "security_spec_base": ["ON", "ON"],
        }
    )
    identity = build_identity([document], observations, sessions, ["OLD", "NEW"])
    assert identity["isin"].unique().to_list() == ["OLD"]
    assert identity["date"].min() == sessions[1]
    assert identity["identity_method"].unique().to_list() == [
        "exact_historical_legal_spelling"
    ]
    mutated = deepcopy(document)
    mutated.update(
        id="11",
        receipt=sessions[2],
        available_index=3,
        version=2,
        sector="New classification",
    )
    changed = build_identity(
        [document, mutated], observations, sessions, ["OLD", "NEW"]
    )
    assert identity.filter(pl.col("date") < sessions[3]).equals(
        changed.filter(pl.col("date") < sessions[3])
    )


def test_contemporaneous_exact_name_collision_is_masked():
    sessions, _, _, _, _ = family_fixture()
    document = {
        "cnpj": "123",
        "cvm_code": "1",
        "id": "10",
        "reference": date(2023, 1, 1),
        "version": 1,
        "receipt": sessions[0],
        "available_index": 1,
        "legal_name": "WEG SA",
        "sector": "Industry",
        "securities": [
            {
                "ticker": "",
                "class": "ON",
                "preferred_class": "",
                "unit_composition": "",
                "start": date(2010, 1, 1),
                "end": date.max,
            }
        ],
    }
    other = {**document, "cnpj": "456", "cvm_code": "2", "id": "11"}
    observations = pl.DataFrame(
        {
            "trade_date": [sessions[0]],
            "isin": ["WEG"],
            "ticker": ["WEGE3"],
            "issuer_short_name": ["WEG"],
            "security_spec_base": ["ON"],
        }
    )
    assert build_identity([document, other], observations, sessions, ["WEG"]).is_empty()


def test_same_version_later_original_receipt_replaces_previous_document():
    sessions, document, rad, identity, market = family_fixture()
    replacement = deepcopy(document)
    replacement.update(id="2", receipt=sessions[2], shares={"ON": 200.0, "PN": 0.0})
    replacement_rad = {**rad[0], "id": "2", "receipt": datetime(2024, 1, 4, 15, 44)}
    result, _ = fundamental_features(
        [document, replacement], [*rad, replacement_rad], identity, sessions, market
    )
    assert result["log_market_cap"][1] == pytest.approx(np.log(1000))
    assert result["log_market_cap"][2] == pytest.approx(np.log(4000))


def test_cnpj_branch_alias_requires_identical_legal_root_and_cvm_code():
    sessions, document, rad, identity, market = family_fixture()
    document["cnpj"] = "60398369000479"
    identity = identity.with_columns(pl.lit("60398369000126").alias("cnpj"))
    result, _ = fundamental_features([document], rad, identity, sessions, market)
    assert result["liabilities_to_assets"].null_count() == 0
    wrong = identity.with_columns(pl.lit("000002").alias("cvm_code"))
    masked, _ = fundamental_features([document], rad, wrong, sessions, market)
    assert masked["liabilities_to_assets"].null_count() == len(sessions)


def test_direct_annual_ttm_survives_unavailable_old_itr_contents():
    annual = report(date(2023, 12, 31), 100)
    missing_itr = {
        "reference": date(2023, 9, 30),
        "version": 1,
        "id": "0",
        "accounts": {},
    }
    ledger = {"annual": annual, "missing_itr": missing_itr}
    assert (
        trailing_twelve_months(ledger, "con", "net_income", annual["reference"])[0]
        == 100
    )
    assert annual["reference"] not in fiscal_quarters(ledger, "con", "net_income")
    assert fundamental_state(ledger, "Industry")["gross_profitability"] == 2.0


def test_interim_ttm_bridge_rejects_mixed_basis_and_fiscal_periods():
    previous = report(date(2022, 9, 30), 60)
    annual = report(date(2022, 12, 31), 100)
    current = report(date(2023, 9, 30), 75)
    ledger = {"prior": previous, "annual": annual, "current": current}
    assert (
        trailing_twelve_months(ledger, "con", "net_income", current["reference"])[0]
        == 115
    )
    mixed = deepcopy(ledger)
    mixed["annual"]["accounts"]["ind"] = mixed["annual"]["accounts"].pop("con")
    assert (
        trailing_twelve_months(mixed, "con", "net_income", current["reference"])[0]
        is None
    )
    mismatched = deepcopy(ledger)
    mismatched["prior"]["accounts"]["con"]["net_income"]["start"] = date(2022, 4, 1)
    assert (
        trailing_twelve_months(mismatched, "con", "net_income", current["reference"])[0]
        is None
    )


def test_joined_ttm_restatement_changes_on_its_own_first_receipt():
    sessions, _, _, identity, market = family_fixture()
    documents = [
        report(date(2022, 9, 30), 60),
        report(date(2022, 12, 31), 100),
        report(date(2023, 9, 30), 75),
    ]
    rad = []
    for i, document in enumerate(documents):
        document.update(
            id=str(i + 1),
            cnpj="123",
            cvm_code="000001",
            kind="DFP" if i == 1 else "ITR",
            receipt=sessions[0],
        )
        rad.append(
            {
                "id": document["id"],
                "group": "structured",
                "cvm_code": "000001",
                "reference": document["reference"],
                "version": "1",
                "receipt": datetime(2024, 1, 2, 15, 44),
            }
        )
    base, _ = fundamental_features(deepcopy(documents), rad, identity, sessions, market)
    revised = report(date(2022, 9, 30), 50, version=2)
    revised.update(
        id="4", cnpj="123", cvm_code="000001", kind="ITR", receipt=sessions[2]
    )
    revised_rad = {
        **rad[0],
        "id": "4",
        "version": "2",
        "receipt": datetime(2024, 1, 4, 15, 44),
    }
    result, _ = fundamental_features(
        [*deepcopy(documents), revised], [*rad, revised_rad], identity, sessions, market
    )
    assert base.head(2).equals(result.head(2))
    assert base["gross_profitability"][2] == pytest.approx(2.3)
    assert result["gross_profitability"][2] == pytest.approx(2.5)
    unavailable = deepcopy(revised)
    unavailable["accounts"] = {}
    masked, _ = fundamental_features(
        [*deepcopy(documents), unavailable],
        [*rad, revised_rad],
        identity,
        sessions,
        market,
    )
    assert masked.head(2).equals(base.head(2))
    assert masked["gross_profitability"][2] is None


def test_old_fca_ticker_cannot_relabel_a_new_isin_after_ticker_reuse():
    sessions, _, _, _, _ = family_fixture()
    old = {
        "cnpj": "123",
        "cvm_code": "1",
        "id": "10",
        "reference": date(2023, 1, 1),
        "version": 1,
        "receipt": sessions[0],
        "available_index": 1,
        "sector": "Industry",
        "securities": [
            {
                "ticker": "ABC3",
                "class": "ON",
                "preferred_class": "",
                "unit_composition": "",
                "start": date(2000, 1, 1),
                "end": date.max,
            }
        ],
    }
    observations = pl.DataFrame(
        {
            "trade_date": [sessions[0], sessions[1]],
            "isin": ["OLD", "NEW"],
            "ticker": ["ABC3", "ABC3"],
        }
    )
    result = build_identity([old], observations, sessions, ["OLD", "NEW"])
    assert result["isin"].to_list() == ["OLD"]
    assert result["date"].to_list() == [sessions[1]]


def test_preannounced_exact_listing_date_admits_new_isin_without_age_rule():
    sessions, _, _, _, _ = family_fixture()
    document = {
        "cnpj": "123",
        "cvm_code": "1",
        "id": "10",
        "reference": date(2023, 1, 1),
        "version": 1,
        "receipt": sessions[0],
        "available_index": 1,
        "sector": "Industry",
        "securities": [
            {
                "ticker": "ABC3",
                "class": "ON",
                "preferred_class": "",
                "unit_composition": "",
                "start": sessions[1],
                "end": date.max,
            }
        ],
    }
    observations = pl.DataFrame(
        {"trade_date": [sessions[1]], "isin": ["NEW"], "ticker": ["ABC3"]}
    )
    result = build_identity([document], observations, sessions, ["NEW"])
    assert result["date"].to_list() == sessions[2:]
    assert result["identity_method"].unique().to_list() == [
        "dated_fca_preannounced_listing"
    ]
