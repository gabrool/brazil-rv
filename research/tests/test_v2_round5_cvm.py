from copy import deepcopy
from datetime import date, datetime
import io
import hashlib
import json
import zipfile
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


@pytest.mark.parametrize("reverse", (False, True))
def test_csv_and_viewer_ignore_stale_nci_label_under_liabilities(
    tmp_path, monkeypatch, reverse
):
    document = {
        "id": "121447",
        "cnpj": "00416968000101",
        "cvm_code": "024406",
        "reference": date(2022, 9, 30),
        "version": 1,
        "kind": "ITR",
    }
    entries = [
        ("2.03", "Passivos Financeiros ao Custo Amortizado", "0"),
        ("2.03.09", "Participacao dos Acionistas Nao Controladores", "0"),
        ("2.08", "Patrimonio Liquido Consolidado", "8337366"),
        ("2.08.09", "Participacao dos Acionistas Nao Controladores", "95984"),
    ]
    if reverse:
        entries.reverse()
    rows = [
        {
            "CNPJ_CIA": document["cnpj"],
            "DT_REFER": "2022-09-30",
            "VERSAO": "1",
            "MOEDA": "REAL",
            "ESCALA_MOEDA": "MIL",
            "ORDEM_EXERC": "ÚLTIMO",
            "DT_FIM_EXERC": "2022-09-30",
            "CD_CONTA": code,
            "DS_CONTA": label,
            "VL_CONTA": value,
        }
        for code, label, value in entries
    ]
    source = tmp_path / "itr_cia_aberta_2022.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "itr_cia_aberta_BPP_con_2022.csv",
            pl.DataFrame(rows).write_csv(separator=";").encode("latin1"),
        )
    monkeypatch.setattr(
        round5_cvm,
        "filing_headers",
        lambda root, kind: [deepcopy(document)] if kind == "itr" else [],
    )
    monkeypatch.setattr(
        round5_cvm, "annual_paths", lambda root, kind: [source] if kind == "itr" else []
    )
    csv_book = round5_cvm.load_accounts(tmp_path)[0]["accounts"]["con"]
    assert csv_book["minority_equity"]["value"] == 95984000
    assert csv_book["minority_equity"]["source_code"] == "2.08.09"
    payload = (
        "<h2>Reais Mil</h2><table><tr><td>Conta</td><td>Descricao</td><td>30/09/2022</td></tr>"
        + "".join(
            f"<tr><td>{code}</td><td>{label}</td><td>{value}</td></tr>"
            for code, label, value in entries
        )
        + "</table>"
    ).encode()
    (tmp_path / "statement.html").write_bytes(payload)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "document": document,
                "pages": [
                    {
                        "file": "statement.html",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "basis": "con",
                    }
                ],
            },
            default=str,
        )
    )
    round5_cvm.attach_viewer_accounts(document, tmp_path)
    assert document["accounts"]["con"] == csv_book


def test_income_attribution_requires_compatible_observed_parent():
    document = {"id": "1"}
    for code, parent in (
        ("3.11.01", "Reversao dos Juros sobre Capital Proprio"),
        ("3.13.01", "Lucro/Prejuizo do Periodo"),
    ):
        assign_account(
            document,
            "con",
            code,
            17,
            date(2024, 1, 1),
            date(2024, 3, 31),
            "Atribuido a Empresa Controladora",
            parent,
        )
    assert document["accounts"]["con"]["parent_income"]["source_code"] == "3.13.01"
    # A missing parent is not an observed incompatible parent or an invented zero.
    partial = {"id": "2"}
    assign_account(
        partial,
        "con",
        "3.13.01",
        17,
        date(2024, 1, 1),
        date(2024, 3, 31),
        "Atribuido a Empresa Controladora",
    )
    assert partial["accounts"]["con"]["parent_income"]["value"] == 17


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


def test_viewer_source_hash_failure_cannot_attach_partial_accounts(tmp_path):
    payload = b"<h2>Reais Mil</h2><table><tr><td>Conta</td><td>Descricao</td><td>01/01/2024 a 31/03/2024</td></tr><tr><td>3.01</td><td>Receita</td><td>100</td></tr></table>"
    document = {"id": "10", "version": 1, "reference": date(2024, 3, 31)}
    (tmp_path / "valid.html").write_bytes(payload)
    (tmp_path / "changed.html").write_bytes(payload + b"changed")
    pages = [
        {"file": name, "sha256": hashlib.sha256(payload).hexdigest(), "basis": "con"}
        for name in ("valid.html", "changed.html")
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps({"document": document, "pages": pages}, default=str)
    )
    with pytest.raises(ValueError, match="differs from its source manifest"):
        round5_cvm.attach_viewer_accounts(document, tmp_path)
    assert "accounts" not in document


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


def test_expected_filing_age_keeps_oldest_used_first_receipt_after_restatement():
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    previous = {
        "cvm_code": "1",
        "kind": "DFP",
        "group": "structured",
        "reference": date(2022, 12, 31),
        "receipt": datetime(2024, 1, 2, 13),
        "version": "1",
    }
    latest = {
        **previous,
        "kind": "ITR",
        "reference": date(2023, 9, 30),
        "receipt": datetime(2024, 1, 3, 13),
    }
    revision = {**previous, "receipt": datetime(2024, 1, 4, 13), "version": "2"}
    baseline = event_features([previous, latest], sessions)
    changed = event_features([previous, latest, revision], sessions)
    assert baseline["sessions_until_expected_filing"].equals(
        changed["sessions_until_expected_filing"]
    )
    assert changed["sessions_until_expected_filing_age_sessions"][2] == 2
    assert changed["sessions_since_financial_filing"][2] == 0


def test_first_observed_later_version_cannot_supply_original_filing_lag():
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    earlier_revision = {
        "cvm_code": "1",
        "kind": "DFP",
        "group": "structured",
        "reference": date(2022, 12, 31),
        "receipt": datetime(2024, 1, 2, 13),
        "version": "2",
    }
    latest = {
        **earlier_revision,
        "kind": "ITR",
        "reference": date(2023, 9, 30),
        "receipt": datetime(2024, 1, 3, 13),
        "version": "1",
    }
    unavailable = event_features([earlier_revision, latest], sessions)
    assert unavailable["sessions_until_expected_filing"].null_count() == len(sessions)
    assert unavailable["sessions_since_financial_filing"].to_list() == [0, 0, 1, 2]
    original = {**earlier_revision, "version": "1"}
    supported = event_features([original, latest], sessions)
    assert supported["sessions_until_expected_filing"][1] is not None


def test_header_only_original_lag_uses_own_date_without_resetting_filing_clocks():
    sessions = [date(2024, 1, d) for d in (2, 3, 4, 5)]
    latest = {
        "id": "20",
        "cvm_code": "1",
        "kind": "ITR",
        "group": "structured",
        "reference": date(2023, 9, 30),
        "receipt": datetime(2024, 1, 2, 15, 44),
        "version": "1",
    }
    header = {
        "id": "10",
        "cvm_code": "1",
        "kind": "DFP",
        "reference": date(2022, 12, 31),
        "receipt": sessions[1],
        "version": 1,
    }
    supplements = round5_cvm.header_only_filing_lags(
        [latest], [header, {**header, "id": "20"}]
    )
    assert len(supplements) == 1  # a linked RAD original keeps its exact minute
    revision = {
        **latest,
        "id": "21",
        "version": "2",
        "receipt": datetime(2024, 1, 4, 14),
    }
    baseline = event_features([latest, revision], sessions)
    supported = event_features([latest, revision, *supplements], sessions)
    assert baseline.head(2).equals(supported.head(2))
    assert supported["sessions_until_expected_filing"][2] is not None
    assert supported["sessions_since_financial_filing"].equals(
        baseline["sessions_since_financial_filing"]
    )
    assert supported["filing_is_dfp"].to_list() == [0, 0, 0, 0]


def test_expected_filing_calendar_is_not_clipped_or_backprojected():
    sessions = [
        date(2024, 11, 4),
        date(2024, 11, 5),
        date(2024, 11, 6),
        date(2024, 12, 30),
    ]
    calendar = {
        "base_through": date(2024, 12, 31),
        "through": date(2025, 12, 31),
        "available_date": date(2024, 11, 6),
        "full_sessions": sessions + [date(2025, 1, d) for d in (2, 3, 6, 7, 8)],
    }
    receipts = {date(2023, 12, 31): date(2024, 1, 8)}
    first = round5_cvm.expected_filing_distance(
        sessions[1], date(2024, 9, 30), receipts, sessions, calendar
    )
    mutated = {
        **calendar,
        "full_sessions": [
            d for d in calendar["full_sessions"] if d != date(2025, 1, 3)
        ],
    }
    assert first == round5_cvm.expected_filing_distance(
        sessions[1], date(2024, 9, 30), receipts, sessions, mutated
    )
    # Before announcement the weekday estimate reaches past the store end.
    assert first > len(sessions)
    exact = round5_cvm.expected_filing_distance(
        sessions[2], date(2024, 9, 30), receipts, sessions, calendar
    )
    assert exact == 6
    assert (
        round5_cvm.expected_filing_distance(
            sessions[2], date(2024, 9, 30), receipts, sessions, mutated
        )
        == exact - 1
    )


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
        "sector_code": "17",
        "sector_label": "Industry",
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
    changed["sector_code"] = "18"
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
        ("2.03.09", 0),
        ("3.01", cumulative * 10),
        ("3.03", cumulative * 2),
        ("3.11", cumulative),
        ("3.11.02", 0),
        ("6.01", cumulative),
    ):
        assign_account(
            d,
            "con",
            code,
            value,
            date(reference.year, 1, 1) if code.startswith(("3", "6")) else None,
            reference,
            {
                "1": "Ativo Total",
                "2.03": "Patrimônio Líquido Consolidado",
                "2.03.09": "Participação dos Acionistas Não Controladores",
                "3.01": "Receita de Venda de Bens e/ou Serviços",
                "3.03": "Resultado Bruto",
                "3.11": "Lucro/Prejuízo Consolidado do Período",
                "3.11.02": "Atribuído a Sócios Não Controladores",
                "6.01": "Caixa Líquido Atividades Operacionais",
            }[code],
        )
    return d


@pytest.mark.parametrize(
    "equity_code,nci_code,income_code,parent_income_code",
    (
        ("2.03", "2.03.09", "3.11", "3.11.01"),
        ("2.05", None, "3.13", "3.13.01"),
        ("2.08", "2.08.09", "3.09", "3.09.01"),
        ("2.07", "2.07.02", "3.11", "3.11.01"),
    ),
)
def test_published_account_descriptions_identify_bank_insurer_and_industrial_roles(
    equity_code,
    nci_code,
    income_code,
    parent_income_code,
):
    reference = date(2023, 12, 31)
    document = {"id": "bank", "reference": reference, "version": 1}
    basis = "ind" if nci_code is None else "con"
    # These account numbers do not mean equity/net income in the bank chart.
    for code, label in (
        ("2.03", "Resultados de Exercícios Futuros"),
        ("2.05", "Passivos Fiscais"),
        ("3.09", "IR Diferido"),
        ("3.11", "Reversão dos Juros sobre Capital Próprio"),
        ("2.07.02", "Reservas de Capital"),
        ("2.07.02", "Passivos sobre Ativos de Operações Descontinuadas"),
    ):
        assign_account(document, basis, code, 9999, None, reference, label)
    assert "accounts" not in document
    observations = [
        ("1", "Ativo Total", 100),
        (equity_code, "Patrimônio Líquido Consolidado", 40),
        (income_code, "Lucro/Prejuízo Consolidado do Período", 12),
    ]
    if nci_code:
        observations.append(
            (nci_code, "Participação dos Acionistas Não Controladores", 3)
        )
        observations.append(
            (parent_income_code, "Atribuído aos Sócios da Empresa Controladora", 10)
        )
    for code, label, value in observations:
        assign_account(
            document,
            basis,
            code,
            value,
            date(2023, 1, 1) if code.startswith("3") else None,
            reference,
            label,
        )
    state = fundamental_state({("DFP", reference): document}, "Bancos")
    assert state["liabilities_to_assets"] == pytest.approx(0.6)
    assert state["_book_equity"] == (37 if nci_code else 40)
    assert state["_earnings_ttm"] == (10 if nci_code else 12)
    assert "gross_profitability" in state and state["gross_profitability"] is None


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


def test_missing_consolidated_nci_is_not_an_observed_zero():
    document = report(date(2023, 12, 31), 100)
    ledger = {document["reference"]: document}
    del document["accounts"]["con"]["minority_equity"]
    missing = fundamental_state(ledger, "Industry")
    assert missing["_book_equity"] is None
    assert missing["liabilities_to_assets"] == pytest.approx(0.6)
    assert missing["_earnings_ttm"] == 100
    assign_account(
        document,
        "con",
        "2.07.01",
        38,
        None,
        document["reference"],
        "Patrimônio Líquido Atribuído ao Controlador",
    )
    assert fundamental_state(ledger, "Industry")["_book_equity"] == 38


def test_source_financial_chart_identifies_bank_without_sector_and_preserves_zero():
    document = report(date(2023, 12, 31), 100)
    document["accounts"]["con"]["gross_profit"].update(
        description="Resultado Bruto de Intermediação Financeira",
        value=0,
    )
    state = fundamental_state({document["reference"]: document})
    assert state["_financial"] == 1
    assert state["gross_profitability"] == 0  # a supported observed economic zero
    assert state["revenue_growth_yoy"] is None
    assert state["accruals_to_assets"] is None


def test_parent_income_may_exceed_group_income_when_nci_has_losses():
    document = report(date(2023, 12, 31), 100)
    assign_account(
        document,
        "con",
        "3.11.01",
        110,
        date(2023, 1, 1),
        document["reference"],
        "Atribuído a Sócios da Empresa Controladora",
    )
    assert fundamental_state({document["reference"]: document})["_earnings_ttm"] == 110


def test_parent_flow_requires_reported_parent_or_same_period_nci_income():
    document = report(date(2023, 12, 31), 100)
    ledger = {document["reference"]: document}
    minority = document["accounts"]["con"].pop("minority_income")
    assert fundamental_state(ledger)["_earnings_ttm"] is None
    document["accounts"]["con"]["minority_income"] = {**minority, "value": -10}
    assert fundamental_state(ledger)["_earnings_ttm"] == 110
    document["accounts"]["con"]["minority_income"]["start"] = date(2023, 4, 1)
    assert fundamental_state(ledger)["_earnings_ttm"] is None
    assert (
        trailing_twelve_months(ledger, "con", "net_income", document["reference"])[0]
        == 100
    )


def test_joined_parent_earnings_restatement_first_eligible_decision():
    sessions, document, rad, identity, market = family_fixture()
    del document["accounts"]["con"]["minority_income"]
    base, _ = fundamental_features(
        [deepcopy(document)], rad, identity, sessions, market
    )
    revised = deepcopy(document)
    revised.update(id="2", version=2, receipt=sessions[2])
    revised["accounts"]["con"]["minority_income"] = {
        **revised["accounts"]["con"]["net_income"],
        "value": -10,
    }
    changed, _ = fundamental_features(
        [deepcopy(document), revised],
        rad
        + [
            {
                **rad[0],
                "id": "2",
                "version": "2",
                "receipt": datetime(2024, 1, 4, 15, 44),
            }
        ],
        identity,
        sessions,
        market,
    )
    assert base["earnings_yield_ttm"].null_count() == len(sessions)
    assert base.head(2).equals(changed.head(2))
    assert changed["earnings_yield_ttm"][2] == pytest.approx(110 / 2000)
    assert changed["earnings_yield_ttm_age_sessions"][2] == 0


def test_statement_financial_flag_age_ignores_later_identity_metadata():
    sessions, document, rad, identity, market = family_fixture()
    document["accounts"]["con"]["gross_profit"]["description"] = (
        "Resultado Bruto de Intermediação Financeira"
    )
    identity = identity.with_columns(
        pl.lit(None, dtype=pl.String).alias("sector"),
        pl.lit(sessions[2]).alias("identity_known_date"),
    )
    values, _ = fundamental_features([document], rad, identity, sessions, market)
    assert values["fundamental_financial_flag"].to_list() == [1, 1, 1, 1]
    assert values["fundamental_financial_flag_age_sessions"].to_list() == [0, 1, 2, 3]


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


def write_account_unit_disposition(root, document):
    evidence = root / "own_note.txt"
    evidence.write_text("Current statement amounts are thousands of reais.")
    path = root / "account_unit_dispositions.json"
    path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document": {
                            key: str(document[key])
                            for key in (
                                "id",
                                "cnpj",
                                "cvm_code",
                                "kind",
                                "reference",
                                "version",
                            )
                        },
                        "accounts_before": document["accounts"],
                        "multiplier": 1000,
                        "reason": "Own-period fixture monetary rows reconcile to the note.",
                        "evidence": [
                            {
                                "path": str(evidence),
                                "sha256": round5_cvm.sha256(evidence),
                            }
                        ],
                    }
                ]
            },
            default=str,
        )
    )
    return path, evidence


def test_account_unit_correction_preserves_receipt_and_independent_capital(tmp_path):
    sessions, document, rad, identity, market = family_fixture()
    revised = deepcopy(document)
    revised.update(id="2", version=2, receipt=sessions[2])
    revised["accounts"]["con"]["cash_flow"]["value"] = -10
    original = deepcopy(revised)
    disposition, evidence = write_account_unit_disposition(tmp_path, revised)
    evidence_before = evidence.read_bytes()
    untouched = deepcopy(document)
    audit = round5_cvm.apply_account_unit_dispositions(tmp_path, [document, revised])
    assert document == untouched
    assert evidence.read_bytes() == evidence_before
    assert audit["sha256"] == round5_cvm.sha256(disposition)
    assert audit["applied_ids"] == ["2"]
    assert revised["accounts"]["con"]["cash_flow"]["value"] == -10000
    for key in original.keys() - {"accounts"}:
        assert revised[key] == original[key]
    changed_rad = rad + [
        {**rad[0], "id": "2", "version": "2", "receipt": datetime(2024, 1, 4, 15, 44)}
    ]
    base, _ = fundamental_features(
        [deepcopy(document)], rad, identity, sessions, market
    )
    changed, _ = fundamental_features(
        [deepcopy(document), revised], changed_rad, identity, sessions, market
    )
    assert base.head(2).equals(changed.head(2))
    assert changed["book_to_market"][2] == pytest.approx(
        base["book_to_market"][2] * 1000
    )
    assert changed["log_market_cap"].to_list() == base["log_market_cap"].to_list()
    assert changed["book_to_market_age_sessions"][2] == 0


@pytest.mark.parametrize("change", ("version", "period", "value", "basis", "evidence"))
def test_account_unit_correction_cannot_repair_different_source_rows(tmp_path, change):
    _, document, _, _, _ = family_fixture()
    _, evidence = write_account_unit_disposition(tmp_path, document)
    if change == "version":
        document["version"] += 1
    elif change == "period":
        document["accounts"]["con"]["assets"]["end"] = date(2022, 12, 31)
    elif change == "value":
        document["accounts"]["con"]["assets"]["value"] *= 1000
    elif change == "basis":
        document["accounts"]["ind"] = document["accounts"].pop("con")
    else:
        evidence.write_text("A different source note.")
    with pytest.raises(ValueError, match="Account unit correction"):
        round5_cvm.apply_account_unit_dispositions(tmp_path, [document])


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
    change = {
        "cnpj": "123",
        "cvm_code": "000001",
        "effective": date(2024, 1, 2),
        "available_index": 2,
    }
    result, _ = fundamental_features(
        [document], rad, identity, sessions, market, [change]
    )
    assert result["log_market_cap"][1] is not None
    assert result["log_market_cap"][2] is None
    unrelated, _ = fundamental_features(
        [document], rad, identity, sessions, market, [{**change, "cvm_code": "000002"}]
    )
    assert unrelated["log_market_cap"][2] is not None


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
        "sector_code": "18",
        "sector_label": "Services",
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
        sector_code="19",
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
        "sector_code": "17",
        "sector_label": "Industry",
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
    result, audit = fundamental_features(
        [*deepcopy(documents), revised], [*rad, revised_rad], identity, sessions, market
    )
    assert base.head(2).equals(result.head(2))
    assert base["gross_profitability"][2] == pytest.approx(2.3)
    assert result["current_balance_version"].to_list() == [1, 1, 1, 1]
    version_counts = {
        row["feature"]: row for row in audit["feature_version_composition"]
    }
    assert version_counts["gross_profitability"]["first_version_only"] == 2
    assert version_counts["gross_profitability"]["includes_later_version"] == 2
    assert version_counts["liabilities_to_assets"]["first_version_only"] == 4
    assert version_counts["liabilities_to_assets"]["includes_later_version"] == 0
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
        "sector_code": "17",
        "sector_label": "Industry",
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
        "sector_code": "17",
        "sector_label": "Industry",
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
