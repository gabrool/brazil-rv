from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing import cvm_rad_events
from brazil_rv.preprocessing.cvm_rad_events import (
    CaptchaRequired,
    FcaSecurity,
    acquire_rad_history,
    build_session_issuer_map,
    first_available_decision,
    parse_rad_data,
    state_rows_for_session,
)


def _rad_row(*, category: str = "Fato Relevante") -> str:
    action = (
        "<i onclick=OpenPopUpVer('frmExibirArquivoIPEExterno.aspx?"
        "NumeroProtocoloEntrega=1186748')></i>"
        "<i onclick=OpenDownloadDocumentos('711474','1','1186748','IPE')></i>"
    )
    return (
        "$&".join(
            [
                "00951-2",
                "PETROLEO BRASILEIRO S.A. PETROBRAS",
                category,
                " - ",
                "<spanOrder>Assunto</spanOrder> - ",
                "<spanOrder>20240126</spanOrder> 26/01/2024",
                "<spanOrder>20240126</spanOrder> 26/01/2024 10:15",
                "Ativo",
                "2",
                "AP",
                action,
                "Assunto",
            ]
        )
        + "$&&*"
    )


def test_rad_parser_preserves_exact_delivery_version_status_and_keys() -> None:
    event = parse_rad_data(_rad_row(), "raw.json")[0]
    assert event.cvm_code == "009512"
    assert event.delivery_timestamp == datetime(2024, 1, 26, 10, 15)
    assert event.reference_date == date(2024, 1, 26)
    assert event.status == "Ativo"
    assert event.version == "2"
    assert event.sequence_id == "711474"
    assert event.protocol_id == "1186748"
    assert event.event_group == "material_fact"


def test_rad_parser_uses_enet_sequence_for_structured_filings() -> None:
    fields = _rad_row(category="ITR - Informações Trimestrais").split("$&")
    fields[10] = (
        "<i onclick=OpenPopUpVer('frmGerenciaPaginaFRE.aspx?"
        "NumeroSequencialDocumento=142766&CodigoTipoInstituicao=1')></i>"
    )
    event = parse_rad_data("$&".join(fields), "structured.json")[0]
    assert event.sequence_id == "142766"
    assert event.protocol_id is None
    assert event.event_group == "itr_dfp"


def test_availability_is_first_decision_strictly_after_receipt() -> None:
    friday = date(2024, 1, 26)
    monday = date(2024, 1, 29)
    sessions = [friday, monday]
    assert first_available_decision(datetime(2024, 1, 26, 10, 14), sessions) == (
        friday,
        0,
    )
    assert first_available_decision(datetime(2024, 1, 26, 10, 15), sessions) == (
        friday,
        1,
    )
    assert first_available_decision(datetime(2024, 1, 26, 14, 44), sessions) == (
        friday,
        54,
    )
    assert first_available_decision(datetime(2024, 1, 26, 14, 45), sessions) == (
        monday,
        0,
    )
    assert first_available_decision(datetime(2024, 1, 27, 9, 0), sessions) == (
        monday,
        0,
    )


def test_identity_uses_exact_fca_ticker_and_broadcasts_only_exact_classes(
    tmp_path: Path,
) -> None:
    trade_date = date(2024, 1, 26)
    store = tmp_path / "store"
    store.mkdir()
    pl.DataFrame({"date_idx": [0], "trade_date": [trade_date]}).with_columns(
        pl.col("date_idx").cast(pl.Int32), pl.col("trade_date").cast(pl.Date)
    ).write_parquet(store / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": [0, 1, 2],
            "security_id": ["ISIN:PETR3", "ISIN:PETR4", "ISIN:PETZ3"],
        }
    ).with_columns(pl.col("equity_slot").cast(pl.Int16)).write_parquet(
        store / "equity_index.parquet"
    )
    np.save(store / "equity_membership.npy", np.ones((1, 3), dtype=np.bool_))
    cotahist = tmp_path / "cotahist" / "year=2024"
    cotahist.mkdir(parents=True)
    pl.DataFrame(
        {
            "trade_date": [trade_date] * 3,
            "security_id": ["ISIN:PETR3", "ISIN:PETR4", "ISIN:PETZ3"],
            "security_id_is_fallback": [False] * 3,
            "market_type": [10] * 3,
            "bdi_code": ["02"] * 3,
            "ticker": ["PETR3", "PETR4", "PETZ3"],
        }
    ).with_columns(pl.col("trade_date").cast(pl.Date)).write_parquet(
        cotahist / "ticker_observations_2024.parquet"
    )
    records = [
        FcaSecurity(
            "33000167000101", "PETR3", date(2023, 1, 1), date(2018, 1, 1), None
        ),
        FcaSecurity(
            "33000167000101", "PETR4", date(2023, 1, 1), date(2018, 1, 1), None
        ),
    ]
    frame, _, audit = build_session_issuer_map(
        cotahist.parent, store, records, through=trade_date
    )
    assert set(frame.get_column("security_id")) == {"ISIN:PETR3", "ISIN:PETR4"}
    assert frame.get_column("cnpj").unique().to_list() == ["33000167000101"]
    assert audit["unmapped_security_days"] == 1


def test_future_event_mutation_cannot_change_earlier_state() -> None:
    issuer_rows = [("ISIN:PETR4", "33000167000101")]
    baseline = {
        "33000167000101": {
            "material_fact": [(1, 10)],
            "itr_dfp": [],
            "market_communication": [],
            "corporate_action": [],
        }
    }
    mutated = {
        "33000167000101": {
            **baseline["33000167000101"],
            "material_fact": [(1, 10), (3, 0)],
        }
    }
    earlier = state_rows_for_session(2, date(2024, 1, 26), issuer_rows, baseline)
    changed = state_rows_for_session(2, date(2024, 1, 26), issuer_rows, mutated)
    assert earlier == changed
    assert all(row[4] == 1.0 for row in earlier)
    assert all(row[-1] is True for row in earlier)


def test_acquisition_stops_and_records_when_rad_requests_captcha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = b'<input id="hdnHabilitaCaptcha" value="N">'
    response = json.dumps(
        {
            "d": {
                "temErro": False,
                "expirouSessao": False,
                "msgErro": "",
                "dados": "",
                "SolicitarCaptcha": "S",
            }
        }
    ).encode()
    calls = iter((page, response))
    monkeypatch.setattr(cvm_rad_events, "_request", lambda *args, **kwargs: next(calls))
    output = tmp_path / "raw"
    with pytest.raises(CaptchaRequired, match="without bypass"):
        acquire_rad_history(output, years=(2024,), pause_seconds=0)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "captcha_required"
    assert manifest["responses"][0]["solicitar_captcha"] == "S"
    assert manifest["captcha_bypass"] is False
