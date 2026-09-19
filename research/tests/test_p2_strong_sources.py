from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

from brazil_rv.preprocessing.b3_options_open_interest import (
    Identity,
    _nested_xml_member,
    parse_instruments,
    parse_price_report,
)
from brazil_rv.preprocessing.bdi_lending_strong import parse_registered_lines
from brazil_rv.preprocessing.dce_iron_ore import _robust


def _nested_archive(path: Path, xml: str) -> None:
    nested_bytes = io.BytesIO()
    with zipfile.ZipFile(nested_bytes, "w", zipfile.ZIP_DEFLATED) as nested:
        nested.writestr("BVBG.000.01_20240101.xml", "<obsolete />")
        nested.writestr("BVBG.000.02_20240101.xml", xml)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("nested.zip", nested_bytes.getvalue())


def test_option_oi_maps_explicit_instrument_ids_to_exact_cash_isin(
    tmp_path: Path,
) -> None:
    instrument_xml = """
    <Root>
      <Instrm><RptParams/><FinInstrmId><OthrId><Id>CASH1</Id></OthrId></FinInstrmId>
        <FinInstrmAttrCmon/><InstrmInf><EqtyInf><ISIN>BRTESTACNOR1</ISIN></EqtyInf></InstrmInf></Instrm>
      <Instrm><RptParams/><FinInstrmId><OthrId><Id>OPTION1</Id></OthrId></FinInstrmId>
        <FinInstrmAttrCmon/><InstrmInf><OptnOnEqtsInf>
          <TckrSymb>WRONGPREFIX</TckrSymb><ExrcPric>10</ExrcPric>
          <XprtnDt>2024-02-01</XprtnDt><OptnTp>PUTT</OptnTp>
          <UndrlygInstrmId><OthrId><Id>CASH1</Id></OthrId></UndrlygInstrmId>
          <TradgStartDt>2023-01-01</TradgStartDt><TradgEndDt>2024-02-01</TradgEndDt>
        </OptnOnEqtsInf></InstrmInf></Instrm>
    </Root>
    """
    price_xml = """
    <Root>
      <PricRpt><FinInstrmId><OthrId><Id>CASH1</Id></OthrId></FinInstrmId>
        <TradDtls><FinInstrmQty>1000</FinInstrmQty><LastPric>8</LastPric></TradDtls></PricRpt>
      <PricRpt><FinInstrmId><OthrId><Id>OPTION1</Id></OthrId></FinInstrmId>
        <FinInstrmAttrbts><OpnIntrst>250</OpnIntrst></FinInstrmAttrbts></PricRpt>
    </Root>
    """
    instrument_path = tmp_path / "IN240102.zip"
    price_path = tmp_path / "PR240102.zip"
    _nested_archive(instrument_path, instrument_xml)
    _nested_archive(price_path, price_xml)
    identities = {
        "BRTESTACNOR1": Identity(
            security_id="ISIN:BRTESTACNOR1",
            isin="BRTESTACNOR1",
            effective_from=date(2023, 1, 1),
            effective_to_inclusive=date(2024, 12, 31),
        )
    }

    cash, options, _ = parse_instruments(instrument_path, date(2024, 1, 2), identities)
    quantity, aggregates, _ = parse_price_report(
        price_path, date(2024, 1, 2), cash, options
    )

    assert cash == {"CASH1": "ISIN:BRTESTACNOR1"}
    assert quantity == {"ISIN:BRTESTACNOR1": 1000}
    aggregate = aggregates["ISIN:BRTESTACNOR1"]
    assert aggregate.put_oi == 250
    assert aggregate.call_oi == 0
    assert aggregate.valid_moneyness_oi == 250
    assert _nested_xml_member(instrument_path)[1].startswith("BVBG.000.02")


def test_lending_registered_rate_row_parses_locale_numbers() -> None:
    rows = parse_registered_lines(
        [
            "27/03/2024   5GTK11     BR5GTKCTF000          BLUESTAR 5G COM            "
            "Registro             12          1.543       141.199,93    5,00%         "
            "5,00%      5,00%     5,00%         5,00%      5,00%"
        ],
        date(2024, 3, 27),
    )

    assert len(rows) == 1
    assert rows[0].quantity == 1543
    assert rows[0].value_brl == 141199.93
    assert rows[0].taker_avg == 5.0


def test_dce_robust_scaler_consumes_prior_history_only() -> None:
    history = [float(index) for index in range(20)]
    baseline = _robust(20.0, history)
    mutated_future = _robust(20.0, history + [10_000.0])

    assert baseline[1]
    assert mutated_future != baseline
    assert _robust(20.0, history) == baseline


def test_lending_rate_wrapped_year_uses_printed_digit_and_keeps_zero_flow() -> None:
    rows = parse_registered_lines(
        [
            "27/12/202 PETR4 BRPETRACNPR6 PETROBRAS Registro 0 0 0.00 "
            "0.10% 0.20% 0.30% 0.40% 0.50% 0.60%",
            "4 SA",
            "27/12/202 VALE3 BRVALEACNOR0 VALE Registro 12 1,543 141,199.93 "
            "0.10% 0.20% 0.30% 0.40% 0.50% 0.60%",
            "4",
        ],
        date(2024, 12, 27),
    )
    assert len(rows) == 2
    assert rows[0].quantity == 0
    assert rows[0].taker_avg == 0.5
    assert rows[1].quantity == 1543
    assert rows[1].value_brl == 141199.93


def test_lending_electronic_rows_can_follow_the_wrapped_date() -> None:
    rows = parse_registered_lines(
        [
            "27/11/202                        ISHARES IBOVESPA   Neg. Elet",
            "4 BOVA11 BRBOVACTF003 FUNDO DE INDICE rônica "
            "601 1,913,212242,002,185.88 0.10% 1.01% 2.25% 0.10% 1.01% 2.25%",
            "                                                   D+1",
        ],
        date(2024, 11, 27),
    )
    assert len(rows) == 1
    assert (rows[0].quantity, rows[0].value_brl) == (1913212, 242002185.88)
    assert rows[0].taker_avg == 1.01


def test_lending_wrapped_isin_uses_its_own_printed_check_digit() -> None:
    rows = parse_registered_lines(
        [
            f"{'27/11/202  BEWW39':24}BRBEWWBDR00         ISHARES Registro "
            "0 0 0.00 8.00% 8.57% 10.00% 8.00% 8.57% 10.00%",
            f"{'4':24}7                    ETF",
            f"{'27/11/202':24}BRBEWWBDR00         ISHARES Neg. Elet",
            f"{'4          BEWW39':24}7                   ETF rônica "
            "2 30 2,216.40 8.57% 8.57% 8.57% 8.57% 8.57% 8.57%",
        ],
        date(2024, 11, 27),
    )
    assert [r.isin for r in rows] == ["BRBEWWBDR007", "BRBEWWBDR007"]
    assert [r.quantity for r in rows] == [0, 30]


def test_lending_wrapped_ticker_and_sparse_table_are_retained() -> None:
    rows = parse_registered_lines(
        [
            "27/11/202 DEBBETF1      BTG PACTUAL TEVA ETF",
            "4         1 BRDEBBCTF000 DEBENTURES DI FUNDO Renda "
            "0 0 0.00 0.10% 0.10% 0.10% 0.10% 0.10% 0.10%",
        ],
        date(2024, 11, 27),
    )
    assert len(rows) == 1
    assert rows[0].ticker == "DEBBETF11"


def test_lending_incomplete_printed_rows_cannot_silently_bias_the_average() -> None:
    import pytest

    with pytest.raises(ValueError, match="Incomplete registered-loan extraction"):
        parse_registered_lines(
            [
                "27/11/2024 PETR4 BROKEN_ID PETROBRAS Registro "
                "1 100 1000.00 0.1% 0.2% 0.3% 0.1% 0.2% 0.3%"
            ],
            date(2024, 11, 27),
        )
    with pytest.raises(ValueError, match="row date differs"):
        parse_registered_lines(
            [
                "27/11/202 PETR4 BRPETRACNPR6 PETROBRAS Registro "
                "1 100 1000.00 0.1% 0.2% 0.3% 0.1% 0.2% 0.3%",
                "3",
            ],
            date(2024, 11, 27),
        )
