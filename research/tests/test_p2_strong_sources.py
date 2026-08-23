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
