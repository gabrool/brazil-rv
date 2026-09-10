from datetime import date
import io
import zipfile

import polars as pl

from brazil_rv.v2.round5_b3 import (
    RATE_FIELDS,
    stitch_registered_rates,
    parse_options_snapshot,
    cotahist_option_quantities,
)


def test_registered_rates_preserve_old_exclude_zero_and_use_exact_next_session():
    days = [date(2024, 12, d) for d in (26, 27, 30)]
    old = pl.DataFrame(
        {
            "source_trade_date": [days[0]],
            "available_date": [days[1]],
            "security_id": ["ISIN:ABC"],
            "registered_contracts": [2],
            "registered_quantity": [100],
            "annual_taker_rate": [0.030000000000000002],
        }
    )
    records = []
    for day, isin, quantity, rate in [
        (days[0], "ABC", 100, 3.0),
        (days[1], "ABC", 0, 999.0),
        (days[1], "DEF", 100, 4.0),
        (days[1], "DEF", 300, 8.0),
        (days[2], "ABC", 100, 5.0),
        (days[0], "DEF", 100, 8.0),
    ]:
        records.append(
            {
                "report_date": day,
                "isin": isin,
                "ticker": isin,
                "contracts": 2,
                "quantity": quantity,
                **{field: rate for field in RATE_FIELDS},
            }
        )
    identities = pl.DataFrame(
        {
            "isin": ["ABC", "DEF"],
            "first_date": [days[0], days[1]],
            "last_date": [days[-1], days[-1]],
        }
    )
    raw = pl.DataFrame(records)
    result, _, audit = stitch_registered_rates(old, raw, identities, days)
    assert result.height == 2
    assert result.row(0, named=True) == old.row(0, named=True)
    assert result.row(1, named=True)["annual_taker_rate"] == 0.07
    assert result.row(1, named=True)["available_date"] == days[2]
    assert audit["zero_flow_rows_excluded"] == 1
    mutated = raw.with_columns(
        pl.when(pl.col("report_date") == days[1])
        .then(pl.col("taker_avg") * 2)
        .otherwise(pl.col("taker_avg"))
        .alias("taker_avg")
    )
    changed, _, _ = stitch_registered_rates(old, mutated, identities, days)
    assert changed.filter(pl.col("available_date") < days[2]).equals(
        result.filter(pl.col("available_date") < days[2])
    )
    assert changed.row(1, named=True)["annual_taker_rate"] == 0.14


def _nested(path, versions):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        for name, xml in versions.items():
            archive.writestr(name, xml)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested.zip", inner.getvalue())


def test_option_snapshot_uses_known_version_and_keeps_missing_oi_unknown(tmp_path):
    header = "<Doc><CreDtAndTm>2024-12-27T20:00:00</CreDtAndTm>"

    def instrument(identifier, body):
        return (
            f"<Instrm><ActvtyInd>true</ActvtyInd><FinInstrmId><Id>{identifier}</Id></FinInstrmId>"
            f"<FinInstrmAttrCmon><Mkt>10</Mkt></FinInstrmAttrCmon><InstrmInf>{body}</InstrmInf></Instrm>"
        )

    cash = instrument("CASH", "<EqtyInf><ISIN>ABC</ISIN></EqtyInf>")
    option = "<OptnOnEqtsInf><UndrlygInstrmId><Id>CASH</Id></UndrlygInstrmId><OptnTp>CALL</OptnTp><TckrSymb>XYZ</TckrSymb><TradgStartDt>2024-12-01</TradgStartDt><TradgEndDt>2025-01-31</TradgEndDt><XprtnDt>2025-01-31</XprtnDt></OptnOnEqtsInf>"
    ins = tmp_path / "IN.zip"
    pr = tmp_path / "PR.zip"
    _nested(ins, {"in.xml": header + cash + instrument("OPT", option) + "</Doc>"})
    report = (
        "<PricRpt><TradDt><Dt>2024-12-27</Dt></TradDt><FinInstrmId><Id>OPT</Id></FinInstrmId>"
        "<TckrSymb>XYZ</TckrSymb><FinInstrmQty>100</FinInstrmQty>{}</PricRpt>"
    )
    _nested(
        pr,
        {
            "before.xml": header + report.format("") + "</Doc>",
            "future.xml": header.replace("2024-12-27", "2025-01-02")
            + report.format("<OpnIntrst>999</OpnIntrst>")
            + "</Doc>",
        },
    )
    rows, _, audit = parse_options_snapshot(
        ins,
        pr,
        date(2024, 12, 27),
        date(2024, 12, 30),
        {"ABC": (date(2020, 1, 1), date(2024, 12, 30))},
    )
    assert rows["oi_observed_series"].to_list() == [0]
    assert rows["oi_all_listed_observed"].to_list() == [False]
    assert audit["pr"]["selected_member"] == "before.xml"
    assert audit["pr"]["versions_after_decision_excluded"] == 1


def test_cotahist_options_use_explicit_dated_isin_not_ticker_prefix(tmp_path):
    row = list(" " * 245)
    for start, text in [
        (0, "01"),
        (2, "20100104"),
        (12, "WRONGA20"),
        (24, "070"),
        (152, "000000000000000123"),
        (202, "20100118"),
        (230, "BRBBASACNOR3"),
    ]:
        row[start : start + len(text)] = text
    path = tmp_path / "COTAHIST_A2010.ZIP"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("COTAHIST_A2010.TXT", "".join(row))
    result = cotahist_option_quantities(
        path,
        {"BRBBASACNOR3": (date(2010, 1, 4), date(2024, 12, 30))},
        end=date(2024, 12, 30),
    )
    assert result["isin"].to_list() == ["BRBBASACNOR3"]
    assert result["call_quantity"].to_list() == [123]
