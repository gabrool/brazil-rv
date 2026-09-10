from datetime import date, timedelta
import io
import zipfile

import polars as pl
import numpy as np

from brazil_rv.v2.round5_b3 import (
    RATE_FIELDS,
    stitch_registered_rates,
    parse_options_snapshot,
    cotahist_option_quantities,
    activity_decision_features,
    stitch_legacy_balances,
    lending_decision_features,
    lending_utilization_features,
    cash_price_report,
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


def test_cash_report_gap_uses_exact_dated_ticker_without_instrument_file(tmp_path):
    pr = tmp_path / "PR.zip"
    _nested(
        pr,
        {
            "known.xml": "<Doc><CreDtAndTm>2023-12-08T20:00:00</CreDtAndTm>"
            "<PricRpt><TradDt><Dt>2023-12-08</Dt></TradDt><TckrSymb>OLD3</TckrSymb>"
            "<FinInstrmQty>100</FinInstrmQty><RglrTraddCtrcts>90</RglrTraddCtrcts>"
            "</PricRpt></Doc>"
        },
    )
    cash = pl.DataFrame(
        {
            "source_trade_date": [date(2023, 12, 8), date(2023, 12, 11)],
            "ticker": ["OLD3", "OLD3"],
            "isin": ["ABC", "DEF"],
        }
    )
    rows, _ = cash_price_report(pr, date(2023, 12, 8), date(2023, 12, 11), cash)
    assert rows["isin"].to_list() == ["ABC"]
    assert rows["quantity"].to_list() == [100.0]
    assert rows["regular_quantity"].to_list() == [90.0]
    assert rows["nonregular_quantity"].to_list() == [None]


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


def test_activity_windows_exclude_same_day_and_do_not_fill_listing_gaps():
    sessions = [date(2024, 1, 1) + timedelta(days=i) for i in range(31)]
    cash = pl.DataFrame(
        {
            "source_trade_date": sessions,
            "isin": ["ABC"] * 31,
            "quantity": [1000.0] * 31,
            "volume_brl": [10000.0] * 31,
            "trades": [10.0] * 31,
        }
    )
    volumes = pl.DataFrame(
        {
            "source_trade_date": sessions,
            "isin": ["ABC"] * 31,
            "call_quantity": [20.0] * 31,
            "put_quantity": [10.0] * 31,
        }
    )
    nonregular = cash.with_columns(
        pl.lit(1000.0).alias("regular_quantity"),
        pl.lit(None, dtype=pl.Float64).alias("nonregular_quantity"),
    )
    option, micro = activity_decision_features(
        cash, volumes, pl.DataFrame(), nonregular, sessions
    )
    assert (
        option.filter(pl.col("date") == sessions[20])[
            "option_to_stock_volume_20"
        ].item()
        == 0.03
    )
    assert micro["avg_trade_size_20"].drop_nulls().unique().to_list() == [1000.0]
    assert micro["after_hours_volume_share_5"].drop_nulls().unique().to_list() == [0.0]
    assert option["put_call_oi_log_ratio"].null_count() == option.height
    changed = volumes.with_columns(
        pl.when(pl.col("source_trade_date") == sessions[22])
        .then(200.0)
        .otherwise(pl.col("call_quantity"))
        .alias("call_quantity")
    )
    mutated, _ = activity_decision_features(
        cash, changed, pl.DataFrame(), nonregular, sessions
    )
    assert option.filter(pl.col("date") <= sessions[22]).equals(
        mutated.filter(pl.col("date") <= sessions[22])
    )
    assert (
        option.filter(pl.col("date") == sessions[23])[
            "option_to_stock_volume_20"
        ].item()
        != mutated.filter(pl.col("date") == sessions[23])[
            "option_to_stock_volume_20"
        ].item()
    )
    missing, _ = activity_decision_features(
        cash,
        volumes.filter(pl.col("source_trade_date") != sessions[22]),
        pl.DataFrame(),
        nonregular,
        sessions,
    )
    assert missing.filter(pl.col("date") == sessions[23]).height == 0

    opening_positions = pl.DataFrame(
        {
            "source_trade_date": sessions,
            "isin": ["ABC"] * 31,
            "listed_series": [2] * 31,
            "call_oi": [100.0 + i for i in range(31)],
            "put_oi": [50.0] * 31,
            "oi_all_listed_observed": [True] * 31,
        }
    )
    changed_cash = cash.with_columns(
        pl.when(pl.col("source_trade_date") == sessions[22])
        .then(100000.0)
        .otherwise(pl.col("quantity"))
        .alias("quantity")
    )
    oi, _ = activity_decision_features(
        changed_cash, volumes, opening_positions, nonregular, sessions
    )
    row = oi.filter(pl.col("date") == sessions[23]).row(0, named=True)
    assert row["delta_oi_to_volume_1"] == 0.001
    assert row["delta_oi_to_volume_1_age_sessions"] == 2
    assert row["put_call_oi_log_ratio_age_sessions"] == 2
    assert row["option_to_stock_volume_20_age_sessions"] == 1


def test_legacy_balance_identity_requires_exact_position_date_and_preserves_old():
    days = [date(2020, 1, d) for d in (2, 3, 6, 7)]
    old = pl.DataFrame(
        {
            "source_position_date": [days[2]],
            "source_report_date": [days[2]],
            "available_date": [days[3]],
            "security_id": ["ISIN:ABC"],
            "source_identity_method": ["old"],
            "lending_balance_quantity": [30],
            "lending_balance_brl": [300.0],
        }
    )
    raw = pl.DataFrame(
        {
            "position_date": [days[0], days[1]],
            "report_date": [days[0], days[1]],
            "ticker": ["OLD3", "OLD3"],
            "isin": [None, None],
            "quantity": [10, 20],
            "balance_brl": [100.0, 200.0],
        }
    )
    cash = pl.DataFrame(
        {
            "source_trade_date": [days[0], days[1]],
            "ticker": ["OLD3", "NEW3"],
            "isin": ["ABC", "ABC"],
        }
    )
    result, audit = stitch_legacy_balances(old, raw, cash, days[:2], days)
    assert audit["added_rows"] == 1
    assert audit["identity_unresolved_rows"] == 1
    assert result.filter(pl.col("available_date") == days[3]).equals(old)
    assert (
        result.filter(pl.col("available_date") == days[1])[
            "lending_balance_quantity"
        ].item()
        == 10
    )


def test_lending_feature_uses_publication_date_and_preserves_reference_age():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(31)]
    balance = pl.DataFrame(
        {
            "source_position_date": [days[20]],
            "source_report_date": [days[21]],
            "available_date": [days[22]],
            "security_id": ["ISIN:ABC"],
            "source_identity_method": ["fixture"],
            "lending_balance_quantity": [30],
            "lending_balance_brl": [300.0],
        }
    )
    rates = pl.DataFrame(
        {
            "source_trade_date": days[:30],
            "available_date": days[1:],
            "security_id": ["ISIN:ABC"] * 30,
            "registered_quantity": list(range(1, 31)),
            "registered_contracts": [1] * 30,
            "annual_taker_rate": [0.03] * 30,
        }
    )
    result = lending_decision_features(
        balance, rates, days, ["ABC"], np.full((31, 1), 100.0)
    )
    row = result.filter(pl.col("date") == days[22]).row(0, named=True)
    assert row["loan_balance_to_volume_20"] == 3.0
    assert row["loan_balance_to_volume_20_age_sessions"] == 2
    assert row["loan_rate_age_sessions"] == 1
    assert row["new_loan_volume_surprise"] is not None
    changed = lending_decision_features(
        balance.with_columns(pl.lit(600.0).alias("lending_balance_brl")),
        rates,
        days,
        ["ABC"],
        np.full((31, 1), 100.0),
    )
    assert result.filter(pl.col("date") < days[22]).equals(
        changed.filter(pl.col("date") < days[22])
    )
    assert (
        changed.filter(pl.col("date") == days[22])["loan_balance_to_volume_20"].item()
        == 6.0
    )
    assert result.filter(pl.col("date") == days[20])[
        "new_loan_volume_surprise"
    ].item() is None
    with_five_gaps = rates.filter(~pl.col("source_trade_date").is_in(days[5:10]))
    sparse = lending_decision_features(
        balance, with_five_gaps, days, ["ABC"], np.full((31, 1), 100.0)
    )
    observed = np.asarray([i + 1 for i in range(20) if not 5 <= i < 10])
    expected = (21 - observed.mean()) / observed.std(ddof=1)
    assert np.isclose(
        sparse.filter(pl.col("date") == days[21])["new_loan_volume_surprise"].item(),
        expected,
    )
    with_six_gaps = with_five_gaps.filter(pl.col("source_trade_date") != days[10])
    insufficient = lending_decision_features(
        balance, with_six_gaps, days, ["ABC"], np.full((31, 1), 100.0)
    )
    assert insufficient.filter(pl.col("date") == days[21])[
        "new_loan_volume_surprise"
    ].item() is None


def test_utilization_uses_received_float_snapshot_and_known_unit_boundaries():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(6)]
    balances = pl.DataFrame(
        {
            "source_position_date": days[1:5],
            "available_date": days[2:6],
            "security_id": ["ISIN:ABC"] * 4,
            "lending_balance_quantity": [10] * 4,
        }
    )
    identity = pl.DataFrame(
        {
            "date": days[2:6],
            "isin": ["ABC"] * 4,
            "cnpj": ["12345678000202"] * 4,
            "cvm_code": ["123"] * 4,
            "class": ["ON"] * 4,
            "identity_effective_start": [days[0]] * 4,
        }
    )
    floats = pl.DataFrame(
        {
            "date": [days[3]],
            "snapshot_date": [days[1]],
            "reference": [date(2023, 1, 1)],
            "cnpj": ["12345678000101"],
            "cvm_code": ["123"],
            "class": ["ON"],
            "free_float_shares": [100.0],
            "document_id": ["1"],
            "version": [1],
        }
    )
    barriers = np.zeros((7, 1), dtype=np.int32)
    barriers[5:] = 1  # A source-session unit change on days[4].
    market = {"columns": {"ABC": 0}, "barrier_prefix": barriers}
    result, _ = lending_utilization_features(
        balances, floats, identity, days, market, []
    )
    assert result["date"].to_list() == days[3:5]
    assert result["utilization_proxy"].to_list() == [0.1, 0.1]
    assert result["utilization_proxy_age_sessions"].to_list() == [1, 1]
    changed, _ = lending_utilization_features(
        balances,
        floats.with_columns(pl.lit(200.0).alias("free_float_shares")),
        identity,
        days,
        market,
        [],
    )
    assert result.filter(pl.col("date") < days[3]).equals(
        changed.filter(pl.col("date") < days[3])
    )
    assert changed["utilization_proxy"].to_list() == [0.05, 0.05]
    wrong_issuer, _ = lending_utilization_features(
        balances,
        floats.with_columns(pl.lit("456").alias("cvm_code")),
        identity,
        days,
        market,
        [],
    )
    assert wrong_issuer.is_empty()
    future_known_change = {
        "cnpj": "12345678000101",
        "cvm_code": "123",
        "available_index": 4,
        "effective": days[2],
    }
    capital, _ = lending_utilization_features(
        balances, floats, identity, days, market, [future_known_change]
    )
    assert capital["date"].to_list() == [days[3]]
    preferred = identity.with_columns(pl.lit("PN").alias("class"))
    preferred = pl.concat(
        [preferred, preferred.with_columns(pl.lit("DEF").alias("isin"))]
    )
    ambiguous, _ = lending_utilization_features(
        balances,
        floats.with_columns(pl.lit("PN").alias("class")),
        preferred,
        days,
        market,
        [],
    )
    assert ambiguous.is_empty()
