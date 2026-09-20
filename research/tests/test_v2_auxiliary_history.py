from datetime import date, timedelta

import polars as pl
import pytest

from brazil_rv.v2.auxiliary_history import (
    renamed_activity_features,
    renamed_oddlot_features,
)
from brazil_rv.v2.round5_b3 import activity_decision_features


def test_activity_continuation_retains_missing_support_and_publication_clock():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(31)]
    cash = pl.DataFrame(
        {
            "source_trade_date": days[:-1],
            "isin": ["OLD"] * 22 + ["NEW"] * 8,
            "quantity": [100.0] * 30,
            "volume_brl": [1000.0] * 30,
            "trades": [10.0] * 30,
        }
    )
    qty = cash.select("source_trade_date", "isin").with_columns(
        pl.lit(10.0).alias("call_quantity"), pl.lit(5.0).alias("put_quantity")
    )
    oi = qty.select("source_trade_date", "isin").with_columns(
        pl.Series("available_date", days[1:]),
        pl.lit(10).alias("listed_series"),
        pl.lit(3).alias("oi_observed_series"),
        pl.lit(4.0).alias("call_oi"),
        pl.lit(2.0).alias("put_oi"),
        pl.lit(False).alias("oi_all_listed_observed"),
    )
    pr = cash.with_columns(
        pl.Series("available_date", days[1:]),
        pl.lit(90.0).alias("regular_quantity"),
        pl.lit(None, dtype=pl.Float64).alias("nonregular_quantity"),
    )
    args = dict(predecessor="OLD", successor="NEW", effective=days[22], known=days[24])
    actual = renamed_activity_features(cash, qty, oi, pr, days, **args)
    uninterrupted = activity_decision_features(
        *(f.with_columns(pl.lit("NEW").alias("isin")) for f in (cash, qty, oi, pr)),
        days,
    )
    for got, expected in zip(actual, uninterrupted):
        assert got.equals(expected.filter(pl.col("date") >= days[24]))
    assert actual[0]["put_call_oi_log_ratio"].null_count() == actual[0].height
    assert actual[0]["delta_oi_to_volume_1"].null_count() == actual[0].height
    assert actual[1]["avg_trade_size_20"].to_list() == [100.0] * 7
    # No future source row can affect earlier output; missing source support is
    # not repaired by borrowing a different day's observation.
    cutoff = days[26]
    earlier = renamed_activity_features(
        *(f.filter(pl.col("source_trade_date") < cutoff) for f in (cash, qty, oi, pr)),
        days,
        **args,
    )
    for got, expected in zip(earlier, actual):
        assert got.filter(pl.col("date") <= cutoff).equals(
            expected.filter(pl.col("date") <= cutoff)
        )
    missing = cash.filter(pl.col("source_trade_date") != days[12])
    assert (
        renamed_activity_features(missing, qty, oi, pr, days, **args)[1][
            "avg_trade_size_20"
        ].null_count()
        == 7
    )


def test_oddlot_exact_lag_carries_only_pre_effect_history():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(14)]
    source = pl.DataFrame(
        {
            "source_trade_date": days[:-1],
            "available_date": days[1:],
            "isin": ["OLD"] * 7 + ["NEW"] * 6,
            "regular_volume_brl": [100.0] * 13,
            "odd_lot_volume_brl": [float(i) for i in range(13)],
        }
    )
    kwargs = dict(predecessor="OLD", successor="NEW", effective=days[7], known=days[9])
    got = renamed_oddlot_features(source, days, **kwargs)
    assert got["available_date"].min() == days[9]
    assert got["oddlot_volume_share_change_5_mask"].all()
    first = got.row(0, named=True)
    assert first["oddlot_volume_share_change_5"] == pytest.approx(8 / 108 - 3 / 103)
    conflict = pl.concat(
        [source, source.head(1).with_columns(pl.lit("NEW").alias("isin"))]
    )
    with pytest.raises(ValueError, match="overlaps"):
        renamed_oddlot_features(conflict, days, **kwargs)
