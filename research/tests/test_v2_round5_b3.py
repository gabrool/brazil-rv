from datetime import date

import polars as pl

from brazil_rv.v2.round5_b3 import RATE_FIELDS, stitch_registered_rates


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
