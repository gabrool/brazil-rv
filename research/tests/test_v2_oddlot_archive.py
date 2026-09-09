from datetime import date

from brazil_rv.preprocessing.odd_lot_activity import Activity
from brazil_rv.v2.oddlot_archive import activity_rows


def test_oddlot_source_identity_observed_zero_and_year_boundary_availability():
    days = (date(2023, 12, 28), date(2024, 1, 2), date(2024, 1, 3))
    activity = {
        (days[0], "old-isin", "regular"): Activity(volume_cents=10000),
        (days[0], "old-isin", "odd_lot"): Activity(volume_cents=500),
        (days[1], "new-isin", "regular"): Activity(volume_cents=20000),
        (days[1], "orphan", "odd_lot"): Activity(volume_cents=500),
        (days[-1], "new-isin", "regular"): Activity(volume_cents=30000),
    }
    rows = activity_rows(activity, days).to_dicts()
    assert len(rows) == 2
    assert rows[0] == {
        "source_trade_date": days[0],
        "available_date": days[1],
        "isin": "old-isin",
        "regular_volume_brl": 100.0,
        "odd_lot_volume_brl": 5.0,
    }
    assert rows[1]["isin"] == "new-isin"
    assert rows[1]["available_date"] == days[2]
    assert rows[1]["odd_lot_volume_brl"] == 0.0
