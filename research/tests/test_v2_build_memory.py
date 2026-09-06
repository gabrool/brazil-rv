from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_all_daily_families_fit_real_axes_below_eight_gib(tmp_path) -> None:
    """Guard the production 4,348 x 933 family-at-a-time RSS contract."""

    script = r"""
import gc
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES, SIDECAR_FEATURES, SLOW_FEATURES
from brazil_rv.v2.normalization import rank_gauss_panel_into
from brazil_rv.v2.sidecars import derive_known_archive_features
from brazil_rv.v2.store import close_memmap, peak_rss_bytes
from brazil_rv.v2.targets import build_multi_day_targets_into

root = Path(sys.argv[1])
scale = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
archive_rows = {}
for archive in scale["archives"]:
    archive_rows.setdefault(archive["group"], []).append(int(archive["rows"]))
date_count = 4_348
name_count = 933
active = np.zeros((date_count, name_count), dtype=np.bool_)
active[:, :64] = True
widths = [
    len(SLOW_FEATURES),
    len(INTRADAY_DAILY_FEATURES),
    *(len(SIDECAR_FEATURES[group]) for group in sorted(SIDECAR_FEATURES)),
]
for family_index, width in enumerate(widths):
    shape = (date_count, name_count, width)
    raw = np.zeros(shape, dtype=np.float32)
    valid = np.broadcast_to(active[:, :, None], shape).copy()
    value_path = root / f"family_{family_index}_values.npy"
    valid_path = root / f"family_{family_index}_valid.npy"
    output = np.lib.format.open_memmap(value_path, mode="w+", dtype=np.float32, shape=shape)
    output_valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=shape)
    rank_gauss_panel_into(raw, valid, active, output, output_valid)
    close_memmap(output)
    close_memmap(output_valid)
    del raw, valid, output, output_valid
    value_path.unlink()
    valid_path.unlink()
    gc.collect()

first_day = date(2010, 1, 1)
dates = [first_day + timedelta(days=index) for index in range(date_count)]

def intraday_archive(row_count, value_names):
    physical = pl.DataFrame(
        {"__row": np.arange(row_count, dtype=np.int32)}
    ).with_columns((pl.col("__row") // 55).alias("__daily"))
    physical = physical.with_columns(
        (
            pl.lit(first_day)
            + pl.duration(
                days=(pl.col("__daily") // name_count) % date_count
            )
        ).alias("available_date"),
        (pl.col("__daily") % name_count).cast(pl.String).alias("isin"),
        (pl.col("__row") % 55).alias("decision_idx"),
        *(
            ((pl.col("__row") + index) % 137 == 0)
            .cast(pl.Float32)
            .alias(value_name)
            for index, value_name in enumerate(value_names)
        ),
        *(pl.lit(True).alias(f"{value_name}_mask") for value_name in value_names),
    ).drop("__daily")
    collapsed = physical.group_by("available_date", "isin").agg(
        pl.col("decision_idx").max(),
        *(
            pl.col(value_name).sort_by("decision_idx").last()
            for value_name in value_names
        ),
        *(
            pl.col(f"{value_name}_mask").sort_by("decision_idx").last()
            for value_name in value_names
        ),
    )
    del physical
    gc.collect()
    return collapsed

# Exercise the two 55-snapshot physical archives at their measured heights.
# They are reduced before collection in production, but their projection and
# reduction still belong to the peak-memory contract.
events = intraday_archive(
    max(archive_rows["events"]), ("event_itr_dfp_recent_5s",)
)
events = derive_known_archive_features(
    events, dates, [str(index) for index in range(name_count)], group="events"
)
assert "sessions_since_earnings" in events.columns
del events
gc.collect()
for group, value_names in (
    ("fundamentals", ("fund_leverage",)),
    ("rebalance", SIDECAR_FEATURES["rebalance"]),
):
    collapsed = intraday_archive(max(archive_rows[group]), value_names)
    assert collapsed.height > 0
    del collapsed
    gc.collect()

def daily_coordinates(row_count, start_index=0):
    return pl.DataFrame(
        {"__row": np.arange(row_count, dtype=np.int32)}
    ).with_columns(
        (
            pl.lit(first_day)
            + pl.duration(
                days=start_index
                + (pl.col("__row") // name_count)
            )
        ).alias("source_trade_date"),
        (
            pl.lit(first_day)
            + pl.duration(
                days=start_index
                + (pl.col("__row") // name_count)
                + 1
            )
        ).alias("available_date"),
        (pl.col("__row") % name_count).cast(pl.String).alias("isin"),
    ).drop("__row")

oddlot = daily_coordinates(max(archive_rows["oddlot"])).with_columns(
    pl.lit(90.0).alias("regular_volume_brl"),
    pl.lit(10.0).alias("odd_lot_volume_brl"),
)
oddlot = derive_known_archive_features(
    oddlot, dates, [str(index) for index in range(name_count)], group="oddlot"
)
assert oddlot.get_column("oddlot_volume_share_mask").any()
del oddlot
gc.collect()

balance_rows, rate_rows = archive_rows["lending"]
daily_volume = np.ones((date_count, name_count), dtype=np.float64)
balance = daily_coordinates(balance_rows, start_index=19).rename(
    {"source_trade_date": "source_position_date"}
).with_columns(pl.lit(100.0).alias("lending_balance_brl"))
balance = derive_known_archive_features(
    balance,
    dates,
    [str(index) for index in range(name_count)],
    group="lending",
    daily_volume_brl=daily_volume,
)
assert balance.get_column("loan_balance_to_volume_20_mask").any()
del balance
gc.collect()
rates = daily_coordinates(rate_rows, start_index=5).with_columns(
    pl.lit(0.05).alias("lending_taker_fee_level_log_tanh"),
    pl.lit(True).alias("lending_taker_fee_level_log_tanh_mask"),
)
rates = derive_known_archive_features(
    rates,
    dates,
    [str(index) for index in range(name_count)],
    group="lending",
    daily_volume_brl=daily_volume,
)
assert rates.get_column("loan_rate_mask").any()
del rates, daily_volume
gc.collect()

options = daily_coordinates(max(archive_rows["options"])).with_columns(
    pl.lit(0.05).alias("options_put_call_oi_log_ratio_tanh"),
    pl.lit(True).alias("options_put_call_oi_log_ratio_tanh_mask"),
    pl.lit(-0.05).alias("options_oi_change_to_stock_adv20_tanh"),
    pl.lit(True).alias("options_oi_change_to_stock_adv20_tanh_mask"),
    pl.lit(0.05).alias("options_put_skew_tanh"),
    pl.lit(True).alias("options_put_skew_tanh_mask"),
)
options = derive_known_archive_features(
    options, dates, [str(index) for index in range(name_count)], group="options"
)
assert options.get_column("put_call_oi_ratio_mask").any()
del options
gc.collect()

target_shape = (date_count, name_count, 5)
target_paths = []
destinations = []
for index, dtype in enumerate((np.float32, np.bool_, np.float32, np.float32, np.bool_, np.float32)):
    path = root / f"target_{index}.npy"
    target_paths.append(path)
    destinations.append(np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=target_shape))
neutralized_return = np.broadcast_to(
    np.linspace(-0.01, 0.01, name_count, dtype=np.float32),
    (date_count, name_count),
).copy()
return_valid = active.copy()
sigma = np.ones_like(neutralized_return)
build_multi_day_targets_into(
    neutralized_log_return=neutralized_return,
    neutralized_log_return_valid=return_valid,
    active=active,
    yang_zhang_sigma_20=sigma,
    primary=destinations[0],
    primary_valid=destinations[1],
    normalized_residual=destinations[2],
    raw_midrank=destinations[3],
    raw_valid=destinations[4],
    raw_log_return=destinations[5],
)
valid_target_count = int(destinations[1].sum())
assert valid_target_count > 0
for destination in destinations:
    close_memmap(destination)
for path in target_paths:
    path.unlink()
print(json.dumps({
    "peak_rss_bytes": peak_rss_bytes(),
    "family_widths": widths,
    "valid_target_count": valid_target_count,
    "archive_rows": archive_rows,
}))
"""
    archive_scale_path = (
        Path(__file__).parents[1] / "configs" / "v2" / "archive_scale.json"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(archive_scale_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["peak_rss_bytes"] < 8 * 1024**3
    assert len(result["family_widths"]) >= 8
    assert result["valid_target_count"] > 0
    assert set(result["archive_rows"]) == {
        "events",
        "fundamentals",
        "lending",
        "oddlot",
        "options",
        "rebalance",
    }
    measured_minimums = {
        "events": 6_072_440,
        "fundamentals": 5_212_350,
        "lending": 94_349,
        "oddlot": 258_845,
        "options": 95_045,
        "rebalance": 2_526_262,
    }
    measured_rows = {
        group: sum(counts) for group, counts in result["archive_rows"].items()
    }
    assert all(
        measured_rows[group] >= minimum
        for group, minimum in measured_minimums.items()
    )
