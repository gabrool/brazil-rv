from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
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
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.v2.feature_spec import (
    feature_specs,
    observation_age_sessions_into,
    transform_feature_panel_into,
)
from brazil_rv.v2.features import build_slow_features_into
from brazil_rv.v2.normalization import rank_gauss_panel_into
from brazil_rv.v2.sidecars import derive_known_archive_features
from brazil_rv.v2.store import close_memmap, peak_rss_bytes
from brazil_rv.v2.targets import build_economic_multi_day_targets_into

root = Path(sys.argv[1])
scale = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
archive_rows = {}
for archive in scale["archives"]:
    archive_rows.setdefault(archive["group"], []).append(int(archive["rows"]))
date_count = 4_348
name_count = 933
first_day = date(2010, 1, 1)
dates = [first_day + timedelta(days=index) for index in range(date_count)]
active = np.zeros((date_count, name_count), dtype=np.bool_)
active[:, :64] = True
widths = [
    len(INTRADAY_DAILY_FEATURES),
    *(len(SIDECAR_FEATURES[group]) for group in sorted(SIDECAR_FEATURES)),
]

# Exercise the production slow producer itself.  Every raw field is handed to
# the transform/age sink and released; there is no synthetic prebuilt 32-field
# cube standing in for this path.
market_paths = {name: root / f"slow_source_{name}.npy" for name in (
    "open", "high", "low", "close", "volume", "trades", "observed"
)}
market = {
    name: np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.bool_ if name == "observed" else np.float32,
        shape=(date_count, name_count),
    )
    for name, path in market_paths.items()
}
name_base = np.linspace(10.0, 30.0, name_count, dtype=np.float32)
for start in range(0, date_count, 64):
    stop = min(start + 64, date_count)
    trend = np.exp(
        np.arange(start, stop, dtype=np.float32)[:, None] * np.float32(1e-4)
    )
    close_block = trend * name_base[None, :]
    market["close"][start:stop] = close_block
    market["open"][start:stop] = close_block * np.float32(0.999)
    market["high"][start:stop] = close_block * np.float32(1.01)
    market["low"][start:stop] = close_block * np.float32(0.99)
    market["volume"][start:stop] = np.float32(2_000_000.0)
    market["trades"][start:stop] = np.float32(100.0)
    market["observed"][start:stop] = True
slow_shape = (date_count, name_count, len(SLOW_FEATURES))
slow_paths = {
    "values": root / "slow_stream_values.npy",
    "valid": root / "slow_stream_valid.npy",
    "age": root / "slow_stream_age.npy",
}
slow_values = np.lib.format.open_memmap(
    slow_paths["values"], mode="w+", dtype=np.float32, shape=slow_shape
)
slow_valid = np.lib.format.open_memmap(
    slow_paths["valid"], mode="w+", dtype=np.bool_, shape=slow_shape
)
slow_age = np.lib.format.open_memmap(
    slow_paths["age"], mode="w+", dtype=np.float32, shape=slow_shape
)
slow_specs = feature_specs("slow", SLOW_FEATURES, minimum_rank_names=20)
source_rows = np.arange(date_count, dtype=np.int64) - 1
decision_rows = np.arange(date_count, dtype=np.int64)

def consume_slow(index, raw_values, raw_valid):
    values_3d = np.asarray(raw_values)[..., None]
    valid_3d = np.asarray(raw_valid, dtype=np.bool_)[..., None]
    transform_feature_panel_into(
        values_3d,
        valid_3d,
        active,
        slow_specs[index:index + 1],
        slow_values[..., index:index + 1],
        slow_valid[..., index:index + 1],
        source_rows=source_rows,
        membership_rows=decision_rows,
        minimum_rank_names=20,
    )
    observation_age_sessions_into(
        valid_3d,
        active,
        slow_age[..., index:index + 1],
        source_rows=source_rows,
        decision_rows=decision_rows,
    )

build_slow_features_into(
    market["open"], market["high"], market["low"], market["close"],
    market["volume"], market["trades"], market["observed"], active, dates,
    raw_high=market["high"],
    raw_low=market["low"],
    raw_close=market["close"],
    price_observed=market["observed"],
    history_observed=market["observed"],
    activity_valid=market["observed"],
    consume=consume_slow,
    cluster_labels=np.full((date_count, name_count), -1, dtype=np.int16),
    ambiguous_action=np.zeros((date_count, name_count), dtype=np.bool_),
)
slow_valid_count = int(slow_valid.sum())
assert slow_valid_count > 0
for value in (*market.values(), slow_values, slow_valid, slow_age):
    close_memmap(value)
del market, slow_values, slow_valid, slow_age
for path in (*market_paths.values(), *slow_paths.values()):
    path.unlink()
gc.collect()

for family_index, width in enumerate(widths):
    shape = (date_count, name_count, width)
    raw_path = root / f"family_{family_index}_raw.npy"
    value_path = root / f"family_{family_index}_values.npy"
    valid_path = root / f"family_{family_index}_valid.npy"
    raw = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=shape)
    raw[:] = 0.0
    valid = np.broadcast_to(active[:, :, None], shape)
    output = np.lib.format.open_memmap(value_path, mode="w+", dtype=np.float32, shape=shape)
    output_valid = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.bool_, shape=shape)
    rank_gauss_panel_into(raw, valid, active, output, output_valid)
    close_memmap(output)
    close_memmap(output_valid)
    close_memmap(raw)
    del raw, valid, output, output_valid
    raw_path.unlink()
    value_path.unlink()
    valid_path.unlink()
    gc.collect()

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
assert "sessions_since_financial_filing" not in events.columns
del events
gc.collect()
for group, value_names in (
    ("fundamentals", ("total_liabilities_brl", "total_assets_brl")),
    ("rebalance", SIDECAR_FEATURES["rebalance"]),
):
    collapsed = intraday_archive(max(archive_rows[group]), value_names)
    collapsed = derive_known_archive_features(
        collapsed, dates, [str(index) for index in range(name_count)], group=group
    )
    if group == "rebalance":
        assert collapsed.is_empty()
    else:
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
    pl.lit(0.05).alias("loan_rate_annual_decimal"),
    pl.lit(True).alias("loan_rate_annual_decimal_mask"),
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
    pl.lit(2).alias("put_open_interest"),
    pl.lit(1).alias("call_open_interest"),
    pl.lit(True).alias("oi_snapshot_complete"),
)
options = derive_known_archive_features(
    options, dates, [str(index) for index in range(name_count)], group="options"
)
assert options.get_column("put_call_log_oi_ratio_mask").any()
del options
gc.collect()

target_shape = (date_count, name_count, 5)
target_paths = []
destinations = []
target_dtypes = (
    np.float32,
    np.bool_,
    np.float32,
    np.bool_,
    np.float32,
    np.bool_,
    np.float32,
    np.float32,
    np.bool_,
    np.float32,
    np.bool_,
    np.float32,
)
for index, dtype in enumerate(target_dtypes):
    path = root / f"target_{index}.npy"
    target_paths.append(path)
    shape = (date_count, 5) if index == 3 else target_shape
    destinations.append(
        np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    )
raw_close = np.broadcast_to(
    np.linspace(10.0, 20.0, name_count, dtype=np.float32),
    (date_count, name_count),
).copy()
close_observed = active.copy()
sigma = np.full_like(raw_close, 0.02)
actions = AlignedActionTerms(
    shares_per_prior_share=np.ones_like(raw_close),
    cash_per_prior_share=np.zeros_like(raw_close),
    session_resolved=np.ones_like(active),
    has_action=np.zeros_like(active),
)
build_economic_multi_day_targets_into(
    raw_close=raw_close,
    close_observed=close_observed,
    active=active,
    sigma_asof=sigma,
    actions=actions,
    primary=destinations[0],
    primary_valid=destinations[1],
    normalized_residual=destinations[2],
    normalized_cross_section_valid=destinations[3],
    shareholder_midrank=destinations[4],
    shareholder_valid=destinations[5],
    shareholder_simple_return=destinations[6],
    terminal_wealth=destinations[7],
    terminal_loss=destinations[8],
    price_midrank=destinations[9],
    price_valid=destinations[10],
    price_simple_return=destinations[11],
)
valid_target_count = int(destinations[1].sum())
assert valid_target_count > 0
for destination in destinations:
    close_memmap(destination)
for path in target_paths:
    path.unlink()
print(json.dumps({
    "peak_rss_bytes": peak_rss_bytes(),
    "family_widths": [len(SLOW_FEATURES), *widths],
    "slow_valid_count": slow_valid_count,
    "valid_target_count": valid_target_count,
    "archive_rows": archive_rows,
}))
"""
    archive_scale_path = (
        Path(__file__).parents[1] / "configs" / "v2" / "archive_scale.json"
    )
    scratch_candidates = [
        Path(value)
        for value in (
            os.environ.get("BRAZIL_RV_TEST_SCRATCH"),
            tmp_path.anchor,
            *(f"{letter}:\\" for letter in "DEFGHIJKLMNOPQRSTUVWXYZ"),
        )
        if value
    ]
    scratch_parent = next(
        (
            candidate
            for candidate in scratch_candidates
            if candidate.is_dir() and shutil.disk_usage(candidate).free >= 2 * 1024**3
        ),
        None,
    )
    assert scratch_parent is not None, (
        "production-axis memory acceptance needs a scratch volume with at least "
        "2 GiB free; set BRAZIL_RV_TEST_SCRATCH"
    )
    with tempfile.TemporaryDirectory(
        prefix="brazil-rv-v2-memory-", dir=scratch_parent
    ) as scratch_root:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                scratch_root,
                str(archive_scale_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "POLARS_MAX_THREADS": "2",
            },
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["peak_rss_bytes"] < 8 * 1024**3
    assert len(result["family_widths"]) >= 8
    assert result["slow_valid_count"] > 0
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
        measured_rows[group] >= minimum for group, minimum in measured_minimums.items()
    )


def test_native_production_axis_collation_and_forward_stay_below_eight_gib() -> None:
    """Exercise native collation/forward on the real name/lookback/feature axes."""

    script = r"""
import json

import numpy as np
import torch

from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.contract import INTRADAY_DAILY_FEATURES, SIDECAR_FEATURES, SLOW_FEATURES
from brazil_rv.v2.data import collate_v2_daily
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.store import peak_rss_bytes
from brazil_rv.v2.train import _model_forward, _to_device

torch.set_num_threads(2)
name_count = 933
lookback = 60
slow_count = len(SLOW_FEATURES) + sum(len(names) for names in SIDECAR_FEATURES.values())
current_count = len(INTRADAY_DAILY_FEATURES)
fast_count = 128
patch_count = 69

def sample(day):
    slow = np.zeros((name_count, lookback, slow_count), dtype=np.float32)
    slow_mask = np.ones(slow.shape, dtype=np.bool_)
    current = np.zeros((name_count, current_count), dtype=np.float32)
    return {
        "date_index": np.int64(day),
        "slow_features": slow,
        "slow_feature_mask": slow_mask,
        "slow_history_mask": np.ones((name_count, lookback), dtype=np.bool_),
        "slow_feature_age_sessions": np.zeros(slow.shape, dtype=np.float32),
        "active_mask": np.ones(name_count, dtype=np.bool_),
        "current_features": current,
        "current_feature_mask": np.ones(current.shape, dtype=np.bool_),
        "current_feature_age_sessions": np.zeros(current.shape, dtype=np.float32),
        "fast_patch_values": np.zeros((fast_count, patch_count, 7), dtype=np.float32),
        "fast_patch_valid": np.ones((fast_count, patch_count, 7), dtype=np.bool_),
        "fast_patch_mask": np.ones((fast_count, patch_count), dtype=np.bool_),
        "fast_name_index": np.arange(fast_count, dtype=np.int64),
        "fast_present": np.arange(name_count) < fast_count,
        "fast_state_position": np.full(fast_count, patch_count, dtype=np.int64),
        "v1_equity_slow": np.zeros((fast_count, 32), dtype=np.float32),
        "targets": np.zeros((name_count, 5), dtype=np.float32),
        "target_mask": np.ones((name_count, 5), dtype=np.bool_),
        "to_close_target": np.zeros(name_count, dtype=np.float32),
        "to_close_mask": np.zeros(name_count, dtype=np.bool_),
    }

# Two dates are the native date-pair unit used by the training sampler.  The
# separate SAM regression covers equivalence across accumulated pairs.
batch = collate_v2_daily((sample(0), sample(1)))
device_batch = _to_device(batch, torch.device("cpu"))
model = DailyMultiHorizonModel(
    ModelConfig(
        slow_feature_count=slow_count,
        current_feature_count=current_count,
        slow_lookback=lookback,
        compile_forward=False,
    )
).train()
with torch.no_grad():
    predictions = _model_forward(model, device_batch)
assert predictions.shape == (2, name_count, 6)
assert torch.isfinite(predictions).all()
print(json.dumps({
    "peak_rss_bytes": peak_rss_bytes(),
    "batch_shape": list(predictions.shape),
    "slow_feature_count": slow_count,
    "fast_name_count": fast_count,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["peak_rss_bytes"] < 8 * 1024**3
    assert result["batch_shape"] == [2, 933, 6]
    assert result["slow_feature_count"] > len(result["batch_shape"])
    assert result["fast_name_count"] > 0
