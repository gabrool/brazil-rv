from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.modeling.contract import (
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.modeling.feature_variant import (
    load_variant_manifest,
    open_variant_arrays,
    variant_parent,
)

from .contract import (
    MIN_ACTIVE_EQUITIES,
    PRICE_FEATURE_CLIP,
    REALIZED_VOL_LOG_CLIP,
    REALIZED_VOL_MIN_FRACTION,
)
from .intraday_normalization import (
    ARMS,
    DECISION_FEATURE_MINUTES,
    PROFILE_BIN_MINUTES,
    VISIBLE_EQUITY_MINUTES,
    load_equity_tod_profile,
    sha256_file,
    write_canonical_json,
)

DIAGNOSTIC_SCHEMA = "EQUITY_INTRADAY_HETEROSKEDASTICITY_V1"
OPENING_BIN = 0
MIDDAY_BIN = 4
CHANNELS = {
    0: "open_move_normalized",
    1: "high_move_normalized",
    2: "low_move_normalized",
    3: "close_move_normalized",
    6: "return_since_open_normalized",
    7: "return_15m_normalized",
    8: "return_30m_normalized",
    9: "return_60m_normalized",
    10: "realized_volatility_15m_log_ratio",
    11: "realized_volatility_30m_log_ratio",
    12: "realized_volatility_60m_log_ratio",
    16: "market_median_return_15m",
    17: "market_median_return_60m",
    20: "market_dispersion_return_15m",
    21: "market_dispersion_return_60m",
    25: "market_rank_realized_volatility_30m",
}
RETURN_CHANNEL_WINDOWS = {7: 15, 8: 30, 9: 60}
REALIZED_VOL_CHANNEL_WINDOWS = {10: 15, 11: 30, 12: 60}


def _finite_statistics(
    values: np.ndarray,
    *,
    possible_count: int,
    lower_clip: float | None,
    upper_clip: float | None,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2 or possible_count <= 0:
        raise ValueError("Diagnostic group has insufficient valid observations")
    median = float(np.median(values))
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= 0.0:
        raise ValueError("Diagnostic group has zero dispersion")
    quantiles = np.quantile(values, (0.01, 0.05, 0.95, 0.99))
    return {
        "valid_count": int(values.size),
        "possible_count": int(possible_count),
        "mean": float(values.mean()),
        "standard_deviation": standard_deviation,
        "signed_log_standard_deviation": float(math.log(standard_deviation)),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "zero_fraction": float(np.mean(values == 0.0)),
        "lower_clipping_fraction": (
            0.0 if lower_clip is None else float(np.mean(values <= lower_clip + 1e-7))
        ),
        "upper_clipping_fraction": (
            0.0 if upper_clip is None else float(np.mean(values >= upper_clip - 1e-7))
        ),
        "observed_fraction": float(values.size / possible_count),
    }


def _clip_bounds(channel: int) -> tuple[float | None, float | None]:
    if channel in REALIZED_VOL_CHANNEL_WINDOWS:
        return -REALIZED_VOL_LOG_CLIP, REALIZED_VOL_LOG_CLIP
    if channel == 25:
        return None, None
    return -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP


def _window_valid(observed: np.ndarray, window: int) -> np.ndarray:
    output = np.zeros_like(observed)
    output[..., window:] = observed[..., window:] & observed[..., :-window]
    return output


def _realized_vol_valid(observed: np.ndarray, window: int) -> np.ndarray:
    adjacent = observed[..., 1:] & observed[..., :-1]
    prefix = np.concatenate(
        (
            np.zeros((*adjacent.shape[:-1], 1), dtype=np.int16),
            np.cumsum(adjacent, axis=-1, dtype=np.int16),
        ),
        axis=-1,
    )
    counts = prefix[..., window:] - prefix[..., :-window]
    output = np.zeros_like(observed)
    output[..., window:] = observed[..., window:] & (
        counts >= math.ceil(REALIZED_VOL_MIN_FRACTION * window)
    )
    return output


def _channel_valid(
    channel: int, observed: np.ndarray, active: np.ndarray
) -> np.ndarray:
    active_minutes = active[..., None]
    if channel in (0, 1, 2, 3, 6):
        return observed & active_minutes
    if channel in RETURN_CHANNEL_WINDOWS:
        return _window_valid(observed, RETURN_CHANNEL_WINDOWS[channel]) & active_minutes
    if channel in REALIZED_VOL_CHANNEL_WINDOWS:
        return (
            _realized_vol_valid(observed, REALIZED_VOL_CHANNEL_WINDOWS[channel])
            & active_minutes
        )
    source_channel = {16: 7, 20: 7, 17: 9, 21: 9}.get(channel)
    if source_channel is not None:
        source = _window_valid(observed, RETURN_CHANNEL_WINDOWS[source_channel])
        enough = (source & active_minutes).sum(axis=1) >= MIN_ACTIVE_EQUITIES
        return active_minutes & enough[:, None, :]
    if channel == 25:
        source = _realized_vol_valid(observed, 30) & active_minutes
        enough = source.sum(axis=1) >= MIN_ACTIVE_EQUITIES
        return source & enough[:, None, :]
    raise ValueError(f"No validity policy for channel {channel}")


def _profile_aggregate(
    profile: pl.DataFrame,
    date_indices: np.ndarray,
    bin_idx: int,
    gamma: float,
) -> dict[str, float]:
    rows = profile.filter(
        pl.col("date_idx").is_in(date_indices.tolist()) & (pl.col("bin_idx") == bin_idx)
    )
    if rows.is_empty():
        raise ValueError("Profile metadata is missing a diagnostic bin")
    q = rows.get_column("relative_variance").to_numpy()
    return {
        "mean_seasonal_relative_variance": float(q.mean()),
        "mean_applied_standard_deviation_multiplier": float(
            np.mean(q ** (0.5 * gamma))
        ),
        "mean_effective_historical_profile_days": float(
            rows.get_column("effective_historical_profile_days").mean()
        ),
        "mean_shrinkage_weight": float(rows.get_column("shrinkage_weight").mean()),
    }


def _selected_values(
    values: np.ndarray,
    valid: np.ndarray,
    active: np.ndarray,
    minutes: np.ndarray,
) -> tuple[np.ndarray, int]:
    selected_values = values[..., minutes]
    selected_valid = valid[..., minutes]
    possible = int(active.sum()) * int(minutes.size)
    return selected_values[selected_valid], possible


def _append_group(
    rows: list[dict[str, object]],
    prefix: dict[str, object],
    values: np.ndarray,
    valid: np.ndarray,
    active: np.ndarray,
    minutes: np.ndarray,
    channel: int,
    *,
    include_empty: bool = False,
) -> None:
    selected, possible = _selected_values(values, valid, active, minutes)
    if selected.size < 2 or possible == 0:
        if include_empty and possible > 0:
            rows.append(
                {
                    **prefix,
                    "valid_count": int(selected.size),
                    "possible_count": possible,
                    "mean": None,
                    "standard_deviation": None,
                    "signed_log_standard_deviation": None,
                    "median": None,
                    "mad": None,
                    "p01": None,
                    "p05": None,
                    "p95": None,
                    "p99": None,
                    "zero_fraction": None,
                    "lower_clipping_fraction": None,
                    "upper_clipping_fraction": None,
                    "observed_fraction": float(selected.size / possible),
                }
            )
        return
    lower, upper = _clip_bounds(channel)
    rows.append(
        {
            **prefix,
            **_finite_statistics(
                selected,
                possible_count=possible,
                lower_clip=lower,
                upper_clip=upper,
            ),
        }
    )


def _open_features(store: Path):
    manifest = load_variant_manifest(store)
    if manifest is None:
        return np.load(store / "equity_features.npy", mmap_mode="r", allow_pickle=False)
    parent = variant_parent(store, manifest)
    return open_variant_arrays(store, parent, manifest, ("equity_features.npy",))[
        "equity_features.npy"
    ]


def _effect_size_rows(
    by_bin: pl.DataFrame,
    by_year: pl.DataFrame,
    by_security: pl.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    legacy = by_bin.filter(pl.col("arm") == "legacy_daily_vol")
    for (arm, split, channel), group in by_bin.group_by(
        "arm", "split", "channel", maintain_order=True
    ):
        group = group.filter(pl.col("standard_deviation").is_not_null())
        group = group.sort("bin_idx")
        std = group["standard_deviation"].to_numpy()
        weights = group["valid_count"].to_numpy().astype(np.float64)
        logs = np.log(std)
        weighted_mean = float(np.average(std, weights=weights))
        weighted_variance = float(
            np.average((std - weighted_mean) ** 2, weights=weights)
        )
        legacy_group = legacy.filter(
            (pl.col("split") == split)
            & (pl.col("channel") == channel)
            & pl.col("standard_deviation").is_not_null()
        ).sort("bin_idx")
        legacy_clipping = (
            legacy_group["lower_clipping_fraction"]
            + legacy_group["upper_clipping_fraction"]
        ).to_numpy()
        clipping = (
            group["lower_clipping_fraction"] + group["upper_clipping_fraction"]
        ).to_numpy()
        year_overall = by_year.filter(
            (pl.col("arm") == arm)
            & (pl.col("split") == split)
            & (pl.col("channel") == channel)
            & (pl.col("bin_idx") == -1)
        )["standard_deviation"].to_numpy()
        security = by_security.filter(
            (pl.col("arm") == arm)
            & (pl.col("split") == split)
            & (pl.col("channel") == channel)
            & (pl.col("bin_idx") == -1)
        ).sort("standard_deviation")
        security_std = security["standard_deviation"].to_numpy()
        security_quantiles = np.quantile(security_std, (0.1, 0.5, 0.9))
        opening = group.filter(pl.col("bin_idx") == OPENING_BIN)
        midday = group.filter(pl.col("bin_idx") == MIDDAY_BIN)
        opening_to_midday = (
            None
            if opening.is_empty() or midday.is_empty()
            else float(
                opening["standard_deviation"][0] / midday["standard_deviation"][0]
            )
        )
        rows.append(
            {
                "arm": arm,
                "split": split,
                "channel": channel,
                "max_to_min_bin_std_ratio": float(std.max() / std.min()),
                "max_to_min_bin_variance_ratio": float((std.max() / std.min()) ** 2),
                "weighted_log_std_rmse": float(
                    np.sqrt(np.average(logs**2, weights=weights))
                ),
                "weighted_bin_std_cv": float(
                    math.sqrt(weighted_variance) / weighted_mean
                ),
                "maximum_absolute_log_std": float(np.max(np.abs(logs))),
                "fraction_of_bins_with_std_outside_0_90_1_10": float(
                    np.mean((std < 0.9) | (std > 1.1))
                ),
                "opening_to_midday_std_ratio": opening_to_midday,
                "year_max_to_min_std_ratio": float(
                    year_overall.max() / year_overall.min()
                ),
                "clipping_fraction_delta_vs_legacy": float(
                    np.average(clipping - legacy_clipping, weights=weights)
                ),
                "valid_fraction_delta_vs_legacy": float(
                    np.average(
                        group["observed_fraction"].to_numpy()
                        - legacy_group["observed_fraction"].to_numpy(),
                        weights=weights,
                    )
                ),
                "security_std_p10": float(security_quantiles[0]),
                "security_std_median": float(security_quantiles[1]),
                "security_std_p90": float(security_quantiles[2]),
                "security_std_p90_minus_p10": float(
                    security_quantiles[2] - security_quantiles[0]
                ),
                "most_under_dispersed_security": security["security_id"][0],
                "most_under_dispersed_std": float(security_std[0]),
                "most_over_dispersed_security": security["security_id"][-1],
                "most_over_dispersed_std": float(security_std[-1]),
            }
        )
    return rows


def _cross_security_rows(by_security: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (arm, split, channel, bin_idx), group in by_security.group_by(
        "arm", "split", "channel", "bin_idx", maintain_order=True
    ):
        if group.height < 10:
            continue
        group = group.sort("standard_deviation")
        values = group["standard_deviation"].to_numpy()
        quantiles = np.quantile(values, (0.1, 0.5, 0.9))
        rows.append(
            {
                "arm": arm,
                "split": split,
                "channel": channel,
                "bin_idx": bin_idx,
                "security_count": group.height,
                "security_std_p10": float(quantiles[0]),
                "security_std_median": float(quantiles[1]),
                "security_std_p90": float(quantiles[2]),
                "security_std_p90_minus_p10": float(quantiles[2] - quantiles[0]),
                "most_under_dispersed_security": group["security_id"][0],
                "most_under_dispersed_std": float(values[0]),
                "most_over_dispersed_security": group["security_id"][-1],
                "most_over_dispersed_std": float(values[-1]),
            }
        )
    return rows


def run_heteroskedasticity_diagnostics(
    stores: dict[str, Path], profile_dir: Path, output_dir: Path
) -> Path:
    """Compute exact train/validation feature-distribution diagnostics."""
    if tuple(stores) != tuple(ARMS):
        raise ValueError("Diagnostics require the exact three-arm ordering")
    profile_manifest, _ = load_equity_tod_profile(profile_dir)
    profile = pl.read_csv(profile_dir / "equity_tod_profile.csv", try_parse_dates=True)
    parent = stores["legacy_daily_vol"]
    date_index = pl.read_parquet(parent / "date_index.parquet").sort("date_idx")
    equity_index = pl.read_parquet(parent / "equity_index.parquet").sort("equity_slot")
    dates = date_index["trade_date"].to_list()
    security_ids = equity_index["security_id"].to_list()
    membership = np.load(
        parent / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    readiness = np.load(
        parent / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    parent_features = np.load(
        parent / "equity_features.npy", mmap_mode="r", allow_pickle=False
    )
    splits = {
        "train": np.asarray(
            [i for i, value in enumerate(dates) if TRAIN_START <= value <= TRAIN_END],
            dtype=np.int64,
        ),
        "validation": np.asarray(
            [
                i
                for i, value in enumerate(dates)
                if VALIDATION_START <= value <= VALIDATION_END
            ],
            dtype=np.int64,
        ),
    }
    by_bin: list[dict[str, object]] = []
    by_year: list[dict[str, object]] = []
    by_security: list[dict[str, object]] = []
    return_decision: list[dict[str, object]] = []
    visible_minutes = np.arange(VISIBLE_EQUITY_MINUTES, dtype=np.int64)

    for arm, store in stores.items():
        gamma = ARMS[arm]
        features = _open_features(store)
        for split, date_indices in splits.items():
            active = np.asarray(
                membership[date_indices] & readiness[date_indices], dtype=bool
            )
            observed = np.asarray(
                parent_features[date_indices, :, :VISIBLE_EQUITY_MINUTES, 5],
                dtype=bool,
            )
            for channel, channel_name in CHANNELS.items():
                values = np.asarray(
                    features[date_indices, :, :VISIBLE_EQUITY_MINUTES, channel],
                    dtype=np.float32,
                )
                valid = _channel_valid(channel, observed, active)
                for bin_idx, start in enumerate(
                    range(0, VISIBLE_EQUITY_MINUTES, PROFILE_BIN_MINUTES)
                ):
                    minutes = visible_minutes[
                        start : min(start + PROFILE_BIN_MINUTES, VISIBLE_EQUITY_MINUTES)
                    ]
                    prefix = {
                        "arm": arm,
                        "gamma": gamma,
                        "split": split,
                        "channel": channel_name,
                        "channel_index": channel,
                        "bin_idx": bin_idx,
                        "session_minute_start": int(start),
                        "session_minute_end_exclusive": int(minutes[-1] + 1),
                        **_profile_aggregate(profile, date_indices, bin_idx, gamma),
                    }
                    _append_group(
                        by_bin,
                        prefix,
                        values,
                        valid,
                        active,
                        minutes,
                        channel,
                        include_empty=True,
                    )
                    if channel in RETURN_CHANNEL_WINDOWS:
                        decision_minutes = np.asarray(
                            [
                                minute
                                for minute in DECISION_FEATURE_MINUTES
                                if start <= minute < start + PROFILE_BIN_MINUTES
                            ],
                            dtype=np.int64,
                        )
                        if decision_minutes.size:
                            _append_group(
                                return_decision,
                                {
                                    **prefix,
                                    "return_window_minutes": RETURN_CHANNEL_WINDOWS[
                                        channel
                                    ],
                                },
                                values,
                                valid,
                                active,
                                decision_minutes,
                                channel,
                                include_empty=True,
                            )

                years = sorted({dates[int(index)].year for index in date_indices})
                for year in years:
                    positions = np.flatnonzero(
                        np.asarray(
                            [dates[int(index)].year == year for index in date_indices]
                        )
                    )
                    year_active = active[positions]
                    year_values = values[positions]
                    year_valid = valid[positions]
                    _append_group(
                        by_year,
                        {
                            "arm": arm,
                            "gamma": gamma,
                            "split": split,
                            "year": year,
                            "channel": channel_name,
                            "channel_index": channel,
                            "bin_idx": -1,
                        },
                        year_values,
                        year_valid,
                        year_active,
                        visible_minutes,
                        channel,
                    )
                    for bin_idx, start in enumerate(
                        range(0, VISIBLE_EQUITY_MINUTES, PROFILE_BIN_MINUTES)
                    ):
                        minutes = visible_minutes[
                            start : min(
                                start + PROFILE_BIN_MINUTES,
                                VISIBLE_EQUITY_MINUTES,
                            )
                        ]
                        _append_group(
                            by_year,
                            {
                                "arm": arm,
                                "gamma": gamma,
                                "split": split,
                                "year": year,
                                "channel": channel_name,
                                "channel_index": channel,
                                "bin_idx": bin_idx,
                            },
                            year_values,
                            year_valid,
                            year_active,
                            minutes,
                            channel,
                        )

                for equity_slot, security_id in enumerate(security_ids):
                    security_values = values[:, equity_slot : equity_slot + 1]
                    security_valid = valid[:, equity_slot : equity_slot + 1]
                    security_active = active[:, equity_slot : equity_slot + 1]
                    before = len(by_security)
                    _append_group(
                        by_security,
                        {
                            "arm": arm,
                            "gamma": gamma,
                            "split": split,
                            "channel": channel_name,
                            "channel_index": channel,
                            "security_id": security_id,
                            "equity_slot": equity_slot,
                            "bin_idx": -1,
                        },
                        security_values,
                        security_valid,
                        security_active,
                        visible_minutes,
                        channel,
                    )
                    if len(by_security) == before:
                        continue
                    for bin_idx, start in enumerate(
                        range(0, VISIBLE_EQUITY_MINUTES, PROFILE_BIN_MINUTES)
                    ):
                        minutes = visible_minutes[
                            start : min(
                                start + PROFILE_BIN_MINUTES,
                                VISIBLE_EQUITY_MINUTES,
                            )
                        ]
                        selected, possible = _selected_values(
                            security_values,
                            security_valid,
                            security_active,
                            minutes,
                        )
                        if selected.size < 100 or possible == 0:
                            continue
                        lower, upper = _clip_bounds(channel)
                        by_security.append(
                            {
                                "arm": arm,
                                "gamma": gamma,
                                "split": split,
                                "channel": channel_name,
                                "channel_index": channel,
                                "security_id": security_id,
                                "equity_slot": equity_slot,
                                "bin_idx": bin_idx,
                                **_finite_statistics(
                                    selected,
                                    possible_count=possible,
                                    lower_clip=lower,
                                    upper_clip=upper,
                                ),
                            }
                        )

    output_dir.mkdir(parents=True, exist_ok=False)
    by_bin_frame = pl.DataFrame(by_bin)
    by_year_frame = pl.DataFrame(by_year)
    by_security_frame = pl.DataFrame(by_security)
    return_frame = pl.DataFrame(return_decision)
    summary_rows = _effect_size_rows(by_bin_frame, by_year_frame, by_security_frame)
    summary_frame = pl.DataFrame(summary_rows)
    cross_security_frame = pl.DataFrame(_cross_security_rows(by_security_frame))
    outputs = {
        "heteroskedasticity_by_bin.csv": by_bin_frame,
        "heteroskedasticity_by_year.csv": by_year_frame,
        "heteroskedasticity_by_security.csv": by_security_frame,
        "return_windows_by_decision_bin.csv": return_frame,
        "heteroskedasticity_effect_sizes.csv": summary_frame,
        "cross_security_dispersion.csv": cross_security_frame,
    }
    for filename, frame in outputs.items():
        frame.write_csv(output_dir / filename)
    primary = summary_frame.filter(
        (pl.col("split") == "validation")
        & (pl.col("channel") == "close_move_normalized")
    ).sort("gamma")
    summary = {
        "schema": DIAGNOSTIC_SCHEMA,
        "test_accessed": False,
        "exact_statistics": True,
        "sampling": None,
        "arms": ARMS,
        "channels": CHANNELS,
        "profile_manifest_sha256": sha256_file(profile_dir / "equity_tod_profile.json"),
        "opening_bin": OPENING_BIN,
        "opening_bin_definition": "session minutes 0 through 29",
        "midday_bin": MIDDAY_BIN,
        "midday_bin_definition": "session minutes 120 through 149",
        "primary_endpoint": (
            "validation weighted_log_std_rmse for close_move_normalized"
        ),
        "primary_results": primary.select(
            "arm",
            "weighted_log_std_rmse",
            "max_to_min_bin_std_ratio",
            "opening_to_midday_std_ratio",
        ).to_dicts(),
        "row_counts": {filename: frame.height for filename, frame in outputs.items()},
    }
    write_canonical_json(output_dir / "heteroskedasticity_summary.json", summary)
    markdown = [
        "# Equity intraday heteroskedasticity diagnostics",
        "",
        "The primary endpoint is validation weighted log-standard-deviation RMSE for the normalized close move.",
        "",
        "| Arm | weighted log-std RMSE | max/min bin std | opening/midday std |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["primary_results"]:
        markdown.append(
            f"| {row['arm']} | {row['weighted_log_std_rmse']:.6f} | "
            f"{row['max_to_min_bin_std_ratio']:.6f} | "
            f"{row['opening_to_midday_std_ratio']:.6f} |"
        )
    (output_dir / "heteroskedasticity_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    output_names = (
        *outputs,
        "heteroskedasticity_summary.json",
        "heteroskedasticity_summary.md",
    )
    write_canonical_json(
        output_dir / "diagnostics_manifest.json",
        {
            "schema": DIAGNOSTIC_SCHEMA,
            "test_accessed": False,
            "profile": {
                "path": str(profile_dir.resolve()),
                "manifest_sha256": sha256_file(profile_dir / "equity_tod_profile.json"),
            },
            "stores": {arm: str(path.resolve()) for arm, path in stores.items()},
            "output_sha256": {
                name: sha256_file(output_dir / name) for name in output_names
            },
        },
    )
    validate_heteroskedasticity_diagnostics(output_dir)
    return output_dir


def _finite_columns(frame: pl.DataFrame, excluded: Iterable[str]) -> None:
    excluded = set(excluded)
    for name, dtype in frame.schema.items():
        if name in excluded or not dtype.is_numeric():
            continue
        if not np.isfinite(frame[name].drop_nulls().to_numpy()).all():
            raise ValueError(f"Non-finite diagnostic column: {name}")


def validate_heteroskedasticity_diagnostics(output_dir: Path) -> None:
    summary = json.loads(
        (output_dir / "heteroskedasticity_summary.json").read_text(encoding="utf-8")
    )
    if (
        summary.get("schema") != DIAGNOSTIC_SCHEMA
        or summary.get("test_accessed") is not False
    ):
        raise ValueError("Invalid heteroskedasticity summary contract")
    lineage = json.loads(
        (output_dir / "diagnostics_manifest.json").read_text(encoding="utf-8")
    )
    if (
        lineage.get("schema") != DIAGNOSTIC_SCHEMA
        or lineage.get("test_accessed") is not False
    ):
        raise ValueError("Invalid heteroskedasticity lineage contract")
    profile = lineage["profile"]
    if (
        sha256_file(Path(profile["path"]) / "equity_tod_profile.json")
        != profile["manifest_sha256"]
    ):
        raise ValueError("Heteroskedasticity profile hash mismatch")
    if set(lineage["stores"]) != set(ARMS):
        raise ValueError("Heteroskedasticity store lineage is incomplete")
    for name, expected_hash in lineage["output_sha256"].items():
        if sha256_file(output_dir / name) != expected_hash:
            raise ValueError(f"Heteroskedasticity output hash mismatch: {name}")
    required = {
        "heteroskedasticity_by_bin.csv": (len(ARMS) * 2 * len(CHANNELS), None),
        "heteroskedasticity_by_year.csv": (1, None),
        "heteroskedasticity_by_security.csv": (1, None),
        "return_windows_by_decision_bin.csv": (1, None),
        "heteroskedasticity_effect_sizes.csv": (
            len(ARMS) * 2 * len(CHANNELS),
            None,
        ),
        "cross_security_dispersion.csv": (len(ARMS) * 2 * len(CHANNELS), None),
    }
    for filename, (minimum, _) in required.items():
        frame = pl.read_csv(output_dir / filename)
        if frame.height < minimum or frame.height != summary["row_counts"][filename]:
            raise ValueError(f"Diagnostic row-count mismatch: {filename}")
        if set(frame["arm"]) != set(ARMS):
            raise ValueError(f"Diagnostic arm mismatch: {filename}")
        _finite_columns(frame, ("security_id", "channel", "arm", "split"))
    primary = summary["primary_results"]
    if [row["arm"] for row in primary] != list(ARMS):
        raise ValueError("Primary diagnostic results have the wrong arm order")
