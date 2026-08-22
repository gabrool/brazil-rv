from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.modeling.contract import (
    EQUITY_COUNT,
    EXPECTED_DECISIONS_PER_DATE,
    HORIZON_COUNT,
)
from brazil_rv.modeling.data import (
    EXTERNAL_SIDECAR_SCHEMA,
    feature_store_axis_identity,
    feature_store_identity,
)
from brazil_rv.modeling.metrics import primary_validation_score

from .contract import (
    DECISION_EQUITY_INDICES,
    EQUITY_SESSION_MINUTES,
    SLOW_CHANNELS,
)
from .io import (
    cotahist_files,
    dense_grid,
    load_assignments,
    load_market_dates_and_security_dates,
    load_source_file,
    prepare_session_bars,
    read_research_interval,
    resolve_inputs,
    validate_physical_source_identity,
    validate_source_date_isolation,
)

P1_FEATURES = (
    "hks_same_interval_return_lag1",
    "hks_same_interval_return_lag5",
    "hks_same_interval_return_lag20",
    "vwap_reversal_15m_cs",
    "vwap_reversal_volume_flip",
    "signed_semivariance_1d",
    "signed_semivariance_5d",
    "realized_skewness_1d",
    "realized_skewness_5d",
    "open_gap_high_attention_fade",
    "late_market_momentum_beta",
    "interval_volume_surprise_15m",
    "high_volume_so_far_surprise",
    "first30m_relative_volume",
    "edge_spread_60m_cs",
    "amihud_30m_cs",
    "vwap_reversal_x_edge",
    "vwap_reversal_x_amihud",
    "overnight_minus_intraday_20d_cs",
)
FEATURE_FAMILY = {
    **{name: "hks_seasonality" for name in P1_FEATURES[:3]},
    **{name: "reversal" for name in P1_FEATURES[3:5]},
    **{name: "signed_moments" for name in P1_FEATURES[5:9]},
    **{name: "time_boxed" for name in P1_FEATURES[9:11]},
    **{name: "volume" for name in P1_FEATURES[11:14]},
    **{name: "liquidity" for name in P1_FEATURES[14:18]},
    P1_FEATURES[18]: "tug_of_war",
}
FAMILY_TIER = {
    "hks_seasonality": 1,
    "reversal": 2,
    "signed_moments": 3,
    "time_boxed": 4,
    "volume": 5,
    "liquidity": 6,
    "tug_of_war": 7,
}
P1_LIBRARY_SCHEMA = "P1_CAUSAL_FEATURE_LIBRARY_V1"
F2_SELECTION_SCHEMA = "P1_F2_SELECTION_V1"
F2_FIT_END_DATE_IDX = 406
F2_TOP_K = 8
F2_MIN_HALF_ABS_IC = 0.001
F2_MAX_FAMILY_MEMBERS = 2
F2_MAX_PAIR_CORRELATION = 0.85


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def edge_spread(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> float:
    """EDGE effective-spread estimate from the authors' published pseudocode."""
    if not (len(open_) == len(high) == len(low) == len(close)) or len(open_) < 3:
        return float("nan")
    o, h, low_log, c = (np.log(values) for values in (open_, high, low, close))
    midpoint = (h + low_log) / 2.0
    h1 = np.roll(h, 1)
    l1 = np.roll(low_log, 1)
    c1 = np.roll(c, 1)
    m1 = np.roll(midpoint, 1)
    h1[0] = l1[0] = c1[0] = m1[0] = np.nan
    tau = np.where(
        np.isfinite(h + low_log + c1), (h != low_log) | (low_log != c1), np.nan
    )
    po1 = np.where(np.isfinite(tau + o + h), (tau == 1) & (o != h), np.nan)
    po2 = np.where(np.isfinite(tau + o + low_log), (tau == 1) & (o != low_log), np.nan)
    pc1 = np.where(np.isfinite(tau + c1 + h1), (tau == 1) & (c1 != h1), np.nan)
    pc2 = np.where(np.isfinite(tau + c1 + l1), (tau == 1) & (c1 != l1), np.nan)
    pt = float(np.nanmean(tau))
    po = float(np.nanmean(po1) + np.nanmean(po2))
    pc = float(np.nanmean(pc1) + np.nanmean(pc2))
    if np.nansum(tau) < 2 or po == 0.0 or pc == 0.0 or pt == 0.0:
        return float("nan")
    r1 = midpoint - o
    r2 = o - m1
    r3 = midpoint - c1
    r4 = c1 - m1
    r5 = o - c1
    d1 = r1 - np.nanmean(r1) / pt * tau
    d3 = r3 - np.nanmean(r3) / pt * tau
    d5 = r5 - np.nanmean(r5) / pt * tau
    x1 = -4.0 / po * d1 * r2 - 4.0 / pc * d3 * r4
    x2 = -4.0 / po * d1 * r5 - 4.0 / pc * d5 * r4
    e1, e2 = float(np.nanmean(x1)), float(np.nanmean(x2))
    v1 = float(np.nanmean(x1 * x1) - e1 * e1)
    v2 = float(np.nanmean(x2 * x2) - e2 * e2)
    total = v1 + v2
    square = (v2 * e1 + v1 * e2) / total if total > 0 else (e1 + e2) / 2.0
    return float(np.sqrt(abs(square)))


def _adjacent_returns(
    raw: np.ndarray, observed: np.ndarray, day: int, cutoff: int
) -> np.ndarray:
    valid = observed[day, 1:cutoff] & observed[day, : cutoff - 1]
    if not valid.any():
        return np.empty(0, dtype=np.float64)
    return np.log(raw[day, 1:cutoff, 3][valid] / raw[day, : cutoff - 1, 3][valid])


def _prior_interval_volumes(
    raw: np.ndarray,
    observed: np.ndarray,
    accepted: np.ndarray,
    day: int,
    start: int,
    end: int,
    *,
    cumulative: bool,
) -> np.ndarray:
    values = []
    for previous in range(max(0, day - 20), day):
        if not accepted[previous]:
            continue
        left = 0 if cumulative else start
        mask = observed[previous, left:end]
        if mask.sum() < max(1, int(0.8 * (end - left))):
            continue
        values.append(float(raw[previous, left:end, 4][mask].sum()))
    return np.asarray(values, dtype=np.float64)


def _robust_log_surprise(current: float, history: np.ndarray) -> float:
    if current <= 0 or history.size < 10 or np.any(history <= 0):
        return float("nan")
    logs = np.log(history)
    median = float(np.median(logs))
    scale = max(1.4826 * float(np.median(np.abs(logs - median))), 0.1)
    return float(np.clip((np.log(current) - median) / scale, -6.0, 6.0))


def _session_open(raw: np.ndarray, observed: np.ndarray, day: int) -> float | None:
    positions = np.flatnonzero(observed[day, :15])
    return float(raw[day, positions[0], 0]) if positions.size else None


def _tug_of_war(
    raw: np.ndarray, observed: np.ndarray, accepted: np.ndarray, day: int
) -> float:
    values = []
    for current in range(max(1, day - 20), day):
        if not (accepted[current - 1] and accepted[current]):
            continue
        opening = _session_open(raw, observed, current)
        if (
            opening is None
            or not observed[current - 1, -1]
            or not observed[current, -1]
        ):
            continue
        previous_close = raw[current - 1, -1, 3]
        close = raw[current, -1, 3]
        values.append(np.log(opening / previous_close) - np.log(close / opening))
    return float(np.mean(values)) if len(values) >= 15 else float("nan")


def build_security_library(
    raw: np.ndarray,
    observed: np.ndarray,
    accepted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one security's causal P1 library on date/decision axes."""
    date_count = raw.shape[0]
    values = np.zeros(
        (date_count, EXPECTED_DECISIONS_PER_DATE, len(P1_FEATURES)), dtype=np.float32
    )
    mask = np.zeros_like(values, dtype=bool)
    for day in range(date_count):
        if not accepted[day]:
            continue
        tug = _tug_of_war(raw, observed, accepted, day)
        for decision, cutoff in enumerate(DECISION_EQUITY_INDICES):
            if cutoff >= 30:
                start = cutoff - 30
                for feature, lag in enumerate((1, 5, 20)):
                    previous = day - lag
                    if (
                        previous >= 0
                        and accepted[previous]
                        and observed[previous, start]
                        and observed[previous, cutoff - 1]
                    ):
                        values[day, decision, feature] = np.log(
                            raw[previous, cutoff - 1, 3] / raw[previous, start, 0]
                        )
                        mask[day, decision, feature] = True

            start15 = max(0, cutoff - 15)
            recent = observed[day, start15:cutoff]
            reversal = float("nan")
            volume_z = float("nan")
            if recent.sum() >= 12 and observed[day, cutoff - 1]:
                prices = raw[day, start15:cutoff, 1:4].mean(axis=1)[recent]
                volumes = raw[day, start15:cutoff, 4][recent]
                vwap = float(np.average(prices, weights=volumes))
                reversal = np.log(vwap / raw[day, cutoff - 1, 3])
                values[day, decision, 3] = reversal
                mask[day, decision, 3] = True
                history = _prior_interval_volumes(
                    raw, observed, accepted, day, start15, cutoff, cumulative=False
                )
                volume_z = _robust_log_surprise(float(volumes.sum()), history)
                if np.isfinite(volume_z):
                    values[day, decision, 4] = reversal * -np.tanh(volume_z)
                    mask[day, decision, 4] = True
                    values[day, decision, 11] = volume_z
                    mask[day, decision, 11] = True

            current_returns = _adjacent_returns(raw, observed, day, cutoff)
            trailing_returns = [current_returns]
            for previous in range(max(0, day - 4), day):
                if accepted[previous]:
                    trailing_returns.insert(
                        0,
                        _adjacent_returns(
                            raw, observed, previous, EQUITY_SESSION_MINUTES
                        ),
                    )
            for feature_offset, returns, minimum in (
                (0, current_returns, 10),
                (1, np.concatenate(trailing_returns), 50),
            ):
                if returns.size < minimum:
                    continue
                variance = float(np.sum(returns**2))
                if variance <= 0:
                    continue
                positive = float(np.sum(returns[returns > 0] ** 2))
                negative = float(np.sum(returns[returns < 0] ** 2))
                values[day, decision, 5 + feature_offset] = (
                    positive - negative
                ) / variance
                mask[day, decision, 5 + feature_offset] = True
                skewness = (
                    np.sqrt(returns.size) * float(np.sum(returns**3)) / variance**1.5
                )
                values[day, decision, 7 + feature_offset] = np.tanh(skewness / 3.0)
                mask[day, decision, 7 + feature_offset] = True

            if day > 0 and cutoff <= 90 and accepted[day - 1]:
                opening = _session_open(raw, observed, day)
                if opening is not None and observed[day - 1, -1]:
                    open_end = min(cutoff, 30)
                    current_volume = float(
                        raw[day, :open_end, 4][observed[day, :open_end]].sum()
                    )
                    history = _prior_interval_volumes(
                        raw, observed, accepted, day, 0, open_end, cumulative=False
                    )
                    attention = _robust_log_surprise(current_volume, history)
                    if np.isfinite(attention) and attention > 0:
                        gap = np.log(opening / raw[day - 1, -1, 3])
                        values[day, decision, 9] = -gap * np.tanh(attention)
                        mask[day, decision, 9] = True

            current_cumulative = float(
                raw[day, :cutoff, 4][observed[day, :cutoff]].sum()
            )
            cumulative_history = _prior_interval_volumes(
                raw, observed, accepted, day, 0, cutoff, cumulative=True
            )
            cumulative_z = _robust_log_surprise(current_cumulative, cumulative_history)
            if np.isfinite(cumulative_z):
                values[day, decision, 12] = cumulative_z
                mask[day, decision, 12] = True
            if cutoff >= 30:
                first30 = float(raw[day, :30, 4][observed[day, :30]].sum())
                history30 = _prior_interval_volumes(
                    raw, observed, accepted, day, 0, 30, cumulative=False
                )
                first30_z = _robust_log_surprise(first30, history30)
                if np.isfinite(first30_z):
                    values[day, decision, 13] = first30_z
                    mask[day, decision, 13] = True

            start60 = max(0, cutoff - 60)
            observed60 = observed[day, start60:cutoff]
            if observed60.sum() >= 48:
                ohlc = raw[day, start60:cutoff, :4].copy()
                ohlc[~observed60] = np.nan
                spread = edge_spread(ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3])
                if np.isfinite(spread):
                    values[day, decision, 14] = spread
                    mask[day, decision, 14] = True
            start30 = max(0, cutoff - 31)
            adjacent30 = (
                observed[day, start30 + 1 : cutoff]
                & observed[day, start30 : cutoff - 1]
            )
            if adjacent30.sum() >= 24:
                returns30 = np.log(
                    raw[day, start30 + 1 : cutoff, 3][adjacent30]
                    / raw[day, start30 : cutoff - 1, 3][adjacent30]
                )
                dollar = (
                    raw[day, start30 + 1 : cutoff, 3][adjacent30]
                    * raw[day, start30 + 1 : cutoff, 4][adjacent30]
                )
                if np.all(dollar > 0):
                    amihud = np.log1p(1e12 * np.mean(np.abs(returns30) / dollar))
                    values[day, decision, 15] = amihud
                    mask[day, decision, 15] = True

            if np.isfinite(tug):
                values[day, decision, 18] = tug
                mask[day, decision, 18] = True
    return values, mask


def _cross_sectional_scale(
    values: np.ndarray,
    masks: np.ndarray,
    active: np.ndarray,
    feature: int,
) -> None:
    for day in range(values.shape[0]):
        for decision in range(values.shape[2]):
            valid = masks[day, :, decision, feature] & active[day]
            if valid.sum() < 30:
                masks[day, :, decision, feature] = False
                values[day, :, decision, feature] = 0.0
                continue
            sample = values[day, valid, decision, feature].astype(np.float64)
            median = float(np.median(sample))
            scale = max(1.4826 * float(np.median(np.abs(sample - median))), 1e-6)
            values[day, valid, decision, feature] = np.clip(
                (sample - median) / scale, -6.0, 6.0
            )
            values[day, ~valid, decision, feature] = 0.0
            masks[day, ~valid, decision, feature] = False


def _write_manifest(
    output_dir: Path,
    store: Path,
    values_path: Path,
    mask_path: Path,
    *,
    provenance: Mapping[str, object],
) -> None:
    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    masks = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    manifest = {
        "schema": EXTERNAL_SIDECAR_SCHEMA,
        "candidate_schema": P1_LIBRARY_SCHEMA,
        "cadence": "intraday",
        "feature_names": list(P1_FEATURES),
        "feature_store_identity": feature_store_identity(store),
        "axes": feature_store_axis_identity(store),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(provenance),
        "arrays": {
            "values.npy": {
                "shape": list(values.shape),
                "dtype": "float32",
                "sha256": _sha256(values_path),
            },
            "mask.npy": {
                "shape": list(masks.shape),
                "dtype": "bool",
                "sha256": _sha256(mask_path),
            },
        },
        "coverage": {
            name: int(masks[..., index].sum()) for index, name in enumerate(P1_FEATURES)
        },
        "explicit_omissions": {
            "sector_demeaned_reversal": (
                "No immutable point-in-time sector-classification axis is bound to "
                "the canonical store; current classifications were not backfilled."
            ),
            "after_1500_loser_reversal": (
                "The last canonical decision is 14:45, before the signal activates."
            ),
            "flat_volume_replacement": (
                "The incumbent already uses causal same-minute 20-session robust "
                "volume normalization; only incremental interval/opening variants "
                "are candidates."
            ),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_library_sidecar(store: Path, output_dir: Path) -> Path:
    store, output_dir = store.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    inputs = resolve_inputs()
    research_start, research_end = read_research_interval(inputs.universe_dir)
    assignments = load_assignments(inputs.assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    market_dates, valid_dates = load_market_dates_and_security_dates(
        cotahist_files(inputs.cotahist_dir), security_ids, research_start, research_end
    )
    validate_source_date_isolation(assignments, valid_dates)
    store_equities = (
        pl.read_parquet(store / "equity_index.parquet")
        .sort("equity_slot")
        .get_column("security_id")
        .to_list()
    )
    if set(store_equities) != set(security_ids):
        raise ValueError("P1 raw assignments differ from the canonical equity axis")
    slot = {security_id: index for index, security_id in enumerate(store_equities)}
    axes = feature_store_axis_identity(store)
    if int(axes["date_count"]) != len(market_dates):
        raise ValueError("P1 market dates differ from the canonical date axis")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        shape = (
            len(market_dates),
            EQUITY_COUNT,
            EXPECTED_DECISIONS_PER_DATE,
            len(P1_FEATURES),
        )
        values = np.lib.format.open_memmap(
            temporary / "values.npy", mode="w+", dtype=np.float32, shape=shape
        )
        masks = np.lib.format.open_memmap(
            temporary / "mask.npy", mode="w+", dtype=np.bool_, shape=shape
        )
        values[...] = 0.0
        masks[...] = False
        for source_number, group in enumerate(
            assignments.partition_by("source_file", maintain_order=True), start=1
        ):
            source_path = Path(group.item(0, "source_file"))
            source = load_source_file(source_path)
            validate_physical_source_identity(group, source, source_path)
            group_ids = tuple(group.get_column("security_id").to_list())
            allowed = frozenset().union(*(valid_dates[name] for name in group_ids))
            session = prepare_session_bars(
                source,
                source_path,
                allowed,
                market_dates,
                10 * 60,
                EQUITY_SESSION_MINUTES,
            )
            for assignment in group.iter_rows(named=True):
                security_id = str(assignment["security_id"])
                raw, observed = dense_grid(
                    session.filter(
                        pl.col("trade_date").is_in(tuple(valid_dates[security_id]))
                    ),
                    len(market_dates),
                    EQUITY_SESSION_MINUTES,
                )
                accepted = np.fromiter(
                    (
                        trade_date in valid_dates[security_id]
                        for trade_date in market_dates
                    ),
                    dtype=bool,
                    count=len(market_dates),
                )
                security_values, security_masks = build_security_library(
                    raw, observed, accepted
                )
                values[:, slot[security_id]] = security_values
                masks[:, slot[security_id]] = security_masks
            if source_number % 20 == 0:
                print(f"P1 raw features {source_number}", flush=True)

        active = np.load(store / "equity_membership.npy", mmap_mode="r") & np.load(
            store / "equity_data_ready.npy", mmap_mode="r"
        )
        for feature in (*range(3), 3, 14, 15, 18):
            _cross_sectional_scale(values, masks, active, feature)
        both = masks[..., 3] & masks[..., 14]
        values[..., 16][both] = values[..., 3][both] * values[..., 14][both]
        masks[..., 16] = both
        both = masks[..., 3] & masks[..., 15]
        values[..., 17][both] = values[..., 3][both] * values[..., 15][both]
        masks[..., 17] = both
        beta = np.load(store / "equity_slow.npy", mmap_mode="r")[
            ..., SLOW_CHANNELS.index("beta_to_WIN")
        ]
        win = np.load(store / "context_features.npy", mmap_mode="r")[:, 0]
        ready = np.load(store / "context_data_ready.npy", mmap_mode="r")[:, 0]
        for decision in range(EXPECTED_DECISIONS_PER_DATE):
            if decision < 39:  # 13:30 and later only.
                continue
            market = win[:, 74 + 5 * decision, 6]
            valid = active & ready[:, None]
            values[:, :, decision, 10][valid] = (beta * market[:, None])[valid]
            masks[:, :, decision, 10] = valid
        values[~masks] = 0.0
        if not np.isfinite(values[masks]).all():
            raise ValueError("P1 valid values are not finite")
        values.flush()
        masks.flush()
        del values, masks
        _write_manifest(
            temporary,
            store,
            temporary / "values.npy",
            temporary / "mask.npy",
            provenance={
                "raw_sources": sorted(
                    set(assignments.get_column("source_file").to_list())
                ),
                "identity": "exact accepted permanent security_id date assignments",
                "availability": "history ends strictly before each decision",
                "normalization_fit": "causal prior observations and contemporaneous cross-section only",
                "edge_reference": "Ardia, Guidotti, Kroencke (JFE 2024) published pseudocode",
            },
        )
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_dir


def _feature_ic(
    feature: np.ndarray,
    feature_mask: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
    dates: np.ndarray,
) -> float:
    predictions = np.repeat(feature[..., None], HORIZON_COUNT, axis=-1)
    mask = label_mask & feature_mask[..., None]
    return primary_validation_score(
        predictions.astype(np.float32), targets, mask, dates
    )


def _pooled_standardized_correlation(
    left: np.ndarray,
    left_mask: np.ndarray,
    right: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    mask = left_mask & right_mask
    counts = mask.sum(axis=1)
    valid_sample = counts >= 30
    if not valid_sample.any():
        return float("nan")
    safe = np.maximum(counts, 1)
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)
    left_mean = np.where(mask, left64, 0.0).sum(axis=1) / safe
    right_mean = np.where(mask, right64, 0.0).sum(axis=1) / safe
    left_centered = np.where(mask, left64 - left_mean[:, None], 0.0)
    right_centered = np.where(mask, right64 - right_mean[:, None], 0.0)
    left_centered[~valid_sample] = 0.0
    right_centered[~valid_sample] = 0.0
    denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    return (
        float(np.sum(left_centered * right_centered) / denominator)
        if denominator > 0
        else float("nan")
    )


def _existing_correlations(
    feature: np.ndarray,
    feature_mask: np.ndarray,
    current: np.ndarray,
    active: np.ndarray,
) -> list[float]:
    mask = feature_mask & active
    counts = mask.sum(axis=1)
    valid_sample = counts >= 30
    if not valid_sample.any():
        return [float("nan")] * current.shape[-1]
    safe = np.maximum(counts, 1).astype(np.float64)
    left = feature.astype(np.float64)
    left_mean = np.where(mask, left, 0.0).sum(axis=1) / safe
    left_centered = np.where(mask, left - left_mean[:, None], 0.0)
    left_centered[~valid_sample] = 0.0
    left_square = float(np.sum(left_centered**2))
    correlations: list[float] = []
    for start in range(0, current.shape[-1], 4):
        right = current[:, :, start : start + 4].astype(np.float64)
        right_mean = np.where(mask[..., None], right, 0.0).sum(axis=1) / safe[:, None]
        right_centered = np.where(mask[..., None], right - right_mean[:, None, :], 0.0)
        right_centered[~valid_sample] = 0.0
        numerator = np.sum(left_centered[..., None] * right_centered, axis=(0, 1))
        denominator = np.sqrt(left_square * np.sum(right_centered**2, axis=(0, 1)))
        correlations.extend(
            np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            ).tolist()
        )
    return correlations


def screen_feature_library(store: Path, library_dir: Path, output_dir: Path) -> Path:
    """Freeze F2 on the 407-date pre-April-2023 fit period only."""
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((library_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("feature_names") != list(P1_FEATURES):
        raise ValueError("P1 feature library contract differs")
    values = np.load(library_dir / "values.npy", mmap_mode="r")[
        : F2_FIT_END_DATE_IDX + 1
    ]
    masks = np.load(library_dir / "mask.npy", mmap_mode="r")[: F2_FIT_END_DATE_IDX + 1]
    targets_all = np.load(store / "targets.npy", mmap_mode="r")[
        : F2_FIT_END_DATE_IDX + 1
    ]
    labels_all = np.load(store / "label_mask.npy", mmap_mode="r")[
        : F2_FIT_END_DATE_IDX + 1
    ]
    sample = (
        pl.read_parquet(store / "sample_index.parquet")
        .filter(pl.col("date_idx") <= F2_FIT_END_DATE_IDX)
        .sort("sample_id")
    )
    date_idx = sample.get_column("date_idx").to_numpy()
    decision_idx = sample.get_column("decision_idx").to_numpy()
    features = values[date_idx, :, decision_idx]
    feature_masks = masks[date_idx, :, decision_idx]
    targets = targets_all[date_idx, :, decision_idx]
    label_mask = labels_all[date_idx, :, decision_idx]
    unique_dates = np.unique(date_idx)
    split_date = unique_dates[len(unique_dates) // 2]
    halves = (date_idx < split_date, date_idx >= split_date)

    existing_dynamic = np.load(store / "equity_features.npy", mmap_mode="r")
    existing_slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    active = np.load(store / "equity_membership.npy", mmap_mode="r") & np.load(
        store / "equity_data_ready.npy", mmap_mode="r"
    )
    current_values = []
    for row, (day, decision) in enumerate(zip(date_idx, decision_idx, strict=True)):
        cutoff = DECISION_EQUITY_INDICES[int(decision)]
        current_values.append(
            np.concatenate(
                (
                    existing_dynamic[day, :, cutoff - 1],
                    existing_slow[day],
                ),
                axis=-1,
            )
        )
    current = np.asarray(current_values, dtype=np.float32)
    current_mask = active[date_idx]

    rows = []
    for index, name in enumerate(P1_FEATURES):
        half_ics = [
            _feature_ic(
                features[half, :, index],
                feature_masks[half, :, index],
                targets[half],
                label_mask[half],
                date_idx[half],
            )
            for half in halves
        ]
        correlations = [
            abs(value)
            for value in _existing_correlations(
                features[:, :, index],
                feature_masks[:, :, index],
                current,
                current_mask,
            )
        ]
        finite_corr = [value for value in correlations if np.isfinite(value)]
        max_corr = max(finite_corr, default=0.0)
        finite_halves = all(np.isfinite(value) for value in half_ics)
        stable = finite_halves and half_ics[0] * half_ics[1] > 0
        minimum_half_ic = min(map(abs, half_ics)) if finite_halves else 0.0
        eligible = stable and minimum_half_ic >= F2_MIN_HALF_ABS_IC
        rows.append(
            {
                "feature": name,
                "family": FEATURE_FAMILY[name],
                "tier": FAMILY_TIER[FEATURE_FAMILY[name]],
                "half_1_ic": half_ics[0] if finite_halves else None,
                "half_2_ic": half_ics[1] if finite_halves else None,
                "fit_mean_ic": float(np.mean(half_ics)) if finite_halves else None,
                "max_abs_correlation_existing": max_corr,
                "incremental_score": minimum_half_ic * (1.0 - max_corr**2),
                "eligible": eligible,
            }
        )

    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["incremental_score"]),
            int(row["tier"]),
            str(row["feature"]),
        ),
    )
    selected: list[str] = []
    family_counts: dict[str, int] = {}
    for row in ordered:
        if not row["eligible"] or len(selected) == F2_TOP_K:
            continue
        family = str(row["family"])
        if family_counts.get(family, 0) >= F2_MAX_FAMILY_MEMBERS:
            continue
        candidate = P1_FEATURES.index(str(row["feature"]))
        if any(
            abs(
                _pooled_standardized_correlation(
                    features[:, :, candidate],
                    feature_masks[:, :, candidate],
                    features[:, :, P1_FEATURES.index(other)],
                    feature_masks[:, :, P1_FEATURES.index(other)],
                )
            )
            >= F2_MAX_PAIR_CORRELATION
            for other in selected
        ):
            continue
        selected.append(str(row["feature"]))
        family_counts[family] = family_counts.get(family, 0) + 1

    selection = {
        "schema": F2_SELECTION_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_dates": {
            "first_date_idx": 0,
            "last_date_idx": F2_FIT_END_DATE_IDX,
            "date_count": F2_FIT_END_DATE_IDX + 1,
            "purpose": "feature selection only; all three F3 selection windows remain unseen",
        },
        "selection_rule": {
            "top_k": F2_TOP_K,
            "minimum_half_abs_ic": F2_MIN_HALF_ABS_IC,
            "same_ic_sign_in_both_chronological_halves": True,
            "maximum_members_per_family": F2_MAX_FAMILY_MEMBERS,
            "maximum_pair_correlation": F2_MAX_PAIR_CORRELATION,
            "incremental_score": "min(abs(half ICs)) * (1 - max_existing_corr^2)",
            "minimum_features_for_f3": 6,
        },
        "features": rows,
        "selected_features": selected,
        "f3_allowed": len(selected) >= 6,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    (output_dir / "f2_selection.json").write_text(
        json.dumps(selection, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if len(selected) < 6:
        return output_dir

    selected_indices = [P1_FEATURES.index(name) for name in selected]
    sidecar = output_dir / "selected_sidecar"
    sidecar.mkdir()
    for filename, dtype in (("values.npy", np.float32), ("mask.npy", np.bool_)):
        source = np.load(library_dir / filename, mmap_mode="r")
        destination = np.lib.format.open_memmap(
            sidecar / filename,
            mode="w+",
            dtype=dtype,
            shape=(*source.shape[:-1], len(selected)),
        )
        destination[...] = source[..., selected_indices]
        destination.flush()
        del destination
    selected_manifest = {
        **manifest,
        "candidate_schema": "P1_F2_SELECTED_SIDECAR_V1",
        "feature_names": selected,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            **manifest["provenance"],
            "source_library": str(library_dir.resolve()),
            "f2_selection": str((output_dir / "f2_selection.json").resolve()),
            "selection_window": "first 407 training dates only",
        },
        "arrays": {
            filename: {
                "shape": list(np.load(sidecar / filename, mmap_mode="r").shape),
                "dtype": dtype,
                "sha256": _sha256(sidecar / filename),
            }
            for filename, dtype in (("values.npy", "float32"), ("mask.npy", "bool"))
        },
    }
    (sidecar / "manifest.json").write_text(
        json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and screen the causal P1 feature library"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--f2-dir", type=Path, required=True)
    args = parser.parse_args()
    library = build_library_sidecar(args.store, args.library_dir)
    print(screen_feature_library(args.store, library, args.f2_dir))


if __name__ == "__main__":
    main()
