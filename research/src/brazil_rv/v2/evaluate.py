from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from brazil_rv.execution.stateful_ledger import (
    LedgerConfig,
    StatefulLedgerResult,
    ledger_configurations,
    ledger_sensitivity_grid,
)
from brazil_rv.modeling.metrics import average_ranks, moving_block_bootstrap

from .artifacts import write_json_atomic
from .config import FULL_PROTOCOL, ProtocolPreset
from .contract import HORIZONS, PRIMARY_HORIZONS
from .splits import (
    PREREGISTRATION_ROOT,
    authorize_dates,
    validate_contiguous_session_axis,
)

MIN_CROSS_SECTION = 30
BOOTSTRAP_SEED = 20260903
ECONOMICS_COSTS_BPS = (2.0, 4.0, 7.0)
ECONOMICS_ANNUAL_BORROW_RATES = (0.02, 0.04)
ECONOMICS_HEADLINE = (4.0, 0.02)


@dataclass(frozen=True)
class EvaluationInputs:
    dates: tuple[date, ...]
    session_indices: NDArray[np.integer]
    calendar_identity_sha256: str
    scores: NDArray[np.floating]
    score_mask: NDArray[np.bool_]
    median_residual_midrank_targets: NDArray[np.floating]
    raw_midrank_targets: NDArray[np.floating]
    raw_log_returns: NDArray[np.floating]
    target_mask: NDArray[np.bool_]
    raw_target_mask: NDArray[np.bool_]
    active: NDArray[np.bool_]
    adjusted_close: NDArray[np.floating]
    neutralized_log_return: NDArray[np.floating]
    neutralized_log_return_valid: NDArray[np.bool_]
    return_neutralized_event: NDArray[np.bool_]
    cross_sectional_median_log_return: NDArray[np.floating]
    target_scale_sigma: NDArray[np.floating]
    prior_feature_values: Mapping[str, NDArray[np.floating]]
    cdi_returns: NDArray[np.floating]
    horizons: tuple[int, ...] = HORIZONS
    source_artifact_hashes: Mapping[str, str] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    report: dict[str, object]
    dates: tuple[date, ...]
    daily_primary_ic: NDArray[np.float64]
    headline_economics_dates: tuple[date, ...]
    headline_net_excess_bps: NDArray[np.float64]


def _array_sha256(values: NDArray[np.generic]) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(value) for value in array.shape).encode("ascii"))
    digest.update(b"\0")
    if array.ndim == 0:
        digest.update(np.ascontiguousarray(array).tobytes())
    else:
        for row in array:
            digest.update(np.ascontiguousarray(row).tobytes())
    return digest.hexdigest()


def _dates_sha256(dates: Sequence[date]) -> str:
    digest = hashlib.sha256()
    for value in dates:
        digest.update(value.isoformat().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _finite_mean(values: NDArray[np.floating]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else math.nan


def _spearman(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    mask: NDArray[np.bool_],
) -> float:
    valid = np.asarray(mask, dtype=bool).copy()
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    valid &= np.isfinite(left_values) & np.isfinite(right_values)
    if int(valid.sum()) < MIN_CROSS_SECTION:
        return math.nan
    left_ranks = average_ranks(left_values[valid])
    right_ranks = average_ranks(right_values[valid])
    left_ranks -= left_ranks.mean()
    right_ranks -= right_ranks.mean()
    denominator = float(
        np.sqrt(np.square(left_ranks).sum() * np.square(right_ranks).sum())
    )
    if denominator == 0.0:
        return math.nan
    return float(np.dot(left_ranks, right_ranks) / denominator)


def _validate(inputs: EvaluationInputs) -> None:
    dates = inputs.dates
    validate_contiguous_session_axis(dates, inputs.session_indices)
    _validate_sha256(inputs.calendar_identity_sha256, label="calendar_identity_sha256")
    if tuple(inputs.horizons) != HORIZONS:
        raise ValueError("evaluation horizon axis differs from the frozen v2 contract")
    scores = np.asarray(inputs.scores)
    active = np.asarray(inputs.active)
    if active.ndim != 2 or active.shape[0] != len(dates):
        raise ValueError("active mask must have date-by-name shape")
    expected = (len(dates), active.shape[1], len(HORIZONS))
    if scores.shape != expected:
        raise ValueError("scores must have date-by-name-by-horizon shape")
    for name, values in (
        ("score_mask", inputs.score_mask),
        (
            "median_residual_midrank_targets",
            inputs.median_residual_midrank_targets,
        ),
        ("raw_midrank_targets", inputs.raw_midrank_targets),
        ("raw_log_returns", inputs.raw_log_returns),
        ("target_mask", inputs.target_mask),
        ("raw_target_mask", inputs.raw_target_mask),
    ):
        if np.asarray(values).shape != expected:
            raise ValueError(f"{name} shape differs from scores")
    matrix_shape = expected[:2]
    if active.shape != matrix_shape:
        raise ValueError("active mask shape differs from scores")
    if np.asarray(inputs.adjusted_close).shape != matrix_shape:
        raise ValueError("adjusted close shape differs from scores")
    for name, values in (
        ("neutralized_log_return", inputs.neutralized_log_return),
        ("neutralized_log_return_valid", inputs.neutralized_log_return_valid),
        ("return_neutralized_event", inputs.return_neutralized_event),
        ("target_scale_sigma", inputs.target_scale_sigma),
    ):
        if np.asarray(values).shape != matrix_shape:
            raise ValueError(f"{name} shape differs from scores")
    if np.asarray(inputs.cross_sectional_median_log_return).shape != (len(dates),):
        raise ValueError("cross-sectional median return axis differs from dates")
    if np.asarray(inputs.cdi_returns).shape != (len(dates),):
        raise ValueError("CDI return axis differs from evaluation dates")
    if np.asarray(inputs.score_mask).dtype != np.bool_:
        raise TypeError("score_mask must be a Boolean array")
    if np.asarray(inputs.target_mask).dtype != np.bool_:
        raise TypeError("target_mask must be a Boolean array")
    if np.asarray(inputs.raw_target_mask).dtype != np.bool_:
        raise TypeError("raw_target_mask must be a Boolean array")
    if active.dtype != np.bool_:
        raise TypeError("active must be a Boolean array")
    if np.asarray(inputs.neutralized_log_return_valid).dtype != np.bool_:
        raise TypeError("neutralized-log-return validity must be Boolean")
    if np.asarray(inputs.return_neutralized_event).dtype != np.bool_:
        raise TypeError("return-neutralized event mask must be Boolean")
    expected_features = {
        "yang_zhang_vol_20",
        "beta_60",
        "log_volume_mean_20",
        "momentum_12_1",
        "log_return_5",
    }
    if set(inputs.prior_feature_values) != expected_features:
        raise ValueError("evaluation prior-feature diagnostic roster differs")
    if any(
        np.asarray(values).shape != matrix_shape
        for values in inputs.prior_feature_values.values()
    ):
        raise ValueError("evaluation prior-feature arrays are misaligned")
    if not np.isfinite(np.asarray(inputs.cdi_returns, dtype=np.float64)).all():
        raise ValueError("CDI returns must be finite")
    for horizon_index, horizon in enumerate(HORIZONS):
        if np.asarray(inputs.target_mask)[-horizon:, :, horizon_index].any():
            raise ValueError(
                "target mask permits a horizon endpoint outside the window"
            )
        if np.asarray(inputs.raw_target_mask)[-horizon:, :, horizon_index].any():
            raise ValueError(
                "raw target mask permits a horizon endpoint outside the window"
            )
    if not inputs.source_artifact_hashes:
        raise ValueError("evaluation requires at least one source artifact identity")
    for name, digest in inputs.source_artifact_hashes.items():
        if not name:
            raise ValueError("source artifact hash names must be nonempty")
        _validate_sha256(digest, label="source artifact hashes")


def _validate_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lower-case SHA-256")


def _daily_metrics(
    inputs: EvaluationInputs,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    list[dict[str, object]],
]:
    scores = np.asarray(inputs.scores, dtype=np.float64)
    score_mask = np.asarray(inputs.score_mask, dtype=bool)
    active = np.asarray(inputs.active, dtype=bool)
    target_mask = np.asarray(inputs.target_mask, dtype=bool)
    raw_target_mask = np.asarray(inputs.raw_target_mask, dtype=bool)
    residual = np.asarray(inputs.median_residual_midrank_targets, dtype=np.float64)
    raw_rank = np.asarray(inputs.raw_midrank_targets, dtype=np.float64)
    raw_return = np.asarray(inputs.raw_log_returns, dtype=np.float64)
    days = len(inputs.dates)
    horizon_count = len(HORIZONS)
    residual_ic = np.full((days, horizon_count), np.nan, dtype=np.float64)
    raw_ic = np.full_like(residual_ic, np.nan)
    decile_spread = np.full_like(residual_ic, np.nan)
    rows: list[dict[str, object]] = []
    for day, day_value in enumerate(inputs.dates):
        for horizon_index, horizon in enumerate(HORIZONS):
            residual_valid = (
                score_mask[day, :, horizon_index]
                & target_mask[day, :, horizon_index]
                & active[day]
                & np.isfinite(scores[day, :, horizon_index])
                & np.isfinite(residual[day, :, horizon_index])
            )
            residual_ic[day, horizon_index] = _spearman(
                scores[day, :, horizon_index],
                residual[day, :, horizon_index],
                residual_valid,
            )
            raw_base_valid = (
                score_mask[day, :, horizon_index]
                & raw_target_mask[day, :, horizon_index]
                & active[day]
                & np.isfinite(scores[day, :, horizon_index])
            )
            raw_rank_valid = raw_base_valid & np.isfinite(
                raw_rank[day, :, horizon_index]
            )
            raw_ic[day, horizon_index] = _spearman(
                scores[day, :, horizon_index],
                raw_rank[day, :, horizon_index],
                raw_rank_valid,
            )
            decile_valid = raw_base_valid & np.isfinite(
                raw_return[day, :, horizon_index]
            )
            names = np.flatnonzero(decile_valid)
            if names.size >= MIN_CROSS_SECTION:
                decile_count = max(1, names.size // 10)
                order = names[
                    np.argsort(scores[day, names, horizon_index], kind="stable")
                ]
                bottom = raw_return[day, order[:decile_count], horizon_index].mean()
                top = raw_return[day, order[-decile_count:], horizon_index].mean()
                decile_spread[day, horizon_index] = (
                    float(top - bottom) * 10_000.0 / horizon
                )
            rows.append(
                {
                    "date": day_value.isoformat(),
                    "horizon_sessions": horizon,
                    "median_residual_valid_name_count": int(residual_valid.sum()),
                    "raw_rank_valid_name_count": int(raw_rank_valid.sum()),
                    "decile_valid_name_count": int(decile_valid.sum()),
                    "median_residual_spearman_ic": _finite_or_none(
                        residual_ic[day, horizon_index]
                    ),
                    "raw_rank_ic": _finite_or_none(raw_ic[day, horizon_index]),
                    "decile_spread_bps_per_holding_session": _finite_or_none(
                        decile_spread[day, horizon_index]
                    ),
                }
            )
    return residual_ic, raw_ic, decile_spread, rows


def _persistence(
    inputs: EvaluationInputs,
) -> tuple[dict[int, NDArray[np.float64]], list[dict[str, object]]]:
    scores = np.asarray(inputs.scores, dtype=np.float64)
    score_mask = np.asarray(inputs.score_mask, dtype=bool)
    active = np.asarray(inputs.active, dtype=bool)
    results: dict[int, NDArray[np.float64]] = {}
    rows: list[dict[str, object]] = []
    for lag in (1, 5):
        values = np.full((len(inputs.dates), len(HORIZONS)), np.nan)
        for day in range(lag, len(inputs.dates)):
            for horizon_index, _ in enumerate(HORIZONS):
                valid = (
                    active[day]
                    & active[day - lag]
                    & score_mask[day, :, horizon_index]
                    & score_mask[day - lag, :, horizon_index]
                )
                values[day, horizon_index] = _spearman(
                    scores[day, :, horizon_index],
                    scores[day - lag, :, horizon_index],
                    valid,
                )
        results[lag] = values
        for day, day_value in enumerate(inputs.dates):
            for horizon_index, horizon in enumerate(HORIZONS):
                rows.append(
                    {
                        "date": day_value.isoformat(),
                        "lag_sessions": lag,
                        "horizon_sessions": horizon,
                        "path_model_count": 1,
                        "spearman": _finite_or_none(values[day, horizon_index]),
                    }
                )
    return results, rows


def _economics_signal(
    inputs: EvaluationInputs,
    scores_input: NDArray[np.floating] | None = None,
    score_mask_input: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    indexes = [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS]
    scores = np.asarray(
        inputs.scores if scores_input is None else scores_input, dtype=np.float64
    )[..., indexes]
    masks = np.asarray(
        inputs.score_mask if score_mask_input is None else score_mask_input, dtype=bool
    )[..., indexes]
    active = np.asarray(inputs.active, dtype=bool)
    composite = np.full(scores.shape[:2], np.nan, dtype=np.float64)
    composite_mask = masks.all(axis=-1) & active & np.isfinite(scores).all(axis=-1)
    for day in range(scores.shape[0]):
        valid = composite_mask[day]
        count = int(valid.sum())
        if count:
            head_ranks = np.stack(
                [
                    average_ranks(scores[day, valid, horizon_index])
                    for horizon_index in range(len(PRIMARY_HORIZONS))
                ]
            )
            averaged_rank = head_ranks.mean(axis=0)
            composite[day, valid] = 2.0 * ((averaged_rank + 0.5) / count) - 1.0
    return composite, composite_mask


def _ledger_rows(
    result: StatefulLedgerResult,
    *,
    cost_bps: float,
    annual_borrow_rate: float,
) -> list[dict[str, object]]:
    return [
        {
            "date": day.isoformat(),
            "cost_bps_per_side": cost_bps,
            "annual_borrow_rate": annual_borrow_rate,
            "interval_valid": True,
            "gross_pnl_bps": _finite_or_none(result.gross_pnl_bps[index]),
            "neutral_flow_bps": _finite_or_none(result.neutral_flow_bps[index]),
            "interest_bps": _finite_or_none(result.interest_bps[index]),
            "turnover_fraction_nav": _finite_or_none(
                result.turnover_fraction_nav[index]
            ),
            "turnover_cost_bps": _finite_or_none(result.cost_bps[index]),
            "borrow_cost_bps": _finite_or_none(result.borrow_bps[index]),
            "net_return": _finite_or_none(result.daily_net_return[index]),
            "net_excess_all_cash_bps": _finite_or_none(
                result.net_excess_all_cash_bps[index]
            ),
            "deployed_gross_fraction_nav": _finite_or_none(
                result.gross_fraction_nav[index]
            ),
            "stale_mark_name_days": int(result.stale_mark_name_days[index]),
            "neutral_marked_name_days": int(result.neutral_marked_name_days[index]),
            "forced_liquidation_count": int(result.forced_liquidation_count[index]),
        }
        for index, day in enumerate(result.dates)
    ]


def _input_hashes(inputs: EvaluationInputs) -> dict[str, str]:
    result = {
        "dates": _dates_sha256(inputs.dates),
        "canonical_calendar": inputs.calendar_identity_sha256,
        "session_indices": _array_sha256(np.asarray(inputs.session_indices)),
        "scores": _array_sha256(np.asarray(inputs.scores)),
        "score_mask": _array_sha256(np.asarray(inputs.score_mask)),
        "median_residual_midrank_targets": _array_sha256(
            np.asarray(inputs.median_residual_midrank_targets)
        ),
        "raw_midrank_targets": _array_sha256(np.asarray(inputs.raw_midrank_targets)),
        "raw_log_returns": _array_sha256(np.asarray(inputs.raw_log_returns)),
        "target_mask": _array_sha256(np.asarray(inputs.target_mask)),
        "raw_target_mask": _array_sha256(np.asarray(inputs.raw_target_mask)),
        "active": _array_sha256(np.asarray(inputs.active)),
        "adjusted_close": _array_sha256(np.asarray(inputs.adjusted_close)),
        "neutralized_log_return": _array_sha256(
            np.asarray(inputs.neutralized_log_return)
        ),
        "neutralized_log_return_valid": _array_sha256(
            np.asarray(inputs.neutralized_log_return_valid)
        ),
        "return_neutralized_event": _array_sha256(
            np.asarray(inputs.return_neutralized_event)
        ),
        "cross_sectional_median_log_return": _array_sha256(
            np.asarray(inputs.cross_sectional_median_log_return)
        ),
        "target_scale_sigma": _array_sha256(np.asarray(inputs.target_scale_sigma)),
        "cdi_returns": _array_sha256(np.asarray(inputs.cdi_returns)),
    }
    for name, values in sorted(inputs.prior_feature_values.items()):
        result[f"prior_feature_{name}"] = _array_sha256(np.asarray(values))
    return result


def _economics_contract() -> dict[str, object]:
    config = LedgerConfig()
    return {
        "signal_construction": (
            "arithmetic mean of each D=1,2,3,5 head's tie-aware "
            "cross-sectional ranks, centered and rescaled to [-1,1]; "
            "all four score masks required and D=10 excluded"
        ),
        "signal_horizons_sessions": list(PRIMARY_HORIZONS),
        "k_per_side": config.k_per_side,
        "buffer_per_side": config.buffer_per_side,
        "gross_target": config.gross_target,
        "annual_sessions": config.annual_sessions,
        "terminal_liquidation": True,
        "stateful_policy": "buffered held inventory; no resizing or drift trades",
        "missing_print_policy": (
            "stale mark until a print, then exit; force after 10 sessions or "
            "inactive-without-print"
        ),
        "event_marking": "neutralized market move with synthetic flow",
        "short_proceeds_remuneration": config.short_proceeds_remuneration,
        "costs_bps_per_side": list(ECONOMICS_COSTS_BPS),
        "annual_borrow_rates": list(ECONOMICS_ANNUAL_BORROW_RATES),
        "headline": {
            "cost_bps_per_side": ECONOMICS_HEADLINE[0],
            "annual_borrow_rate": ECONOMICS_HEADLINE[1],
        },
    }


def _diagnostics(inputs: EvaluationInputs) -> dict[str, object]:
    scores = np.asarray(inputs.scores, dtype=np.float64)
    masks = np.asarray(inputs.score_mask, dtype=np.bool_)
    active = np.asarray(inputs.active, dtype=np.bool_)
    target_mask = np.asarray(inputs.target_mask, dtype=np.bool_)
    residual = np.asarray(inputs.median_residual_midrank_targets, dtype=np.float64)
    composite, composite_mask = _economics_signal(inputs)
    exposure_rows = []
    exposure_summary = []
    for name, raw_values in sorted(inputs.prior_feature_values.items()):
        values = np.asarray(raw_values, dtype=np.float64)
        daily = np.asarray(
            [
                _spearman(composite[day], values[day], composite_mask[day])
                for day in range(len(inputs.dates))
            ],
            dtype=np.float64,
        )
        exposure_rows.extend(
            {
                "date": day.isoformat(),
                "feature": name,
                "spearman": _finite_or_none(daily[index]),
            }
            for index, day in enumerate(inputs.dates)
        )
        finite = daily[np.isfinite(daily)]
        interval = {
            "estimate": _finite_or_none(_finite_mean(daily)),
            "lower_95": None,
            "upper_95": None,
        }
        if finite.size >= 20:
            interval = _bootstrap_payload(
                finite, replications=10_000, block_length=min(20, len(finite))
            )
        exposure_summary.append({"feature": name, **interval})

    matched_rows = []
    d10 = HORIZONS.index(10)
    for horizon_index, horizon in enumerate(HORIZONS):
        values = []
        for day in range(len(inputs.dates)):
            valid = (
                masks[day, :, horizon_index]
                & target_mask[day, :, horizon_index]
                & target_mask[day, :, d10]
                & active[day]
            )
            values.append(
                _spearman(
                    scores[day, :, horizon_index],
                    residual[day, :, horizon_index],
                    valid,
                )
            )
        matched_rows.append(
            {
                "horizon_sessions": horizon,
                "mean_median_residual_ic": _finite_or_none(
                    _finite_mean(np.asarray(values))
                ),
            }
        )

    neutral_returns = np.asarray(inputs.neutralized_log_return, dtype=np.float64)
    neutral_valid = np.asarray(inputs.neutralized_log_return_valid, dtype=np.bool_)
    sigma = np.asarray(inputs.target_scale_sigma, dtype=np.float64)
    intervals = ((0, 1), (1, 3), (3, 5), (5, 10))
    incremental_rows = []
    for head_index, head_horizon in enumerate(HORIZONS):
        for start, stop in intervals:
            daily = []
            for day in range(len(inputs.dates)):
                if day + stop >= len(inputs.dates):
                    daily.append(np.nan)
                    continue
                path = neutral_returns[day + start + 1 : day + stop + 1]
                path_valid = neutral_valid[day + start + 1 : day + stop + 1].all(axis=0)
                realized = path.sum(axis=0, dtype=np.float64)
                valid = (
                    masks[day, :, head_index]
                    & active[day]
                    & path_valid
                    & np.isfinite(sigma[day])
                    & (sigma[day] > 0)
                )
                if valid.any():
                    realized[valid] = (realized[valid] - np.median(realized[valid])) / (
                        sigma[day, valid] * np.sqrt(stop - start)
                    )
                daily.append(_spearman(scores[day, :, head_index], realized, valid))
            incremental_rows.append(
                {
                    "head_horizon_sessions": head_horizon,
                    "increment_start_sessions": start,
                    "increment_end_sessions": stop,
                    "mean_spearman_ic": _finite_or_none(
                        _finite_mean(np.asarray(daily))
                    ),
                }
            )
    return {
        "exposure_daily": exposure_rows,
        "exposure_summary": exposure_summary,
        "incremental_horizon_ic": incremental_rows,
        "matched_universe_ic": matched_rows,
    }


def evaluate_scores(
    inputs: EvaluationInputs,
    *,
    window_name: str,
    registration_path: Path | None = None,
    preregistration_root: Path = PREREGISTRATION_ROOT,
) -> EvaluationResult:
    """Evaluate one score cube after enforcing the v2 access boundary."""
    ledger = authorize_dates(
        inputs.dates,
        purpose="evaluation",
        registration_path=registration_path,
        preregistration_root=preregistration_root,
    )
    _validate(inputs)
    if not window_name:
        raise ValueError("window_name must be nonempty")
    residual_ic, raw_ic, decile_spread, metric_rows = _daily_metrics(inputs)
    persistence, persistence_rows = _persistence(inputs)
    primary_indexes = [HORIZONS.index(horizon) for horizon in PRIMARY_HORIZONS]
    daily_primary = np.asarray(
        [
            _finite_mean(residual_ic[day, primary_indexes])
            for day in range(len(inputs.dates))
        ],
        dtype=np.float64,
    )
    economics_score, economics_mask = _economics_signal(inputs)
    grid = ledger_sensitivity_grid(
        dates=inputs.dates,
        scores=economics_score,
        score_mask=economics_mask,
        active=np.asarray(inputs.active, dtype=np.bool_),
        adjusted_close=inputs.adjusted_close,
        neutralized_log_return=inputs.neutralized_log_return,
        neutralized_log_return_valid=inputs.neutralized_log_return_valid,
        return_neutralized_event_mask=inputs.return_neutralized_event,
        cross_sectional_median_log_return=(inputs.cross_sectional_median_log_return),
        cdi_returns=inputs.cdi_returns,
    )
    configurations = ledger_configurations()
    headline_name = f"cost_{ECONOMICS_HEADLINE[0]:g}_borrow_{ECONOMICS_HEADLINE[1]:g}"
    headline = grid[headline_name]
    horizon_rows: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(HORIZONS):
        horizon_rows.append(
            {
                "horizon_sessions": horizon,
                "primary_horizon": horizon in PRIMARY_HORIZONS,
                "mean_median_residual_spearman_ic": _finite_or_none(
                    _finite_mean(residual_ic[:, horizon_index])
                ),
                "mean_raw_rank_ic": _finite_or_none(
                    _finite_mean(raw_ic[:, horizon_index])
                ),
                "mean_decile_spread_bps_per_holding_session": _finite_or_none(
                    _finite_mean(decile_spread[:, horizon_index])
                ),
                "mean_persistence_1_session": _finite_or_none(
                    _finite_mean(persistence[1][:, horizon_index])
                ),
                "mean_persistence_5_sessions": _finite_or_none(
                    _finite_mean(persistence[5][:, horizon_index])
                ),
            }
        )
    economics_summaries: list[dict[str, object]] = []
    economics_daily: list[dict[str, object]] = []
    for scenario, result in grid.items():
        config = configurations[scenario]
        economics_summaries.append(
            {
                "scenario": scenario,
                "cost_bps_per_side": config.cost_bps_per_side,
                "annual_borrow_rate": config.annual_borrow_rate,
                "buffer_per_side": config.buffer_per_side,
                "short_proceeds_remuneration": (config.short_proceeds_remuneration),
                "ex_date_marking": config.ex_date_marking,
                "forced_liquidation_haircut": (config.forced_liquidation_haircut),
                "path_model_count": 1,
                **{
                    key: _finite_or_none(value) if isinstance(value, float) else value
                    for key, value in result.summary().items()
                },
            }
        )
        if scenario.startswith("cost_"):
            economics_daily.extend(
                {
                    **row,
                    "scenario": scenario,
                }
                for row in _ledger_rows(
                    result,
                    cost_bps=config.cost_bps_per_side,
                    annual_borrow_rate=config.annual_borrow_rate,
                )
            )
    report: dict[str, object] = {
        "schema": "BRAZIL_RV_V2_EVALUATION_V2",
        "window": {
            "name": window_name,
            "start": inputs.dates[0].isoformat(),
            "end": inputs.dates[-1].isoformat(),
            "date_count": len(inputs.dates),
            "date_identity_sha256": _dates_sha256(inputs.dates),
            "canonical_calendar_sha256": inputs.calendar_identity_sha256,
            "first_session_index": int(np.asarray(inputs.session_indices)[0]),
            "last_session_index": int(np.asarray(inputs.session_indices)[-1]),
        },
        "access": ledger.payload(),
        "official_validation_accessed": ledger.official_validation_accessed,
        "test_accessed": ledger.test_accessed,
        "horizons_sessions": list(HORIZONS),
        "primary_horizons_sessions": list(PRIMARY_HORIZONS),
        "pooled_primary_median_residual_ic": _finite_or_none(
            _finite_mean(
                np.asarray(
                    [_finite_mean(residual_ic[:, index]) for index in primary_indexes]
                )
            )
        ),
        "daily_primary_ic": [
            {
                "date": value.isoformat(),
                "mean_primary_horizon_ic": _finite_or_none(daily_primary[index]),
            }
            for index, value in enumerate(inputs.dates)
        ],
        "horizon_readouts": horizon_rows,
        "daily_metric_table": metric_rows,
        "persistence_table": persistence_rows,
        "economics": {
            "contract": _economics_contract(),
            "summaries": economics_summaries,
            "daily_table": economics_daily,
        },
        "diagnostics": _diagnostics(inputs),
        "input_hashes": _input_hashes(inputs),
        "source_artifact_hashes": dict(
            sorted((inputs.source_artifact_hashes or {}).items())
        ),
        "mask_coverage": {
            "score_mask_true": int(np.asarray(inputs.score_mask).sum()),
            "target_mask_true": int(np.asarray(inputs.target_mask).sum()),
            "raw_target_mask_true": int(np.asarray(inputs.raw_target_mask).sum()),
            "economics_score_mask_true": int(economics_mask.sum()),
            "economics_path_model_count": 1,
            "return_neutralized_event_true": int(
                np.asarray(inputs.return_neutralized_event, dtype=np.bool_).sum()
            ),
            "neutralized_log_return_valid_true": int(
                np.asarray(inputs.neutralized_log_return_valid, dtype=np.bool_).sum()
            ),
            "stale_mark_name_days": int(headline.stale_mark_name_days.sum()),
            "neutral_marked_name_days": int(headline.neutral_marked_name_days.sum()),
        },
    }
    return EvaluationResult(
        report=report,
        dates=inputs.dates,
        daily_primary_ic=daily_primary,
        headline_economics_dates=headline.dates,
        headline_net_excess_bps=headline.net_excess_all_cash_bps,
    )


def _bootstrap_payload(
    values: NDArray[np.floating],
    *,
    replications: int,
    block_length: int,
) -> dict[str, float | None]:
    if replications == 0:
        return {
            "estimate": _finite_or_none(_finite_mean(values)),
            "lower_95": None,
            "upper_95": None,
        }
    output = moving_block_bootstrap(
        values,
        replications=replications,
        block_length=block_length,
        seed=BOOTSTRAP_SEED,
    )
    return {
        key: _finite_or_none(np.asarray(value).reshape(-1)[0])
        for key, value in output.items()
    }


def paired_comparison(
    candidate: EvaluationResult,
    baseline: EvaluationResult,
    *,
    registration_path: Path | None = None,
    preregistration_root: Path = PREREGISTRATION_ROOT,
    protocol: ProtocolPreset = FULL_PROTOCOL,
) -> dict[str, object]:
    """Compare aligned daily metrics without estimating on another window."""
    ledger = authorize_dates(
        candidate.dates,
        purpose="evaluation",
        registration_path=registration_path,
        preregistration_root=preregistration_root,
    )
    if candidate.dates != baseline.dates:
        raise ValueError("paired IC comparison requires identical date axes")
    if candidate.headline_economics_dates != baseline.headline_economics_dates:
        raise ValueError("paired economics comparison requires identical date axes")
    _validate_paired_identity(candidate.report, baseline.report)
    if candidate.daily_primary_ic.shape != (len(candidate.dates),) or (
        baseline.daily_primary_ic.shape != (len(baseline.dates),)
    ):
        raise ValueError("paired primary-IC arrays differ from their date axes")
    if candidate.headline_net_excess_bps.shape != (
        len(candidate.headline_economics_dates),
    ) or baseline.headline_net_excess_bps.shape != (
        len(baseline.headline_economics_dates),
    ):
        raise ValueError("paired economics arrays differ from their date axes")
    ic_delta = candidate.daily_primary_ic - baseline.daily_primary_ic
    economics_delta = (
        candidate.headline_net_excess_bps - baseline.headline_net_excess_bps
    )
    if (
        protocol.bootstrap_replications > 0
        and min(len(ic_delta), len(economics_delta)) < protocol.bootstrap_block_length
    ):
        raise ValueError(
            "paired comparison requires at least the preset bootstrap block length"
        )
    return {
        "schema": "BRAZIL_RV_V2_PAIRED_COMPARISON_V1",
        "access": ledger.payload(),
        "official_validation_accessed": ledger.official_validation_accessed,
        "test_accessed": ledger.test_accessed,
        "date_identity_sha256": _dates_sha256(candidate.dates),
        "protocol": protocol.name,
        "block_length_sessions": protocol.bootstrap_block_length,
        "replications": protocol.bootstrap_replications,
        "seed": BOOTSTRAP_SEED,
        "daily_primary_ic_delta": _bootstrap_payload(
            ic_delta,
            replications=protocol.bootstrap_replications,
            block_length=protocol.bootstrap_block_length,
        ),
        "daily_headline_net_excess_bps_delta": _bootstrap_payload(
            economics_delta,
            replications=protocol.bootstrap_replications,
            block_length=protocol.bootstrap_block_length,
        ),
        "daily_primary_ic_delta_table": [
            {
                "date": value.isoformat(),
                "delta": _finite_or_none(ic_delta[index]),
            }
            for index, value in enumerate(candidate.dates)
        ],
        "daily_headline_net_excess_bps_delta_table": [
            {
                "date": value.isoformat(),
                "delta": _finite_or_none(economics_delta[index]),
            }
            for index, value in enumerate(candidate.headline_economics_dates)
        ],
    }


_PAIRED_INPUT_KEYS = (
    "dates",
    "canonical_calendar",
    "session_indices",
    "score_mask",
    "median_residual_midrank_targets",
    "raw_midrank_targets",
    "raw_log_returns",
    "target_mask",
    "raw_target_mask",
    "active",
    "adjusted_close",
    "neutralized_log_return",
    "neutralized_log_return_valid",
    "return_neutralized_event",
    "cross_sectional_median_log_return",
    "target_scale_sigma",
    "cdi_returns",
    "prior_feature_beta_60",
    "prior_feature_log_return_5",
    "prior_feature_log_volume_mean_20",
    "prior_feature_momentum_12_1",
    "prior_feature_yang_zhang_vol_20",
)


def _validate_paired_identity(
    candidate_report: Mapping[str, object],
    baseline_report: Mapping[str, object],
) -> None:
    """Require both reports to describe the same paired evaluation population."""
    for label, report in (
        ("candidate", candidate_report),
        ("baseline", baseline_report),
    ):
        if report.get("schema") != "BRAZIL_RV_V2_EVALUATION_V2":
            raise ValueError(f"{label} is not a v2 evaluation report")
        input_hashes = report.get("input_hashes")
        if not isinstance(input_hashes, Mapping) or any(
            not isinstance(input_hashes.get(key), str) for key in _PAIRED_INPUT_KEYS
        ):
            raise ValueError(f"{label} report lacks paired input identities")
        for key in _PAIRED_INPUT_KEYS:
            digest = input_hashes[key]
            assert isinstance(digest, str)
            _validate_sha256(digest, label=f"{label} {key} identity")
        source_hashes = report.get("source_artifact_hashes")
        if not isinstance(source_hashes, Mapping) or not source_hashes:
            raise ValueError(f"{label} report lacks source artifact identities")
        if any(
            not isinstance(name, str) or not name or not isinstance(digest, str)
            for name, digest in source_hashes.items()
        ):
            raise ValueError(f"{label} report has malformed source identities")
        for digest in source_hashes.values():
            assert isinstance(digest, str)
            _validate_sha256(digest, label=f"{label} source identity")
        economics = report.get("economics")
        if not isinstance(economics, Mapping) or not isinstance(
            economics.get("contract"), Mapping
        ):
            raise ValueError(f"{label} report lacks the economics contract")
    candidate_inputs = candidate_report["input_hashes"]
    baseline_inputs = baseline_report["input_hashes"]
    assert isinstance(candidate_inputs, Mapping)
    assert isinstance(baseline_inputs, Mapping)
    mismatched = [
        key
        for key in _PAIRED_INPUT_KEYS
        if candidate_inputs[key] != baseline_inputs[key]
    ]
    if mismatched:
        raise ValueError(
            "paired comparison requires identical dates, targets, masks, adjusted "
            "close, neutralized returns, event marks, target scales, and CDI inputs; "
            "mismatched identities: "
            f"{mismatched}"
        )
    if (
        candidate_report["source_artifact_hashes"]
        != baseline_report["source_artifact_hashes"]
    ):
        raise ValueError("paired comparison requires identical source identities")
    candidate_economics = candidate_report["economics"]
    baseline_economics = baseline_report["economics"]
    assert isinstance(candidate_economics, Mapping)
    assert isinstance(baseline_economics, Mapping)
    if candidate_economics["contract"] != baseline_economics["contract"]:
        raise ValueError("paired comparison requires an identical economics contract")


def write_evaluation_report(path: Path, result: EvaluationResult) -> str:
    return write_json_atomic(path, result.report)
