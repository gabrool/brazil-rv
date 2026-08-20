from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ..modeling.contract import HORIZONS, TRAIN_END, VALIDATION_END, workspace_path
from ..modeling.data import (
    DI_TILT_SIDECAR_SCHEMA,
    feature_store_identity,
    resolve_feature_store,
)
from .analyze_preprocessing import AuditArrays, AuditDates, causal_factor_betas
from .contract import (
    BETA_MIN_PAIRED_SESSIONS,
    CATALOGUE_PATH,
    CONTEXT_SESSION_MINUTES,
    CONTEXT_SESSION_START_MINUTE,
    DECISION_CONTEXT_INDICES,
    FIXED_RATE_CONTEXT_SYMBOLS,
    MIN_ACTIVE_EQUITIES,
)
from .io import dense_grid, load_source_file, prepare_session_bars
from .preprocessing_audit_di import (
    DIInputs,
    _daily_rate_factors,
    load_equity_causal_state,
    load_rate_grids,
    prepare_equity_causal_scope,
)
from .transforms import build_causal_features, centered_midranks

SHORT_WIN_HALF_LIFE_DAYS = 5
BETA_TO_WIN_SLOW_INDEX = 20
BETA_TO_WDO_SLOW_INDEX = 21
BETA_TO_DI1F28_SLOW_INDEX = 23
D2_VARIANTS = (
    "short_unclipped_win",
    "win_wdo",
    "win_wdo_di_level",
)
D2_TRAINING_GATE = 0.90


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _recorded_input(manifest: Mapping[str, object], name: str) -> Path:
    canonical = manifest.get("canonical_inputs")
    if not isinstance(canonical, Mapping):
        raise ValueError("Feature manifest has no canonical inputs")
    row = canonical.get(name)
    if not isinstance(row, Mapping) or not isinstance(row.get("resolved_path"), str):
        raise ValueError(f"Feature manifest has no resolved canonical input {name}")
    return workspace_path(str(row["resolved_path"]))


def exact_context_returns(
    raw_grid: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    is_rate: bool,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    if raw_grid.ndim != 3 or raw_grid.shape[2] != 5:
        raise ValueError("Context raw grid must be [date, minute, OHLCV]")
    if observed.shape != raw_grid.shape[:2]:
        raise ValueError("Context observation mask does not align")
    output = np.zeros(
        (raw_grid.shape[0], len(DECISION_CONTEXT_INDICES), len(HORIZONS)),
        dtype=np.float32,
    )
    endpoint_mask = np.zeros(output.shape, dtype=bool)
    for decision_idx, entry_idx in enumerate(DECISION_CONTEXT_INDICES):
        entry_valid = observed[:, entry_idx]
        entry_open = raw_grid[:, entry_idx, 0]
        for horizon_idx, horizon in enumerate(HORIZONS):
            exit_idx = entry_idx + horizon - 1
            valid = entry_valid & observed[:, exit_idx]
            endpoint_mask[:, decision_idx, horizon_idx] = valid
            if is_rate:
                values = 100.0 * (raw_grid[valid, exit_idx, 3] - entry_open[valid])
            else:
                values = np.log(raw_grid[valid, exit_idx, 3] / entry_open[valid])
            output[valid, decision_idx, horizon_idx] = values.astype(np.float32)
    return output, endpoint_mask


def paired_beta_readiness(
    equity_valid: NDArray[np.bool_], factor_valid: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    count = np.zeros(equity_valid.shape[1], dtype=np.int32)
    ready = np.zeros(equity_valid.shape, dtype=bool)
    for date_idx in range(equity_valid.shape[0]):
        ready[date_idx] = count >= BETA_MIN_PAIRED_SESSIONS
        count += equity_valid[date_idx] & factor_valid[date_idx]
    return ready


def causal_unclipped_beta(
    equity_change: NDArray[np.float64],
    equity_valid: NDArray[np.bool_],
    factor_change: NDArray[np.float64],
    factor_valid: NDArray[np.bool_],
    *,
    half_life_days: int = SHORT_WIN_HALF_LIFE_DAYS,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """EWMA beta emitted before the current observation, without floor or clip."""
    alpha = 1.0 - 2.0 ** (-1.0 / half_life_days)
    dates, equities = equity_change.shape
    output = np.zeros((dates, equities), dtype=np.float32)
    ready_output = np.zeros((dates, equities), dtype=bool)
    count = np.zeros(equities, dtype=np.int32)
    mean_x = np.zeros(equities, dtype=np.float64)
    mean_y = np.zeros(equities, dtype=np.float64)
    mean_xy = np.zeros(equities, dtype=np.float64)
    mean_y2 = np.zeros(equities, dtype=np.float64)
    for date_idx in range(dates):
        variance = mean_y2 - mean_y**2
        covariance = mean_xy - mean_x * mean_y
        ready = (count >= BETA_MIN_PAIRED_SESSIONS) & (variance > 0.0)
        beta = np.zeros(equities, dtype=np.float64)
        beta[ready] = covariance[ready] / variance[ready]
        finite = ready & np.isfinite(beta)
        output[date_idx, finite] = beta[finite].astype(np.float32)
        ready_output[date_idx] = finite

        paired = equity_valid[date_idx] & factor_valid[date_idx]
        if not paired.any():
            continue
        x = equity_change[date_idx]
        y = np.full(equities, factor_change[date_idx], dtype=np.float64)
        first = paired & (count == 0)
        continuing = paired & ~first
        mean_x[first] = x[first]
        mean_y[first] = y[first]
        mean_xy[first] = x[first] * y[first]
        mean_y2[first] = y[first] ** 2
        mean_x[continuing] = (1.0 - alpha) * mean_x[continuing] + alpha * x[continuing]
        mean_y[continuing] = (1.0 - alpha) * mean_y[continuing] + alpha * y[continuing]
        mean_xy[continuing] = (1.0 - alpha) * mean_xy[continuing] + alpha * x[
            continuing
        ] * y[continuing]
        mean_y2[continuing] = (1.0 - alpha) * mean_y2[continuing] + alpha * y[
            continuing
        ] ** 2
        count[paired] += 1
    return output, ready_output


def _context_grid(
    context_index: pl.DataFrame,
    symbol: str,
    trade_dates: tuple,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    row = context_index.filter(pl.col("symbol") == symbol)
    if row.height != 1:
        raise ValueError(f"Feature store must contain exactly one {symbol} context")
    path = workspace_path(str(row.item(0, "source_file")))
    source = load_source_file(path)
    bars = prepare_session_bars(
        source,
        path,
        frozenset(trade_dates),
        trade_dates,
        CONTEXT_SESSION_START_MINUTE,
        CONTEXT_SESSION_MINUTES,
    )
    return dense_grid(bars, len(trade_dates), CONTEXT_SESSION_MINUTES)


def _nonrate_context(
    context_index: pl.DataFrame,
    symbol: str,
    trade_dates: tuple,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.bool_],
    NDArray[np.float64],
    NDArray[np.bool_],
]:
    grid, observed = _context_grid(context_index, symbol, trade_dates)
    future, endpoints = exact_context_returns(grid, observed, is_rate=False)
    causal = build_causal_features(
        grid,
        observed,
        np.ones(len(trade_dates), dtype=bool),
        is_rate=False,
        market_dates=trade_dates,
        include_dollar_volume=False,
    )
    return future, endpoints, causal.daily_change, causal.daily_change_valid


def _di_level_future(
    context_index: pl.DataFrame, trade_dates: tuple
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    changes = []
    masks = []
    for symbol in FIXED_RATE_CONTEXT_SYMBOLS:
        grid, observed = _context_grid(context_index, symbol, trade_dates)
        change, mask = exact_context_returns(grid, observed, is_rate=True)
        changes.append(change)
        masks.append(mask)
    values = np.stack(changes, axis=0)
    valid = np.stack(masks, axis=0)
    count = valid.sum(axis=0)
    level = np.divide(
        (values * valid).sum(axis=0),
        count,
        out=np.zeros_like(values[0]),
        where=count > 0,
    )
    return level.astype(np.float32), count > 0


def _validate_target_parity(
    store: Path, scale: NDArray[np.float64], date_count: int
) -> None:
    raw = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    medians = np.load(
        store / "cross_section_median.npy", mmap_mode="r", allow_pickle=False
    )
    for date_idx in range(date_count):
        for decision_idx in range(raw.shape[2]):
            for horizon_idx, horizon in enumerate(HORIZONS):
                valid = masks[date_idx, :, decision_idx, horizon_idx]
                if not valid.any():
                    continue
                normalized = (
                    raw[date_idx, valid, decision_idx, horizon_idx].astype(np.float64)
                    - float(medians[date_idx, decision_idx, horizon_idx])
                ) / (scale[date_idx, valid] * np.sqrt(horizon))
                if not np.allclose(
                    centered_midranks(normalized),
                    targets[date_idx, valid, decision_idx, horizon_idx],
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise ValueError("Rebuilt causal target scale fails target parity")


def _residual_variant(
    store: Path,
    dates: AuditDates,
    scale: NDArray[np.float64],
    components: Sequence[
        tuple[
            NDArray[np.float32],
            NDArray[np.bool_],
            NDArray[np.float32],
            NDArray[np.bool_],
        ]
    ],
) -> tuple[NDArray[np.float32], NDArray[np.bool_], dict[str, object]]:
    raw = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    main_targets = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    main_mask = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    medians = np.load(
        store / "cross_section_median.npy", mmap_mode="r", allow_pickle=False
    )
    shape = (len(dates.trade_dates), *raw.shape[1:])
    residual_targets = np.zeros(shape, dtype=np.float32)
    residual_mask = np.zeros(shape, dtype=bool)
    metrics = {
        horizon: {
            "main": 0,
            "matched": 0,
            "beta": 0,
            "endpoint": 0,
            "correlations": [],
            "rank_shift": 0.0,
            "factor_sq": 0.0,
            "main_sq": 0.0,
        }
        for horizon in HORIZONS
    }
    for date_idx in dates.train:
        for decision_idx in range(raw.shape[2]):
            for horizon_idx, horizon in enumerate(HORIZONS):
                valid = np.asarray(
                    main_mask[date_idx, :, decision_idx, horizon_idx], dtype=bool
                )
                beta_ready = valid.copy()
                endpoint_ready = True
                factor = np.zeros(valid.size, dtype=np.float64)
                for beta, ready, future, endpoint in components:
                    beta_ready &= ready[date_idx]
                    endpoint_ready &= bool(
                        endpoint[date_idx, decision_idx, horizon_idx]
                    )
                    factor += beta[date_idx].astype(np.float64) * float(
                        future[date_idx, decision_idx, horizon_idx]
                    )
                matched = beta_ready if endpoint_ready else np.zeros_like(valid)
                if int(matched.sum()) >= MIN_ACTIVE_EQUITIES:
                    residual = (
                        raw[date_idx, matched, decision_idx, horizon_idx].astype(
                            np.float64
                        )
                        - factor[matched]
                    )
                    residual -= np.median(residual)
                    residual /= scale[date_idx, matched] * np.sqrt(horizon)
                    residual_targets[date_idx, matched, decision_idx, horizon_idx] = (
                        centered_midranks(residual)
                    )
                    residual_mask[date_idx, matched, decision_idx, horizon_idx] = True
                row = metrics[horizon]
                row["main"] += int(valid.sum())
                row["beta"] += int(beta_ready.sum())
                if endpoint_ready:
                    row["endpoint"] += int(valid.sum())
                row["matched"] += int(matched.sum())
                if int(matched.sum()) < MIN_ACTIVE_EQUITIES:
                    continue
                residual_rank = residual_targets[
                    date_idx, matched, decision_idx, horizon_idx
                ].astype(np.float64)
                main_rank = main_targets[
                    date_idx, matched, decision_idx, horizon_idx
                ].astype(np.float64)
                correlation = float(np.corrcoef(residual_rank, main_rank)[0, 1])
                if np.isfinite(correlation):
                    row["correlations"].append(correlation)
                row["rank_shift"] += float(np.abs(residual_rank - main_rank).sum())
                denominator = scale[date_idx, matched] * np.sqrt(horizon)
                main_z = (
                    raw[date_idx, matched, decision_idx, horizon_idx].astype(np.float64)
                    - float(medians[date_idx, decision_idx, horizon_idx])
                ) / denominator
                factor_z = factor[matched] / denominator
                row["factor_sq"] += float(np.square(factor_z).sum())
                row["main_sq"] += float(np.square(main_z).sum())

    horizon_audit = {}
    correlations = []
    for horizon, row in metrics.items():
        correlations.extend(row["correlations"])
        horizon_audit[str(horizon)] = {
            "main_label_count": row["main"],
            "beta_coverage": row["beta"] / row["main"],
            "factor_endpoint_coverage": row["endpoint"] / row["main"],
            "matched_residual_coverage": row["matched"] / row["main"],
            "auxiliary_main_rank_correlation": float(np.mean(row["correlations"])),
            "mean_absolute_rank_shift": row["rank_shift"] / row["matched"],
            "factor_to_main_rms_ratio": float(
                np.sqrt(row["factor_sq"] / row["main_sq"])
            ),
        }
    return (
        residual_targets,
        residual_mask,
        {
            "aggregate_auxiliary_main_rank_correlation": float(np.mean(correlations)),
            "horizons": horizon_audit,
        },
    )


def _mutation_checks() -> dict[str, bool]:
    grid = np.zeros((1, CONTEXT_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(grid.shape[:2], dtype=bool)
    entry = DECISION_CONTEXT_INDICES[0]
    exit_idx = entry + HORIZONS[0] - 1
    grid[0, entry, 0] = 100.0
    grid[0, exit_idx, 3] = 101.0
    observed[0, (entry, exit_idx)] = True
    baseline, baseline_mask = exact_context_returns(grid, observed, is_rate=False)
    post = grid.copy()
    post[0, exit_idx + 1, 3] = 1e6
    mutated, _ = exact_context_returns(post, observed, is_rate=False)
    missing = observed.copy()
    missing[0, exit_idx] = False
    _, missing_mask = exact_context_returns(grid, missing, is_rate=False)

    dates = BETA_MIN_PAIRED_SESSIONS + 3
    changes = np.arange(dates, dtype=np.float64)[:, None]
    factors = np.linspace(-0.02, 0.03, dates)
    valid = np.ones_like(changes, dtype=bool)
    factor_valid = np.ones(dates, dtype=bool)
    beta, ready = causal_unclipped_beta(changes, valid, factors, factor_valid)
    changed_factors = factors.copy()
    changed_factors[-1] = 100.0
    changed_beta, _ = causal_unclipped_beta(
        changes, valid, changed_factors, factor_valid
    )
    checks = {
        "post_exit_mutation_invariant": bool(baseline[0, 0, 0] == mutated[0, 0, 0]),
        "missing_exact_exit_masks_factor": bool(
            baseline_mask[0, 0, 0] and not missing_mask[0, 0, 0]
        ),
        "short_beta_emits_before_current_update": bool(
            np.array_equal(beta[-1], changed_beta[-1]) and ready[-1, 0]
        ),
    }
    if not all(checks.values()):
        raise ValueError("Next-stage causality mutation checks failed")
    return checks


def build_next_stage_sidecars(store: Path, output_dir: Path) -> Path:
    store_identity = feature_store_identity(store)
    if output_dir.exists():
        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("schema") != DI_TILT_SIDECAR_SCHEMA
            or manifest.get("source_feature_store") != store_identity
        ):
            raise ValueError("Existing next-stage sidecar has a different contract")
        return output_dir

    store_manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    date_frame = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= VALIDATION_END)
        .sort("date_idx")
    )
    dates = AuditDates.from_frame(date_frame)
    arrays = AuditArrays(store, dates)
    inputs = DIInputs(
        context_dir=_recorded_input(store_manifest, "xp_context_archive"),
        catalogue_path=CATALOGUE_PATH.resolve(),
        assignments_dir=_recorded_input(store_manifest, "accepted_xp_assignments"),
        cotahist_dir=_recorded_input(store_manifest, "parsed_cotahist"),
    )
    equity_index = pl.read_parquet(store / "equity_index.parquet")
    scope = prepare_equity_causal_scope(inputs, dates, equity_index, arrays)
    equity = load_equity_causal_state(inputs, dates, scope)
    date_count = int(dates.validation[-1]) + 1
    training_date_count = int(dates.train[-1]) + 1
    _validate_target_parity(store, equity.sigma, training_date_count)
    equity_ready = np.asarray(
        arrays.array("equity_data_ready.npy")[:date_count], dtype=bool
    )

    context_index = pl.read_parquet(store / "context_index.parquet")
    trade_dates = dates.trade_dates[:training_date_count]
    win_future, win_endpoint, win_change, win_valid = _nonrate_context(
        context_index, "WIN$", trade_dates
    )
    wdo_future, wdo_endpoint, wdo_change, wdo_valid = _nonrate_context(
        context_index, "WDO$", trade_dates
    )
    di_level_future, di_level_endpoint = _di_level_future(context_index, trade_dates)

    slow = np.load(store / "equity_slow.npy", mmap_mode="r", allow_pickle=False)
    beta_win = np.asarray(
        slow[:date_count, :, BETA_TO_WIN_SLOW_INDEX], dtype=np.float32
    )
    beta_wdo = np.asarray(
        slow[:date_count, :, BETA_TO_WDO_SLOW_INDEX], dtype=np.float32
    )
    beta_di = np.asarray(
        slow[:date_count, :, BETA_TO_DI1F28_SLOW_INDEX], dtype=np.float32
    )
    training_equity_ready = equity_ready[:training_date_count]
    ready_win = (
        paired_beta_readiness(equity.change_valid[:training_date_count], win_valid)
        & training_equity_ready
    )
    ready_wdo = (
        paired_beta_readiness(equity.change_valid[:training_date_count], wdo_valid)
        & training_equity_ready
    )

    rate_grids = load_rate_grids(inputs, dates)
    rate_factors, rate_factor_valid, contract_change_valid = _daily_rate_factors(
        rate_grids, dates
    )
    factor_betas, factor_beta_ready = causal_factor_betas(
        equity.change,
        equity.change_valid,
        rate_factors,
        rate_factor_valid,
    )
    tilt_exposure = factor_betas[:, :, 1] * equity_ready
    tilt_ready = factor_beta_ready[:, :, 1] & equity_ready
    ready_di = (
        paired_beta_readiness(equity.change_valid, contract_change_valid[:, 1])
        & equity_ready
    )

    short_beta, short_ready = causal_unclipped_beta(
        equity.change[:training_date_count],
        equity.change_valid[:training_date_count],
        win_change,
        win_valid,
    )
    short_ready &= training_equity_ready
    component_sets = {
        "short_unclipped_win": ((short_beta, short_ready, win_future, win_endpoint),),
        "win_wdo": (
            (beta_win, ready_win, win_future, win_endpoint),
            (beta_wdo, ready_wdo, wdo_future, wdo_endpoint),
        ),
        "win_wdo_di_level": (
            (beta_win, ready_win, win_future, win_endpoint),
            (beta_wdo, ready_wdo, wdo_future, wdo_endpoint),
            (beta_di, ready_di, di_level_future, di_level_endpoint),
        ),
    }
    audits = {}
    residual_arrays = {}
    for variant in D2_VARIANTS:
        targets, mask, audit = _residual_variant(
            store,
            dates,
            equity.sigma,
            component_sets[variant],
        )
        audits[variant] = audit
        residual_arrays[variant] = (targets, mask)
    selected = min(
        D2_VARIANTS,
        key=lambda name: audits[name]["aggregate_auxiliary_main_rank_correlation"],
    )
    selected_correlation = audits[selected]["aggregate_auxiliary_main_rank_correlation"]
    training_gate_passed = bool(selected_correlation <= D2_TRAINING_GATE)

    partial = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    partial.mkdir(parents=True)
    try:
        np.save(
            partial / "tilt_exposure.npy",
            tilt_exposure.astype(np.float32),
            allow_pickle=False,
        )
        np.save(partial / "tilt_ready.npy", tilt_ready, allow_pickle=False)
        if training_gate_passed:
            np.save(
                partial / "residual_targets.npy",
                residual_arrays[selected][0],
                allow_pickle=False,
            )
            np.save(
                partial / "residual_mask.npy",
                residual_arrays[selected][1],
                allow_pickle=False,
            )
        audit = {
            "scope": {
                "training_end": TRAIN_END.isoformat(),
                "official_validation_targets_accessed": False,
                "test_accessed": False,
            },
            "residual_variants": audits,
            "selected_lowest_correlation_variant": selected,
            "selected_correlation": selected_correlation,
            "training_gate": {
                "threshold": D2_TRAINING_GATE,
                "passed": training_gate_passed,
                "rule": (
                    "train exactly one three-seed residual auxiliary candidate "
                    "only when correlation <= 0.90"
                ),
            },
            "di_tilt": {
                "training_ready_fraction": float(
                    tilt_ready[dates.train].sum()
                    / training_equity_ready[dates.train].sum()
                ),
                "coverage_denominator": "active training equity-date cells",
                "source": "causal beta to fitted DI tilt factor",
            },
            "mutation_causality_checks": _mutation_checks(),
        }
        _atomic_json(partial / "audit.json", audit)
        files = ["tilt_exposure.npy", "tilt_ready.npy", "audit.json"]
        if training_gate_passed:
            files.extend(("residual_targets.npy", "residual_mask.npy"))
        hashes = {name: _sha256(partial / name) for name in files}
        manifest = {
            "schema": DI_TILT_SIDECAR_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_feature_store": store_identity,
            "through": VALIDATION_END.isoformat(),
            "test_accessed": False,
            "official_validation_targets_accessed": False,
            "files_sha256": hashes,
            "di_tilt_contract": (
                "causal per-equity beta to the daily fitted DI tilt factor; "
                "20 prior paired sessions; emit before current update"
            ),
            "d2_contract": {
                "target_through": TRAIN_END.isoformat(),
                "short_win_half_life_days": SHORT_WIN_HALF_LIFE_DAYS,
                "short_win_clip": None,
                "short_win_variance_floor": None,
                "di_level_future": (
                    "mean basis-point open[T]-to-close[T+h-1] change over all "
                    "endpoint-ready fixed DI contracts; at least one required"
                ),
                "selected_variant": selected,
                "training_gate_passed": training_gate_passed,
            },
        }
        _atomic_json(partial / "manifest.json", manifest)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit next-stage D2 residual and C3 DI-tilt sidecars"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(build_next_stage_sidecars(resolve_feature_store(), args.output_dir))


if __name__ == "__main__":
    main()
