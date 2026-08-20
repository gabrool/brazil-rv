from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

from ..modeling.contract import (
    HORIZONS,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    workspace_path,
)
from ..modeling.data import (
    AUXILIARY_IDENTITY_FILES,
    AUXILIARY_TARGET_SCHEMA,
    auxiliary_target_identity,
    feature_store_identity,
    resolve_feature_store,
)
from .contract import (
    BETA_MIN_PAIRED_SESSIONS,
    CONTEXT_SESSION_MINUTES,
    CONTEXT_SESSION_START_MINUTE,
    DECISION_CONTEXT_INDICES,
    EQUITY_SESSION_MINUTES,
    EQUITY_SESSION_START_MINUTE,
    MIN_ACTIVE_EQUITIES,
)
from .io import (
    cotahist_files,
    dense_grid,
    load_assignments,
    load_market_dates_and_security_dates,
    load_source_file,
    prepare_session_bars,
    read_research_interval,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from .transforms import (
    build_causal_features,
    build_equity_features,
    causal_exposure_betas,
    centered_midranks,
)

BETA_TO_WIN_SLOW_INDEX = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_input(store_manifest: dict[str, object], name: str) -> Path:
    inputs = store_manifest.get("canonical_inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get(name), dict):
        raise ValueError(f"Feature store does not record canonical input {name}")
    value = inputs[name].get("resolved_path")
    if not isinstance(value, str):
        raise ValueError(f"Feature store canonical input {name} has no resolved path")
    return workspace_path(value)


def exact_win_returns(
    raw_grid: NDArray[np.float64], observed: NDArray[np.bool_]
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    if raw_grid.ndim != 3 or raw_grid.shape[2] != 5:
        raise ValueError("WIN raw grid must be [date, minute, OHLCV]")
    if observed.shape != raw_grid.shape[:2]:
        raise ValueError("WIN observation mask must align to the raw grid")
    output = np.zeros(
        (raw_grid.shape[0], len(DECISION_CONTEXT_INDICES), len(HORIZONS)),
        dtype=np.float32,
    )
    endpoint_mask = np.zeros(output.shape, dtype=bool)
    for decision_idx, entry_index in enumerate(DECISION_CONTEXT_INDICES):
        entry_observed = observed[:, entry_index]
        entry_open = raw_grid[:, entry_index, 0]
        for horizon_idx, horizon in enumerate(HORIZONS):
            exit_index = entry_index + horizon - 1
            valid = entry_observed & observed[:, exit_index]
            endpoint_mask[:, decision_idx, horizon_idx] = valid
            output[valid, decision_idx, horizon_idx] = np.log(
                raw_grid[valid, exit_index, 3] / entry_open[valid]
            ).astype(np.float32)
    return output, endpoint_mask


def paired_beta_readiness(
    equity_valid: NDArray[np.bool_], factor_valid: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    if equity_valid.ndim != 2 or factor_valid.shape != (equity_valid.shape[0],):
        raise ValueError("Beta validity inputs must share one date axis")
    count = np.zeros(equity_valid.shape[1], dtype=np.int32)
    ready = np.zeros(equity_valid.shape, dtype=bool)
    for date_idx in range(equity_valid.shape[0]):
        ready[date_idx] = count >= BETA_MIN_PAIRED_SESSIONS
        count += equity_valid[date_idx] & factor_valid[date_idx]
    return ready


def _validate_target_parity(store: Path, scale: NDArray[np.float64]) -> None:
    raw = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    medians = np.load(
        store / "cross_section_median.npy", mmap_mode="r", allow_pickle=False
    )
    for date_idx in range(scale.shape[0]):
        for decision_idx in range(raw.shape[2]):
            for horizon_idx, horizon in enumerate(HORIZONS):
                valid = masks[date_idx, :, decision_idx, horizon_idx]
                if not valid.any():
                    continue
                values = (
                    raw[date_idx, valid, decision_idx, horizon_idx].astype(np.float64)
                    - float(medians[date_idx, decision_idx, horizon_idx])
                ) / (scale[date_idx, valid] * np.sqrt(horizon))
                if not np.allclose(
                    centered_midranks(values),
                    targets[date_idx, valid, decision_idx, horizon_idx],
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise ValueError(
                        "Exact target scale disagrees with stored main ranks at "
                        f"date_idx={date_idx}, decision_idx={decision_idx}, "
                        f"horizon={horizon}"
                    )


def _mutation_causality_audit() -> dict[str, bool]:
    raw = np.zeros((1, CONTEXT_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(raw.shape[:2], dtype=bool)
    entry = DECISION_CONTEXT_INDICES[0]
    exit_index = entry + HORIZONS[0] - 1
    raw[0, entry, 0] = 100.0
    raw[0, exit_index, 3] = 101.0
    observed[0, (entry, exit_index)] = True
    baseline, baseline_mask = exact_win_returns(raw, observed)

    after = raw.copy()
    after_observed = observed.copy()
    after[0, exit_index + 1, 3] = 50_000.0
    after_observed[0, exit_index + 1] = True
    post_exit, _ = exact_win_returns(after, after_observed)
    at_exit = raw.copy()
    at_exit[0, exit_index, 3] = 102.0
    changed, _ = exact_win_returns(at_exit, observed)
    missing = observed.copy()
    missing[0, exit_index] = False
    _, missing_mask = exact_win_returns(raw, missing)

    equity_valid = np.ones((BETA_MIN_PAIRED_SESSIONS + 2, 1), dtype=bool)
    factor_valid = np.ones(equity_valid.shape[0], dtype=bool)
    readiness = paired_beta_readiness(equity_valid, factor_valid)
    mutated = factor_valid.copy()
    mutated[BETA_MIN_PAIRED_SESSIONS] = False
    mutated_readiness = paired_beta_readiness(equity_valid, mutated)
    checks = {
        "win_post_exit_mutation_invariant": bool(
            baseline[0, 0, 0] == post_exit[0, 0, 0]
        ),
        "win_exact_exit_mutation_sensitive": bool(
            baseline[0, 0, 0] != changed[0, 0, 0]
        ),
        "missing_win_exit_masks_label": bool(
            baseline_mask[0, 0, 0] and not missing_mask[0, 0, 0]
        ),
        "beta_readiness_emit_before_update": bool(
            not readiness[BETA_MIN_PAIRED_SESSIONS - 1, 0]
            and readiness[BETA_MIN_PAIRED_SESSIONS, 0]
            and readiness[BETA_MIN_PAIRED_SESSIONS, 0]
            == mutated_readiness[BETA_MIN_PAIRED_SESSIONS, 0]
        ),
    }
    if not all(checks.values()):
        raise ValueError("Auxiliary-target mutation causality audit failed")
    return checks


def _build_audit(
    store: Path,
    *,
    date_count: int,
    scale: NDArray[np.float64],
    beta: NDArray[np.float32],
    beta_ready: NDArray[np.bool_],
    win_returns: NDArray[np.float32],
    win_mask: NDArray[np.bool_],
    residual_targets: NDArray[np.float32],
    residual_mask: NDArray[np.bool_],
    sign_targets: NDArray[np.bool_],
    magnitude_targets: NDArray[np.float32],
) -> dict[str, object]:
    date_index = pl.read_parquet(store / "date_index.parquet").head(date_count)
    training_dates = (
        date_index.filter(pl.col("trade_date").is_between(TRAIN_START, TRAIN_END))
        .get_column("date_idx")
        .to_numpy()
    )
    main_targets = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    main_mask = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    raw = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    medians = np.load(
        store / "cross_section_median.npy", mmap_mode="r", allow_pickle=False
    )

    horizons: dict[str, object] = {}
    for horizon_idx, horizon in enumerate(HORIZONS):
        correlations: list[float] = []
        main_count = beta_count = win_count = matched_count = 0
        rank_shift_sum = factor_sq_sum = main_sq_sum = 0.0
        sign_sum = magnitude_sum = 0.0
        for date_idx in training_dates:
            for decision_idx in range(len(DECISION_CONTEXT_INDICES)):
                valid = main_mask[date_idx, :, decision_idx, horizon_idx]
                count = int(valid.sum())
                main_count += count
                beta_count += int((valid & beta_ready[date_idx]).sum())
                if win_mask[date_idx, decision_idx, horizon_idx]:
                    win_count += count
                matched = residual_mask[date_idx, :, decision_idx, horizon_idx]
                matched_count += int(matched.sum())
                sign_sum += float(
                    sign_targets[date_idx, valid, decision_idx, horizon_idx].sum()
                )
                magnitude_sum += float(
                    magnitude_targets[date_idx, valid, decision_idx, horizon_idx].sum()
                )
                if int(matched.sum()) < 2:
                    continue
                residual_rank = residual_targets[
                    date_idx, matched, decision_idx, horizon_idx
                ].astype(np.float64)
                main_rank = main_targets[
                    date_idx, matched, decision_idx, horizon_idx
                ].astype(np.float64)
                correlations.append(float(np.corrcoef(residual_rank, main_rank)[0, 1]))
                rank_shift_sum += float(np.abs(residual_rank - main_rank).sum())
                denominator = scale[date_idx, matched] * np.sqrt(horizon)
                main_z = (
                    raw[date_idx, matched, decision_idx, horizon_idx].astype(np.float64)
                    - float(medians[date_idx, decision_idx, horizon_idx])
                ) / denominator
                factor = (
                    beta[date_idx, matched].astype(np.float64)
                    * float(win_returns[date_idx, decision_idx, horizon_idx])
                    / denominator
                )
                factor_sq_sum += float(np.square(factor).sum())
                main_sq_sum += float(np.square(main_z).sum())
        horizons[str(horizon)] = {
            "main_label_count": main_count,
            "beta_coverage": beta_count / main_count,
            "win_endpoint_coverage": win_count / main_count,
            "matched_residual_coverage": matched_count / main_count,
            "auxiliary_main_rank_correlation": float(np.mean(correlations)),
            "mean_absolute_rank_shift": rank_shift_sum / matched_count,
            "factor_to_main_rms_ratio": np.sqrt(factor_sq_sum / main_sq_sum),
            "sign_positive_fraction": sign_sum / main_count,
            "mean_absolute_normalized_return": magnitude_sum / main_count,
        }
    return {
        "scope": {
            "split": "training",
            "start": TRAIN_START.isoformat(),
            "end": TRAIN_END.isoformat(),
            "date_count": int(training_dates.size),
            "official_validation_evaluated": False,
            "test_accessed": False,
        },
        "horizons": horizons,
        "mutation_causality_checks": _mutation_causality_audit(),
    }


def _close_memmaps(arrays: dict[str, np.memmap]) -> None:
    failure: BaseException | None = None
    for array in arrays.values():
        try:
            array.flush()
        except BaseException as error:
            failure = failure or error
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            try:
                mapping.close()
            except BaseException as error:
                failure = failure or error
    if failure is not None:
        raise RuntimeError("Failed to close auxiliary-target memmaps") from failure


def build_auxiliary_target_sidecar(store: Path, output_dir: Path) -> Path:
    store_identity = feature_store_identity(store)
    if output_dir.exists():
        auxiliary_target_identity(output_dir, store_identity)
        return output_dir

    store_manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assignments_dir = _recorded_input(store_manifest, "accepted_xp_assignments")
    cotahist_dir = _recorded_input(store_manifest, "parsed_cotahist")
    universe_dir = _recorded_input(store_manifest, "point_in_time_universe")
    research_start, research_end = read_research_interval(universe_dir)
    through = min(research_end, VALIDATION_END)
    assignments = load_assignments(assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    market_dates, assignment_dates = load_market_dates_and_security_dates(
        cotahist_files(cotahist_dir),
        security_ids,
        research_start,
        through,
        allow_empty_security_dates=True,
    )
    validate_source_date_isolation(assignments, assignment_dates)
    date_index = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= through)
        .sort("date_idx")
    )
    if tuple(date_index["trade_date"]) != market_dates or not np.array_equal(
        date_index["date_idx"].to_numpy(), np.arange(len(market_dates))
    ):
        raise ValueError("Feature-store dates differ from recorded source dates")
    equity_index = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    if tuple(equity_index["security_id"]) != security_ids:
        raise ValueError("Feature-store equity identity differs from assignments")

    date_count = len(market_dates)
    equity_count = len(security_ids)
    decisions = len(DECISION_CONTEXT_INDICES)
    horizons = len(HORIZONS)
    partial = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    partial.mkdir(parents=True)
    specs = {
        "target_scale.npy": (np.float64, (date_count, equity_count)),
        "beta_to_win.npy": (np.float32, (date_count, equity_count)),
        "beta_ready.npy": (bool, (date_count, equity_count)),
        "win_returns.npy": (np.float32, (date_count, decisions, horizons)),
        "win_endpoint_mask.npy": (bool, (date_count, decisions, horizons)),
        "residual_targets.npy": (
            np.float32,
            (date_count, equity_count, decisions, horizons),
        ),
        "residual_mask.npy": (bool, (date_count, equity_count, decisions, horizons)),
        "sign_targets.npy": (bool, (date_count, equity_count, decisions, horizons)),
        "magnitude_targets.npy": (
            np.float32,
            (date_count, equity_count, decisions, horizons),
        ),
    }
    arrays = {
        name: open_memmap(partial / name, mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in specs.items()
    }
    for array in arrays.values():
        array[...] = 0

    equity_change = np.zeros((date_count, equity_count), dtype=np.float64)
    equity_change_valid = np.zeros((date_count, equity_count), dtype=bool)
    slot_by_security = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    try:
        groups = assignments.partition_by("source_file", maintain_order=True)
        for source_number, group in enumerate(groups, start=1):
            source_path = Path(group.item(0, "source_file"))
            source = load_source_file(source_path)
            validate_physical_source_identity(group, source, source_path)
            allowed_dates = frozenset().union(
                *(assignment_dates[value] for value in group["security_id"])
            )
            session_bars = prepare_session_bars(
                source,
                source_path,
                allowed_dates,
                market_dates,
                EQUITY_SESSION_START_MINUTE,
                EQUITY_SESSION_MINUTES,
            )
            for assignment in group.iter_rows(named=True):
                security_id = assignment["security_id"]
                bars = session_bars.filter(
                    pl.col("trade_date").is_in(tuple(assignment_dates[security_id]))
                )
                raw_grid, observed = dense_grid(
                    bars, date_count, EQUITY_SESSION_MINUTES
                )
                identity_day = np.fromiter(
                    (
                        assignment["first_overlap_date"]
                        <= trade_date
                        <= assignment["last_overlap_date"]
                        for trade_date in market_dates
                    ),
                    dtype=bool,
                    count=date_count,
                )
                result = build_equity_features(
                    raw_grid,
                    observed,
                    identity_day,
                    market_dates=market_dates,
                )
                slot = slot_by_security[security_id]
                arrays["target_scale.npy"][:, slot] = result.sigma
                equity_change[:, slot] = result.daily_change
                equity_change_valid[:, slot] = result.daily_change_valid
            if source_number % 20 == 0 or source_number == len(groups):
                print(f"Built auxiliary equity state {source_number}/{len(groups)}")

        context_index = pl.read_parquet(store / "context_index.parquet")
        win_row = context_index.filter(pl.col("symbol") == "WIN$")
        if win_row.height != 1:
            raise ValueError("Feature store must contain exactly one WIN$ context")
        win_path = workspace_path(win_row.item(0, "source_file"))
        win_source = load_source_file(win_path)
        win_bars = prepare_session_bars(
            win_source,
            win_path,
            frozenset(market_dates),
            market_dates,
            CONTEXT_SESSION_START_MINUTE,
            CONTEXT_SESSION_MINUTES,
        )
        win_grid, win_observed = dense_grid(
            win_bars, date_count, CONTEXT_SESSION_MINUTES
        )
        win_returns, win_endpoint_mask = exact_win_returns(win_grid, win_observed)
        arrays["win_returns.npy"][...] = win_returns
        arrays["win_endpoint_mask.npy"][...] = win_endpoint_mask
        win_features = build_causal_features(
            win_grid,
            win_observed,
            np.ones(date_count, dtype=bool),
            is_rate=False,
            market_dates=market_dates,
            include_dollar_volume=False,
        )
        reconstructed_beta = causal_exposure_betas(
            equity_change,
            equity_change_valid,
            win_features.daily_change[:, None],
            win_features.daily_change_valid[:, None],
        )[:, :, 0]
        equity_ready = np.load(
            store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False
        )[:date_count]
        reconstructed_beta *= equity_ready
        stored_beta = np.asarray(
            np.load(store / "equity_slow.npy", mmap_mode="r", allow_pickle=False)[
                :date_count, :, BETA_TO_WIN_SLOW_INDEX
            ],
            dtype=np.float32,
        )
        if not np.allclose(reconstructed_beta, stored_beta, rtol=0.0, atol=1e-6):
            raise ValueError(
                "Stored beta_to_WIN differs from exact causal reconstruction"
            )
        readiness = paired_beta_readiness(
            equity_change_valid, win_features.daily_change_valid
        )
        readiness &= equity_ready
        arrays["beta_to_win.npy"][...] = stored_beta
        arrays["beta_ready.npy"][...] = readiness

        scale = arrays["target_scale.npy"]
        required = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)[
            :date_count
        ].any(axis=(2, 3))
        if np.any(scale[required] <= 0.0) or not np.isfinite(scale).all():
            raise ValueError(
                "Valid labels require finite positive causal target scales"
            )
        _validate_target_parity(store, scale)

        raw_returns = np.load(
            store / "raw_returns.npy", mmap_mode="r", allow_pickle=False
        )
        label_mask = np.load(
            store / "label_mask.npy", mmap_mode="r", allow_pickle=False
        )
        medians = np.load(
            store / "cross_section_median.npy", mmap_mode="r", allow_pickle=False
        )
        for date_idx in range(date_count):
            for decision_idx in range(decisions):
                for horizon_idx, horizon in enumerate(HORIZONS):
                    valid = label_mask[date_idx, :, decision_idx, horizon_idx]
                    if not valid.any():
                        continue
                    centered = raw_returns[
                        date_idx, valid, decision_idx, horizon_idx
                    ].astype(np.float64) - float(
                        medians[date_idx, decision_idx, horizon_idx]
                    )
                    normalized = centered / (scale[date_idx, valid] * np.sqrt(horizon))
                    arrays["sign_targets.npy"][
                        date_idx, valid, decision_idx, horizon_idx
                    ] = centered > 0.0
                    arrays["magnitude_targets.npy"][
                        date_idx, valid, decision_idx, horizon_idx
                    ] = np.abs(normalized).astype(np.float32)

                    matched = valid & readiness[date_idx]
                    if not win_endpoint_mask[date_idx, decision_idx, horizon_idx]:
                        continue
                    if int(matched.sum()) < MIN_ACTIVE_EQUITIES:
                        continue
                    residual = raw_returns[
                        date_idx, matched, decision_idx, horizon_idx
                    ].astype(np.float64) - stored_beta[date_idx, matched].astype(
                        np.float64
                    ) * float(win_returns[date_idx, decision_idx, horizon_idx])
                    residual -= np.median(residual)
                    residual /= scale[date_idx, matched] * np.sqrt(horizon)
                    arrays["residual_targets.npy"][
                        date_idx, matched, decision_idx, horizon_idx
                    ] = centered_midranks(residual)
                    arrays["residual_mask.npy"][
                        date_idx, matched, decision_idx, horizon_idx
                    ] = True

        audit = _build_audit(
            store,
            date_count=date_count,
            scale=arrays["target_scale.npy"],
            beta=arrays["beta_to_win.npy"],
            beta_ready=arrays["beta_ready.npy"],
            win_returns=arrays["win_returns.npy"],
            win_mask=arrays["win_endpoint_mask.npy"],
            residual_targets=arrays["residual_targets.npy"],
            residual_mask=arrays["residual_mask.npy"],
            sign_targets=arrays["sign_targets.npy"],
            magnitude_targets=arrays["magnitude_targets.npy"],
        )
        (partial / "audit.json").write_text(
            json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
        )
        _close_memmaps(arrays)
        arrays.clear()
        array_hashes = {
            name: _sha256(partial / name) for name in AUXILIARY_IDENTITY_FILES
        }
        manifest = {
            "schema": AUXILIARY_TARGET_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_feature_store": store_identity,
            "through": through.isoformat(),
            "test_accessed": False,
            "official_validation_evaluated": False,
            "audit_scope": "training",
            "array_sha256": array_hashes,
            "array_shapes": {name: list(shape) for name, (_, shape) in specs.items()},
            "construction": {
                "beta": "stored causal beta_to_WIN before loader neutralization",
                "beta_readiness": (
                    "20 prior paired completed sessions; emit before current update"
                ),
                "factor_return": "WIN open[T] to exact close[T+h-1]",
                "residual": "equity_return - beta_to_WIN * WIN_return",
                "residual_target": (
                    "residual cross-sectional median, causal equity sigma*sqrt(h), "
                    "average-tie centered midrank"
                ),
                "sign_target": "equity return minus main cross-sectional median > 0",
                "magnitude_target": (
                    "absolute main median-centered return/(causal sigma*sqrt(h))"
                ),
            },
            "audit_file_sha256": _sha256(partial / "audit.json"),
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output_dir)
    except BaseException:
        if arrays:
            _close_memmaps(arrays)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    auxiliary_target_identity(output_dir, store_identity)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and audit the immutable Phase B auxiliary-target sidecar"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    print(
        build_auxiliary_target_sidecar(resolve_feature_store(), parse_args().output_dir)
    )


if __name__ == "__main__":
    main()
