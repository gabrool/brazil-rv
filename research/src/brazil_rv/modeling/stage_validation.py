from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from brazil_rv.preprocessing.contract import CONTRACT_VERSION

from .contract import (
    EXPECTED_DECISIONS_PER_DATE,
    EXPECTED_SPLIT_DATE_COUNTS,
    EXPECTED_SPLIT_SAMPLE_COUNTS,
    HORIZONS,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    TCN_READOUTS,
)
from .data import feature_store_identity, sample_window_metadata
from .evaluate import load_current_neural_run
from .horizon_diagnostics import PERMUTATION_SEED
from .model import count_trainable_parameters


CONTEXT_FAMILIES = ("wdo", "br_rates", "us_rates")
FROZEN_CANDIDATES = (
    *(f"block_{index}" for index in range(1, 7)),
    "uniform_mean",
    "concatenated",
    "final_post_fusion",
    "incumbent_predictions",
)
GRADIENT_GROUPS = (
    "input_projection",
    *(f"block_{index}" for index in range(1, 7)),
    "slow_projection",
    "peer_adapter",
    "shared_fusion",
    "prediction_head",
)
RESIDUAL_PROBES = ("slow", "selected_peer", "macro_interaction", "combined")
HORIZON_PAIRS = {(30, 60), (30, 120), (60, 120)}
COMPARISON_SEEDS = {
    **{f"horizon_multiscale_vs_final_seed{seed}": seed for seed in (11, 29, 47)},
    "shared_multiscale_vs_final_seed29": 29,
    "final_score_mlp_vs_final_seed29": 29,
    "horizon_multiscale_vs_shared_seed29": 29,
    "horizon_multiscale_vs_final_three_seed": 0,
}
GATE_RUNS = {
    ("shared_multiscale_seed29", 29),
    *((f"horizon_multiscale_seed{seed}", seed) for seed in (11, 29, 47)),
    ("horizon_multiscale_three_seed", 0),
}


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _read_csv(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"Invalid CSV artifact {path}: {error}") from error


def _read_parquet(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ValueError(f"Invalid Parquet artifact {path}: {error}") from error


def _require_keys(value: dict[str, Any], required: set[str], path: Path) -> None:
    missing = required - set(value)
    if missing:
        raise ValueError(f"Artifact {path} is missing keys: {sorted(missing)}")


def _require_columns(frame: pl.DataFrame, required: set[str], path: Path) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Artifact {path} is missing columns: {sorted(missing)}")
    if frame.is_empty():
        raise ValueError(f"Artifact is empty: {path}")


def _require_finite_columns(
    frame: pl.DataFrame, columns: tuple[str, ...], path: Path
) -> None:
    for column in columns:
        series = frame.get_column(column)
        if series.null_count():
            raise ValueError(f"Artifact {path} has null values in {column}")
        try:
            values = series.cast(pl.Float64, strict=True).to_numpy()
        except (TypeError, ValueError, pl.exceptions.PolarsError) as error:
            raise ValueError(
                f"Artifact {path} has non-numeric values in {column}"
            ) from error
        if not np.isfinite(values).all():
            raise ValueError(f"Artifact {path} has non-finite values in {column}")


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} differs: expected {expected!r}, found {actual!r}")


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} is non-finite")
    return result


def _finite_array(value: object, name: str, shape: tuple[int, ...]) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")


def validate_split_contract(rows: pl.DataFrame, split: str) -> dict[str, object]:
    specs = {
        "train": (TRAIN_START, TRAIN_END),
        "validation": (VALIDATION_START, VALIDATION_END),
    }
    if split not in specs:
        raise ValueError(f"Unsupported stage split: {split}")
    if rows.is_empty():
        raise ValueError(f"{split} rows are empty")
    dates = rows.get_column("trade_date")
    if dates.max() >= TEST_START:
        raise ValueError(f"{split} rows include the held-out test period")
    expected_start, expected_end = specs[split]
    _require_equal(f"{split} start", dates.min(), expected_start)
    _require_equal(f"{split} end", dates.max(), expected_end)
    _require_equal(
        f"{split} date count",
        dates.n_unique(),
        EXPECTED_SPLIT_DATE_COUNTS[split],
    )
    _require_equal(
        f"{split} sample count",
        rows.height,
        EXPECTED_SPLIT_SAMPLE_COUNTS[split],
    )
    if rows.get_column("sample_id").n_unique() != rows.height:
        raise ValueError(f"{split} sample identifiers are not unique")
    decisions = rows.group_by("trade_date").agg(pl.col("decision_idx").sort())
    expected = list(range(EXPECTED_DECISIONS_PER_DATE))
    if any(values.to_list() != expected for values in decisions["decision_idx"]):
        raise ValueError(f"{split} dates must contain decisions 0..54 exactly once")
    return sample_window_metadata(rows, split)


def validate_preflight(
    path: Path,
    store: Path,
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
) -> None:
    summary = read_json_object(path)
    _require_keys(
        summary,
        {
            "feature_store",
            "test_accessed",
            "parameter_counts",
            "added_parameter_counts",
            "tap_receptive_field_minutes",
            "readout_modes",
            "split_contract",
        },
        path,
    )
    _require_equal(
        "feature-store identity",
        summary["feature_store"],
        feature_store_identity(store),
    )
    _require_equal(
        "feature-store contract",
        summary["feature_store"]["contract_version"],
        CONTRACT_VERSION,
    )
    _require_equal("preflight test access", summary["test_accessed"], False)
    _require_equal("preflight readouts", summary["readout_modes"], list(TCN_READOUTS))
    _require_equal(
        "tap receptive fields",
        summary["tap_receptive_field_minutes"],
        [15, 35, 75, 155, 315, 635],
    )
    counts = summary["parameter_counts"]
    additions = summary["added_parameter_counts"]
    _require_equal("parameter-count modes", set(counts), set(TCN_READOUTS))
    _require_equal(
        "added parameter counts",
        additions,
        {
            "final": 0,
            "shared_multiscale": 6,
            "horizon_multiscale": 18,
            "final_score_mlp": 17,
        },
    )
    _require_equal(
        "preflight split contract",
        summary["split_contract"],
        {
            "train": validate_split_contract(train_rows, "train"),
            "validation": validate_split_contract(validation_rows, "validation"),
        },
    )


def validate_target_basis(output_dir: Path) -> None:
    summary_path = output_dir / "target_basis_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "train_end",
            "date_count",
            "pooled_target_correlation",
            "complete_case_target_covariance",
            "eigenvalues",
            "eigenvectors_columns",
            "variance_shares",
            "fixed_basis_rows",
            "fixed_basis_variance",
            "raw_return_headline_correlation",
        },
        summary_path,
    )
    _require_equal("target audit train end", summary["train_end"], str(TRAIN_END))
    _require_equal(
        "target audit date count",
        summary["date_count"],
        EXPECTED_SPLIT_DATE_COUNTS["train"],
    )
    for name in (
        "pooled_target_correlation",
        "complete_case_target_covariance",
        "eigenvectors_columns",
        "fixed_basis_rows",
        "raw_return_headline_correlation",
    ):
        _finite_array(summary[name], name, (3, 3))
    for name in ("eigenvalues", "variance_shares", "fixed_basis_variance"):
        _finite_array(summary[name], name, (3,))
    pairs = {(left, right) for left in HORIZONS for right in HORIZONS if left <= right}
    pair_path = output_dir / "target_pairwise.csv"
    pairwise = _read_csv(pair_path)
    _require_columns(
        pairwise,
        {
            "scope",
            "left_horizon",
            "right_horizon",
            "valid_count",
            "covariance",
            "correlation",
        },
        pair_path,
    )
    _require_equal(
        "target scopes",
        set(pairwise["scope"]),
        {"pooled", "equal_date", "equal_decision"},
    )
    for scope in ("pooled", "equal_date", "equal_decision"):
        selected = pairwise.filter(pl.col("scope") == scope)
        _require_equal(
            f"{scope} horizon pairs",
            set(selected.select("left_horizon", "right_horizon").iter_rows()),
            pairs,
        )
    date_path = output_dir / "target_basis_by_date.parquet"
    by_date = _read_parquet(date_path)
    _require_columns(
        by_date,
        {"trade_date", "left_horizon", "right_horizon", "covariance", "correlation"},
        date_path,
    )
    _require_equal(
        "target by-date rows",
        by_date.height,
        EXPECTED_SPLIT_DATE_COUNTS["train"] * len(pairs),
    )
    if by_date["trade_date"].max() > TRAIN_END:
        raise ValueError("Target audit contains post-training dates")
    decision_path = output_dir / "target_basis_by_decision.csv"
    by_decision = _read_csv(decision_path)
    _require_columns(
        by_decision,
        {"decision_idx", "left_horizon", "right_horizon", "covariance", "correlation"},
        decision_path,
    )
    _require_equal(
        "target decision rows",
        by_decision.height,
        EXPECTED_DECISIONS_PER_DATE * len(pairs),
    )
    _require_equal(
        "target decisions",
        set(by_decision["decision_idx"]),
        set(range(EXPECTED_DECISIONS_PER_DATE)),
    )


def validate_frozen_probes(output_dir: Path) -> None:
    table_path = output_dir / "frozen_block_probes.csv"
    frame = _read_csv(table_path)
    _require_columns(
        frame,
        {
            "candidate",
            "horizon_minutes",
            "validation_ic",
            "selected_penalty",
            "feature_dim",
            "valid_count",
            "coefficient_norm",
        },
        table_path,
    )
    _require_equal("frozen candidates", set(frame["candidate"]), set(FROZEN_CANDIDATES))
    _require_equal("frozen row count", frame.height, len(FROZEN_CANDIDATES) * 4)
    for candidate in FROZEN_CANDIDATES:
        _require_equal(
            f"{candidate} horizons",
            set(frame.filter(pl.col("candidate") == candidate)["horizon_minutes"]),
            {*HORIZONS, 0},
        )
    if not np.isfinite(frame["validation_ic"].to_numpy()).all():
        raise ValueError("Frozen-probe validation IC contains non-finite values")
    summary_path = output_dir / "frozen_block_probe_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "best_tap_by_horizon",
            "concatenation_beats_every_individual_tap_all_horizons",
            "concatenated_beats_final_post_fusion_by_horizon",
            "earlier_tap_beats_final_post_fusion_by_horizon",
        },
        summary_path,
    )
    for name in (
        "best_tap_by_horizon",
        "concatenated_beats_final_post_fusion_by_horizon",
        "earlier_tap_beats_final_post_fusion_by_horizon",
    ):
        _require_equal(
            f"{name} horizons", set(summary[name]), {str(value) for value in HORIZONS}
        )


def validate_context_inference(output_dir: Path) -> None:
    modes = {
        "baseline",
        *(
            f"{family}_{suffix}"
            for family in CONTEXT_FAMILIES
            for suffix in ("masked", "permuted")
        ),
    }
    table_path = output_dir / "context_inference_probes.csv"
    frame = _read_csv(table_path)
    _require_columns(
        frame,
        {
            "mode",
            "horizon_minutes",
            "baseline_ic",
            "candidate_ic",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
        },
        table_path,
    )
    _require_equal("context inference modes", set(frame["mode"]), modes)
    _require_equal("context inference row count", frame.height, len(modes) * 4)
    for mode in modes:
        _require_equal(
            f"{mode} horizons",
            set(frame.filter(pl.col("mode") == mode)["horizon_minutes"]),
            {*HORIZONS, 0},
        )
    _require_finite_columns(
        frame,
        (
            "baseline_ic",
            "candidate_ic",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
        ),
        table_path,
    )
    manifest_path = output_dir / "context_permutation_manifest.json"
    manifest = read_json_object(manifest_path)
    _require_keys(
        manifest,
        {"seed", "mapping_sha256", "group_sizes", "self_map_count"},
        manifest_path,
    )
    _require_equal("context permutation seed", manifest["seed"], PERMUTATION_SEED)
    if (
        not isinstance(manifest["mapping_sha256"], str)
        or len(manifest["mapping_sha256"]) != 64
    ):
        raise ValueError("Context permutation digest is invalid")
    _require_equal("context permutation self maps", manifest["self_map_count"], 0)
    group_sizes = manifest["group_sizes"]
    if not isinstance(group_sizes, dict):
        raise ValueError("Context permutation group sizes must be an object")
    _require_equal(
        "context permutation quarters",
        set(group_sizes),
        {"2024Q3", "2024Q4", "2025Q1", "2025Q2"},
    )
    if any(not isinstance(size, int) or size < 2 for size in group_sizes.values()):
        raise ValueError("Context permutation quarters must contain at least two dates")
    _require_equal(
        "context permutation date count",
        sum(group_sizes.values()),
        EXPECTED_SPLIT_DATE_COUNTS["validation"],
    )
    summary_path = output_dir / "context_family_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(summary, {"interpretation", "inference"}, summary_path)
    _require_equal("context summary modes", set(summary["inference"]), modes)


def validate_gradient_audit(output_dir: Path) -> None:
    table_path = output_dir / "horizon_gradient_audit.parquet"
    frame = _read_parquet(table_path)
    _require_columns(
        frame,
        {
            "date_idx",
            "decision_idx",
            "group",
            "left_horizon",
            "right_horizon",
            "cosine",
            "undefined_reason",
            "left_gradient_norm",
            "right_gradient_norm",
        },
        table_path,
    )
    _require_equal("gradient groups", set(frame["group"]), set(GRADIENT_GROUPS))
    _require_equal(
        "gradient horizon pairs",
        set(frame.select("left_horizon", "right_horizon").iter_rows()),
        HORIZON_PAIRS,
    )
    _require_finite_columns(
        frame, ("left_gradient_norm", "right_gradient_norm"), table_path
    )
    summary_path = output_dir / "horizon_gradient_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "train_only",
            "train_end",
            "sample_count",
            "by_group_and_horizon_pair",
            "single_horizon_controls",
        },
        summary_path,
    )
    _require_equal("gradient train-only flag", summary["train_only"], True)
    _require_equal("gradient train end", summary["train_end"], str(TRAIN_END))
    _require_equal("gradient sample count", summary["sample_count"], 20)
    rows = summary["by_group_and_horizon_pair"]
    _require_equal(
        "gradient summary row count",
        len(rows),
        len(GRADIENT_GROUPS) * len(HORIZON_PAIRS),
    )
    _require_equal(
        "gradient summary groups", {row["group"] for row in rows}, set(GRADIENT_GROUPS)
    )


def validate_oof_plan(path: Path, expected: dict[str, object]) -> None:
    plan = read_json_object(path)
    _require_equal("OOF plan", plan, expected)
    _require_equal("OOF block ordering", list(plan["blocks"]), ["B0", "B1", "B2", "B3"])
    previous_end = None
    for block in plan["blocks"].values():
        if block["end"] > str(TRAIN_END):
            raise ValueError("OOF plan exceeds the training period")
        if previous_end is not None and previous_end >= block["start"]:
            raise ValueError("OOF blocks are not strictly chronological")
        previous_end = block["end"]


def validate_oof_residual(output_dir: Path) -> None:
    table_path = output_dir / "oof_residual_probes.csv"
    frame = _read_csv(table_path)
    _require_columns(
        frame,
        {
            "probe",
            "horizon_minutes",
            "residual_prediction_ic",
            "base_ic",
            "base_plus_correction_ic",
            "delta_from_base",
            "selected_penalty",
            "valid_count",
            "feature_dim",
        },
        table_path,
    )
    _require_equal("OOF residual probes", set(frame["probe"]), set(RESIDUAL_PROBES))
    _require_equal(
        "OOF residual row count", frame.height, len(RESIDUAL_PROBES) * len(HORIZONS)
    )
    for probe in RESIDUAL_PROBES:
        _require_equal(
            f"{probe} OOF horizons",
            set(frame.filter(pl.col("probe") == probe)["horizon_minutes"]),
            set(HORIZONS),
        )
    _require_finite_columns(
        frame,
        (
            "residual_prediction_ic",
            "base_ic",
            "base_plus_correction_ic",
            "delta_from_base",
            "selected_penalty",
            "valid_count",
            "feature_dim",
        ),
        table_path,
    )
    summary_path = output_dir / "oof_residual_probe_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {"fit_block", "evaluation_block", "calibration_start", "results"},
        summary_path,
    )
    _require_equal("OOF residual fit block", summary["fit_block"], "B2")
    _require_equal("OOF residual evaluation block", summary["evaluation_block"], "B3")
    _require_equal("OOF residual summary rows", len(summary["results"]), frame.height)


def validate_completed_run(
    run_dir: Path,
    store: Path,
    expected: dict[str, Any],
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json_object(manifest_path)
    _require_keys(
        manifest,
        {
            "status",
            "feature_store",
            "feature_store_identity",
            "split",
            "seed",
            "global_context",
            "training_horizon",
            "selection_horizon",
            "context_family_ablation",
            "training",
            "model",
            "parameter_count",
            "objective",
            "optimizer",
            "sam",
            "best_epoch",
            "epochs_completed",
            "total_run_seconds",
        },
        manifest_path,
    )
    _require_equal("run status", manifest["status"], "completed")
    _require_equal(
        "run feature-store path",
        str(Path(manifest["feature_store"]).resolve()),
        str(store.resolve()),
    )
    _require_equal(
        "run feature-store identity",
        manifest["feature_store_identity"],
        feature_store_identity(store),
    )
    for name in (
        "seed",
        "global_context",
        "training_horizon",
        "selection_horizon",
        "context_family_ablation",
        "parameter_count",
        "objective",
        "optimizer",
        "sam",
    ):
        _require_equal(f"run {name}", manifest[name], expected[name])
    split = manifest["split"]
    _require_equal(
        "run fit name", split.get("training"), expected["fit_window"]["name"]
    )
    _require_equal(
        "run selection name",
        split.get("selection"),
        expected["selection_window"]["name"],
    )
    _require_equal("run fit window", split.get("fit_window"), expected["fit_window"])
    _require_equal(
        "run selection window",
        split.get("selection_window"),
        expected["selection_window"],
    )
    _require_equal("run test access", split.get("test_accessed"), False)
    model_metadata = manifest["model"]
    _require_equal("run model name", model_metadata.get("model_name"), "tcn")
    _require_equal(
        "run TCN settings", model_metadata.get("tcn_settings"), expected["tcn_settings"]
    )
    _require_equal(
        "run peer mode",
        model_metadata.get("peer_features", {}).get("mode"),
        expected["peer_features"],
    )
    _require_equal(
        "run date-replacement policy",
        manifest["training"].get("allow_date_replacement"),
        expected["allow_date_replacement"],
    )

    checkpoint_path = run_dir / "best_checkpoint.pt"
    try:
        raw_checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, EOFError) as error:
        raise ValueError(
            f"Invalid checkpoint artifact {checkpoint_path}: {error}"
        ) from error
    required_checkpoint = {
        "model_name",
        "architecture",
        "tcn_settings",
        "peer_features",
        "optimizer_variant",
        "objective",
        "sam",
        "seed",
        "feature_store",
        "feature_store_identity",
        "global_context",
        "training_horizon",
        "selection_horizon",
        "context_family_ablation",
        "fit_window",
        "selection_window",
        "parameter_count",
        "model_state_dict",
    }
    missing = required_checkpoint - set(raw_checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing stage provenance: {sorted(missing)}")
    checkpoint_expectations = {
        "model_name": "tcn",
        "tcn_settings": expected["tcn_settings"],
        "optimizer_variant": expected["optimizer"],
        "objective": expected["objective"],
        "sam": expected["sam"],
        "seed": expected["seed"],
        "feature_store_identity": feature_store_identity(store),
        "global_context": expected["global_context"],
        "training_horizon": expected["training_horizon"],
        "selection_horizon": expected["selection_horizon"],
        "context_family_ablation": expected["context_family_ablation"],
        "fit_window": expected["fit_window"],
        "selection_window": expected["selection_window"],
        "parameter_count": expected["parameter_count"],
    }
    for name, value in checkpoint_expectations.items():
        _require_equal(f"checkpoint {name}", raw_checkpoint[name], value)
    _require_equal(
        "checkpoint peer mode",
        raw_checkpoint["peer_features"].get("mode"),
        expected["peer_features"],
    )
    _require_equal(
        "checkpoint architecture",
        raw_checkpoint["architecture"],
        expected["architecture"],
    )
    _require_equal(
        "checkpoint feature-store path",
        str(Path(raw_checkpoint["feature_store"]).resolve()),
        str(store.resolve()),
    )
    try:
        model, restored, restored_store = load_current_neural_run(run_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Checkpoint strict reconstruction failed: {error}") from error
    _require_equal("restored feature store", restored_store.resolve(), store.resolve())
    _require_equal(
        "restored parameter count",
        count_trainable_parameters(model),
        expected["parameter_count"],
    )
    _require_equal(
        "restored training horizon",
        restored["training_horizon"],
        expected["training_horizon"],
    )

    metrics_path = run_dir / "validation_metrics.json"
    metrics = read_json_object(metrics_path)
    _require_keys(metrics, {"primary_score", "horizons"}, metrics_path)
    _finite(metrics["primary_score"], "validation primary score")
    _require_equal(
        "validation metric horizons",
        {row["horizon_minutes"] for row in metrics["horizons"]},
        set(HORIZONS),
    )
    for row in metrics["horizons"]:
        _finite(row["mean_daily_spearman_ic"], "validation horizon IC")

    daily_path = run_dir / "validation_daily_metrics.parquet"
    daily = _read_parquet(daily_path)
    _require_columns(daily, {"date_idx", "horizon_minutes", "spearman_ic"}, daily_path)
    _require_equal(
        "validation daily horizons", set(daily["horizon_minutes"]), set(HORIZONS)
    )
    _require_equal(
        "validation daily rows",
        daily.height,
        expected["selection_window"]["date_count"] * len(HORIZONS),
    )
    _require_finite_columns(daily, ("spearman_ic",), daily_path)

    history_path = run_dir / "history.csv"
    history = _read_csv(history_path)
    _require_columns(
        history,
        {
            "epoch",
            "train_objective_loss",
            "validation_objective_loss",
            "validation_primary_ic",
            "optimizer_steps",
        },
        history_path,
    )
    _require_equal("history epoch count", history.height, manifest["epochs_completed"])
    if int(manifest["best_epoch"]) not in set(history["epoch"]):
        raise ValueError("Best epoch is absent from training history")
    _require_finite_columns(
        history,
        (
            "train_objective_loss",
            "validation_objective_loss",
            "validation_primary_ic",
            "optimizer_steps",
        ),
        history_path,
    )
    _finite(manifest["total_run_seconds"], "training duration")


def validate_consolidated(output_dir: Path, arm_names: tuple[str, ...]) -> None:
    run_matrix_path = output_dir / "run_matrix.csv"
    run_matrix = _read_csv(run_matrix_path)
    _require_columns(
        run_matrix,
        {
            "arm",
            "seed",
            "readout",
            "training_horizon",
            "context_family_ablation",
            "validation_ic",
            "ic_30",
            "ic_60",
            "ic_120",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        },
        run_matrix_path,
    )
    _require_equal("run-matrix arms", set(run_matrix["arm"]), set(arm_names))
    _require_equal("run-matrix row count", run_matrix.height, len(arm_names))
    _require_finite_columns(
        run_matrix,
        (
            "validation_ic",
            "ic_30",
            "ic_60",
            "ic_120",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        ),
        run_matrix_path,
    )

    comparison_path = output_dir / "multiscale_comparison.csv"
    comparisons = _read_csv(comparison_path)
    _require_columns(
        comparisons,
        {
            "comparison",
            "seed",
            "horizon_minutes",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
        },
        comparison_path,
    )
    _require_equal(
        "multiscale comparisons", set(comparisons["comparison"]), set(COMPARISON_SEEDS)
    )
    _require_equal(
        "multiscale comparison row count",
        comparisons.height,
        len(COMPARISON_SEEDS) * (len(HORIZONS) + 1),
    )
    for name, seed in COMPARISON_SEEDS.items():
        selected = comparisons.filter(pl.col("comparison") == name)
        _require_equal(
            f"{name} horizons",
            set(selected["horizon_minutes"]),
            {*HORIZONS, 0},
        )
        _require_equal(f"{name} seed", set(selected["seed"]), {seed})
    _require_equal(
        "multiscale comparison keys",
        comparisons.select("comparison", "seed", "horizon_minutes").n_unique(),
        comparisons.height,
    )
    _require_finite_columns(
        comparisons,
        ("delta_ic", "delta_lower_95", "delta_upper_95"),
        comparison_path,
    )

    paired_path = output_dir / "multiscale_paired_daily.parquet"
    paired = _read_parquet(paired_path)
    _require_columns(
        paired,
        {"comparison", "seed", "date_idx", "horizon_minutes", "delta_ic"},
        paired_path,
    )
    _require_equal("paired horizons", set(paired["horizon_minutes"]), set(HORIZONS))
    _require_equal(
        "paired comparisons", set(paired["comparison"]), set(COMPARISON_SEEDS)
    )
    _require_equal(
        "paired keys",
        paired.select("comparison", "seed", "date_idx", "horizon_minutes").n_unique(),
        paired.height,
    )
    paired_group_sizes = []
    for name, seed in COMPARISON_SEEDS.items():
        selected = paired.filter(pl.col("comparison") == name)
        _require_equal(f"paired {name} seed", set(selected["seed"]), {seed})
        _require_equal(
            f"paired {name} horizons",
            set(selected["horizon_minutes"]),
            set(HORIZONS),
        )
        paired_group_sizes.extend(
            selected.group_by("horizon_minutes").len().get_column("len").to_list()
        )
    if not paired_group_sizes or len(set(paired_group_sizes)) != 1:
        raise ValueError("Paired comparison groups have inconsistent date counts")
    _require_finite_columns(paired, ("delta_ic",), paired_path)

    gates_path = output_dir / "multiscale_gate_weights.csv"
    gates = _read_csv(gates_path)
    _require_columns(
        gates,
        {
            "arm",
            "seed",
            "horizon_minutes",
            "block",
            "receptive_field_minutes",
            "weight",
            "entropy",
        },
        gates_path,
    )
    _require_equal("gate blocks", set(gates["block"]), set(range(1, 7)))
    _require_equal("gate horizons", set(gates["horizon_minutes"]), set(HORIZONS))
    _require_equal(
        "gate runs",
        set(gates.select("arm", "seed").iter_rows()),
        GATE_RUNS,
    )
    _require_equal(
        "gate row count",
        gates.height,
        len(GATE_RUNS) * len(HORIZONS) * 6,
    )
    _require_finite_columns(gates, ("weight", "entropy"), gates_path)
    for key, group in gates.group_by("arm", "seed", "horizon_minutes"):
        if group.height != 6 or not np.isclose(group["weight"].sum(), 1.0, atol=1e-6):
            raise ValueError(f"Scale weights are invalid for {key}")
        if not np.isfinite(group["weight"].to_numpy()).all():
            raise ValueError(f"Scale weights are non-finite for {key}")

    context_path = output_dir / "audits" / "context" / "context_training_ablations.csv"
    context = _read_csv(context_path)
    _require_columns(
        context,
        {
            "context_family",
            "horizon_minutes",
            "control_ic",
            "candidate_ic",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
            "worst_horizon_delta",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        },
        context_path,
    )
    _require_equal(
        "context retraining families",
        set(context["context_family"]),
        set(CONTEXT_FAMILIES),
    )
    _require_equal(
        "context retraining row count", context.height, len(CONTEXT_FAMILIES) * 4
    )
    for family in CONTEXT_FAMILIES:
        _require_equal(
            f"{family} retraining horizons",
            set(context.filter(pl.col("context_family") == family)["horizon_minutes"]),
            {*HORIZONS, 0},
        )
    _require_finite_columns(
        context,
        (
            "control_ic",
            "candidate_ic",
            "delta_ic",
            "delta_lower_95",
            "delta_upper_95",
            "worst_horizon_delta",
            "best_epoch",
            "epochs_completed",
            "parameter_count",
            "training_duration_seconds",
        ),
        context_path,
    )

    controls_path = output_dir / "audits" / "gradient" / "single_horizon_controls.csv"
    controls = _read_csv(controls_path)
    _require_columns(
        controls,
        {
            "training_horizon",
            "paired_control_ic",
            "selected_horizon_ic",
            "delta_from_control",
        },
        controls_path,
    )
    _require_equal(
        "single-horizon controls",
        set(controls["training_horizon"].cast(pl.Utf8)),
        {str(value) for value in HORIZONS},
    )
    _require_equal("single-horizon control row count", controls.height, len(HORIZONS))
    _require_finite_columns(
        controls,
        (
            "paired_control_ic",
            "selected_horizon_ic",
            "delta_from_control",
        ),
        controls_path,
    )

    summary_path = output_dir / "stage_summary.json"
    summary = read_json_object(summary_path)
    _require_keys(
        summary,
        {
            "training_run_count",
            "test_accessed",
            "hypotheses",
            "bottleneck_interpretation",
            "promotion",
            "artifacts",
        },
        summary_path,
    )
    _require_equal(
        "stage training-run count", summary["training_run_count"], len(arm_names)
    )
    _require_equal("stage test access", summary["test_accessed"], False)
    _require_equal(
        "stage hypotheses",
        set(summary["hypotheses"]),
        {
            "representation_information_loss",
            "shared_scale_aggregation",
            "horizon_scale_specialization",
            "trained_multiscale_result",
            "score_capacity_control",
            "horizon_conflict",
            "context_source_information",
            "target_structure",
        },
    )
    _require_equal("stage promotion", summary["promotion"], "none")
    markdown = (output_dir / "stage_summary.md").read_text(encoding="utf-8")
    if not markdown.strip() or "Held-out test accessed: no" not in markdown:
        raise ValueError("Stage Markdown summary is incomplete")
