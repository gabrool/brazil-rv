from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from brazil_rv.modeling.contract import (
    FEATURE_STORE_POINTER,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.modeling.data import feature_store_identity, resolve_feature_store

from .analyze_preprocessing import AUDIT_SEED, OUTPUT_FILES, AuditArrays, AuditDates
from .contract import (
    ASSIGNMENTS_POINTER,
    CATALOGUE_PATH,
    CONTEXT_POINTER,
    COTAHIST_POINTER,
    DECISION_GLOBAL_INDICES,
    DYNAMIC_CHANNELS,
    GLOBAL_CONTEXT_SYMBOLS,
)
from .io import resolve_pointer
from .preprocessing_audit_di import (
    DIInputs,
    load_equity_causal_state,
    run_di_audit,
)
from .preprocessing_audit_features import (
    build_samples,
    largest_material_shifts,
    run_normalization_audit,
    run_redundancy_and_shift,
)
from .preprocessing_audit_target import run_target_audit


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def _csv_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    normalized: list[dict[str, object]] = []
    for row in rows:
        normalized.append(
            {
                key: (
                    json.dumps(_json_safe(value), separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else _json_safe(value)
                )
                for key, value in row.items()
            }
        )
    return pl.from_dicts(normalized, infer_schema_length=None, strict=False)


def atomic_output_directory(output_dir: Path, writer: Callable[[Path], None]) -> Path:
    """Publish one complete immutable audit directory or nothing."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Audit output already exists: {output_dir}")
    partial = output_dir.with_name(f"{output_dir.name}.{uuid4().hex}.partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        writer(partial)
        missing = [name for name in OUTPUT_FILES if not (partial / name).is_file()]
        if missing:
            raise ValueError(f"Audit output is incomplete: {missing}")
        os.replace(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return output_dir


def _resolve_store(override: Path | None) -> Path:
    if override is None:
        return resolve_feature_store(FEATURE_STORE_POINTER)
    store = override.expanduser().resolve()
    if not store.is_dir():
        raise FileNotFoundError(f"Explicit feature store does not exist: {store}")
    return store


def _resolve_inputs(args: argparse.Namespace, manifest: dict[str, object]) -> DIInputs:
    canonical = manifest["canonical_inputs"]

    def directory(override: Path | None, pointer: Path) -> Path:
        return (
            override.expanduser().resolve()
            if override is not None
            else resolve_pointer(pointer)
        )

    inputs = DIInputs(
        context_dir=directory(args.context_dir, CONTEXT_POINTER),
        catalogue_path=(
            args.catalogue_path.expanduser().resolve()
            if args.catalogue_path is not None
            else CATALOGUE_PATH.resolve()
        ),
        assignments_dir=directory(args.assignments_dir, ASSIGNMENTS_POINTER),
        cotahist_dir=directory(args.cotahist_dir, COTAHIST_POINTER),
    )
    for value in (
        inputs.context_dir,
        inputs.assignments_dir,
        inputs.cotahist_dir,
    ):
        if not value.is_dir():
            raise FileNotFoundError(value)
    if not inputs.catalogue_path.is_file():
        raise FileNotFoundError(inputs.catalogue_path)
    if args.context_dir is None and str(inputs.context_dir) != str(
        canonical["xp_context_archive"]["resolved_path"]
    ):
        raise ValueError(
            "Canonical context pointer disagrees with the feature manifest"
        )
    if args.assignments_dir is None and str(inputs.assignments_dir) != str(
        canonical["accepted_xp_assignments"]["resolved_path"]
    ):
        raise ValueError(
            "Canonical assignment pointer disagrees with the feature manifest"
        )
    if args.cotahist_dir is None and str(inputs.cotahist_dir) != str(
        canonical["parsed_cotahist"]["resolved_path"]
    ):
        raise ValueError(
            "Canonical COTAHIST pointer disagrees with the feature manifest"
        )
    return inputs


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _global_mapping_changes(store: Path, dates: AuditDates) -> np.ndarray:
    mapping_changed = np.zeros(
        (
            len(dates.trade_dates),
            len(GLOBAL_CONTEXT_SYMBOLS),
            max(DECISION_GLOBAL_INDICES),
        ),
        dtype=bool,
    )
    rows = (
        pl.scan_parquet(store / "global_context_index.parquet")
        .filter(
            pl.col("mapping_changed"),
            pl.col("date_idx") <= int(dates.validation[-1]),
            pl.col("minute_idx") < max(DECISION_GLOBAL_INDICES),
        )
        .select("date_idx", "global_slot", "minute_idx")
        .collect()
    )
    if rows.height:
        mapping_changed[
            rows["date_idx"].to_numpy(),
            rows["global_slot"].to_numpy(),
            rows["minute_idx"].to_numpy(),
        ] = True
    return mapping_changed


def _normalization_findings(
    rows: list[dict[str, object]], security_rows: list[dict[str, object]]
) -> list[str]:
    overall = [
        row
        for row in rows
        if row["scope_kind"] == "overall"
        and row["entity_kind"] == "equity"
        and row["feature"] in DYNAMIC_CHANNELS[:5]
        and row.get("std") is not None
    ]
    findings: list[str] = []
    if overall:
        clipped = max(
            overall,
            key=lambda row: (
                float(row.get("lower_clipping_fraction") or 0.0)
                + float(row.get("upper_clipping_fraction") or 0.0)
            ),
        )
        findings.append(
            f"{clipped['feature']} has the largest equity base-channel clipping rate "
            f"({float(clipped.get('lower_clipping_fraction') or 0.0) + float(clipped.get('upper_clipping_fraction') or 0.0):.3%})."
        )
        dispersed = max(overall, key=lambda row: float(row["std"]))
        centered = max(overall, key=lambda row: abs(float(row["median"])))
        yearly = [
            row
            for row in rows
            if row["entity_kind"] == "equity"
            and row["scope_kind"] == "year"
            and row["feature"] == "close_move_normalized"
            and row.get("std") is not None
        ]
        intraday = [
            row
            for row in rows
            if row["entity_kind"] == "equity"
            and row["scope_kind"] == "time_bin_30m"
            and row["feature"] == "close_move_normalized"
            and row.get("std") is not None
        ]
        stability = ""
        if yearly and intraday:
            stability = (
                " Close-move std ranges "
                f"{min(float(row['std']) for row in yearly):.3f}-"
                f"{max(float(row['std']) for row in yearly):.3f} by year and "
                f"{min(float(row['std']) for row in intraday):.3f}-"
                f"{max(float(row['std']) for row in intraday):.3f} by session bin."
            )
        findings.append(
            f"Equity base-channel dispersion is widest for {dispersed['feature']} "
            f"(std {float(dispersed['std']):.3f}); the largest median offset is "
            f"{centered['feature']} ({float(centered['median']):.3f}).{stability}"
        )
    dispersion = [
        row
        for row in security_rows
        if row.get("row_type") == "cross_security_summary"
        and row.get("summary_metric") == "std"
    ]
    by_feature: dict[str, dict[str, float]] = {}
    for row in dispersion:
        by_feature.setdefault(str(row["feature"]), {})[
            str(row["summary_percentile"])
        ] = float(row["summary_value"])
    if by_feature:
        feature, values = max(
            by_feature.items(),
            key=lambda item: item[1].get("p90", 0.0) - item[1].get("p10", 0.0),
        )
        candidates = [
            row
            for row in security_rows
            if row.get("row_type") == "security"
            and row.get("feature") == feature
            and row.get("std") is not None
        ]
        outlier = max(
            candidates,
            key=lambda row: abs(float(row["std"]) - values.get("median", 0.0)),
        )
        findings.append(
            f"Cross-security dispersion varies most for {feature}: security-level "
            f"std p10/median/p90 is {values.get('p10', float('nan')):.3f}/"
            f"{values.get('median', float('nan')):.3f}/{values.get('p90', float('nan')):.3f}; "
            f"the furthest security is {outlier['latest_ticker']} ({outlier['security_id']}, "
            f"std {float(outlier['std']):.3f})."
        )
    volume_bins = [
        row
        for row in rows
        if row["entity_kind"] == "equity"
        and row["scope_kind"] == "time_bin_30m"
        and row["feature"] == "volume_surprise"
        and row.get("median") is not None
    ]
    if volume_bins:
        worst = max(volume_bins, key=lambda row: abs(float(row["median"])))
        findings.append(
            f"The largest session-bin volume-surprise median offset is "
            f"{float(worst['median']):.3f} in bin {worst['scope_value']}."
        )
    contexts = [
        row
        for row in rows
        if row["scope_kind"] == "context_symbol"
        and row["feature"] == "close_move_normalized"
        and row.get("std") is not None
    ]
    if contexts:
        high = max(contexts, key=lambda row: float(row["std"]))
        low = min(contexts, key=lambda row: float(row["std"]))
        findings.append(
            f"Context close-move dispersion ranges from {low['scope_value']} "
            f"({float(low['std']):.3f}) to {high['scope_value']} ({float(high['std']):.3f})."
        )
    return findings[:5]


def _target_findings(target: dict[str, object]) -> list[str]:
    parity = target["target_parity"]
    horizon_rows = target["horizon_summary"]
    overall_prerank = [
        row
        for row in target["distribution_rows"]
        if row["stage"] == "prerank_scaled_residual" and row["scope_kind"] == "overall"
    ]
    year_dispersion = [
        float(row["mean_prerank_std"])
        for row in target["coverage_by_training_year"]
        if row.get("mean_prerank_std") is not None
    ]
    time_dispersion = [
        float(row["mean_prerank_std"])
        for row in target["coverage_by_decision_time_bin_30m"]
        if row.get("mean_prerank_std") is not None
    ]
    worst_coverage = target["worst_security_coverage"][0]
    worst_dispersion = target["worst_security_prerank_dispersion"][0]
    worst_time = min(
        target["coverage_by_decision_time_bin_30m"],
        key=lambda row: float(row["valid_label_fraction"]),
    )
    return [
        f"Target reconstruction reproduced {int(parity['checked_value_count']):,} stored values "
        f"with {int(parity['mismatch_count'])} mismatches (max error "
        f"{float(parity['maximum_absolute_error']):.3g}), confirming the implemented "
        "residual, causal-volatility, sqrt(horizon), and midrank stages.",
        f"Overall pre-rank std spans {min(float(row['std']) for row in overall_prerank):.3f}-"
        f"{max(float(row['std']) for row in overall_prerank):.3f} across horizons; mean "
        f"date/decision dispersion spans {min(year_dispersion):.3f}-{max(year_dispersion):.3f} "
        f"by training year and {min(time_dispersion):.3f}-{max(time_dispersion):.3f} by session bin.",
        f"The lowest covered security is {worst_coverage['latest_ticker']} "
        f"({worst_coverage['security_id']}, {float(worst_coverage['valid_label_fraction']):.3%}); "
        f"the most atypical security pre-rank std is {worst_dispersion['latest_ticker']} "
        f"({float(worst_dispersion['prerank_std']):.3f}).",
        f"Across horizons, tie rates span {min(float(row['tie_fraction'] or 0.0) for row in horizon_rows):.3%}-"
        f"{max(float(row['tie_fraction'] or 0.0) for row in horizon_rows):.3%}, degenerate "
        f"cross-sections {min(float(row['degenerate_cross_section_fraction'] or 0.0) for row in horizon_rows):.3%}-"
        f"{max(float(row['degenerate_cross_section_fraction'] or 0.0) for row in horizon_rows):.3%}, "
        f"and below-minimum cross-sections {min(float(row['below_minimum_cross_section_fraction'] or 0.0) for row in horizon_rows):.3%}-"
        f"{max(float(row['below_minimum_cross_section_fraction'] or 0.0) for row in horizon_rows):.3%}.",
        f"The weakest decision-time coverage is bin {worst_time['decision_time_bin_30m']} at "
        f"{float(worst_time['valid_label_fraction']):.3%}; causal-volatility inversion "
        f"encountered {int(target['vol_regime_values_at_clip_used_for_reconstruction'])} clipped sigma states.",
    ]


def build_summary(
    normalization_rows: list[dict[str, object]],
    security_rows: list[dict[str, object]],
    target: dict[str, object],
    pairs: list[dict[str, object]],
    pca_rows: list[dict[str, object]],
    shifts: list[dict[str, object]],
    di: dict[str, object],
) -> dict[str, object]:
    top_shifts = largest_material_shifts(shifts)
    shifts_by_entity = {
        entity: largest_material_shifts(
            [row for row in shifts if row["entity_kind"] == entity], count=5
        )
        for entity in ("equity", "local", "global")
    }
    predominant_shift_components: dict[str, int] = {}
    for row in top_shifts:
        component = str(row.get("dominant_shift_component") or "unclassified")
        predominant_shift_components[component] = (
            predominant_shift_components.get(component, 0) + 1
        )
    compressible = sorted(
        [row for row in pca_rows if row.get("components_95") is not None],
        key=lambda row: float(row["components_95"]) / max(int(row["feature_count"]), 1),
    )[:5]
    redundancy = [
        {
            "entity_kind": row["entity_kind"],
            "feature_kind": row["feature_kind"],
            "feature_left": row["feature_left"],
            "feature_right": row["feature_right"],
            "spearman_rho": row["spearman_rho"],
            "semantic_caution": "High correlation alone does not erase distinct causal semantics.",
        }
        for row in pairs[:10]
    ]
    experiments: list[str] = []
    if di["computability"]["causal_level_tilt_candidate_computable"]:
        experiments.append(
            "Chronological ablation of raw DI contract channels versus the candidate level/tilt representation."
        )
    overall_normalization = [
        row
        for row in normalization_rows
        if row["scope_kind"] == "overall" and row["entity_kind"] == "equity"
    ]
    if overall_normalization:
        worst = max(
            overall_normalization,
            key=lambda row: (
                float(row.get("lower_clipping_fraction") or 0.0)
                + float(row.get("upper_clipping_fraction") or 0.0)
            ),
        )
        experiments.append(
            f"Chronological recalibration of the causal normalization mechanism for {worst['feature']}, the most clipped audited equity channel."
        )
    if pairs:
        experiments.append(
            f"Controlled removal ablation for {pairs[0]['feature_left']} versus {pairs[0]['feature_right']}, preserving both if their semantics matter despite correlation."
        )
    experiments.append(
        "Training-only recalibration study for the largest documented volume-surprise session-bin offset."
    )
    return {
        "normalization_findings": _normalization_findings(
            normalization_rows, security_rows
        ),
        "target_findings": _target_findings(target),
        "redundancy_findings": redundancy,
        "compressible_families": compressible,
        "largest_train_validation_shifts": top_shifts,
        "largest_train_validation_shifts_by_entity": shifts_by_entity,
        "predominant_shift_components": predominant_shift_components,
        "di_verdict": di["verdict"],
        "di_computability": di["computability"],
        "di_fit_quality_diagnostics": di["fit_quality_diagnostics"],
        "di_empirical_usefulness": di["empirical_usefulness"],
        "di_bivariate_contract_beta_alignment": di["bivariate_contract_beta_alignment"],
        "ranked_preprocessing_experiments": experiments,
        "interpretation": {
            "normalization_unit_std_not_assumed_optimal": True,
            "pca_is_diagnostic_only": True,
            "alternative_targets_proposed": False,
            "validation_targets_used": False,
            "held_out_test_used": False,
        },
    }


def summary_markdown(summary: dict[str, object]) -> str:
    lines = ["# Focused preprocessing audit", "", "## Normalization"]
    lines.extend(f"- {value}" for value in summary["normalization_findings"])
    lines.extend(("", "## Target"))
    lines.extend(f"- {value}" for value in summary["target_findings"])
    lines.extend(("", "## Redundancy and shift"))
    if summary["redundancy_findings"]:
        first = summary["redundancy_findings"][0]
        lines.append(
            f"- Strongest duplicate candidate: {first['entity_kind']} "
            f"{first['feature_left']} / {first['feature_right']} "
            f"(rho {float(first['spearman_rho']):.4f}); semantics still require review."
        )
    if summary["compressible_families"]:
        first = summary["compressible_families"][0]
        lines.append(
            f"- Most compressible family: {first['entity_kind']} {first['semantic_family']} "
            f"needs {first['components_95']} of {first['feature_count']} components for 95% variance."
        )
    for row in summary["largest_train_validation_shifts"][:5]:
        lines.append(
            f"- Shift: {row['entity_kind']} {row['feature']} has KS "
            f"{float(row.get('ks_statistic') or 0.0):.3f}, standardized mean shift "
            f"{float(row.get('absolute_standardized_mean_difference') or 0.0):.3f}, "
            f"availability change {float(row.get('observed_fraction_change') or 0.0):+.3f}, "
            f"and dominant evidence in {row.get('dominant_shift_component', 'unclassified')}."
        )
    lines.extend(("", "## DI feasibility", "", str(summary["di_verdict"])))
    lines.extend(("", "## Ranked experiments"))
    lines.extend(
        f"{index}. {value}"
        for index, value in enumerate(
            summary["ranked_preprocessing_experiments"], start=1
        )
    )
    return "\n".join(lines) + "\n"


def run_audit(args: argparse.Namespace) -> Path:
    store = _resolve_store(args.feature_store)
    manifest_path = store / "manifest.json"
    schema_path = store / "feature_schema.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = _resolve_inputs(args, manifest)
    date_index = pl.read_parquet(store / "date_index.parquet")
    equity_index = pl.read_parquet(store / "equity_index.parquet")
    dates = AuditDates.from_frame(date_index)
    arrays = AuditArrays(store, dates)
    global_mapping_changed = _global_mapping_changes(store, dates)

    def write(partial: Path) -> None:
        normalization_rows, security_rows = run_normalization_audit(
            arrays,
            dates,
            equity_index,
            global_mapping_changed=global_mapping_changed,
        )
        equity_state = load_equity_causal_state(inputs, dates, equity_index)
        target, target_coverage, target_security = run_target_audit(
            arrays, dates, equity_index, causal_sigma=equity_state.sigma
        )
        training_samples, validation_samples, sampling = build_samples(
            arrays, dates, global_mapping_changed=global_mapping_changed
        )
        pairwise, pairs, feature_summary, pca_rows, shifts = run_redundancy_and_shift(
            training_samples, validation_samples
        )
        di_coverage, di_fits, di_betas, di, di_access = run_di_audit(
            arrays,
            dates,
            equity_index,
            inputs,
            equity_state=equity_state,
        )
        summary = build_summary(
            normalization_rows,
            security_rows,
            target,
            pairs,
            pca_rows,
            shifts,
            di,
        )
        created = datetime.now(timezone.utc)
        audit_manifest: dict[str, object] = {
            "created_at_utc": created,
            "audit_seed": AUDIT_SEED,
            "repository_commit": _git_commit(),
            "feature_store": {
                **feature_store_identity(store),
                "manifest_sha256": _sha256(manifest_path),
                "feature_schema_sha256": _sha256(schema_path),
                "canonical_inputs": manifest["canonical_inputs"],
            },
            "resolved_inputs": {
                "feature_store_pointer": str(FEATURE_STORE_POINTER),
                "feature_store": str(store),
                "context_dir": str(inputs.context_dir),
                "catalogue_path": str(inputs.catalogue_path),
                "assignments_dir": str(inputs.assignments_dir),
                "cotahist_dir": str(inputs.cotahist_dir),
                **di_access,
            },
            "split_boundaries": {
                "train": [str(TRAIN_START), str(TRAIN_END)],
                "validation": [str(VALIDATION_START), str(VALIDATION_END)],
                "held_out_test": [str(TEST_START), str(TEST_END)],
            },
            "split_access": {
                "training_target_indices": int(dates.train.size),
                "validation_feature_indices": int(dates.validation.size),
                "validation_target_indices": 0,
                "test_indices": 0,
                "validation_features_only": True,
                "targets_training_only": True,
            },
            "sampling": sampling,
            "quantile_sample_capacity": 4096,
            "ks_sampling": "bounded deterministic feature samples recorded above",
            "outputs": list(OUTPUT_FILES),
        }
        _csv_frame(normalization_rows).write_csv(
            partial / "normalization_effectiveness.csv"
        )
        _csv_frame(security_rows).write_csv(
            partial / "normalization_security_summary.csv"
        )
        _write_json(partial / "target_audit.json", target)
        _csv_frame(target_coverage).write_csv(partial / "target_coverage.csv")
        _csv_frame(target_security).write_csv(partial / "target_security_summary.csv")
        _csv_frame(pairs).write_csv(partial / "redundancy_pairs.csv")
        _csv_frame(pairwise).write_csv(partial / "redundancy_pairwise.csv")
        _csv_frame(feature_summary).write_csv(
            partial / "redundancy_feature_summary.csv"
        )
        _csv_frame(pca_rows).write_csv(partial / "redundancy_family_pca.csv")
        _csv_frame(shifts).write_csv(partial / "train_validation_shift.csv")
        _csv_frame(di_coverage).write_csv(partial / "di_contract_coverage.csv")
        _csv_frame(di_fits).write_csv(partial / "di_factor_fit_summary.csv")
        _csv_frame(di_betas).write_csv(partial / "di_factor_beta_summary.csv")
        _write_json(partial / "di_feasibility.json", di)
        _write_json(partial / "preprocessing_audit_summary.json", summary)
        (partial / "preprocessing_audit_summary.md").write_text(
            summary_markdown(summary), encoding="utf-8"
        )
        _write_json(partial / "audit_manifest.json", audit_manifest)

    return atomic_output_directory(args.output_dir, write)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Brazil-RV preprocessing without model training or test access"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-store", type=Path)
    parser.add_argument("--context-dir", type=Path)
    parser.add_argument("--catalogue-path", type=Path)
    parser.add_argument("--assignments-dir", type=Path)
    parser.add_argument("--cotahist-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    output = run_audit(parse_args(argv))
    print(output)


if __name__ == "__main__":
    main()
