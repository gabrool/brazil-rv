from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, DECISION_GLOBAL_INDICES
from .data import (
    di_tilt_sidecar_identity,
    feature_store_identity,
    resolve_feature_store,
)
from .engine import EvaluationObservations
from .model import (
    CAPACITY_96_VARIANT,
    COMPRESSED_GLOBAL_RISK_VARIANT,
    COMPETITIVE_FEATURE_GATE_VARIANT,
    DI_TILT_EXPOSURE_VARIANT,
    FACTOR_MIXER_K4_VARIANT,
    FACTOR_MIXER_K8_VARIANT,
    RESIDUAL_AUXILIARY_VARIANT,
    SET_POOL_FACTOR_MIXER_VARIANT,
)
from .next_stage_diagnostics import crossfit_patience_observations
from .provenance import repository_commit
from .train import run_training
from .trajectory import predictions_for_rule

DISCOVERY_FOLDS = ("fold_a", "fold_b")
PHASE_C_READOUTS = ("patience3_raw", "final_ema_0995")
INITIAL_VARIANTS = (
    COMPRESSED_GLOBAL_RISK_VARIANT,
    FACTOR_MIXER_K4_VARIANT,
    DI_TILT_EXPOSURE_VARIANT,
)
C2_EXTENSION_VARIANTS = (
    FACTOR_MIXER_K8_VARIANT,
    SET_POOL_FACTOR_MIXER_VARIANT,
)
NULL_ONLY_VARIANTS = (
    CAPACITY_96_VARIANT,
    COMPETITIVE_FEATURE_GATE_VARIANT,
)
C2_EXTENSION_THRESHOLD = 0.001


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _manifest(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _variant_name(manifest: Mapping[str, object]) -> str:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Run manifest has no model metadata")
    variant = model.get("variant")
    if isinstance(variant, Mapping):
        return str(variant.get("name"))
    return "parent"


def _parent_run(parent_campaign: Path, fold: str, seed: int) -> Path:
    return parent_campaign / fold / f"seed_{seed}"


def _candidate_run(output_dir: Path, variant: str, fold: str, seed: int) -> Path:
    return output_dir / "runs" / variant / fold / f"seed_{seed}"


def _validate_parent_campaign(
    parent_campaign: Path,
    *,
    store: Path,
    identity: Mapping[str, object],
) -> None:
    campaign = json.loads(
        (parent_campaign / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    if (
        campaign.get("status") != "completed"
        or Path(str(campaign.get("feature_store"))).resolve() != store.resolve()
        or campaign.get("feature_store_identity") != identity
        or campaign.get("official_validation_accessed") is not False
        or campaign.get("test_accessed") is not False
    ):
        raise ValueError("Trajectory parent campaign does not match Phase C")
    for fold in DISCOVERY_FOLDS:
        for seed in ALLOWED_SEEDS:
            run = _parent_run(parent_campaign, fold, seed)
            manifest = _manifest(run)
            if (
                manifest.get("status") != "completed"
                or int(manifest.get("seed", -1)) != seed
                or manifest.get("split", {}).get("training") != fold
                or manifest.get("split", {}).get("test_accessed") is not False
            ):
                raise ValueError(f"Trajectory parent run does not match: {run}")


def _completed_candidate_matches(
    run_dir: Path,
    *,
    store: Path,
    identity: Mapping[str, object],
    commit: str,
    variant: str,
    fold: str,
    seed: int,
) -> bool:
    if not (run_dir / "run_manifest.json").is_file():
        return False
    manifest = _manifest(run_dir)
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == commit
        and Path(str(manifest.get("feature_store"))).resolve() == store.resolve()
        and manifest.get("feature_store_identity") == identity
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and _variant_name(manifest) == variant
    )


def _readout_observations(
    run_dir: Path, readout: str
) -> tuple[EvaluationObservations, list[dict[str, object]]]:
    if readout == "patience3_raw":
        return crossfit_patience_observations(run_dir)
    if readout != "final_ema_0995":
        raise ValueError(f"Unsupported Phase C readout: {readout}")
    return (
        replace(
            load_run_observations(run_dir, "final_raw"),
            predictions=predictions_for_rule(run_dir, readout),
        ),
        [],
    )


def _es_volatility_guardrail(store: Path, daily_delta_path: Path) -> dict[str, object]:
    daily = pl.read_parquet(daily_delta_path).sort("date_idx")
    dates = daily["date_idx"].to_numpy()
    features = np.load(store / "global_features.npy", mmap_mode="r", allow_pickle=False)
    readiness = np.load(
        store / "global_data_ready.npy", mmap_mode="r", allow_pickle=False
    )
    states = []
    for date_idx in dates:
        values = []
        for decision_idx, cutoff in enumerate(DECISION_GLOBAL_INDICES):
            if readiness[date_idx, 0, decision_idx]:
                values.append(float(features[date_idx, 0, cutoff - 1, 11]))
        states.append(float(np.mean(values)) if values else float("nan"))
    states = np.asarray(states, dtype=np.float64)
    finite = np.isfinite(states)
    if not finite.any():
        raise ValueError("C1 ES volatility guardrail has no ready dates")
    threshold = float(np.median(states[finite]))
    deltas = daily["candidate_minus_parent_ic"].to_numpy()
    low = finite & (states <= threshold)
    high = finite & (states > threshold)
    return {
        "state": "daily mean causal ES realized_vol_30m_log_ratio at decisions",
        "median_threshold": threshold,
        "low_date_count": int(low.sum()),
        "high_date_count": int(high.sum()),
        "low_vol_candidate_minus_parent_ic": float(np.mean(deltas[low])),
        "high_vol_candidate_minus_parent_ic": float(np.mean(deltas[high])),
    }


def _analyze_variant(
    *,
    store: Path,
    variant: str,
    output_dir: Path,
    parent_campaign: Path,
) -> dict[str, object]:
    summary_path = output_dir / "analysis" / variant / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("variant") != variant:
            raise ValueError(f"Existing Phase C analysis differs: {summary_path}")
        return summary
    readouts = {}
    for readout in PHASE_C_READOUTS:
        folds = {}
        for fold in DISCOVERY_FOLDS:
            candidate_members = {}
            parent_members = {}
            candidate_replays = {}
            parent_replays = {}
            for seed in ALLOWED_SEEDS:
                key = f"seed_{seed}"
                candidate, candidate_replay = _readout_observations(
                    _candidate_run(output_dir, variant, fold, seed), readout
                )
                parent, parent_replay = _readout_observations(
                    _parent_run(parent_campaign, fold, seed), readout
                )
                candidate_members[key] = candidate
                parent_members[key] = parent
                candidate_replays[key] = candidate_replay
                parent_replays[key] = parent_replay
            comparison = output_dir / "analysis" / variant / fold / readout
            compare_observation_ensembles(
                candidate_members,
                parent_members,
                candidate_rule=readout,
                parent_rule=readout,
                output_dir=comparison,
                comparison_metadata={
                    "variant": variant,
                    "fold": fold,
                    "seeds": list(ALLOWED_SEEDS),
                    "adaptive_checkpoint_crossfit": readout == "patience3_raw",
                    "candidate_patience_replays": candidate_replays,
                    "parent_patience_replays": parent_replays,
                    "official_validation_accessed": False,
                    "test_accessed": False,
                },
            )
            report = json.loads(
                (comparison / "analysis.json").read_text(encoding="utf-8")
            )
            fold_summary = {
                "candidate_ensemble_ic": report["candidate"]["ensemble_ic"],
                "parent_ensemble_ic": report["parent"]["ensemble_ic"],
                "candidate_minus_parent_primary_ic": report[
                    "candidate_minus_parent_primary_ic"
                ],
                "per_date_delta_bootstrap": report["per_date_delta_bootstrap"],
                "horizon_guardrails": report["horizon_guardrails"],
                "time_of_day_guardrails": report["time_of_day_guardrails"],
                "analysis": str((comparison / "analysis.json").resolve()),
            }
            if variant == COMPRESSED_GLOBAL_RISK_VARIANT:
                fold_summary["es_volatility_guardrail"] = _es_volatility_guardrail(
                    store, comparison / "daily_delta.parquet"
                )
            folds[fold] = fold_summary
        mean_delta = float(
            np.mean(
                [
                    folds[fold]["candidate_minus_parent_primary_ic"]
                    for fold in DISCOVERY_FOLDS
                ]
            )
        )
        readouts[readout] = {
            "folds": folds,
            "mean_fold_candidate_minus_parent_ic": mean_delta,
            "positive_both_folds": all(
                folds[fold]["candidate_minus_parent_primary_ic"] > 0.0
                for fold in DISCOVERY_FOLDS
            ),
        }
    summary = {
        "schema": "PHASE_C_VARIANT_ANALYSIS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "seeds": list(ALLOWED_SEEDS),
        "readouts": readouts,
        "retained": readouts["patience3_raw"]["positive_both_folds"],
        "retention_rule": "primary patience3_raw paired delta strictly positive on both folds",
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, summary)
    return summary


def c2_extensions_allowed(summary: Mapping[str, object]) -> bool:
    primary = summary["readouts"]["patience3_raw"]
    folds = primary["folds"]
    return bool(
        primary["mean_fold_candidate_minus_parent_ic"] >= C2_EXTENSION_THRESHOLD
        and all(
            folds[fold]["candidate_minus_parent_primary_ic"] >= 0.0
            for fold in DISCOVERY_FOLDS
        )
    )


def _run_candidate(
    *,
    store: Path,
    identity: Mapping[str, object],
    commit: str,
    output_dir: Path,
    variant: str,
    fold: str,
    seed: int,
    tilt_sidecar: Path,
) -> Path:
    run_dir = _candidate_run(output_dir, variant, fold, seed)
    if run_dir.exists():
        if not _completed_candidate_matches(
            run_dir,
            store=store,
            identity=identity,
            commit=commit,
            variant=variant,
            fold=fold,
            seed=seed,
        ):
            raise ValueError(f"Existing Phase C run differs: {run_dir}")
        return run_dir
    return run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        variant=variant,
        tilt_sidecar=(
            tilt_sidecar
            if variant in (DI_TILT_EXPOSURE_VARIANT, RESIDUAL_AUXILIARY_VARIANT)
            else None
        ),
    )


def _run_and_analyze(
    *,
    store: Path,
    identity: Mapping[str, object],
    commit: str,
    output_dir: Path,
    parent_campaign: Path,
    variant: str,
    tilt_sidecar: Path,
) -> dict[str, object]:
    for fold in DISCOVERY_FOLDS:
        for seed in ALLOWED_SEEDS:
            _run_candidate(
                store=store,
                identity=identity,
                commit=commit,
                output_dir=output_dir,
                variant=variant,
                fold=fold,
                seed=seed,
                tilt_sidecar=tilt_sidecar,
            )
    return _analyze_variant(
        store=store,
        variant=variant,
        output_dir=output_dir,
        parent_campaign=parent_campaign,
    )


def run_phase_c_campaign(
    store: Path,
    parent_campaign: Path,
    tilt_sidecar: Path,
    output_dir: Path,
) -> Path:
    commit = repository_commit()
    identity = feature_store_identity(store)
    _validate_parent_campaign(parent_campaign, store=store, identity=identity)
    sidecar_manifest = di_tilt_sidecar_identity(tilt_sidecar, identity)
    sidecar_audit = json.loads(
        (tilt_sidecar / "audit.json").read_text(encoding="utf-8")
    )
    d2_training_gate_passed = bool(sidecar_audit["training_gate"]["passed"])
    if d2_training_gate_passed:
        di_tilt_sidecar_identity(tilt_sidecar, identity, require_residual=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "campaign_manifest.json"
    immutable = {
        "schema": "PHASE_C_CAMPAIGN",
        "repository_commit": commit,
        "feature_store": str(store.resolve()),
        "feature_store_identity": identity,
        "trajectory_parent": str(parent_campaign.resolve()),
        "next_stage_sidecar": str(tilt_sidecar.resolve()),
        "next_stage_sidecar_manifest": sidecar_manifest,
        "d2_training_gate_passed": d2_training_gate_passed,
        "d2_selected_variant": sidecar_audit["selected_lowest_correlation_variant"],
        "initial_variants": list(INITIAL_VARIANTS),
        "c2_extension_variants": list(C2_EXTENSION_VARIANTS),
        "null_only_variants": list(NULL_ONLY_VARIANTS),
        "c2_extension_gate": (
            "K4 primary mean delta >= +0.001 and both fold deltas nonnegative"
        ),
        "null_only_gate": (
            "run capacity and competitive gate only when C1, C2-K4, and C3 "
            "all fail the positive-both-fold retention rule"
        ),
        "seeds": list(ALLOWED_SEEDS),
        "readouts": list(PHASE_C_READOUTS),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    created_at = datetime.now(timezone.utc).isoformat()
    results = {}
    sequence = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise ValueError("Existing Phase C campaign has a different contract")
        created_at = str(existing["created_at"])
        results = dict(existing.get("results", {}))
        sequence = list(existing.get("candidate_sequence", []))
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "status": "running",
            "created_at": created_at,
            "results": results,
            "candidate_sequence": sequence,
        },
    )

    def run(variant: str) -> dict[str, object]:
        summary = _run_and_analyze(
            store=store,
            identity=identity,
            commit=commit,
            output_dir=output_dir,
            parent_campaign=parent_campaign,
            variant=variant,
            tilt_sidecar=tilt_sidecar,
        )
        results[variant] = {
            "retained": summary["retained"],
            "analysis": str(
                (output_dir / "analysis" / variant / "summary.json").resolve()
            ),
        }
        if variant not in sequence:
            sequence.append(variant)
        _atomic_json(
            manifest_path,
            {
                **immutable,
                "status": "running",
                "created_at": created_at,
                "results": results,
                "candidate_sequence": sequence,
            },
        )
        return summary

    if d2_training_gate_passed:
        run(RESIDUAL_AUXILIARY_VARIANT)
    initial = {variant: run(variant) for variant in INITIAL_VARIANTS}
    if c2_extensions_allowed(initial[FACTOR_MIXER_K4_VARIANT]):
        for variant in C2_EXTENSION_VARIANTS:
            run(variant)
    all_initial_null = all(
        not initial[variant]["retained"] for variant in INITIAL_VARIANTS
    )
    if all_initial_null:
        for variant in NULL_ONLY_VARIANTS:
            run(variant)

    retained = [variant for variant in sequence if results[variant]["retained"]]
    completed = {
        **immutable,
        "status": "completed",
        "created_at": created_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "candidate_sequence": sequence,
        "candidate_count": len(sequence),
        "trajectory_count": 6 * len(sequence),
        "retained_variants": retained,
        "c2_extensions_ran": FACTOR_MIXER_K8_VARIANT in results,
        "null_only_variants_ran": CAPACITY_96_VARIANT in results,
    }
    _atomic_json(manifest_path, completed)
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the gated Phase C representation/capacity campaign"
    )
    parser.add_argument("--parent-campaign", required=True, type=Path)
    parser.add_argument("--tilt-sidecar", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(
        run_phase_c_campaign(
            resolve_feature_store(),
            args.parent_campaign,
            args.tilt_sidecar,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
