from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS
from .data import (
    di_tilt_sidecar_identity,
    feature_store_identity,
    resolve_feature_store,
)
from .model import DI_TILT_EXPOSURE_VARIANT, RESIDUAL_AUXILIARY_VARIANT
from .next_stage_diagnostics import _parent_validation_members
from .provenance import repository_commit
from .train import run_training
from .trajectory import load_frozen_selection

EXPERIMENT_COUNT_BEFORE_STAGE_0 = 14
SIDECAR_VARIANTS = frozenset((DI_TILT_EXPOSURE_VARIANT, RESIDUAL_AUXILIARY_VARIANT))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_name(manifest: Mapping[str, object]) -> str:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Run manifest has no model metadata")
    variant = model.get("variant")
    if not isinstance(variant, Mapping) or not isinstance(variant.get("name"), str):
        raise ValueError("Run manifest has no model variant")
    return str(variant["name"])


def selected_stage3_variants(
    phase_c_manifest: Mapping[str, object],
    r1_summary: Mapping[str, object],
) -> tuple[str, ...]:
    if (
        phase_c_manifest.get("status") != "completed"
        or phase_c_manifest.get("official_validation_accessed") is not False
        or phase_c_manifest.get("test_accessed") is not False
    ):
        raise ValueError("Phase C campaign is incomplete or has crossed a lockbox")
    if (
        r1_summary.get("official_validation_accessed") is not False
        or r1_summary.get("test_accessed") is not False
    ):
        raise ValueError("R1 summary has crossed a lockbox")
    if r1_summary.get("selected_candidate") is not None:
        raise ValueError("This Stage 3 driver supports the observed null R1 decision")
    results = phase_c_manifest.get("results")
    sequence = phase_c_manifest.get("candidate_sequence")
    if not isinstance(results, Mapping) or not isinstance(sequence, list):
        raise ValueError("Phase C campaign has no completed candidate sequence")
    selected: list[str] = []
    if phase_c_manifest.get("d2_training_gate_passed") is True:
        selected.append(RESIDUAL_AUXILIARY_VARIANT)
    for variant in sequence:
        if not isinstance(variant, str):
            raise ValueError("Phase C candidate sequence contains a non-string")
        result = results.get(variant)
        if not isinstance(result, Mapping):
            raise ValueError(f"Phase C result is missing for {variant}")
        if result.get("retained") is True and variant not in selected:
            selected.append(variant)
    return tuple(selected)


def experiment_numbering(
    phase_c_manifest: Mapping[str, object],
) -> dict[str, object]:
    sequence = phase_c_manifest.get("candidate_sequence")
    if not isinstance(sequence, list) or not all(
        isinstance(value, str) for value in sequence
    ):
        raise ValueError("Phase C campaign has no valid candidate sequence")
    first_candidate = EXPERIMENT_COUNT_BEFORE_STAGE_0 + 4
    candidate_numbers = {
        str(variant): first_candidate + offset
        for offset, variant in enumerate(sequence)
    }
    return {
        "count_before_stage_0": EXPERIMENT_COUNT_BEFORE_STAGE_0,
        "stage_0_and_1": {"D1": 15, "D2": 16, "R1": 17},
        "candidate_numbers": candidate_numbers,
        "stage_3_number": first_candidate + len(sequence),
    }


def _official_run(output_dir: Path, variant: str, seed: int) -> Path:
    return output_dir / "official_runs" / variant / f"seed_{seed}"


def _completed_run_matches(
    run_dir: Path,
    *,
    commit: str,
    store: Path,
    identity: Mapping[str, object],
    variant: str,
    seed: int,
    selection_rule_file: Path,
) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _read_json(manifest_path)
    frozen = manifest.get("frozen_selection")
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == commit
        and Path(str(manifest.get("feature_store"))).resolve() == store.resolve()
        and manifest.get("feature_store_identity") == identity
        and int(manifest.get("seed", -1)) == seed
        and manifest.get("split", {}).get("training") == "official"
        and manifest.get("split", {}).get("test_accessed") is False
        and _variant_name(manifest) == variant
        and isinstance(frozen, Mapping)
        and frozen.get("selected_rule") == "patience3_raw"
        and frozen.get("selection_file_sha256")
        == hashlib.sha256(selection_rule_file.read_bytes()).hexdigest()
    )


def _run_official_candidate(
    *,
    store: Path,
    identity: Mapping[str, object],
    commit: str,
    output_dir: Path,
    selection_rule_file: Path,
    sidecar: Path,
    variant: str,
    seed: int,
) -> Path:
    run_dir = _official_run(output_dir, variant, seed)
    if run_dir.exists():
        if not _completed_run_matches(
            run_dir,
            commit=commit,
            store=store,
            identity=identity,
            variant=variant,
            seed=seed,
            selection_rule_file=selection_rule_file,
        ):
            raise ValueError(f"Existing official run differs: {run_dir}")
        return run_dir
    return run_training(
        store=store,
        seed=seed,
        selection_window="official",
        run_dir=run_dir,
        selection_rule_file=selection_rule_file,
        variant=variant,
        tilt_sidecar=sidecar if variant in SIDECAR_VARIANTS else None,
    )


def _clears_paired_interval(report: Mapping[str, object]) -> bool:
    intervals = report.get("per_date_delta_bootstrap")
    if not isinstance(intervals, Mapping):
        raise ValueError("Official comparison has no paired intervals")
    return all(float(intervals[str(block)]["lower_95"][0]) > 0.0 for block in (5, 10))


def run_stage3(
    store: Path,
    phase_c_campaign: Path,
    r1_summary_path: Path,
    d1_summary_path: Path,
    parent_reproduction: Path,
    selection_rule_file: Path,
    sidecar: Path,
    output_dir: Path,
) -> Path:
    commit = repository_commit()
    identity = feature_store_identity(store)
    phase_manifest = _read_json(phase_c_campaign / "campaign_manifest.json")
    r1_summary = _read_json(r1_summary_path)
    d1_summary = _read_json(d1_summary_path)
    variants = selected_stage3_variants(phase_manifest, r1_summary)
    if not variants:
        raise ValueError("No Stage 1/2 survivor or gated D2 candidate reached Stage 3")
    selection = load_frozen_selection(selection_rule_file)
    if selection["selected_rule"] != "patience3_raw":
        raise ValueError("Stage 3 requires the frozen Raw Patience-3 rule")
    require_residual = RESIDUAL_AUXILIARY_VARIANT in variants
    sidecar_manifest = di_tilt_sidecar_identity(
        sidecar, identity, require_residual=require_residual
    )
    if (
        Path(str(phase_manifest.get("next_stage_sidecar"))).resolve()
        != sidecar.resolve()
    ):
        raise ValueError("Stage 3 sidecar differs from the Phase C sidecar")
    numbering = experiment_numbering(phase_manifest)
    immutable = {
        "schema": "NEXT_STAGE_OFFICIAL_STACK",
        "repository_commit": commit,
        "feature_store": str(store.resolve()),
        "feature_store_identity": identity,
        "phase_c_campaign": str(phase_c_campaign.resolve()),
        "r1_summary": str(r1_summary_path.resolve()),
        "d1_summary": str(d1_summary_path.resolve()),
        "parent_reproduction": str(parent_reproduction.resolve()),
        "selection_rule_file": str(selection_rule_file.resolve()),
        "selection_rule_sha256": hashlib.sha256(
            selection_rule_file.read_bytes()
        ).hexdigest(),
        "selected_rule": "patience3_raw",
        "sidecar": str(sidecar.resolve()),
        "sidecar_manifest": sidecar_manifest,
        "stack_variants": list(variants),
        "seeds": list(ALLOWED_SEEDS),
        "experiment_numbering": numbering,
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "stage3_manifest.json"
    created_at = datetime.now(timezone.utc).isoformat()
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise ValueError("Existing Stage 3 campaign has a different contract")
        created_at = str(existing["created_at"])
    _atomic_json(
        manifest_path,
        {**immutable, "created_at": created_at, "status": "running"},
    )
    for variant in variants:
        for seed in ALLOWED_SEEDS:
            _run_official_candidate(
                store=store,
                identity=identity,
                commit=commit,
                output_dir=output_dir,
                selection_rule_file=selection_rule_file,
                sidecar=sidecar,
                variant=variant,
                seed=seed,
            )

    parent = _parent_validation_members(parent_reproduction)
    candidate = {
        f"parent_seed_{seed}": parent[f"seed_{seed}"] for seed in ALLOWED_SEEDS
    }
    for variant in variants:
        for seed in ALLOWED_SEEDS:
            candidate[f"{variant}_seed_{seed}"] = load_run_observations(
                _official_run(output_dir, variant, seed), "patience3_raw"
            )
    comparison = output_dir / "official_stack_comparison"
    if not comparison.exists():
        compare_observation_ensembles(
            candidate,
            parent,
            candidate_rule="uniform_rank_parent_plus_stage_survivors",
            parent_rule="canonical_parent3_patience3_raw",
            output_dir=comparison,
            comparison_metadata={
                "stack_variants": list(variants),
                "seeds": list(ALLOWED_SEEDS),
                "selection_rule": "patience3_raw",
                "d1_staleness_summary": d1_summary,
                "experiment_numbering": numbering,
                "official_validation_accessed": True,
                "test_accessed": False,
            },
        )
    report = _read_json(comparison / "analysis.json")
    clears = _clears_paired_interval(report)
    _atomic_json(
        manifest_path,
        {
            **immutable,
            "created_at": created_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "official_comparison": str((comparison / "analysis.json").resolve()),
            "candidate_minus_parent_primary_ic": report[
                "candidate_minus_parent_primary_ic"
            ],
            "clears_paired_interval_block_5_and_10": clears,
            "held_out_test_read_justified": clears,
        },
    )
    return output_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare the one sparse official next-stage stack"
    )
    parser.add_argument("--phase-c-campaign", required=True, type=Path)
    parser.add_argument("--r1-summary", required=True, type=Path)
    parser.add_argument("--d1-summary", required=True, type=Path)
    parser.add_argument("--parent-reproduction", required=True, type=Path)
    parser.add_argument("--selection-rule-file", required=True, type=Path)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    print(
        run_stage3(
            resolve_feature_store(),
            args.phase_c_campaign,
            args.r1_summary,
            args.d1_summary,
            args.parent_reproduction,
            args.selection_rule_file,
            args.sidecar,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
