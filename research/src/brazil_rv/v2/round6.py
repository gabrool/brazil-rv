"""Registered Round-6 input freeze and staged GPU plans; no host lifecycle policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .config import PROJECT_ROOT, ModelConfig
from .contract import (
    ALLOWED_SEEDS,
    DEVELOPMENT_END,
    DEVELOPMENT_FOLDS,
    RUN_MANY_PLAN_SCHEMA,
)
from .data_roots import resolve_external_root
from .round5_screens import FAMILIES
from .store import open_store_for_dates
from .train import model_config_contract, stage_p_model_config

PROTOCOL = PROJECT_ROOT / "research/preregistrations/v2_round6.json"
INPUTS = PROJECT_ROOT / "docs/v2_round6_inputs.json"
SCHEMA = "BRAZIL_RV_V2_ROUND6_V1"
SESSION1_FAMILIES = (
    "magnitudes",
    "oddlot",
    "options",
    "events",
    "fundamentals",
    "lending",
)
SESSION1 = (*SESSION1_FAMILIES, "finetune_lr_1", "time_decay_756", "mlp")
SESSION2 = ("cross_market", "sector", "microstructure", "rebalance")


def resolve_file(record: dict) -> Path:
    directory, _, name = record["path"].replace("\\", "/").rpartition("/")
    root, _ = resolve_external_root(directory)
    path = root / name
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Round-6 input differs: {path}")
    return path


def input_coverage(store_root: Path) -> dict:
    """Input-only, decision-row support; fit and evaluation must both be nonempty."""
    _, dates = rr._read_store_header(store_root)
    fit, selection, evaluation, _, _ = rr._fold_indices(dates)
    rows = np.unique(
        np.concatenate([*fit.values(), *selection.values(), *evaluation.values()])
    )
    store, access = open_store_for_dates(store_root, rows, purpose="training")
    result = {}
    try:
        active = store.read("active", rows)
        for family in FAMILIES:
            present = active & store.read(f"sidecar_{family}_valid", rows).any(axis=-1)
            coverage = {}
            for fold in DEVELOPMENT_FOLDS:
                counts = {}
                for label, panel in (
                    ("fit", fit),
                    ("selection", selection),
                    ("evaluation", evaluation),
                ):
                    subset = present[np.searchsorted(rows, panel[fold])]
                    counts[label] = int(subset.sum())
                    counts[f"{label}_sessions"] = int(subset.any(axis=1).sum())
                counts["informative"] = counts["fit"] > 0 and counts["evaluation"] > 0
                coverage[fold] = counts
            result[family] = coverage
    finally:
        store.close()
    return {
        "rule": "active at decision t and any family field valid at t; positive fit and evaluation support",
        "coverage": result,
        "folds": {
            family: [f for f, c in folds.items() if c["informative"]]
            for family, folds in result.items()
        },
        "access": access.payload(),
        "scores_read": False,
        "store_manifest_sha256": sha256_file(store_root / "manifest.json"),
    }


def families_for(arm: str, roster: dict | None = None) -> tuple[str, ...]:
    if arm in FAMILIES:
        return (arm,)
    if arm in ("S0", "finetune_lr_1", "time_decay_756", "mlp"):
        return ()
    if roster is not None and arm in ("C6", "C6_fresh_p", "best_single_fresh_p"):
        return (
            tuple(roster["c6_families"])
            if arm != "best_single_fresh_p"
            else (roster["best_single"],)
        )
    raise ValueError(f"unregistered Round-6 arm or missing roster: {arm}")


def arm_config(feature_names: dict, arm: str, *, stage="F", roster=None) -> ModelConfig:
    config = ModelConfig(
        slow_feature_count=len(feature_names["slow"]),
        current_feature_count=0,
        disable_fast_stream=True,
        use_bf16=True,
        slow_encoder_kind="mlp" if arm == "mlp" else "gru",
        time_decay_half_life_sessions=756.0
        if arm == "time_decay_756" and stage != "P"
        else None,
        sidecar_feature_counts=tuple(
            (f, len(feature_names[f"sidecar_{f}"])) for f in families_for(arm, roster)
        ),
    )
    return stage_p_model_config(config) if stage == "P" else config


def freeze(output: Path) -> str:
    code = rr._git_identity()
    protocol = rr._read_json(PROTOCOL)
    inputs = rr._read_json(INPUTS)
    store_root, _ = resolve_external_root(inputs["store"]["root"])
    manifest, dates = rr._read_store_header(store_root)
    if sha256_file(store_root / "manifest.json") != inputs["store"]["manifest_sha256"]:
        raise ValueError("Round-6 store changed after CPU acceptance")
    if dates[-1] > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("Round 6 refuses held-out rows")
    # Local measured coverage is already sealed; only its input manifest must match.
    if (
        inputs["input_coverage"]["store_manifest_sha256"]
        != inputs["store"]["manifest_sha256"]
    ):
        raise ValueError("coverage was measured on another store")
    source_store, _ = resolve_external_root(inputs["s0_store"]["root"])
    if (
        sha256_file(source_store / "manifest.json")
        != inputs["s0_store"]["manifest_sha256"]
    ):
        raise ValueError("S0 transfer source store changed")
    checkpoints = {
        s: {"path": str(resolve_file(r)), "sha256": r["sha256"]}
        for s, r in protocol["s0_stage_p"].items()
    }
    parent_root, _ = resolve_external_root(protocol["s0_panels"]["root"])
    if (
        sha256_file(parent_root / "artifact_inventory.json")
        != protocol["s0_panels"]["inventory_sha256"]
    ):
        raise ValueError("sealed S0 inventory changed")
    bindings = {
        path.name: sha256_file(path)
        for path in sorted(PROTOCOL.parent.glob("v2_round6*"))
        if path.suffix in (".json", ".md")
    }
    design = {
        "schema": SCHEMA,
        "status": "frozen_before_training",
        "implementation": code,
        "store": {**inputs["store"], "root": str(store_root)},
        "s0_store": {**inputs["s0_store"], "root": str(source_store)},
        "s0_stage_p": checkpoints,
        "mlp_stage_p": {
            s: {"path": str(resolve_file(r)), "sha256": r["sha256"]}
            for s, r in protocol["mlp_stage_p"].items()
        },
        "s0_panels": {**protocol["s0_panels"], "root": str(parent_root)},
        "preregistration": bindings,
        "baseline": "matched_refit_S0",
        "training_recipe": {
            "maximum_epochs": 60,
            "selection_interval": 2,
            "patience": 3,
            "date_sampling": "unique_dates",
        },
        "input_acceptance_sha256": sha256_file(INPUTS),
        "input_coverage": inputs["input_coverage"],
        "feature_names": manifest["feature_names"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "model_contracts": {
            a: model_config_contract(arm_config(manifest["feature_names"], a))
            for a in ("S0", *SESSION1, *SESSION2)
        },
        "enabled_sidecars": [],
        "fast_initialization": {
            "mode": "native_fresh",
            "transfer_chronology_clean": True,
        },
        "inference_ablation": "each enabled family forced invalid separately, plus all jointly when multiple; same checkpoint, dates and score mask",
        "evaluation_sources": inputs["evaluation_sources"],
        **rr.RESEARCH_FLAGS,
        **rr._source_tier_labels(manifest),
    }
    output.mkdir(parents=True, exist_ok=False)
    return write_json_atomic(output / "frozen_design.json", design)


def design_at(root: Path) -> dict:
    design = rr._read_json(root / "frozen_design.json")
    if design["schema"] != SCHEMA or design["implementation"] != rr._git_identity():
        raise ValueError("Round-6 training differs from its clean frozen commit")
    for name, digest in design["preregistration"].items():
        if sha256_file(PROTOCOL.parent / name) != digest:
            raise ValueError("Round-6 registration changed after freeze")
    if sha256_file(INPUTS) != design["input_acceptance_sha256"]:
        raise ValueError("Round-6 input acceptance changed after freeze")
    return design


def completed(
    run: Path, design: dict, arm: str, stage: str, seed: int, fold: str, *, roster=None
) -> dict:
    manifest = rr._read_json(run / "run_manifest.json")
    rr._assert_current_clean_training(manifest, path=run / "run_manifest.json")
    if (manifest["stage"], manifest["seed"], manifest["fold"]) != (stage, seed, fold):
        raise ValueError(f"trajectory identity differs: {run}")
    contract = manifest["checkpoint_input_contract"]
    if contract["model_config"] != model_config_contract(
        arm_config(design["feature_names"], arm, stage=stage, roster=roster)
    ):
        raise ValueError(f"trajectory model contract differs: {run}")
    if contract["implementation_commit"] != design["implementation"]["commit"]:
        raise ValueError("trajectory training commit changed")
    if manifest["compiled_graphs"] != {"training": 1, "selection": 1, "total": 2}:
        raise ValueError(f"trajectory compile smoke failed: {run}")
    optimizer = manifest["optimizer"]
    if (
        optimizer["date_sampling"] != "unique_dates"
        or optimizer["selection_interval_epochs"] != 2
        or optimizer["padded_name_count"] is None
    ):
        raise ValueError("trajectory differs from the compact unique-date recipe")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"trajectory artifact changed: {run / name}")
    for relative, digest in manifest.get("score_manifests", {}).items():
        if sha256_file(run / relative / "score_manifest.json") != digest:
            raise ValueError("trajectory scoring or attribution artifact changed")
    expected_lr = 1.0 if arm == "finetune_lr_1" else 0.3
    if (
        stage == "F"
        and manifest["optimizer"]["pretrained_lr_multiplier"] != expected_lr
    ):
        raise ValueError("trajectory pretrained LR multiplier differs")
    return manifest


def trajectory(root: Path, arm: str, seed: int, fold: str, stage="F") -> Path:
    return (
        root
        / "trajectories"
        / arm
        / (f"stage_P/seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
    )


def training_command(
    design,
    run,
    arm,
    seed,
    fold,
    stage,
    *,
    smoke=False,
    checkpoint=None,
    digest=None,
    reused_s0=False,
    roster=None,
):
    families = families_for(arm, roster)
    command = rr._training_command(
        design={**design, "enabled_sidecars": families},
        output_dir=run,
        stage=stage,
        seed=seed,
        fold=None if stage == "P" else fold,
        maximum_epochs=1 if smoke else 60,
        score_output=not smoke and stage == "F",
        pretrain_checkpoint=checkpoint,
        pretrain_sha256=digest,
    )
    command += [
        "--slow-only",
        "--record-branch-diagnostics",
        "--use-bf16",
        "--selection-interval",
        "2",
    ]
    if families and not smoke and stage == "F":
        command.append("--score-sidecar-ablations")
    if reused_s0:
        command += ["--pretrain-parent-store", design["s0_store"]["root"]]
    if arm == "mlp":
        command += ["--slow-encoder-kind", "mlp"]
    if arm == "finetune_lr_1" and stage == "F":
        command += ["--pretrained-lr-multiplier", "1.0"]
    if arm == "time_decay_756" and stage == "F":
        command += ["--time-decay-half-life", "756"]
    return command


def derive_session2_roster(paired_points: dict[str, float]) -> dict:
    if set(paired_points) != set(SESSION1_FAMILIES) or not all(
        np.isfinite(v) for v in paired_points.values()
    ):
        raise ValueError(
            "C6 requires all six defined session-1 informative-fold points"
        )
    best = max(SESSION1_FAMILIES, key=lambda f: paired_points[f])
    c6 = [f for f in SESSION1_FAMILIES if paired_points[f] > 0]
    return {
        "best_single": best,
        "c6_families": c6,
        "c6_identity_s0": not c6,
        "paired_primary_ic_points": paired_points,
    }


def session2_roster(root: Path) -> dict:
    record = rr._read_json(root / "session2_roster.json")
    result = root / "session1_result.json"
    if record["session1_result_sha256"] != sha256_file(result):
        raise ValueError("session-2 roster result binding differs")
    if record["frozen_design_sha256"] != sha256_file(root / "frozen_design.json"):
        raise ValueError("session-2 roster design binding differs")
    return record


def write_plan(root: Path, phase: str) -> str:
    design = design_at(root)
    roster = session2_roster(root) if phase.startswith("session2") else None
    second = (
        *SESSION2,
        "best_single_fresh_p",
        *(("C6", "C6_fresh_p") if roster and roster["c6_families"] else ()),
    )
    smoke = phase.endswith("smoke")
    if phase == "session1_smoke":
        tasks = [(a, 11, "F14", "F") for a in ("S0", *SESSION1)]
    elif phase == "session2_smoke":
        tasks = [(a, 11, "F14", "F") for a in second]
        tasks += [
            (a, 11, "pretrain_internal", "P") for a in second if a.endswith("fresh_p")
        ]
    elif phase == "session1":
        tasks = [
            (a, s, f, "F")
            for a in ("S0", *SESSION1)
            for f in DEVELOPMENT_FOLDS
            for s in ALLOWED_SEEDS
        ]
    elif phase == "session2_p":
        tasks = [
            (a, s, "pretrain_internal", "P")
            for a in second
            if a.endswith("fresh_p")
            for s in ALLOWED_SEEDS
        ]
    elif phase == "session2":
        tasks = [
            (a, s, f, "F")
            for a in second
            for f in DEVELOPMENT_FOLDS
            for s in ALLOWED_SEEDS
        ]
    else:
        raise ValueError(f"unknown Round-6 phase: {phase}")
    if not smoke:
        needed = {(a, stage) for a, _, _, stage in tasks}
        for arm, stage in sorted(needed):
            fold = "pretrain_internal" if stage == "P" else "F14"
            manifest = completed(
                root / "smoke" / f"{arm}_{stage}",
                design,
                arm,
                stage,
                11,
                fold,
                roster=roster,
            )
            if (
                manifest["epochs_completed"] != 1
                or (root / "smoke" / f"{arm}_{stage}" / "scores").exists()
            ):
                raise ValueError("smoke must be one epoch without scores")
    jobs = []
    for arm, seed, fold, stage in tasks:
        fresh = arm.endswith("fresh_p")
        run = (
            root / "smoke" / f"{arm}_{stage}"
            if smoke
            else trajectory(root, arm, seed, fold, stage)
        )
        checkpoint, digest = None, None
        reused = stage == "F" and not fresh and arm != "mlp"
        if arm == "mlp" and stage == "F":
            record = design["mlp_stage_p"][str(seed)]
            checkpoint, digest = Path(record["path"]), record["sha256"]
        elif reused:
            record = design["s0_stage_p"][str(seed)]
            checkpoint, digest = Path(record["path"]), record["sha256"]
        elif stage == "F" and not smoke:
            pretrain = trajectory(root, arm, seed, "pretrain_internal", "P")
            manifest = completed(
                pretrain, design, arm, "P", seed, "pretrain_internal", roster=roster
            )
            checkpoint, digest = (
                pretrain / "raw_patience.pt",
                manifest["artifacts"]["raw_patience.pt"],
            )
        command = training_command(
            design,
            run,
            arm,
            seed,
            fold,
            stage,
            smoke=smoke,
            checkpoint=checkpoint,
            digest=digest,
            reused_s0=reused,
            roster=roster,
        )
        jobs.append(
            rr._plan_job(
                name=f"{arm}_{stage}_{fold}_{seed}",
                seed=seed,
                fold=fold,
                run_dir=run,
                command=command,
                stage=stage,
                source_tiers=rr._source_tier_labels({"metadata": design}),
            )
        )
        if not smoke and stage == "F":
            jobs[-1]["expected_manifest"]["scoring_complete"] = True
    path = root / f"round6_plan_{phase}.json"
    if path.exists():
        raise FileExistsError(path)
    return write_json_atomic(
        path,
        {
            "schema": RUN_MANY_PLAN_SCHEMA,
            "phase": phase,
            "max_parallel": 6,
            "first_failure_stop": True,
            "jobs": jobs,
            "research_candidate_score": not smoke,
            "smoke_reused": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "plan", "coverage"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.root))
    elif args.action == "coverage":
        print(write_json_atomic(args.output, input_coverage(args.root)))
    else:
        print(write_plan(args.root, args.phase))


if __name__ == "__main__":
    main()
