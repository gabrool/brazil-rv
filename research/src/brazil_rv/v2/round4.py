"""Round-4 graph contracts and staged GPU plans; launching a host is separate."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .config import PROJECT_ROOT, ModelConfig
from .contract import (
    ALLOWED_SEEDS,
    CONFIRMATION_SEEDS,
    DEVELOPMENT_END,
    DEVELOPMENT_FOLDS,
    RUN_MANY_PLAN_SCHEMA,
)
from .data_roots import portable_name, resolve_external_root
from .research_checkpoint import (
    SCHEMA as CHECKPOINT_SCHEMA,
    CELLS,
    _completed as cpu_cell_completed,
    _finish_cell,
    _context_arguments,
)
from .train import model_config_contract, stage_p_model_config

REGISTRATION = PROJECT_ROOT / "research/preregistrations/v2_round4.md"
SCHEMA = "BRAZIL_RV_V2_ROUND4_V1"
ARMS = ("fast_off", "S0", "H", "P", "L", "C")
FRESH_P = ("fast_off", "S0", "L", "C")
WEIGHTS_H = tuple(value / 3.5 for value in (0.25, 0.25, 1.0, 1.0, 1.0))


def arm_config(feature_names: dict, arm: str, *, stage: str = "F") -> ModelConfig:
    if arm not in (*ARMS, "selection_1235"):
        raise ValueError(f"unknown Round-4 arm: {arm}")
    config = ModelConfig(
        slow_feature_count=len(feature_names["slow"]),
        current_feature_count=len(feature_names["intraday"]),
        disable_fast_stream=True,
    )
    if arm == "S0":
        config = replace(config, current_feature_count=0)
    elif arm == "H":
        config = replace(config, horizon_loss_weights=WEIGHTS_H)
    elif arm == "P":
        config = replace(config, lambda_persistence=0.1)
    elif arm == "L":
        config = replace(
            config,
            slow_feature_count=config.slow_feature_count
            + len(feature_names["sidecar_lending"]),
        )
    elif arm == "C":
        config = replace(config, common_state_feature_count=3)
    elif arm == "selection_1235":
        config = replace(config, selection_horizons=(1, 2, 3, 5))
    return stage_p_model_config(config) if stage == "P" else config


def parent_graph(arm: str) -> str:
    return arm if arm in FRESH_P else "fast_off"


def freeze(cpu_root: Path, output: Path) -> str:
    code = rr._git_identity()
    result_path = cpu_root / "checkpoint_cpu_result.json"
    result = rr._read_json(result_path)
    if (
        result.get("schema") != CHECKPOINT_SCHEMA
        or result.get("status") != "cpu_rebaseline_complete"
    ):
        raise ValueError("Round 4 requires the completed CPU re-baseline")
    inventory_path = cpu_root / "artifact_inventory.json"
    sealed = rr._read_json(inventory_path)
    if (
        sealed.get("status") != "passed"
        or rr.inventory(cpu_root, exclude=set(sealed["excluded_self"]))
        != sealed["files"]
    ):
        raise ValueError("Round 4 requires a verified sealed CPU root")
    for name in (*rr._BASELINE_SIGNAL_NAMES, *CELLS):
        family = "gbdt" if name in CELLS else "baselines"
        for fold in DEVELOPMENT_FOLDS:
            if not cpu_cell_completed(cpu_root / family / name / fold):
                raise ValueError(f"CPU acceptance missing: {name}/{fold}")
    design = copy.deepcopy(rr._read_json(cpu_root / "frozen_design.json"))
    resolutions = []
    for record, key in (
        (design["store"], "root"),
        (design["bova11"], "root"),
        (design["bova11"], "hedge_beta_root"),
        (design["lending_archive"], "root"),
        (design["execution_policy"], "root"),
    ):
        path, audit = resolve_external_root(record[key])
        record[key] = str(path)
        resolutions.append(audit.payload())
    for record in design["cdi"].values():
        original = record["path"]
        directory = original.replace("\\", "/").rsplit("/", 1)[0]
        path, audit = resolve_external_root(directory)
        record["path"] = str(path / portable_name(original))
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("relocated CDI file differs from its binding")
        resolutions.append(audit.payload())
    manifest, dates = rr._read_store_header(Path(design["store"]["root"]))
    if (
        sha256_file(Path(design["store"]["root"]) / "manifest.json")
        != design["store"]["manifest_sha256"]
    ):
        raise ValueError("Round-4 store differs from the accepted CPU binding")
    if dates[-1] > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("Round 4 refuses a store extending beyond development")
    rr._fold_indices(dates)
    design.update(
        schema=SCHEMA,
        implementation=code,
        status="frozen_before_training",
        cpu_checkpoint={
            "root": str(cpu_root.resolve()),
            "result_sha256": sha256_file(result_path),
            "frozen_design_sha256": sha256_file(cpu_root / "frozen_design.json"),
            "inventory_sha256": sha256_file(inventory_path),
        },
        preregistration={
            "path": str(REGISTRATION),
            "sha256": sha256_file(REGISTRATION),
        },
        feature_names=manifest["feature_names"],
        feature_schema_sha256=manifest["feature_schema_sha256"],
        model_contracts={
            arm: model_config_contract(arm_config(manifest["feature_names"], arm))
            for arm in (*ARMS, "selection_1235")
        },
        enabled_sidecars=[],
        fast_initialization={"mode": "native_fresh", "transfer_chronology_clean": True},
        path_resolutions=resolutions,
        max_parallel_trajectories=6,
        **rr.RESEARCH_FLAGS,
        **rr._source_tier_labels(manifest),
    )
    output.mkdir(parents=True, exist_ok=False)
    return write_json_atomic(output / "frozen_design.json", design)


def _design(root: Path) -> dict:
    design = rr._read_json(root / "frozen_design.json")
    if design["schema"] != SCHEMA or design["implementation"] != rr._git_identity():
        raise ValueError("Round-4 implementation differs from its frozen commit")
    if sha256_file(REGISTRATION) != design["preregistration"]["sha256"]:
        raise ValueError("Round-4 registration changed after freeze")
    return design


def completed(
    run: Path, design: dict, arm: str, stage: str, seed: int, fold: str
) -> dict:
    manifest = rr._read_json(run / "run_manifest.json")
    rr._assert_current_clean_training(manifest, path=run / "run_manifest.json")
    if (manifest["stage"], manifest["seed"], manifest["fold"]) != (stage, seed, fold):
        raise ValueError(f"trajectory identity differs: {run}")
    expected = model_config_contract(
        arm_config(design["feature_names"], arm, stage=stage)
    )
    contract = manifest["checkpoint_input_contract"]
    if (
        contract["model_config"] != expected
        or contract["implementation_commit"] != design["implementation"]["commit"]
    ):
        raise ValueError(f"trajectory model or code contract differs: {run}")
    if manifest["compiled_graphs"] != {"training": 1, "selection": 1, "total": 2}:
        raise ValueError(f"trajectory compile smoke failed: {run}")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"trajectory artifact mismatch: {run / name}")
    return manifest


def _command(
    design, run, arm, seed, fold, stage, *, smoke=False, checkpoint=None, digest=None
):
    command = rr._training_command(
        design=design,
        output_dir=run,
        stage=stage,
        seed=seed,
        fold=fold,
        maximum_epochs=1 if smoke else 20,
        score_output=not smoke,
        pretrain_checkpoint=checkpoint,
        pretrain_sha256=digest,
    )
    command += ["--disable-fast-stream", "--record-branch-diagnostics"]
    if arm == "S0":
        command.append("--slow-only")
    elif arm == "L":
        command += ["--sidecar", "lending"]
    elif arm == "C":
        command.append("--common-state")
    elif stage == "F" and arm == "H":
        command += ["--horizon-loss-weights", *(repr(x) for x in WEIGHTS_H)]
    elif stage == "F" and arm == "P":
        command += ["--lambda-persistence", "0.1"]
    elif stage == "F" and arm == "selection_1235":
        command += ["--selection-horizons", "1", "2", "3", "5"]
    return command


def write_plan(root: Path, phase: str, *, confirmation_arms=()) -> str:
    design = _design(root)
    jobs = []
    tiers = rr._source_tier_labels({"metadata": design})
    if phase != "smoke":
        for arm in ARMS:
            run = root / "smoke" / arm
            manifest = completed(
                run, design, arm, "F", 11, "F14" if arm == "L" else "F1"
            )
            if manifest["epochs_completed"] != 1 or (run / "scores").exists():
                raise ValueError(
                    "smoke must be one epoch without reusable score artifacts"
                )
    if phase in ("arms", "selection", "confirmation_p", "confirmation"):
        for fold in DEVELOPMENT_FOLDS:
            if not cpu_cell_completed(root / "aggregates/screening/fast_off" / fold):
                raise ValueError(
                    "evaluate and accept the re-baselined parent before arms"
                )
    if phase.startswith("confirmation"):
        if not confirmation_arms or any(arm not in ARMS for arm in confirmation_arms):
            raise ValueError("confirmation needs an explicit screened arm roster")
        if not (root / "screening_result.json").exists():
            raise ValueError("confirmation follows the complete screening readout")
        confirmation_arms = tuple(
            arm for arm in ARMS if arm in {*confirmation_arms, "fast_off", "S0"}
        )
    if phase == "smoke":
        roster = [(a, 11, "F14" if a == "L" else "F1", "F") for a in ARMS]
    elif phase == "p":
        roster = [
            (a, s, "pretrain_internal", "P") for a in FRESH_P for s in ALLOWED_SEEDS
        ]
    elif phase in ("parent", "arms"):
        roster = [
            (a, s, f, "F")
            for a in (("fast_off",) if phase == "parent" else ARMS[1:])
            for f in DEVELOPMENT_FOLDS
            for s in ALLOWED_SEEDS
        ]
    elif phase == "selection":
        roster = [
            ("selection_1235", s, f, "F")
            for f in DEVELOPMENT_FOLDS[-3:]
            for s in ALLOWED_SEEDS
        ]
    elif phase == "confirmation_p":
        graphs = tuple(
            a for a in FRESH_P if a in {parent_graph(x) for x in confirmation_arms}
        )
        roster = [
            (a, s, "pretrain_internal", "P") for a in graphs for s in CONFIRMATION_SEEDS
        ]
    elif phase == "confirmation":
        roster = [
            (a, s, f, "F")
            for a in confirmation_arms
            for f in DEVELOPMENT_FOLDS
            for s in CONFIRMATION_SEEDS
        ]
    else:
        raise ValueError("unknown Round-4 phase")
    for arm, seed, fold, stage in roster:
        smoke = phase == "smoke"
        run = (
            root / "smoke" / arm
            if smoke
            else root
            / "trajectories"
            / arm
            / (f"stage_P/seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
        )
        checkpoint, digest = None, None
        if stage == "F" and not smoke:
            graph = parent_graph(arm)
            pretrain = root / "trajectories" / graph / "stage_P" / f"seed_{seed}"
            manifest = completed(
                pretrain, design, graph, "P", seed, "pretrain_internal"
            )
            checkpoint, digest = (
                pretrain / "raw_patience.pt",
                manifest["artifacts"]["raw_patience.pt"],
            )
        command = _command(
            design,
            run,
            arm,
            seed,
            fold,
            stage,
            smoke=smoke,
            checkpoint=checkpoint,
            digest=digest,
        )
        jobs.append(
            rr._plan_job(
                name=f"{arm}_{stage}_{fold}_{seed}",
                seed=seed,
                fold=fold,
                run_dir=run,
                command=command,
                stage=stage,
                source_tiers=tiers,
            )
        )
    path = root / f"round4_plan_{phase}.json"
    if path.exists():
        raise FileExistsError(path)
    return write_json_atomic(
        path,
        {
            "schema": RUN_MANY_PLAN_SCHEMA,
            "phase": phase,
            "max_parallel": 1 if phase == "smoke" else 6,
            "first_failure_stop": True,
            "jobs": jobs,
            "confirmation_arms": list(confirmation_arms),
            "research_candidate_score": phase != "smoke",
            "smoke_reused": False,
        },
    )


def promotion_trace(readouts: dict, comparisons: dict, *, confirmed: bool) -> dict:
    """Apply the registered tie/eligibility rules; screening never promotes."""
    eligible = [
        a
        for a in ARMS
        if a in readouts
        and (
            readouts[a]["headline_net_excess_bps"]["estimate"] is not None
            and readouts[a]["headline_net_excess_bps"]["estimate"] >= 0
        )
    ]
    eligible = [
        a
        for a in eligible
        if readouts[a]["primary_neutral_target_ic"]["estimate"] is not None
    ]
    leader = max(
        eligible,
        key=lambda a: readouts[a]["primary_neutral_target_ic"]["estimate"],
        default=None,
    )
    override = None
    if leader is not None:
        options = []
        for arm in eligible:
            if arm == leader:
                continue
            paired = comparisons[f"{arm}_minus_{leader}"]
            if rr._interval_includes_zero(
                paired["primary_neutral_target_ic"]
            ) and rr._interval_is_positive(paired["headline_net_excess_bps"]):
                options.append(arm)
        if options:
            override = max(
                options,
                key=lambda a: readouts[a]["headline_net_excess_bps"]["estimate"],
            )
    s0_delta = comparisons.get("S0_minus_fast_off", {}).get(
        "primary_neutral_target_ic", {}
    )
    upper = s0_delta.get("upper_95")
    s0_adopt = "S0" in eligible and upper is not None and upper >= 0
    return {
        "confirmed_six_seed_panel": confirmed,
        "eligible": eligible,
        "ic_leader": leader,
        "economics_override": override,
        "designation": (override or leader) if confirmed else None,
        "next_round_parent": ("S0" if s0_adopt else "fast_off") if confirmed else None,
        "retained_parent_eligible": "fast_off" in eligible,
        "provisional_parent": "S0" if s0_adopt else "fast_off",
        "S0_reason": "simpler_with_no_demonstrated_IC_inferiority"
        if s0_adopt
        else "negative_or_undefined_economics"
        if "S0" not in eligible
        else "paired_IC_upper_bound_below_zero_or_undefined",
        "read_2025_authorized": False,
        "economics_basis": "registered_resolved_fold_pool; inspect per_candidate_economics_coverage; not an implementability claim",
        "read_bar_status": "Gabriel_to_set",
        "in_sample_selection_label": True,
    }


def evaluate_phase(root: Path, phase: str) -> str:
    from .checkpoint_readouts import candidate_readout, paired_readouts, retained
    from .research_diagnostics import momentum_diagnostics, pooled_momentum_diagnostics

    design = _design(root)
    if phase not in ("parent", "screening", "confirmation"):
        raise ValueError("unknown evaluation phase")
    result_path = root / f"{phase}_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    seeds = ALLOWED_SEEDS
    arms = ("fast_off",) if phase == "parent" else ARMS
    if phase == "confirmation":
        plan = rr._read_json(root / "round4_plan_confirmation.json")
        arms = tuple(plan["confirmation_arms"])
        seeds = (*ALLOWED_SEEDS, *CONFIRMATION_SEEDS)
    group = "confirmation" if phase == "confirmation" else "screening"
    cpu_root = Path(design["cpu_checkpoint"]["root"])
    if (
        sha256_file(cpu_root / "checkpoint_cpu_result.json")
        != design["cpu_checkpoint"]["result_sha256"]
    ):
        raise ValueError("CPU checkpoint changed after Round-4 freeze")
    source_hashes = rr._read_json(
        cpu_root / "baselines/momentum_12_1/F1/evaluation.json"
    )["source_artifact_hashes"]
    context = rr._open_ledger_replay(design)
    _, dates = rr._read_store_header(Path(design["store"]["root"]))
    policy, _ = rr.load_selected_policy(
        Path(design["execution_policy"]["root"]),
        expected_result_sha256=design["execution_policy"]["result_sha256"],
    )
    paths, trajectories = {}, {}
    try:
        for arm in (*arms, *(("selection_1235",) if phase == "screening" else ())):
            folds = (
                DEVELOPMENT_FOLDS[-3:] if arm == "selection_1235" else DEVELOPMENT_FOLDS
            )
            arm_seeds = ALLOWED_SEEDS if arm == "selection_1235" else seeds
            paths[arm] = {}
            for fold in folds:
                destination = root / "aggregates" / group / arm / fold
                paths[arm][fold] = destination
                if cpu_cell_completed(destination):
                    metadata = rr._read_json(destination / "score_manifest.json")[
                        "metadata"
                    ]
                    if any(
                        metadata.get(key) != expected
                        for key, expected in {
                            "arm": arm,
                            "fold": fold,
                            "seeds": list(arm_seeds),
                            "round4_registration_sha256": design["preregistration"][
                                "sha256"
                            ],
                        }.items()
                    ):
                        raise ValueError(
                            "accepted aggregate differs from the requested arm/seed panel"
                        )
                    continue
                if (destination / "score_manifest.json").exists():
                    raise RuntimeError(
                        "partial aggregate needs diagnosis before any retry"
                    )
                members, reference_mask, records = [], None, []
                for seed in arm_seeds:
                    run = root / "trajectories" / arm / f"{fold}_seed_{seed}"
                    manifest = completed(run, design, arm, "F", seed, fold)
                    graph = parent_graph(arm)
                    pretrain = (
                        root / "trajectories" / graph / "stage_P" / f"seed_{seed}"
                    )
                    p_manifest = completed(
                        pretrain, design, graph, "P", seed, "pretrain_internal"
                    )
                    if (
                        manifest["pretrain_checkpoint_sha256"]
                        != p_manifest["artifacts"]["raw_patience.pt"]
                    ):
                        raise ValueError(
                            "fine-tuning did not use its registered same-seed P graph"
                        )
                    score, mask = rr._score_artifact(
                        run / "scores",
                        require_clean_transfer=True,
                        expected_dates=dates[context.evaluation[fold]],
                        expected_isins=context.store.isins,
                        expected_feature_schema_sha256=design["feature_schema_sha256"],
                    )
                    if reference_mask is not None and not np.array_equal(
                        reference_mask, mask
                    ):
                        raise ValueError("seed ensemble masks differ")
                    reference_mask = mask
                    members.append(score)
                    records.append(
                        {
                            "seed": seed,
                            "run_manifest_sha256": sha256_file(
                                run / "run_manifest.json"
                            ),
                            "selected_epoch": manifest["selected_epoch"],
                            "epochs_completed": manifest["epochs_completed"],
                        }
                    )
                scores = rr.rank_average_ensemble(members, reference_mask)
                rr._persist_scores(
                    destination,
                    {"scores": scores, "score_mask": reference_mask},
                    {
                        **rr._source_tier_labels(context.store.manifest),
                        "arm": arm,
                        "seeds": list(arm_seeds),
                        "fold": fold,
                        "evaluation_date_indices": context.evaluation[fold].tolist(),
                        "trajectories": records,
                        "round4_registration_sha256": design["preregistration"][
                            "sha256"
                        ],
                    },
                )
                evaluated = rr._evaluate(
                    **_context_arguments(context, source_hashes),
                    indices=context.evaluation[fold],
                    scores=scores,
                    score_mask=reference_mask,
                    fold=fold,
                    output=destination / "evaluation.json",
                    execution_policy=policy,
                )
                _finish_cell(destination, evaluated, name=arm, fold=fold)
                trajectories[f"{arm}/{fold}"] = records
                del members, evaluated
        readouts = {arm: candidate_readout(paths[arm]) for arm in arms}
        if phase == "parent":
            return write_json_atomic(
                result_path,
                {
                    "status": "parent_accepted",
                    "readouts": readouts,
                    **rr.RESEARCH_FLAGS,
                },
            )
        paired = paired_readouts(
            context, {arm: paths[arm] for arm in arms}, root / "paired" / group
        )
        selection_comparison = None
        if phase == "screening":
            selection_comparison = paired_readouts(
                context,
                {
                    "selection_1235": paths["selection_1235"],
                    "fast_off": {
                        fold: paths["fast_off"][fold] for fold in DEVELOPMENT_FOLDS[-3:]
                    },
                },
                root / "diagnostics/selection_metric",
            )
        momentum = {}
        for fold in DEVELOPMENT_FOLDS:
            parent = retained(context, paths["fast_off"][fold], fold)
            m_score, m_mask = rr._score_artifact(
                cpu_root / "baselines/momentum_12_1" / fold, require_clean_transfer=True
            )
            diagnostic = momentum_diagnostics(parent.inputs, m_score, m_mask)
            write_json_atomic(
                root / "diagnostics" / group / "momentum" / f"{fold}.json", diagnostic
            )
            momentum[fold] = diagnostic
        trace = promotion_trace(
            {a: r["pooled"] for a, r in readouts.items()},
            {key: value["pooled"] for key, value in paired.items()},
            confirmed=phase == "confirmation",
        )
        return write_json_atomic(
            result_path,
            {
                "schema": SCHEMA,
                "status": "confirmed_readout"
                if phase == "confirmation"
                else "screened_requires_confirmation",
                "seed_roster": list(seeds),
                "readouts": readouts,
                "paired": paired,
                "selection_metric_only": selection_comparison,
                "momentum_diagnostics": pooled_momentum_diagnostics(momentum),
                "promotion_trace": trace,
                "trajectories": trajectories,
                "access": context.access,
                "read_2025_bar": "unapproved_proposal_Gabriel_to_set",
                **rr.RESEARCH_FLAGS,
            },
        )
    except BaseException as error:
        write_json_atomic(
            root / f"{phase}_stop.json", {"error": str(error), "at_utc": rr._utc_now()}
        )
        raise
    finally:
        context.store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "plan", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path)
    parser.add_argument(
        "--phase",
        choices=(
            "smoke",
            "p",
            "parent",
            "arms",
            "selection",
            "confirmation_p",
            "confirmation",
            "screening",
        ),
    )
    parser.add_argument("--confirmation-arm", choices=ARMS, action="append", default=[])
    args = parser.parse_args(argv)
    if args.action == "freeze":
        if args.cpu_root is None:
            parser.error("freeze requires --cpu-root")
        print(freeze(args.cpu_root, args.root))
    elif args.action == "plan":
        if args.phase is None:
            parser.error("plan requires --phase")
        print(
            write_plan(args.root, args.phase, confirmation_arms=args.confirmation_arm)
        )
    else:
        print(evaluate_phase(args.root, args.phase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
