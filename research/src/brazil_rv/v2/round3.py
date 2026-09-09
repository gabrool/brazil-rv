"""Registered six-seed Arm-B experiment and matched native-fast ablation."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .config import PROJECT_ROOT
from .contract import DEVELOPMENT_END, HORIZONS, RUN_MANY_PLAN_SCHEMA
from .data_roots import portable_name, resolve_external_root
from .evaluate import _primary_daily_metrics, _primary_population_components, _spearman
from .execution_policy import load_selected_policy

SCHEMA = "BRAZIL_RV_V2_ROUND3_V4F"
SEEDS = (11, 29, 47, 61, 79, 97)
PAIRED_SEEDS = SEEDS[:3]
FOLDS = ("F1", "F2", "F3")
REGISTRATION = PROJECT_ROOT / "research/preregistrations/v2_round3.md"


def protocol() -> dict[str, object]:
    return {
        "stage_p_seeds": list(SEEDS),
        "arm_B": {"seeds": list(SEEDS), "folds": list(FOLDS)},
        "fast_off": {
            "seeds": list(PAIRED_SEEDS),
            "folds": list(FOLDS),
            "fast_present": "forced_zero",
            "TCN": "bypassed",
            "handoff": "same_seed_fresh_stage_P_as_arm_B",
            "current_intraday_scalars": "unchanged",
        },
        "maximum_epochs": 20,
        "patience": 3,
        "lookback": 60,
        "pairs_per_batch": 8,
        "lambda_persistence": 0.0,
        "to_close_weight": 0.0,
        "seed_extension_rule": "keep_six_if_pooled_neutral_IC_not_lower_than_three",
        "ablation_decision": "none_report_paired_delta_and_interval",
        "comparators": ["gbdt_b_intraday", "momentum_12_1", "equal_rank_ensemble"],
        "diagnostics": {
            "gradients": "branch_L2_second_SAM_pass_before_global_clipping_by_epoch",
            "gates": "selected_checkpoint_eager_inference_by_fold_and_fast_present",
        },
        "validation_2025_readout_only": {
            "candidate": "six_seed_B",
            "neutral_IC_lower_95_minimum": 0.015,
            "headline_net_bps_minimum": 3.0,
            "headline_net_lower_95_minimum": -2.0,
            "sterile_minus_headline_bps_minimum": -5.0,
            "open_validation_in_this_round": False,
        },
    }


def freeze(*, round1_root: Path, output: Path, max_parallel: int = 6) -> str:
    code = rr._git_identity()
    if not 1 <= max_parallel <= 6:
        raise ValueError("Round-3 concurrency must be between one and six")
    parent_root = round1_root.resolve(strict=True)
    parent = rr._verify_sealed_root(parent_root, expected_schema=rr.ROUND1_SCHEMA)
    prior = rr._read_json(parent_root / "frozen_design.json")
    if (
        prior.get("fixed_parent_rung") != "b_intraday"
        or parent["gbdt_ladder"]["parent_rung"] != "b_intraday"
    ):
        raise ValueError(
            "Round 3 requires the completed fixed-parent Round 1' ablation"
        )
    if prior.get("round3_registration", {}).get("sha256") != sha256_file(REGISTRATION):
        raise ValueError("Round 1' and Round 3 registrations differ")
    design = {
        key: copy.deepcopy(prior[key])
        for key in (
            "store",
            "cdi",
            "bova11",
            "lending_archive",
            "execution_policy",
            "bootstrap",
            "action_terms_source",
            "schedule_source",
        )
    }
    resolutions = []
    for record, key in (
        (design["store"], "root"),
        (design["bova11"], "root"),
        (design["bova11"], "hedge_beta_root"),
        (design["lending_archive"], "root"),
        (design["execution_policy"], "root"),
    ):
        resolved, audit = resolve_external_root(record[key])
        record[key] = str(resolved)
        resolutions.append(audit.payload())
    for record in design["cdi"].values():
        recorded = str(record["path"])
        directory = recorded.replace("\\", "/").rsplit("/", 1)[0]
        resolved, audit = resolve_external_root(directory)
        path = resolved / portable_name(recorded)
        if sha256_file(path) != record["sha256"]:
            raise ValueError("relocated CDI identity differs")
        record["path"] = str(path)
        resolutions.append(audit.payload())
    store = Path(design["store"]["root"])
    manifest, dates = rr._read_store_header(store)
    if dates[-1] != np.datetime64(DEVELOPMENT_END):
        raise ValueError("Round-3 store must stop at 2024-12-30")
    if sha256_file(store / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("Round-3 store differs from Round 1'")
    rr._fold_indices(dates)
    _, binding = load_selected_policy(Path(design["execution_policy"]["root"]))
    if binding != design["execution_policy"]:
        raise ValueError("Round-3 execution policy differs from Round 1'")
    design.update(
        {
            "schema": SCHEMA,
            "status": "frozen_before_score",
            **rr.RESEARCH_FLAGS,
            "frozen_at_utc": rr._utc_now(),
            "implementation": code,
            "preregistration": {
                "path": str(REGISTRATION),
                "sha256": sha256_file(REGISTRATION),
            },
            "round1": {
                "root": str(parent_root),
                "result_sha256": sha256_file(parent_root / "round1_result.json"),
                "inventory_sha256": sha256_file(
                    parent_root / "artifact_inventory.json"
                ),
                "gbdt_parent_rung": "b_intraday",
            },
            "enabled_sidecars": [],
            "fast_initialization": {
                "mode": "native_fresh",
                "transfer_chronology_clean": True,
            },
            "max_parallel_trajectories": max_parallel,
            "protocol": protocol(),
            "path_resolutions": resolutions,
            "feature_schema_sha256": manifest["feature_schema_sha256"],
            "target_view": rr.REGISTERED_PRIMARY_TARGET,
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    return write_json_atomic(output / "frozen_design.json", design)


def _design(root: Path) -> dict:
    design = rr._read_json(root / "frozen_design.json")
    if design.get("schema") != SCHEMA or design.get("protocol") != protocol():
        raise ValueError("Round-3 protocol differs from its freeze")
    if design["implementation"] != rr._git_identity():
        raise ValueError("Round-3 code differs from the clean frozen implementation")
    if design["preregistration"]["sha256"] != sha256_file(REGISTRATION):
        raise ValueError("Round-3 registration changed after freeze")
    return design


def _completed(run: Path, *, stage: str, seed: int, fold: str, disabled: bool) -> dict:
    manifest = rr._read_json(run / "run_manifest.json")
    rr._assert_current_clean_training(manifest, path=run / "run_manifest.json")
    if (manifest["stage"], manifest["seed"], manifest["fold"]) != (stage, seed, fold):
        raise ValueError(f"trajectory identity differs: {run}")
    if manifest["model_config"]["disable_fast_stream"] != disabled:
        raise ValueError(f"trajectory fast-stream contract differs: {run}")
    if manifest["compiled_graphs"] != {"training": 1, "selection": 1, "total": 2}:
        raise ValueError(f"trajectory compile smoke did not pass: {run}")
    for name, digest in manifest["artifacts"].items():
        if sha256_file(run / name) != digest:
            raise ValueError(f"trajectory artifact mismatch: {run / name}")
    return manifest


def write_plan(*, root: Path, phase: str) -> str:
    design = _design(root)
    jobs = []
    tiers = {key: design[key] for key in ("action_terms_source", "schedule_source")}

    def job(arm, seed, fold, stage, *, smoke=False, checkpoint=None, digest=None):
        run = (
            root / "smoke" / arm
            if smoke
            else root
            / "trajectories"
            / arm
            / (f"stage_P/seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
        )
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
        command.append("--record-branch-diagnostics")
        if arm == "fast_off":
            command.append("--disable-fast-stream")
        jobs.append(
            rr._plan_job(
                name=f"{arm}_{stage}_{fold}_seed_{seed}",
                seed=seed,
                fold=fold,
                run_dir=run,
                command=command,
                stage=stage,
                source_tiers=tiers,
            )
        )

    if phase == "smoke":
        for arm in ("arm_B", "fast_off"):
            job(arm, 11, "F1", "F", smoke=True)
    else:
        for arm in ("arm_B", "fast_off"):
            run = root / "smoke" / arm
            manifest = _completed(
                run, stage="F", seed=11, fold="F1", disabled=arm == "fast_off"
            )
            if manifest["epochs_completed"] != 1 or (run / "scores").exists():
                raise ValueError(
                    "smoke must be exactly one epoch without score artifacts"
                )
        if phase == "p":
            for seed in SEEDS:
                job("arm_B", seed, "pretrain_internal", "P")
        elif phase == "main":
            for arm, seeds in (("arm_B", SEEDS), ("fast_off", PAIRED_SEEDS)):
                for fold in FOLDS:
                    for seed in seeds:
                        run = root / "trajectories/arm_B/stage_P" / f"seed_{seed}"
                        manifest = _completed(
                            run,
                            stage="P",
                            seed=seed,
                            fold="pretrain_internal",
                            disabled=False,
                        )
                        job(
                            arm,
                            seed,
                            fold,
                            "F",
                            checkpoint=run / "raw_patience.pt",
                            digest=manifest["artifacts"]["raw_patience.pt"],
                        )
        else:
            raise ValueError("unknown Round-3 phase")
    path = root / f"round3_plan_{phase}.json"
    if path.exists():
        raise FileExistsError(path)
    return write_json_atomic(
        path,
        {
            "schema": RUN_MANY_PLAN_SCHEMA,
            "phase": phase,
            "max_parallel": 1
            if phase == "smoke"
            else design["max_parallel_trajectories"],
            "first_failure_stop": True,
            "jobs": jobs,
            "research_candidate_score": phase != "smoke",
            "smoke_reused": False,
        },
    )


def paired_horizons(candidate, baseline) -> dict:
    """D1/D2/D3/D5 retain the primary common population; D10 uses its own mask."""
    result = {}
    for head, horizon in enumerate(HORIZONS):
        arrays, audits = [], {}
        for fold in FOLDS:
            left, right = candidate[fold].inputs, baseline[fold].inputs
            rr._validate_paired_identity(
                candidate[fold].result.report, baseline[fold].result.report
            )
            if horizon in rr.PRIMARY_HORIZONS:
                _, _, left_outcome, left_score = _primary_population_components(left)
                _, _, right_outcome, right_score = _primary_population_components(right)
                mask = left_outcome & right_outcome & left_score & right_score
            else:
                mask = (
                    left.active
                    & right.active
                    & left.neutral_target_mask[..., head]
                    & right.neutral_target_mask[..., head]
                    & left.score_mask[..., head]
                    & right.score_mask[..., head]
                    & np.isfinite(left.scores[..., head])
                    & np.isfinite(right.scores[..., head])
                    & np.isfinite(left.neutral_midrank_targets[..., head])
                    & np.isfinite(right.neutral_midrank_targets[..., head])
                )
            delta = np.full(len(left.dates), np.nan)
            for day, population in enumerate(mask):
                if population.sum() >= 20:
                    target = left.neutral_midrank_targets[day, :, head]
                    delta[day] = _spearman(
                        left.scores[day, :, head], target, population
                    ) - _spearman(right.scores[day, :, head], target, population)
            arrays.append(delta)
            audits[fold] = rr._paired_population_rows(left.dates, mask, delta)
        result[f"D{horizon}"] = {
            "pooled": rr._folded_bootstrap(tuple(arrays)),
            "folds": {
                fold: rr._folded_bootstrap((value,))
                for fold, value in zip(FOLDS, arrays, strict=True)
            },
            "population_audit": audits,
        }
    return result


def validation_readout(readout: dict, sterile_delta: dict) -> dict:
    ic = readout["pooled"]["primary_neutral_target_ic"]
    net = readout["pooled"]["headline_net_excess_bps"]
    tests = {
        "neutral_IC_lower_95": (ic["lower_95"], 0.015),
        "headline_net_bps": (net["estimate"], 3.0),
        "headline_net_lower_95": (net["lower_95"], -2.0),
        "sterile_minus_headline_bps": (sterile_delta["estimate"], -5.0),
    }
    checks = {
        key: {
            "actual": actual,
            "minimum": minimum,
            "met": actual is not None and actual >= minimum,
        }
        for key, (actual, minimum) in tests.items()
    }
    return {
        "candidate": "six_seed_B",
        "checks": checks,
        "development_rule_met": all(row["met"] for row in checks.values()),
        "official_validation_opened": False,
        "read_is_spent": False,
        "rule_is_readout_only": True,
    }


def finalize(*, root: Path) -> str:
    design = _design(root)
    if (root / "round3_result.json").exists():
        raise FileExistsError(root / "round3_result.json")
    parent_root = Path(design["round1"]["root"])
    rr._verify_sealed_root(parent_root, expected_schema=rr.ROUND1_SCHEMA)
    for name, key in (
        ("round1_result.json", "result_sha256"),
        ("artifact_inventory.json", "inventory_sha256"),
    ):
        if sha256_file(parent_root / name) != design["round1"][key]:
            raise ValueError("Round 1' changed after Round-3 freeze")
    store_root = Path(design["store"]["root"])
    _, dates = rr._read_store_header(store_root)
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("Round-3 store changed after freeze")
    fit, selection, evaluation, target_window, _ = rr._fold_indices(dates)
    store, access = rr._open_round_store(
        store_root,
        fit,
        selection,
        evaluation,
        target_window,
        rr._pretrain_indices(dates),
    )
    cdi_record = design["cdi"]
    cdi, _ = rr._load_development_cdi(
        dates=dates,
        cdi_path=Path(cdi_record["development_extension"]["path"]),
        expected_sha256=cdi_record["development_extension"]["sha256"],
        experiment52_cdi_path=Path(cdi_record["experiment52_reference"]["path"]),
        experiment52_expected_sha256=cdi_record["experiment52_reference"]["sha256"],
    )
    bova_record = design["bova11"]
    bova = rr.load_bova11_series(
        Path(bova_record["root"]),
        expected_manifest_sha256=bova_record["manifest_sha256"],
        canonical_dates=dates.astype("datetime64[D]").astype(object).tolist(),
    )
    if bova.data_sha256 != bova_record["data_sha256"]:
        raise ValueError("BOVA11 changed after freeze")
    lending = rr._load_frozen_lending(design, store_root=store_root, dates=dates)
    policy, binding = load_selected_policy(Path(design["execution_policy"]["root"]))
    if binding != design["execution_policy"]:
        raise ValueError("execution policy changed after freeze")
    sources = {
        "v2_store_manifest": design["store"]["manifest_sha256"],
        "cdi_development_extension": cdi_record["development_extension"]["sha256"],
        "cdi_experiment52_reference": cdi_record["experiment52_reference"]["sha256"],
        "bova11_manifest": bova.manifest_sha256,
        "bova11_data": bova.data_sha256,
        "lending_archive_manifest": lending.manifest_sha256,
        "lending_archive_balances": lending.balance_sha256,
        "lending_archive_rates": lending.rate_sha256,
        "preregistration": design["preregistration"]["sha256"],
        "round1_result": design["round1"]["result_sha256"],
        "execution_sweep_result": binding["result_sha256"],
    }
    reports, artifacts, trajectories, dispersion = {}, {}, {}, {}
    stage_p = {}
    try:
        for seed in SEEDS:
            run = root / "trajectories/arm_B/stage_P" / f"seed_{seed}"
            manifest = _completed(
                run, stage="P", seed=seed, fold="pretrain_internal", disabled=False
            )
            stage_p[str(seed)] = manifest

        def evaluate_panel(name, fold, scores, mask, metadata):
            output = root / "aggregates" / name / fold
            rr._persist_scores(
                output,
                {"scores": scores, "score_mask": mask},
                {
                    **rr._source_tier_labels(store.manifest),
                    "fold": fold,
                    "evaluation_date_indices": evaluation[fold].tolist(),
                    **metadata,
                },
            )
            evaluated = rr._evaluate(
                store=store,
                indices=evaluation[fold],
                scores=scores,
                score_mask=mask,
                cdi=cdi,
                bova11_close_by_index=bova.close_by_session,
                bova11_binding=bova_record,
                lending_borrow=lending,
                source_hashes=sources,
                fold=fold,
                output=output / "evaluation.json",
                execution_policy=policy,
            )
            reports.setdefault(name, {})[fold] = evaluated
            artifacts.setdefault(name, {})[fold] = (
                rr._existing_score_and_evaluation_record(output)
            )
            return evaluated

        for fold in FOLDS:
            members = {}
            masks = {}
            for arm, seeds in (("arm_B", SEEDS), ("fast_off", PAIRED_SEEDS)):
                for seed in seeds:
                    run = root / "trajectories" / arm / f"{fold}_seed_{seed}"
                    manifest = _completed(
                        run, stage="F", seed=seed, fold=fold, disabled=arm == "fast_off"
                    )
                    if (
                        manifest["pretrain_checkpoint_sha256"]
                        != stage_p[str(seed)]["artifacts"]["raw_patience.pt"]
                    ):
                        raise ValueError(
                            "fine-tuning did not use its paired Stage-P checkpoint"
                        )
                    if (
                        manifest["checkpoint_input_contract"]["implementation_commit"]
                        != design["implementation"]["commit"]
                    ):
                        raise ValueError("training implementation differs from freeze")
                    scores, mask = rr._score_artifact(
                        run / "scores",
                        require_clean_transfer=True,
                        expected_dates=dates[evaluation[fold]],
                        expected_isins=store.isins,
                        expected_feature_schema_sha256=design["feature_schema_sha256"],
                    )
                    if masks and not np.array_equal(mask, next(iter(masks.values()))):
                        raise ValueError("paired seed/arm masks differ")
                    members[arm, seed], masks[arm, seed] = scores, mask
                    score_manifest = rr._read_json(run / "scores/score_manifest.json")
                    gate = score_manifest["gate_diagnostics"]
                    if sha256_file(run / "scores" / gate["path"]) != gate["sha256"]:
                        raise ValueError("gate diagnostic hash differs")
                    history = json.loads((run / "history.json").read_text())
                    if not all("branch_gradient_norms" in row for row in history):
                        raise ValueError("trajectory lacks branch gradient diagnostics")
                    trajectories[f"{arm}/{fold}/{seed}"] = {
                        "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
                        "selected_epoch": manifest["selected_epoch"],
                        "epochs_completed": manifest["epochs_completed"],
                        "gates": rr._read_json(run / "scores" / gate["path"]),
                        "history": history,
                    }
            mask = masks["arm_B", 11]
            panels = {}
            for name, arm, seeds in (
                ("B3", "arm_B", PAIRED_SEEDS),
                ("B6", "arm_B", SEEDS),
                ("fast_off", "fast_off", PAIRED_SEEDS),
            ):
                scores = rr.rank_average_ensemble(
                    [members[arm, seed] for seed in seeds], mask
                )
                panels[name] = scores
                evaluate_panel(
                    name,
                    fold,
                    scores,
                    mask,
                    {
                        "engine": "starter_network",
                        "arm": arm,
                        "seeds": list(seeds),
                        "seed_aggregation": "tie-aware rank average",
                    },
                )
            for name, path in (
                ("gbdt", parent_root / "gbdt_ladder/b_intraday" / fold),
                ("momentum", parent_root / "baselines/momentum_12_1" / fold),
            ):
                scores, other_mask = rr._score_artifact(
                    path, require_clean_transfer=True
                )
                if not np.array_equal(mask, other_mask):
                    raise ValueError("comparator mask differs from network")
                panels[name] = scores
                evaluate_panel(
                    name,
                    fold,
                    scores,
                    mask,
                    {"engine": "sealed_round1_comparator", "source_root": str(path)},
                )
            for name in ("B3", "B6", "fast_off"):
                scores = rr.rank_average_ensemble((panels[name], panels["gbdt"]), mask)
                evaluate_panel(
                    f"ensemble_{name}",
                    fold,
                    scores,
                    mask,
                    {
                        "engine": "equal_weight_rank_average",
                        "members": [name, "gbdt"],
                        "weights": [0.5, 0.5],
                    },
                )
            reference = reports["B3"][fold].inputs
            values = {}
            for seed in SEEDS:
                inputs = replace(
                    reference, scores=members["arm_B", seed], score_mask=mask
                )
                components = _primary_population_components(inputs)
                _, daily, _ = _primary_daily_metrics(*components, inputs.dates)
                values[str(seed)] = (
                    float(np.nanmean(daily)) if np.isfinite(daily).any() else None
                )
            dispersion[fold] = {
                "seed_IC": values,
                **{
                    label: {
                        "defined_seeds": len(finite),
                        "standard_deviation": float(np.std(finite, ddof=1))
                        if len(finite) >= 2
                        else None,
                    }
                    for label, seeds in (
                        ("three_seed", PAIRED_SEEDS),
                        ("six_seed", SEEDS),
                    )
                    for finite in [
                        [
                            values[str(seed)]
                            for seed in seeds
                            if values[str(seed)] is not None
                        ]
                    ]
                },
            }
            print(
                f"Round 3 evaluated {fold}: all registered panels and comparators",
                flush=True,
            )

        readouts = {name: rr._pooled_readouts(value) for name, value in reports.items()}
        comparisons = [
            ("B3", "fast_off"),
            ("B6", "B3"),
            ("B6", "gbdt"),
            ("B6", "momentum"),
            ("ensemble_B6", "gbdt"),
            ("ensemble_B6", "B6"),
            ("ensemble_B6", "momentum"),
        ]
        paired = {
            f"{left}_minus_{right}": rr._paired_readouts(reports[left], reports[right])
            for left, right in comparisons
        }
        per_horizon = paired_horizons(reports["B3"], reports["fast_off"])
        six_ic = readouts["B6"]["pooled"]["primary_neutral_target_ic"]["estimate"]
        three_ic = readouts["B3"]["pooled"]["primary_neutral_target_ic"]["estimate"]
        keep_six = six_ic is not None and three_ic is not None and six_ic >= three_ic
        sterile_arrays = []
        for fold, value in reports["B6"].items():
            headline = rr._scenario_daily_rows(value, "borrow_balance")
            sterile = rr._scenario_daily_rows(value, "comparator_sterile_proceeds")
            summaries = {
                row["scenario"]: row
                for row in value.result.report["economics"]["summaries"]
            }
            unresolved = any(
                summaries[scenario]["economics_unresolved"]
                for scenario in ("borrow_balance", "comparator_sterile_proceeds")
            )
            sterile_arrays.append(
                np.asarray(
                    [
                        np.nan
                        if unresolved
                        else b["net_excess_all_cash_bps"] - a["net_excess_all_cash_bps"]
                        for a, b in zip(headline, sterile, strict=True)
                    ]
                )
            )
        sterile_delta = rr._folded_bootstrap(tuple(sterile_arrays))
        designation, designation_detail = rr._weighted_candidate_designation(
            {
                "network": reports["B6"],
                "gbdt": reports["gbdt"],
                "ensemble": reports["ensemble_B6"],
            },
            exact_tie_priority=("network", "gbdt", "ensemble"),
        )
        result = {
            "schema": SCHEMA,
            "status": "completed",
            **rr.RESEARCH_FLAGS,
            **rr._source_tier_labels(store.manifest),
            "implementation": design["implementation"],
            "completed_at_utc": rr._utc_now(),
            "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
            "sources": sources,
            "store_access": access,
            "execution_policy": binding,
            "stage_p": stage_p,
            "trajectory_diagnostics": trajectories,
            "fast_present_coverage": {
                fold: {
                    "absent_active_name_days": counts[0],
                    "present_active_name_days": counts[1],
                    "present_fraction": counts[1] / sum(counts)
                    if sum(counts)
                    else None,
                }
                for fold in FOLDS
                for counts in [
                    trajectories[f"arm_B/{fold}/11"]["gates"][
                        "archive_fast_present_counts"
                    ]
                ]
            },
            "artifacts": artifacts,
            "readouts": readouts,
            "paired_deltas": paired,
            "ablation_paired_per_horizon": per_horizon,
            "seed_dispersion": dispersion,
            "seed_decision": {
                "keep_six": keep_six,
                "parent": "B6" if keep_six else "B3",
                "six_IC": six_ic,
                "three_IC": three_ic,
            },
            "parent_comparison": {
                "designation": designation,
                "detail": designation_detail,
            },
            "sterile_minus_headline": sterile_delta,
            "validation_2025_rule": validation_readout(readouts["B6"], sterile_delta),
        }
        write_json_atomic(
            root / "economics_detail.json", rr._round1_economics_detail(reports)
        )
        return write_json_atomic(root / "round3_result.json", result)
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--round1-root", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)
    freezing.add_argument("--max-parallel", type=int, default=6)
    for command in ("plan-smoke", "plan-p", "plan-main", "finalize"):
        commands.add_parser(command).add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        digest = freeze(
            round1_root=args.round1_root,
            output=args.output,
            max_parallel=args.max_parallel,
        )
    elif args.command.startswith("plan-"):
        digest = write_plan(root=args.root, phase=args.command[5:])
    else:
        digest = finalize(root=args.root)
    print(digest)


if __name__ == "__main__":
    main()
