"""Fold-bounded Round-6 ensembles and matched parent comparisons."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import candidate_readout, paired_readouts, retained
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, TRADED_PRIMARY_HORIZONS
from .evaluate import _primary_daily_metrics, _primary_population_components
from .data_roots import resolve_external_root
from .research_checkpoint import _completed, _context_arguments, _finish_cell
from .research_diagnostics import momentum_diagnostics, pooled_momentum_diagnostics
from .round6 import (
    SESSION1,
    SESSION1_FAMILIES,
    completed,
    derive_session2_roster,
    resolve_file,
    trajectory,
)


def evaluation_design(design: dict) -> dict:
    # Evaluation uses the unchanged parent arrays and repaired Round-5 hedge.
    # The training store's exact protected-array proof authorizes this reuse.
    result = copy.deepcopy(design["evaluation_sources"]["design"])
    for record, key in (
        (result["store"], "root"),
        (result["bova11"], "root"),
        (result["bova11"], "hedge_beta_root"),
        (result["lending_archive"], "root"),
        (result["execution_policy"], "root"),
    ):
        record[key] = str(resolve_external_root(record[key])[0])
    for record in result["cdi"].values():
        record["path"] = str(resolve_file(record))
    return result


def informative_folds(design: dict, arm: str, roster: dict | None = None) -> list[str]:
    coverage = design["input_coverage"]["folds"]
    if arm in coverage:
        return coverage[arm]
    if arm == "best_single_fresh_p":
        return coverage[roster["best_single"]]
    if arm in ("C6", "C6_fresh_p"):
        return [
            f
            for f in DEVELOPMENT_FOLDS
            if any(f in coverage[a] for a in roster["c6_families"])
        ]
    return list(DEVELOPMENT_FOLDS)


def ensemble(
    design: dict,
    root: Path,
    arm: str,
    fold: str,
    seeds,
    dates,
    isins,
    *,
    roster=None,
    ablation=None,
):
    members, mask, records = [], None, []
    parent_root = None
    inventory_files = {}
    if arm == "S0" and any(seed in ALLOWED_SEEDS for seed in seeds):
        parent_root = resolve_external_root(design["s0_panels"]["root"])[0]
        inventory_path = parent_root / "artifact_inventory.json"
        if sha256_file(inventory_path) != design["s0_panels"]["inventory_sha256"]:
            raise ValueError("parent score inventory changed")
        inventory_files = {
            record["path"]: record for record in rr._read_json(inventory_path)["files"]
        }
    for seed in seeds:
        run = trajectory(root, arm, seed, fold)
        if arm == "S0" and seed in ALLOWED_SEEDS:
            run = trajectory(parent_root, arm, seed, fold)
            manifest = rr._read_json(run / "run_manifest.json")
            rr._assert_current_clean_training(manifest, path=run / "run_manifest.json")
            # Every reused byte is bound by the sealed, registered inventory.
            relative = run.relative_to(parent_root).as_posix()
            for name in (
                "run_manifest.json",
                "scores/score_manifest.json",
                "scores/scores.npy",
                "scores/score_mask.npy",
                "scores/date_index.npy",
                "scores/isin_index.npy",
            ):
                record = inventory_files[f"{relative}/{name}"]
                digest = record["sha256"]
                if sha256_file(run / name) != digest:
                    raise ValueError("sealed parent artifact changed")
        else:
            manifest = completed(run, design, arm, "F", seed, fold, roster=roster)
        directory = run / "scores" if ablation is None else run / "ablations" / ablation
        score, member_mask = rr._score_artifact(
            directory,
            require_clean_transfer=True,
            expected_dates=dates,
            expected_isins=isins,
            expected_feature_schema_sha256=rr._read_json(
                resolve_external_root(design["s0_store"]["root"])[0] / "manifest.json"
            )["feature_schema_sha256"]
            if arm == "S0" and seed in ALLOWED_SEEDS
            else design["feature_schema_sha256"],
        )
        if mask is not None and not np.array_equal(mask, member_mask):
            raise ValueError("Round-6 seed populations differ")
        mask = member_mask
        members.append(score)
        records.append(
            {
                "seed": seed,
                "selected_epoch": manifest["selected_epoch"],
                "epochs_completed": manifest["epochs_completed"],
                "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
                "score_manifest_sha256": sha256_file(directory / "score_manifest.json"),
            }
        )
    return rr.rank_average_ensemble(members, mask), mask, records


def evaluate(root: Path, arms, *, seeds=ALLOWED_SEEDS, group="session1", roster=None):
    design = rr._read_json(root / "frozen_design.json")
    original = evaluation_design(design)
    context = rr._open_ledger_replay(original)
    policy, _ = rr.load_selected_policy(
        Path(original["execution_policy"]["root"]),
        expected_result_sha256=original["execution_policy"]["result_sha256"],
    )
    source_hashes = {
        "v2_store_manifest": original["store"]["manifest_sha256"],
        "round6_training_store": design["store"]["manifest_sha256"],
        "round6_frozen_design": sha256_file(root / "frozen_design.json"),
    }
    paths = {arm: {} for arm in arms}
    try:
        # Fold outermost keeps warm arrays useful across all candidates.
        for fold in DEVELOPMENT_FOLDS:
            for arm in arms:
                output = root / "aggregates" / group / arm / fold
                paths[arm][fold] = output
                if _completed(output):
                    metadata = rr._read_json(output / "score_manifest.json")["metadata"]
                    if metadata["seeds"] != list(seeds) or metadata["arm"] != arm:
                        raise ValueError("completed aggregate uses another panel")
                    continue
                if (output / "score_manifest.json").exists():
                    raise RuntimeError(
                        f"partial aggregate requires diagnosis: {output}"
                    )
                scores, mask, records = ensemble(
                    design,
                    root,
                    arm,
                    fold,
                    seeds,
                    context.store.dates[context.evaluation[fold]],
                    context.store.isins,
                    roster=roster,
                )
                rr._persist_scores(
                    output,
                    {"scores": scores, "score_mask": mask},
                    {
                        **rr._source_tier_labels(context.store.manifest),
                        "arm": arm,
                        "fold": fold,
                        "seeds": list(seeds),
                        "evaluation_date_indices": context.evaluation[fold].tolist(),
                        "trajectories": records,
                        "round6_frozen_design_sha256": sha256_file(
                            root / "frozen_design.json"
                        ),
                    },
                )
                evaluated = rr._evaluate(
                    **_context_arguments(context, source_hashes),
                    indices=context.evaluation[fold],
                    scores=scores,
                    score_mask=mask,
                    fold=fold,
                    output=output / "evaluation.json",
                    execution_policy=policy,
                    settle_terminal_residuals=True,
                )
                _finish_cell(output, evaluated, name=arm, fold=fold)
                del evaluated
                print(f"accepted {group}/{arm}/{fold}", flush=True)
        readouts = {a: candidate_readout(p) for a, p in paths.items()}
        pairs = {}
        support = {a: informative_folds(design, a, roster) for a in arms}
        if "S0" in arms:
            for arm in arms:
                if arm != "S0":
                    pairs.update(
                        paired_readouts(
                            context,
                            {arm: paths[arm], "S0": paths["S0"]},
                            root / "paired" / group,
                            informative_folds=support,
                        )
                    )
        # Reuse the protected momentum panels; no baseline refit or sweep.
        baseline_root = resolve_external_root(original["cpu_checkpoint"]["root"])[0]
        momentum = {a: {} for a in arms}
        for fold in DEVELOPMENT_FOLDS:
            m_score, m_mask = rr._score_artifact(
                baseline_root / "baselines/momentum_12_1" / fold,
                require_clean_transfer=True,
            )
            for arm in arms:
                candidate = retained(context, paths[arm][fold], fold)
                momentum[arm][fold] = momentum_diagnostics(
                    candidate.inputs, m_score, m_mask
                )
                del candidate
        result = {
            "schema": "BRAZIL_RV_ROUND6_READOUT_V1",
            "status": "completed",
            "group": group,
            "seeds": list(seeds),
            "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
            "readouts": readouts,
            "paired": pairs,
            "momentum_diagnostics": {
                a: {"pooled": pooled_momentum_diagnostics(v), "folds": v}
                for a, v in momentum.items()
            },
            **rr.RESEARCH_FLAGS,
        }
        path = root / f"{group}_result.json"
        if path.exists():
            raise FileExistsError(path)
        return write_json_atomic(path, result)
    finally:
        context.store.close()


def attribution_reference_matches(metadata: dict, records, seeds, arm: str) -> bool:
    return (
        metadata["seeds"] == list(seeds)
        and metadata["arm"] == arm
        and [r["run_manifest_sha256"] for r in metadata["trajectories"]]
        == [r["run_manifest_sha256"] for r in records]
    )


def attribution(root: Path, arms, *, ablation, reference_group, seeds=ALLOWED_SEEDS):
    """Paired forecast attribution; no refit, new candidate or portfolio gate."""
    if not ablation or not reference_group:
        raise ValueError(
            "attribution requires a family/all and a trained reference group"
        )
    design = rr._read_json(root / "frozen_design.json")
    roster = (
        rr._read_json(root / "session2_roster.json")
        if (root / "session2_roster.json").exists()
        else None
    )
    output = root / "attribution" / reference_group / ablation
    output.mkdir(parents=True, exist_ok=False)
    context = rr._open_ledger_replay(evaluation_design(design))
    daily = {a: {} for a in arms}
    try:
        for fold in DEVELOPMENT_FOLDS:
            ix = context.evaluation[fold]
            for arm in arms:
                trained_path = root / "aggregates" / reference_group / arm / fold
                metadata = rr._read_json(trained_path / "score_manifest.json")[
                    "metadata"
                ]
                scores, mask, records = ensemble(
                    design,
                    root,
                    arm,
                    fold,
                    seeds,
                    context.store.dates[ix],
                    context.store.isins,
                    roster=roster,
                    ablation=ablation,
                )
                if not attribution_reference_matches(metadata, records, seeds, arm):
                    raise ValueError(
                        "attribution reference uses different trained models"
                    )
                trained = retained(context, trained_path, fold)
                np.testing.assert_array_equal(mask, trained.inputs.score_mask)
                invalid_inputs = replace(trained.inputs, scores=scores, score_mask=mask)
                components = _primary_population_components(
                    invalid_inputs, TRADED_PRIMARY_HORIZONS
                )
                _, invalid_ic, _ = _primary_daily_metrics(
                    *components, invalid_inputs.dates, TRADED_PRIMARY_HORIZONS
                )
                invalid_result = replace(
                    trained.result,
                    daily_primary_ic=invalid_ic,
                    primary_scores=components[0],
                    primary_targets=components[1],
                    primary_outcome_mask=components[2],
                    primary_score_mask=components[3],
                )
                delta, population = rr._paired_primary_daily(
                    trained.result, invalid_result
                )
                daily[arm][fold] = delta
                write_json_atomic(
                    output / arm / f"{fold}.json",
                    {
                        "arm": arm,
                        "fold": fold,
                        "ablation": ablation,
                        "seeds": list(seeds),
                        "direction": "trained_minus_forced_invalid",
                        "source_score_manifests": records,
                        "trained_aggregate_sha256": sha256_file(
                            trained_path / "score_manifest.json"
                        ),
                        "population": population,
                        "daily_primary_ic_delta": [
                            float(v) if np.isfinite(v) else None for v in delta
                        ],
                        "promotion_weight": 0,
                    },
                )
                del trained, invalid_inputs, invalid_result
        summary = {}
        for arm, values in daily.items():
            informative = informative_folds(design, arm, roster)
            summary[arm] = {
                "all_folds": rr._folded_bootstrap(tuple(values.values())),
                "informative_folds": informative,
                "informative": rr._folded_bootstrap(
                    tuple(values[f] for f in informative)
                ),
                "folds": {f: rr._folded_bootstrap((v,)) for f, v in values.items()},
            }
        return write_json_atomic(
            output / "result.json",
            {
                "schema": "BRAZIL_RV_ROUND6_ATTRIBUTION_V1",
                "status": "completed",
                "ablation": ablation,
                "reference_group": reference_group,
                "seeds": list(seeds),
                "readouts": summary,
                "promotion_weight": 0,
                "direction": "trained_minus_forced_invalid",
                "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def freeze_roster(root: Path):
    result_path = root / "session1_result.json"
    result = rr._read_json(result_path)
    if (
        result["status"] != "completed"
        or result["seeds"] != list(ALLOWED_SEEDS)
        or set(result["readouts"]) != set((*SESSION1, "S0"))
    ):
        raise ValueError("session 2 requires all nine accepted session-1 arms and S0")
    points = {
        a: result["paired"][f"{a}_minus_S0"]["informative_subsets"][a]["pooled"][
            "primary_neutral_target_ic"
        ]["estimate"]
        for a in SESSION1_FAMILIES
    }
    roster = {
        **derive_session2_roster(points),
        "session1_result_sha256": sha256_file(result_path),
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
    }
    path = root / "session2_roster.json"
    if path.exists():
        raise FileExistsError(path)
    return write_json_atomic(path, roster)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "roster", "attribution"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arms", nargs="+", default=["S0", *SESSION1])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ALLOWED_SEEDS))
    parser.add_argument("--group", default="session1")
    parser.add_argument("--ablation")
    parser.add_argument("--reference-group")
    args = parser.parse_args()
    if args.action == "roster":
        print(freeze_roster(args.root))
    elif args.action == "attribution":
        print(
            attribution(
                args.root,
                args.arms,
                ablation=args.ablation,
                reference_group=args.reference_group,
                seeds=tuple(args.seeds),
            )
        )
    else:
        roster = (
            rr._read_json(args.root / "session2_roster.json")
            if (args.root / "session2_roster.json").exists()
            else None
        )
        print(
            evaluate(
                args.root,
                args.arms,
                seeds=tuple(args.seeds),
                group=args.group,
                roster=roster,
            )
        )


if __name__ == "__main__":
    main()
