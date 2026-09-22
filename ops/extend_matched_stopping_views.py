"""Smoothed-centre parent selection views on the executed patience-20 trajectories.

The matched stopping experiment executed six attention parent trajectories under
patience 20 and labelled two selection views on each: the patience-5 prefix and
the full patience-20 selection. This driver adds views that apply the trailing
three-epoch smoothed selection score to the same saved histories, binds the saved
epoch files those views choose, and freezes a children plan on the same eight
periods, seeds and child recipes. A child whose selected parent epoch equals an
existing view's epoch reuses that view's completed child through an explicit
alias; the others are trained once the GPU worker has finished. Smoothing, like
patience, touches only selection, never gradients, schedule, sampler or RNG.

    uv run --project research --no-sync python ops/extend_matched_stopping_views.py --freeze
    uv run --project research --no-sync python ops/extend_matched_stopping_views.py

Views: ``s3`` scans the whole executed trajectory (truncated wherever patience 20
ended it); ``s3p5`` applies patience 5 to the smoothed score, so it sees only its
own prefix. Evaluate the children with ``ops/replay_forecast_variants.py``.
"""

import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.forecast_variants import epoch_checkpoint
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_training import TrainingRecipe, train
from brazil_rv.v2.selection_rules import smoothed_view
from brazil_rv.v2.train import compile_forward

PROJECT = Path(__file__).resolve().parents[1]
POINTER = PROJECT / "docs/v2_economic_data_scaling_run.json"
PLAN_KEY = "scaling_smoothed_views_plan"
VIEWS = {"s3": {"window": 3, "patience": None}, "s3p5": {"window": 3, "patience": 5}}


def runtime_files():
    model = Path(inspect.getfile(CharacteristicModel))
    return {
        "training": Path(inspect.getfile(train)),
        "model": model,
        "temporal": model.with_name("temporal_pathway.py"),
        "configuration": Path(inspect.getfile(configuration)),
        "compiler": Path(inspect.getfile(compile_forward)),
    }


def parent_views(parent, manifest, history, existing, rules, minimum_improvement):
    """Smoothed views of one executed parent, each binding a sealed epoch file."""
    views = {}
    for rule in rules:
        spec = VIEWS[rule]
        view = smoothed_view(
            history,
            window=spec["window"],
            patience=spec["patience"],
            minimum_improvement=minimum_improvement if spec["patience"] else 0.0,
        )
        epoch = view["selected_epoch"]
        checkpoint, digest = epoch_checkpoint(parent, manifest, epoch)
        same = [
            name
            for name, record in existing.items()
            if record["selected_epoch"] == epoch
        ]
        views[rule] = {
            **view,
            "checkpoint": {"path": str(checkpoint), "sha256": digest},
            "executed_trajectory": binding(parent / "run_manifest.json"),
            "selection_view": True,
            "independent_fit": False,
            "same_epoch_as": same,
        }
    return views


def freeze(run, args):
    stopping_ref = run["scaling_matched_stopping_plan"]
    stopping = bound_json(stopping_ref)
    root = Path(stopping["root"])
    progress = json.loads((root / "refits.json").read_text())
    done = {r["key"] for r in progress["completed"]}
    rules = args.view
    out = root / "smoothed_views"
    out.mkdir(exist_ok=False)
    views = {}
    for cell in stopping["cells"]:
        for seed in stopping["seeds"]:
            if f"{cell}/P/{seed}" not in done:
                raise SystemExit(f"parent {cell}/P/{seed} has not completed")
            parent = root / "parents" / cell / f"P_seed_{seed}"
            manifest = bound_json(binding(parent / "run_manifest.json"))
            history = json.loads((parent / "history.json").read_text())
            assert (
                sha256_file(parent / "history.json")
                == manifest["artifacts"]["history.json"]
            )
            existing = bound_json(binding(parent / "stopping_views.json"))["views"]
            record = parent_views(
                parent,
                manifest,
                history,
                existing,
                rules,
                stopping["p_recipe"]["minimum_improvement"],
            )
            path = out / "views" / cell / f"P_seed_{seed}" / "views.json"
            write_json_atomic(
                path,
                {
                    "plan": stopping_ref,
                    "existing_views": binding(parent / "stopping_views.json"),
                    "views": record,
                },
            )
            views[f"{cell}/{seed}"] = binding(path)
    arms = {
        f"{cell}_{rule}": {"reference": cell, "view": rule}
        for cell in stopping["cells"]
        for rule in rules
    }
    plan = {
        "status": "frozen_before_smoothed_view_outcomes",
        "root": str(out),
        "source": stopping_ref,
        "store": stopping["store"],
        "cells": stopping["cells"],
        "arms": arms,
        "views": views,
        "view_rules": {rule: VIEWS[rule] for rule in rules},
        "seeds": stopping["seeds"],
        "folds": stopping["folds"],
        "f_recipes": stopping["f_recipes"],
        "maximum_epochs": stopping["maximum_epochs"],
        "ema_half_life_epochs": stopping["ema_half_life_epochs"],
        "maximum_child_fits": len(arms)
        * len(stopping["seeds"])
        * len(stopping["folds"]),
        "runtime": {
            "git": _git_identity(),
            "files": {k: binding(v) for k, v in runtime_files().items()},
        },
        "driver": binding(Path(__file__)),
        "registration": binding(
            PROJECT / "research/preregistrations/v2_posthoc_selection_and_ensembles.md"
        ),
        "scope": (
            "Additional labelled selection views of the six executed patience-20 "
            "attention parents: the trailing-3 smoothed selection score selects the "
            "centre epoch of its best window, over the whole executed trajectory "
            "(s3) or under patience 5 on the smoothed score (s3p5). Same store, "
            "seeds, eight periods, child recipes, 60-epoch maximum/schedule, "
            "optimizer, selector population and fold boundaries. No new parent "
            "trajectory, no best-seed choice, no reserved fold."
        ),
        "exact_reuse": (
            "A view whose epoch equals the patience-5 or patience-20 view of the "
            "same parent binds that view's completed child by explicit alias; "
            "identical selected parent, recipe, seed and fold mean an identical fit."
        ),
        "comparisons": (
            "Evaluate with ops/replay_forecast_variants.py: paired fold deltas of "
            "each smoothed-view arm against its patience-20 reference on the same "
            "dates, account and policy; block 40 primary; IC diagnostic."
        ),
    }
    write_json_atomic(out / "plan.json", plan)
    write_json_atomic(out / "frozen_design.json", {"store": stopping["store"]})
    run[PLAN_KEY] = binding(out / "plan.json")
    write_json_atomic(POINTER, run)
    print(
        json.dumps(
            {
                "plan": run[PLAN_KEY],
                "arms": list(arms),
                "views": {
                    key: {
                        rule: {
                            "epoch": v["selected_epoch"],
                            "same_epoch_as": v["same_epoch_as"],
                        }
                        for rule, v in bound_json(ref)["views"].items()
                    }
                    for key, ref in views.items()
                },
            },
            indent=1,
        ),
        flush=True,
    )


def alias(output, target):
    """An explicit logical alias of a completed fit; never a copy or a modification."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return
    if sys.platform == "win32":
        quoted = ["'" + str(p).replace("'", "''") + "'" for p in (output, target)]
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"New-Item -ItemType Junction -Path {quoted[0]} -Target {quoted[1]} | Out-Null",
            ],
            check=True,
        )
    else:
        os.symlink(target, output, target_is_directory=True)
    assert output.resolve() == target.resolve()


def gpu_is_free(run, force):
    stopping = bound_json(run["scaling_matched_stopping_plan"])
    busy = []
    progress = Path(stopping["root"]) / "refits.json"
    if json.loads(progress.read_text())["status"] != "complete":
        busy.append(str(progress))
    if "scaling_common_model_plan" in run:
        common = (
            Path(bound_json(run["scaling_common_model_plan"])["root"]) / "refits.json"
        )
        if common.exists() and json.loads(common.read_text())["status"] != "complete":
            busy.append(str(common))
    if busy and not force:
        raise SystemExit(
            "another fit queue is incomplete; never compete for the 6 GB GPU: "
            + ", ".join(busy)
        )
    return not busy


def execute(run, force=False):
    reference = run[PLAN_KEY]
    plan = bound_json(reference)
    assert sha256_file(Path(__file__)) == plan["driver"]["sha256"]
    for key, path in runtime_files().items():
        assert sha256_file(path) == plan["runtime"]["files"][key]["sha256"], key
    gpu_is_free(run, force)
    stopping = bound_json(plan["source"])
    stopping_root = Path(stopping["root"])
    store, root = Path(plan["store"]["root"]), Path(plan["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    torch.set_num_threads(1)
    progress = root / "refits.json"
    completed = (
        json.loads(progress.read_text())["completed"] if progress.exists() else []
    )
    done = {r["key"] for r in completed}

    def save(row, status="running"):
        completed.append(row)
        done.add(row["key"])
        write_json_atomic(
            progress,
            {
                "status": status,
                "plan": reference,
                "completed": completed,
                "logical_jobs": plan["maximum_child_fits"],
                "actual_fits": sum(not x.get("reused", False) for x in completed),
            },
        )
        print(json.dumps(row), flush=True)

    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            for arm, spec in plan["arms"].items():
                key = f"{arm}/F/{fold}/{seed}"
                if key in done:
                    continue
                cell, rule = spec["reference"], spec["view"]
                view = bound_json(plan["views"][f"{cell}/{seed}"])["views"][rule]
                output = root / "fits" / arm / f"{fold}_seed_{seed}"
                if view["same_epoch_as"]:
                    patience = view["same_epoch_as"][0]
                    target = (
                        stopping_root / "fits" / f"{cell}_p{patience}" / output.name
                    )
                    manifest = target / "run_manifest.json"
                    if not manifest.exists():
                        raise SystemExit(
                            f"reusable child is not complete yet: {target}"
                        )
                    alias(output, target)
                    save(
                        {
                            "key": key,
                            "reused": True,
                            "source": str(target),
                            "manifest": binding(manifest),
                            "seconds": 0,
                            "reason": "Identical selected parent epoch, child recipe, seed and fold",
                        }
                    )
                    continue
                if (output / "run_manifest.json").exists():
                    result = bound_json(binding(output / "run_manifest.json"))
                    assert result["status"] == "completed"
                    save(
                        {
                            "key": key,
                            "manifest": binding(output / "run_manifest.json"),
                            "parent": view,
                            "seconds": 0,
                        }
                    )
                    continue
                if shutil.disk_usage(root).free < 2_000_000_000:
                    raise RuntimeError(
                        "less than 2 GB free before fit; reclaim space first"
                    )
                parent = Path(view["checkpoint"]["path"])
                assert sha256_file(parent) == view["checkpoint"]["sha256"]
                print(
                    json.dumps({"starting": key, "completed": len(completed)}),
                    flush=True,
                )
                torch._dynamo.reset()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                tick = perf_counter()
                train(
                    store,
                    output,
                    cell=stopping["cells"][cell],
                    stage="F",
                    fold=fold,
                    seed=seed,
                    epochs=plan["maximum_epochs"],
                    recipe=TrainingRecipe(**plan["f_recipes"][cell]),
                    parent=parent,
                    parent_sha256=view["checkpoint"]["sha256"],
                    compiled=True,
                    export_scores=True,
                    ema_half_life_epochs=plan["ema_half_life_epochs"],
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                save(
                    {
                        "key": key,
                        "manifest": binding(output / "run_manifest.json"),
                        "parent": view,
                        "seconds": perf_counter() - tick,
                    }
                )
    write_json_atomic(
        progress,
        {
            "status": "complete",
            "plan": reference,
            "completed": completed,
            "logical_jobs": plan["maximum_child_fits"],
            "actual_fits": sum(not x.get("reused", False) for x in completed),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "--view", action="append", choices=sorted(VIEWS), help="default: s3"
    )
    parser.add_argument("--force", action="store_true", help="skip the busy-GPU guard")
    args = parser.parse_args()
    if not args.view:
        args.view = ["s3"]
    run = json.loads(POINTER.read_text())
    freeze(run, args) if args.freeze else execute(run, args.force)
