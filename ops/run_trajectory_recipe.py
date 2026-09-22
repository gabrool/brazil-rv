"""Short fully annealed F schedule from existing parents; selection unchanged.

Under the matched stopping recipe an F child runs a 60-epoch cosine schedule with
patience 5, so the raw selection lands on a near-peak-learning-rate iterate at
epochs three to six. This one-factor wave keeps everything else and changes only
the child's schedule: ``schedule_epochs = epochs = --f-epochs`` (default 8) with
patience equal to that budget, so the trajectory anneals fully and the selector
compares low-learning-rate iterates. Parents are the existing patience-20
attention selection views, or the common-model C6/GRU parents once trained; no
new P fit is run. Learning rate, SAM, transferred-parameter multiplier, loss,
selector population, store and fold boundaries are inherited unchanged.

    uv run --project research --no-sync python ops/run_trajectory_recipe.py \
        --freeze --cell TE_full
    uv run --project research --no-sync python ops/run_trajectory_recipe.py

At most two cells per wave. Children land under ``<attention root>/f_schedule/
fits/<cell>_f8/<fold>_seed_<seed>`` and are evaluated with
``ops/replay_forecast_variants.py`` (arm ``<cell>_f8``), whose post-hoc views
(centre3/top3/around3) also apply to these shorter trajectories. Execution
refuses to start while another fit queue is incomplete on the 6 GB GPU.
"""

import argparse
import gc
import json
from pathlib import Path
import shutil
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7_training import TrainingRecipe, train
from extend_matched_stopping_views import gpu_is_free, runtime_files

PROJECT = Path(__file__).resolve().parents[1]
POINTER = PROJECT / "docs/v2_economic_data_scaling_run.json"
PLAN_KEY = "scaling_f_schedule_plan"


def short_schedule_recipe(f_recipe, f_epochs):
    """Only the schedule length and the patience change; every other field is kept."""
    child = {**f_recipe, "schedule_epochs": int(f_epochs), "patience": int(f_epochs)}
    TrainingRecipe(**child)
    return child


def parents_for(run, stopping, cell):
    """Hash-bound parent checkpoints per seed for one cell; None while untrained."""
    root = Path(stopping["root"])
    parents = {}
    for seed in stopping["seeds"]:
        if cell in stopping["cells"]:
            views = bound_json(
                binding(
                    root / "parents" / cell / f"P_seed_{seed}" / "stopping_views.json"
                )
            )["views"]
            parents[str(seed)] = dict(views["20"]["checkpoint"], view="p20")
        else:
            common = bound_json(run["scaling_common_model_plan"])
            path = (
                Path(common["root"]) / "fits" / cell / f"P_seed_{seed}" / "selected.pt"
            )
            if not path.exists():
                return None
            parents[str(seed)] = dict(binding(path), view="selected")
    return parents


def freeze(run, args):
    stopping_ref = run["scaling_matched_stopping_plan"]
    stopping = bound_json(stopping_ref)
    common = (
        bound_json(run["scaling_common_model_plan"])
        if "scaling_common_model_plan" in run
        else {"cells": {}, "f_recipes": {}}
    )
    cells = {**common["cells"], **stopping["cells"]}
    f_recipes = {**common["f_recipes"], **stopping["f_recipes"]}
    chosen = args.cell
    if not chosen or len(chosen) > 2 or set(chosen) - set(cells):
        raise SystemExit(f"choose one or two cells from {sorted(cells)}")
    parents = {}
    for cell in chosen:
        found = parents_for(run, stopping, cell)
        if found is None:
            raise SystemExit(f"{cell} parents are not trained yet")
        parents[cell] = found
    root = Path(stopping["root"]) / "f_schedule"
    root.mkdir(exist_ok=False)
    arms = {f"{cell}_f{args.f_epochs}": {"reference": cell} for cell in chosen}
    plan = {
        "status": "frozen_before_short_schedule_outcomes",
        "root": str(root),
        "source": stopping_ref,
        "common_source": run.get("scaling_common_model_plan"),
        "store": stopping["store"],
        "cells": {cell: cells[cell] for cell in chosen},
        "arms": arms,
        "parents": parents,
        "seeds": stopping["seeds"],
        "folds": stopping["folds"],
        "f_epochs": int(args.f_epochs),
        "f_recipes": {
            cell: short_schedule_recipe(f_recipes[cell], args.f_epochs)
            for cell in chosen
        },
        "source_f_recipes": {cell: f_recipes[cell] for cell in chosen},
        "ema_half_life_epochs": stopping["ema_half_life_epochs"],
        "planned_fits": len(chosen) * len(stopping["seeds"]) * len(stopping["folds"]),
        "runtime": {
            "git": _git_identity(),
            "files": {k: binding(v) for k, v in runtime_files().items()},
        },
        "driver": binding(Path(__file__)),
        "registration": binding(
            PROJECT / "research/preregistrations/v2_posthoc_selection_and_ensembles.md"
        ),
        "contrast": (
            f"Child schedule only: 60-epoch cosine with patience 5 -> fully annealed "
            f"{args.f_epochs}-epoch schedule with patience {args.f_epochs}. Same "
            "parents (patience-20 attention views or trained common-model parents), "
            "learning rate, SAM, transferred multiplier, loss, selector, store, seeds "
            "and eight periods. Raw earlier-tie selection is unchanged; post-hoc "
            "views are evaluated separately."
        ),
        "gate": (
            "Paired fold deltas of each short-schedule arm against its reference arm's "
            "existing books on the same policy (block 40 primary): retain only with "
            "the 95 percent lower bound above zero and a majority of seeds and folds "
            "positive. No per-fold or per-seed schedule choice."
        ),
    }
    write_json_atomic(root / "plan.json", plan)
    write_json_atomic(root / "frozen_design.json", {"store": stopping["store"]})
    run[PLAN_KEY] = binding(root / "plan.json")
    write_json_atomic(POINTER, run)
    print(
        json.dumps(
            {"plan": run[PLAN_KEY], "arms": list(arms), "fits": plan["planned_fits"]}
        ),
        flush=True,
    )


def execute(run, force=False):
    reference = run[PLAN_KEY]
    plan = bound_json(reference)
    assert sha256_file(Path(__file__)) == plan["driver"]["sha256"]
    for key, path in runtime_files().items():
        assert sha256_file(path) == plan["runtime"]["files"][key]["sha256"], key
    gpu_is_free(run, force)
    store, root = Path(plan["store"]["root"]), Path(plan["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    torch.set_num_threads(1)
    progress = root / "refits.json"
    completed = (
        json.loads(progress.read_text())["completed"] if progress.exists() else []
    )
    done = {r["key"] for r in completed}
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            for arm, spec in plan["arms"].items():
                key = f"{arm}/F/{fold}/{seed}"
                if key in done:
                    continue
                cell = spec["reference"]
                parent = plan["parents"][cell][str(seed)]
                output = root / "fits" / arm / f"{fold}_seed_{seed}"
                tick = perf_counter()
                if not (output / "run_manifest.json").exists():
                    if shutil.disk_usage(root).free < 2_000_000_000:
                        raise RuntimeError("less than 2 GB free before fit")
                    print(
                        json.dumps({"starting": key, "completed": len(completed)}),
                        flush=True,
                    )
                    torch._dynamo.reset()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    train(
                        store,
                        output,
                        cell=plan["cells"][cell],
                        stage="F",
                        fold=fold,
                        seed=seed,
                        epochs=plan["f_epochs"],
                        recipe=TrainingRecipe(**plan["f_recipes"][cell]),
                        parent=Path(parent["path"]),
                        parent_sha256=parent["sha256"],
                        compiled=True,
                        export_scores=True,
                        ema_half_life_epochs=plan["ema_half_life_epochs"],
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                manifest = bound_json(binding(output / "run_manifest.json"))
                assert manifest["status"] == "completed"
                completed.append(
                    {
                        "key": key,
                        "manifest": binding(output / "run_manifest.json"),
                        "parent": parent,
                        "seconds": perf_counter() - tick,
                    }
                )
                done.add(key)
                write_json_atomic(
                    progress,
                    {
                        "status": "complete"
                        if len(completed) == plan["planned_fits"]
                        else "running",
                        "plan": reference,
                        "completed": completed,
                        "planned": plan["planned_fits"],
                    },
                )
                print(json.dumps(completed[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--cell", action="append", help="one or two cells")
    parser.add_argument("--f-epochs", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="skip the busy-GPU guard")
    args = parser.parse_args()
    run = json.loads(POINTER.read_text())
    freeze(run, args) if args.freeze else execute(run, args.force)
