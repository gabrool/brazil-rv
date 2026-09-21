"""Outcome-informed parent-stopping diagnostic; preserves all selected controls."""

import argparse
import gc
import inspect
import json
from pathlib import Path
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7_training import TrainingRecipe, train

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    out = Path(run["scaling_investigation"]["path"]).parent / "parent_patience"
    plan_path = out / "plan.json"
    if args.freeze:
        out.mkdir(exist_ok=False)
        original = bound_json(run["stage_c_refit_plan"])
        controls = Path(run["stage_c_refit_root"]) / "fits/TE_wide"
        p_recipe = original["p_recipe"].copy()
        assert p_recipe["patience"] == 5
        p_recipe["patience"] = 20
        plan = dict(
            status="frozen_before_new_fit",
            original_plan=run["stage_c_refit_plan"],
            store=original["store"],
            cell=original["cells"]["TE_wide"],
            seed=29,
            fold="F10",
            epochs=original["maximum_epochs"],
            p_recipe=p_recipe,
            f_recipe=original["f_recipes"]["TE_wide"],
            ema_half_life_epochs=original["attention_gru_ema_half_life_epochs"],
            controls={
                name: binding(controls / name / "run_manifest.json")
                for name in ("P_seed_29", "F10_seed_29")
            },
            runtime=dict(
                code=_git_identity(), training=binding(Path(inspect.getfile(train)))
            ),
            driver=binding(Path(__file__)),
            scope="One outcome-informed mechanism test, not candidate adoption or an unbiased confirmation. Corrected TE_wide P seed29 changes ONLY patience5 to20, preserving the original maximum60 and schedule60, loss/selector/optimizer/full population/full60 and same data. Its F10 seed29 child uses the original unchanged F recipe. The corrected original parent stopped9/selected4; original-data parent stopped25/selected20. No exact epoch9 resume exists: epoch files omit optimizer/RNG, while selected.pt is epoch4. Therefore train a fresh trajectory and compare its first nine epochs/weights to saved corrected controls before attributing any difference to patience. Preserve divergence if any; never declare exact continuation without evidence. Compare the child's own raw forecasts and corrected R10m neutral account with saved corrected and original-data seed29 controls. No new test-period checkpoint selection, held-out reads, other seed/patience search or automatic adoption.",
        )
        write_json_atomic(plan_path, plan)
        (out / "executed.py").write_bytes(Path(__file__).read_bytes())
        run["scaling_parent_patience_plan"] = binding(plan_path)
        write_json_atomic(pointer, run)
        print(
            json.dumps(dict(frozen=str(plan_path), runtime=plan["runtime"])), flush=True
        )
        return
    plan = bound_json(run["scaling_parent_patience_plan"])
    assert sha256_file(Path(__file__)) == plan["driver"]["sha256"]
    assert _git_identity() == plan["runtime"]["code"]
    assert (
        sha256_file(Path(inspect.getfile(train)))
        == plan["runtime"]["training"]["sha256"]
    )
    store = Path(plan["store"]["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    torch.set_num_threads(1)
    progress = out / "fits.json"
    completed = bound_json(binding(progress))["completed"] if progress.exists() else []
    parent = out / "P_seed_29/selected.pt"
    for stage, fold, name in (
        ("P", "pretrain_internal", "P_seed_29"),
        ("F", "F10", "F10_seed_29"),
    ):
        if any(row["stage"] == stage for row in completed):
            continue
        started = perf_counter()
        print(json.dumps(dict(starting=stage, name=name)), flush=True)
        torch._dynamo.reset()
        train(
            store,
            out / name,
            cell=plan["cell"],
            stage=stage,
            fold=fold,
            seed=plan["seed"],
            epochs=plan["epochs"],
            recipe=TrainingRecipe(**plan["p_recipe" if stage == "P" else "f_recipe"]),
            parent=parent if stage == "F" else None,
            parent_sha256=sha256_file(parent) if stage == "F" else None,
            compiled=True,
            export_scores=stage == "F",
            ema_half_life_epochs=plan["ema_half_life_epochs"] if stage == "F" else None,
        )
        completed.append(
            dict(
                stage=stage,
                manifest=binding(out / name / "run_manifest.json"),
                seconds=perf_counter() - started,
            )
        )
        write_json_atomic(
            progress,
            dict(
                plan=binding(plan_path),
                completed=completed,
                status="complete" if len(completed) == 2 else "running",
            ),
        )
        gc.collect()
        torch.cuda.empty_cache()
    run = json.loads(pointer.read_text())
    run["scaling_parent_patience_fits"] = binding(progress)
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
