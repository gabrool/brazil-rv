"""Conditional ten-fold replication of the two registered economic leads."""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import DEVELOPMENT_FOLDS
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7_training import TrainingRecipe, train

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    admission = bound_json(run["stage_c_refit_admission"])
    results = bound_json(run["stage_c_refit_results"])
    source = bound_json(run["stage_c_refit_plan"])
    assert admission["results"] == run["stage_c_refit_results"]
    assert admission["technical_comparisons_valid"]
    comparisons = bound_json(results["comparisons"])
    candidates = [
        arm
        for arm in ("TE_wide", "GRU_early")
        if next(
            r
            for r in comparisons
            if r["candidate"] == arm
            and r["reference"] == "TE_full"
            and r["capital"] == 10000000
        )["members"]["ensemble"]["equal_fold_mean_bps_day"]
        > 0
    ]
    assert candidates == admission["replicate"]
    arms = ["TE_full", *candidates] if candidates else []
    folds = [f for f in DEVELOPMENT_FOLDS if f not in source["folds"]]
    assert len(folds) == 10
    parents = {}
    for arm in arms:
        for seed in source["seeds"]:
            root = Path(run["stage_c_refit_root"]) / "fits" / arm / f"P_seed_{seed}"
            rec = bound_json(binding(root / "run_manifest.json"))
            checkpoint = binding(root / "selected.pt")
            assert rec["status"] == "completed" and rec["seed"] == seed
            assert rec["contract"]["cell"] == source["cells"][arm]
            assert (
                rec["contract"]["store_manifest_sha256"]
                == source["store"]["manifest_sha256"]
            )
            assert rec["artifacts"]["selected.pt"] == checkpoint["sha256"]
            parents[f"{arm}/{seed}"] = dict(
                manifest=binding(root / "run_manifest.json"), checkpoint=checkpoint
            )
    root = Path(run["root"]) / "economic_lead_replication"
    root.mkdir(exist_ok=False)
    plan = dict(
        status="frozen_before_other_ten_fold_results" if arms else "conditional_stop",
        admission=run["stage_c_refit_admission"],
        original_screen=run["stage_c_refit_plan"],
        candidates=candidates,
        arms=arms,
        folds=folds,
        seeds=source["seeds"],
        store=source["store"],
        cells={arm: source["cells"][arm] for arm in arms},
        parents=parents,
        f_recipes={arm: source["f_recipes"][arm] for arm in arms},
        maximum_epochs=source["maximum_epochs"],
        ema_half_life_epochs=source["attention_gru_ema_half_life_epochs"],
        planned_fits=len(arms) * len(folds) * len(source["seeds"]),
        economic_rule="R$10m neutral net above CDI; equal-fold mean advantage >=0.25bp/day, at least2/3 positive seed differences, within-fold40-session one-sided97.5% lower bound >0. Also report20/60-session bounds and R$1m/R$5m. A fall in BRL/CDI Sharpe or worse drawdown requires risk attribution and prevents automatic adoption.",
        limits="Previously nominated leads and reused development, not untouched confirmation. Same graphs, new-store compatible parents, original selector/optimizer/learning budget, all933/full60. No extra parent fit, C6 replication, capacity contrast or held-out consumer.",
        driver=binding(Path(__file__)),
        registration=binding(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
        ),
    )
    write_json_atomic(root / "plan.json", plan)
    write_json_atomic(root / "frozen_design.json", dict(store=source["store"]))
    run["stage_c_replication_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(
        json.dumps(dict(candidates=candidates, planned=plan["planned_fits"])),
        flush=True,
    )


def execute(run):
    torch.set_num_threads(1)
    _git_identity()
    reference = run["stage_c_replication_plan"]
    plan = bound_json(reference)
    assert plan["driver"]["sha256"] == sha256_file(Path(__file__))
    store = Path(plan["store"]["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    root = Path(reference["path"]).parent
    path = root / "refits.json"
    completed = json.loads(path.read_text())["completed"] if path.exists() else []
    done = {r["key"] for r in completed}
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            for arm in plan["arms"]:
                key = f"{arm}/F/{fold}/{seed}"
                if key in done:
                    bound_json(
                        next(r["manifest"] for r in completed if r["key"] == key)
                    )
                    continue
                parent = plan["parents"][f"{arm}/{seed}"]["checkpoint"]
                output = root / "fits" / arm / f"{fold}_seed_{seed}"
                tick = perf_counter()
                print(
                    json.dumps(dict(starting=key, completed=len(completed))), flush=True
                )
                torch._dynamo.reset()
                torch.cuda.reset_peak_memory_stats()
                train(
                    store,
                    output,
                    cell=plan["cells"][arm],
                    stage="F",
                    fold=fold,
                    seed=seed,
                    epochs=plan["maximum_epochs"],
                    recipe=TrainingRecipe(**plan["f_recipes"][arm]),
                    parent=Path(parent["path"]),
                    parent_sha256=parent["sha256"],
                    compiled=True,
                    export_scores=True,
                    ema_half_life_epochs=plan["ema_half_life_epochs"],
                )
                completed.append(
                    dict(
                        key=key,
                        manifest=binding(output / "run_manifest.json"),
                        seconds=perf_counter() - tick,
                    )
                )
                write_json_atomic(
                    path,
                    dict(
                        status="complete"
                        if len(completed) == plan["planned_fits"]
                        else "running",
                        plan=reference,
                        completed=completed,
                        planned=plan["planned_fits"],
                    ),
                )
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run)
