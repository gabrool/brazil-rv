"""Matched registered C refits, original recipes, new compatible parents only."""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.objective_program import adapt_c6_parent
from brazil_rv.v2.round6 import training_command
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_training import TrainingRecipe, train

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run, root):
    root.mkdir(exist_ok=False)
    source = bound_json(run["stage_c_plan"])
    accepted = bound_json(run["economic_refit_inputs"])
    store = Path(accepted["store"]["root"])
    assert sha256_file(store / "manifest.json") == accepted["store"]["manifest_sha256"]
    manifest = json.loads((store / "manifest.json").read_text())
    prior, foundation = Path(source["prior_root"]), Path(source["foundation_root"])
    original = json.loads((foundation / "frozen_design.json").read_text())
    cells, recipes, counts, controls = {}, {}, {}, {}
    for arm in source["arms"]:
        path = (
            prior / "phase3/fits/C6/neutral/F2_seed_11"
            if arm == "C6"
            else foundation / "fits" / arm / "F2_seed_11"
        ) / "run_manifest.json"
        fit = bound_json(binding(path))
        cells[arm] = fit["contract"]["cell"]
        recipes[arm] = fit["contract"]["recipe"]
        config = configuration(cells[arm], manifest["feature_names"])
        assert json.loads(json.dumps(asdict(config))) == fit["contract"]["config"]
        model = (
            DailyMultiHorizonModel(config)
            if arm == "C6"
            else CharacteristicModel(config)
        )
        counts[arm] = sum(p.numel() for p in model.parameters())
        controls[arm] = binding(path)
    design = dict(
        status="frozen_before_repaired_data_fits",
        stage="C",
        store=accepted["store"],
        acceptance=run["economic_refit_inputs"],
        cells=cells,
        parameter_counts=counts,
        matched_graph_recipe_sources=controls,
        f_recipes=recipes,
        p_recipe=original["p_recipe"],
        folds=source["folds"],
        seeds=[11, 29, 47],
        maximum_epochs=60,
        raw_selection_unchanged=True,
        attention_gru_ema_half_life_epochs=original["ema_half_life_epochs"],
        c6_parent="original slow-only S0 P,60epochs maximum/3patience/selection every2, then exact zero-family C6 adaptation; fresh repaired-store parent only",
        optional_to_close_loss=0,
        no_population_or_history_reduction=True,
        no_heldout_consumers=True,
        driver=binding(Path(__file__)),
        stage_c_plan=run["stage_c_plan"],
    )
    write_json_atomic(root / "refit_plan.json", design)
    print(json.dumps(dict(frozen=True, parameter_counts=counts)), flush=True)


def execute(run, root):
    torch.set_num_threads(1)
    design = json.loads((root / "refit_plan.json").read_text())
    assert design["driver"]["sha256"] == sha256_file(Path(__file__))
    store = Path(design["store"]["root"])
    assert sha256_file(store / "manifest.json") == design["store"]["manifest_sha256"]
    jobs = [
        (arm, "P", "pretrain_internal", seed)
        for seed in design["seeds"]
        for arm in ("TE_full", "TE_wide", "GRU_early")
    ]
    jobs += [
        (arm, "F", fold, seed)
        for fold in design["folds"]
        for seed in design["seeds"]
        for arm in ("TE_full", "TE_wide", "GRU_early")
    ]
    jobs += [("C6", "P", "pretrain_internal", seed) for seed in design["seeds"]]
    jobs += [
        ("C6", "F", fold, seed) for fold in design["folds"] for seed in design["seeds"]
    ]
    progress = root / "refits.json"
    complete = (
        json.loads(progress.read_text())["completed"] if progress.exists() else []
    )
    done = {x["key"] for x in complete}
    for arm, stage, fold, seed in jobs:
        key = f"{arm}/{stage}/{fold}/{seed}"
        if key in done:
            bound_json(next(x["manifest"] for x in complete if x["key"] == key))
            continue
        output = (
            root
            / "fits"
            / arm
            / (f"P_seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
        )
        parent = root / "fits" / arm / f"P_seed_{seed}/selected.pt"
        print(
            json.dumps(dict(starting=key, completed=len(complete), planned=len(jobs))),
            flush=True,
        )
        tick = perf_counter()
        torch._dynamo.reset()
        torch.cuda.reset_peak_memory_stats()
        if arm == "C6" and stage == "P":
            plain = output / "slow_parent"
            command = training_command(
                dict(
                    store=design["store"],
                    fast_initialization=dict(
                        mode="native_fresh", transfer_chronology_clean=True
                    ),
                ),
                plain,
                "S0",
                seed,
                fold,
                "P",
            )
            subprocess.run(command, check=True, cwd=PROJECT)
            adapt_c6_parent(
                plain / "raw_patience.pt",
                parent,
                sha256_file(plain / "raw_patience.pt"),
                seed,
                json.loads((store / "manifest.json").read_text()),
                design["store"]["manifest_sha256"],
            )
            write_json_atomic(
                output / "run_manifest.json",
                dict(
                    status="completed",
                    stage="P",
                    seed=seed,
                    plain_parent=binding(plain / "run_manifest.json"),
                    adapted_parent=binding(parent),
                ),
            )
        else:
            train(
                store,
                output,
                cell=design["cells"][arm],
                stage=stage,
                fold=fold,
                seed=seed,
                epochs=design["maximum_epochs"],
                recipe=TrainingRecipe(
                    **(design["p_recipe"] if stage == "P" else design["f_recipes"][arm])
                ),
                parent=parent if stage == "F" else None,
                parent_sha256=sha256_file(parent) if stage == "F" else None,
                compiled=True,
                export_scores=stage == "F",
                ema_half_life_epochs=design["attention_gru_ema_half_life_epochs"]
                if stage == "F" and arm != "C6"
                else None,
            )
        complete.append(
            dict(
                key=key,
                manifest=binding(output / "run_manifest.json"),
                seconds=perf_counter() - tick,
            )
        )
        write_json_atomic(
            progress,
            dict(
                status="complete" if len(complete) == len(jobs) else "running",
                plan=binding(root / "refit_plan.json"),
                completed=complete,
                planned=len(jobs),
            ),
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["stage_c_refit_root"])
    freeze(run, root) if args.freeze else execute(run, root)
