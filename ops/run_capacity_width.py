"""Registered two-cell width wave, admitted only after Stage C disposition."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import gc
import inspect
import json
from pathlib import Path
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_training import TrainingRecipe, train
from brazil_rv.v2.train import compile_forward

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    admission = bound_json(run["stage_c_refit_admission"])
    assert admission["technical_comparisons_valid"] and admission["stage_c_complete"]
    source = bound_json(run["stage_c_refit_plan"])
    store = bound_json(
        dict(
            path=str(Path(source["store"]["root"]) / "manifest.json"),
            sha256=source["store"]["manifest_sha256"],
        )
    )
    cells, controls, counts = {}, {}, {}
    for candidate, reference, old_width, width in (
        ("TE_128", "TE_wide", 96, 128),
        ("GRU_96", "GRU_early", 64, 96),
    ):
        old_cell = source["cells"][reference]
        old_config = configuration(old_cell, store["feature_names"])
        assert old_config.hidden_width == old_width
        cell = deepcopy(old_cell)
        cell["cell"] = candidate
        cell["model"] = dict(cell.get("model", {}), hidden_width=width)
        config = configuration(cell, store["feature_names"])
        changes = {k for k, v in asdict(config).items() if v != asdict(old_config)[k]}
        assert changes == {"hidden_width"}
        torch.manual_seed(11)
        model = CharacteristicModel(config)
        by_module = {}
        for name, parameter in model.named_parameters():
            module = name.split(".")[0]
            by_module[module] = by_module.get(module, 0) + parameter.numel()
        baseline_modules = {}
        for name, parameter in CharacteristicModel(old_config).named_parameters():
            module = name.split(".")[0]
            baseline_modules[module] = (
                baseline_modules.get(module, 0) + parameter.numel()
            )
        deltas = {k: value - baseline_modules[k] for k, value in by_module.items()}
        assert {k for k, value in deltas.items() if value} <= {
            "slow_input_projection",
            "slow_input_norm",
            "slow_encoder",
            "temporal_peer",
            "joint",
        }
        counts[candidate] = dict(
            total=sum(by_module.values()),
            by_module=by_module,
            reference_by_module=baseline_modules,
            module_deltas=deltas,
            config=asdict(config),
            reference=reference,
        )
        cells[candidate] = cell
        for seed in source["seeds"]:
            for fold in source["folds"]:
                path = (
                    Path(run["stage_c_refit_root"])
                    / "fits"
                    / reference
                    / f"{fold}_seed_{seed}"
                    / "run_manifest.json"
                )
                fit = bound_json(binding(path))
                assert fit["status"] == "completed"
                assert fit["contract"]["cell"] == old_cell
                assert (
                    fit["contract"]["store_manifest_sha256"]
                    == source["store"]["manifest_sha256"]
                )
                controls[f"{reference}/{fold}/{seed}"] = binding(path)
    root = Path(run["root"]) / "capacity_width"
    root.mkdir(exist_ok=False)
    plan = dict(
        status="frozen_before_width_outcomes",
        stage="D",
        admission=run["stage_c_refit_admission"],
        source=run["stage_c_refit_plan"],
        store=source["store"],
        cells=cells,
        controls=controls,
        parameters=counts,
        seeds=source["seeds"],
        folds=source["folds"],
        p_recipe=source["p_recipe"],
        f_recipes={c: source["f_recipes"][v["reference"]] for c, v in counts.items()},
        maximum_epochs=source["maximum_epochs"],
        ema_half_life_epochs=source["attention_gru_ema_half_life_epochs"],
        planned_fits=len(cells) * len(source["seeds"]) * (1 + len(source["folds"])),
        driver=binding(Path(__file__)),
        runtime=dict(
            git=_git_identity(),
            files={
                "training": binding(Path(inspect.getfile(train))),
                "model": binding(Path(inspect.getfile(CharacteristicModel))),
                "temporal": binding(
                    Path(inspect.getfile(CharacteristicModel)).with_name(
                        "temporal_pathway.py"
                    )
                ),
                "configuration": binding(Path(inspect.getfile(configuration))),
            },
        ),
        contrast="Existing hidden_width96->128 for attention and64->96 for GRU, one configuration field per cell; new compatible P parents. Other cleaned inputs, FiLM, early-peer timing, history pooling, heads, optimizer, selector and full learning budgets stay fixed. Reuse all matched C controls.",
        dimensional_coupling="The existing hidden_width also sizes the input projection, same-stage peer attention and history query, and the temporal input columns of joint fusion. Report their module parameter deltas; this is the model's coupled temporal-path width contrast, not an isolated recurrent matrix scaling law or an independently expanded final context/trunk. No extra peer layer or final-context width changes.",
        gate="Four-fold primary R10m net-CDI delta >=.25bps/day, at least2/3 positive seed and3/4 positive fold deltas; no unexplained BRL/CDI Sharpe decline or worse drawdown. IC diagnostic. At most one retained candidate per architecture receives ten-fold confirmation under the registered paired20/40/60,40primary, one-sided97.5% lower>0 rule. No automatic grid or added seeds.",
        width_retention="Retain the larger width only after the registered economic, seed, fold and risk screen passes. Otherwise retain its existing matched reference width (attention96 or GRU64) for the single subsequent depth contrast. No best-seed choice or additional width search.",
        conditional_next="One attention-depth and one recurrent-depth contrast at retained widths follow the width diagnostics; LSTM and additional peer/context/general capacity remain separately conditional and unstarted.",
    )
    write_json_atomic(root / "plan.json", plan)
    write_json_atomic(root / "frozen_design.json", dict(store=source["store"]))
    run["stage_d_width_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


def execute(run, arms=None):
    torch.set_num_threads(1)
    reference = run["stage_d_width_plan"]
    plan = bound_json(reference)
    assert plan["driver"]["sha256"] == sha256_file(Path(__file__))
    selected_arms = set(arms or plan["cells"])
    assert selected_arms <= set(plan["cells"])
    imported = {
        "training": Path(inspect.getfile(train)),
        "model": Path(inspect.getfile(CharacteristicModel)),
        "temporal": Path(inspect.getfile(CharacteristicModel)).with_name(
            "temporal_pathway.py"
        ),
        "configuration": Path(inspect.getfile(configuration)),
        "compiler": Path(inspect.getfile(compile_forward)),
    }
    for arm in selected_arms:
        runtime = plan.get("runtime_by_arm", {}).get(arm, plan["runtime"])
        for name, record in runtime["files"].items():
            assert sha256_file(Path(record["path"])) == record["sha256"]
            assert sha256_file(imported[name]) == record["sha256"], (arm, name)
    root = Path(reference["path"]).parent
    store = Path(plan["store"]["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    path = root / "refits.json"
    completed = json.loads(path.read_text())["completed"] if path.exists() else []
    done = {r["key"] for r in completed}
    jobs = [
        (arm, "P", "pretrain_internal", seed)
        for seed in plan["seeds"]
        for arm in plan["cells"]
    ]
    jobs += [
        (arm, "F", fold, seed)
        for fold in plan["folds"]
        for seed in plan["seeds"]
        for arm in plan["cells"]
    ]
    for arm, stage, fold, seed in jobs:
        if arm not in selected_arms:
            continue
        key = f"{arm}/{stage}/{fold}/{seed}"
        if key in done:
            bound_json(next(r["manifest"] for r in completed if r["key"] == key))
            continue
        output = (
            root
            / "fits"
            / arm
            / (f"P_seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}")
        )
        parent = root / "fits" / arm / f"P_seed_{seed}" / "selected.pt"
        tick = perf_counter()
        print(
            json.dumps(dict(starting=key, completed=len(completed), planned=len(jobs))),
            flush=True,
        )
        torch._dynamo.reset()
        torch.cuda.reset_peak_memory_stats()
        train(
            store,
            output,
            cell=plan["cells"][arm],
            stage=stage,
            fold=fold,
            seed=seed,
            epochs=plan["maximum_epochs"],
            recipe=TrainingRecipe(
                **(plan["p_recipe"] if stage == "P" else plan["f_recipes"][arm])
            ),
            parent=parent if stage == "F" else None,
            parent_sha256=sha256_file(parent) if stage == "F" else None,
            compiled=True,
            export_scores=stage == "F",
            ema_half_life_epochs=plan["ema_half_life_epochs"] if stage == "F" else None,
        )
        completed.append(
            dict(
                key=key,
                manifest=binding(output / "run_manifest.json"),
                seconds=perf_counter() - tick,
            )
        )
        done.add(key)
        write_json_atomic(
            path,
            dict(
                status="complete" if len(completed) == len(jobs) else "running",
                plan=reference,
                completed=completed,
                planned=len(jobs),
            ),
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--arm", action="append")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run, args.arm)
