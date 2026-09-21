"""Freeze the registered second temporal layer after width disposition."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import inspect
import json
from pathlib import Path

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_training import train
import run_capacity_width

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    admission = bound_json(run["stage_d_width_admission"])
    assert admission["technical_comparisons_valid"]
    engineering = bound_json(run["stage_d_depth_engineering"])
    assert engineering["status"] == "implementation_qualified_not_depth_experiment"
    model_source = Path(inspect.getfile(CharacteristicModel))
    for name, record in engineering["runtime"].items():
        assert sha256_file(model_source.with_name(name)) == record["sha256"]
    original = bound_json(run["stage_c_refit_plan"])
    store = bound_json(
        dict(
            path=str(Path(original["store"]["root"]) / "manifest.json"),
            sha256=original["store"]["manifest_sha256"],
        )
    )
    cells, counts, controls, recipes, reference_results = {}, {}, {}, {}, {}
    for candidate, architecture in (("TE_depth2", "attention"), ("GRU_depth2", "gru")):
        retained = admission["retained"][architecture]
        parent_plan = bound_json(retained["plan"])
        reference = retained["arm"]
        old_cell = parent_plan["cells"][reference]
        old_config = configuration(old_cell, store["feature_names"])
        assert old_config.temporal_encoder == architecture
        cell = deepcopy(old_cell)
        cell["cell"] = candidate
        cell["model"] = dict(
            cell.get("model", {}), temporal_encoder=architecture + "_depth2"
        )
        config = configuration(cell, store["feature_names"])
        assert {
            k for k, value in asdict(config).items() if value != asdict(old_config)[k]
        } == {"temporal_encoder"}
        torch.manual_seed(11)
        model = CharacteristicModel(config)
        modules, old_modules = {}, {}
        for target, instance in (
            (modules, model),
            (old_modules, CharacteristicModel(old_config)),
        ):
            for name, parameter in instance.named_parameters():
                key = name.split(".")[0]
                target[key] = target.get(key, 0) + parameter.numel()
        deltas = {key: value - old_modules[key] for key, value in modules.items()}
        assert {key for key, value in deltas.items() if value} == {"slow_encoder"}
        cells[candidate] = cell
        counts[candidate] = dict(
            total=sum(modules.values()),
            by_module=modules,
            reference_by_module=old_modules,
            module_deltas=deltas,
            config=asdict(config),
            reference=reference,
            retained=retained,
        )
        recipes[candidate] = parent_plan["f_recipes"][reference]
        reference_results[reference] = retained["results"]
        fit_root = Path(retained["fit_root"])
        for seed in original["seeds"]:
            for fold in original["folds"]:
                manifest = binding(
                    fit_root
                    / "fits"
                    / reference
                    / f"{fold}_seed_{seed}"
                    / "run_manifest.json"
                )
                fit = bound_json(manifest)
                assert fit["status"] == "completed"
                assert fit["contract"]["cell"] == old_cell
                assert (
                    fit["contract"]["store_manifest_sha256"]
                    == original["store"]["manifest_sha256"]
                )
                controls[f"{reference}/{fold}/{seed}"] = manifest
    root = Path(run["root"]) / "capacity_depth"
    root.mkdir(exist_ok=False)
    plan = dict(
        status="frozen_before_depth_outcomes",
        stage="D",
        admission=run["stage_d_width_admission"],
        engineering=run["stage_d_depth_engineering"],
        store=original["store"],
        cells=cells,
        parameters=counts,
        controls=controls,
        reference_results=reference_results,
        seeds=original["seeds"],
        folds=original["folds"],
        p_recipe=original["p_recipe"],
        f_recipes=recipes,
        maximum_epochs=original["maximum_epochs"],
        ema_half_life_epochs=original["attention_gru_ema_half_life_epochs"],
        planned_fits=30,
        driver=binding(Path(run_capacity_width.__file__)),
        freezer=binding(Path(__file__)),
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
        contrast="Exactly one second temporal block for attention and one second recurrent GRU layer, at widths retained by the prior economic screen. Position encoding once, unchanged pooling/early peers/FiLM/heads/final context/trunk. GRU has no added interlayer dropout. Fresh compatible P parents and original seeds/folds/learning budgets/optimizer/selector; saved single-layer controls reused.",
        gate="R10m equal-four-fold net-CDI delta >=.25bps/day, >=2/3 positive seeds and >=3/4 positive folds; no unexplained BRL/CDI Sharpe decline or worse drawdown. IC diagnostic. At most one candidate per architecture proceeds to ten-fold confirmation under the registered family-adjusted uncertainty rule; width and depth are not independent additional replication slots.",
        next="LSTM versus retained GRU is authorized after recurrent diagnostics with explicit parameter/runtime differences. Further peer/context/general capacity requires those diagnostics and economic evidence; no automatic grid or extra seeds.",
    )
    write_json_atomic(root / "plan.json", plan)
    write_json_atomic(root / "frozen_design.json", dict(store=original["store"]))
    run["stage_d_depth_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    if args.freeze:
        freeze(run)
    else:
        # Reuse the exact frozen fit loop; only its explicit plan binding changes.
        run_capacity_width.execute(
            dict(run, stage_d_width_plan=run["stage_d_depth_plan"])
        )
