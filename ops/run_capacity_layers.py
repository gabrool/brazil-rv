"""Freeze registered depth or matched LSTM after the preceding capacity screen."""

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


def freeze(run, wave):
    previous = "width" if wave == "depth" else "depth"
    prefix = f"stage_d_{wave}"
    admission = bound_json(run[f"stage_d_{previous}_admission"])
    assert admission["technical_comparisons_valid"]
    engineering = bound_json(run[prefix + "_engineering"])
    assert engineering["status"] == f"implementation_qualified_not_{wave}_experiment"
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
    candidates = (
        (("TE_depth2", "attention"), ("GRU_depth2", "gru"))
        if wave == "depth"
        else (("LSTM_matched", "gru"),)
    )
    for candidate, architecture in candidates:
        retained = admission["retained"][architecture]
        parent_plan = bound_json(retained["plan"])
        reference = retained["arm"]
        old_cell = parent_plan["cells"][reference]
        old_config = configuration(old_cell, store["feature_names"])
        assert old_config.temporal_encoder in (
            (architecture,) if wave == "depth" else ("gru", "gru_depth2")
        )
        cell = deepcopy(old_cell)
        cell["cell"] = candidate
        encoder = (
            architecture + "_depth2"
            if wave == "depth"
            else old_config.temporal_encoder.replace("gru", "lstm")
        )
        cell["model"] = dict(cell.get("model", {}), temporal_encoder=encoder)
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
    root = Path(run["root"]) / f"capacity_{wave}"
    root.mkdir(exist_ok=False)
    plan = dict(
        status=f"frozen_before_{wave}_outcomes",
        stage="D",
        admission=run[f"stage_d_{previous}_admission"],
        engineering=run[prefix + "_engineering"],
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
        planned_fits=len(cells) * len(original["seeds"]) * (1 + len(original["folds"])),
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
        contrast=(
            "Exactly one second temporal block for attention and one second recurrent GRU layer, at widths retained by the prior economic screen. Position encoding once; GRU has no added interlayer dropout."
            if wave == "depth"
            else "Exactly one LSTM-versus-GRU cell at the retained GRU width and layer count. Only the recurrent cell changes, with its additional gate parameters recorded by module. No extra interlayer dropout. Compare complete fit runtimes and memory explicitly; this is not a parameter-matched GRU width contrast."
        )
        + " Unchanged pooling/early peers/FiLM/heads/final context/trunk. Fresh compatible P parents and original seeds/folds/learning budgets/optimizer/selector; saved controls reused.",
        gate="R10m equal-four-fold net-CDI delta >=.25bps/day, >=2/3 positive seeds and >=3/4 positive folds; no unexplained BRL/CDI Sharpe decline or worse drawdown. IC diagnostic. At most one candidate per architecture proceeds to ten-fold confirmation under the registered family-adjusted uncertainty rule; width and depth are not independent additional replication slots.",
        next="Further peer/context/general capacity requires module/fit diagnostics and economic evidence; no automatic grid or extra seeds. At most one final candidate per architecture receives registered ten-fold confirmation.",
    )
    write_json_atomic(root / "plan.json", plan)
    write_json_atomic(root / "frozen_design.json", dict(store=original["store"]))
    run[prefix + "_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--wave", choices=("depth", "lstm"), required=True)
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    if args.freeze:
        freeze(run, args.wave)
    else:
        # Reuse the exact frozen fit loop; only its explicit plan binding changes.
        run_capacity_width.execute(
            dict(run, stage_d_width_plan=run[f"stage_d_{args.wave}_plan"])
        )
