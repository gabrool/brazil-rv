"""Matched width/parent-stopping experiment with exact shared-prefix selection."""

import argparse
import gc
import inspect
import json
from pathlib import Path
import shutil
import subprocess
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


def stopped_selection(history, patience, minimum_improvement):
    """Apply the original selector, ignoring every epoch after its stopping point."""
    best, selected, stale, completed = float("-inf"), None, 0, 0
    for record in history:
        if stale >= patience:
            break
        value = record["selection"]["mean_ic"]
        if value > best + minimum_improvement:
            best, selected, stale = value, record["epoch"], 0
        else:
            stale += 1
        completed = record["epoch"]
    assert selected is not None
    return dict(
        selected_epoch=selected,
        selection_ic=best,
        epochs_completed=completed,
        stop_reason="patience" if stale >= patience else "ceiling",
    )


def runtime_files():
    model = Path(inspect.getfile(CharacteristicModel))
    return dict(
        training=Path(inspect.getfile(train)),
        model=model,
        temporal=model.with_name("temporal_pathway.py"),
        configuration=Path(inspect.getfile(configuration)),
        compiler=Path(inspect.getfile(compile_forward)),
    )


def freeze(run):
    code = _git_identity()
    source = bound_json(run["stage_c_refit_plan"])
    investigation = bound_json(run["scaling_investigation"])
    accepted = bound_json(run["scaling_data_inputs"])
    recovery = bound_json(run["scaling_input_recovery"])
    assert recovery["acceptance"] == run["scaling_data_inputs"]
    root = Path("C:/quant-data/b3/processed/model_runs/v2_attention_stopping_20260921")
    root.mkdir(exist_ok=False)
    recipe = dict(source["p_recipe"])
    assert (
        recipe["patience"] == 5
        and recipe["schedule_epochs"] == source["maximum_epochs"] == 60
    )
    recipe["patience"] = 20
    cells = {name: source["cells"][name] for name in ("TE_full", "TE_wide")}
    arms = {
        f"{name}_p{p}": dict(reference=name, patience=p)
        for name in cells
        for p in (5, 20)
    }
    assert source["seeds"] == [11, 29, 47]
    plan = dict(
        status="frozen_before_matched_stopping_outcomes",
        root=str(root),
        store=accepted["store"],
        data_admission=run["scaling_data_inputs"],
        recovery=run["scaling_input_recovery"],
        account_terms=accepted["account_terms"],
        economic_account=run["economic_account"],
        source_recipe=run["stage_c_refit_plan"],
        investigation=run["scaling_investigation"],
        cells=cells,
        arms=arms,
        seeds=source["seeds"],
        folds=investigation["expansion_folds"],
        p_recipe=recipe,
        f_recipes={a: source["f_recipes"][a] for a in cells},
        maximum_epochs=60,
        ema_half_life_epochs=source["attention_gru_ema_half_life_epochs"],
        parent_trajectories=6,
        parent_selection_views=12,
        maximum_child_fits=96,
        runtime=dict(
            git=code, files={k: binding(v) for k, v in runtime_files().items()}
        ),
        driver=binding(Path(__file__)),
        registration=binding(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
        ),
        scope="Two existing attention graphs64/96, three fixed seeds, eight fixed periods and the newly accepted data. Parent stopping5/20 only; child recipes,60maximum/schedule, optimizer, earlier-tie selector, full933/full60 and fold boundaries unchanged. No best-seed replacement or old-checkpoint rescore.",
        shared_prefix="Run each patience20 parent trajectory once. The patience5 selection uses ONLY history through the first five-stale stopping point and that saved epoch's weights. Patience does not enter gradient, LR schedule, sampler or RNG. These are twelve labelled selection views of six executed trajectories, never twelve independent fits. A chosen epoch file omits optimizer state; child F already initializes its own optimizer and transfers compatible weights/preprocessing only. The stored parent contract honestly remains the executed patience20 contract.",
        exact_reuse="If both selectors choose the same epoch, use the same selected.pt for both, execute each seed/fold child once and bind the second logical arm to that exact completed fit. Otherwise train both children. No reuse based on similar IC or PnL.",
        comparisons="Report width96-minus64 under each stopping rule, stopping20-minus5 within each width, their interaction, all seeds/folds and paired20/40/60-session uncertainty (40primary), three currency-consistent Sharpes and drawdown. IC is diagnostic. Investigative development comparison, no automatic adoption or claim of untouched testing.",
        boundaries="Six development reserves and2025/2026 consumers remain unopened for new comparisons. Later C6/GRU/common-policy and bounded capacity work follow the four-step instruction.",
    )
    write_json_atomic(root / "plan.json", plan)
    write_json_atomic(root / "frozen_design.json", dict(store=accepted["store"]))
    (root / "executed.py").write_bytes(Path(__file__).read_bytes())
    run["scaling_matched_stopping_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(
        json.dumps(
            dict(
                plan=run["scaling_matched_stopping_plan"],
                parents=6,
                maximum_children=96,
            )
        ),
        flush=True,
    )


def execute(run):
    reference = run["scaling_matched_stopping_plan"]
    plan = bound_json(reference)
    assert sha256_file(Path(__file__)) == plan["driver"]["sha256"]
    for key, path in runtime_files().items():
        assert sha256_file(path) == plan["runtime"]["files"][key]["sha256"], key
    store, root = Path(plan["store"]["root"]), Path(plan["root"])
    assert sha256_file(store / "manifest.json") == plan["store"]["manifest_sha256"]
    torch.set_num_threads(1)
    progress = root / "refits.json"
    completed = (
        json.loads(progress.read_text())["completed"] if progress.exists() else []
    )
    done = {r["key"] for r in completed}

    def save(row):
        completed.append(row)
        done.add(row["key"])
        write_json_atomic(
            progress,
            dict(
                status="running",
                plan=reference,
                completed=completed,
                logical_jobs=102,
                actual_fits=sum(not x.get("reused", False) for x in completed),
            ),
        )
        print(json.dumps(row), flush=True)

    def fit(output, cell, stage, fold, seed, parent=None):
        manifest = output / "run_manifest.json"
        if manifest.exists():
            result = bound_json(binding(manifest))
            assert (
                result["status"] == "completed"
                and result["contract"]["store_manifest_sha256"]
                == plan["store"]["manifest_sha256"]
            )
            return 0.0
        if shutil.disk_usage(root).free < 2_000_000_000:
            raise RuntimeError(
                "Less than2GB free before fit; reclaim verified redundant artifacts before resuming"
            )
        torch._dynamo.reset()
        torch.cuda.reset_peak_memory_stats()
        tick = perf_counter()
        train(
            store,
            output,
            cell=plan["cells"][cell],
            stage=stage,
            fold=fold,
            seed=seed,
            epochs=60,
            recipe=TrainingRecipe(
                **(plan["p_recipe"] if stage == "P" else plan["f_recipes"][cell])
            ),
            parent=parent,
            parent_sha256=sha256_file(parent) if parent else None,
            compiled=True,
            export_scores=stage == "F",
            ema_half_life_epochs=plan["ema_half_life_epochs"] if stage == "F" else None,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return perf_counter() - tick

    for seed in plan["seeds"]:
        for cell in plan["cells"]:
            key = f"{cell}/P/{seed}"
            output = root / "parents" / cell / f"P_seed_{seed}"
            if key in done:
                continue
            print(json.dumps(dict(starting=key, completed=len(completed))), flush=True)
            seconds = fit(output, cell, "P", "pretrain_internal", seed)
            manifest = bound_json(binding(output / "run_manifest.json"))
            history = json.loads((output / "history.json").read_text())
            selected = {
                p: stopped_selection(
                    history, p, plan["p_recipe"]["minimum_improvement"]
                )
                for p in (5, 20)
            }
            for k, v in selected[20].items():
                assert manifest[k] == v, k
            views = {}
            for patience, record in selected.items():
                epoch = record["selected_epoch"]
                checkpoint = output / (
                    "selected.pt"
                    if epoch == selected[20]["selected_epoch"]
                    else f"epochs/epoch_{epoch:03d}.pt"
                )
                relative = str(checkpoint.relative_to(output))
                assert sha256_file(checkpoint) == manifest["artifacts"][relative]
                views[str(patience)] = dict(
                    **record,
                    patience=patience,
                    checkpoint=binding(checkpoint),
                    executed_trajectory=binding(output / "run_manifest.json"),
                    selection_view=True,
                    independent_fit=False,
                )
            write_json_atomic(
                output / "stopping_views.json", dict(plan=reference, views=views)
            )
            save(
                dict(
                    key=key,
                    manifest=binding(output / "run_manifest.json"),
                    views=binding(output / "stopping_views.json"),
                    seconds=seconds,
                )
            )
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            for arm, spec in plan["arms"].items():
                key = f"{arm}/F/{fold}/{seed}"
                if key in done:
                    continue
                cell, patience = spec["reference"], spec["patience"]
                views = bound_json(
                    binding(
                        root
                        / "parents"
                        / cell
                        / f"P_seed_{seed}"
                        / "stopping_views.json"
                    )
                )["views"]
                parent = Path(views[str(patience)]["checkpoint"]["path"])
                output = root / "fits" / arm / f"{fold}_seed_{seed}"
                if (
                    patience == 20
                    and views["5"]["checkpoint"] == views["20"]["checkpoint"]
                ):
                    target = root / "fits" / f"{cell}_p5" / output.name
                    assert (target / "run_manifest.json").exists()
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not output.exists():
                        # Directory junction is an explicit logical alias, never a
                        # copy or modification of an accepted model artifact.
                        quoted = [
                            "'" + str(p).replace("'", "''") + "'"
                            for p in (output, target)
                        ]
                        subprocess.run(
                            [
                                "powershell",
                                "-NoProfile",
                                "-Command",
                                f"New-Item -ItemType Junction -Path {quoted[0]} -Target {quoted[1]} | Out-Null",
                            ],
                            check=True,
                        )
                    assert output.is_junction() and output.resolve() == target.resolve()
                    save(
                        dict(
                            key=key,
                            reused=True,
                            source=str(target),
                            manifest=binding(target / "run_manifest.json"),
                            seconds=0,
                            reason="Identical selected parent epoch, child recipe, seed and fold",
                        )
                    )
                    continue
                print(
                    json.dumps(dict(starting=key, completed=len(completed))), flush=True
                )
                seconds = fit(output, cell, "F", fold, seed, parent)
                save(
                    dict(
                        key=key,
                        manifest=binding(output / "run_manifest.json"),
                        parent=views[str(patience)],
                        seconds=seconds,
                    )
                )
    write_json_atomic(
        progress,
        dict(
            status="complete",
            plan=reference,
            completed=completed,
            logical_jobs=102,
            actual_fits=sum(not x.get("reused", False) for x in completed),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run)
