"""Broaden the two unchanged attention models to eight development periods."""

import argparse
import inspect
import json
from pathlib import Path

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_training import train
from brazil_rv.v2.train import compile_forward
import replay_data_refits
import run_capacity_width

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    investigation = bound_json(run["scaling_investigation"])
    original = bound_json(run["stage_c_refit_plan"])
    out = Path(run["scaling_investigation"]["path"]).parent / "expanded_folds"
    out.mkdir(exist_ok=False)
    source = Path(run["stage_c_refit_root"])
    aliases, completed = [], []
    for arm in investigation["arms"]:
        for stage, fold, seed in [
            *(("P", "pretrain_internal", s) for s in original["seeds"]),
            *(("F", f, s) for f in original["folds"] for s in original["seeds"]),
        ]:
            name = f"P_seed_{seed}" if stage == "P" else f"{fold}_seed_{seed}"
            target = source / "fits" / arm / name
            record = binding(target / "run_manifest.json")
            fit = bound_json(record)
            assert fit["status"] == "completed"
            assert (
                fit["contract"]["store_manifest_sha256"]
                == original["store"]["manifest_sha256"]
            )
            aliases.append(
                dict(path=str(out / "fits" / arm / name), target=str(target))
            )
            completed.append(
                dict(
                    key=f"{arm}/{stage}/{fold}/{seed}",
                    manifest=record,
                    seconds=0,
                    reused=True,
                )
            )
    model_source = Path(inspect.getfile(CharacteristicModel))
    plan = dict(
        status="frozen_before_additional_fold_outcomes",
        investigation=run["scaling_investigation"],
        original=run["stage_c_refit_plan"],
        cells={a: original["cells"][a] for a in investigation["arms"]},
        store=original["store"],
        seeds=original["seeds"],
        folds=investigation["expansion_folds"],
        p_recipe=original["p_recipe"],
        f_recipes={a: original["f_recipes"][a] for a in investigation["arms"]},
        maximum_epochs=original["maximum_epochs"],
        ema_half_life_epochs=original["attention_gru_ema_half_life_epochs"],
        runtime=dict(
            git=_git_identity(),
            files={
                "training": binding(Path(inspect.getfile(train))),
                "model": binding(model_source),
                "temporal": binding(model_source.with_name("temporal_pathway.py")),
                "configuration": binding(Path(inspect.getfile(configuration))),
                "compiler": binding(Path(inspect.getfile(compile_forward))),
            },
        ),
        driver=binding(Path(run_capacity_width.__file__)),
        freezer=binding(Path(__file__)),
        directory_aliases=aliases,
        planned_fits=54,
        reused_fits=30,
        new_fits=24,
        scope="Two existing attention graphs (64/96), unchanged corrected-data P parents and recipes, eight development evaluation folds with expanding training. Reuse all six P and24 completed F fits, train only F3/F7/F11/F13 for both arms and all three original seeds. This does not propagate the outcome-informed patience diagnostic or adopt a model. It tests the existing model comparison over more chronological regimes. Six fixed development reserves and all 2025/2026 consumers remain unread. Full933/full60, original max60/schedule/optimizer/selection/purges preserved. Report paired20/40/60-session development intervals, all seeds/folds and economic risks; no retrospective claim of unbiased confirmation.",
    )
    write_json_atomic(out / "plan.json", plan)
    write_json_atomic(out / "frozen_design.json", dict(store=original["store"]))
    write_json_atomic(
        out / "refits.json",
        dict(
            status="frozen",
            plan=binding(out / "plan.json"),
            completed=completed,
            planned=54,
        ),
    )
    old_eval = bound_json(run["stage_c_data_replay_plan"])
    old_progress = bound_json(
        binding(Path(run["stage_c_data_replay_plan"]["path"]).parent / "replays.json")
    )
    reused_books = [
        r
        for r in old_progress["completed"]
        if r["key"].split("/")[2] in investigation["arms"]
    ]
    assert len(reused_books) == 48
    evaluation = dict(
        old_eval,
        fit_root=str(out),
        arms=investigation["arms"],
        folds=investigation["expansion_folds"],
        planned_books=96,
        refits=binding(out / "plan.json"),
        registration=run["scaling_investigation"],
        scope="48 new primary books on the four additional folds plus48 reused original corrected books. Original R10m seeds/ensemble and R1m/R5m ensemble checks. No new policy tuning or repeated broad sensitivity grid; inspect actual new path exposures and apply only necessary existing bounds before a material economic conclusion.",
    )
    evaluation_root = out / "evaluation"
    evaluation_root.mkdir()
    write_json_atomic(evaluation_root / "plan.json", evaluation)
    write_json_atomic(
        evaluation_root / "replays.json",
        dict(
            status="frozen",
            plan=binding(evaluation_root / "plan.json"),
            completed=reused_books,
            planned=96,
        ),
    )
    run["scaling_expanded_fit_plan"] = binding(out / "plan.json")
    run["scaling_expanded_evaluation_plan"] = binding(evaluation_root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(
        json.dumps(
            dict(
                frozen=True, new_fits=24, reused_fits=30, new_books=48, reused_books=48
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    if args.freeze:
        freeze(run)
    elif args.evaluate:
        replay_data_refits.execute(
            dict(run, stage_c_data_replay_plan=run["scaling_expanded_evaluation_plan"])
        )
    else:
        plan = bound_json(run["scaling_expanded_fit_plan"])
        assert sha256_file(Path(__file__)) == plan["freezer"]["sha256"]
        for alias in plan["directory_aliases"]:
            assert Path(alias["path"]).is_junction()
            assert Path(alias["path"]).resolve() == Path(alias["target"]).resolve()
        run_capacity_width.execute(
            dict(run, stage_d_width_plan=run["scaling_expanded_fit_plan"])
        )
        run = json.loads(pointer.read_text())
        run["scaling_expanded_fits"] = binding(
            Path(run["scaling_expanded_fit_plan"]["path"]).parent / "refits.json"
        )
        write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
