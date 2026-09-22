"""Fresh C6/GRU controls on the accepted scaling data and eight fixed periods."""

import argparse
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
import run_economic_refits

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    code = _git_identity()
    source = bound_json(run["stage_c_refit_plan"])
    stopping = bound_json(run["scaling_matched_stopping_plan"])
    accepted = bound_json(run["scaling_data_inputs"])
    assert accepted["store"] == stopping["store"]
    root = Path(stopping["root"]).parent / "v2_common_model_comparison_20260921"
    root.mkdir(exist_ok=False)
    arms = ("GRU_early", "C6")
    plan = dict(
        source,
        status="frozen_before_common_model_refits",
        stage="user_step3",
        root=str(root),
        store=accepted["store"],
        acceptance=run["scaling_data_inputs"],
        cells={a: source["cells"][a] for a in arms},
        f_recipes={a: source["f_recipes"][a] for a in arms},
        parameter_counts={a: source["parameter_counts"][a] for a in arms},
        matched_graph_recipe_sources={
            a: source["matched_graph_recipe_sources"][a] for a in arms
        },
        folds=stopping["folds"],
        seeds=stopping["seeds"],
        driver=binding(Path(run_economic_refits.__file__)),
        freezer=binding(Path(__file__)),
        implementation=code,
        source_plan=run["stage_c_refit_plan"],
        comparison_attention=run["scaling_matched_stopping_plan"],
        registration=binding(
            PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
        ),
        planned_fits=54,
        scope="Two existing controls, three seeds, eight fixed periods, fresh compatible parents. Original GRU parent patience5 and C6 slow-parent patience3/every2-selection remain; original child recipes/max60/optimizer/selector are unchanged. This compares complete model recipes, not a pure encoder-only intervention or a new stopping search. Attention5/20 results will both remain visible. No held-out or six reserve-period consumers.",
        evaluation="Use the same admitted account, inputs and dates as the stopping test. Report neutral5% and flexible45% net policies separately, fixed beta5% and all other constraints/calibration unchanged. Freeze that executable evaluation before any book outcome; no best-seed selection or policy training.",
        continuous="The existing1738-session C6 reference remains a historical result. A complete corrected continuous counterfactual would consume the six protected development reserve periods and is deferred under the current boundary. Report only fully covered adjacent authorized blocks as continuous blocks; never splice the eight periods into one account or label them unbiased tests.",
    )
    write_json_atomic(root / "refit_plan.json", plan)
    write_json_atomic(root / "frozen_design.json", dict(store=accepted["store"]))
    run["scaling_common_model_plan"] = binding(root / "refit_plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(json.dumps(dict(plan=run["scaling_common_model_plan"], fits=54)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    if args.freeze:
        freeze(run)
    else:
        plan = bound_json(run["scaling_common_model_plan"])
        run_economic_refits.execute(run, Path(plan["root"]))
