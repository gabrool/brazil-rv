"""Bind the unchanged primary account and common dates for the stopping test."""

import argparse
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
import qualify_refit_books
import replay_data_refits

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    code = _git_identity()
    fits = bound_json(run["scaling_matched_stopping_plan"])
    inputs = bound_json(run["scaling_refit_economics"])
    quality = bound_json(run["scaling_refit_economics_qualification"])
    assert quality["passed"] and quality["inputs"] == run["scaling_refit_economics"]
    assert inputs["store"] == fits["store"]
    old = bound_json(run["scaling_expanded_evaluation_plan"])
    root = Path(fits["root"]) / "evaluation"
    root.mkdir(exist_ok=False)
    plan = dict(
        old,
        status="frozen_before_matched_stopping_economic_outcomes",
        store=fits["store"],
        fit_root=fits["root"],
        arms=list(fits["arms"]),
        folds=fits["folds"],
        inputs=run["scaling_refit_economics"],
        input_qualification=run["scaling_refit_economics_qualification"],
        economic_account=fits["economic_account"],
        terms=fits["account_terms"],
        refits=run["scaling_matched_stopping_plan"],
        planned_books=4 * 8 * 6,
        driver=binding(Path(replay_data_refits.__file__)),
        freezer=binding(Path(__file__)),
        runtime=code,
        registration=fits["registration"],
        policy="Original neutral5% net equal-rank calibration, same limits, original allocator and no policy retraining. R10m primary and all three seeds; R1m/R5m ensembles. The later flexible45% exposure contrast is separate.",
        attribution="Within the new accepted store: width and parent stopping contrasts. Comparison to the preceding expanded baseline combines specifically evidenced new data plus required refits; it is not a pure stopping effect across old and new stores. Current account terms compose the already qualified source overlays before these outcomes.",
        scope="192 logical primary books on the same993sessions/eight independent period accounts. Reuse identical jobs if identical selected parents lead to identical forecasts. No new cost grid, reserved-fold or held-out consumer. Retain measured adaptive minimum-fee uncertainty; apply existing conditional bounds only to actual new exposures before economic conclusions.",
    )
    write_json_atomic(root / "plan.json", plan)
    run["scaling_matched_stopping_evaluation_plan"] = binding(root / "plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(
        json.dumps(
            dict(
                plan=run["scaling_matched_stopping_evaluation_plan"], logical_books=192
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    if args.freeze:
        freeze(run)
    elif args.qualify:
        qualify_refit_books.main(
            plan_key="scaling_matched_stopping_evaluation_plan",
            output_key="scaling_matched_stopping_qualification",
        )
    else:
        replay_data_refits.execute(
            dict(
                run,
                stage_c_data_replay_plan=run[
                    "scaling_matched_stopping_evaluation_plan"
                ],
            )
        )
