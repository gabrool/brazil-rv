"""Common-date neutral and flexible exposure comparisons without policy fitting."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.research_rounds import _git_identity
import qualify_refit_books
import replay_data_refits

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    code = _git_identity()
    comparison = bound_json(run["scaling_common_model_plan"])
    primary = bound_json(run["scaling_matched_stopping_evaluation_plan"])
    assert comparison["store"] == primary["store"]
    root = Path(comparison["root"])
    plans = {}
    for policy, net_cap in (("neutral", 0.05), ("flexible", 0.45)):
        arms = list(comparison["cells"])
        if policy == "flexible":
            arms += list(primary["arms"])
        destination = root / policy
        destination.mkdir(exist_ok=False)
        allocation = replace(AllocationConfig(), net_cap=net_cap)
        plan = dict(
            primary,
            status="frozen_before_common_model_policy_outcomes",
            arms=arms,
            fit_root=comparison["root"],
            fit_roots={
                a: comparison["root"]
                if a in comparison["cells"]
                else primary["fit_root"]
                for a in arms
            },
            allocation=asdict(allocation),
            refits=run["scaling_common_model_plan"],
            attention_refits=run["scaling_matched_stopping_plan"],
            planned_books=len(arms) * len(primary["folds"]) * 6,
            driver=binding(Path(replay_data_refits.__file__)),
            freezer=binding(Path(__file__)),
            runtime=code,
            policy=f"Fixed {net_cap:.0%} absolute net and5% beta caps, original equal-rank calibration/allocator. No calibration or policy fitting. All other allocation and account settings unchanged.",
            attribution="Same accepted store, qualified account inputs and eight fixed periods. C6/GRU retain their original distinct training recipes; attention parent-stopping arms remain explicit. Flexible-minus-neutral changes net cap only in both the optimizer and ledger permission. Neutral attention books are reused from the stopping plan, never replayed here.",
            scope="R10m all seeds and ensembles, R1m/R5m ensembles. Eight independent period accounts; not a continuous1738-session counterpart. Conditional exposure bounds and unresolved claims remain. Exact identical logical forecasts can reuse one book within this policy only.",
            neutral_attention=run["scaling_matched_stopping_evaluation_plan"],
        )
        write_json_atomic(destination / "plan.json", plan)
        plans[policy] = binding(destination / "plan.json")
    run["scaling_common_evaluation_plans"] = plans
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(json.dumps(plans), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--policy", choices=("neutral", "flexible"))
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    if args.freeze:
        freeze(run)
    else:
        assert args.policy
        key = f"scaling_common_{args.policy}_evaluation_plan"
        run[key] = run["scaling_common_evaluation_plans"][args.policy]
        if args.qualify:
            # The existing qualifier resolves flat canonical pointer keys.
            pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
            write_json_atomic(pointer, run)
            qualify_refit_books.main(
                plan_key=key,
                output_key=f"scaling_common_{args.policy}_qualification",
            )
        else:
            replay_data_refits.execute(dict(run, stage_c_data_replay_plan=run[key]))
