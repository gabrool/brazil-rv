"""Reuse the qualified account drivers for the admitted ten-fold replication."""

import argparse
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
import bound_refit_debit
import qualify_refit_books
import replay_data_refits
import run_refit_sensitivities

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run):
    ref = run["stage_c_replication_plan"]
    fits = bound_json(ref)
    assert fits["candidates"], "No replication was admitted"
    root = Path(ref["path"]).parent
    context = root / "evaluation_context"
    (context / "docs").mkdir(parents=True, exist_ok=False)
    primary = root / "primary_books"
    primary.mkdir(exist_ok=False)
    old = bound_json(run["stage_c_data_replay_plan"])
    # Only folds, admitted arms and fit provenance change; accounting is frozen.
    plan = dict(
        old,
        status="frozen_before_replication_economics",
        original_screen=run["stage_c_data_replay_plan"],
        fit_root=str(root),
        refits=ref,
        arms=fits["arms"],
        folds=fits["folds"],
        planned_books=len(fits["arms"]) * len(fits["folds"]) * 6,
        attribution="Other ten development folds of the nominated economic replication; same accepted coordinates/account/source hypotheses and neutral policy. No old-coordinate pure-data attribution or untouched-validation claim.",
    )
    assert plan["store"] == fits["store"]
    write_json_atomic(primary / "plan.json", plan)
    local = dict(
        run,
        root=str(root),
        stage_c_data_replay_plan=binding(primary / "plan.json"),
    )
    pointer = context / "docs/v2_economic_data_scaling_run.json"
    write_json_atomic(pointer, local)
    prior_scenarios = bound_json(run["stage_c_refit_sensitivity_plan"])
    scenarios = root / "matched_refit_sensitivities"
    scenarios.mkdir(exist_ok=False)
    write_json_atomic(
        scenarios / "plan.json",
        dict(
            primary=local["stage_c_data_replay_plan"],
            original_hypotheses=run["stage_c_refit_sensitivity_plan"],
            phases=prior_scenarios["phases"],
            planned_primary_ensembles=len(fits["arms"]) * len(fits["folds"]) * 3,
            scope="The existing cost/loan/delivery phase definitions on every admitted replication ensemble and capital, using their unchanged actual-exposure predicates.",
            additional_conditional="The other ten folds also intersect2019 and2023 corporate windows. Apply the previously qualified Natura bonus/JCP/custody, BRML/Dommo/Copel disposal/conversion/payment and other inherited engineering hypotheses if the new portfolios expose them. These extra event-specific bounds are not supplied by this phase list. Cielo cents and opposing-fill daytrade remain exposure-conditional. Final replication admission requires the complete exposure disposition.",
            driver=binding(Path(run_refit_sensitivities.__file__)),
        ),
    )
    local["stage_c_refit_sensitivity_plan"] = binding(scenarios / "plan.json")
    bound_refit_debit.PROJECT = context
    bound_refit_debit.freeze(local)
    specification = dict(
        replication=ref,
        primary=local["stage_c_data_replay_plan"],
        sensitivities=local["stage_c_refit_sensitivity_plan"],
        funded_denial=local["stage_c_refit_debit_plan"],
        context=str(context),
        initial_context=binding(pointer),
        drivers={
            name: binding(Path(module.__file__))
            for name, module in (
                ("primary", replay_data_refits),
                ("sensitivities", run_refit_sensitivities),
                ("funded_denial", bound_refit_debit),
                ("qualification", qualify_refit_books),
            )
        },
        scope="Reuse all completed keys and proofs. Each invocation consumes only ready three-seed groups, then runs the existing phase list on actual exposure and conditional funded denial. Additional2019/2023 event-specific hypotheses require their own actual-exposure disposition before final admission; completion of these books is not completion of that disposition. No new accounting engine, forecast inference, source census, checkpoint choice or fit. All private outcome pointers remain outside the source checkout.",
    )
    # The mutable private run pointer subsequently acquires qualification pointers.
    (context / "initial_run.json").write_bytes(pointer.read_bytes())
    specification["initial_context"] = binding(context / "initial_run.json")
    write_json_atomic(root / "evaluation_plan.json", specification)
    run["stage_c_replication_evaluation_plan"] = binding(root / "evaluation_plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


def execute(run):
    spec = bound_json(run["stage_c_replication_evaluation_plan"])
    context = Path(spec["context"])
    pointer = context / "docs/v2_economic_data_scaling_run.json"
    local = json.loads(pointer.read_text())
    assert local["stage_c_data_replay_plan"] == spec["primary"]
    assert local["stage_c_refit_sensitivity_plan"] == spec["sensitivities"]
    assert local["stage_c_refit_debit_plan"] == spec["funded_denial"]
    for name, module in (
        ("primary", replay_data_refits),
        ("sensitivities", run_refit_sensitivities),
        ("funded_denial", bound_refit_debit),
        ("qualification", qualify_refit_books),
    ):
        assert binding(Path(module.__file__)) == spec["drivers"][name]
        module.PROJECT = context
    replay_data_refits.execute(local)
    qualify_refit_books.main()
    local = json.loads(pointer.read_text())
    run_refit_sensitivities.execute(local)
    qualify_refit_books.main(sensitivities=True)
    local = json.loads(pointer.read_text())
    bound_refit_debit.execute(local)
    local = json.loads(pointer.read_text())
    receipt = {
        key: local[key]
        for key in (
            "stage_c_data_replay_qualification",
            "stage_c_refit_sensitivity_qualification",
            "stage_c_refit_debit_qualification",
        )
    }
    receipt["plan"] = run["stage_c_replication_evaluation_plan"]
    receipt["additional_corporate_hypotheses_disposition_required"] = True
    output = Path(spec["replication"]["path"]).parent / "evaluation_progress.json"
    write_json_atomic(output, receipt)
    run["stage_c_replication_evaluation"] = binding(output)
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run) if args.freeze else execute(run)
