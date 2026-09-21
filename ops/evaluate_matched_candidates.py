"""Reuse qualified account drivers for admitted replication or width fits."""

import argparse
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
import bound_refit_debit
import bound_refit_fraction_precision
import qualify_refit_books
import replay_data_refits
import run_refit_sensitivities

PROJECT = Path(__file__).resolve().parents[1]


def freeze(run, width=False):
    ref = run["stage_d_width_plan" if width else "stage_c_replication_plan"]
    fits = bound_json(ref)
    arms = list(fits["cells"]) if width else fits["arms"]
    assert arms, "No candidate fits were admitted"
    root = Path(ref["path"]).parent
    context = root / "evaluation_context"
    (context / "docs").mkdir(parents=True, exist_ok=False)
    primary = root / "primary_books"
    primary.mkdir(exist_ok=False)
    old = bound_json(run["stage_c_data_replay_plan"])
    # Only folds, admitted arms and fit provenance change; accounting is frozen.
    plan = dict(
        old,
        status="frozen_before_candidate_economics",
        original_screen=run["stage_c_data_replay_plan"],
        fit_root=str(root),
        refits=ref,
        arms=arms,
        folds=fits["folds"],
        planned_books=len(arms) * len(fits["folds"]) * 6,
        attribution=(
            "Registered width candidates on four screen folds; existing matched C controls are reused separately. "
            if width
            else "Other ten development folds of the nominated replication. "
        )
        + "Same accepted coordinates/account/source hypotheses and neutral policy. No pure-data attribution or untouched-validation claim.",
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
    (context / "docs/v2_opportunity_run.json").write_bytes(
        (PROJECT / "docs/v2_opportunity_run.json").read_bytes()
    )
    prior_scenarios = bound_json(run["stage_c_refit_sensitivity_plan"])
    scenarios = root / "matched_refit_sensitivities"
    scenarios.mkdir(exist_ok=False)
    write_json_atomic(
        scenarios / "plan.json",
        dict(
            primary=local["stage_c_data_replay_plan"],
            original_hypotheses=run["stage_c_refit_sensitivity_plan"],
            phases=prior_scenarios["phases"],
            planned_primary_ensembles=len(arms) * len(fits["folds"]) * 3,
            scope="The existing cost/loan/delivery phase definitions on every admitted replication ensemble and capital, using their unchanged actual-exposure predicates.",
            additional_conditional=(
                "Four original screen folds; no2019/2023 corporate windows. "
                if width
                else "Other ten folds also intersect2019/2023: apply qualified Natura, BRML/Dommo/Copel and other event hypotheses when exposed; this phase list does not supply those bounds. "
            )
            + "Cielo cents, opposing-fill daytrade and additional actual exposure require disposition before final admission.",
            driver=binding(Path(run_refit_sensitivities.__file__)),
        ),
    )
    local["stage_c_refit_sensitivity_plan"] = binding(scenarios / "plan.json")
    bound_refit_debit.PROJECT = context
    bound_refit_debit.freeze(local)
    bound_refit_fraction_precision.PROJECT = context
    bound_refit_fraction_precision.freeze(local)
    specification = dict(
        candidate_fits=ref,
        width=width,
        primary=local["stage_c_data_replay_plan"],
        sensitivities=local["stage_c_refit_sensitivity_plan"],
        funded_denial=local["stage_c_refit_debit_plan"],
        fraction_precision=local["stage_c_refit_fraction_precision_plan"],
        context=str(context),
        initial_context=binding(pointer),
        drivers={
            name: binding(Path(module.__file__))
            for name, module in (
                ("primary", replay_data_refits),
                ("sensitivities", run_refit_sensitivities),
                ("funded_denial", bound_refit_debit),
                ("fraction_precision", bound_refit_fraction_precision),
                ("qualification", qualify_refit_books),
            )
        },
        scope="Reuse all completed keys and proofs. Consume only ready three-seed groups, existing actual-exposure phases, conditional funded denial and qualified SOMA precision. Replication-only2019/2023 hypotheses and newly exposed cases need separate disposition before admission. No new account engine, inference, source census, selector or fit. Private outcome pointers stay outside the checkout.",
    )
    # The mutable private run pointer subsequently acquires qualification pointers.
    (context / "initial_run.json").write_bytes(pointer.read_bytes())
    specification["initial_context"] = binding(context / "initial_run.json")
    write_json_atomic(root / "evaluation_plan.json", specification)
    run[
        "stage_d_width_evaluation_plan"
        if width
        else "stage_c_replication_evaluation_plan"
    ] = binding(root / "evaluation_plan.json")
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)


def execute(run, width=False):
    reference = run[
        "stage_d_width_evaluation_plan"
        if width
        else "stage_c_replication_evaluation_plan"
    ]
    spec = bound_json(reference)
    assert spec["width"] == width
    context = Path(spec["context"])
    pointer = context / "docs/v2_economic_data_scaling_run.json"
    local = json.loads(pointer.read_text())
    assert local["stage_c_data_replay_plan"] == spec["primary"]
    assert local["stage_c_refit_sensitivity_plan"] == spec["sensitivities"]
    assert local["stage_c_refit_debit_plan"] == spec["funded_denial"]
    assert local["stage_c_refit_fraction_precision_plan"] == spec["fraction_precision"]
    for name, module in (
        ("primary", replay_data_refits),
        ("sensitivities", run_refit_sensitivities),
        ("funded_denial", bound_refit_debit),
        ("fraction_precision", bound_refit_fraction_precision),
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
    bound_refit_fraction_precision.execute(local)
    local = json.loads(pointer.read_text())
    receipt = {
        key: local[key]
        for key in (
            "stage_c_data_replay_qualification",
            "stage_c_refit_sensitivity_qualification",
            "stage_c_refit_debit_qualification",
            "stage_c_refit_fraction_precision_qualification",
            "stage_c_refit_fraction_precision_arithmetic",
        )
    }
    receipt["plan"] = reference
    receipt["additional_corporate_hypotheses_disposition_required"] = not width
    output = Path(spec["candidate_fits"]["path"]).parent / "evaluation_progress.json"
    write_json_atomic(output, receipt)
    run["stage_d_width_evaluation" if width else "stage_c_replication_evaluation"] = (
        binding(output)
    )
    write_json_atomic(PROJECT / "docs/v2_economic_data_scaling_run.json", run)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--width", action="store_true")
    args = parser.parse_args()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    freeze(run, args.width) if args.freeze else execute(run, args.width)
