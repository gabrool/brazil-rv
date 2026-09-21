"""Publish the exact account/source composition already qualified by Stage C."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_refit_admission"])
    assert admission["stage_c_complete"] and admission["technical_comparisons_valid"]
    plan = bound_json(run["stage_c_data_replay_plan"])
    previous = bound_json(plan["economic_account"])
    assert run["economic_account"] == plan["economic_account"]
    assert plan["terms"] == run["stage_c_event_candidate_terms"]
    accepted_data = bound_json(run["economic_refit_inputs"])
    assert accepted_data["account_terms"] == plan["terms"]
    # Reuse the actual completed books, cached qualification and recovery receipts.
    evidence = {
        name: run[name]
        for name in (
            "stage_c_event_source_admission",
            "stage_c_event_qualification",
            "stage_c_refit_economics_qualification",
            "stage_c_refit_action_identity",
            "stage_c_pending_fraction_qualification",
            "stage_c_refit_admission",
            "stage_c_refit_recovery",
        )
    }
    for record in evidence.values():
        bound_json(record)
    terms = bound_json(plan["terms"])
    root = Path(run["stage_c_root"]) / "account_composition"
    root.mkdir(exist_ok=False)
    (root / "executed.py").write_bytes(Path(__file__).read_bytes())
    account = dict(
        previous,
        status="matched_account_admitted_with_explicit_hypotheses",
        supersedes=plan["economic_account"],
        corporate_terms=plan["terms"],
        new_refit_inputs=run["economic_refit_inputs"],
        matched_portfolio_inputs=plan["inputs"],
        evidence={**previous["evidence"], **evidence},
        scenario_dependencies={
            **previous["scenario_dependencies"],
            "matched_results": run["stage_c_refit_results"],
        },
    )
    assert account["primary_config"] == previous["primary_config"]
    assert account["source_inputs"] == previous["source_inputs"]
    write_json_atomic(root / "account.json", account)
    report = dict(
        status="qualified_matched_account_composition_admitted",
        account=binding(root / "account.json"),
        previous_account=plan["economic_account"],
        previous_terms=run["corporate_replay"],
        corporate_terms=plan["terms"],
        accepted_model_inputs=run["economic_refit_inputs"],
        evidence=evidence,
        unchanged="Exact primary configuration, cash/lending/source inputs and already executed corporate terms. Candidate producer status/provenance is preserved byte-for-byte; this wrapper supplies its subsequent acceptance. Existing frozen plans, stores, fits and books retain their original bindings.",
        admission_scope="Canonical account/source metadata only, reusing complete Stage C qualification. No new source retrieval, numerical replay, model inference, fit or repeated verification campaign. Ongoing capacity plans continue their exact frozen account and term bindings.",
        pending_source_cases=terms["pending_cases"],
        primary_exposure_disposition=admission["primary_exposure_disposition"],
        stress_disposition=admission["stress_disposition"],
        limits=admission["limits"],
        recipe=binding(root / "executed.py"),
    )
    write_json_atomic(root / "manifest.json", report)
    write_json_atomic(PROJECT / "docs/v2_matched_account_acceptance.json", report)
    run["stage_c_account_composition"] = binding(root / "manifest.json")
    run["economic_account"] = report["account"]
    run["economic_account_acceptance"] = binding(
        PROJECT / "docs/v2_matched_account_acceptance.json"
    )
    run["corporate_replay"] = plan["terms"]
    write_json_atomic(pointer, run)
    print(json.dumps(dict(status=report["status"], account=report["account"])))


if __name__ == "__main__":
    main()
