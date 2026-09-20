"""Bind the completed cost block and bounded calendar search without book replay."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    base = Path(run["root"])
    root = base / "spot_invoice"
    sources = base / "closeout_calendar_sources"
    audit = bound_json(binding(root / "manifest.json"))
    qualified = bound_json(binding(root / "qualification/report.json"))
    assert qualified["books"] == len(bound_json(audit["completed"])) == 6
    assert (qualified["cells"], qualified["actual_fills"]) == (1722762, 4287)
    runtime = []
    for folder in ("execution", "v2"):
        for path in sorted((PROJECT / "research/src/brazil_rv" / folder).glob("*.py")):
            current = binding(path)
            executed = binding(root / f"executed_{folder}_{path.name}")
            assert current["sha256"] == executed["sha256"]
            runtime.append(dict(current=current, executed=executed))
    write_json_atomic(root / "runtime_identity.json", runtime)
    attempts = dict(
        test_logs={
            k: binding(root / k / "stdout.txt")
            for k in ("initial_tests", "qualified_tests", "final_tests")
        },
        distinct_new_tests=5,
        affected_existing_tests=53,
        first="4pass/1fail: fixture notionals yielded exact 2024 cents. Only test target adjusted to exercise nonzero precision.",
        second="57pass/1fail: prior funding equality differed by2.22e-16; only narrow fixture absolute tolerance5e-15 applied. qualified_tests is not an all-passed folder.",
        final="2affected parameter cases pass,3deselected. Overlapping counts are not additive. No production correction or historical book rerun.",
        books="All6 first invocation, qualification first invocation. All6 saved parent books reused without replay.",
        reconstructed_preflight="Plan iteration accidentally visited count/variants keys before resolved plan/books; corrected selection of two named windows. Read-only guessed qualifier paths, rg wildcard and incompatible Get-Content Raw/TotalCount queries changed no artifacts. Prose reconstructed, not preserved exact shell bytes.",
        lint=binding(root / "final_ruff.log"),
    )
    write_json_atomic(root / "attempts.json", attempts)
    receipts = {
        path.stem.removesuffix("_receipt"): binding(path)
        for path in sorted(sources.glob("*_receipt.json"))
    }
    visual = {
        "b3_131_2015_calendar": [
            binding(sources / f"b3_131_2015_page-{p}.png") for p in range(1, 6)
        ],
        "cblc_sita_manual": [binding(sources / "cblc_sita_manual_p1.png")],
        "b3_prior_manual": [
            binding(sources / f"b3_prior_manual_p{p}.png") for p in (1, 134)
        ],
    }
    calendar = dict(
        status="bounded_source_attempt_complete_two_dates_still_unresolved",
        prior_resolution=run["clearing_calendar_resolution"],
        receipts=receipts,
        originals={k: binding(sources / f"{k}.pdf") for k in visual},
        visual_pages=visual,
        dispositions={
            "b3_131_2015_calendar": "OriginalDecember8 2015/scanned5pages. December30 nontrading and separateFX/OTC/Treasury rules do not explicitly prove former equity-clearing delivery or loan-rent calendar. Extracted text empty because scanned, all5visualpages readable.",
            "cblc_sita_manual": "BMFBOVESPA-authored March2011, third-party SITA mirror. Cover visually checked; no2016/2017 revision continuity or new calendar rule admitted. Not a complete manual audit.",
            "b3_prior_manual": "B3-hosted February24 2017 draft explicitly NOT approved byBCB/CVM. Cover andp134 readable; trading-day definition cannot be backdated/admitted as an active old-equity rule.",
            "cblc_modal_manual": "Selected mirror403; not bypassed.",
            "elektro_20160820": "Selected official-gazette URL404. Secondary Escavador reproduction is a lead only, not admitted original issuer evidence.",
        },
        unchanged_resolved_money_only_dates=21,
        remaining_ambiguous_dates=["2016-12-30", "2017-01-25"],
        changed_calendar_or_model_cells=0,
        no_calendar_bound_established=True,
        next="Specific dated disposition or separately frozen date hypotheses within composition/integrated admission. Do not repeat this bounded search or add all money-only dates. Cash delivery and rent accrual remain distinct.",
        attempt_notes="Three executed retrieval recipes and original receipts retained. Initial official-gazette404 and Modal403 retained. Poppler font warnings did not prevent8selected-page review; rendering durations/failed web queries not comprehensively timed. No source-completeness or exact historical web-publication claim.",
    )
    write_json_atomic(sources / "qualification.json", calendar)
    evidence = {
        "spot_invoice_audit": binding(root / "manifest.json"),
        "spot_invoice_qualification": binding(root / "qualification/report.json"),
        "spot_invoice_runtime": binding(root / "runtime_identity.json"),
        "spot_invoice_attempts": binding(root / "attempts.json"),
        "closeout_calendar_sources": binding(sources / "qualification.json"),
    }
    dependencies = {
        k: run[k]
        for k in (
            "historical_cost_recovery",
            "historical_cost_acceptance",
            "historical_tariff_sources",
            "historical_cost_sources",
            "enat_settlement_terms",
            "cash_calendar",
            "loan_source_panels",
            "bova_loan_reference_audit",
            "lending_feature_recovery",
            "natura_store_recovery",
            "cielo_foundation_exposure_recovery",
            "clearing_calendar_resolution",
        )
    }
    for record in dependencies.values():
        bound_json(record)
    accepted = dict(
        status="historical_cost_block_complete_for_bounded_contract_not_stage_a_admission",
        evidence=evidence,
        report=binding(PROJECT / "docs/v2_SPOT_INVOICE_CLOSEOUT.md"),
        closeout=binding(PROJECT / "docs/v2_STAGE_A_CLOSEOUT.md"),
        books=6,
        reused_parent_books=6,
        account_array_cells=qualified["cells"],
        sessions=qualified["sessions"],
        fills=qualified["actual_fills"],
        errors=qualified["errors"],
        adaptive_max_bps=qualified["adaptive_max_bps"],
        retained_prior_uncertainty="All larger V31-V41 and historical corporate fixed-fee adaptive uncertainty retained; new small discrepancies do not replace them.",
        contrasts=qualified["contrasts"],
        engineering_seconds=audit["seconds"],
        summed_case_seconds=audit["summed_case_seconds"],
        qualification_seconds=qualified["seconds"],
        contracts=[
            "Unrounded fractional research costs primary; security_day_6dp_cent is a separate one-factor hypothesis. Original source017/2023pp17-19 reused. Integer quantity applicability, notional half-up and prior-era backcast are not observed invoices.",
            "Invoice-minus-unrounded adjustment belongs to originalT3/T2 spot settlement. execution_charges and cost already include it; Fill.cost and hedge diagnostic remain unrounded. Funding/intentions precede recognition, no double charge.",
            "Normal one-account/phase cash market, all4287historicalgroupssinglefill/same-direction. Opposing same-day guard retained; actual future exposure needs specific daytrade admission. Multi-fill arithmetic fixture-only.",
            "All933names/frozenOLDcoordinates/fullhistory/originalcalibration. Synthetic adaptive paths, no neuralforward/scoring/fit/modelprofits. Oldstores/fits unchanged.",
        ],
        open_items=[
            "Two older clearing dates require sourced disposition or separate bounded hypotheses; new source search establishes no numerical bound.",
            "Primary event/source composition and separately attributed succession-data implications.",
            "Cielo cent variant only if actual loan cash redemption; none of six new paths exposed.",
            "Integrated StageA acceptance followed by registered C and conditional D.",
        ],
        prior_recovery_dependencies=dependencies,
        immutable_contracts={
            k: run[k] for k in ("corporate_replay", "economic_refit_inputs")
        },
    )
    path = PROJECT / "docs/v2_spot_invoice_acceptance.json"
    write_json_atomic(path, accepted)
    run.update(evidence)
    run["spot_invoice_acceptance"] = binding(path)
    run["stage_a_closeout_plan"] = binding(PROJECT / "docs/v2_STAGE_A_CLOSEOUT.md")
    write_json_atomic(pointer, run)
    print(json.dumps(dict(acceptance=run["spot_invoice_acceptance"], books=6)))


if __name__ == "__main__":
    main()
