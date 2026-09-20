"""Bind the new evidence and intermediate amendments without accepting a store."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    keys = (
        "held_event_source_audit",
        "held_event_source_qualification",
        "natura_gross_target_attribution",
        "natura_wealth_amendment",
    )
    reports = {key: bound_json(run[key]) for key in keys}
    source, clock, targets, wealth = [reports[k] for k in keys]
    result = {
        "status": "original_sources_and_bounded_gross_data_amendments_verified_A_and_new_dependency_admission_incomplete",
        "evidence": {key: run[key] for key in keys},
        "qualified_gross_terms": run["natura_gross_action_amendments"],
        "accepted_baseline_preserved": source["accepted_derived_store"],
        "source_counts": {
            "bounded_issuer_records": source["index_rows"],
            "new_original_pdfs": len(source["source_files"]),
            "initial_quote_rows": source["bounded_quote_rows"],
            "wealth_quote_rows_overlapping_initial": wealth["quote_rows"],
        },
        "scope": "Original NATU/ALSC/SOMA shareholder source contracts; two source-proven Natura q/cash replacements, clock/signed-entitlement qualification, bounded gross targets at fixed accepted risk and daily Float32 wealth. No final features/store, account treatment or model result.",
        "action_corrections": source["natura_scalar_comparisons"],
        "available_at_resolution": clock["clock_windows"],
        "targets": {
            k: targets[k]
            for k in (
                "dates",
                "names",
                "control_cells",
                "original_endpoint_oracles",
                "future_endpoint_prefix_cells",
                "effects",
            )
        },
        "wealth": {
            k: wealth[k]
            for k in (
                "quote_rows",
                "control_cells",
                "effects",
                "event_decimal_checks",
                "seed",
            )
        },
        "source_contract_gaps": [
            "ALSC physical credit and auction terms",
            "NATU/ALSC/SOMA loan event clocks, fraction/rent treatment and causal opening valuation",
            "Natura same-ISIN bonus custody; JCP withholding, supported tax-credit and lender compensation",
        ],
        "remaining_data": "Newly evidenced scalar and succession propagation through daily/native/auxiliary history/risk coordinates and final virtual neutral targets/actual tensors requires separate complete acceptance before new P/F fits. The previous full store remains its sealed baseline, not a substitute for these amendments.",
        "remaining_program": "ENAT next by held exposure; adaptive pre-custody disposal, allocation/timing/sweep/fraction/invoice/grouping/old-minimum and older clearing bounds, final corporate costs, C matched economics/data refits and conditional D waves.",
        "failed_attempts": {
            "source_index": source["attempt_dispositions"]["index_first_attempt"],
            "initial_target": "Index files were incorrectly looked up under manifest arrays; failure before target construction.",
            "second_target": "2D normalized-cross-section validity was indexed as 3D in future-prefix audit. Old control and Decimal endpoints passed before this harness failure. Target results had not been persisted; only the short calculation was repeated.",
            "clock_metadata": clock["qualification"],
            "wealth_inspection": "A read-only search for an invalid prior Natura wealth row found none; the explicit common prior close seeds the successful recurrence. No historical wealth reconstruction failed.",
        },
        "source_limits": source[
            "first_internet_publication_and_unselected_disclosure_completeness"
        ],
        "validation": "Canonical term loader/alignment/payment/gross arithmetic; actual target producer/old controls and independent Decimal endpoints; bounded original wealth recurrence/control and future deletion. Ruff passes. No old lifecycle suite or source census repeated; no neural input acceptance claimed.",
        "timing_seconds_excluding_retrieval_and_failed_attempts": {
            key: reports[key]["seconds"] for key in keys
        },
        "no_old_store_fit_or_production_research_mutation": True,
    }
    path = PROJECT / "docs/v2_held_event_source_acceptance.json"
    write_json_atomic(path, result)
    run["held_event_source_acceptance"] = binding(path)
    run["new_source_amendment_readiness"]["qualification"] = binding(path)
    write_json_atomic(pointer, run)
    print(json.dumps({"acceptance": binding(path), "status": result["status"]}))


if __name__ == "__main__":
    main()
