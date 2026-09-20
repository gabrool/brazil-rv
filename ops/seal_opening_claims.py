"""Independent saved-book NAV identity and bounded engineering acceptance."""

import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    audit = bound_json(run["opening_claim_audit"])
    root = Path(audit["audit_root"])
    output = root / "account_identity"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "executed.py")
    cases = bound_json(audit["completed"])
    max_error = 0.0
    for case in cases:
        with np.load(root / case["book"] / "account.npz") as a:
            nav = (
                a["free_cash"]
                + a["restricted_cash"]
                + a["hedge_restricted_cash"]
                + a["unsettled_cash"]
                + a["receivables"]
                - a["payables"]
                + (a["signed_shares"] * np.nan_to_num(a["mark_price"])).sum(1)
                + a["hedge_signed_shares"] * np.nan_to_num(a["hedge_mark_price"])
                - a["loan_liability"]
            )
            err = float(np.max(np.abs(nav - a["nav"])))
            assert err < 5e-8
            max_error = max(max_error, err)
    identity = dict(
        status="passed_independent_component_identity",
        days=30 * len(cases),
        max_error_brl=max_error,
        preceding_qualification=run["opening_claim_audit"],
        readout_correction="The executed completed.json opening_claim_value is end-of-effect-day undelivered value after current quotes. Independent scalar oracles report the causal opening value separately. Canonical producer label is effect_close_claim_value; no array/book rerun.",
        runtime_qualification="Actual account references/marks are Float64. Widening recognize_distribution's scalar input changes only its earlier direct Float32 oracle; the executed accounts are numerically unchanged. V33 is a subsequent evaluation-schema label; no book numeric changes.",
        seconds=perf_counter() - started,
    )
    write_json_atomic(output / "manifest.json", identity)
    run["opening_claim_identity"] = binding(output / "manifest.json")
    run["opening_claim_terms"] = audit["terms"]
    acceptance = dict(
        status="opening_claim_engineering_accepted_stage_A_incomplete",
        schema="BRAZIL_RV_V2_EVALUATION_V33",
        evidence={
            k: run[k]
            for k in (
                "held_event_source_audit",
                "held_event_source_qualification",
                "opening_claim_audit",
                "opening_claim_identity",
                "opening_claim_terms",
            )
        },
        report=binding(PROJECT / "docs/v2_OPENING_CLAIM_VALUATION.md"),
        unchanged="All old stores, source observations, original fits,933names/fullhistory and previous V31/V32 obligations; no GPU/forward/scoring.",
        scope="30x30-session all933-name actual adaptive engineering books; focus-long/short, three capital sizes, explicit SOMA fraction/price one-factor bounds. Synthetic preferences and fixed risks, bundled4bp execution bridge, no B3 spot addition.",
        remaining="NATU same-ISIN bonus custody/JCP withholding/payment; event-specific loan treatment and ALSC custody/auction; ENAT and remaining execution/cost/clearing bounds; registered C/D.",
        no_final_corporate_replay_replacement=True,
        tests="17 share-distribution tests passed; five of those reran after adding independent unquoted-delivery state; two new/affected first-quote and actual loader cases passed. Do not sum overlapping batches. Ruff passes.",
    )
    write_json_atomic(PROJECT / "docs/v2_opening_claim_acceptance.json", acceptance)
    run["opening_claim_acceptance"] = binding(
        PROJECT / "docs/v2_opening_claim_acceptance.json"
    )
    write_json_atomic(pointer, run)
    print(json.dumps(identity))


if __name__ == "__main__":
    main()
