"""Dispose of the registered C screen, preserving conditional account limits."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    result = bound_json(run["stage_c_refit_results"])
    books = bound_json(result["books"])
    scenarios = bound_json(result["sensitivities"]["books"])
    comparisons = bound_json(result["comparisons"])
    proof = bound_json(result["qualification"])
    fraction = bound_json(run["stage_c_pending_fraction_qualification"])
    assert fraction["passed"] and len(fraction["books"]) == 3
    assert proof["status"] == "complete" and proof["qualified"] == len(books) == 96
    for source in result["sensitivities"]["sources"]:
        assert bound_json(source["qualification"])["status"] == "complete"
    assert not any(x["loan_cash_bounds_pending"] for x in [*books, *scenarios])
    assert not any(x["maximum_overdue_principal"] for x in books)
    terminal = [
        dict(book=x["source"]["key"], claim=claim)
        for x in books
        for claim in x["terminal_unquoted"]
    ]
    # This is the sourced unknown-auction case already admitted by Stage A.
    # A newly exposed terminal identity needs its own disposition.
    assert all(x["claim"]["isin"] == "BRENATACNOR0" for x in terminal)
    leads = [
        next(
            r
            for r in comparisons
            if r["candidate"] == arm
            and r["reference"] == "TE_full"
            and r["capital"] == 10000000
        )
        for arm in ("TE_wide", "GRU_early")
    ]
    replicate = [
        r["candidate"]
        for r in leads
        if r["members"]["ensemble"]["equal_fold_mean_bps_day"] > 0
    ]
    admission = dict(
        status="screen_disposed_replication_required"
        if replicate
        else "stage_c_complete",
        results=run["stage_c_refit_results"],
        pending_fraction_qualification=run["stage_c_pending_fraction_qualification"],
        technical_comparisons_valid=True,
        stage_c_complete=not replicate,
        replicate=replicate,
        lead_decisions=[
            dict(
                candidate=r["candidate"],
                delta_bps_day=r["members"]["ensemble"]["equal_fold_mean_bps_day"],
                decision="replicate"
                if r["candidate"] in replicate
                else "conditional_stop",
                reason="Registered corrected four-fold mean advantage is positive"
                if r["candidate"] in replicate
                else "Registered corrected four-fold mean advantage is not positive; no ten-fold fit is authorized by this gate",
            )
            for r in leads
        ],
        primary_terminal_claims=terminal,
        primary_unresolved_books=sum(x["economics_unresolved"] for x in books),
        scenario_overdue_books=sum(bool(x["overdue_dates"]) for x in scenarios),
        scenario_terminal_overdue_books=sum(
            x["terminal_overdue_principal"] > 0 for x in scenarios
        ),
        primary_exposure_disposition="No primary overdue loan principal or loan cash redemption. Terminal unquoted ENAT fractions retain their signed, marked, unpaid claims under the already admitted unknown-auction contract. Known corporate delivery intervals and isolated missing own quotes retain qualified marks and obligations. This validates comparison under explicit hypotheses, not liquidation proceeds or an executable broker guarantee.",
        stress_disposition="Recall/denial overdue intervals and any terminal principal remain unresolved execution stress. No buy-in, penalty, replacement locate or extra charge is invented. Their arithmetic-qualified results are conditional and never adopted as executable gains. Financing variants apply to actual prior debit; skipped unexposed variants are not bounds on held exposures.",
        limits="100% historical CDI on eligible settled short proceeds and zero brokerage remain negotiated hypotheses, not quotes. Preserve historical tariff bounds, delivery/loan/invoice/valuation assumptions, unknown ENAT auction, source revision limits and all prior adaptive fixed-fee numerical uncertainty. Cielo held-loan cents and opposing daytrade fills remain exposure-conditional. Reused development and post-selection nomination are explicit; IC is diagnostic. No model adoption, held-out result or live capacity claim follows from technical admission.",
        next="Registered two-cell width wave may begin only when stage_c_complete is true. Depth, LSTM and further capacity retain their original conditions.",
        recipe=binding(Path(__file__)),
    )
    out = Path(result["books"]["path"]).parent / "admission.json"
    assert not out.exists()
    write_json_atomic(out, admission)
    run["stage_c_refit_admission"] = binding(out)
    write_json_atomic(pointer, run)
    print(json.dumps(admission["lead_decisions"]), flush=True)


if __name__ == "__main__":
    main()
