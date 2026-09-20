"""Bind the bounded settlement engineering result and current runtime qualification."""

import difflib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    audit = bound_json(run["natura_settlement_audit"])
    probe = bound_json(run["natura_bonus_disposal_audit"])
    qualification = bound_json(run["natura_settlement_qualification"])
    output = Path(run["root"]) / "natura_settlement/runtime_qualification"
    output.mkdir(exist_ok=False)
    shutil.copyfile(__file__, output / "executed.py")
    old_path = Path(audit["audit_root"]) / "executed_execution_stateful_ledger.py"
    current_path = PROJECT / "research/src/brazil_rv/execution/stateful_ledger.py"
    old, current = old_path.read_text(), current_path.read_text()
    before = "& (np.abs(desired - decision_state.reserved_weights) > tol)"
    after = "& (\n                    (np.abs(desired) > np.abs(decision_state.reserved_weights) + tol)\n                    | (desired * decision_state.reserved_weights < -tol)\n                )"
    assert old.replace(before, after) == current
    assert (
        Path(probe["audit_root"]) / "executed_execution_stateful_ledger.py"
    ).read_text() == old
    (output / "runtime.patch").write_text(
        "".join(
            difflib.unified_diff(
                old.splitlines(True),
                current.splitlines(True),
                fromfile=str(old_path),
                tofile=str(current_path),
            )
        ),
        encoding="utf-8",
    )
    # The changed predicate only differs with a nonzero reserved bonus AND a
    # required exit. The one affected name/date has fresh prints, active scores,
    # young loans and no notices in every completed historical engineering book.
    terms = bound_json(audit["terms"])
    store = Path(terms["store"]["root"])
    dates = np.load(terms["calendar"]["path"])
    rows = np.flatnonzero(
        (dates >= np.datetime64("2019-09-10")) & (dates <= np.datetime64("2019-09-19"))
    )
    active = np.load(store / "active.npy", mmap_mode="r")
    close = np.load(store / "raw_close.npy", mmap_mode="r")
    assert active[rows, 605].all()
    assert np.isfinite(close[rows, 605]).all() and (close[rows, 605] > 0).all()
    for receipt in (audit, probe):
        for case in bound_json(receipt["completed"]):
            book = json.loads(
                (Path(receipt["audit_root"]) / case["book"] / "book.json").read_text()
            )
            config = book["provenance"]["config"]
            assert not config["loan_recalls"] and config["approve_loan_renewals"]
            assert config["loan_term_sessions"] - 4 > len(rows)
    tests = PROJECT / "research/tests/test_action_settlement.py"
    shutil.copyfile(tests, output / "qualified_tests.py")
    runtime = dict(
        status="qualified_without_book_reruns",
        current=binding(current_path),
        executed=binding(old_path),
        patch=binding(output / "runtime.patch"),
        proof="Only the mandatory-exit predicate changed. Completed books have no required Natura exit during reserved bonus dates: all active/fresh, flat-start loans younger than renewal/term, no recalls. Other coordinates have zero reserve, where both predicates are identical.",
        tests="14 distinct new settlement cases passed across focused batches; three affected existing split/custody/minimum cases also passed. Overlapping repeats not summed. Ruff passes.",
        failures=[
            "Initial credit-hypothesis prefix assertion demanded bitwise NAV equality; classification ordering differed by2.220446049250313e-16. Narrow absolute1e-15 arithmetic tolerance qualified it; no account formula changed.",
            "A new recalled-bonus/zero-target fixture found the strict desired-equals-reserve rule incorrectly rejected an intention to liquidate. Current guard allows zero or a reduction within the reserved portion, while actual fills retain undelivered shares and overdue loans. One affected case reran and passed;42historical books reused.",
        ],
        seconds=perf_counter() - tick,
    )
    write_json_atomic(output / "manifest.json", runtime)
    run["natura_settlement_runtime_qualification"] = binding(output / "manifest.json")
    acceptance = dict(
        status="natura_settlement_engineering_accepted_stage_A_incomplete",
        schema="BRAZIL_RV_V2_EVALUATION_V34",
        evidence={
            k: run[k]
            for k in (
                "natura_gross_action_amendments",
                "held_event_source_audit",
                "held_event_source_qualification",
                "opening_claim_terms",
                "natura_settlement_audit",
                "natura_bonus_disposal_audit",
                "natura_settlement_qualification",
                "natura_settlement_runtime_qualification",
                "natura_settlement_terms",
            )
        },
        report=binding(PROJECT / "docs/v2_NATURA_ACCOUNT_SETTLEMENT.md"),
        books=qualification["books"],
        daily_identities=qualification["daily_identities"],
        primary="Source q2/effectSep18/creditSep20; grossJCP.12784527353 Nov7/payFeb26, long15%withholding/no taxcredit; gross lender compensation and whole-loan-return deferral are explicit hypotheses.",
        unchanged="Both accepted stores, original sources/fits, frozen old PolicyData coordinates and full933-name axes; no model scoring, new fit, held-out consumer, history change or corporate_replay pointer replacement.",
        limitations="Trade-date entitlement; no event-specific lender instruction or tax-credit value. Synthetic fixed-risk books/oldbundled4bp bridge only. Preserve all measured adaptive fixed-minimum uncertainty. Remaining NATU/ALSC/SOMA custody/loan/fraction evidence, ENAT, BRML/DMMO/Copel disposal and fee/clearing bounds; final A and C/D incomplete.",
    )
    path = PROJECT / "docs/v2_natura_settlement_acceptance.json"
    write_json_atomic(path, acceptance)
    run["natura_settlement_acceptance"] = binding(path)
    write_json_atomic(pointer, run)
    print(json.dumps(runtime))


if __name__ == "__main__":
    main()
