"""Qualify new gross terms at the actual dated clock without repeating sources."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    align_action_payment_sessions,
    align_decision_known_action_terms,
    apply_contractual_action,
    verified_action_terms_from_table,
    verified_action_terms_to_table,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule, normalize_publication_time

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    initial = bound_json(run["held_event_source_audit"])
    root = Path(run["held_event_source_audit"]["path"]).parent.parent
    out = root / "clock_qualification"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "current_source_audit.py").write_bytes(
        (PROJECT / "ops/audit_held_event_sources.py").read_bytes()
    )
    before = verified_action_terms_from_table(
        pl.read_parquet(initial["gross_action_amendments"]["path"])
    )
    after = tuple(
        replace(
            term,
            available_at=normalize_publication_time(
                term.available_at, precision="minute"
            ),
        )
        for term in before
    )
    table = verified_action_terms_to_table(after)
    table.write_parquet(out / "natura_gross_action_amendments.parquet")
    assert (
        verified_action_terms_from_table(
            pl.read_parquet(out / "natura_gross_action_amendments.parquet")
        )
        == after
    )
    old = initial["old_store"]
    manifest = bound_json(
        {
            "path": str(Path(old["root"]) / "manifest.json"),
            "sha256": old["manifest_sha256"],
        }
    )
    schedule_rec = next(
        x
        for x in manifest["sources"]
        if Path(x["path"]).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    schedule = tuple(
        s
        for s in load_session_schedule(Path(schedule_rec["path"]))
        if str(s.trade_date) <= "2024-12-30"
    )
    isins = (after[0].isin,)
    assert all(t.isin == isins[0] for t in after)
    windows, counts = [], 0
    for old_term, term in zip(before, after, strict=True):
        idx = next(i for i, s in enumerate(schedule) if s.trade_date == term.ex_date)
        selected = schedule[idx - 2 : idx + 3]
        dates = [s.trade_date for s in selected]
        cutoffs = [s.decision_at for s in selected]
        coverage = np.ones((len(dates), 1), bool)

        def aligned(terms):
            return align_decision_known_action_terms(
                terms,
                dates,
                isins,
                coverage_resolved=coverage,
                decision_timestamps=cutoffs,
            )

        control, actual = aligned([old_term]), aligned([term])
        for key in (
            "shares_per_prior_share",
            "cash_per_prior_share",
            "session_resolved",
            "has_action",
            "successor_index",
        ):
            np.testing.assert_array_equal(getattr(control, key), getattr(actual, key))
            counts += getattr(actual, key).size
        empty = aligned([])
        late = aligned([replace(term, available_at=cutoffs[2] + timedelta(seconds=1))])
        assert not late.has_action[2, 0]
        np.testing.assert_array_equal(actual.has_action[:2], empty.has_action[:2])
        np.testing.assert_array_equal(
            actual.shares_per_prior_share[:2], empty.shares_per_prior_share[:2]
        )
        assert actual.has_action[2, 0]
        # Minute precision must not expose the term inside its recorded minute.
        for offset, expected in ((-1, False), (0, True)):
            boundary = align_decision_known_action_terms(
                [term],
                [term.ex_date],
                isins,
                coverage_resolved=np.ones((1, 1), bool),
                decision_timestamps=[
                    term.available_at + timedelta(microseconds=offset)
                ],
            )
            assert bool(boundary.has_action[0, 0]) == expected
        windows.append(
            {
                "ex_date": str(term.ex_date),
                "initial_available_at": old_term.available_at.isoformat(),
                "qualified_available_at": term.available_at.isoformat(),
                "actual_dates": [str(d) for d in dates],
                "decision_cutoffs": [c.isoformat() for c in cutoffs],
            }
        )
    payments = align_action_payment_sessions(
        after, [s.trade_date for s in schedule], isins
    )
    bonus_ix = next(
        i for i, s in enumerate(schedule) if s.trade_date == after[0].ex_date
    )
    jcp_ix = next(i for i, s in enumerate(schedule) if s.trade_date == after[1].ex_date)
    payment_ix = next(
        i for i, s in enumerate(schedule) if s.trade_date == after[1].payment_date
    )
    assert payments[bonus_ix, 0] == -1 and payments[jcp_ix, 0] == payment_ix
    # Gross entitlement helper, explicitly not settled cash or loan custody.
    for signed in (-100, 100):
        quantity, receivable = float(signed), 0.0
        expected_q, expected_cash = Decimal(signed), Decimal(0)
        for term in after:
            expected_cash += expected_q * Decimal(str(term.cash_per_prior_share))
            expected_q *= Decimal(str(term.shares_per_prior_share))
            quantity, receivable = apply_contractual_action(
                quantity,
                receivable,
                shares_per_prior_share=term.shares_per_prior_share,
                cash_per_prior_share=term.cash_per_prior_share,
            )
        assert quantity == float(expected_q)
        np.testing.assert_allclose(receivable, float(expected_cash), rtol=0, atol=4e-15)
    result = {
        "status": "passed_source_terms_and_clock_only_not_propagated_store_or_account",
        "source_audit": run["held_event_source_audit"],
        "amendments": binding(out / "natura_gross_action_amendments.parquet"),
        "schedule": schedule_rec,
        "clock_windows": windows,
        "actual_clock_control_cells": counts,
        "same_gross_terms": True,
        "qualification": "Initial rows used the start of each minute. The existing conservative minute-end clock adds one minute. All actual five-session q/cash/mask/successor controls remain exact; event effects are on later sessions. Initial rows and executed source audit remain preserved.",
        "future_and_delayed_receipt_checks": "Two future-source deletion prefixes, two delayed-event clocks, four minute-boundary checks pass.",
        "signed_gross_entitlement_oracle": "Both +/-100 original shares match independent Decimal q/cash arithmetic; this is neither settled cash nor custody, loan compensation or tax-credit accounting.",
        "cash_payment_date": str(after[1].payment_date),
        "bonus_credit_scope": "Sep20 physical share credit remains distinct from Sep18 q2. No cash payment is assigned to the bonus.",
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", result)
    run["held_event_source_qualification"] = binding(out / "manifest.json")
    run["natura_gross_action_amendments"] = result["amendments"]
    write_json_atomic(pointer, run)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
