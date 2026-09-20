"""Qualify prior delayed-auction scope and zero-exposure Cielo controls."""

import json
from pathlib import Path
import shutil

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["payment_bounds_audit"])
    root = Path(audit["audit_root"])
    candidates = [
        run[k]
        for k in (
            "corporate_replay",
            "opening_claim_terms",
            "natura_settlement_terms",
            "enat_settlement_terms",
        )
    ]
    for key in ("precredit_disposal_acceptance", "copel_loan_bounds_acceptance"):
        candidates.extend(bound_json(run[key])["terms"].values())
    checked = {}
    for record in candidates:
        terms = bound_json(record)
        branch = [
            a
            for event in terms["share_distributions"]
            for leg in event["legs"]
            if (a := leg.get("fractional_auction"))
            and a.get("provision_loan_fractions")
            and a.get("available_date") is not None
            and a["available_date"] == a["payment_date"]
        ]
        assert not branch
        checked[record["sha256"]] = record
    runtime = {}
    for folder, name in (
        ("execution", "portfolio_account.py"),
        ("execution", "stateful_ledger.py"),
        ("v2", "evaluate.py"),
    ):
        saved = root / f"executed_{folder}_{name}"
        assert sha256_file(saved) == sha256_file(
            PROJECT / "research/src/brazil_rv" / folder / name
        )
        runtime[f"{folder}/{name}"] = binding(saved)
    cases = bound_json(audit["completed"])
    cielo = [c for c in cases if c["event"] == "CIEL"]
    for case in cielo:
        folder = root / case["book"]
        with (
            np.load(folder / "account.npz") as a,
            np.load(
                root / f"CIEL_primary_{case['sign']}_{case['capital']}" / "account.npz"
            ) as b,
        ):
            for field in a.files:
                np.testing.assert_array_equal(a[field], b[field])
            if case["sign"] < 0:
                assert not np.any(a["signed_shares"][:, case["source"]])
                assert not np.any(a["targets"][:, case["source"]] < 0)
        rows = bound_json(binding(folder / "payments.json"))
        assert all(
            not row["source_loans"] and row["loan_cash_settlement_payment"] == 0
            for row in rows
        )
    shutil.copyfile(__file__, root / "runtime_qualification_executed.py")
    result = dict(
        status="qualified",
        prior_delayed_term_contracts=list(checked.values()),
        prior_scope="These prior term variants never invoke the corrected provisioned-auction same-day release branch; completed prior books are reused without replay. This is term-scope qualification, not another source/store/book audit.",
        executed_runtime=runtime,
        cielo_books=len(cielo),
        cielo_long_held=9,
        cielo_short_unheld=9,
        cielo_all_scenario_arrays_exact=True,
        cielo_scope="No Cielo loan exposure in any of these18 paths. Long shareholder holdings remain source5.89; the negative preference books never allocated Cielo. +/-cent terms are available but their actual-held-loan adaptive price sensitivity remains unqualified and must use an actually exposed model/engineering book later. Do not report zero as a bound on held Cielo loans.",
    )
    output = root / "runtime_qualification.json"
    write_json_atomic(output, result)
    run["payment_bounds_runtime_qualification"] = binding(output)
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("prior_delayed_term_contracts", "executed_runtime")
            }
        )
    )


if __name__ == "__main__":
    main()
