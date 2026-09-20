"""Bind the invoice hypothesis checkpoint and its exact runtime qualification."""

from collections import Counter
import difflib
import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["loan_invoice_audit"])
    q = bound_json(run["loan_invoice_qualification"])
    root = Path(audit["audit_root"])
    runtime = {}
    for folder in ("execution", "v2"):
        for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            saved = root / f"executed_{folder}_{path.name}"
            if path.name == "evaluate.py":
                current, executed = (
                    path.read_text(encoding="utf8"),
                    saved.read_text(encoding="utf8"),
                )
                qualified = current.replace(
                    'f"residual payment allocation {config.loan_minimum_allocation} is an explicit research assumption"',
                    '"residual payment on final return is an explicit research assumption"',
                ).replace(
                    '        "loan_invoice_convention": config.loan_invoice_convention,\n',
                    "",
                )
                assert qualified == executed
                patch = root / "evaluation_metadata.patch"
                patch.write_text(
                    "".join(
                        difflib.unified_diff(
                            executed.splitlines(True), current.splitlines(True)
                        )
                    ),
                    encoding="utf8",
                )
            else:
                assert sha256_file(path) == sha256_file(saved)
            runtime[f"{folder}/{path.name}"] = dict(
                current=binding(path), executed=binding(saved)
            )
    attempts = dict(
        initial_tests="8passed/3failed: cash roundoff1.82e-12 atR10000 against an absolute1e-12 helper; T+3 credit probe incorrectly day4, correctly day5. Narrow fixtures corrected: comparison atR1000, probe day5 and mutate one credit cell. Production unchanged. Initial test and implementation saved. Qualified11 passed, added threshold-crossing case gives12 plus9loan/6payment existing=27pass; two additional delivery/renewal credit cases produce14distinct new tests. Overlapping batches not additive.",
        initial_qualification="Initial Decimal recomposition of independently printed accrued fees and minimum residual straddled R10 by ulps, causing floor/ceil cent mismatches. Verify component arithmetic then round actual Float64 invoice coordinate in independent Decimal.72 enumerated boundary cases retained; all coordinate-based invoice amounts exact. No production change or book rerun.",
        books="All36 books completed first invocation, no numerical failure or repeated book.",
        metadata="Post-book evaluation report now states chosen invoice/minimum hypothesis instead of always describing final-return allocation. Exactly two metadata text changes verified; all other executed production bytes match current.",
    )
    write_json_atomic(root / "attempts.json", attempts)
    boundary_counts = Counter(row["book"] for row in q["precision_boundaries"])
    acceptance = {
        k: v for k, v in q.items() if k not in ("checks", "runtime", "seconds")
    }
    acceptance.update(
        status="invoice_and_minimum_engineering_accepted_final_A_C_D_pending",
        qualification=run["loan_invoice_qualification"],
        runtime=runtime,
        metadata_patch=binding(patch),
        attempts=attempts,
        precision_contract="Independent components match within1.1e-14BRL; Decimal rounding of saved Float64 invoice coordinates matches all9503 groups exactly.72 recomposition/cent straddles are operand-printing precision boundaries around an actualR10 coordinate, not observed invoice uncertainty or evidence for9.99/10.01 exact fees. All are enumerated; actual lower/upper scenario paths retain their frozen coordinate contract.",
        boundary_counts=dict(boundary_counts),
        tests="14distinctnewtests,9existingloancontracttests,6existingfractionpaymenttests pass; overlapping batches not summed. Ruff passes.",
        one_factor="security_day_nearest minus contract_nearest isolates invoice grouping; other variants compare primary. Original-security/day invoice grouping NEVER merges registrations, rates, principal, renewals or minimums. Minimum pro-rata advances only existing residual minimum and offsets future fees; no duplicate toll.",
        uncertainty="Independentadaptive maxpath .010282502409070731/.00000816549058072269/.000022326227277517318bp atR10m/R1m/R5m. Preserve larger V32-V38 fixed-fee uncertainty. Many2019R10m path signs are below this bound. Total synthetic paths, not dailyalpha/modelprofit or an all-interior adaptive extremum proof.",
        timings=dict(
            engineering_seconds=audit["seconds"],
            summed_case_seconds=audit["summed_case_seconds"],
            saved_qualification_seconds=q["seconds"],
        ),
        remaining="Actual-held-Cielo cent bound; final historical B3 spot/custody/rent-intermediation/brokerage/shortfall/cash/debit separation and negotiated sensitivities; three older clearing ambiguities; held source/succession data admission; then registeredC/D. ALSC credit/exactnet/actualreceipt and ENATfractionauction remainunknown. Both acceptedstores/oldfits and all prior books are sealed; no newfit/modelprofit.",
    )
    path = PROJECT / "docs/v2_loan_invoice_acceptance.json"
    write_json_atomic(path, acceptance)
    run["loan_invoice_acceptance"] = binding(path)
    write_json_atomic(pointer, run)
    table = [
        "| Start / variant minus base | Capital | Final bp | Max path bp |",
        "| --- | ---: | ---: | ---: |",
    ]
    for c in q["contrasts"]:
        table.append(
            f"| {c['start']} / {c['scenario']} minus {c['base']} | {c['capital']:,} | {c['final_bps']:.9f} | {c['max_path_bps']:.9f} |"
        )
    report = """# Loan invoice cents and historical minimum allocation

2026-09-20. V39 qualifies explicitly bounded accounting engineering; final A/C/D remain open. Resolve loan_invoice_audit, loan_invoice_qualification, loan_invoice_acceptance and loan_invoice_recovery from the economic run pointer. Both accepted Stage B stores, original fits, canonical corporate replay and all prior event books remain immutable.

## Contract

The primary remains unrounded loan payments and residual R10 historical minimum on final return. New one-factor hypotheses round rent and total B3 loan fees separately on their actual physical payment date, per original contract: nearest half-up, down or up to cents. A separate nearest-original-security/day scenario changes invoice grouping only, compared with contract-nearest. It never combines loan registrations, changes rates/principal, reduces the number of historical minimum obligations, groups across payment dates or merges rent with B3 fees. These are analyst invoice hypotheses, not sourced broker/B3 instructions. Original-security grouping remains explicit through corporate successor conversion.

Both accounts recognize the paid-minus-unrounded difference as expense at payment, separately exposed in loan_invoice_adjustment (rent/B3 columns). No prior intention or prior-close interest changes. Old loan_charges retain accrued rent/fees; invoice adjustments must be added separately when reconciling them to paid expense. Exact floor/ceil arithmetic retains its actual local derivative, zero away from cent boundaries; no straight-through gradient approximation. The metadata states each selected hypothesis.

Another one-factor hypothesis allocates the currently unpaid minimum residual pro rata to original principal actually returned. Early amounts become minimum_credit, offsetting subsequent B3 invoices. The credit reduces remaining loan liabilities and survives partials, corporate leg allocation and independent SAM/TBPTT copies; final return or renewal settles the old root exactly once. No refund or second minimum is invented. Same-intention closed-form tests preserve total max(accrued B3 fees,R10), including a path crossing the minimum threshold. Adaptive scenario paths can trade differently and therefore create different later contracts; they are not a pure interest-only arithmetic difference.

## Frozen books and checks

36 new64-session books start February1 2019 (through May7) and February1 2024 (through May6), six variants and R10m/R1m/R5m. All933 names and original histories remain. The actual allocator receives sin(axis*.31+localday*.07+head*.2), fixed beta1/idio.0004/market.0001 and original calibration. OLD PolicyData is shallow-copied for corrected CDI/qualified lending/strict-prior references. No neural forward or model score is used. The old bundled4bp bridge has no additional B3 spot and is not final corporate pricing. All36 completed on the first invocation; none was repeated.

"""
    report += f"Independent saved cash/proceeds/unsettled/claims/stock/hedge/loan arithmetic checks {q['sessions']} NAVs, max R{q['errors']['nav']}; {q['account_array_cells']} account cells. Prior funding max R{q['errors']['funding']}; liability/expense rollforward max R{max(q['errors']['liability'], q['errors']['expense_rollforward'])}. {q['root_payments']} root-payment records produce {q['invoice_groups']} invoice groups; independent Decimal minimum allocation max R{q['errors']['decimal_minimum']}, grouping max R{q['errors']['decimal_grouping']}, actual-coordinate invoice error zero. Before the first differing payment/credit, all NAV prefixes and current intentions are exact.\n\n"
    report += "The2019 pro-rata variants have81/235/97 actual partial minimum allocations and maximum fee credits R34.0620348024/R63.0762494277/R38.9071349453 atR10m/R1m/R5m. The2024 books have no old minimum and their pro-rata/primary paths are exact. Security/day grouping actually merges roots in every tested capital/window. Final-boundary unsettled loans/charges remain accounted for.\n\n"
    report += acceptance["precision_contract"] + "\n\n"
    report += f"Identical-intention independent accounts agree within R{q['identical_intention_max_brl']}. {acceptance['uncertainty']} Maximum target distance {q['adaptive_target_max']}. AtR10m in2019, nearest rent/B3 rounding totals R.0182052/.0751801 while the path contrast is+.00995962bp; this includes adaptive trading/minimum effects and is below measured implementation uncertainty. Do not call it a beneficial rounding effect. In2024, down/up total final contrasts are+.002207685/-.002196860bp. These are bounded synthetic paths, not corrected model economics.\n\n"
    report += "\n".join(table) + "\n\n## Execution and limitations\n\n"
    report += "\n\n".join(attempts.values()) + "\n\n" + acceptance["tests"]
    report += (
        f" Engineering {audit['seconds']:.6f}s (sumcases {audit['summed_case_seconds']:.6f}s); saved qualification {q['seconds']:.6f}s. These are CPU audit runtimes, not fit ETAs.\n\n"
        + acceptance["remaining"]
        + "\n"
    )
    (PROJECT / "docs/v2_LOAN_INVOICES.md").write_text(report, encoding="utf8")
    durable = "The [loan invoice checkpoint](docs/v2_LOAN_INVOICES.md) qualifies V39 cent/payment grouping and partial old-minimum credit hypotheses in BOTH accounts,36full933 adaptive books/2304NAVs. Independent9503invoice groups and minimum-credit arithmetic pass;14newtests, prior books/stores/oldfits preserved.72 Decimal operand-recomposition cent boundaries are enumerated; actual Float64 invoice coordinates round exactly. Larger fixed-fee adaptive uncertainty remains; no model profit. Actual-held-Cielo, final historical corporate costs/source-data admission and C/D remain open."
    context = PROJECT / "PROJECT_CONTEXT.md"
    text = context.read_text(encoding="utf8")
    assert durable not in text
    context.write_text(
        text.replace(
            "Last verified: 2026-09-20.\n",
            "Last verified: 2026-09-20.\n\n" + durable + "\n",
            1,
        ),
        encoding="utf8",
    )
    with (PROJECT / "docs/v2_economic_data_scaling_progress.md").open(
        "a", encoding="utf8"
    ) as f:
        f.write(
            "\n\n## 2026-09-20 — Loan invoice and minimum allocation checkpoint\n\n"
            + durable
            + "\n\n"
            + acceptance["uncertainty"]
            + "\n\n"
            + acceptance["tests"]
            + f" Engineering{audit['seconds']:.6f}s/qualification{q['seconds']:.6f}s; all36 first-pass, no oldbook/source/store rerun.\n"
        )
    print(
        json.dumps(
            dict(
                acceptance=run["loan_invoice_acceptance"],
                errors=q["errors"],
                boundaries=len(q["precision_boundaries"]),
            )
        )
    )


if __name__ == "__main__":
    main()
