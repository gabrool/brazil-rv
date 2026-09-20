"""Bind payment engineering acceptance, including unexposed Cielo limits."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    audit = bound_json(run["payment_bounds_audit"])
    q = bound_json(run["payment_bounds_qualification"])
    runtime = bound_json(run["payment_bounds_runtime_qualification"])
    root = Path(audit["audit_root"])
    accepted = {k: v for k, v in q.items() if k not in ("checks", "seconds")}
    accepted.update(
        status="fraction_sweep_precision_engineering_accepted_Cielo_held_loan_bound_and_final_A_pending",
        qualification=run["payment_bounds_qualification"],
        runtime_qualification=run["payment_bounds_runtime_qualification"],
        terms={p.stem: binding(p) for p in sorted(root.glob("terms_*.json"))},
        runtime=runtime["executed_runtime"],
        validation="Six distinct new tests pass, including both signed same-day payments, continuous/provisioned fractions, finite-difference gradients and independent SAM/TBPTT copies. Initial four-case batch overlaps final six. Ruff passes.",
        attempts="Initial newly-known/paid fraction fixture exposed an already-due restricted-proceeds queue in BOTH accounts; fixed only that branch before historical books, V38. Historical harness stopped after saving46 fully computed books because first Cielo negative-focus book had no position. All46 were reused; only remaining8 frozen books ran after qualifying the guard. No historical numerical/account failure or repeated book. Saved runtime qualifier initially demanded all Cielo targets zero; a later positive intention after sign reversal was unfilled without a quote. Narrow no-negative-target/no-position qualification passed; saved NAV proof/books reused.",
        cielo_disposition="Nine long books hold source shares, all18 have no Cielo loan. Nine negative-focus books have no negative targets or actual Cielo positions; later positive intentions can remain unfilled. All Cielo scenario account fields are exact. The cent loan bound is UNQUALIFIED for actually held loans; this zero is not evidence of immaterial invoice precision. No preferences, rates, borrowing availability or source quotes were changed to manufacture exposure.",
        costs="Old bundled4bp bridge has no additional B3 spot. Not final corporate pricing or model alpha. All synthetic/fixed-risk OLD PolicyData coordinates preserved by shallow copy.",
        prior_scope="Sixteen distinct previous term contracts never enter the corrected provisioned-fraction same-day payment branch. No previous V31-V37 book, source audit or accepted store was repeated.",
        timings=dict(
            summed_case_seconds=audit["summed_case_seconds"],
            final_resumption_seconds=audit["final_invocation_seconds"],
            initial_invocation_wall_seconds=None,
            saved_qualification_seconds=q["seconds"],
        ),
        remaining="Actual-exposure Cielo cent bound; invoice cents, security-day grouping and older R10 final-partial minimum payment allocation; dated B3 spot/custody separation and other negotiated-account components; three older clearing dates; held-source and succession data admission; final A then C/D. ALSC physical credit/exact net receipts and ENAT auction remain unknown. No completed program or corrected model profit.",
    )
    path = PROJECT / "docs/v2_payment_bounds_acceptance.json"
    write_json_atomic(path, accepted)
    run["payment_bounds_acceptance"] = binding(path)
    write_json_atomic(pointer, run)
    table = [
        "| Event / variant minus primary | Side | Capital | Final bp | Maximum path bp |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for c in q["contrasts"]:
        if c["event"] != "CIEL":
            table.append(
                f"| {c['event']} / {c['scenario']} | {'long' if c['sign'] > 0 else 'short'} | {c['capital']:,} | {c['final_bps']:.9f} | {c['max_path_bps']:.9f} |"
            )
    report = """# Fraction payment sweeps and cash precision

2026-09-20. V38 qualifies bounded payment engineering; final Stage A and C/D remain open. Resolve `payment_bounds_audit`, `payment_bounds_qualification`, `payment_bounds_runtime_qualification`, `payment_bounds_acceptance` and `payment_bounds_recovery` from the economic run pointer. Canonical corporate replay, accepted stores and old fits are unchanged.

## Contract and correction

Use the existing signed fraction and loan ledgers. BRML's source-known January26 2023 auction result can pay at the February2 deadline primarily or January26 as the separate earliest-known settlement hypothesis. Dommo's March31 known result pays April6 primarily or March31 as the corresponding hypothesis. Neither alternative is an observed client receipt. Source result clocks, January11 physical custody and January20 BRML/January17 Dommo ordinary cash legs remain unchanged. No extra CDI is appended to already indexed BRML cash.

Separately compare Dommo's approximate printed31.94031/share with565024/17690; the quotient is not an observed invoice precision. Original tiny loans delivering zero whole shares stop rent at delivery primarily; the separate bound continues original-principal rent through known April6 payment. Every original loan's own fraction is retained; no net-holding floor or duplicate proceeds release. Cielo's sourced continuous loan payment5.842895570784521 has separately frozen +/-R.01 per-share terms, while the shareholder5.89 and all dates remain unchanged.

The initial same-day fixture found a real accounting classification defect: both accounts appended a restricted-to-free transfer already due that day when a provisioned fraction was first recognized and paid. The value-date queue left that release restricted through the close. V38 only creates that queue for a later payment date. Same-day principal payment and proceeds release now both settle at the close; prior-close funding and earlier intentions stay frozen. Existing later-date queues are unchanged. At the100%-CDI primary this classification need not alter aggregate interest, but it matters to the next funding classification and different proceeds-remuneration hypotheses. Sixteen distinct prior term contracts do not enter the changed branch; all prior books remain reusable without rerunning them.

## Frozen experiments and exposure limits

Fifty-four actual constrained-allocation books use all933 names and frozen synthetic preferences/fixed risks, original calibration, corrected CDI, qualified lending and exact prior references. Accounting inputs shallow-copy OLD PolicyData. The old bundled4bp execution bridge has no additional B3 spot component; final corporate costs remain separate. Source preferences are +/-4 before effect and opposite afterward, successors opposite from effect, all other scores sinusoidal. No neural scoring, model alpha or new fit is involved.

BRML has12 books of27 sessions (December29 2022–February6 2023), Dommo24 of71 (December29–April12), Cielo18 of30 (August22–October2 2024). Each uses both focus signs and R10m/R1m/R5m. All BRML/Dommo focus positions were actually held. Nine Cielo long books hold shareholder inventory, but **none of the18 Cielo books has a Cielo loan**. All nine negative-focus cases have no negative Cielo target or filled position. A later positive intention after the frozen sign reversal can be unfilled without a source quote. The earlier blanket all-targets-zero assertion was too broad; the qualified statement is no short exposure. All Cielo scenario arrays are exact, but this does not qualify the cent bound for held Cielo loans. Apply it later to an actually exposed book; no cost, preference, source availability or quote was changed to force exposure.

The original harness stopped after46 fully computed/saved books when the first unheld Cielo short failed its exposure assertion. Resumption verified the frozen plan and runtime, reused all46, and ran only the remaining eight original cases. This was a scope assertion failure, not a numerical/accounting failure or a failed book. Exact initial/executed/resumed recipes and outputs are retained. The saved runtime qualifier's zero-target assertion was narrowed after inspecting the positive unfilled intention; no book or NAV qualification was repeated.

## Independent evidence and uncertainty

"""
    report += f"Independent cash, settled/restricted/unsettled balances, receivables/payables, marked stock/hedge and loan liabilities reconcile {q['sessions']:,} saved daily NAVs within R${q['independent_nav_identity_max_brl']}; {q['account_array_cells']:,} account cells are retained. Prior-close funding checks every saved day, maximum R${q['independent_prior_funding_max_brl']}. Decimal actual-fill/original-cohort fraction arithmetic differs by at most {q['decimal_fraction_max_error']} shares; signed ordinary/fraction cash checks differ by at most R${q['decimal_cash_max_error']}. No post-effect source fill, pre-recognition fraction cash, duplicate release or unpaid claim after its selected payment remains. Effect/recognition prefixes and intentions before the first differing realization are exact. Continuous BRML short and Dommo long rent-only variants are exact across all account fields.\n\n"
    report += f"All54 identical-intention account comparisons agree within R${q['identical_intention_max_brl']}. Independently adaptive maximum total-path differences in bp at R10m/R1m/R5m are {q['independent_adaptive_max_path_bps']['10000000']}/{q['independent_adaptive_max_path_bps']['1000000']}/{q['independent_adaptive_max_path_bps']['5000000']}; maximum target distance {q['adaptive_target_max']}. These are not bitwise parity. Preserve larger prior V32–V37 fixed-fee uncertainty too. The precision contrasts below are smaller than the measured adaptive discrepancy; do not interpret their sign as a resolved economic improvement. Tables are total synthetic-path NAV differences in bp of initial capital, never daily alpha or model profitability.\n\n"
    report += "\n".join(table) + "\n\n"
    report += "The Dommo short books contain one original zero-whole loan at each capital size. Independent compounded principal/rate arithmetic gives additional rent R.02864014215948/.00286401421594/.01432007107974 at R10m/R1m/R5m, matching saved source-attributed charges. This small tested-cohort result does not bound a portfolio with more tiny loans. No historical R10 minimum is present in these2023 contracts; that separate older bound remains open.\n\n"
    report += f"Six distinct new tests pass, including four signed/continuous/provisioned cases and two gradient/copy cases; the earlier four-test batch overlaps these six. They check same-day and delayed payments, no earlier intention or interest change, signed Decimal cash, restricted release, finite-difference derivatives and independent SAM/TBPTT state. Ruff passes. Sum of54 complete case runtimes {audit['summed_case_seconds']:.6f}s; the final eight-case resumption was {audit['final_invocation_seconds']:.6f}s and saved qualification {q['seconds']:.6f}s. Initial invocation total wall time was not separately recorded. These are CPU audit timings, not fit ETAs.\n\n"
    report += "## Remaining work\n\n" + accepted["remaining"] + "\n"
    (PROJECT / "docs/v2_PAYMENT_BOUNDS.md").write_text(report, encoding="utf8")
    durable = "The [payment checkpoint](docs/v2_PAYMENT_BOUNDS.md) qualifies V38 same-day newly-known fraction proceeds settlement in both accounts and54 frozen full933 adaptive books. BRML/Dommo sweep/precision/tiny-loan variants have held exposure; Cielo cent variants have no held loans and remain unqualified for that exposure. Independent2568NAVs, prior-close funding and Decimal cash/cohort checks pass. All earlier books/terms and accepted stores remain sealed; no model profitability. Invoice/security-day/old-minimum costs, actual-exposure Cielo and final A/C/D remain open."
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
            "\n\n## 2026-09-20 — Payment sweep and precision qualification\n\n"
            + durable
            + f"\n\n54books/2568dailyNAVs/{q['account_array_cells']}accountcells; independentNAVmaxR{q['independent_nav_identity_max_brl']}, fundingR{q['independent_prior_funding_max_brl']}, sixnewtests/Ruff. Initial same-day release queue defect fixed before books;46saved books reused after Cielo exposure assertion, only remaining8 ran. Runtime target assertion narrowed to no negative targets; later positive unfilled intention retained. No oldbook/source/store rerun. Adaptive discrepancy~.00002bp exceeds the precision contrasts; earlier uncertainty retained. Sumcase89.283086s/resumption8.831308s/qualification.720953s, initialwallunknown. Canonicalcorporate replay unchanged.\n"
        )
    print(json.dumps(binding(path)))


if __name__ == "__main__":
    main()
