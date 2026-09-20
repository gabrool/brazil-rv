"""Seal bounded Copel engineering; do not promote final corporate economics."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    q = bound_json(run["copel_loan_bounds_qualification"])
    audit = bound_json(run["copel_loan_bounds_audit"])
    root = Path(audit["audit_root"])
    runtime = {}
    for folder, filename in [
        ("execution", "share_distributions.py"),
        ("execution", "portfolio_account.py"),
        ("execution", "stateful_ledger.py"),
        ("execution", "portfolio_policy.py"),
        ("v2", "corporate_replay.py"),
        ("v2", "evaluate.py"),
    ]:
        current = PROJECT / "research/src/brazil_rv" / folder / filename
        executed = root / f"executed_{folder}_{filename}"
        assert sha256_file(current) == sha256_file(executed)
        runtime[f"{folder}/{filename}"] = binding(executed)
    accepted = {
        key: value for key, value in q.items() if key not in ("checks", "seconds")
    }
    accepted.update(
        status="copel_net_borrowed_timing_and_principal_engineering_accepted_final_A_pending",
        qualification=run["copel_loan_bounds_qualification"],
        terms={p.stem: binding(p) for p in sorted(root.glob("terms_*.json"))},
        executed_runtime=runtime,
        validation="11 distinct new cases pass in focused batches (initial9, added partial-cover and pending-owned-receipt cases);7 affected precredit cases pass. Ruff passes. Overlapping receipt retry not additive.",
        attempts="All30 historical books and saved qualification completed on first invocation. Extra pending-owned fixture initially requested40% of current NAV after rent, buying fewer units than original loan; fixed only test sizing to exact held units. Exact initial9-case and failed expanded test files retained. Read-only Polars pretty-print hit cp1252 after successful schema/sample reads; no research artifact changed.",
        timings_seconds=dict(
            engineering=audit["seconds"], saved_qualification=q["seconds"]
        ),
        source_loan_nonzero_day_checks=sum(
            float(row["actual_rent"]) != 0
            for case in q["checks"]
            for row in case["daily_source_loan_charges"]
        ),
        remaining="Cash/fraction receipt sweeps and precision/invoice/security-day grouping/old minimum; dated B3 spot/custody and other corporate components; three older clearing dates; held-source admission and separately attributed succession data implications; final A then C/D. Timing for flat/positive source positions with pending old loans remains outside this net-borrowed bound.",
    )
    path = PROJECT / "docs/v2_copel_loan_bounds_acceptance.json"
    write_json_atomic(path, accepted)
    run["copel_loan_bounds_acceptance"] = binding(path)
    write_json_atomic(pointer, run)
    table = [
        "| Short-focus variant minus primary | Capital | Final path bp | Maximum absolute path bp | Copel original-root rent difference R$ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in q["contrasts"]:
        if row["sign"] < 0:
            table.append(
                f"| {row['scenario']} | {row['capital']:,} | {row['final_bps']:.9f} | {row['max_path_bps']:.9f} | {row['source_rent_delta_brl']:.9f} |"
            )
    report = """# Copel original-principal and loan conversion bounds

2026-09-20. V37 is bounded engineering acceptance, with final Stage A and C/D still open. Resolve `copel_loan_bounds_audit`, `copel_loan_bounds_qualification`, `copel_loan_bounds_acceptance` and `copel_loan_bounds_recovery` from the economic run pointer. Canonical `corporate_replay`, both accepted stores and all old fits remain unchanged.

## Contract and implementation

Copel units BRCPLECDAM13 split economically on December26 2023 into one BRCPLEACNOR8 and four BRCPLEACNPB9. The recovered issuer receipt fixes positive shareholder custody on December28. The general B3 manual requires issuer factors for original loan-principal allocation; the event-specific K and loan-conversion instruction remain unrecovered. The earlier source receipts and prescribed36 cases are retained; none was retrieved or replayed again.

Use K=.2 ON/.8 PN primary and K0/K1 as separate endpoint sensitivities. The existing loan subledger already preserves total original principal/rate and allocates accrued rent/fees. K is unrelated to the1:4 quantities or the market-value allocation of inventory basis and proceeds. Tested endpoints do not prove an extremum over every interior K for an adaptive policy.

Both actual accounts now admit an optional `loan_conversion_session`/`loan_conversion_date`. A negative source claim uses this date to become successor loan obligations; positive shareholder inventory uses the original physical credit. The one-factor timing variants use December27/28/29, with December28 primary. A conversion retains original rate/reference principal, open date, fees and pending-return dates. It does not open a new loan or infer a locate, reference price, loan alias or shareholder delivery. Existing positive successor inventory can offset the converted debt only with its actual settled/dated owned receipt; physical return and restricted-proceeds release follow that receipt. A cover after conversion retains normal spot T+2.

This is explicitly a **net borrowed claim** timing hypothesis. Flat or positive source inventory with pending old loan returns keeps the prior convention; no new claim is made about that separate gross-offset case. Separate timing with a fraction auction is rejected because that requires additional event-specific terms. The current qualified Copel basket has no auction leg. Both economic accounts, claim-date readouts and sector exposure scheduling use the same sign-dependent clock. The leg's shareholder credit remains separately stored. Loader/slicing preserve both dates, and existing independent SAM/TBPTT copies preserve the immutable event and mutable account state.

## Frozen adaptive books

Thirty new30-session/all933 books cover December15 2023–January30 2024, six pre-effect plus24 effect/following sessions. Five separate variants (primary, K0, K1, early loan, late loan), both source sides and R10m/R1m/R5m use the actual constrained allocator. Synthetic source preferences are +/-4 before effect and opposite afterward; ON reverses at effect while PN keeps the original source sign until credit+2, then reverses. All other sinusoidal preferences, beta1/idio.0004/market.0001 risks and original calibration are frozen. This creates staggered disposal pressure without model scores or outcome-selected parameters.

Accounting inputs shallow-copy frozen OLD PolicyData, using corrected CDI, qualified lending and exact prior references. The old bundled4bp cost bridge has no additional B3 spot component; it is not final corporate pricing. Original Copel loan roots here use the recovered .0002 annual rate (0.02%), unlike the earlier prescribed4% rent example. No old book or source/store audit was repeated.

## Independent saved results

"""
    report += f"All30 identical-intention accounts agree within R${q['identical_intention_max_brl']}. Independent saved cash, restricted proceeds, unsettled money, claims, marked inventory/hedge and loan liability reconcile900 daily NAVs within R${q['independent_nav_identity_max_brl']}; {q['account_array_cells']:,} account cells are retained. Decimal signed entitlements plus actual fills reproduce both legs with maximum {q['decimal_share_max_error']} share discrepancy. Source post-effect fills are absent. All long-focus arrays are exact across variants; NAV prefixes before first differing realization and all effect-day targets are exact. This branch equality is distinct from parity of independently adaptive implementations.\n\n"
    report += f"For each converted original root, saved before/prepared states independently reconcile quantities1:4, principal K:(1-K), accrued rent/fees, original rate/opening/reference principal and unchanged fee-growth/minimum state. Maximum Decimal allocation discrepancy is R${q['decimal_cohort_max_error']}. Independent Decimal daily compounded rent and both B3 fee components match660 source/day checks (including zero controls; {accepted['source_loan_nonzero_day_checks']} nonzero rent checks), maximum errors R${q['decimal_rent_max_error']} and R${q['decimal_fee_max_error']}. These are repeated scenario checks, not distinct source observations. Post-2020 minimums here are zero; the older R$10 minimum bound remains separate.\n\n"
    report += f"Independent adaptive maximum total-path differences at R10m/R1m/R5m are {q['independent_adaptive_max_path_bps']['10000000']}/{q['independent_adaptive_max_path_bps']['1000000']}/{q['independent_adaptive_max_path_bps']['5000000']}bp, target distance {q['adaptive_target_max']}. Preserve larger V32–V36 measured fixed-fee uncertainty for future actual-model comparisons. These are neither daily alpha nor bitwise adaptive equality.\n\n"
    report += (
        "The table gives total synthetic-path NAV contrasts in bp of initial capital, with original Copel-root rent shown separately. Timing changes adaptive disposal/exposure, so its NAV difference must not be described as just one day's rent. Every long-focus contrast is zero. K's small effect at the recovered low rate does not establish small effects for different rates, timings or actual model exposures.\n\n"
        + "\n".join(table)
        + "\n\n"
    )
    report += f"Eleven distinct new tests and seven affected existing precredit tests passed in focused batches; Ruff passes. The added pending-owned fixture initially bought fewer units by applying40% to NAV already reduced by rent. Using the actual original units qualified its Decimal NAV/gradient oracle; production code and all historical books were unchanged. Initial/failed test bytes are retained. A read-only Polars pretty-print failed under Windows cp1252 after successful sample/schema reads and changed no data. All30 books and saved qualification completed on their first invocation. Engineering {audit['seconds']:.6f}s and saved qualification {q['seconds']:.6f}s are CPU audit timings, not fit estimates. Executed research runtime hashes match current implementation.\n\n"
    report += "## Remaining admission\n\nApply these explicit variants later to matched actual-model books. Cash/fraction payment sweeps, precision/invoice/security-day grouping/older minimum, dated B3 spot/custody and other corporate cost components, three older equity-clearing ambiguities and remaining held-event source/data admission are still open. ALSC physical credit/exact net fees/payment and ENAT auction remain unknown; their completed source/engineering records stay sealed. No corrected model profitability, new neural forward, GPU fit or final economic acceptance is claimed.\n"
    (PROJECT / "docs/v2_COPEL_LOAN_BOUNDS.md").write_text(report, encoding="utf8")
    context = PROJECT / "PROJECT_CONTEXT.md"
    text = context.read_text(encoding="utf8")
    addition = "\nThe [Copel loan checkpoint](docs/v2_COPEL_LOAN_BOUNDS.md) qualifies V37's separate net-borrowed conversion date and existing original-principal K0/.2/1 allocation in30 new30-session/all933 adaptive synthetic books. Positive shareholder custody stays December28; negative claims convert December27/28/29 under explicit hypotheses and owned successor custody still bounds physical returns/proceeds. Original loan terms/fees survive; flat/positive source pending returns remain outside this net-borrowed timing bound. Independent900NAV/Decimalcohort and660source-day rent/fee checks pass. Canonical corporate replay, accepted stores and old fits remain sealed; no model profitability/fit. Do not repeat these or earlier V31–V36 books. Remaining payment/rounding/cost/clearing/source/data admission precedes C/D.\n"
    assert addition not in text
    context.write_text(
        text.replace(
            "Last verified: 2026-09-20.\n", "Last verified: 2026-09-20.\n" + addition, 1
        ),
        encoding="utf8",
    )
    with (PROJECT / "docs/v2_economic_data_scaling_progress.md").open(
        "a", encoding="utf8"
    ) as f:
        f.write(
            f"\n\n## 2026-09-20 — Copel original principal and net-loan timing\n\nV37 separates net-borrowed conversion from sourced positive custody, preserving original loan rates/principal/fees and actual owned receipts.30x30/all933 adaptive synthetic books, K0/.2/1 and December27/28/29 one-factor dates, both signs/R10m/R1m/R5m, completed once.900NAV maxR{q['independent_nav_identity_max_brl']}; {q['account_array_cells']}cells;660Decimal source/day charges includingzero controls. R10m shortK0/K1 final-.000435061/+.001740257bp; early/late final-3.999501232/-3.949972862bp(max7.449071841/4.839939996), total synthetic path not alpha. Long arrays exact acrossvariants; independentadaptive max.000006878bp retains earlier uncertainty.11new+7affected tests/Ruff pass. Extra fixture sizing failure fixed only test after saving exact failedbytes; allbooks reused. Audit{audit['seconds']:.6f}s/qualification{q['seconds']:.6f}s. No retrieval/store/oldfit changes. Flat/positive source pendingloan timing, othercost/payment/rounding/clearing/source/data admission and A/C/D remain open.\n"
        )
    print(json.dumps(binding(path)))


if __name__ == "__main__":
    main()
