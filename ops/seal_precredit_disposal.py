"""Seal the prearranged-sale engineering checkpoint without economic promotion."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    q = bound_json(run["precredit_disposal_qualification"])
    audit = bound_json(run["precredit_disposal_audit"])
    root = Path(audit["audit_root"])
    accepted = {k: v for k, v in q.items() if k not in ("entitlements", "seconds")}
    accepted.update(
        status="precredit_owned_disposal_engineering_accepted_final_A_pending",
        qualification=run["precredit_disposal_qualification"],
        terms={p.stem: binding(p) for p in sorted(root.glob("terms_*.json"))},
        validation="7 new tests and17 existing share-custody/undated-fraction cases pass; three focused existing bonus custody/disposal cases pass. Ruff passes. No prior engineering books repeated.",
        failed_attempts="Initial Dommo selector used incorrect ISIN suffix before any book/plan output; canonical terms resolved BRDMMOACNOR0. Exact initial recipe and sources retained, qualified invocation completes all54books without retry.",
        timings_seconds=dict(
            engineering=audit["seconds"], saved_qualification=q["seconds"]
        ),
        remaining="Final corporate costs; allocation/timing/sweep/invoice/security-day grouping/old-minimum/older-clearing bounds; other held source admission and separately attributed succession data effects. ALSC physical credit/exact net fees/payment, ENAT fraction auction remain unknown. Actual-model A acceptance/C/D unstarted.",
    )
    path = PROJECT / "docs/v2_precredit_disposal_acceptance.json"
    write_json_atomic(path, accepted)
    run["precredit_disposal_acceptance"] = binding(path)
    write_json_atomic(pointer, run)
    table = [
        "| Event | Sale permission | Capital | Final path bp | Maximum absolute path bp |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in q["contrasts"]:
        if row["sign"] > 0:
            table.append(
                f"| {row['event']} | {row['scenario']} | R${row['capital']:,} | {row['final_bps']:.9f} | {row['max_absolute_path_bps']:.9f} |"
            )
    report = """# Prearranged disposal before physical custody

2026-09-20. V36 is bounded engineering acceptance. Resolve `precredit_disposal_audit`, `precredit_disposal_qualification`, `precredit_disposal_acceptance` and `precredit_disposal_recovery` from the economic program pointer. Canonical `corporate_replay`, both accepted Stage B stores and all old fits remain unchanged. No corrected model profitability, neural forward or new GPU fit is claimed.

## What changed

The prior primary held corporate entitlements until custody. Both accounts now accept a separate, explicit `disposal_session`/`disposal_date` for prearranged sales of positive, unencumbered entitlements. This is an analyst execution hypothesis, not an obtained custodian agreement or newly recovered legal permission. The scheduled dates are January9/January10 2023 for BRML and Dommo, and December26/December27 2023 for Copel. Sourced credit remains January11 and December28 respectively.

On the selected disposal date, a positive source position with no outstanding source loan can become successor economic inventory. Whole shares remain backed by an incoming custody receipt on the original credit date. All original source purchases must settle by that date. This uses the existing regular-way owned-purchase mechanism; each sale must settle no earlier than its own incoming receipt. Failed or partial fills leave unsold shares and their receipts. A sale consumes its earmarked receipt under the existing scheduled-settlement convention; this does not claim to simulate an actual failed custodian delivery.

Fractional shareholder entitlements remain on the locked source claim until separately known auction/payment terms. Cash legs retain their original payment dates. Source shorts and encumbered source positions remain on the original credit path; this bound does not change source-loan conversion, principal, rate or fees. Existing successor shorts may economically offset the early owned entitlement, but actual loan return and proceeds release wait for credit. Existing successor purchases can cover source shorts at credit under the earlier custody contract. No loan alias, new locate, source quote, early spendable sale proceeds or fictitious physical delivery is introduced.

`ShareCustody` now rejects a sale settling before its incoming owned receipt. Both independent and differentiable accounts pass the explicit credit date to custody when economic ownership is advanced. State-copy behavior reuses the independent SAM/TBPTT custody/settlement copies. Optional disposal dates survive actual term loading and rebasing; unknown physical credit cannot support this hypothesis. The default remains custody-first. No model feature, label or membership changes.

## Frozen engineering experiment

Fifty-four new14-session books use all933 names: three events, both starting focus sides, R10m/R1m/R5m, and custody/effect/effect-plus-one permission. Each window has six pre-effect and eight effect/following sessions: December29 2022–January18 2023, or December15 2023–January8 2024. Preferences are sinusoidal with source focus +/-4 before effect, reversing thereafter; successor preferences reverse the original source sign from effect. Fixed beta/idio/market engineering risks and the actual constrained allocator are preserved. This is a deliberately frozen pressure test, not model scores or favorable outcome selection.

Accounting inputs shallow-copy frozen OLD PolicyData and use corrected CDI, qualified lending and exact prior references. The old bundled4bp execution bridge receives no additional B3 spot charge. Final corporate pricing and actual-model economic admission remain open. The short-focus paths are exact across permissions because advancing a short entitlement was not authorized by this hypothesis; independent implementations still have their separately measured numerical path differences.

## Saved-book qualification

"""
    report += f"All54 identical-intention account comparisons agree within R${q['identical_intention_max_brl']}. Independent cash/proceeds/unsettled/receivables/payables/marked inventory/hedge/loan-liability arithmetic reconciles {q['daily_navs']} daily NAVs within R${q['independent_nav_identity_max_brl']}. The saved account arrays contain {q['account_array_cells']} cells. Independent Decimal entitlement plus actual signed-fill recurrence verifies whole share arrivals and subsequent quantities, maximum discrepancy {q['decimal_quantity_max_error']} shares. Original fractions persist and no post-effect source fills occur. Early successor sales' T+2 dates are no earlier than original credit.\n\n"
    report += f"Independent adaptive maximum path differences at R10m/R1m/R5m are {q['independent_adaptive_max_path_bps']['10000000']}/{q['independent_adaptive_max_path_bps']['1000000']}/{q['independent_adaptive_max_path_bps']['5000000']}bp, maximum target distance {q['adaptive_target_max']}. These are neither daily alpha nor bitwise adaptive equality. Preserve larger earlier V32–V35 uncertainties for model contrasts. Scenario NAV prefixes are exact before their first differing realization; all effect-day targets are exact. Every saved account-array field is exact across short-side permission variants.\n\n"
    report += (
        "The table shows long-focus permission minus custody-first total-path NAV in basis points of initial capital. Every short-focus contrast is zero. BRML's very small contrast reflects economic offset against existing successor shorts, whose loan returns and restricted proceeds still wait for custody. It does not prove all pre-custody disposal choices are immaterial.\n\n"
        + "\n".join(table)
        + "\n\n"
    )
    report += f"Seven new focused cases cover single/basket and long/short paths, missing/partial fills, receipt-date rejection, unknown credit, delayed loan return/proceeds and gradient/independent copies. Seventeen existing custody/unknown-fraction cases and three bonus-specific cases pass; these are tests, not reruns of older books. The initial Dommo selector had the wrong ISIN suffix and failed before any book; BRDMMOACNOR0 was then read from canonical terms. Initial/executed recipes are retained. No completed new book failed or was repeated. Engineering {audit['seconds']:.6f}s and saved qualification {q['seconds']:.6f}s are measured CPU audit timings, not GPU-fit estimates.\n\n"
    report += "## Remaining work\n\nThis completes the frozen owned-disposal engineering bound. Apply the explicit alternatives in later actual-model economic contrasts. Other allocation/timing/sweep/fraction/invoice/security-day/minimum/clearing and corporate-cost bounds remain, together with unresolved held-event sources and separate new succession data attribution. ALSC physical credit, exact net fraction fees/payment and ENAT fraction auction are unknown. Prior originals and completed source/store/engineering work must not be repeated. Final A acceptance and registered C/D are pending.\n"
    (PROJECT / "docs/v2_PRECREDIT_DISPOSAL.md").write_text(report, encoding="utf8")
    context = PROJECT / "PROJECT_CONTEXT.md"
    text = context.read_text(encoding="utf8")
    anchor = "Last verified: 2026-09-20.\n"
    addition = "\nThe [pre-custody disposal checkpoint](docs/v2_PRECREDIT_DISPOSAL.md) qualifies V36's explicit prearranged owned-sale hypothesis. Positive source holdings without outstanding source loans may enter successor economic inventory on a selected pre-credit date; incoming custody and sale settlement remain separately dated. Source loans, shareholder fractions, cash legs and deferred proceeds retain prior contracts. Fifty-four14-session/all933 adaptive synthetic books and independent756dailyNAV/Decimalfill checks pass. Custody-first remains primary, no model profitability or data/fit change; use the separate precredit_disposal pointers for later actual-model bounds. Do not repeat these books or earlier V31–V35/source/store audits. Remaining costs, event/rounding/clearing bounds and C/D are still active.\n"
    assert addition not in text
    context.write_text(text.replace(anchor, anchor + addition, 1), encoding="utf8")
    progress = PROJECT / "docs/v2_economic_data_scaling_progress.md"
    with progress.open("a", encoding="utf8") as f:
        f.write(
            f"\n\n## 2026-09-20 — Prearranged owned disposal before custody\n\nV36 adds explicit economic sale permission while preserving incoming custody, physical loan return and proceeds release.54x14/all933 adaptive synthetic books, custody/effect/next for BRML/DMMO/Copel and both sides/R10m/R1m/R5m, completed without book retry. Independent756NAVs maxR{q['independent_nav_identity_max_brl']}; {q['account_array_cells']}cells; Decimalshareflow max{q['decimal_quantity_max_error']}; identical-intention maxR{q['identical_intention_max_brl']}. Short permission variants exact; adaptive implementations retain measured uncertainty. R10m long effect-vs-custody finalBRML-.000144115bp/DMMO-5.846263298bp/Copel-.330398353bp, total synthetic path not alpha.7new+17existing+3bonus tests pass/Ruff. Initial wrong Dommo ISIN suffix stopped beforebooks, retained/resolved from canonical terms. CPUaudit{audit['seconds']:.6f}s/qualification{q['seconds']:.6f}s. No source retrieval/store alteration/neural scoring/fit; canonicalcorporate_replay remains unchanged. See v2_PRECREDIT_DISPOSAL.md and acceptance/qualification pointers; remaining costs/eventbounds/A/C/D incomplete.\n"
        )
    print(json.dumps(binding(path)))


if __name__ == "__main__":
    main()
