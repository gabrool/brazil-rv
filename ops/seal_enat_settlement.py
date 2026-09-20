"""Publish the bounded ENAT/fraction engineering and ALSC original-source result."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    q = bound_json(run["enat_settlement_qualification"])
    audit = bound_json(run["enat_settlement_audit"])
    sources = bound_json(q["sources"])
    acceptance = dict(
        status="ENAT_undated_fraction_engineering_and_ALSC_auction_sources_accepted_stage_A_incomplete",
        evidence={
            k: run[k]
            for k in (
                "enat_settlement_audit",
                "enat_settlement_qualification",
                "remaining_held_source_audit",
            )
        },
        terms=audit["terms"],
        books=q["books"],
        daily_navs=q["sessions"],
        account_array_cells=q["account_array_cells"],
        identical_intention_max_brl=q["identical_intention_max_brl"],
        independent_nav_identity_max_brl=q["independent_nav_identity_max_brl"],
        independent_adaptive_max_path_bps=q["independent_adaptive_max_path_bps"],
        adaptive_target_max=q["adaptive_target_max"],
        contrasts=q["contrasts"],
        new_original_pdfs=5,
        new_issuer_records=194,
        alsc_auction=sources["alsc_auction"],
        validation="7 new unknown-fraction/loader/gradient/copy cases,7 affected loan-fraction cases and16 target cases pass. Earlier five new cases repeated within the final seven are not additional tests. Ruff passes; no earlier engineering books or source censuses rerun.",
        limitations=q["limits"],
        failed_attempts="One ALSC2020 issuer-index selection shadowed a list with a boolean, before index/PDF output; exact failed bytes preserved. Five original PDFs and18 books completed without numerical/rerun failure. Poppler emitted display-font warnings; eight relevant pages are legible and visually qualified.",
        timings_seconds=dict(
            engineering=audit["seconds"], saved_qualification=q["seconds"]
        ),
        remaining="ALSC physical credit/exact net fraction cash; ENAT auction and event-specific loans; prior source-specific loan hypotheses, pre-custody disposal, corporate cost/rounding/grouping/minimum/calendar bounds, final StageA, C/D. New succession model-target/history propagation remains separate; no accepted store reassembly merely for source revision uncertainty.",
    )
    path = PROJECT / "docs/v2_enat_settlement_acceptance.json"
    write_json_atomic(path, acceptance)
    run["enat_settlement_terms"] = audit["terms"]
    run["enat_settlement_acceptance"] = binding(path)
    write_json_atomic(pointer, run)
    table = [
        "| Contrast | Side | Capital | Final path bp | Max absolute path bp |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in q["contrasts"]:
        table.append(
            f"| {row['contrast']} | {'Long' if row['sign'] > 0 else 'Short'} | R${row['capital']:,} | {row['final_bps']:.9f} | {row['max_absolute_path_bps']:.9f} |"
        )
    report = """# ENAT undated fraction claims and remaining held-source evidence

2026-09-20. Bounded engineering/source acceptance, not final economic admission or model profitability. Resolve `enat_settlement_audit`, `enat_settlement_qualification`, `remaining_held_source_audit`, `enat_settlement_terms` and `enat_settlement_recovery` through the run pointer. The canonical `corporate_replay` remains unchanged. Both accepted StageB stores and all old fits stay immutable.

## Original evidence

Four new CVM originals establish ENAT's share exchange into3R: tentative timetable1260409, final confirmation1264467, consummation1265227 and withdrawal result1263891. The legal close/register is July31 2024, trading/economic effect August1, physical share credit August5. Use the final ratio **.805012676** from July30, not the tentative rounded .805013. July30's21:02local receipt becomes July31 00:03UTC; the prior-decision source is causal. August1's9:22local confirmation is available12:23UTC. The successor is **BRRRRPACNOR5**, not a current BRAV ticker substituted backwards. ENAT is BRENATACNOR0. Own prior successor mark27.059999465942383 and independent strict-prior loan reference26.87 already exist; no opening-value bridge or alias is invented. Source last close21.639999389648438 retains original stored precision.

Fractions must be grouped and sold, with net proceeds later distributed. Their auction date, price and payment are unrecovered from the bounded2024 original index. The34 dissenting shares/R14.59 withdrawal are separate from the ordinary non-electing portfolio's share exchange; no elective cash is silently added. The notice's nonresident capital-gain withholding passage neither imposes that tax on domestic CNPJ nor grants a tax exemption. These accounts exclude entity income tax under the registered research metric.

The prior2019 ALSC index was insufficient for the later auction. The existing2020 issuer22357 receipts identify **732740**, received January30 2020 18:33local, usable21:34UTC/January31 decision. Its original page reports **2076 shares auctioned January15**, **R$54.26688776859 per share**, distributed **net of unspecified fees within seven business days**. February10 is the derived deadline under the seven following accepted sessions; it is not an observed payment. Do not backdate knowledge to January15 or call the printed auction proceeds exact net investor cash. Physical share credit remains unknown; earlier ALSC books stay wholly locked. This partially resolves the earlier source gap without inventing custody.

The search reuses the prior216 records and adds153 ENAT/3R2024 plus41 issuer223572020 records from existing RAD responses. Current archive labels do not establish historical legal issuers. Five new PDFs and eight relevant original pages are extracted and visually qualified. Poppler's optional-display-font warnings did not prevent legible numerical/date verification. Bounded source searches do not prove disclosure completeness, original internet publication times, or historical revision share.

## Account contract

V35 permits an explicitly undated fraction auction: all knowledge/price/payment fields remain null. Existing delivery in both accounts supplies whole long shares and leaves their signed fraction in the original non-tradable claim, marked from the successor. No zero cash, fabricated auction or loan reference is created. The existing training copy keeps independent claim lists; gradients through the unliquidated fractional position remain verified. Known-auction behavior is unchanged.

Loan conversion at shareholder credit, original principal/rate/fees and continuous loan quantity are explicit hypotheses. A separate per-original-contract whole-quantity variant retains a signed residual obligation, using independent Decimal pro-rata cohort arithmetic. Tiny zero-deliverable contracts retain the existing explicit stopped-rent convention; continued rent until an unknown payment is rejected. No event-specific B3 loan instruction or locate guarantee is claimed. Unknown timing also cannot authorize post-delivery lot-dependent gross unit labels. This new contract changes no accepted data arrays or old coordinates.

## Verification and effects

Eighteen30-session/all933-name books span July24–September3 2024, six pre-effect and24 effect/following sessions, both focus sides and R10m/R1m/R5m. Three cases are frozen before outcomes: unresolved source, sourced whole delivery with continuous loans, and provisioned loan fractions. All use actual adaptive allocation, synthetic fixed preferences/engineering risks, correctedCDI/qualified lending/exact prior references, and shallow-copied frozen OLD PolicyData. The old bundled4bp bridge has no additionalB3spot charge; final corporate pricing is pending. No model forecast, neural forward or GPU fit was used.

"""
    report += f"All18 identical-intention comparisons agree within R${q['identical_intention_max_brl']}. Independent cash/proceeds/unsettled/claims/marked inventory/loan-liability arithmetic reconciles{q['sessions']}daily NAVs, max R${q['independent_nav_identity_max_brl']}; {q['account_array_cells']} saved account cells. All ENAT positions are actually held, no post-effect source fills occur, whole entitlements wait for credit, and residuals remain locked. Decimal entitlement and per-original-loan fraction oracles pass. Scenario prefixes are exact through their first differing realization; loan-only long paths are exact.\n\n"
    report += f"Independent adaptive maximum total-path differences at R10m/R1m/R5m are {q['independent_adaptive_max_path_bps']['10000000']}/{q['independent_adaptive_max_path_bps']['1000000']}/{q['independent_adaptive_max_path_bps']['5000000']}bp; maximum target distance{q['adaptive_target_max']}. Preserve larger earlierV32–V34 uncertainties; this does not establish bitwise equality or daily alpha.\n\n"
    report += (
        "The table reports total-path NAV effects in basis points of initial capital. Source means primary minus unresolved; loan_fraction means provisioned minus continuous. These are synthetic engineering contrasts, not model profits.\n\n"
        + "\n".join(table)
        + "\n\n"
    )
    report += "At R10m the final long fraction is .3588203548785 successor shares/R9.1391541924; the provisioned-short bound retains1.792071018635 shares/R45.6440476141. The continuous short has no undelivered fraction by assumption. The source control retains unquoted predecessor inventory; its zero *undelivered-share* readout does not mean inventory disappeared.\n\n"
    report += f"Seven new cases cover signed unknown fractions, loader preservation, unit-endpoint exclusion and gradient/independent copy; seven existing loan-fraction and16 target cases pass. Overlapping focused repeats are not summed. One new2020 index attempt failed before outputs because a local boolean shadowed the row list; its bytes are retained and only the bounded index selection was retried. No ENAT account book failed or was repeated. Engineering wall{audit['seconds']:.6f}s, saved qualification{q['seconds']:.6f}s; PDF retrieval batches separately record3.220011/0.752919/1.069098s, excluding manual selection/rendering/failed index. These are not fitETAs.\n\n"
    report += "## Remaining admission\n\nALSC physical credit and exact net fraction deductions remain unresolved. ENAT auction and source-specific loan rules remain unknown; NATU/SOMA loan conventions retain their prior hypothesis status. Next resolve the pre-custody disposal and allocation/timing/sweep/fraction/invoice/security-day/minimum/clearing bounds, then separated corporate costs and final StageA. C matched economic/source replays, separately accepted new-data refits and conditionalD remain unstarted. New succession target/history implications must be attributed separately before future data acceptance; do not rerun the completed scalar/store/source work to revisit unknown vintages.\n"
    (PROJECT / "docs/v2_ENAT_FRACTION_SETTLEMENT.md").write_text(
        report, encoding="utf8"
    )
    print(json.dumps(binding(path)))


if __name__ == "__main__":
    main()
