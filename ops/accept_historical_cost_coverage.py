"""Bind completed tariff/corporate evidence without replaying accepted books."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    root = Path(run["root"]) / "tariff_coverage"
    sources = bound_json(binding(root / "qualification.json"))
    ordinary = bound_json(binding(root / "runtime_qualification/ordinary_report.json"))
    corporate = bound_json(
        binding(root / "corporate_books/saved_qualification/report.json")
    )
    runtime = bound_json(binding(root / "runtime_qualification/report.json"))
    for row in runtime["corporate_runtime"]:
        assert binding(Path(row["current"]["path"])) == row["current"]
        assert row["current"]["sha256"] == row["executed"]["sha256"]
    audits = [
        bound_json(binding(root / f"{p}/manifest.json"))
        for p in ("books", "corporate_books")
    ]
    counts = [len(bound_json(a["completed"])) for a in audits]
    assert counts == [18, 34]
    assert (ordinary["cells"], corporate["cells"]) == (3513141, 8883040)
    assert len(sources["monthly_sources"]) == 50
    assert sum(sources["counts"].values()) == 2099
    assert not corporate["cielo_cash_loan_exposed_books"]
    prior_calendar = bound_json(run["clearing_calendar_sources"])
    calendar_2017 = next(r for r in prior_calendar if r["id"] == "b3_calendar_2017")
    calendar = dict(
        status="one_additional_date_resolved_by_dated_source_composition",
        prior_audit=run["clearing_calendar_audit"],
        prior_sources=run["clearing_calendar_sources"],
        holiday_source=calendar_2017,
        migration_original=binding(root / "b3_047_2017_integration.pdf"),
        migration_receipt=binding(root / "b3_047_2017_integration_receipt.json"),
        migration_visual_page=binding(root / "b3_047_2017_integration_p1.png"),
        qualified_pages=[1],
        resolved_date="2017-11-20",
        disposition="Equity clearing closed; accepted equity calendar remains unchanged.",
        reasoning="047/2017-DP dated August18 ends the former equity clearing house on August25 and migrates equities to Camara BM&FBOVESPA on August28. The already qualified111/2016 holiday table closes that named clearing house on November20. This is a two-source inference, not an observed settlement receipt.",
        remaining_ambiguous_dates=["2016-12-30", "2017-01-25"],
        resolved_money_only_dates=21,
        total_money_only_dates=23,
        changed_settlement_indices=0,
        changed_quote_or_model_arrays=0,
        failed_targeted_2016_index=binding(
            root / "index_calendar_2016_calendar_index_plan_failure.json"
        ),
        limitation="January25 predates the clearing migration; do not extend the November inference backward. A closed delivery day alone does not prove all rent-accrual conventions. No new calendar-wide census.",
    )
    write_json_atomic(root / "calendar_resolution.json", calendar)
    logs = [
        "custody_tests_first.log",
        "spot_tests_first.log",
        "corporate_first/tests.log",
        "corporate_second/tests.log",
        "corporate_third/tests.log",
        "corporate_fourth/tests.log",
        "regression/stdout.txt",
        "corporate_qualification_first.log",
        "corporate_qualification_second.log",
        "runtime_qualification.log",
        "final_ruff.log",
    ]
    attempts = dict(
        bound_logs={p: binding(root / p) for p in logs},
        new_historical_tests=4,
        new_corporate_tests=17,
        affected_regression_tests=72,
        overlap="Historical12custody/9spot include10/7previous cases. Corporate third batch15 plus fourth2 are17, not the sum of earlier attempts. Third batch also repeats those12/9 historical cases.",
        maintenance="Initial fixed-amount probe found Float32 creation; qualified full_like Float64 implementation preceded all books. Initial code/probe retained in implementation_initial.",
        corporate_fixtures="First9pass/1fail cash precision at100000; only fixture capital became1000. Second11pass/2fail expected zero stock before D2 spot sale, overlooking actual D1 borrowed receipt; corrected fixture expectations only. Third36pass and fourth2pass. No production change/book rerun resulted from these fixture corrections.",
        saved_qualifier="Initial np.max on an empty current-positive-price comparison failed. Qualified empty reduction uses initial=0, retaining all34books; exact failed/qualified recipes and logs saved.",
        prose_correction=runtime["correction"],
        retrieval="All selected attempts, empty responses/timeouts and successful originals retained. No broad repeat census. An initial011/2017 www-host failure was overwritten by its next attempt; the following record is reconstructed from tool output, not an original receipt.",
        reconstructed_first_011_failure=dict(
            url="https://www.b3.com.br/lumis/portal/file/fileDownload.jsp?fileId=8AA8D0976075EB9901609E1F19067AEA",
            error="TimeoutError('The read operation timed out')",
            seconds=35.07613590001711,
        ),
        reconstructed_read_only_attempts="Wrong read-only paths and Windows rg wildcard queries, an oversized truncated report print and an initial assumed JSON pages key changed no source/book artifacts. These are prose reconstructions, not retained exact failed shell bytes.",
        timing="Book/source/qualification timings are CPU audit runtimes. Manual search, rendering and all retrieval overhead were not measured consistently; no total wall time or fit ETA inferred.",
    )
    write_json_atomic(root / "attempts.json", attempts)
    bindings = {
        "historical_cost_sources": binding(root / "qualification.json"),
        "historical_cost_audit": binding(root / "books/manifest.json"),
        "historical_cost_qualification": binding(
            root / "runtime_qualification/ordinary_report.json"
        ),
        "corporate_custody_audit": binding(root / "corporate_books/manifest.json"),
        "corporate_custody_qualification": binding(
            root / "corporate_books/saved_qualification/report.json"
        ),
        "historical_cost_runtime": binding(root / "runtime_qualification/report.json"),
        "historical_cost_attempts": binding(root / "attempts.json"),
        "clearing_calendar_resolution": binding(root / "calendar_resolution.json"),
    }
    dependencies = {
        k: run[k]
        for k in (
            "custody_fee_recovery",
            "historical_spot_recovery",
            "loan_invoice_recovery",
            "payment_bounds_recovery",
            "copel_loan_bounds_recovery",
            "precredit_disposal_recovery",
            "enat_settlement_recovery",
            "enat_settlement_terms",
            "natura_settlement_recovery",
            "loan_return_notice_recovery",
            "economic_store_recovery",
            "natura_store_recovery",
            "lending_feature_recovery",
            "cash_calendar",
            "loan_source_panels",
            "bova_loan_reference_audit",
            "clearing_calendar_sources",
            "clearing_calendar_audit",
            "cielo_foundation_exposure_recovery",
        )
    }
    for record in dependencies.values():
        bound_json(record)
    acceptance = dict(
        status="historical_tariff_and_corporate_custody_engineering_accepted_not_final_stage_a",
        evidence=bindings,
        report=binding(PROJECT / "docs/v2_HISTORICAL_COST_COVERAGE.md"),
        closeout=binding(PROJECT / "docs/v2_STAGE_A_CLOSEOUT.md"),
        books=52,
        book_counts=dict(historical=18, corporate=34),
        skipped_unexposed_rights_variants=len(corporate["skipped"]),
        account_array_cells=ordinary["cells"] + corporate["cells"],
        sessions=ordinary["sessions"] + corporate["sessions"],
        actual_fills=ordinary["actual_fills"] + corporate["fills"],
        monthly_assessments=ordinary["assessments"] + corporate["assessments"],
        monthly_original_archives=50,
        new_original_pdfs=10,
        visual_pages=34,
        coverage=sources["counts"],
        historical_errors=ordinary["errors"],
        corporate_errors=corporate["errors"],
        historical_adaptive_max_bps=ordinary["adaptive_max_bps"],
        corporate_adaptive_max_bps=corporate["adaptive_max_bps"],
        source_entitlement_checks=corporate["source_entitlement_checks"],
        independent_ordinary_physical_cells=corporate["ordinary_physical_flow_cells"],
        runtime_state_checks=runtime["ordinary_current_close_saved_state_days"],
        runtime_max_error=runtime["ordinary_current_close_max_error"],
        engineering_seconds=dict(
            historical=audits[0]["seconds"], corporate=audits[1]["seconds"]
        ),
        saved_qualification_seconds=dict(
            historical=ordinary["seconds"], corporate=corporate["seconds"]
        ),
        contracts=[
            "b3_spot_dated replaces both stock/hedge bundled execution costs for2016-07-18 through2024-12-30. Previous-month global-market rates plus dated clearing; source-supported0.2/0.5bp trading bounds only on109 unrecovered/not-yet-known dates. Zero primary brokerage,1bp shortfall; no fund or account-capital ADTV discount.",
            "Dated progressive monthly value custody plus explicit one-open-active-CNPJ/custodian maintenance. Total custody_fee already includes separately reported maintenance. Own available mark and positive corporate rights inclusion/exclusion are hypotheses, not invoices.",
            "Pending spot/new-loan flows, delayed bonus, successor deliveries, old loan principals/charges and cash closeout remain separate. Unknown physical credit stays locked. No new alias, locate, quote or fabricated receipt.",
            "Frozen synthetic full933 adaptive books, shallow-copied old coordinates, correctedCDI/qualified loans/strict prior references. No model forward/scoring/fit. Total path contrasts retain all prior adaptive uncertainty.",
        ],
        open_items=[
            "Spot invoice rounding and actual-exposure daytrade disposition.",
            "2016-12-30 and2017-01-25 equity-clearing ambiguity.",
            "Bound primary event/source composition and separately attributed succession-data implications.",
            "Cielo held-loan cents remain conditional; all34 new books have no loan cash redemption and do not supply a general bound.",
            "Integrated StageA admission, registered C replays/refits and conditional D.",
        ],
        prior_recovery_dependencies=dependencies,
        immutable_contracts={
            k: run[k] for k in ("corporate_replay", "economic_refit_inputs")
        },
    )
    accepted_path = PROJECT / "docs/v2_historical_cost_coverage_acceptance.json"
    write_json_atomic(accepted_path, acceptance)
    run.update(bindings)
    run["historical_cost_acceptance"] = binding(accepted_path)
    run["stage_a_closeout_plan"] = binding(PROJECT / "docs/v2_STAGE_A_CLOSEOUT.md")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                acceptance=run["historical_cost_acceptance"],
                books=52,
                cells=acceptance["account_array_cells"],
            )
        )
    )


if __name__ == "__main__":
    main()
