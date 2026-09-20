"""Compose the qualified primary account and its explicit conditional limits."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, time
import json
from pathlib import Path
import shutil
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from brazil_rv.execution.custody_fees import CustodyAssessment, custody_schedule
from brazil_rv.execution.spot_costs import MonthlySpotTariff, execution_bps
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    root = Path(run["root"]) / "integrated_admission"
    out = root / "qualified"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    evidence = {
        k: run[k]
        for k in (
            "loan_contract_acceptance",
            "loan_accrual_acceptance",
            "loan_allocation_acceptance",
            "spot_settlement_acceptance",
            "loan_cash_acceptance",
            "share_custody_acceptance",
            "loan_renewal_acceptance",
            "loan_return_notice_acceptance",
            "opening_claim_acceptance",
            "natura_settlement_acceptance",
            "enat_settlement_acceptance",
            "precredit_disposal_acceptance",
            "copel_loan_bounds_acceptance",
            "payment_bounds_acceptance",
            "loan_invoice_acceptance",
            "historical_spot_acceptance",
            "custody_fee_acceptance",
            "historical_cost_acceptance",
            "spot_invoice_acceptance",
            "january_calendar_acceptance",
            "composed_data_inputs",
            "event_composition_qualification",
            "surviving_rename_admission",
            "identity_source_admission",
            "composed_store_recovery",
            "january_calendar_recovery",
        )
    }
    for record in evidence.values():
        bound_json(record)
    primary, dates = load_corporate_replay(
        run["composed_primary_event_terms"]["path"],
        run["composed_primary_event_terms"]["sha256"],
    )
    primary = deepcopy(primary)
    primary["supersedes"] = run["composed_primary_event_terms"]
    links_path = PROJECT / "research/configs/v2/isin_links_allowlist.csv"
    links = pl.read_csv(links_path).to_dicts()
    assert len(links) == 5
    identities = []
    for link in links:
        assert (
            link["shares_received_per_prior_share"] == 1
            and link["cash_entitlement_per_prior_share"] == 0
        )
        known = datetime.fromisoformat(link["first_known_at"].replace("Z", "+00:00"))
        local_known = known.astimezone(ZoneInfo("America/Sao_Paulo"))
        known_date = np.datetime64(local_known.date())
        first_day = int(np.searchsorted(dates, known_date))
        if dates[first_day] == known_date and local_known.time() > time(15, 45):
            first_day += 1
        available = str(dates[first_day])
        assert available <= link["effective_date"]
        identities.append(
            dict(
                predecessor_isin=link["predecessor_isin"],
                successor_isin=link["successor_isin"],
                effective_date=link["effective_date"],
                available_date=available,
                first_known_at=link["first_known_at"],
                source=dict(path=link["source"], sha256=link["evidence_sha256"]),
                contract="Same surviving legal issuer/class, unit/no cash. Existing positions, pending receipts and original contractual cohorts rekey under explicit book-identity hypothesis; no new physical receipt, loan rate/balance alias, locate, principal, maturity or minimum.",
            )
        )
    primary["identity_actions"] = identities
    primary["sources"] += [binding(links_path)]
    primary["sources"] += list(
        {x["source"]["sha256"]: x["source"] for x in identities}.values()
    )
    primary["status"] = "integrated_primary_account_terms_with_explicit_hypotheses"
    primary["pending_cases"] = [
        "ALSC physical credit/exact net cash unknown: whole signed claim remains locked",
        "ENAT fraction auction unknown: signed residual retained and lot-dependent unit labels unsupported",
        "Cielo held-loan cent, opposing same-security/day spot fills and new actual untested corporate interactions require exposure-specific qualification",
        "Same-class held-loan identity is a bookkeeping hypothesis, not a new published-rate or borrow-balance alias; conditional actual-path interpretation retained",
        "January25 closed-day primary with separately qualified cash/delivery/accrual hypotheses; historical clearing fact remains unknown",
    ]
    write_json_atomic(out / "primary_terms.json", primary)
    terms_ref = binding(out / "primary_terms.json")
    primary, calendar = load_corporate_replay(terms_ref["path"], terms_ref["sha256"])
    store = Path(primary["store"]["root"])
    names = np.load(store / "isin_index.npy").tolist()
    base = inputs_on_axes(store, calendar, np.arange(933), np.arange(len(calendar)))
    revised = apply_corporate_replay(base, primary, calendar, terms_ref["sha256"])
    retained = []
    for key in (
        "scores",
        "score_mask",
        "active",
        "raw_close",
        "prior_feature_values",
        "scaled_midrank_targets",
        "shareholder_simple_returns",
        "target_scale_sigma",
        "annual_borrow_rate_by_name",
        "shortable_by_borrow_source",
        "loan_reference_prices",
    ):
        assert getattr(revised, key) is getattr(base, key)
        retained.append(key)
    checks = []
    for event in identities:
        t = int(np.searchsorted(calendar, np.datetime64(event["effective_date"])))
        n, dest = (
            names.index(event["predecessor_isin"]),
            names.index(event["successor_isin"]),
        )
        np.testing.assert_array_equal(
            revised.action_successor_index[:t, n], base.action_successor_index[:t, n]
        )
        assert revised.action_successor_index[t, n] == dest
        assert revised.action_shares_per_prior_share[t, n] == 1
        assert revised.action_cash_per_prior_share[t, n] == 0
        assert revised.action_has_action[t, n]
        assert revised.action_session_resolved[t:, n].all()
        checks.append(
            dict(
                source=event["predecessor_isin"],
                destination=event["successor_isin"],
                effect=event["effective_date"],
                source_index=n,
                successor_index=dest,
            )
        )
    # Reuse the published calendar proof; do not reparse originals or replay books.
    tariff = bound_json(run["historical_cost_sources"])
    historical = bound_json(run["historical_cost_audit"])
    parent = next(
        c
        for c in bound_json(historical["completed"])
        if c["start"] == "2016-10-24"
        and c["scenario"] == "primary"
        and c["capital"] == 10000000
    )
    book = bound_json(
        binding(Path(historical["audit_root"]) / parent["book"] / "book.json")
    )
    cfg = book["provenance"]["config"].copy()
    cfg["monthly_spot_tariffs"] = tuple(
        MonthlySpotTariff(**x) for x in cfg["monthly_spot_tariffs"]
    )
    cfg["custody_assessments"] = tuple(
        CustodyAssessment(**x) for x in cfg["custody_assessments"]
    )
    original = LedgerConfig(**cfg)
    schedule_path = (
        PROJECT / "research/configs/v2/b3_session_schedule_reconstructed_v1.csv"
    )
    schedule = (
        pl.read_csv(schedule_path, columns=["trade_date"])["trade_date"]
        .to_numpy()
        .astype("datetime64[D]")
    )
    # Calendar dates only. January metadata establishes the last December month-end
    # and retains future payment dates; no 2025 market rows/features are consumed.
    schedule = schedule[schedule <= np.datetime64("2025-01-31")]
    np.testing.assert_array_equal(
        schedule[(schedule >= calendar[0]) & (schedule <= calendar[-1])], calendar
    )
    config = replace(
        original,
        custody_assessments=custody_schedule(schedule, "2016-07-18", "2024-12-30"),
    )
    coverage = []
    for row in tariff["calendar"]:
        d = row["date"]
        bps = execution_bps(config, d)
        expected = [
            0.0,
            0.5 if row["trading_bps"] is None else row["trading_bps"],
            2.75 if d < "2021-02-02" else 2.5,
            0.0,
            1.0,
        ]
        np.testing.assert_array_equal(bps, [expected, expected])
        coverage.append(dict(date=d, components=bps[0].tolist()))
    assert len(coverage) == 2099
    assert len(config.custody_assessments) == 102
    assert config.custody_assessments[-1].date == "2024-12-30"
    assert config.short_proceeds_remuneration == 1 and config.loan_term_sessions == 63
    assert config.cost_bps_per_side == config.hedge_cost_bps_per_side == 0
    write_json_atomic(
        out / "coverage.json",
        dict(
            spot=coverage,
            custody=[asdict(x) for x in config.custody_assessments],
            calendar_source=binding(schedule_path),
            calendar_only_through="2025-01-31",
            market_consumers_end="2024-12-30",
        ),
    )
    account = dict(
        schema="BRAZIL_RV_ECONOMIC_ACCOUNT",
        status="integrated_account_admitted",
        primary_config=asdict(config),
        capital_checks=[1000000, 5000000],
        development_start="2016-07-18",
        development_end="2024-12-30",
        corporate_terms=terms_ref,
        source_inputs={
            k: run[k]
            for k in (
                "cash_calendar",
                "qualified_lending",
                "loan_source_panels",
                "bova_loan_reference_audit",
            )
        },
        new_refit_inputs=run["composed_data_inputs"],
        evidence=evidence,
        scenario_dependencies={
            k: run[k]
            for k in (
                "loan_return_notice_acceptance",
                "precredit_disposal_acceptance",
                "copel_loan_bounds_acceptance",
                "payment_bounds_acceptance",
                "loan_invoice_acceptance",
                "natura_settlement_acceptance",
                "enat_settlement_acceptance",
                "historical_spot_acceptance",
                "custody_fee_acceptance",
                "historical_cost_acceptance",
                "spot_invoice_acceptance",
                "january_calendar_acceptance",
            )
        },
        rules=[
            "Old forecast accounting/source attribution shallow-copies frozen OLDPolicyData; never regenerate old static features or feed changed model coordinates to old weights.",
            "Repaired data refits require new conditioning and P weights, compatible F only; fixed schema/population/history/budgets remain.",
            "Keep all prior adaptive uncertainty; every actual path reports unresolved obligations and exposure-triggered bounds. No forced Cielo holding or invented daytrade execution.",
            "Primary100%CDI/zero brokerage, account payment/valuation, loan lifecycle and unresolved lender terms are explicit hypotheses, not obtained quotes or exact client invoices.",
            "Reuse qualified mechanics/source/consumer evidence. New actual model-path interactions may require narrow qualification before economic adoption.",
        ],
    )
    write_json_atomic(out / "account.json", account)
    report = dict(
        status="integrated_stage_a_admitted_with_explicit_conditional_exposure_limits",
        account=binding(out / "account.json"),
        plan=binding(root / "plan.json"),
        evidence=evidence,
        identity_actions=checks,
        preserved_input_objects=retained,
        dated_spot_sessions=len(coverage),
        monthly_assessments=len(config.custody_assessments),
        coverage=binding(out / "coverage.json"),
        new_historical_books=0,
        reused_proofs="V31-V41, historical tariffs/corporate custody/invoices, calendar15books, complete composed store and all family/source proofs",
        tests=binding(root / "identity_axis_qualified_tests/stdout.txt"),
        model_scoring_or_fitting=False,
        source_fact_uncertainties=primary["pending_cases"],
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    write_json_atomic(PROJECT / "docs/v2_economic_account_acceptance.json", report)
    run["economic_account"] = binding(out / "account.json")
    run["economic_account_acceptance"] = binding(
        PROJECT / "docs/v2_economic_account_acceptance.json"
    )
    run["corporate_replay"] = terms_ref
    run["integrated_account_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("evidence", "source_fact_uncertainties")
            }
        )
    )


if __name__ == "__main__":
    main()
