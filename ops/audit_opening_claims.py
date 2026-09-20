"""Bounded original-axis claim valuations and adaptive engineering books."""

from copy import copy, deepcopy
from dataclasses import asdict, replace
from decimal import Decimal
import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    PolicyData,
    account_decision,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.execution.share_distributions import recognize_distribution
from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.cash_calendar import apply_cash_calendar
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.portfolio_inputs import Calibration
from brazil_rv.v2.portfolio_readouts import book_summary, save_book
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main(label):
    started = perf_counter()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    root = Path(run["root"]) / "opening_claims" / label
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / "executed.py")
    for f in (PROJECT / "research/src/brazil_rv/execution").glob("*.py"):
        shutil.copyfile(f, root / f"executed_{f.name}")
    shutil.copyfile(
        PROJECT / "research/src/brazil_rv/v2/corporate_replay.py",
        root / "executed_corporate_replay.py",
    )
    source = bound_json(run["held_event_source_audit"])
    old, dates = load_corporate_replay(
        run["corporate_replay"]["path"], run["corporate_replay"]["sha256"]
    )
    cases = source["cases"]
    terms = deepcopy(old)
    terms["supersedes"] = run["corporate_replay"]
    terms["held_source_evidence"] = run["held_event_source_audit"]
    terms["status"] = "opening_valuation_engineering_not_final_held_account_admission"
    for case in cases:
        auction = None
        if case["isin"] == "BRSOMAACNOR3":
            auction = dict(
                available_date="2024-08-23",
                payment_date="2024-08-26",
                cash_per_share=50.20718,
                provision_loan_fractions=True,
                zero_quantity_rent_through_payment=False,
                hypothesis="per-original-loan whole shares and separate fraction; event-specific loan instructions unrecovered",
            )
        terms["share_distributions"].append(
            dict(
                isin=case["isin"],
                effective_date=case["effective"],
                available_date=case["available"],
                legal_consummation=case["legal_consummation"],
                carry_source_value=True,
                cash_per_prior_share=0.0,
                payment_date=None,
                legs=[
                    dict(
                        successor_isin=case["successor_isin"],
                        shares_per_prior_share=float(case["ratio"]),
                        delivery_date=case["credit"],
                        loan_principal_fraction=1.0,
                        fractional_auction=auction,
                    )
                ],
                loan_hypothesis="original principal/rate/fees retained, conversion at known share credit; unknown credit leaves original loan outstanding",
                unresolved=case["loan_disposition"] + "; " + case["fraction"],
            )
        )
        for protocol in case["protocols"]:
            record = source["source_files"][protocol]["pdf"]
            if record not in terms["sources"]:
                terms["sources"].append(record)
    write_json_atomic(root / "terms.json", terms)
    plan = dict(
        cases=cases,
        source=run["held_event_source_audit"],
        terms=binding(root / "terms.json"),
        sessions="six pre-effect and twenty-four effect/following sessions",
        capital=[10_000_000, 1_000_000, 5_000_000],
        signs=[1, -1],
        scenarios={
            "NATU": ["primary"],
            "ALSC": ["unknown_custody"],
            "SOMA": ["provisioned", "continuous_loan", "auction_quotient"],
        },
        preferences="all-name sin(axis*.31+localday*.003+head*.2), focus name +/-4; original .002/.001/.0005 calibration coefficients",
        risk="fixed engineering beta1, idio.0004, market.0001; actual constrained allocator, no neural scores",
        costs="old bundled4bp bridge with no extra B3 spot; sourced loan/CDI/reference panels; final corporate pricing pending",
        limits="NATU bonus/JCP outside flat-start window. Unknown ALSC custody/fractions persist. No source/fit/store changes, no alpha conclusion.",
        bindings={
            k: run[k]
            for k in (
                "corporate_replay",
                "qualified_lending",
                "loan_source_panels",
                "cash_calendar",
                "bova_loan_reference_audit",
            )
        },
    )
    write_json_atomic(root / "plan.json", plan)
    store = Path(old["store"]["root"])
    isins = np.load(store / "isin_index.npy").tolist()
    names = np.arange(len(isins))
    assert len(names) == 933 and dates[-1] <= np.datetime64("2024-12-30")
    lend = run["qualified_lending"]
    lending = load_lending_borrow_panels(
        Path(lend["root"]),
        expected_manifest_sha256=lend["manifest_sha256"],
        canonical_dates=dates.astype(object).tolist(),
        canonical_isins=isins,
    )
    panels = bound_json(run["loan_source_panels"])["panels"]
    assert sha256_file(Path(panels["path"])) == panels["sha256"]
    with np.load(panels["path"]) as z:
        references, hedge_rates = (
            z["loan_reference_prices"],
            z["hedge_annual_borrow_rate"],
        )
    cash = bound_json(run["cash_calendar"])["panel"]
    assert sha256_file(Path(cash["path"])) == cash["sha256"]
    with np.load(cash["path"]) as z:
        cdi = z["cdi_returns"]
    hedge = pl.read_parquet(
        bound_json(run["bova_loan_reference_audit"])["data"]["path"]
    )
    hmap = dict(zip(hedge["trade_date"], hedge["close_brl"]))
    hedge_close = np.array([hmap.get(d, np.nan) for d in dates.astype(object)])
    reports, oracle = [], []
    for case in cases:
        name, successor = case["indices"]
        effect = int(np.searchsorted(dates, np.datetime64(case["effective"])))
        rows = np.arange(effect - 6, effect + 24)
        local_effect = 6
        inputs = inputs_on_axes(store, dates, names, rows)
        prior = np.load(store / "prior_reference_close.npy", mmap_mode="r")[rows]
        assert not np.isfinite(prior[local_effect, successor])
        expected = Decimal(str(case["source_last_close"])) / Decimal(case["ratio"])
        for sign in plan["signs"]:
            scores = np.sin(
                names[None, :, None] * 0.31
                + np.arange(len(rows))[:, None, None] * 0.003
                + np.arange(5)[None, None, :] * 0.2
            ).astype(np.float32)
            scores[:, name, :] = 4 * sign
            base = replace(
                inputs,
                scores=scores,
                score_mask=np.repeat(inputs.active[..., None], 5, -1),
                annual_borrow_rate_by_name=lending.annual_taker_rate[rows],
                borrow_rate_imputed=lending.rate_imputed[rows],
                borrow_rate_placeholder=lending.rate_placeholder[rows],
                shortable_by_borrow_source={
                    k: v[rows] for k, v in lending.availability.items()
                },
                loan_reference_prices=references[rows],
                hedge_annual_borrow_rate=hedge_rates[rows],
                bova11_close=hedge_close[rows],
                initial_hedge_reference_price=hedge_close[rows[0] - 1],
            )
            shape = base.active.shape
            frozen = PolicyData(
                base,
                np.ones(shape),
                np.full(shape, 0.0004),
                np.full(len(rows), 0.0001),
                cdi[rows - 1],
                prior,
            )
            label_name = case["isin"][2:6]
            scenarios = plan["scenarios"][label_name]
            for scenario in scenarios:
                variant = deepcopy(terms)
                evt = next(
                    x
                    for x in variant["share_distributions"]
                    if x["isin"] == case["isin"]
                )
                if scenario == "continuous_loan":
                    evt["legs"][0]["fractional_auction"]["provision_loan_fractions"] = (
                        False
                    )
                if scenario == "auction_quotient":
                    evt["legs"][0]["fractional_auction"]["cash_per_share"] = float(
                        Decimal("916532.03") / Decimal("18255")
                    )
                variant_path = root / f"terms_{label_name}_{scenario}.json"
                if not variant_path.exists():
                    write_json_atomic(variant_path, variant)
                data = copy(frozen)
                data.inputs = apply_cash_calendar(
                    apply_corporate_replay(
                        base, variant, dates, sha256_file(variant_path)
                    ),
                    dates,
                    cdi,
                    run["cash_calendar"]["sha256"],
                )
                assert (
                    data.static is frozen.static
                    and data.references is frozen.references
                )
                assert data.inputs.loan_reference_prices is base.loan_reference_prices
                event = next(
                    e for e in data.inputs.share_distributions if e.source_index == name
                )
                legs = recognize_distribution(event, prior[local_effect])
                np.testing.assert_allclose(
                    legs[0].opening_mark, float(expected), rtol=1e-7
                )
                oracle.append(
                    dict(
                        case=label_name,
                        sign=sign,
                        scenario=scenario,
                        decimal_opening=str(expected),
                            actual=float(legs[0].opening_mark),
                        source_prior=float(prior[local_effect, name]),
                        source_expected=case["source_last_close"],
                    )
                )
                model = CalibratedPolicy(
                    Calibration(
                        np.zeros(3), np.ones(3), np.array([0.002, 0.001, 0.0005]), 0
                    )
                )
                for capital in plan["capital"]:
                    book = f"{label_name}_{scenario}_{sign}_{capital}"
                    tick = perf_counter()
                    config = policy_ledger_config(initial_capital_brl=capital)
                    result, targets, previous = exact_replay(
                        data, model, 0, len(rows), config=config
                    )
                    # Save a completed exact book before checking another path.
                    save_book(
                        root / book,
                        data,
                        result,
                        targets,
                        previous,
                        0,
                        0,
                        dict(
                            policy="synthetic_focus_optimizer",
                            scenario="base",
                            config=asdict(config),
                        ),
                    )
                    fixed, adaptive = (
                        data.initial_account(0, config),
                        data.initial_account(0, config),
                    )
                    fixed_error = adaptive_error = decision_error = 0.0
                    with torch.no_grad():
                        for day, target in enumerate(targets):
                            actual_target = account_decision(data, model, adaptive, day)
                            if day < len(rows) - 1:
                                decision_error = max(
                                    decision_error,
                                    float(
                                        np.max(np.abs(actual_target.numpy() - target))
                                    ),
                                )
                            fixed_row = data.step(
                                fixed,
                                torch.from_numpy(target),
                                day,
                                terminal=day == len(rows) - 1,
                            )
                            data.step(
                                adaptive,
                                actual_target,
                                day,
                                terminal=day == len(rows) - 1,
                            )
                            fixed_error = max(
                                fixed_error,
                                abs(float(fixed_row["nav"]) - result.nav[day]),
                            )
                            adaptive_error = max(
                                adaptive_error,
                                abs(float(adaptive.nav) - result.nav[day]),
                            )
                    summary, _ = book_summary(data, result, previous, 0, 0)
                    report = dict(
                        book=book,
                        source_quantity_before=float(
                            result.signed_shares[local_effect - 1, name]
                        ),
                        effect_close_claim_value=float(
                            result.undelivered_share_notional[local_effect]
                        ),
                        final_undelivered=float(result.undelivered_share_notional[-1]),
                        identical_intention_nav_error=fixed_error,
                        adaptive_nav_difference_bps=adaptive_error / capital * 1e4,
                        adaptive_target_difference=decision_error,
                        summary=summary,
                        seconds=perf_counter() - tick,
                    )
                    reports.append(report)
                    write_json_atomic(root / "completed.json", reports)
                    print(
                        json.dumps({k: v for k, v in report.items() if k != "summary"}),
                        flush=True,
                    )
                    assert fixed_error < capital * 1e-8, (book, fixed_error)
                    assert sign * report["source_quantity_before"] > 0, book
    write_json_atomic(root / "oracle.json", oracle)
    manifest = dict(
        status="bounded_opening_claim_books_complete_pending_economic_admission",
        plan=binding(root / "plan.json"),
        terms=binding(root / "terms.json"),
        cases=reports,
        oracle=binding(root / "oracle.json"),
        seconds=perf_counter() - started,
    )
    write_json_atomic(root / "manifest.json", manifest)
    run["opening_claim_audit"] = binding(root / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    main(parser.parse_args().label)
