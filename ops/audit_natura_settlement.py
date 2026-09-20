"""Bounded original-axis claim valuations and adaptive engineering books."""

from copy import copy, deepcopy
from dataclasses import asdict, replace
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
from brazil_rv.v2.artifacts import write_json_atomic, sha256_file
from brazil_rv.v2.cash_calendar import apply_cash_calendar
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.portfolio_inputs import Calibration
from brazil_rv.v2.portfolio_readouts import book_summary, save_book
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main(label, *, probe=False):
    started = perf_counter()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    root = Path(run["root"]) / "natura_settlement" / label
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / "executed.py")
    for folder in ("execution", "v2"):
        for f in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            shutil.copyfile(f, root / f"executed_{folder}_{f.name}")
    old, dates = load_corporate_replay(
        run["opening_claim_terms"]["path"], run["opening_claim_terms"]["sha256"]
    )
    source = bound_json(run["held_event_source_audit"])
    amendment = run["natura_gross_action_amendments"]
    assert sha256_file(Path(amendment["path"])) == amendment["sha256"]
    records = pl.read_parquet(amendment["path"]).to_dicts()
    terms = deepcopy(old)
    terms["supersedes"] = run["opening_claim_terms"]
    terms["scalar_source_terms"] = amendment
    terms["status"] = "natura_settlement_engineering_not_final_economic_admission"
    terms["scalar_actions"] = []
    for row in records:
        bonus = row["action_type"] == "bonus"
        terms["scalar_actions"].append(
            dict(
                isin=row["isin"],
                effective_date=str(row["effective_date"]),
                available_date=str(row["available_at"].date()),
                available_at=row["available_at"].isoformat(),
                shares_per_prior_share=row["shares_per_prior_share"],
                gross_cash_per_prior_share=row["cash_per_prior_share"],
                payment_date=None
                if row["payment_date"] is None
                else str(row["payment_date"]),
                bonus_delivery_date="2019-09-20" if bonus else None,
                withholding_rate=0 if bonus else 0.15,
                short_cash_fraction=1,
                source=row["source"],
                evidence=row["evidence"],
                loan_hypothesis="economic quantity at effect; preserve original principal/rate/minimum; all same-name returns and their proceeds wait at least to bonus credit"
                if bonus
                else "gross lender compensation primary; net85% one-factor bound; no tax credit asset",
            )
        )
        pdf = source["source_files"][row["source"].split(":")[1]]["pdf"]
        if pdf not in terms["sources"]:
            terms["sources"].append(pdf)
    write_json_atomic(root / "terms.json", terms)
    plan = dict(
        dates=["2019-09-10", "2020-02-28"],
        capital=[10_000_000, 1_000_000, 5_000_000],
        signs=[1, -1],
        scenarios=[
            "legacy_scalar",
            "gross_instant",
            "primary",
            "immediate_credit",
            "net_compensation",
        ],
        contrasts={
            "source_gross": "gross_instant minus legacy_scalar",
            "account_withholding": "immediate_credit minus gross_instant",
            "bonus_custody": "primary minus immediate_credit",
            "lender_compensation": "net_compensation minus primary",
        },
        preferences="sin(axis*.31+localday*.003+head*.2), NATU focus +/-4; frozen engineering risks and original .002/.001/.0005 coefficients; no neural scoring/model-alpha",
        costs="old bundled4bp with NO additional B3 spot; correctedCDI/qualifiedloan sources; final corporate pricing pending",
        terms=binding(root / "terms.json"),
        source_terms=amendment,
        frozen_old_policy_coordinates=True,
        accepted_stores_and_old_fits_unchanged=True,
        boundaries="trade-date economic entitlement; bonus incremental shares cannot be disposed until credit; loan whole-return deferral conservative hypothesis; no exemption/tax-credit/loan alias; old data targets stay gross and unchanged",
    )
    if probe:
        plan["dates"] = ["2019-09-10", "2019-09-27"]
        plan["scenarios"] = ["primary", "immediate_credit"]
        plan["preferences"] += (
            "; independently frozen disposal probe reverses NATU focus sign from bonus effect through two sessions after credit"
        )
        plan["reason"] = (
            "Incremental bonus custody needs actual adaptive exit pressure as well as constant-focus long-span entitlement/payment paths; frozen before probe outcomes."
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

    rows = np.flatnonzero(
        (dates >= np.datetime64(plan["dates"][0]))
        & (dates <= np.datetime64(plan["dates"][1]))
    )
    assert np.array_equal(rows, np.arange(rows[0], rows[-1] + 1))
    name = isins.index("BRNATUACNOR6")
    effects = {
        key: int(np.searchsorted(dates[rows], np.datetime64(value)))
        for key, value in dict(
            bonus="2019-09-18", credit="2019-09-20", jcp="2019-11-07", pay="2020-02-26"
        ).items()
    }
    inputs = inputs_on_axes(store, dates, names, rows)
    prior = np.load(store / "prior_reference_close.npy", mmap_mode="r")[rows]
    reports = []
    for sign in plan["signs"]:
        scores = np.sin(
            names[None, :, None] * 0.31
            + np.arange(len(rows))[:, None, None] * 0.003
            + np.arange(5)[None, None, :] * 0.2
        ).astype(np.float32)
        scores[:, name, :] = 4 * sign
        if probe:
            scores[effects["bonus"] : effects["credit"] + 2, name, :] *= -1
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
        model = CalibratedPolicy(
            Calibration(np.zeros(3), np.ones(3), np.array([0.002, 0.001, 0.0005]), 0)
        )
        for scenario in plan["scenarios"]:
            variant = deepcopy(terms)
            if scenario == "legacy_scalar":
                variant["scalar_actions"] = []
            else:
                bonus, jcp = variant["scalar_actions"]
                if scenario in ("gross_instant", "immediate_credit"):
                    bonus["bonus_delivery_date"] = bonus["effective_date"]
                if scenario == "gross_instant":
                    jcp["withholding_rate"] = 0
                if scenario == "net_compensation":
                    jcp["short_cash_fraction"] = 0.85
            path = root / f"terms_{scenario}.json"
            if not path.exists():
                write_json_atomic(path, variant)
            data = copy(frozen)
            data.inputs = apply_cash_calendar(
                apply_corporate_replay(base, variant, dates, sha256_file(path)),
                dates,
                cdi,
                run["cash_calendar"]["sha256"],
            )
            assert data.static is frozen.static and data.references is frozen.references
            assert (
                data.inputs.scores is base.scores
                and data.inputs.loan_reference_prices is base.loan_reference_prices
            )
            for capital in plan["capital"]:
                book = f"{scenario}_{sign}_{capital}"
                tick = perf_counter()
                config = policy_ledger_config(initial_capital_brl=capital)
                result, targets, previous = exact_replay(
                    data, model, 0, len(rows), config=config
                )
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
                        terms=binding(path),
                    ),
                )
                fixed, adaptive = (
                    data.initial_account(0, config),
                    data.initial_account(0, config),
                )
                fixed_error = adaptive_error = target_error = 0.0
                with torch.no_grad():
                    for day, target in enumerate(targets):
                        adaptive_target = account_decision(data, model, adaptive, day)
                        if day < len(rows) - 1:
                            target_error = max(
                                target_error,
                                float(np.max(np.abs(adaptive_target.numpy() - target))),
                            )
                        a = data.step(
                            fixed,
                            torch.from_numpy(target),
                            day,
                            terminal=day == len(rows) - 1,
                        )
                        data.step(
                            adaptive,
                            adaptive_target,
                            day,
                            terminal=day == len(rows) - 1,
                        )
                        fixed_error = max(
                            fixed_error, abs(float(a["nav"]) - result.nav[day])
                        )
                        adaptive_error = max(
                            adaptive_error, abs(float(adaptive.nav) - result.nav[day])
                        )
                summary, _ = book_summary(data, result, previous, 0, 0)
                report = dict(
                    book=book,
                    scenario=scenario,
                    sign=sign,
                    capital=capital,
                    sessions=len(rows),
                    bonus_pre_quantity=float(
                        result.signed_shares[effects["bonus"] - 1, name]
                    ),
                    jcp_pre_quantity=None
                    if effects["jcp"] >= len(rows)
                    else float(result.signed_shares[effects["jcp"] - 1, name]),
                    withholding_accrual=float(result.withholding_accrual.sum()),
                    lender_compensation=float(result.lender_compensation.sum()),
                    identical_intention_nav_error=fixed_error,
                    adaptive_nav_difference_bps=adaptive_error / capital * 1e4,
                    adaptive_target_difference=target_error,
                    final_nav=float(result.nav[-1]),
                    summary=summary,
                    seconds=perf_counter() - tick,
                )
                reports.append(report)
                write_json_atomic(root / "completed.json", reports)
                print(
                    json.dumps({k: v for k, v in report.items() if k != "summary"}),
                    flush=True,
                )
                assert fixed_error < capital * 1e-10, (book, fixed_error)
                assert sign * report["bonus_pre_quantity"] > 0 and (
                    report["jcp_pre_quantity"] is None
                    or sign * report["jcp_pre_quantity"] > 0
                )
    manifest = dict(
        status="bounded_natura_settlement_books_complete_pending_final_economics",
        audit_root=str(root),
        plan=binding(root / "plan.json"),
        terms=binding(root / "terms.json"),
        completed=binding(root / "completed.json"),
        effects=effects,
        name=name,
        seconds=perf_counter() - started,
    )
    write_json_atomic(root / "manifest.json", manifest)
    run["natura_bonus_disposal_audit" if probe else "natura_settlement_audit"] = (
        binding(root / "manifest.json")
    )
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    main(args.label, probe=args.probe)
