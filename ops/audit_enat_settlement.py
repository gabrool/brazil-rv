"""Original ENAT terms and new adaptive books with undated fraction claims."""

from copy import copy, deepcopy
from dataclasses import asdict, replace
from datetime import datetime
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
from brazil_rv.v2.decision_clock import normalize_publication_time
from brazil_rv.v2.lending_archive import load_lending_borrow_panels
from brazil_rv.v2.portfolio_inputs import Calibration
from brazil_rv.v2.portfolio_readouts import book_summary, save_book
from verify_corporate_replay import inputs_on_axes

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    source_root = Path(run["root"]) / "remaining_held_sources"
    root = Path(run["root"]) / "enat_settlement"
    root.mkdir(exist_ok=False)
    shutil.copyfile(__file__, root / "executed.py")
    for folder in ("execution", "v2"):
        for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            shutil.copyfile(path, root / f"executed_{folder}_{path.name}")
    index = bound_json(binding(source_root / "enat_index.json"))
    sources = {}
    for protocol, phrase in {
        "1260409": "05/08/2024",
        "1264467": "0,805012676",
        "1265227": "31 de julho de 2024",
        "1263891": "34 (trinta e",
    }.items():
        text_path = source_root / (protocol + ".txt")
        assert phrase in text_path.read_text(encoding="utf8")
        row = next(x for x in index["rows"] if x["protocol"] == protocol)
        dt = datetime.strptime(row["receipt"][9:], "%d/%m/%Y %H:%M")
        sources[protocol] = dict(
            pdf=binding(source_root / (protocol + ".pdf")),
            text=binding(text_path),
            receipt=binding(source_root / (protocol + "_receipt.json")),
            available_at=normalize_publication_time(
                dt, naive_timezone="America/Sao_Paulo", precision="minute"
            ).isoformat(),
        )
    old, dates = load_corporate_replay(
        run["natura_settlement_terms"]["path"], run["natura_settlement_terms"]["sha256"]
    )
    event = dict(
        isin="BRENATACNOR0",
        effective_date="2024-08-01",
        available_date="2024-07-31",
        legal_consummation="2024-07-31",
        carry_source_value=False,
        cash_per_prior_share=0.0,
        payment_date=None,
        legs=[
            dict(
                successor_isin="BRRRRPACNOR5",
                shares_per_prior_share=0.805012676,
                delivery_date="2024-08-05",
                loan_principal_fraction=1.0,
                fractional_auction=dict(
                    available_date=None,
                    payment_date=None,
                    cash_per_share=None,
                    provision_loan_fractions=False,
                ),
            )
        ],
        loan_hypothesis="Continuous loan quantity and conversion at shareholder credit, preserving original principal/rate/fees; no event-specific loan instruction recovered.",
        fraction_contract="Whole shareholder shares credited; undated fractions stay signed locked claims. No invented auction or zero cash.",
        withdrawal="34 source-reported dissenting shares only; ordinary portfolio is non-electing. R14.59 is not merger cash for every holder.",
        tax_scope="Original nonresident IRRF passage does not establish a charge or exemption for domestic CNPJ.",
        sources=sources,
    )
    terms = deepcopy(old)
    terms["share_distributions"].append(event)
    terms["supersedes"] = run["natura_settlement_terms"]
    terms["status"] = "enat_undated_fraction_engineering_not_final_economic_admission"
    terms["sources"] += [x["pdf"] for x in sources.values()]
    write_json_atomic(root / "terms.json", terms)
    plan = dict(
        source_index=binding(source_root / "enat_index.json"),
        original_sources=sources,
        capital=[10_000_000, 1_000_000, 5_000_000],
        signs=[1, -1],
        scenarios=["unresolved_source", "primary", "provisioned_loan"],
        sessions="6 pre-effect +24 effect/following; no later RRRP/BRAV rename consumed",
        preferences="sin(axis*.31+localday*.003+head*.2), ENAT focus +/-4; frozen fixed risks beta1/idio.0004/market.0001, original .002/.001/.0005 calibration; no neural scores",
        costs="old bundled4bp; no extra B3 spot, corrected CDI/qualified loans/prior references; not final corporate pricing",
        terms=binding(root / "terms.json"),
        prior_completed_books_not_repeated=True,
        source_limits="ALSC unknown custody/fraction remains locked. ENAT auction date/price/payment and event-specific loans unknown. No issuer/loan aliases, history or accepted data amended.",
    )
    write_json_atomic(root / "plan.json", plan)
    store = Path(old["store"]["root"])
    isins = np.load(store / "isin_index.npy").tolist()
    names = np.arange(len(isins))
    name = isins.index(event["isin"])
    successor = isins.index(event["legs"][0]["successor_isin"])
    assert len(names) == 933 and dates[-1] <= np.datetime64("2024-12-30")
    effect = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
    rows = np.arange(effect - 6, effect + 24)
    credit = int(np.searchsorted(dates[rows], np.datetime64("2024-08-05")))
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
    inputs = inputs_on_axes(store, dates, names, rows)
    prior = np.load(store / "prior_reference_close.npy", mmap_mode="r")[rows]
    assert np.isfinite(prior[6, successor]) and np.isfinite(
        references[rows[6], successor]
    )
    reports = []
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
        frozen = PolicyData(
            base,
            np.ones(base.active.shape),
            np.full(base.active.shape, 0.0004),
            np.full(len(rows), 0.0001),
            cdi[rows - 1],
            prior,
        )
        for scenario in plan["scenarios"]:
            variant = deepcopy(terms)
            if scenario == "unresolved_source":
                variant["share_distributions"].pop()
            elif scenario == "provisioned_loan":
                variant["share_distributions"][-1]["legs"][0]["fractional_auction"][
                    "provision_loan_fractions"
                ] = True
            variant_path = root / f"terms_{scenario}.json"
            if not variant_path.exists():
                write_json_atomic(variant_path, variant)
            data = copy(frozen)
            data.inputs = apply_cash_calendar(
                apply_corporate_replay(base, variant, dates, sha256_file(variant_path)),
                dates,
                cdi,
                run["cash_calendar"]["sha256"],
            )
            assert data.static is frozen.static and data.references is frozen.references
            assert data.inputs.loan_reference_prices is base.loan_reference_prices
            model = CalibratedPolicy(
                Calibration(
                    np.zeros(3), np.ones(3), np.array([0.002, 0.001, 0.0005]), 0
                )
            )
            for capital in plan["capital"]:
                started = perf_counter()
                book = f"ENAT_{scenario}_{sign}_{capital}"
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
                                float(np.max(np.abs(actual_target.numpy() - target))),
                            )
                        fixed_row = data.step(
                            fixed,
                            torch.from_numpy(target),
                            day,
                            terminal=day == len(rows) - 1,
                        )
                        data.step(
                            adaptive, actual_target, day, terminal=day == len(rows) - 1
                        )
                        fixed_error = max(
                            fixed_error, abs(float(fixed_row["nav"]) - result.nav[day])
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
                    source_quantity_before=float(result.signed_shares[5, name]),
                    final_undelivered=float(result.undelivered_share_notional[-1]),
                    identical_intention_nav_error=fixed_error,
                    adaptive_nav_difference_bps=adaptive_error / capital * 1e4,
                    adaptive_target_difference=decision_error,
                    summary=summary,
                    seconds=perf_counter() - started,
                )
                reports.append(report)
                write_json_atomic(root / "completed.json", reports)
                print(
                    json.dumps({k: v for k, v in report.items() if k != "summary"}),
                    flush=True,
                )
                assert (
                    fixed_error < capital * 1e-8
                    and sign * report["source_quantity_before"] > 0
                )
    manifest = dict(
        status="enat_engineering_books_complete_pending_saved_qualification",
        audit_root=str(root),
        plan=binding(root / "plan.json"),
        terms=binding(root / "terms.json"),
        completed=binding(root / "completed.json"),
        name=name,
        successor=successor,
        effect=6,
        credit=credit,
        source_prior=float(prior[6, name]),
        successor_prior=float(prior[6, successor]),
        successor_loan_reference=float(references[rows[6], successor]),
        dates=[str(dates[rows[0]]), str(dates[rows[-1]])],
        seconds=perf_counter() - tick,
    )
    write_json_atomic(root / "manifest.json", manifest)
    run["enat_settlement_audit"] = binding(root / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
