"""Fifteen dated calendar scenarios; three qualified primary books are reused."""

from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.custody_fees import CustodyAssessment
from brazil_rv.execution.loan_contracts import LoanContracts
from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.execution.stateful_ledger import LedgerConfig
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    PolicyData,
    account_decision,
    exact_replay,
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


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    root = Path(run["root"]) / "january_calendar"
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8-sig"))
    assert not (root / "resolved_plan.json").exists()
    parents = []
    audit = bound_json(run["historical_cost_audit"])
    for case in bound_json(audit["completed"]):
        if case["start"] != "2016-10-24" or case["scenario"] != "primary":
            continue
        folder = Path(audit["audit_root"]) / case["book"]
        for scenario, changes in plan["variants"].items():
            parents.append(
                dict(
                    case=case,
                    scenario=scenario,
                    changes=changes,
                    book=binding(folder / "book.json"),
                    account=binding(folder / "account.npz"),
                )
            )
    assert len(parents) == 15
    write_json_atomic(
        root / "resolved_plan.json",
        dict(
            plan=binding(root / "plan.json"),
            parents=parents,
            sources={
                k: run[k]
                for k in (
                    "enat_settlement_terms",
                    "qualified_lending",
                    "loan_source_panels",
                    "cash_calendar",
                    "bova_loan_reference_audit",
                )
            },
            contract="Only one dated cash/delivery/accrual hypothesis changes; saved primary parents are not replayed.",
            historical_parent_default="custody_claim_fraction=1 was added after the historical parent; historical_cost_runtime qualifies those saved ordinary states without replay.",
            historical_runtime=run["historical_cost_runtime"],
        ),
    )
    shutil.copyfile(__file__, root / "executed.py")
    for folder in ("execution", "v2"):
        for path in (PROJECT / "research/src/brazil_rv" / folder).glob("*.py"):
            shutil.copyfile(path, root / f"executed_{folder}_{path.name}")
    terms, dates = load_corporate_replay(
        run["enat_settlement_terms"]["path"], run["enat_settlement_terms"]["sha256"]
    )
    store = Path(terms["store"]["root"])
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
    reports = []
    for parent in parents:
        started = perf_counter()
        case, scenario = parent["case"], parent["scenario"]
        meta = bound_json(parent["book"])
        cfg = dict(meta["provenance"]["config"])
        cfg["monthly_spot_tariffs"] = tuple(
            MonthlySpotTariff(**t) for t in cfg["monthly_spot_tariffs"]
        )
        cfg["custody_assessments"] = tuple(
            CustodyAssessment(**a) for a in cfg["custody_assessments"]
        )
        primary = LedgerConfig(**cfg)
        config = replace(primary, **parent["changes"])
        original_config = asdict(primary)
        original_config = {k: original_config[k] for k in meta["provenance"]["config"]}
        assert json.loads(json.dumps(original_config)) == meta["provenance"]["config"]
        first = int(np.searchsorted(dates, np.datetime64(meta["state_dates"][0])))
        rows = np.arange(first, first + case["sessions"])
        assert [str(x) for x in dates[rows]] == meta["state_dates"]
        parent_folder = Path(parent["book"]["path"]).parent
        if scenario == "extra_accrual":
            evidence = pl.read_parquet(parent_folder / "loan_charges.parquet")
            exposure = evidence.filter(
                pl.col("session") == meta["state_dates"].index("2017-01-24")
            )
        else:
            evidence = pl.read_parquet(parent_folder / "fills.parquet")
            selected_dates = next(iter(parent["changes"].values()))
            exposure = evidence.filter(
                pl.col("fill_session").is_in(
                    [meta["state_dates"].index(d) for d, _ in selected_dates]
                )
            )
        assert exposure.height, "unexposed calendar variant must be skipped"
        inputs = inputs_on_axes(store, dates, names, rows)
        scores = np.sin(
            names[None, :, None] * 0.31
            + np.arange(len(rows))[:, None, None] * 0.07
            + np.arange(5)[None, None, :] * 0.2
        ).astype(np.float32)
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
            initial_hedge_reference_price=hedge_close[first - 1],
        )
        frozen = PolicyData(
            base,
            np.ones(base.active.shape),
            np.full(base.active.shape, 0.0004),
            np.full(len(rows), 0.0001),
            cdi[rows - 1],
            np.load(store / "prior_reference_close.npy", mmap_mode="r")[rows],
        )
        data = copy(frozen)
        data.inputs = apply_cash_calendar(
            apply_corporate_replay(
                base, terms, dates, run["enat_settlement_terms"]["sha256"]
            ),
            dates,
            cdi,
            run["cash_calendar"]["sha256"],
        )
        assert data.static is frozen.static and data.references is frozen.references
        model = CalibratedPolicy(
            Calibration(np.zeros(3), np.ones(3), np.array([0.002, 0.001, 0.0005]), 0)
        )
        book = f"{scenario}_{case['capital']}"
        snapshots = []
        phase = "independent"
        original_accrue = LoanContracts.accrue

        def captured_accrue(loans, day, date, **kwargs):
            if not kwargs.get("extra_interval"):
                return original_accrue(loans, day, date, **kwargs)

            def state():
                return {
                    key: getattr(loans, key).detach().numpy().tolist()
                    if isinstance(getattr(loans, key), torch.Tensor)
                    else getattr(loans, key).tolist()
                    for key in (
                        "name",
                        "opened",
                        "extra_accrual_days",
                        "value_lag",
                        "return_day",
                        "return_requested",
                        "accrual_end",
                        "principal",
                        "annual_rate",
                        "fee_rate",
                        "fee_growth",
                        "rent_due",
                        "fees_due",
                        "root",
                        "root_fees",
                        "minimum",
                        "started",
                        "minimum_credit",
                    )
                }

            before = state()
            value = original_accrue(loans, day, date, **kwargs)
            snapshots.append(
                dict(
                    phase=phase,
                    day=day,
                    date=str(date),
                    before=before,
                    after=state(),
                    fee_multiplier=loans.fee_multiplier,
                    expense=[float(x.sum()) for x in value],
                )
            )
            return value

        LoanContracts.accrue = captured_accrue
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
                policy="frozen_parent_calendar_variant",
                scenario="base",
                config=asdict(config),
                parent=parent,
            ),
        )
        fixed, adaptive = (
            data.initial_account(0, config),
            data.initial_account(0, config),
        )
        fixed_error = adaptive_error = decision_error = 0.0
        details = []
        with torch.no_grad():
            for day, target in enumerate(targets):
                actual = account_decision(data, model, adaptive, day)
                if day < len(rows) - 1:
                    decision_error = max(
                        decision_error, float(np.max(np.abs(actual.numpy() - target)))
                    )
                phase = "identical_intention"
                row = data.step(
                    fixed, torch.from_numpy(target), day, terminal=day == len(rows) - 1
                )
                phase = "adaptive"
                data.step(adaptive, actual, day, terminal=day == len(rows) - 1)
                fixed_error = max(fixed_error, abs(float(row["nav"]) - result.nav[day]))
                adaptive_error = max(
                    adaptive_error, abs(float(adaptive.nav) - result.nav[day])
                )
                details.append(
                    dict(
                        day=day,
                        funding_cash=float(fixed.funding_cash),
                        funding_restricted=float(fixed.funding_restricted),
                        components=row["execution_charges"].tolist(),
                        adjustment=row["spot_invoice_adjustment"].tolist(),
                        cost=float(row["cost"]),
                        free_income=float(row["free_cash_income"]),
                        proceeds_income=float(row["short_proceeds_interest"]),
                        debit_financing=float(row["debit_financing"]),
                        interest=float(row["interest"]),
                        physical_error=float(
                            np.max(
                                np.abs(
                                    row["physical_custody"].numpy()
                                    - result.physical_custody[day]
                                )
                            )
                        ),
                    )
                )
        LoanContracts.accrue = original_accrue
        write_json_atomic(root / book / "accrual_snapshots.json", snapshots)
        write_json_atomic(root / book / "funding_and_costs.json", details)
        write_json_atomic(
            root / book / "loan_cash_payments.json",
            [asdict(p) for p in result.loan_cash_payments],
        )
        summary, _ = book_summary(data, result, previous, 0, 0)
        with np.load(parent["account"]["path"]) as old:
            changed = np.flatnonzero(
                (result.nav != old["nav"])
                | (result.free_cash != old["free_cash"])
                | (result.restricted_cash != old["restricted_cash"])
                | np.any(result.physical_custody != old["physical_custody"], axis=1)
            )
            first_change = int(changed[0]) if len(changed) else len(rows)
            np.testing.assert_array_equal(
                result.nav[:first_change], old["nav"][:first_change]
            )
            np.testing.assert_array_equal(
                targets[: first_change + 1], old["targets"][: first_change + 1]
            )
            difference = (result.nav - old["nav"]) / case["capital"] * 1e4
        report = dict(
            book=book,
            scenario=scenario,
            start=case["start"],
            parent=parent,
            capital=case["capital"],
            sessions=len(rows),
            identical_intention_nav_error=fixed_error,
            adaptive_nav_difference_bps=adaptive_error / case["capital"] * 1e4,
            adaptive_target_difference=decision_error,
            final_contrast_bps=float(difference[-1]),
            max_contrast_bps=float(np.max(np.abs(difference))),
            accrual_snapshots=len(snapshots),
            changed_dates=len(changed),
            first_difference=first_change,
            cielo_loan_cash_quantity=sum(
                p.quantity
                for p in result.loan_cash_payments
                if p.security_index == isins.index("BRCIELACNOR3")
            ),
            summary=summary,
            seconds=perf_counter() - started,
        )
        reports.append(report)
        write_json_atomic(root / "completed.json", reports)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("summary", "parent")}
            ),
            flush=True,
        )
        assert fixed_error < 1e-6
    write_json_atomic(
        root / "manifest.json",
        dict(
            status="complete_pending_saved_qualification",
            audit_root=str(root),
            plan=binding(root / "resolved_plan.json"),
            completed=binding(root / "completed.json"),
            seconds=perf_counter() - tick,
            summed_case_seconds=sum(c["seconds"] for c in reports),
        ),
    )


if __name__ == "__main__":
    main()
