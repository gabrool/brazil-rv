"""Independent dated flow and Decimal accrual checks of saved calendar books."""

from collections import defaultdict
from decimal import Decimal, localcontext
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "january_calendar"
    out = root / "qualification"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    audit = bound_json(binding(root / "manifest.json"))
    cases = bound_json(audit["completed"])
    store = Path(bound_json(run["enat_settlement_terms"])["store"]["root"])
    dates = np.load(store / "date_index.npy").astype("datetime64[D]")
    qs = np.load(store / "action_shares_per_prior_share.npy", mmap_mode="r")
    resolved = np.load(store / "action_session_resolved.npy", mmap_mode="r")
    with np.load(bound_json(run["cash_calendar"])["panel"]["path"]) as z:
        cdi = z["cdi_returns"]
    errors = defaultdict(float)
    cells = fills_count = days = cohorts = 0
    checks = []

    def error(name, expected, actual):
        errors[name] = max(
            errors[name],
            float(np.max(np.abs(np.asarray(expected) - actual), initial=0)),
        )

    for case in cases:
        folder = root / case["book"]
        meta = json.loads((folder / "book.json").read_text())
        for file, digest in meta["files"].items():
            assert sha256_file(folder / file) == digest
        assert not meta["legacy_slot_diagnostics_applicable"]
        cfg = meta["provenance"]["config"]
        with np.load(folder / "account.npz") as z:
            a = {k: z[k] for k in z.files}
        cells += sum(x.size for x in a.values())
        rows = json.loads((folder / "funding_and_costs.json").read_text())
        fills = pl.read_parquet(folder / "fills.parquet").to_dicts()
        fills_count += len(fills)
        by_day = defaultdict(list)
        for f in fills:
            by_day[f["fill_session"]].append(f)
        nav = (
            a["free_cash"]
            + a["restricted_cash"]
            + a["hedge_restricted_cash"]
            + a["unsettled_cash"]
            + a["receivables"]
            - a["payables"]
            + (a["signed_shares"] * np.nan_to_num(a["mark_price"])).sum(1)
            + a["hedge_signed_shares"] * np.nan_to_num(a["hedge_mark_price"])
            - a["loan_liability"]
            - a["custody_liability"]
        )
        error("saved_nav", nav, a["nav"])
        cash_lags = dict(cfg["spot_cash_lag_overrides"])
        delivery_lags = dict(cfg["spot_delivery_lag_overrides"])
        pending_cash = []
        economic = np.zeros(934)
        physical = np.zeros(934)
        flows = defaultdict(lambda: np.zeros(934))
        selected = []
        for day, row in enumerate(rows):
            date = meta["state_dates"][day]
            index = int(np.searchsorted(dates, np.datetime64(date)))
            q = np.r_[np.where(resolved[index], qs[index], 1), 1.0].astype(float)
            assert np.all(q > 0)
            economic *= q
            physical *= q
            for due in flows:
                flows[due] *= q
            physical += flows.pop(day, np.zeros(934))
            flow = Decimal(0)
            for f in by_day[day]:
                name = 933 if f["purpose"] == "hedge" else f["security_index"]
                quantity = f["quantity"] * (1 if f["side"] == "buy" else -1)
                cover = min(max(quantity, 0), max(-economic[name], 0))
                opening = max(-quantity - max(economic[name], 0), 0)
                economic[name] += quantity
                # D0 new-loan receipt; ordinary bought cover and loan return
                # share the physical date, so their physical flows net to zero.
                flows[day + delivery_lags.get(date, 3)][name] += quantity - cover
                physical[name] += opening
                flow += Decimal(str(f["gross_notional"])) * (
                    1 if f["side"] == "sell" else -1
                ) - Decimal(str(f["cost"]))
            pending_cash.append((day + cash_lags.get(date, 3), flow))
            pending_cash = [(d, v) for d, v in pending_cash if d > day]
            error(
                "decimal_spot_cash_queue",
                float(sum((v for _, v in pending_cash), Decimal(0))),
                a["unsettled_cash"][day],
            )
            error("independent_physical_flow", physical, a["physical_custody"][day])
            error(
                "fill_share_conservation",
                economic,
                np.r_[a["signed_shares"][day], a["hedge_signed_shares"][day]],
            )
            error("identical_intention_physical", 0, row["physical_error"])
            cash = a["free_cash"][day - 1] if day else case["capital"]
            restricted = (
                (a["restricted_cash"][day - 1] + a["hedge_restricted_cash"][day - 1])
                if day
                else 0
            )
            error(
                "prior_funding",
                [cash, restricted],
                np.array([row["funding_cash"], row["funding_restricted"]]),
            )
            error("income", (cash + restricted) * cdi[index], row["interest"])
            if date in ("2017-01-20", "2017-01-23", "2017-01-24"):
                selected.append(
                    dict(
                        date=date,
                        fills=len(by_day[day]),
                        cash_date=meta["state_dates"][day + cash_lags.get(date, 3)],
                        delivery_date=meta["state_dates"][
                            day + delivery_lags.get(date, 3)
                        ],
                    )
                )
            days += 1
        snapshots = json.loads((folder / "accrual_snapshots.json").read_text())
        increments = []
        for shot in snapshots:
            b = {k: np.array(v) for k, v in shot["before"].items()}
            post = {k: np.array(v) for k, v in shot["after"].items()}
            day = shot["day"]
            age = day - b["opened"]
            alive = (
                ((b["accrual_end"] < 0) | (b["accrual_end"] > day))
                & ((b["return_day"] < 0) | (b["return_day"] > day))
                & (age >= b["value_lag"])
            )
            active = ((b["accrual_end"] < 0) | (day <= b["accrual_end"])) & (
                age >= b["value_lag"]
            )
            rent_active = active & ~(
                (b["value_lag"] == 0)
                & (b["return_day"] == day)
                & (b["accrual_end"] < 0)
            )
            fee_active = active & ~((b["return_requested"] == b["opened"]) & (age == 0))
            rents, fees, extra_rents, extra_fees = [], [], [], []
            with localcontext() as ctx:
                ctx.prec = 42
                for i, principal in enumerate(b["principal"]):
                    p = Decimal(str(principal))
                    rate = 1 + Decimal(str(b["annual_rate"][i]))
                    power = (
                        Decimal(
                            int(age[i] + b["extra_accrual_days"][i] - b["value_lag"][i])
                        )
                        / 252
                    )
                    daily = rate ** (Decimal(1) / 252) - 1
                    regular = p * rate**power * daily if rent_active[i] else Decimal(0)
                    extra = (
                        p * rate ** (power + Decimal(1) / 252) * daily
                        if alive[i]
                        else Decimal(0)
                    )
                    rents.append(float(regular + extra))
                    extra_rents.append(float(extra))
                    one_fee, additional_fee = [], []
                    for j in range(2):
                        daily_fee = (1 + Decimal(str(b["fee_rate"][i, j]))) ** (
                            Decimal(1) / 252
                        ) - 1
                        growth = Decimal(str(b["fee_growth"][i, j]))
                        first = (
                            p
                            * growth
                            * daily_fee
                            * Decimal(str(shot["fee_multiplier"]))
                            if fee_active[i]
                            else Decimal(0)
                        )
                        next_growth = growth * (1 + daily_fee if fee_active[i] else 1)
                        second = (
                            p
                            * next_growth
                            * daily_fee
                            * Decimal(str(shot["fee_multiplier"]))
                            if alive[i]
                            else Decimal(0)
                        )
                        one_fee.append(float(first + second))
                        additional_fee.append(float(second))
                    fees.append(one_fee)
                    extra_fees.append(additional_fee)
            rent = np.array(rents)
            fee = np.array(fees)
            error("decimal_contract_rent", rent, post["rent_due"] - b["rent_due"])
            error("decimal_contract_fee", fee, post["fees_due"] - b["fees_due"])
            np.testing.assert_array_equal(
                post["extra_accrual_days"], b["extra_accrual_days"] + alive
            )
            for key in (
                "opened",
                "return_day",
                "root",
                "minimum",
                "minimum_credit",
                "principal",
                "annual_rate",
            ):
                np.testing.assert_array_equal(b[key], post[key])
            old_min = np.maximum(b["minimum"] * b["started"] - b["root_fees"], 0).sum()
            root_fees = b["root_fees"].copy()
            np.add.at(root_fees, b["root"], fee.sum(1))
            started = b["started"].copy()
            started[np.unique(b["root"][fee_active])] = True
            new_min = np.maximum(b["minimum"] * started - root_fees, 0).sum()
            error(
                "decimal_accrual_expense",
                [rent.sum(), fee.sum() + new_min - old_min],
                np.array(shot["expense"]),
            )
            error("root_fee_rollforward", root_fees, post["root_fees"])
            # Direct extra-fee expense can be absorbed by the already accrued minimum.
            without_extra = b["root_fees"].copy()
            np.add.at(without_extra, b["root"], (fee - np.array(extra_fees)).sum(1))
            without_min = np.maximum(b["minimum"] * started - without_extra, 0).sum()
            increments.append(
                dict(
                    phase=shot["phase"],
                    live_cohorts=int(alive.sum()),
                    original_roots=len(b["minimum"]),
                    extra_rent=sum(extra_rents),
                    extra_raw_fee=float(np.sum(extra_fees)),
                    extra_fee_expense=float(np.sum(extra_fees) + new_min - without_min),
                )
            )
            cohorts += len(rent)
        with np.load(case["parent"]["account"]["path"]) as old:
            first = case["first_difference"]
            np.testing.assert_array_equal(a["nav"][:first], old["nav"][:first])
            np.testing.assert_array_equal(
                a["targets"][: first + 1], old["targets"][: first + 1]
            )
        checks.append(
            dict(
                book=case["book"],
                selected_value_dates=selected,
                accrual_increments=increments,
            )
        )
    report = dict(
        status="qualified_bounded_calendar_hypotheses_not_historical_fact_or_model_profit",
        books=len(cases),
        days=days,
        cells=cells,
        fills=fills_count,
        accrual_cohort_checks=cohorts,
        errors=dict(errors),
        seconds=perf_counter() - tick,
        cases=checks,
    )
    write_json_atomic(out / "report.json", report)
    assert max(errors.values()) < 1e-6, dict(errors)
    write_json_atomic(
        out / "manifest.json",
        dict(
            report=binding(out / "report.json"),
            audit=binding(root / "manifest.json"),
            status=report["status"],
        ),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    main()
