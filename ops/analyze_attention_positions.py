"""Saved F10 mark/fill and exposure attribution for the frozen forecast contrast."""

import json
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    book = bound_json(binding(path / "book.json"))
    with np.load(path / "account.npz") as z:
        a = {
            k: z[k]
            for k in ["signed_shares", "mark_price", "nav", "start_nav", "targets"]
        }
    value = a["signed_shares"] * np.nan_to_num(a["mark_price"])
    marked = value - np.vstack([np.zeros((1, value.shape[1])), value[:-1]])
    for fill in pl.read_parquet(path / "fills.parquet").iter_rows(named=True):
        if fill["purpose"] != "hedge":
            marked[fill["fill_session"], fill["security_index"]] -= fill[
                "gross_notional"
            ] * (1 if fill["side"] == "buy" else -1)
    marked = marked / a["start_nav"][:, None] * 1e4
    residual = np.asarray(book["daily"]["equity_gross_bps"]) - marked.sum(1)
    return book, marked, residual, value / a["nav"][:, None], a["targets"]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    plan = bound_json(run["scaling_investigation"])
    refit = bound_json(run["stage_c_refit_plan"])
    names = np.load(Path(refit["store"]["root"]) / "isin_index.npy").tolist()
    root = Path(run["scaling_portfolio_decomposition"]["path"]).parent
    output = root.parent / "position_attribution.json"
    assert not output.exists()
    (root.parent / "executed_positions.py").write_bytes(Path(__file__).read_bytes())
    results = []
    for arm in plan["arms"]:
        path = root / "books" / arm / "F10"
        old, a, ar, aw, at = read(path / "old_forecast_old_risk")
        new, b, br, bw, bt = read(path / "new_forecast_old_risk")
        assert old["state_dates"] == new["state_dates"]
        delta = b - a
        contribution = delta.mean(0)
        weight_change = np.mean(np.abs(bw - aw), axis=0)
        daily = (
            np.asarray(new["daily"]["net_excess_bps"]) - old["daily"]["net_excess_bps"]
        )
        rows = [
            dict(
                isin=names[i],
                mark_fill_delta_bps_day=float(contribution[i]),
                mean_absolute_weight_difference=float(weight_change[i]),
                old_mean_weight=float(aw[:, i].mean()),
                new_mean_weight=float(bw[:, i].mean()),
            )
            for i in np.argsort(np.abs(contribution))[::-1]
        ]
        dates = old["dates"]
        monthly = []
        for month in sorted({d[:7] for d in dates}):
            take = np.array([d.startswith(month) for d in dates])
            monthly.append(
                dict(
                    month=month,
                    sessions=int(take.sum()),
                    net_delta_bps_day=float(daily[take].mean()),
                )
            )
        components = {
            k: new["summary"]["mean"][k] - old["summary"]["mean"][k]
            for k in [
                "net_excess_bps",
                "equity_gross_bps",
                "hedge_gross_bps",
                "trading_cost_bps",
                "borrow_bps",
                "interest_bps",
                "custody_bps",
                "gross",
                "signed_net",
                "turnover",
            ]
        }
        results.append(
            dict(
                arm=arm,
                fold="F10",
                old=binding(path / "old_forecast_old_risk/book.json"),
                new=binding(path / "new_forecast_old_risk/book.json"),
                component_deltas=components,
                mark_fill_delta_bps_day=float(contribution.sum()),
                unassigned_cash_claim_delta_bps_day=float((br - ar).mean()),
                mean_absolute_weight_distance=float(np.abs(bw - aw).sum(1).mean()),
                mean_absolute_target_distance=float(np.abs(bt - at).sum(1).mean()),
                largest_daily_differences=[
                    dict(
                        date=dates[i],
                        net_delta_bps=float(daily[i]),
                        largest_mark_fill=[
                            dict(isin=names[j], bps=float(delta[i, j]))
                            for j in np.argsort(abs(delta[i]))[-5:][::-1]
                        ],
                        cash_claim_residual_bps=float(br[i] - ar[i]),
                    )
                    for i in np.argsort(abs(daily))[-12:][::-1]
                ],
                monthly=monthly,
                all_names=rows,
            )
        )
    report = dict(
        plan=run["scaling_investigation"],
        results=results,
        scope="Both arms, all 933 names, entire F10. This period was selected diagnostically because the already observed reversal is largest there, not as an unbiased test.",
        definition="Mark-and-fill contribution is daily change in signed marked stock value minus signed fill notional, divided by that path's start NAV. Corporate transfers can move contributions between ISINs. Cash distributions and claim-value changes remain an explicitly unassigned residual. This is a saved-book decomposition, not an independent per-security total-return oracle or causal data-repair attribution.",
    )
    write_json_atomic(output, report)
    run["scaling_position_attribution"] = binding(output)
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ["all_names", "largest_daily_differences"]
                }
                | {"top_names": r["all_names"][:8]}
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
