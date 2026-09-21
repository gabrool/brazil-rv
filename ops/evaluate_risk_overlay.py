"""Evaluate causal volatility-targeting overlays on saved daily books; no refit.

For each saved ``book.json`` (continuous or fold account) and each rule, the
overlay multiplies the book's CDI-excess return by ``clip(target / trailing vol,
floor, 1)`` where the trailing volatility uses only sessions strictly before the
decision day. Own-return volatility is always evaluated; an optional external
daily series (for example BOVA11 returns, columns ``date,return``) adds a
market-stress signal and the elementwise minimum of both. Rescaling turnover is
charged at ``--cost-bps``.

    uv run --project research python ops/evaluate_risk_overlay.py \
        --book <run>/continuous/C6_flexible_net/book.json \
        --market-csv D:/quant-data/b3/interim/bova11_daily_returns.csv \
        --window 20 60 --target 0.08 0.10 0.12 --floor 0.25 \
        --output docs/v2_risk_overlay_readout.json

This is an ex-post linear approximation on frozen forecasts and frozen books.
It selects nothing; report every rule. Confirm a candidate rule by rerunning the
ledger with a daily gross cap equal to the scale times the planned cap.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.risk_overlay import (
    combine_scales,
    evaluate_overlay,
    trailing_external_volatility,
    trailing_volatility,
    volatility_target_scale,
)


def read_market(path):
    dates, returns = [], []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        closes = "return" not in reader.fieldnames and "close" in reader.fieldnames
        previous = None
        for row in reader:
            dates.append(row["date"])
            if closes:
                close = float(row["close"])
                returns.append(np.nan if previous is None else close / previous - 1.0)
                previous = close
            else:
                returns.append(float(row["return"]))
    return np.asarray(dates, dtype="datetime64[D]"), np.asarray(returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=Path, action="append", required=True)
    parser.add_argument("--market-csv", type=Path)
    parser.add_argument("--window", type=int, nargs="+", default=[20, 60])
    parser.add_argument("--target", type=float, nargs="+", default=[0.08, 0.10, 0.12])
    parser.add_argument("--floor", type=float, default=0.25)
    parser.add_argument("--cost-bps", type=float, default=4.0)
    parser.add_argument("--block", type=int, default=40)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    market = read_market(args.market_csv) if args.market_csv else None
    report = {"books": {}, "rules": [], "cost_bps": args.cost_bps, "floor": args.floor}
    for path in args.book:
        book = json.loads(path.read_text(encoding="utf-8"))
        daily = {k: np.asarray(v, dtype=np.float64) for k, v in book["daily"].items()}
        dates = np.asarray(book["dates"], dtype="datetime64[D]")
        own = daily["net_excess_bps"] / 1e4
        results = {}
        for window in args.window:
            own_vol = trailing_volatility(own, window)
            market_vol = (
                trailing_external_volatility(dates, market[0], market[1], window)
                if market is not None
                else None
            )
            for target in args.target:
                signals = {
                    "own": volatility_target_scale(own_vol, target, floor=args.floor)
                }
                if market_vol is not None:
                    signals["market"] = volatility_target_scale(
                        market_vol, target, floor=args.floor
                    )
                    signals["min_own_market"] = combine_scales(
                        signals["own"], signals["market"]
                    )
                for signal, scale in signals.items():
                    label = f"{signal}_w{window}_t{target:g}"
                    results[label] = evaluate_overlay(
                        daily,
                        scale,
                        cost_bps=args.cost_bps,
                        block=args.block,
                        draws=args.draws,
                    )
                    if label not in report["rules"]:
                        report["rules"].append(label)
        report["books"][str(path)] = {
            "sha256": sha256_file(path),
            "sessions": len(dates),
            "first_date": str(dates[0]),
            "last_date": str(dates[-1]),
            "results": results,
        }
        base = next(iter(results.values()))["base"]
        print(f"\n{path}")
        print(
            f"  base: mean excess {base['mean_excess_bps']:.3f} bps/day, "
            f"vol {base['annualized_volatility']:.3%}, Sharpe0 {base['sharpe_zero_rate']:.3f}, "
            f"drawdown {base['maximum_drawdown']:.2%}"
        )
        for label, result in results.items():
            overlay, paired = result["overlay"], result["paired"]
            print(
                f"  {label:>28}: excess {overlay['mean_excess_bps']:.3f} "
                f"vol {overlay['annualized_volatility']:.3%} "
                f"Sharpe0 {overlay['sharpe_zero_rate']:.3f} "
                f"(delta {result['sharpe_zero_rate_delta']:+.3f}, 95% "
                f"{paired['sharpe_zero_rate_delta_interval']}) "
                f"drawdown {overlay['maximum_drawdown']:.2%} "
                f"mean scale {result['mean_scale']:.3f}"
            )
    if args.output is not None:
        write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
