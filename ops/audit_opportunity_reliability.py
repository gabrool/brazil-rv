"""Registered post-hoc shrinkage/compact-context diagnostic; no neural fits."""

import argparse
import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.opportunity_research import bound, checked, fit_ridge
from brazil_rv.v2.portfolio_program import read
from brazil_rv.v2.portfolio_readouts import interval
from brazil_rv.v2.research_rounds import _git_identity

COMPACT = (
    "recent_market_log_return",
    "median_raw_daily_volatility",
    "raw_cross_sectional_return_dispersion",
    "shadow_payoff_bps",
    "shadow_volatility_bps",
    "mean_seed_disagreement",
    "mean_horizon_disagreement",
)
PENALTIES = (1.0, 10.0, 100.0, 1000.0, 10000.0, None)


def predict(x, valid, y, blocks, first):
    result = np.full_like(y, np.nan)
    baseline = np.full_like(y, np.nan)
    records = {}
    for fold, rows in blocks.items():
        if int(fold[1:]) < 5:
            continue
        cutoff = int(rows[0])
        available = np.arange(first, cutoff - 5)
        selection = available[-126:]
        training = available[available + 5 < selection[0]]
        selection = selection[np.isfinite(y[selection]).all(1)]
        trials = []
        for strength in PENALTIES:
            if strength is None:
                p = np.broadcast_to(np.nanmean(y[training], axis=0), y.shape)
            else:
                p, _, _ = fit_ridge(x, valid, y, training, strength)
            trials.append(np.mean((p[selection] - y[selection]) ** 2, axis=0))
        # Prefer the simpler/constant candidate on an exact selection-loss tie.
        selected = len(PENALTIES) - 1 - np.argmin(trials[::-1], axis=0)
        means = np.nanmean(y[available], axis=0)
        baseline[rows] = means
        for index in np.unique(selected):
            strength = PENALTIES[index]
            p = (
                np.broadcast_to(means, y.shape)
                if strength is None
                else fit_ridge(x, valid, y, available, strength)[0]
            )
            for head in np.flatnonzero(selected == index):
                result[rows, head] = p[rows, head]
        records[fold] = {
            "first_decision": cutoff,
            "last_origin": int(available[-1]),
            "selected_penalty": [PENALTIES[i] for i in selected],
            "selection_mse": np.asarray(trials).tolist(),
            "fit_dates": len(available),
        }
    return result, baseline, records


def run(root):
    code = _git_identity()
    output = root / "reliability_followup"
    output.mkdir(exist_ok=False)
    design = read(root / "frozen_design.json")
    registration = (
        Path(__file__).resolve().parents[1]
        / "research/preregistrations/v2_opportunity_portfolios.md"
    )
    record = {
        "implementation": code,
        "registration": bound(registration),
        "original_design": bound(root / "frozen_design.json"),
        "post_hoc": True,
        "penalties": PENALTIES,
        "compact_features": COMPACT,
        "arms": {},
        "financial_survivors": [],
    }
    for arm, entry in design["arms"].items():
        with np.load(checked(entry["prepared"]), allow_pickle=False) as p:
            x, valid, y, dates = (
                p[n].copy() for n in ("common", "common_valid", "target", "dates")
            )
        y = y[:, :2]
        blocks = {f: np.asarray(rows) for f, rows in entry["blocks"].items()}
        first = int(blocks["F1"][0])
        arms = {}
        for label in ("common_context", "compact_context"):
            if label == "compact_context":
                indices = [entry["context_names"].index(n) for n in COMPACT]
                values = np.column_stack((x[:, indices], valid[:, indices]))
                mask = np.column_stack(
                    (valid[:, indices], np.ones_like(valid[:, indices]))
                )
            else:
                values, mask = x, valid
            p, base, fits = predict(values, mask, y, blocks, first)
            path = output / f"{arm}_{label}.npz"
            np.savez_compressed(path, prediction=p, unconditional=base)
            result = {"forecasts": bound(path), "fits": fits, "heads": {}}
            for head, name in enumerate(("shadow_payoff", "market_excess")):
                periods = {}
                for period, lo, hi in (
                    ("screen_2020_2021", "2020-01-01", "2022-01-01"),
                    ("later_2022_2024", "2022-01-01", "2025-01-01"),
                ):
                    within = (dates >= np.datetime64(lo)) & (dates < np.datetime64(hi))
                    last = int(np.flatnonzero(within)[-1])
                    use = (
                        within
                        & np.isfinite(p[:, head])
                        & np.isfinite(y[:, head])
                        & (np.arange(len(dates)) + 5 <= last)
                    )
                    gain = (
                        (base[use, head] - y[use, head]) ** 2
                        - (p[use, head] - y[use, head]) ** 2
                    ) * 1e8
                    periods[period] = {
                        "conditional_rmse_bps": float(
                            np.sqrt(np.mean((p[use, head] - y[use, head]) ** 2)) * 1e4
                        ),
                        "unconditional_rmse_bps": float(
                            np.sqrt(np.mean((base[use, head] - y[use, head]) ** 2))
                            * 1e4
                        ),
                        "improvement_bps_squared": {
                            str(b): interval([gain], block_length=b)
                            for b in (20, 40, 60)
                        },
                    }
                admitted = all(
                    v["improvement_bps_squared"]["40"]["lower_95"] > 0
                    for v in periods.values()
                )
                result["heads"][name] = {
                    "periods": periods,
                    "passes_financial_replay_gate": admitted,
                    "constant_selected_folds": sum(
                        r["selected_penalty"][head] is None for r in fits.values()
                    ),
                }
                if admitted:
                    record["financial_survivors"].append(
                        {"arm": arm, "features": label, "head": name}
                    )
            arms[label] = result
        record["arms"][arm] = arms
    write_json_atomic(output / "results.json", record)
    print(
        json.dumps(
            {
                "output": str(output),
                "financial_survivors": record["financial_survivors"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
