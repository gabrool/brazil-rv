"""Read sealed policy artifacts; quantify mechanisms without rerunning a backtest."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import torch

from brazil_rv.execution.allocation import AllocationConfig, allocate


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def account_series(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as account:
        nav = account["nav"]
        stock = account["signed_shares"] * np.nan_to_num(account["mark_price"])
        stock /= nav[:, None]
        hedge = account["hedge_signed_shares"] * account["hedge_mark_price"] / nav
        series = {
            "stock_net": stock.sum(1),
            "hedge_weight": hedge,
            "hedge_at_short_cap": hedge < -0.59,
        }
        if "targets" in account:
            targets = account["targets"]
            series.update(
                planned_gross=np.abs(targets).sum(1),
                planned_hedge=targets[:, -1],
                planned_hedge_at_short_cap=targets[:, -1] < -0.5999,
                planned_gross_at_cap=np.abs(targets).sum(1) > 2.2499,
                planned_stock_at_cap=(np.abs(targets[:, :-1]) > 0.04999).sum(1),
            )
        return series


def intercept_demonstration():
    """Synthetic identical stocks: isolate the common stock/hedge preference."""
    n = 60
    config = AllocationConfig()
    cases = {}
    for net_cap in (config.net_cap, 0.0):
        for label, stock_mu, hedge_mu in (
            ("zero_alpha", 0.0, 0.0),
            ("stock_intercept_only", 0.0008, 0.0),
            ("common_return_for_stocks_and_hedge", 0.0008, 0.0008),
        ):
            preference = torch.tensor([stock_mu] * n + [hedge_mu], dtype=torch.float64)
            weights = allocate(
                preference,
                torch.zeros(n + 1, dtype=torch.float64),
                beta=np.ones(n + 1),
                idiosyncratic_variance=np.r_[np.full(n, 0.02**2), 1e-8],
                market_variance=0.015**2,
                daily_borrow=np.full(n + 1, 0.000025),
                lower=torch.tensor([-0.05] * n + [-0.6], dtype=torch.float64),
                upper=torch.tensor([0.05] * n + [0.6], dtype=torch.float64),
                config=replace(config, net_cap=net_cap),
            ).numpy()
            cases[f"net_cap_{net_cap}/{label}"] = {
                "gross": float(np.abs(weights).sum()),
                "stock_net": float(weights[:-1].sum()),
                "hedge": float(weights[-1]),
                "net_and_beta": float(weights.sum()),
            }
    return {
        "interpretation": "mechanism demonstration, not historical P&L or a fitted strategy",
        "stocks": n,
        "stock_idiosyncratic_volatility": 0.02,
        "market_volatility": 0.015,
        "daily_borrow_bps": 0.25,
        "cases": cases,
    }


def blocked_interval(arrays, length):
    """Descriptive circular blocks preserve marginal weight of boundary days."""
    rng = np.random.default_rng(20260914)
    draws = np.zeros(10_000)
    for values in arrays:
        size = len(values)
        starts = rng.integers(0, size, (len(draws), (size + length - 1) // length))
        widths = np.full(starts.shape[1], length)
        widths[-1] = size - length * (len(widths) - 1)
        prefix = np.r_[0.0, np.cumsum(np.r_[values, values[:length]])]
        draws += (prefix[starts + widths] - prefix[starts]).sum(1)
    draws /= sum(map(len, arrays))
    return {
        "mean": float(np.concatenate(arrays).mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def noncircular_expected_mean(arrays, length=20):
    """Exact expected mean under the original finite-block sampling scheme."""
    total = 0.0
    for values in arrays:
        starts = np.arange(len(values) - length + 1)
        widths = np.full((len(values) + length - 1) // length, length)
        widths[-1] = len(values) - length * (len(widths) - 1)
        prefix = np.r_[0.0, np.cumsum(values)]
        total += sum(
            (prefix[starts + width] - prefix[starts]).mean() for width in widths
        )
    return float(total / sum(map(len, arrays)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory_path = args.archives / "inventory.json"
    inventory = {r["path"]: r for r in json.loads(inventory_path.read_text())}
    books, accounts, policies, checked = {}, {}, {}, {}
    # Stream the archive once. Never unpack, overwrite, or modify source files.
    with tarfile.open(args.archives / "evidence.tar.gz", "r|gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            is_book = (
                name.startswith("books/")
                and "/base/" in name
                and name.endswith(("/account.npz", "/book.json"))
            )
            is_policy = name.startswith("policies/") and name.endswith(
                "/run_manifest.json"
            )
            if not (is_book or is_policy):
                continue
            payload = archive.extractfile(member).read()
            sha = hashlib.sha256(payload).hexdigest()
            if sha != inventory[name]["sha256"]:
                raise ValueError(f"sealed member hash differs: {name}")
            checked[name] = sha
            key = name.rsplit("/", 1)[0]
            if is_policy:
                policies[name] = json.loads(payload)
            elif name.endswith("account.npz"):
                accounts[key] = account_series(payload)
            else:
                books[key] = json.loads(payload)
    result = {}
    for key, book in books.items():
        source = key
        if "equivalent_book" in book:
            source = "books/" + book["equivalent_book"]["path"].split("/books/")[1]
            source = source.removesuffix("/book.json")
        trim = book["burn_in_sessions"]
        series = accounts[source]
        result[key] = {
            "summary": book["summary"],
            "allocation_mean": {k: float(v[trim:].mean()) for k, v in series.items()},
            "allocation_sessions": len(series["hedge_weight"][trim:]),
            "account_source": source,
        }
        if result[key]["allocation_sessions"] != book["summary"]["sessions"]:
            raise ValueError("account and reported dates differ")
    training = {}
    for arm in ("S0", "TE_all", "C6"):
        trials = {k: v for k, v in policies.items() if f"/{arm}/" in k}
        records = list(trials.values())
        selection_gains = [
            next(
                h["utility_bps"]
                for h in r["history"]
                if h["epoch"] == r["selected_epoch"]
            )
            - r["history"][0]["utility_bps"]
            for r in records
        ]
        training[arm] = {
            "fits": len(records),
            "epoch_zero_selected": sum(r["selected_epoch"] == 0 for r in records),
            "selected_selection_utility_below_cash_count": sum(
                next(
                    h["utility_bps"]
                    for h in r["history"]
                    if h["epoch"] == r["selected_epoch"]
                )
                < 0
                for r in records
            ),
            "mean_selected_minus_epoch_zero_selection_utility_bps": float(
                np.mean(selection_gains)
            ),
            "median_online_fit_last_minus_first_utility_bps": float(
                np.median(
                    [
                        r["history"][-1]["fit_utility_bps"]
                        - r["history"][1]["fit_utility_bps"]
                        for r in records
                    ]
                )
            ),
            "median_terminal_minus_epoch_zero_selection_utility_bps": float(
                np.median(
                    [
                        r["history"][-1]["utility_bps"] - r["history"][0]["utility_bps"]
                        for r in records
                    ]
                )
            ),
            "calibration_and_windows": {
                k.split("/")[2]: {
                    "calibration": v["calibration"],
                    "windows": v["windows"],
                }
                for k, v in trials.items()
                if "seed_11" in k
            },
            "selected_trials": {
                k: {
                    "selected_epoch": v["selected_epoch"],
                    "epochs_completed": v["epochs_completed"],
                    "initial_selection": v["history"][0],
                    "selected_selection": next(
                        h for h in v["history"] if h["epoch"] == v["selected_epoch"]
                    ),
                    "terminal_selection": v["history"][-1],
                }
                for k, v in trials.items()
            },
        }
    intervals, continuous_intervals, original_centers = {}, {}, {}
    for arm in ("S0", "TE_all", "C6"):
        arrays = []
        for fold in range(1, 15):
            prefix = f"books/{arm}/F{fold}/base/"
            learned = np.mean(
                [
                    books[prefix + f"learned_{seed}"]["daily"]["utility_bps"]
                    for seed in (11, 29, 47)
                ],
                axis=0,
            )
            arrays.append(
                learned
                - np.asarray(books[prefix + "optimizer"]["daily"]["utility_bps"])
            )
        intervals[arm] = {
            str(length): blocked_interval(arrays, length) for length in (20, 40, 60)
        }
        original_centers[arm] = noncircular_expected_mean(arrays)
        prefix = f"books/{arm}/continuous/base/"
        delta = np.mean(
            [
                books[prefix + f"learned_{seed}"]["daily"]["utility_bps"]
                for seed in (11, 29, 47)
            ],
            axis=0,
        ) - np.asarray(books[prefix + "optimizer"]["daily"]["utility_bps"])
        continuous_intervals[arm] = {
            str(length): blocked_interval([delta], length) for length in (20, 40, 60)
        }
    output = {
        "status": "read_only_posthoc_development_audit",
        "code_commit_at_run": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "audit_script_sha256": digest(Path(__file__)),
        "sources": {
            "archives": str(args.archives),
            "inventory_sha256": digest(inventory_path),
            "verified_members": checked,
        },
        "books": result,
        "training": training,
        "intercept_demonstration": intercept_demonstration(),
        "uncertainty_sensitivity": {
            "contrast": "mean of three learned policy outcomes minus optimizer utility",
            "method": "10000 circular within-fold block draws, seed20260914, paired dates; nominal posthoc intervals, not promotion tests",
            "results": intervals,
            "original_noncircular_exact_expected_mean": original_centers,
            "continuous_circular_results": continuous_intervals,
        },
        "new_historical_backtests": False,
        "new_training": False,
        "heldout_accessed": False,
        "forward_capture": False,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "books": len(books),
                "verified_members": len(checked),
                "toy": output["intercept_demonstration"],
                "intervals": intervals,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
