"""Historical gradient/causality admission and source-bound closeout export."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.objective_readouts import calibration
from brazil_rv.v2.opportunity_research import bound, checked
from brazil_rv.v2.portfolio_objective import (
    TensorPreference,
    clone_account,
    utility_path,
)
from brazil_rv.v2.portfolio_program import PROJECT, read
from brazil_rv.v2.portfolio_training import load_data
from brazil_rv.v2.research_rounds import _git_identity


def audit_gradients(root):
    torch.set_num_threads(1)
    old = Path(read(PROJECT / "docs/v2_decision_run.json")["root"])
    data, binding = load_data(old, "C6")
    m = calibration(read(old / "phase3/mappings/F2.json")["arms"]["C6"])
    values = (data.ranks.mean(-1) - m.mean[0]) / m.scale[0] * m.coefficient[
        0
    ] + m.intercept
    results = []
    for start in (0, 120, 300):  # all within F2 fit, not financial evaluation
        rows = np.arange(start, start + 12)
        preference = torch.tensor(values[rows])
        initial = data.initial_account(start, policy_ledger_config())
        rng = np.random.default_rng(start + 20260917)
        direction = torch.tensor(rng.normal(size=preference.shape) * 0.0001)

        def value(coefficient):
            return utility_path(
                data,
                preference + coefficient * direction,
                clone_account(initial),
                rows,
                terminal=True,
            )

        t = time.monotonic()
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        loss, _, plans, nav = value(parameter)
        loss.backward()
        analytic = parameter.grad.item()
        numerical = []
        for epsilon in (1e-4, 1e-5):
            plus, minus = value(epsilon)[0].item(), value(-epsilon)[0].item()
            numerical.append((plus - minus) / (2 * epsilon))
        exact, expected, _ = exact_replay(
            data, TensorPreference(preference, start), start, start + 12
        )
        result = {
            "first_row": start,
            "last_row": start + 11,
            "analytic": analytic,
            "finite_difference": numerical,
            "nav_error": float(np.max(np.abs(nav - exact.nav))),
            "target_error": float(np.max(np.abs(plans - expected))),
            "seconds": time.monotonic() - t,
        }
        results.append(result)
        write_json_atomic(
            root / "historical_gradient_audit.json",
            {
                "implementation": _git_identity(),
                "cache_binding": binding,
                "tests": results,
                "completed": False,
            },
        )
        if abs(analytic - numerical[-1]) > 0.02 * abs(numerical[-1]) + 0.001:
            raise ValueError(f"historical account derivative differs: {result}")
        if result["nav_error"] > 1e-7 or result["target_error"] > 1e-7:
            raise ValueError("historical account differs from independent ledger")
    write_json_atomic(
        root / "historical_gradient_audit.json",
        {
            "implementation": _git_identity(),
            "cache_binding": binding,
            "tests": results,
            "completed": True,
            "financial_evaluation_read": False,
        },
    )
    print(results)


def export_closeout(root):
    frozen = read(root / "closeout_design.json")
    store = Path(frozen["store"]["root"])
    manifest = read(store / "manifest.json")
    dates = np.load(store / "date_index.npy", allow_pickle=False)
    isins = np.load(store / "isin_index.npy", allow_pickle=False)
    arrays = {
        k: np.load(
            store / manifest["arrays"][k]["path"], mmap_mode="r", allow_pickle=False
        )
        for k in (
            "raw_close",
            "action_successor_index",
            "action_has_action",
            "action_session_resolved",
            "active",
        )
    }
    report = {
        "design": bound(root / "closeout_design.json"),
        "arms": {},
        "largest_events": {},
    }
    for arm in ("C6", "TE_all"):
        source = root / "closeout" / arm / "summary.json"
        summary = read(source)
        report["arms"][arm] = {
            "source": bound(source),
            "books": summary["books"],
            "intercept_daily_bps": summary["intercept_daily_bps"],
        }
        for scenario, record in summary["settlements"].items():
            checked(record["source"])
            for event in sorted(
                record["records"], key=lambda x: abs(x["gross_notional"]), reverse=True
            )[:10]:
                key = f"{event['security']}:{event['fill_date']}"
                if key in report["largest_events"]:
                    continue
                j = int(np.flatnonzero(isins == event["security"])[0])
                t = int(np.searchsorted(dates, np.datetime64(event["fill_date"])))
                observed = np.flatnonzero(
                    np.isfinite(arrays["raw_close"][:, j])
                    & (arrays["raw_close"][:, j] > 0)
                )
                before = observed[observed <= t]
                after = observed[observed > t]
                begin = max(0, int(before[-1]) - 5) if len(before) else max(0, t - 20)
                report["largest_events"][key] = {
                    "example_arm": arm,
                    "example_scenario": scenario,
                    "last_quote": str(dates[before[-1]]) if len(before) else None,
                    "next_quote_same_isin": str(dates[after[0]])
                    if len(after)
                    else None,
                    "observed_fill": False,
                    "example_settlement": event,
                    "action_rows": [
                        {
                            "date": str(dates[i]),
                            "resolved": bool(arrays["action_session_resolved"][i, j]),
                            "successor": str(
                                isins[arrays["action_successor_index"][i, j]]
                            ),
                        }
                        for i in range(begin, min(t + 1, len(dates)))
                        if arrays["action_has_action"][i, j]
                    ],
                    "source_limit": "No verified contractual successor/cash settlement in accepted store; retain sensitivity flag.",
                }
    report["corporate_action_contract"] = manifest["metadata"][
        "corporate_action_contract"
    ]
    report["isin_succession_link_count"] = manifest["metadata"][
        "isin_succession_link_count"
    ]
    write_json_atomic(root / "closeout_summary.json", report)
    write_json_atomic(PROJECT / "docs/v2_portfolio_objective_closeout.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("gradients", "closeout"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (audit_gradients if args.command == "gradients" else export_closeout)(args.root)


if __name__ == "__main__":
    main()
