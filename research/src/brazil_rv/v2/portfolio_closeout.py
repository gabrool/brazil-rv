"""Bounded mapping decomposition and financing sensitivity before objective fits."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    exact_replay,
    policy_ledger_config,
)
from .artifacts import write_json_atomic
from .objective_readouts import calibration, rank_view
from .opportunity_research import bound, checked, enriched_book
from .portfolio_program import PROJECT, read
from .portfolio_readouts import interval, save_book, verify_book
from .portfolio_training import load_data
from .research_rounds import _git_identity


def prepare(root):
    source = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    design = read(source / "frozen_design.json")
    root.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        root / "closeout_design.json",
        {
            "implementation": _git_identity(),
            "source": bound(source / "frozen_design.json"),
            "source_root": str(source),
            "registration": bound(
                PROJECT / "research/preregistrations/v2_portfolio_objective.md"
            ),
            "store": read(PROJECT / "docs/v2_data_inputs.json")["store"],
            "arms": list(design["arms"]),
            "heldout_accessed": False,
        },
    )


def closeout(root, arm):
    torch.set_num_threads(1)
    frozen = read(root / "closeout_design.json")
    if frozen["implementation"] != _git_identity():
        raise ValueError("closeout source differs")
    source = Path(frozen["source_root"])
    design = read(checked(frozen["source"]))
    data, binding = load_data(Path(design["source_root"]), arm)
    entry = design["arms"][arm]
    if binding != entry["cache_binding"]:
        raise ValueError("closeout cache changed")
    with np.load(checked(entry["prepared"]), allow_pickle=False) as p:
        data = rank_view(data, np.arange(len(data.valid)), p["ranks"], p["valid"])
        sectors, issuers = p["sectors"].copy(), p["issuers"].copy()
    blocks = {f: np.asarray(rows) for f, rows in entry["blocks"].items()}
    rows = np.sort(np.concatenate(list(blocks.values())))
    first, stop = int(rows[0]), int(rows[-1] + 1)
    mappings = {
        f: calibration(read(checked(entry["sources"][f]["mapping"]))["arms"][arm])
        for f in blocks
    }
    trials = []
    for cap in (0.05, 0.45):
        for mode in ("rank", "intercept"):
            trials.append((f"{mode}_net{cap:g}", mode, cap, {}))
        for scenario, changes in (
            ("cost8", {"cost_bps_per_side": 8.0, "hedge_cost_bps_per_side": 8.0}),
            ("proceeds0", {"short_proceeds_remuneration": 0.0}),
            ("debit3", {"annual_debit_spread": 0.03}),
        ):
            trials.append((f"{scenario}_net{cap:g}", "full", cap, changes))
    trials.append(("full_net0.25", "full", 0.25, {}))
    analyses = {}
    for name, mode, cap, stress in trials:
        cfg = replace(AllocationConfig(), net_cap=cap)
        models = {}
        for fold, days in blocks.items():
            mapping = mappings[fold]
            if mode == "rank":
                mapping = replace(mapping, intercept=0.0)
            elif mode == "intercept":
                mapping = replace(
                    mapping, coefficient=np.zeros_like(mapping.coefficient)
                )
            model = CalibratedPolicy(mapping)
            models.update({int(day): model for day in days})
        path = root / "closeout" / arm / name
        provenance = {
            "scenario": name if stress else "base",
            "policy": name,
            "arm": arm,
            "design": bound(root / "closeout_design.json"),
            "allocation": asdict(cfg),
            "ledger_changes": stress,
        }
        t = time.monotonic()
        if (path / "book.json").exists():
            verify_book(path, provenance)
        else:
            result, target, previous = exact_replay(
                data,
                models,
                first,
                stop,
                allocation=cfg,
                config=policy_ledger_config(planned_absolute_net_cap=cap, **stress),
            )
            save_book(path, data, result, target, previous, first, first, provenance)
            del result, target, previous
        analyses[name] = enriched_book(source, path, data, sectors, issuers, cfg)
        print(
            {"arm": arm, "closeout": name, "seconds": round(time.monotonic() - t, 1)},
            flush=True,
        )
    for cap, name in ((0.05, "neutral"), (0.45, "flexible_net")):
        path = source / arm / "books" / name
        analyses[f"full_net{cap:g}"] = read(path / "analysis.json")
    summaries = {}
    market = data.inputs.bova11_close[rows] / data.inputs.bova11_close[rows - 1] - 1
    for name, a in analyses.items():
        book = read(checked(a["book"]))
        daily = {k: np.asarray(v) for k, v in book["daily"].items()}
        benchmark = "full_net0.45" if name.endswith("0.45") else "full_net0.05"
        other = analyses[benchmark]["daily"]
        net = daily["net_excess_bps"]
        years = np.asarray([str(d)[:4] for d in book["dates"]])
        summaries[name] = {
            "book": a["book"],
            "metrics": a["metrics"],
            "mean": book["summary"]["mean"],
            "limitations": a["economic_limitations"],
            "paired_net_bps": interval([net - np.asarray(other["net_excess_bps"])]),
            "paired_control": benchmark,
            "year_net_bps": {
                year: float(net[years == year].mean()) for year in np.unique(years)
            },
            "realized_beta": float(
                np.cov(net / 1e4, market, ddof=1)[0, 1] / np.var(market, ddof=1)
            ),
            "issuer_contribution_proxy_top": sorted(
                a["attribution"]["issuer"].items(),
                key=lambda p: abs(p[1]["mean_contribution_proxy_bps"]),
                reverse=True,
            )[:10],
        }
    settlements = {}
    for name in ("neutral", "flexible_net"):
        path = source / arm / "books" / name / "fills.parquet"
        fills = pl.read_parquet(path)
        settlement = fills.filter(pl.col("purpose") == "terminal_settlement")
        settlements[name] = {
            "source": bound(path),
            "schema": dict(zip(fills.columns, map(str, fills.dtypes))),
            "records": settlement.to_dicts(),
        }
    write_json_atomic(
        root / "closeout" / arm / "summary.json",
        {
            "books": summaries,
            "settlements": settlements,
            "intercept_daily_bps": {
                f: float(m.intercept) * 1e4 for f, m in mappings.items()
            },
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("C6", "TE_all"))
    args = parser.parse_args()
    prepare(args.root) if args.command == "prepare" else closeout(args.root, args.arm)


if __name__ == "__main__":
    main()
