"""Matched retained-control financing scenarios; no broker terms are asserted."""

from pathlib import Path

import numpy as np
import torch

from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    exact_replay,
    policy_ledger_config,
)
from .artifacts import write_json_atomic
from .foundation_readouts import new_panel, paired_interval
from .objective_readouts import calibration, rank_view, read_panel
from .opportunity_research import bound
from .portfolio_program import read
from .portfolio_readouts import save_book, verify_book
from .portfolio_training import load_data, windows
from .research_rounds import _git_identity
from .round7 import SCREEN_FOLDS


def run(root):
    torch.set_num_threads(1)
    design = read(root / "frozen_design.json")
    prior = Path(design["prior_decision_root"])
    data, binding = load_data(prior, "C6")
    results = {}
    for arm in ("C6", "TE_full"):
        changes = {
            "proceeds0": {"short_proceeds_remuneration": 0.0},
            "debit3": {"annual_debit_spread": 0.03},
        }
        differences = {name: [] for name in changes}
        for fold in SCREEN_FOLDS:
            rows = windows(prior, data, fold)["evaluation"]
            if arm == "C6":
                panels, _, valid, sources = read_panel(
                    prior, data, arm, "neutral", fold, rows
                )
                base = prior / "phase3/books" / fold / "C6/neutral/ensemble"
            else:
                panels, valid, sources = new_panel(root, data, arm, fold, rows, "raw")
                base = root / "books/TE_full/raw" / fold / "ensemble"
            mapping_path = prior / "phase3/mappings" / f"{fold}.json"
            mapping = calibration(
                read(mapping_path)["arms"]["C6" if arm == "C6" else "TE_all"]
            )
            view = rank_view(data, rows, panels["ensemble"], valid)
            original = read(base / "book.json")
            for scenario, delta in changes.items():
                output = root / "financing" / arm / scenario / fold
                provenance = {
                    "implementation": _git_identity(),
                    "policy": "equal_rank",
                    "scenario": scenario,
                    "ledger_changes": delta,
                    "economic_cache_binding": binding,
                    "mapping": bound(mapping_path),
                    "forecasts": sources,
                    "base": bound(base / "book.json"),
                }
                if (output / "book.json").exists():
                    record = verify_book(output, provenance)
                else:
                    account, targets, previous = exact_replay(
                        view,
                        CalibratedPolicy(mapping),
                        int(rows[0]),
                        int(rows[-1] + 1),
                        config=policy_ledger_config(**delta),
                    )
                    record = save_book(
                        output,
                        view,
                        account,
                        targets,
                        previous,
                        int(rows[0]),
                        int(rows[0]),
                        provenance,
                    )
                differences[scenario].append(
                    np.asarray(record["daily"]["net_excess_bps"])
                    - np.asarray(original["daily"]["net_excess_bps"])
                )
                print({"financing": [arm, scenario, fold]}, flush=True)
        results[arm] = {
            name: {
                str(length): paired_interval(parts, length) for length in (20, 40, 60)
            }
            for name, parts in differences.items()
        }
    write_json_atomic(root / "financing_summary.json", results)

