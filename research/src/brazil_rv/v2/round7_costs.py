"""Round-7 rate-only sensitivities, reusing the accepted original-rate book."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import retained
from .contract import DEVELOPMENT_FOLDS
from .execution_policy import ledger_gate_failures
from .round6 import resolve_file
from .round6_costs import headline_ledger, load_rates
from .round7 import SEEDS
from .round7_data import PROJECT
from .round7_program import read
from .round7_readouts import aggregate_path, economics_design


def run(root, cells, folds, seeds):
    design = read(root / "frozen_design.json")
    context = rr._open_ledger_replay(economics_design(design))
    source = read(PROJECT / "docs/v2_round6_inputs.json")["preflight"][
        "borrow_sensitivity_calibration"
    ]
    borrow_root = resolve_file(source).parent
    panels, borrowing = load_rates(borrow_root, context)
    output = root / "borrow_sensitivities" / ("seeds_" + "_".join(map(str, seeds)))
    records = {c: {} for c in cells}
    try:
        for fold in folds:
            ix = context.evaluation[fold]
            for cell in cells:
                path = output / cell / f"{fold}.json"
                aggregate = aggregate_path(root, cell, fold, seeds)
                digest = sha256_file(aggregate / "evaluation.json")
                if path.exists():
                    saved = read(path)
                    if (
                        saved["original_evaluation_sha256"] != digest
                        or saved["borrow_manifest_sha256"] != source["sha256"]
                    ):
                        raise ValueError(
                            "rate replay differs from the accepted source book"
                        )
                    records[cell][fold] = saved
                    continue
                evaluated = retained(context, aggregate, fold)
                original = evaluated.result.report["economics"]
                scenarios = {
                    "original": {
                        "summary": original["headline"],
                        "daily": original["headline_audit"]["daily_state"],
                        "gates": ledger_gate_failures(
                            original["headline"], headline=True
                        ),
                        "promotion_weight": 1,
                        "reuse": "exact accepted ledger; load_rates verifies original arrays against this economic context",
                    }
                }
                for label in ("placeholder_v2", "latest_vintage"):
                    rates = panels[label]
                    changed = replace(
                        evaluated.inputs,
                        annual_borrow_rate_by_name=rates["annual_taker_rate"][ix],
                        borrow_rate_imputed=rates["rate_imputed"][ix],
                        borrow_rate_placeholder=rates["rate_placeholder"][ix],
                        borrow_source_label=label,
                    )
                    _, summary, daily = headline_ledger(changed)
                    scenarios[label] = {
                        "summary": summary,
                        "daily": daily,
                        "gates": ledger_gate_failures(summary, headline=True),
                        "promotion_weight": 0,
                    }
                saved = {
                    "schema": "BRAZIL_RV_ROUND7_RATE_REPLAY_V1",
                    "cell": cell,
                    "fold": fold,
                    "seeds": list(seeds),
                    "original_evaluation_sha256": digest,
                    "borrow_manifest_sha256": source["sha256"],
                    "scenarios": scenarios,
                    "unchanged_shortability": True,
                    "latest_availability_differences_not_consumed": borrowing[
                        "latest_availability_differences_not_consumed"
                    ],
                }
                write_json_atomic(path, saved)
                records[cell][fold] = saved
                print(f"rate scenarios complete {cell}/{fold}", flush=True)
        summary = {
            cell: {
                label: {
                    key: rr._folded_bootstrap(
                        tuple(
                            np.asarray(
                                [d[key] for d in row["scenarios"][label]["daily"]],
                                float,
                            )
                            for row in by_fold.values()
                        )
                    )
                    for key in (
                        "net_excess_all_cash_bps",
                        "turnover_fraction_nav",
                        "borrow_cost_bps",
                    )
                }
                for label in panels
            }
            for cell, by_fold in records.items()
        }
        write_json_atomic(
            output / "summary.json",
            {
                "cells": summary,
                "folds": list(folds),
                "seeds": list(seeds),
                "rule": "all calendar days; sensitivities change rates only and never choose the model",
            },
        )
        return summary
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--folds", nargs="+", default=list(DEVELOPMENT_FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    run(args.root, args.cells, args.folds, args.seeds)


if __name__ == "__main__":
    main()
