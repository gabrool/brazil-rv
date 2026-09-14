"""Causal calibrated TL optimizer confirmation against sealed TE books."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from brazil_rv.execution.portfolio_policy import (
    PreferenceModel,
    exact_replay,
    policy_ledger_config,
)
from brazil_rv.v2 import research_rounds as rr
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import DEVELOPMENT_FOLDS
from brazil_rv.v2.portfolio_program import read
from brazil_rv.v2.portfolio_readouts import SCENARIOS, interval, save_book
from brazil_rv.v2.portfolio_training import calibration_for, load_data, windows
from brazil_rv.v2.round7 import SCREEN_FOLDS, SEEDS


def run(root, source):
    torch.set_num_threads(1)
    design, original = (
        read(root / "frozen_design.json"),
        read(source / "frozen_design.json"),
    )
    if sha256_file(source / "frozen_design.json") != design["source_design_sha256"]:
        raise ValueError("source design changed")
    base, binding = load_data(source, "TE_all")
    bounds = {f: windows(source, base, f) for f in DEVELOPMENT_FOLDS}
    first_eval = int(bounds["F1"]["evaluation"][0])
    blocks = {
        "prelude": np.arange(first_eval),
        **{f: b["evaluation"] for f, b in bounds.items()},
    }
    if not np.array_equal(
        np.concatenate(list(blocks.values())), np.arange(len(base.inputs.dates))
    ):
        raise ValueError(
            "TL forecast blocks do not cover the causal cache exactly once"
        )
    values, masks, sources = [], [], {}
    for fold, rows in blocks.items():
        members, mask, records = [], None, []
        dates = np.asarray([base.inputs.dates[i] for i in rows], dtype="datetime64[D]")
        for seed in SEEDS:
            directory = (
                root / "prelude" / f"seed_{seed}"
                if fold == "prelude"
                else root / "fine" / f"{fold}_seed_{seed}"
            )
            record = read(directory / "scores/score_manifest.json")
            if record["store"]["manifest_sha256"] != design["store"]["manifest_sha256"]:
                raise ValueError("TL forecast store changed")
            scores, valid = rr._score_artifact(
                directory / "scores",
                require_clean_transfer=True,
                expected_dates=dates,
                expected_isins=base.inputs.security_ids,
                expected_feature_schema_sha256=original["feature_schema_sha256"],
            )
            if mask is not None and not np.array_equal(mask, valid):
                raise ValueError("TL seed populations differ")
            members.append(scores)
            mask = valid
            records.append(
                {
                    "path": str(directory),
                    "score_manifest_sha256": sha256_file(
                        directory / "scores/score_manifest.json"
                    ),
                }
            )
        if not np.array_equal(mask, base.inputs.score_mask[rows]):
            raise ValueError("early/late forecast populations differ")
        values.append(rr.rank_average_ensemble(members, mask))
        masks.append(mask)
        sources[fold] = records
    data = replace(
        base,
        inputs=replace(
            base.inputs, scores=np.concatenate(values), score_mask=np.concatenate(masks)
        ),
    )
    provenance = {
        "source_policy_data_sha256": binding,
        "forecasts": sources,
        "design_sha256": sha256_file(root / "frozen_design.json"),
        "readout_script_sha256": sha256_file(Path(__file__)),
        "policy": "optimizer",
        "arm": "TL_all",
        "planned_transaction_cost_bps": 4.0,
    }
    models = {
        f: PreferenceModel(data, calibration_for(data, b["fit"]), b["fit"]).eval()
        for f, b in bounds.items()
    }
    collected = {}
    for label in (*DEVELOPMENT_FOLDS, "continuous"):
        if label == "continuous":
            first = first_eval
            start = int(bounds["F1"]["selection"][-1] + 1)
            stop = int(bounds["F14"]["evaluation"][-1] + 1)
            model = {}
            for f, b in bounds.items():
                rows = (
                    np.arange(start, b["evaluation"][-1] + 1)
                    if f == "F1"
                    else b["evaluation"]
                )
                model.update({int(day): models[f] for day in rows})
        else:
            b = bounds[label]
            first, stop = int(b["evaluation"][0]), int(b["evaluation"][-1] + 1)
            start, model = int(b["selection"][-1] + 1), models[label]
        collected[label] = {}
        for scenario, changes in SCENARIOS.items():
            output = root / "optimizer_readouts" / label / scenario
            if output.exists():
                raise ValueError(
                    "TL optimizer book already exists; inspect before replacing"
                )
            result, targets, previous = exact_replay(
                data, model, start, stop, config=policy_ledger_config(**changes)
            )
            book = save_book(
                output,
                data,
                result,
                targets,
                previous,
                start,
                first,
                {
                    **provenance,
                    "scenario": scenario,
                    "window": label,
                    "realized_changes": changes,
                },
            )
            record = {
                "book": str(output / "book.json"),
                "book_sha256": sha256_file(output / "book.json"),
                "summary": book["summary"],
                "unresolved_action_name_days": int(
                    result.unresolved_action_name_days.sum()
                ),
                "settlement_fraction_sum": float(
                    result.terminal_settlement_notional_fraction_nav.sum()
                ),
                "unresolved_inventory_notional": result.unresolved_inventory_notional,
                "insolvent": result.insolvent,
            }
            write_json_atomic(output / "diagnostics.json", record)
            reference = read(
                source / "books/TE_all" / label / scenario / "optimizer/book.json"
            )
            if reference["dates"] != book["dates"]:
                raise ValueError("optimizer comparisons have different dates")
            collected[label][scenario] = {
                **record,
                "paired_daily": {
                    m: (
                        np.asarray(book["daily"][m]) - np.asarray(reference["daily"][m])
                    ).tolist()
                    for m in ("net_excess_bps", "utility_bps")
                },
            }
        print(label, "complete", flush=True)
    groups = {
        "original_screen": SCREEN_FOLDS,
        "remaining_ten": [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS],
        "all_fourteen": DEVELOPMENT_FOLDS,
    }
    summary = {
        "status": "completed",
        "heldout_accessed": False,
        "forward_capture": False,
        "groups": {
            g: {
                s: {
                    m: interval([collected[f][s]["paired_daily"][m] for f in folds])
                    for m in ("net_excess_bps", "utility_bps")
                }
                for s in SCENARIOS
            }
            for g, folds in groups.items()
        },
        "books": {
            f: {
                s: {k: v for k, v in r.items() if k != "paired_daily"}
                for s, r in scenarios.items()
            }
            for f, scenarios in collected.items()
        },
        "continuous_paired_means": {
            s: {m: float(np.mean(v)) for m, v in r["paired_daily"].items()}
            for s, r in collected["continuous"].items()
        },
    }
    write_json_atomic(root / "optimizer_comparison.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.source)
