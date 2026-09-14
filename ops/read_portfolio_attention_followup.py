"""Paired, empty-start rich-attention timing screen on sealed economic inputs."""

import argparse
from copy import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from brazil_rv.v2 import research_rounds as rr
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import DEVELOPMENT_FOLDS, HORIZONS, TRADED_PRIMARY_HORIZONS
from brazil_rv.v2.evaluate import _primary_daily_metrics, _primary_population_components
from brazil_rv.v2.portfolio_program import read
from brazil_rv.v2.portfolio_readouts import interval, legacy_replay, save_book
from brazil_rv.v2.portfolio_training import load_data, windows
from brazil_rv.v2.round7 import SCREEN_FOLDS, SEEDS


def run(root, source, *, remaining=False):
    torch.set_num_threads(1)
    design, original = (
        read(root / "frozen_design.json"),
        read(source / "frozen_design.json"),
    )
    if design["source_design_sha256"] != sha256_file(source / "frozen_design.json"):
        raise ValueError("source design changed")
    data, binding = load_data(source, "TE_all")
    results = {}
    folds = (
        [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS]
        if remaining
        else SCREEN_FOLDS
    )
    for fold in folds:
        rows = windows(source, data, fold)["evaluation"]
        start, stop = int(rows[0]), int(rows[-1] + 1)
        expected_dates = np.asarray(
            data.inputs.dates[start:stop], dtype="datetime64[D]"
        )
        paired = {}
        for arm in ("TE_all", "TL_all"):
            members, sources, mask = [], [], None
            for seed in SEEDS:
                reused = original["reused_fits"]["TE_all"].get(fold, {}).get(str(seed))
                directory = (
                    (
                        Path(reused["manifest"]["path"]).parent
                        if reused
                        else source / "forecasters/TE_all" / f"{fold}_seed_{seed}"
                    )
                    if arm == "TE_all"
                    else root / "fine" / f"{fold}_seed_{seed}"
                )
                manifest = read(directory / "scores/score_manifest.json")
                if (
                    manifest["store"]["manifest_sha256"]
                    != design["store"]["manifest_sha256"]
                ):
                    raise ValueError("paired forecasts use different stores")
                values, valid = rr._score_artifact(
                    directory / "scores",
                    require_clean_transfer=True,
                    expected_dates=expected_dates,
                    expected_isins=data.inputs.security_ids,
                    expected_feature_schema_sha256=original["feature_schema_sha256"],
                )
                if mask is not None and not np.array_equal(valid, mask):
                    raise ValueError("seed score populations differ")
                mask = valid
                members.append(values)
                sources.append(
                    {
                        "path": str(directory),
                        "score_manifest_sha256": sha256_file(
                            directory / "scores/score_manifest.json"
                        ),
                    }
                )
            if not np.array_equal(mask, data.inputs.score_mask[start:stop]):
                raise ValueError("timing comparison score populations differ")
            predictions = {
                "ensemble": rr.rank_average_ensemble(members, mask),
                **{str(s): v for s, v in zip(SEEDS, members)},
            }
            paired[arm] = {}
            for label, scores in predictions.items():
                view = copy(data)
                full = data.inputs.scores.copy()
                full[start:stop] = scores
                target_mask = data.inputs.neutral_target_mask.copy()
                for h, horizon in enumerate(HORIZONS):
                    target_mask[max(start, stop - horizon) : stop, :, h] = False
                view.inputs = replace(
                    data.inputs, scores=full, neutral_target_mask=target_mask
                )
                output = root / "readouts" / fold / arm / label
                provenance = {
                    "scenario": "base",
                    "policy": "legacy",
                    "arm": arm,
                    "fold": fold,
                    "member": label,
                    "policy_data_sha256": binding,
                    "sources": sources,
                    "readout_script_sha256": sha256_file(Path(__file__)),
                }
                if output.exists():
                    raise ValueError(
                        "readouts already exist; inspect before replacing results"
                    )
                config = replace(
                    data.inputs.execution_policy.ledger_config(),
                    settle_terminal_residuals=True,
                )
                result, targets, previous = legacy_replay(view, start, stop, config)
                book = save_book(
                    output, view, result, targets, previous, start, start, provenance
                )
                if arm == "TE_all" and label == "ensemble":
                    reference = read(source / "bridge/TE_all" / fold / "book.json")
                    for metric in ("net_excess_bps", "utility_bps"):
                        if not np.allclose(
                            book["daily"][metric],
                            reference["daily"][metric],
                            atol=1e-8,
                            rtol=0,
                        ):
                            raise ValueError("sealed reference replay changed")
                components = _primary_population_components(
                    view.inputs, TRADED_PRIMARY_HORIZONS
                )
                head, ic, _ = _primary_daily_metrics(
                    *components, view.inputs.dates, TRADED_PRIMARY_HORIZONS
                )
                audit = {
                    "head_ic": np.nanmean(head[start:stop], axis=0).tolist(),
                    "mean_ic": float(np.nanmean(ic[start:stop])),
                    "daily_ic": [
                        float(v) if np.isfinite(v) else None for v in ic[start:stop]
                    ],
                    "settlement_fraction_sum": float(
                        result.terminal_settlement_notional_fraction_nav.sum()
                    ),
                    "unresolved_action_name_days": int(
                        result.unresolved_action_name_days.sum()
                    ),
                    "unresolved_inventory_notional": result.unresolved_inventory_notional,
                    "insolvent": result.insolvent,
                }
                write_json_atomic(output / "diagnostics.json", audit)
                paired[arm][label] = {"book": book, "diagnostics": audit}
        results[fold] = paired
    summary = {
        "status": "completed",
        "heldout_accessed": False,
        "forward_capture": False,
        "paired": {},
    }
    for label in ("ensemble", *(str(s) for s in SEEDS)):
        summary["paired"][label] = {
            metric: interval(
                [
                    np.asarray(r["TL_all"][label]["book"]["daily"][metric])
                    - np.asarray(r["TE_all"][label]["book"]["daily"][metric])
                    for r in results.values()
                ]
            )
            for metric in ("net_excess_bps", "utility_bps")
        }
    summary["folds"] = {
        f: {
            a: {
                s: {"summary": r["book"]["summary"], "diagnostics": r["diagnostics"]}
                for s, r in ss.items()
            }
            for a, ss in arms.items()
        }
        for f, arms in results.items()
    }
    summary["positive_utility_folds"] = sum(
        r["TL_all"]["ensemble"]["book"]["summary"]["mean"]["utility_bps"]
        > r["TE_all"]["ensemble"]["book"]["summary"]["mean"]["utility_bps"]
        for r in results.values()
    )
    summary["point_estimate_gate"] = (
        remaining or summary["positive_utility_folds"] >= 3
    ) and all(
        summary["paired"]["ensemble"][m]["estimate"] > 0
        for m in ("net_excess_bps", "utility_bps")
    )
    summary["accounting_acceptance_requires_review"] = any(
        r[a]["ensemble"]["book"]["summary"]["economics_unresolved"]
        for r in results.values()
        for a in r
    )
    if remaining:
        summary["all_fourteen_paired"] = {
            label: {
                metric: interval(
                    [
                        np.asarray(
                            read(
                                root / "readouts" / f / "TL_all" / label / "book.json"
                            )["daily"][metric]
                        )
                        - np.asarray(
                            read(
                                root / "readouts" / f / "TE_all" / label / "book.json"
                            )["daily"][metric]
                        )
                        for f in DEVELOPMENT_FOLDS
                    ]
                )
                for metric in ("net_excess_bps", "utility_bps")
            }
            for label in ("ensemble", *(str(s) for s in SEEDS))
        }
    write_json_atomic(
        root
        / (
            "timing_remaining_comparison.json"
            if remaining
            else "timing_comparison.json"
        ),
        summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--remaining", action="store_true")
    args = parser.parse_args()
    run(args.root, args.source, remaining=args.remaining)
