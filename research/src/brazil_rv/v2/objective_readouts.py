"""Fixed-policy, causal blend and paired readouts for the auxiliary experiment."""

from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import numpy as np
import torch

from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, HORIZONS
from .controller_program import checked_book, paired
from .data_roots import resolve_external_root
from .decision_program import (
    benchmark_excess_returns,
    calibration_payload,
    calibrations,
)
from .evaluate import _primary_daily_metrics
from .portfolio_inputs import Calibration, HEADS, normalized_ranks
from .portfolio_program import read
from .portfolio_readouts import interval, save_book, verify_book
from .portfolio_training import load_data, selection_value, windows
from .research_rounds import _git_identity, _score_artifact
from .round7 import SCREEN_FOLDS
from .train import rank_average_ensemble

ARMS = ("TE_all", "C6")
VARIANTS = ("neutral", "economic")
MEMBERS = (*map(str, ALLOWED_SEEDS), "ensemble")


def rank_view(data, rows, ranks, valid):
    """Read-only economics, with no old forecast outside the supplied new block."""
    result = copy(data)
    result.ranks = np.zeros_like(data.ranks)
    result.valid = np.zeros_like(data.valid)
    result.ranks[rows] = ranks
    result.valid[rows] = valid
    return result


def combine(a, b, weight):
    if a.shape != b.shape:
        raise ValueError("blend rank axes differ")
    return (1 - weight) * a + weight * b


def choose_blend(data, other, bounds):
    if not np.array_equal(data.valid, other.valid):
        raise ValueError("blend cannot silently intersect different populations")
    rows = np.arange(len(data.valid))
    trials = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        view = rank_view(
            data, rows, combine(data.ranks, other.ranks, weight), data.valid
        )
        mapping = calibrations(view, bounds["fit"], benchmark_excess_returns(view))[
            "equal_rank"
        ]
        selection = selection_value(view, CalibratedPolicy(mapping), bounds)
        trials.append(
            {
                "te_weight": weight,
                "calibration": calibration_payload(mapping),
                **selection,
            }
        )
    selected = max(trials, key=lambda row: row["utility_bps"])
    return {"selected": selected, "trials": trials}


def prepare_mappings(root, *, confirmation=False):
    """Only old OOS fits/selection windows choose mappings and blend weights."""
    torch.set_num_threads(1)
    data, c6_binding = load_data(root, "C6")
    other, te_binding = load_data(root, "TE_all")
    if not np.array_equal(data.inputs.dates, other.inputs.dates) or not np.array_equal(
        data.inputs.security_ids, other.inputs.security_ids
    ):
        raise ValueError("old OOS blend caches have different identity/date axes")
    folds = (
        [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS]
        if confirmation
        else SCREEN_FOLDS
    )
    sources = {"C6": c6_binding, "TE_all": te_binding}
    for fold in folds:
        destination = root / "phase3/mappings" / f"{fold}.json"
        if destination.exists():
            if read(destination)["source_cache_bindings"] != sources:
                raise ValueError("frozen historical mappings bind other caches")
            continue
        bounds = windows(root, data, fold)
        result = {
            "implementation": _git_identity(),
            "source_cache_bindings": sources,
            "fit": bounds["fit"].tolist(),
            "selection": bounds["selection"].tolist(),
            "arms": {
                arm: calibration_payload(
                    calibrations(
                        values, bounds["fit"], benchmark_excess_returns(values)
                    )["equal_rank"]
                )
                for arm, values in (("C6", data), ("TE_all", other))
            },
            "blend": choose_blend(data, other, bounds),
            "heldout_accessed": False,
        }
        write_json_atomic(destination, result)


def calibration(record):
    return Calibration(
        **{
            key: np.asarray(value)
            if key in ("mean", "scale", "coefficient", "covariance")
            and value is not None
            else value
            for key, value in record.items()
        }
    )


def read_panel(root, data, arm, variant, fold, rows):
    design = read(root / "phase3/frozen_design.json")
    store, _ = resolve_external_root(design["store"]["root"])
    schema = read(store / "manifest.json")["metadata"]["feature_schema"]["sha256"]
    panels, heads, sources, common = {}, {}, {}, None
    for seed in ALLOWED_SEEDS:
        directory = root / "phase3/fits" / arm / variant / f"{fold}_seed_{seed}"
        record = read(directory / "run_manifest.json")
        score_path = directory / "scores"
        score_record = read(score_path / "score_manifest.json")
        contract = record["contract"]
        if (
            record["status"] != "completed"
            or contract["code"] != design["implementation"]
            or record["seed"] != seed
            or record["fold"] != fold
            or contract["store_manifest_sha256"] != design["store"]["manifest_sha256"]
            or contract["cell"] != design["cells"][arm]
        ):
            raise ValueError(
                "objective score block is not its registered completed fit"
            )
        if contract != score_record["round7_contract"] or bool(
            contract["economic_auxiliary"]
        ) != (variant == "economic"):
            raise ValueError("objective scoring/training contract differs")
        if (
            sha256_file(directory / "selected.pt")
            != score_record["checkpoint"]["sha256"]
        ):
            raise ValueError("objective selected checkpoint changed")
        values, mask = _score_artifact(
            score_path,
            require_clean_transfer=True,
            expected_dates=np.asarray(data.inputs.dates, dtype="datetime64[D]")[rows],
            expected_isins=data.inputs.security_ids,
            expected_feature_schema_sha256=schema,
        )
        mask = mask[..., HEADS]
        valid = mask.all(-1) & data.inputs.active[rows]
        if not np.array_equal(valid, data.inputs.active[rows]) or (
            common is not None and not np.array_equal(valid, common)
        ):
            raise ValueError("objective forecasts lost eligible names")
        common = valid
        panels[str(seed)] = normalized_ranks(
            rank_average_ensemble([values[..., HEADS]], mask), mask
        )
        if variant == "economic":
            name = "economic_daily_residual.npy"
            if (
                sha256_file(score_path / name)
                != score_record["artifacts"][name]["sha256"]
            ):
                raise ValueError("economic head score changed")
            head = np.load(score_path / name, allow_pickle=False)
            if head.shape != valid.shape or not np.isfinite(head[valid]).all():
                raise ValueError("economic head population differs")
            heads[str(seed)] = head
        sources[str(seed)] = {
            "fit_manifest_sha256": sha256_file(directory / "run_manifest.json"),
            "score_manifest_sha256": sha256_file(score_path / "score_manifest.json"),
            "selected_epoch": record["selected_epoch"],
        }
    panels["ensemble"] = np.mean(list(panels.values()), axis=0)
    if heads:
        heads["ensemble"] = np.mean(list(heads.values()), axis=0)
    return panels, heads, common, sources


def cardinal_readout(predicted, outcome, valid, ranks):
    """Equal-date errors/slopes; masks do not alter the decision population."""
    counts = valid.sum(1)
    weights = valid / np.maximum(counts[:, None], 1) / max(int((counts > 0).sum()), 1)
    p, y = np.where(valid, predicted, 0), np.where(valid, outcome, 0)
    pm, ym = float((weights * p).sum()), float((weights * y).sum())
    variance = float((weights * (p - pm) ** 2).sum())
    slope = (
        float((weights * (p - pm) * (y - ym)).sum() / variance)
        if variance > 1e-20
        else None
    )
    tails = {}
    for name, selected in (
        ("upper", ranks.mean(-1) > 0.6),
        ("lower", ranks.mean(-1) < -0.6),
    ):
        use = valid & selected
        n = use.sum(1)
        daily = np.where(use, outcome, 0).sum(1) / np.maximum(n, 1)
        tails[name] = float(daily[n > 0].mean() * 1e4) if np.any(n) else None
    return {
        "valid_stock_days": int(valid.sum()),
        "dates": int((counts > 0).sum()),
        "predicted_mean_bps": pm * 1e4,
        "realized_mean_bps": ym * 1e4,
        "rmse_bps": float(np.sqrt((weights * (p - y) ** 2).sum()) * 1e4),
        "calibration_slope": slope,
        "fixed_rank_tail_outcomes_bps": tails,
    }


def forecast_readout(data, rows, ranks, valid, mapping, head=None):
    targets = data.inputs.neutral_midrank_targets[rows][..., HEADS]
    outcome_mask = data.inputs.neutral_target_mask[rows][..., HEADS].all(-1)
    outcome_mask &= (rows + 10 <= rows[-1])[:, None]
    _, ic, _ = _primary_daily_metrics(
        ranks,
        targets,
        outcome_mask,
        valid,
        [data.inputs.dates[i] for i in rows],
        horizons=(3, 5, 10),
    )
    market = benchmark_excess_returns(data)[rows]
    cash = np.array(
        [np.prod(1 + data.inputs.cdi_returns[t + 1 : t + 6]) - 1 for t in rows]
    )
    outcome = (
        data.inputs.shareholder_simple_returns[rows, :, HORIZONS.index(5)]
        - cash[:, None]
        - data.beta[rows] * market[:, None]
    ) / 5
    mask = (
        valid
        & data.inputs.shareholder_target_mask[rows, :, HORIZONS.index(5)]
        & np.isfinite(outcome)
        & (rows + 5 <= rows[-1])[:, None]
    )
    result = {
        "dates": [data.inputs.dates[i].isoformat() for i in rows],
        "neutral_ic": [float(v) if np.isfinite(v) else None for v in ic],
        "neutral_ic_mean": float(np.nanmean(ic)),
        "fixed_mapping": cardinal_readout(mapping.predict(ranks), outcome, mask, ranks),
        "economic_head": cardinal_readout(head, outcome, mask, ranks)
        if head is not None
        else None,
    }
    return result


def evaluate_fold(root, fold, *, confirmation=False):
    torch.set_num_threads(1)
    data, binding = load_data(root, "C6")
    rows = windows(root, data, fold)["evaluation"]
    frozen = read(root / "phase3/mappings" / f"{fold}.json")
    cache, head_cache, masks, provenance = {}, {}, {}, {}
    arms = (
        read(root / "phase3/screen_summary.json")["survivors"] if confirmation else ARMS
    )
    for arm in arms:
        for variant in VARIANTS:
            panels, heads, valid, source = read_panel(
                root, data, arm, variant, fold, rows
            )
            (
                cache[arm, variant],
                head_cache[arm, variant],
                masks[arm, variant],
                provenance[arm, variant],
            ) = panels, heads, valid, source
    common = masks[arms[0], "neutral"]
    if any(not np.array_equal(common, valid) for valid in masks.values()):
        raise ValueError("matched objective/architecture populations differ")
    blend_weight = frozen["blend"]["selected"]["te_weight"]
    for arm in (*arms, *(("blend",) if len(arms) == 2 else ())):
        mapping = calibration(
            frozen["blend"]["selected"]["calibration"]
            if arm == "blend"
            else frozen["arms"][arm]
        )
        for variant in VARIANTS:
            for member in MEMBERS:
                ranks = (
                    combine(
                        cache["C6", variant][member],
                        cache["TE_all", variant][member],
                        blend_weight,
                    )
                    if arm == "blend"
                    else cache[arm, variant][member]
                )
                head = (
                    None
                    if variant == "neutral"
                    else combine(
                        head_cache["C6", variant][member],
                        head_cache["TE_all", variant][member],
                        blend_weight,
                    )
                    if arm == "blend"
                    else head_cache[arm, variant][member]
                )
                output = root / "phase3/books" / fold / arm / variant / member
                source = {
                    "implementation": _git_identity(),
                    "scenario": "base",
                    "policy": "equal_rank",
                    "economic_cache_binding": binding,
                    "mapping_sha256": sha256_file(
                        root / "phase3/mappings" / f"{fold}.json"
                    ),
                    "forecasts": {
                        a: provenance[a, variant] for a in arms if arm in (a, "blend")
                    },
                    "member": member,
                    "initial_state": "cash at first evaluation date",
                }
                if (output / "book.json").exists():
                    verify_book(output, source)
                else:
                    view = rank_view(data, rows, ranks, common)
                    result, targets, previous = exact_replay(
                        view, CalibratedPolicy(mapping), int(rows[0]), int(rows[-1] + 1)
                    )
                    save_book(
                        output,
                        view,
                        result,
                        targets,
                        previous,
                        int(rows[0]),
                        int(rows[0]),
                        source,
                    )
                write_json_atomic(
                    output / "forecast_readout.json",
                    forecast_readout(data, rows, ranks, common, mapping, head),
                )


def summarize(root, *, confirmation=False):
    folds = (
        [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS]
        if confirmation
        else SCREEN_FOLDS
    )
    arms = (
        read(root / "phase3/screen_summary.json")["survivors"] if confirmation else ARMS
    )
    output = {"cells": {}, "survivors": [], "heldout_accessed": False}
    for arm in (*arms, *(("blend",) if len(arms) == 2 else ())):
        contrasts, seeds, summaries, forecast = {}, {}, {}, {}
        for member in MEMBERS:
            contrasts[member] = {}
            for metric in ("net_excess_bps", "utility_bps"):
                arrays = []
                for fold in folds:
                    paths = {
                        v: root / "phase3/books" / fold / arm / v / member
                        for v in VARIANTS
                    }
                    books = {v: checked_book(p) for v, p in paths.items()}
                    arrays.append(paired(books["economic"], books["neutral"], metric))
                    summaries[f"{fold}/{member}"] = {
                        v: b["summary"] for v, b in books.items()
                    }
                    forecast[f"{fold}/{member}"] = {
                        v: read(p / "forecast_readout.json") for v, p in paths.items()
                    }
                contrasts[member][metric] = {
                    "mean": float(np.concatenate(arrays).mean()),
                    "fold_means": {f: float(a.mean()) for f, a in zip(folds, arrays)},
                    "intervals": {
                        str(b): interval(arrays, block_length=b) for b in (20, 40, 60)
                    },
                }
            ic_arrays = []
            for fold in folds:
                pair = forecast[f"{fold}/{member}"]
                candidate = np.asarray(pair["economic"]["neutral_ic"], dtype=float)
                control = np.asarray(pair["neutral"]["neutral_ic"], dtype=float)
                if not np.array_equal(np.isfinite(candidate), np.isfinite(control)):
                    raise ValueError("matched objective IC populations differ")
                ic_arrays.append((candidate - control)[np.isfinite(candidate)])
            contrasts[member]["neutral_ic"] = {
                "mean": float(np.concatenate(ic_arrays).mean()),
                "intervals": {
                    str(b): interval(ic_arrays, block_length=b) for b in (20, 40, 60)
                },
                "undefined_dates_excluded_identically": True,
            }
            if member != "ensemble":
                seeds[member] = {
                    m: contrasts[member][m]["mean"]
                    for m in ("net_excess_bps", "utility_bps")
                }
        primary = contrasts["ensemble"]
        checks = {
            "positive_paired_net": primary["net_excess_bps"]["mean"] > 0,
            "positive_paired_utility": primary["utility_bps"]["mean"] > 0,
            "positive_utility_three_quarters_of_folds": sum(
                v > 0 for v in primary["utility_bps"]["fold_means"].values()
            )
            >= int(np.ceil(0.75 * len(folds))),
            "no_material_seed_reversal": min(
                v for s in seeds.values() for v in s.values()
            )
            >= -0.25,
        }
        if arm != "blend" and all(checks.values()):
            output["survivors"].append(arm)
        output["cells"][arm] = {
            "gate": checks,
            "contrasts": contrasts,
            "books": summaries,
            "forecasts": forecast,
        }
    output["interpretation"] = (
        "Nominal development results; primary ensemble is an executed rank ensemble; the blend is a prior-selection diagnostic. Old Phase1/2 fold books have a different burn-in boundary."
    )
    write_json_atomic(
        root
        / "phase3"
        / ("confirmation_summary.json" if confirmation else "screen_summary.json"),
        output,
    )
    return output


def evaluate_continuous(root, arm):
    """One account across all fourteen new forecast blocks; no P&L splicing."""
    torch.set_num_threads(1)
    if arm not in read(root / "phase3/screen_summary.json")["survivors"]:
        raise ValueError("continuous confirmation requires an admitted architecture")
    data, binding = load_data(root, arm)
    blocks = {
        fold: windows(root, data, fold)["evaluation"] for fold in DEVELOPMENT_FOLDS
    }
    rows = np.concatenate(list(blocks.values()))
    if not np.all(np.diff(rows) == 1):
        raise ValueError("continuous new forecasts must cover every intervening date")
    mappings = {
        fold: read(root / "phase3/mappings" / f"{fold}.json") for fold in blocks
    }
    models = {
        int(day): CalibratedPolicy(calibration(mappings[fold]["arms"][arm]))
        for fold, block in blocks.items()
        for day in block
    }
    for variant in VARIANTS:
        panels = {member: [] for member in MEMBERS}
        masks, sources = [], {}
        for fold, block in blocks.items():
            values, _, valid, source = read_panel(root, data, arm, variant, fold, block)
            for member in MEMBERS:
                panels[member].append(values[member])
            masks.append(valid)
            sources[fold] = {
                "forecasts": source,
                "mapping_sha256": sha256_file(
                    root / "phase3/mappings" / f"{fold}.json"
                ),
            }
        common = np.concatenate(masks)
        for member in MEMBERS:
            output = root / "phase3/continuous" / arm / variant / member
            provenance = {
                "implementation": _git_identity(),
                "scenario": "base",
                "policy": "equal_rank",
                "economic_cache_binding": binding,
                "arm": arm,
                "variant": variant,
                "member": member,
                "sources": sources,
                "inventory": "one initial cash account; actual inventory survives fold model changes; one final liquidation",
            }
            if (output / "book.json").exists():
                verify_book(output, provenance)
                continue
            view = rank_view(data, rows, np.concatenate(panels[member]), common)
            result, targets, previous = exact_replay(
                view, models, int(rows[0]), int(rows[-1] + 1)
            )
            save_book(
                output,
                view,
                result,
                targets,
                previous,
                int(rows[0]),
                int(rows[0]),
                provenance,
            )
    contrasts = {}
    for member in MEMBERS:
        books = {
            v: checked_book(root / "phase3/continuous" / arm / v / member)
            for v in VARIANTS
        }
        contrasts[member] = {
            metric: {
                str(b): interval(
                    [paired(books["economic"], books["neutral"], metric)],
                    block_length=b,
                )
                for b in (20, 40, 60)
            }
            for metric in ("net_excess_bps", "utility_bps")
        }
    write_json_atomic(
        root / "phase3/continuous" / arm / "comparison.json",
        {"contrasts": contrasts, "heldout_accessed": False},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("mappings", "evaluate", "summarize", "continuous")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fold", choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.command == "mappings":
        prepare_mappings(args.root, confirmation=args.confirmation)
    elif args.command == "evaluate":
        evaluate_fold(args.root, args.fold, confirmation=args.confirmation)
    elif args.command == "continuous":
        evaluate_continuous(args.root, args.arm)
    else:
        result = summarize(args.root, confirmation=args.confirmation)
        print({k: v for k, v in result.items() if k != "cells"})


if __name__ == "__main__":
    main()
