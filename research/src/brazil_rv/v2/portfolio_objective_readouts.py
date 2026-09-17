"""Two selectors per trained trajectory, magnitude-preserving ensemble books."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
from pathlib import Path

import numpy as np
import torch

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS
from .opportunity_research import benchmark_for, bound, checked
from .performance import performance
from .portfolio_objective import TensorPreference
from .portfolio_objective_training import preparation, readout, VARIANTS
from .portfolio_program import PROJECT, read
from .portfolio_readouts import interval, save_book, verify_book
from .portfolio_training import load_data, windows
from .research_rounds import _git_identity
from .round7 import SCREEN_FOLDS
from .round7_training import daily_primary_ic
from .train import compile_forward, rank_average_ensemble, set_deterministic_seed


SELECTORS = ("ic", "utility_bps")


def evaluate(root, arm, fold, *, confirmation=False):
    torch.set_num_threads(1)
    design = read(root / "training_design.json")
    if _git_identity() != design["implementation"]:
        raise ValueError("objective readout differs from frozen implementation")
    old = Path(design["decision_root"])
    checked(design["caches"][arm])
    data, _ = load_data(old, arm)
    bounds = windows(old, data, fold)
    variants = (
        ("rank", *read(root / "screen_summary.json")["survivors"][arm])
        if confirmation
        else VARIANTS
    )
    forecasts = {}
    for seed in ALLOWED_SEEDS:
        set_deterministic_seed(seed)
        model, caches, axes, view, _ = preparation(
            design, data, arm, fold, seed, include_evaluation=True
        )
        neural_forward = compile_forward(model)
        for variant in variants:
            directory = root / "fits" / arm / variant / f"{fold}_seed_{seed}"
            manifest = read(directory / "run_manifest.json")
            for selector in (*SELECTORS, "initial") if variant == "rank" else SELECTORS:
                file = (
                    "initial.pt" if selector == "initial" else f"selected_{selector}.pt"
                )
                checkpoint = directory / file
                if sha256_file(checkpoint) != manifest["files"][file]:
                    raise ValueError("objective readout checkpoint changed")
                label = "initial" if selector == "initial" else variant
                key = (label, selector)
                output = root / "readouts" / arm / fold / label / selector / str(seed)
                output.mkdir(parents=True, exist_ok=True)
                forecast = output / "forecast.npz"
                binding = {
                    "checkpoint": bound(checkpoint),
                    "design": bound(root / "training_design.json"),
                }
                if forecast.exists():
                    receipt = read(output / "forecast.json")
                    if (
                        receipt["binding"] != binding
                        or sha256_file(forecast) != receipt["sha256"]
                    ):
                        raise ValueError("saved objective forecast differs")
                    with np.load(forecast, allow_pickle=False) as p:
                        panel = {k: p[k].copy() for k in p.files}
                else:
                    state = torch.load(
                        checkpoint, map_location="cuda", weights_only=True
                    )
                    model.load_state_dict(state["model"], strict=True)
                    _, (scores, preferences), _ = readout(
                        model,
                        neural_forward,
                        caches["evaluation"],
                        axes["evaluation"],
                        view,
                    )
                    panel = {
                        "scores": scores,
                        "preferences": preferences,
                        "rows": axes["evaluation"],
                        "valid": view.valid[axes["evaluation"]],
                        "isins": np.asarray(data.inputs.security_ids),
                    }
                    np.savez_compressed(forecast, **panel)
                    write_json_atomic(
                        output / "forecast.json",
                        {
                            "binding": binding,
                            "sha256": sha256_file(forecast),
                            "epoch": state["epoch"],
                        },
                    )
                forecasts.setdefault(key, {})[str(seed)] = (
                    panel,
                    bound(output / "forecast.json"),
                )
        del neural_forward, model, caches
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    benchmark_root = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    for (variant, selector), members in forecasts.items():
        panels = [members[str(s)][0] for s in ALLOWED_SEEDS]
        for p in panels[1:]:
            if not np.array_equal(p["valid"], panels[0]["valid"]) or not np.array_equal(
                p["rows"], panels[0]["rows"]
            ):
                raise ValueError(
                    "objective ensembles may not intersect different populations"
                )
        mask = np.repeat(panels[0]["valid"][..., None], 3, -1)
        ensemble = {
            **panels[0],
            "preferences": np.mean([p["preferences"] for p in panels], axis=0),
            "scores": rank_average_ensemble([p["scores"] for p in panels], mask),
        }
        members["ensemble"] = (
            ensemble,
            {"members": [members[str(s)][1] for s in ALLOWED_SEEDS]},
        )
        for member, (panel, binding) in members.items():
            rows = panel["rows"]
            target = data.inputs.neutral_midrank_targets[rows][..., [2, 3, 4]]
            valid = data.inputs.neutral_target_mask[rows][..., [2, 3, 4]]
            ic = daily_primary_ic(panel["scores"], target, valid, panel["valid"])
            first = int(bounds["evaluation"][0])
            trim = first - int(rows[0])
            scenarios = {"base": (AllocationConfig(), {})}
            # Secondary assumption checks use the same selected ensemble and
            # preferences. No retraining, reselection or economic cherry-picking.
            if member == "ensemble":
                scenarios.update(
                    {
                        "flexible_net": (replace(AllocationConfig(), net_cap=0.45), {}),
                        "cost8": (
                            AllocationConfig(),
                            {"cost_bps_per_side": 8.0, "hedge_cost_bps_per_side": 8.0},
                        ),
                        "proceeds0": (
                            AllocationConfig(),
                            {"short_proceeds_remuneration": 0.0},
                        ),
                    }
                )
            for scenario, (allocation, changes) in scenarios.items():
                output = (
                    root
                    / "readouts"
                    / arm
                    / fold
                    / variant
                    / selector
                    / member
                    / scenario
                )
                provenance = {
                    "scenario": scenario,
                    "policy": variant,
                    "selector": selector,
                    "member": member,
                    "forecasts": binding,
                    "allocation": allocation.__dict__,
                    "ledger_changes": changes,
                    "benchmark": bound(benchmark_root / "inputs/benchmarks.npz"),
                }
                if (output / "book.json").exists():
                    book = verify_book(output, provenance)
                else:
                    values = torch.from_numpy(panel["preferences"])
                    result, targets, previous = exact_replay(
                        data,
                        TensorPreference(values, int(rows[0])),
                        int(rows[0]),
                        int(rows[-1] + 1),
                        allocation=allocation,
                        config=policy_ledger_config(
                            planned_absolute_net_cap=allocation.net_cap, **changes
                        ),
                    )
                    book = save_book(
                        output,
                        data,
                        result,
                        targets,
                        previous,
                        int(rows[0]),
                        first,
                        provenance,
                    )
                daily = book["daily"]
                dates = np.asarray(book["dates"], dtype="datetime64[D]")
                benchmarks = benchmark_for(
                    benchmark_root, dates, data.inputs.dates[first - 1]
                )
                metrics = performance(
                    np.asarray(daily["absolute_bps"]) / 1e4,
                    np.asarray(daily["cdi_bps"]) / 1e4,
                    benchmarks,
                )
                write_json_atomic(
                    output / "metrics.json",
                    {
                        "performance": metrics,
                        "ic": float(np.nanmean(ic[trim:])),
                        "daily_ic": [
                            float(x) if np.isfinite(x) else None for x in ic[trim:]
                        ],
                        "book": bound(output / "book.json"),
                    },
                )
    write_json_atomic(
        root / "readouts" / arm / fold / "complete.json",
        {"status": "completed", "variants": list(variants), "selectors": SELECTORS},
    )


def summarize(root, *, confirmation=False):
    design = read(root / "training_design.json")
    folds = (
        [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS]
        if confirmation
        else SCREEN_FOLDS
    )
    admitted = read(root / "screen_summary.json")["survivors"] if confirmation else None
    summary = {
        "design": bound(root / "training_design.json"),
        "folds": list(folds),
        "arms": {},
        "survivors": {},
        "promoted": [],
    }
    for arm in design["recipes"]:
        if confirmation and arm not in admitted:
            continue
        variants = ("rank", *admitted[arm]) if confirmation else VARIANTS
        cells = {}

        def books(variant, selector, member, scenario="base"):
            return [
                read(
                    root
                    / "readouts"
                    / arm
                    / f
                    / variant
                    / selector
                    / member
                    / scenario
                    / "book.json"
                )
                for f in folds
            ]

        for variant in variants:
            for selector in SELECTORS:
                own = books(variant, selector, "ensemble")
                control = books("rank", selector, "ensemble")
                differences = {
                    key: [
                        np.asarray(a["daily"][key]) - np.asarray(b["daily"][key])
                        for a, b in zip(own, control)
                    ]
                    for key in ("net_excess_bps", "utility_bps")
                }
                seed_means = {}
                for seed in ALLOWED_SEEDS:
                    candidate = books(variant, selector, str(seed))
                    baseline = books("rank", selector, str(seed))
                    seed_means[str(seed)] = float(
                        np.mean(
                            np.concatenate(
                                [
                                    np.asarray(a["daily"]["utility_bps"])
                                    - np.asarray(b["daily"]["utility_bps"])
                                    for a, b in zip(candidate, baseline)
                                ]
                            )
                        )
                    )
                record = {
                    "mean": {
                        k: float(np.mean(np.concatenate([b["daily"][k] for b in own])))
                        for k in own[0]["daily"]
                    },
                    "paired": {
                        key: {
                            str(block): interval(arrays, block_length=block)
                            for block in (20, 40, 60)
                        }
                        for key, arrays in differences.items()
                    },
                    "fold_paired_utility": {
                        f: float(x.mean())
                        for f, x in zip(folds, differences["utility_bps"])
                    },
                    "seed_paired_utility": seed_means,
                    "economics_unresolved": any(
                        b["summary"]["economics_unresolved"] for b in own
                    ),
                    "fold_metrics": {
                        f: read(
                            root
                            / "readouts"
                            / arm
                            / f
                            / variant
                            / selector
                            / "ensemble/base/metrics.json"
                        )
                        for f in folds
                    },
                    "stresses": {
                        scenario: {
                            k: float(
                                np.mean(
                                    np.concatenate(
                                        [
                                            b["daily"][k]
                                            for b in books(
                                                variant, selector, "ensemble", scenario
                                            )
                                        ]
                                    )
                                )
                            )
                            for k in (
                                "net_excess_bps",
                                "utility_bps",
                                "gross",
                                "signed_net",
                                "beta",
                            )
                        }
                        for scenario in ("flexible_net", "cost8", "proceeds0")
                    },
                }
                for label, comparison in (
                    (
                        "paired_against_warm_start",
                        books("initial", "initial", "ensemble"),
                    ),
                    ("selector_minus_ic", books(variant, "ic", "ensemble")),
                ):
                    record[label] = {
                        key: interval(
                            [
                                np.asarray(a["daily"][key])
                                - np.asarray(b["daily"][key])
                                for a, b in zip(own, comparison)
                            ]
                        )
                        for key in ("net_excess_bps", "utility_bps")
                    }
                cells[f"{variant}:{selector}"] = record
                if selector == "utility_bps" and variant != "rank":
                    positive_seeds = sum(v > 0 for v in seed_means.values()) >= 2
                    if not confirmation:
                        passes = (
                            positive_seeds
                            and sum(x.mean() > 0 for x in differences["utility_bps"])
                            >= 3
                            and all(
                                record["paired"][key]["40"]["estimate"] > 0
                                for key in differences
                            )
                        )
                        if passes:
                            summary["survivors"].setdefault(arm, []).append(variant)
                    elif positive_seeds and all(
                        record["paired"][key]["40"]["lower_95"] > 0
                        for key in differences
                    ):
                        summary["promoted"].append(
                            {"arm": arm, "objective": variant, "research_only": True}
                        )
        summary["arms"][arm] = cells
    write_json_atomic(
        root / ("confirmation_summary.json" if confirmation else "screen_summary.json"),
        summary,
    )
    return summary


def continuous(root):
    """Roll confirmed selected models through one account, without fold resets."""
    torch.set_num_threads(1)
    design = read(root / "training_design.json")
    old = Path(design["decision_root"])
    admitted = read(root / "screen_summary.json")["survivors"]
    benchmark_root = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    result_summary = {}
    for arm, variants in admitted.items():
        checked(design["caches"][arm])
        data, _ = load_data(old, arm)
        for variant in ("rank", *variants):
            for selector in SELECTORS:
                panels = {
                    str(s): np.full_like(data.valid, np.nan, dtype=float)
                    for s in ALLOWED_SEEDS
                }
                sources = []
                for fold in DEVELOPMENT_FOLDS:
                    bounds = windows(old, data, fold)
                    for seed in ALLOWED_SEEDS:
                        directory = (
                            root
                            / "readouts"
                            / arm
                            / fold
                            / variant
                            / selector
                            / str(seed)
                        )
                        receipt = read(directory / "forecast.json")
                        path = directory / "forecast.npz"
                        if sha256_file(path) != receipt["sha256"]:
                            raise ValueError("continuous objective forecast changed")
                        with np.load(path, allow_pickle=False) as p:
                            use = (
                                np.ones(len(p["rows"]), dtype=bool)
                                if fold == "F1"
                                else p["rows"] >= bounds["evaluation"][0]
                            )
                            panels[str(seed)][p["rows"][use]] = p["preferences"][use]
                            if fold == "F1":
                                start = int(p["rows"][0])
                            stop = int(p["rows"][-1] + 1)
                        sources.append(bound(directory / "forecast.json"))
                panels["ensemble"] = np.mean(list(panels.values()), axis=0)
                first = int(windows(old, data, "F1")["evaluation"][0])
                for member, values in panels.items():
                    if not np.isfinite(values[start:stop]).all():
                        raise ValueError(
                            "continuous objective account has a missing decision"
                        )
                    output = root / "continuous" / arm / variant / selector / member
                    provenance = {
                        "scenario": "base",
                        "policy": variant,
                        "selector": selector,
                        "member": member,
                        "forecasts": sources,
                        "allocation": AllocationConfig().__dict__,
                    }
                    if (output / "book.json").exists():
                        book = verify_book(output, provenance)
                    else:
                        result, targets, previous = exact_replay(
                            data,
                            TensorPreference(torch.from_numpy(values)),
                            start,
                            stop,
                        )
                        book = save_book(
                            output,
                            data,
                            result,
                            targets,
                            previous,
                            start,
                            first,
                            provenance,
                        )
                    daily = book["daily"]
                    bench = benchmark_for(
                        benchmark_root,
                        np.asarray(book["dates"], dtype="datetime64[D]"),
                        data.inputs.dates[first - 1],
                    )
                    metrics = performance(
                        np.asarray(daily["absolute_bps"]) / 1e4,
                        np.asarray(daily["cdi_bps"]) / 1e4,
                        bench,
                    )
                    write_json_atomic(
                        output / "metrics.json",
                        {"book": bound(output / "book.json"), "performance": metrics},
                    )
                    result_summary[f"{arm}:{variant}:{selector}:{member}"] = {
                        "book": bound(output / "book.json"),
                        "performance": metrics,
                        "mean": book["summary"]["mean"],
                        "economics_unresolved": book["summary"]["economics_unresolved"],
                    }
    write_json_atomic(root / "continuous_summary.json", result_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "summarize", "continuous"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("C6", "TE_all"))
    parser.add_argument("--fold", choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate(args.root, args.arm, args.fold, confirmation=args.confirmation)
    elif args.command == "summarize":
        summarize(args.root, confirmation=args.confirmation)
    else:
        continuous(args.root)


if __name__ == "__main__":
    main()
