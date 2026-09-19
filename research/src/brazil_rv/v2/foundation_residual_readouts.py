"""Matched neutral accounts for the registered residual-information screen."""

from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS
from .foundation_readouts import book, compare
from .objective_readouts import calibration
from .opportunity_research import bound
from .portfolio_program import PROJECT, read
from .portfolio_training import load_data, windows
from .research_rounds import _git_identity


def evaluate(root, folds):
    torch.set_num_threads(1)
    design = read(root / "frozen_design.json")
    prior = Path(design["prior_decision_root"])
    data, binding = load_data(prior, "C6")
    store_dates = np.load(Path(design["store"]["root"]) / "date_index.npy")
    benchmark = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    implementation = _git_identity()
    cells = ("ridge_score", "ridge_rich", "tree_score", "tree_rich")
    for fold in folds:
        rows = windows(prior, data, fold)["evaluation"]
        active = data.inputs.active[rows]
        day, name = np.nonzero(active)
        mapping_path = prior / "phase3/mappings" / f"{fold}.json"
        mapping = calibration(read(mapping_path)["arms"]["C6"])
        for cell in cells:
            panels, sources = {}, {}
            for seed in ALLOWED_SEEDS:
                fit = root / "residual" / cell / f"{fold}_seed_{seed}"
                result = read(fit / "result.json")
                if result["score_sha256"] != sha256_file(fit / "scores.npz"):
                    raise ValueError("residual forecasts changed")
                if (
                    result["binding"]["seed"] != seed
                    or result["binding"]["fold"] != fold
                ):
                    raise ValueError("residual fit identity differs")
                with np.load(fit / "scores.npz") as scores:
                    if not np.array_equal(
                        store_dates[scores["dates"]],
                        np.asarray(data.inputs.dates, dtype="datetime64[D]")[rows[day]],
                    ) or not np.array_equal(scores["names"], name):
                        raise ValueError(
                            "residual forecast population differs from original control"
                        )
                    values = scores["scores"]
                if not np.isfinite(values).all():
                    raise ValueError("nonfinite residual predictions")
                panel = np.zeros((*active.shape, 3), np.float64)
                for d in range(len(rows)):
                    mask = day == d
                    count = mask.sum()
                    if count:
                        panel[d, name[mask]] = (
                            2 * (rankdata(values[mask], axis=0) - 0.5) / count - 1
                        )
                panels[str(seed)] = panel
                sources[str(seed)] = {
                    "result": bound(fit / "result.json"),
                    "scores": bound(fit / "scores.npz"),
                }
            panels["ensemble"] = np.mean(list(panels.values()), axis=0)
            for member, ranks in panels.items():
                book(
                    root / "books" / cell / "raw" / fold / member,
                    data,
                    rows,
                    ranks,
                    active,
                    mapping,
                    {
                        "implementation": implementation,
                        "scenario": "base",
                        "policy": "equal_rank",
                        "economic_cache_binding": binding,
                        "mapping": bound(mapping_path),
                        "forecasts": sources,
                        "member": member,
                        "initial_state": "cash at first evaluation date",
                        "benchmark": bound(benchmark / "inputs/benchmarks.npz"),
                    },
                    benchmark,
                )
            print({"residual_readout": [cell, fold]}, flush=True)
    references = {
        (fold, member): prior / "phase3/books" / fold / "C6/neutral" / member
        for fold in folds
        for member in (*map(str, ALLOWED_SEEDS), "ensemble")
    }
    results = {}
    for learner in ("ridge", "tree"):
        rich, score = f"{learner}_rich", f"{learner}_score"
        versus_anchor = compare(
            root,
            rich,
            "C6",
            folds,
            allow_noninferiority=False,
            control_paths=references,
        )
        versus_score = compare(root, rich, score, folds, allow_noninferiority=False)
        results[learner] = {
            "versus_anchor": versus_anchor,
            "versus_score": versus_score,
            "admitted": versus_anchor["admitted"]
            and versus_score["screen_improvement"],
        }
    write_json_atomic(root / "residual_summary.json", results)
