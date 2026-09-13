"""Paired temporal-factorial inference from retained, matched daily panels."""

import argparse
import math
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .data_roots import resolve_external_root
from .round7_pathway import IC, NET
from .round7_program import read

CONTRASTS = {
    "GE_minus_GL": {"GE": 1, "GL": -1},
    "TE_minus_TL": {"TE": 1, "TL": -1},
    "TL_minus_GL": {"TL": 1, "GL": -1},
    "TE_minus_GE": {"TE": 1, "GE": -1},
    "interaction": {"TE": 1, "TL": -1, "GE": -1, "GL": 1},
}


def infer(arrays):
    """NW lag 10 keeps gaps and fold boundaries; bootstrap uses the same dates."""
    arrays = tuple(np.asarray(a, float) for a in arrays)
    result = rr._folded_bootstrap(arrays)
    n = sum(np.isfinite(a).sum() for a in arrays)
    if n < 2:
        return {**result, "nw_lag10_se": None, "nominal_two_sided_p": None}
    mean = result["estimate"]
    variance = 0.0
    for a in arrays:
        centered = np.where(np.isfinite(a), a - mean, 0.0)
        variance += float(centered @ centered)
        for lag in range(1, min(10, len(a) - 1) + 1):
            variance += 2 * (1 - lag / 11) * float(centered[lag:] @ centered[:-lag])
    se = math.sqrt(max(variance, 0.0)) / n
    p = math.erfc(abs(mean) / (math.sqrt(2) * se)) if se else float(mean == 0)
    return {**result, "nw_lag10_se": se, "nominal_two_sided_p": p}


def holm_primary(rows):
    """The two prespecified pathway tests remain the family after screening."""
    keys = ("GE_minus_GL", "TE_minus_TL")
    tests = sorted(
        (rows[k][IC]["nominal_two_sided_p"], k)
        for k in keys
        if k in rows and rows[k][IC]["nominal_two_sided_p"] is not None
    )
    running = 0.0
    for index, (p, key) in enumerate(tests):
        running = min(1.0, max(running, (2 - index) * p))
        rows[key][IC]["holm_two_primary_tests_p"] = running


def run(root, group):
    source = root / f"{group}_result.json"
    result = read(source)
    panels, masks, hashes = {}, {}, {}
    for cell in CONTRASTS["interaction"]:
        if cell not in result["aggregate_paths"]:
            continue
        panels[cell], masks[cell] = {}, {}
        for fold, location in result["aggregate_paths"][cell].items():
            path = resolve_external_root(location)[0]
            panel = read(path / "daily_readouts.json")
            # All four cells have the same three traded heads. Equality of
            # masks makes their daily IC contrasts genuinely population matched.
            _, mask = rr._score_artifact(path, require_clean_transfer=True)
            panels[cell][fold] = panel
            masks[cell][fold] = mask
            hashes[f"{cell}/{fold}"] = {
                name: sha256_file(path / name)
                for name in ("daily_readouts.json", "score_manifest.json")
            }
    rows, unavailable = {}, []
    for contrast, weights in CONTRASTS.items():
        if not set(weights) <= panels.keys():
            unavailable.append(contrast)
            continue
        first = next(iter(weights))
        values = {IC: [], NET: []}
        folds = {}
        for fold in result["folds"]:
            reference = panels[first][fold]
            for cell in weights:
                if panels[cell][fold]["dates"] != reference[
                    "dates"
                ] or not np.array_equal(masks[cell][fold], masks[first][fold]):
                    raise ValueError("factorial dates or score populations differ")
            folds[fold] = {}
            for metric in values:
                delta = sum(
                    weight * np.asarray(panels[cell][fold]["series"][metric], float)
                    for cell, weight in weights.items()
                )
                values[metric].append(delta)
                folds[fold][metric] = [
                    float(v) if np.isfinite(v) else None for v in delta
                ]
        rows[contrast] = {
            **{metric: infer(arrays) for metric, arrays in values.items()},
            "daily_by_fold": folds,
        }
    holm_primary(rows)
    comparisons = {}
    for key, row in result["paired"].items():
        if "opposite_of" in row:
            continue
        values = {IC: [], NET: []}
        for fold in result["folds"]:
            pair = read(root / "paired" / group / key / f"{fold}.json")
            for metric in values:
                values[metric].append(
                    np.array(
                        [r["delta"] for r in pair["population_audit"][fold][metric]],
                        float,
                    )
                )
        comparisons[key] = {metric: infer(arrays) for metric, arrays in values.items()}
    report = {
        "contrasts": rows,
        "all_pairwise_comparisons": comparisons,
        "unavailable_contrasts": unavailable,
        "seeds": result["seeds"],
        "folds": result["folds"],
        "source_result_sha256": sha256_file(source),
        "sources": hashes,
        "interpretation": "primary placement family corrected with Holm; encoder and interaction secondary; all development-selected",
        "missing_contrasts": "not estimated when their full matched cell panel was not advanced",
    }
    write_json_atomic(root / f"{group}_factorial.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--group", required=True)
    args = parser.parse_args()
    run(args.root, args.group)


if __name__ == "__main__":
    main()
