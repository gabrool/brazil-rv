"""Fit-period, within-family dependence audit. No labels or automatic deletions."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.portfolio_program import read
from brazil_rv.v2.round7_preprocessing import common_snapshot, cross_market_partition
from brazil_rv.v2.train import _cli_stage_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    design = read(args.root / "frozen_design.json")
    store = Path(design["store"]["root"])
    manifest = read(store / "manifest.json")
    if sha256_file(store / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("field audit store changed")
    active = np.load(store / "active.npy", mmap_mode="r")
    results = {}
    for stage, fold in (("P", "pretrain_internal"), ("F", "F14")):
        rows = _cli_stage_indices(store, stage, fold)[0]
        days, names = np.nonzero(active[rows])
        # A fixed outcome-independent sample bounds CPU work. Common shocks use
        # one observation per date, never repeated once per active stock.
        selection = np.linspace(0, len(days) - 1, min(50_000, len(days)), dtype=int)
        pair_rows = rows[days[selection]], names[selection]
        families = {}
        for family, fields in manifest["feature_names"].items():
            if not family.startswith("sidecar_"):
                continue
            common = (
                cross_market_partition(fields)[0]
                if family == "sidecar_cross_market"
                else ()
            )
            arrays = {
                s: np.load(
                    store / manifest["arrays"][family + "_" + s]["path"], mmap_mode="r"
                )
                for s in ("values", "valid", "age_sessions")
            }
            sampled = {}
            for f, name in enumerate(fields):
                if name in common:
                    sampled[f] = common_snapshot(
                        arrays["values"][rows, :, f : f + 1],
                        arrays["valid"][rows, :, f : f + 1] & active[rows, :, None],
                        np.where(
                            active[rows, :, None],
                            arrays["age_sessions"][rows, :, f : f + 1],
                            -1,
                        ),
                    )
                    sampled[f] = tuple(v[:, 0] for v in sampled[f])
                else:
                    sampled[f] = tuple(
                        arrays[s][pair_rows[0], pair_rows[1], f]
                        for s in ("values", "valid", "age_sessions")
                    )
            highly_related, tested, insufficient = [], 0, 0
            for first, second in combinations(range(len(fields)), 2):
                a_common, b_common = fields[first] in common, fields[second] in common
                if a_common != b_common:
                    continue
                va, vb = sampled[first][1], sampled[second][1]
                both = va & vb
                if both.sum() < 100:
                    insufficient += 1
                    continue
                x, y = (sampled[f][0][both] for f in (first, second))
                if np.ptp(x) == 0 or np.ptp(y) == 0:
                    continue
                correlation = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
                tested += 1
                if abs(correlation) >= 0.97:
                    highly_related.append(
                        {
                            "fields": [fields[first], fields[second]],
                            "spearman": correlation,
                            "overlap": int(both.sum()),
                            "common_field_date_observations": a_common,
                            "sample_masks_equal": bool(np.array_equal(va, vb)),
                            "sample_observed_values_equal": bool(np.array_equal(x, y)),
                            "sample_ages_equal": bool(
                                np.array_equal(
                                    sampled[first][2],
                                    sampled[second][2],
                                )
                            ),
                        }
                    )
            families[family] = {
                "tested_pairs": tested,
                "fewer_than_100_joint_observations": insufficient,
                "highly_related": highly_related,
            }
        results[fold] = {"fit_date_indices": rows.tolist(), "families": families}
    write_json_atomic(
        args.root / "field_overlap.json",
        {
            "script_sha256": sha256_file(Path(__file__)),
            "store": design["store"],
            "fit_periods": results,
            "labels_accessed": False,
            "automatic_deletions": [],
            "interpretation": "Exploratory support/dependence only; high rank correlation is not redundancy or absence of conditional information. No evaluation rows in each fit-period calculation.",
        },
    )
    print(
        {
            fold: {
                family: len(v["highly_related"]) for family, v in r["families"].items()
            }
            for fold, r in results.items()
        }
    )


if __name__ == "__main__":
    main()
