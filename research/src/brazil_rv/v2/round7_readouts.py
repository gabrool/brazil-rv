"""Shared-population Round-7 score, momentum and adopted-policy readouts."""

from __future__ import annotations

import argparse
from itertools import combinations
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import candidate_readout, paired_readouts, retained
from .contract import DEVELOPMENT_FOLDS, TRADED_PRIMARY_HORIZONS
from .data_roots import resolve_external_root
from .evaluate import _primary_daily_metrics, _primary_population_components
from .research_checkpoint import _completed, _context_arguments, _finish_cell
from .research_diagnostics import momentum_diagnostics, pooled_momentum_diagnostics
from .round6 import resolve_file
from .round6_readouts import evaluation_design
from .round7 import SEEDS, SCREEN_FOLDS
from .round7_program import read, trajectory

NEIGHBOURS = {
    "A1": "A0",
    "A2": "A1",
    "A3": "A2",
    "B1": "A1",
    "B2": "B1",
    "B3": "B2",
    "B4": "B3",
    "B5": "B4",
    "B6": "B4",
    "B7": "B4",
    "B8": "B4",
    "B9": "B4",
    "B10": "B4",
    "B11": "B4",
}


def economics_design(design):
    path = resolve_file(design["evaluation_design"])
    return evaluation_design({"evaluation_sources": {"design": read(path)}})


def ensemble(root, design, cell, fold, seeds, dates, isins):
    members, mask, records = [], None, []
    for seed in seeds:
        directory = trajectory(root, cell, fold, seed)
        manifest = read(directory / "run_manifest.json")
        if manifest["status"] != "completed":
            raise ValueError("cannot read an incomplete trajectory")
        scores, valid = rr._score_artifact(
            directory / "scores",
            require_clean_transfer=True,
            expected_dates=dates,
            expected_isins=isins,
            expected_feature_schema_sha256=design["feature_schema_sha256"],
        )
        if mask is not None and not np.array_equal(valid, mask):
            raise ValueError("matched seeds have different prediction populations")
        mask = valid
        members.append(scores)
        records.append(
            {
                "seed": seed,
                "epochs": manifest["epochs_completed"],
                "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
                "score_manifest_sha256": sha256_file(
                    directory / "scores/score_manifest.json"
                ),
            }
        )
    return rr.rank_average_ensemble(members, mask), mask, records, members


def aggregate_path(root, cell, fold, seeds):
    return root / "aggregates" / ("seeds_" + "_".join(map(str, seeds))) / cell / fold


def evaluate(root, cells, folds, group, seeds=SEEDS, *, cells_only=False):
    design = read(root / "frozen_design.json")
    economic = economics_design(design)
    context = rr._open_ledger_replay(economic)
    policy, _ = rr.load_selected_policy(
        Path(economic["execution_policy"]["root"]),
        expected_result_sha256=economic["execution_policy"]["result_sha256"],
    )
    hashes = {
        "v2_store_manifest": design["store"]["manifest_sha256"],
        "round7_frozen_design": sha256_file(root / "frozen_design.json"),
    }
    paths = {cell: {} for cell in cells}
    try:
        for fold in folds:
            ix = context.evaluation[fold]
            for cell in cells:
                output = aggregate_path(root, cell, fold, seeds)
                paths[cell][fold] = output
                if _completed(output):
                    metadata = read(output / "score_manifest.json")["metadata"]
                    if (
                        metadata["seeds"] != list(seeds)
                        or metadata["round7_frozen_design_sha256"]
                        != hashes["round7_frozen_design"]
                    ):
                        raise ValueError(
                            "retained aggregate differs from the requested comparison"
                        )
                    continue
                scores, mask, records, members = ensemble(
                    root,
                    design,
                    cell,
                    fold,
                    seeds,
                    context.store.dates[ix],
                    context.store.isins,
                )
                if (output / "score_manifest.json").exists():
                    raise RuntimeError(f"partial aggregate needs diagnosis: {output}")
                rr._persist_scores(
                    output,
                    {"scores": scores, "score_mask": mask},
                    {
                        **rr._source_tier_labels(context.store.manifest),
                        "arm": cell,
                        "fold": fold,
                        "seeds": list(seeds),
                        "evaluation_date_indices": ix.tolist(),
                        "trajectories": records,
                        "round7_frozen_design_sha256": hashes["round7_frozen_design"],
                    },
                )
                evaluated = rr._evaluate(
                    **_context_arguments(context, hashes),
                    indices=ix,
                    scores=scores,
                    score_mask=mask,
                    fold=fold,
                    output=output / "evaluation.json",
                    execution_policy=policy,
                    settle_terminal_residuals=True,
                )
                seed_ic = {}
                for seed, values in zip(seeds, members, strict=True):
                    single = replace(evaluated.inputs, scores=values)
                    _, daily, _ = _primary_daily_metrics(
                        *_primary_population_components(
                            single, TRADED_PRIMARY_HORIZONS
                        ),
                        single.dates,
                        TRADED_PRIMARY_HORIZONS,
                    )
                    seed_ic[str(seed)] = [
                        float(v) if np.isfinite(v) else None for v in daily
                    ]
                write_json_atomic(output / "seed_ic.json", seed_ic)
                _finish_cell(output, evaluated, name=cell, fold=fold)
                print(f"accepted {cell}/{fold} seeds={seeds}", flush=True)
        if cells_only:
            return {"status": "cells_complete", "cells": len(cells) * len(folds)}
        readouts = {cell: candidate_readout(p) for cell, p in paths.items()}
        requested = {(cell, "A0") for cell in cells if cell != "A0" and "A0" in cells}
        requested |= {
            (cell, NEIGHBOURS[cell])
            for cell in cells
            if cell in NEIGHBOURS and NEIGHBOURS[cell] in cells
        }
        if "B4" in cells and "A3" in cells:
            requested.add(("B4", "A3"))
        if "B3" in cells and "B1" in cells:
            requested.add(("B3", "B1"))
        if group == "confirmation":
            requested |= set(combinations(cells, 2))
        pairs = {}
        for left, right in sorted(requested):
            pairs.update(
                paired_readouts(
                    context,
                    {left: paths[left], right: paths[right]},
                    root / "paired" / group,
                )
            )
        momentum = {cell: {} for cell in cells}
        baseline = resolve_external_root(economic["cpu_checkpoint"]["root"])[0]
        for fold in folds:
            m_score, m_mask = rr._score_artifact(
                baseline / "baselines/momentum_12_1" / fold, require_clean_transfer=True
            )
            for cell in cells:
                candidate = retained(context, paths[cell][fold], fold)
                momentum[cell][fold] = momentum_diagnostics(
                    candidate.inputs, m_score, m_mask
                )
        seed_points = {}
        for cell in cells:
            seed_points[cell] = {
                str(seed): float(
                    np.nanmean(
                        np.asarray(
                            [
                                value
                                for fold in folds
                                for value in read(paths[cell][fold] / "seed_ic.json")[
                                    str(seed)
                                ]
                            ],
                            float,
                        )
                    )
                )
                for seed in seeds
            }
        result = {
            "schema": "BRAZIL_RV_ROUND7_READOUT_V1",
            "status": "completed",
            "group": group,
            "cells": list(cells),
            "folds": list(folds),
            "seeds": list(seeds),
            "readouts": readouts,
            "paired": pairs,
            "seed_primary_ic": seed_points,
            "momentum": {
                c: {"pooled": pooled_momentum_diagnostics(v), "folds": v}
                for c, v in momentum.items()
            },
            "screen_label": "screened_on_F2_F6_F10_F14"
            if group != "anchor"
            else "fixed_repaired_anchor",
            "frozen_design_sha256": hashes["round7_frozen_design"],
            **rr.RESEARCH_FLAGS,
        }
        write_json_atomic(root / f"{group}_result.json", result)
        return result
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--folds", nargs="+", default=list(DEVELOPMENT_FOLDS))
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--cells-only", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.root,
        args.cells,
        SCREEN_FOLDS if args.screen else args.folds,
        args.group,
        args.seeds,
        cells_only=args.cells_only,
    )


if __name__ == "__main__":
    main()
