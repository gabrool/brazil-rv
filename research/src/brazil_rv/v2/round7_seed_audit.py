"""Six-seed and leave-one-seed-out forecast stability without redundant ledgers."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import retained
from .contract import DEVELOPMENT_FOLDS, TRADED_PRIMARY_HORIZONS
from .evaluate import _primary_daily_metrics, _primary_population_components
from .round7_program import read
from .round7_readouts import aggregate_path, economics_design, ensemble

SIX_SEEDS = (11, 29, 47, 61, 79, 97)


def run(root):
    candidate = read(root / "confirmation_leader.json")["cell"]
    design = read(root / "frozen_design.json")
    context = rr._open_ledger_replay(economics_design(design))
    panels = {str(s): {} for s in SIX_SEEDS}
    bindings = {}
    try:
        for fold in DEVELOPMENT_FOLDS:
            ix = context.evaluation[fold]
            reference = retained(
                context, aggregate_path(root, "A0", fold, SIX_SEEDS), fold
            )
            members, masks = {}, {}
            for cell in dict.fromkeys(("A0", candidate)):
                _, mask, records, values = ensemble(
                    root,
                    design,
                    cell,
                    fold,
                    SIX_SEEDS,
                    context.store.dates[ix],
                    context.store.isins,
                )
                members[cell], masks[cell] = values, mask
                bindings[f"{cell}_{fold}"] = records
            common = masks["A0"] & masks[candidate]
            for omitted in SIX_SEEDS:
                daily = {}
                for cell in members:
                    scores = rr.rank_average_ensemble(
                        [
                            m
                            for s, m in zip(SIX_SEEDS, members[cell], strict=True)
                            if s != omitted
                        ],
                        masks[cell],
                    )
                    inputs = replace(reference.inputs, scores=scores, score_mask=common)
                    _, values, _ = _primary_daily_metrics(
                        *_primary_population_components(
                            inputs, TRADED_PRIMARY_HORIZONS
                        ),
                        inputs.dates,
                        TRADED_PRIMARY_HORIZONS,
                    )
                    daily[cell] = values
                delta = daily[candidate] - daily["A0"]
                panels[str(omitted)][fold] = [
                    float(v) if np.isfinite(v) else None for v in delta
                ]
        summary = {
            s: rr._folded_bootstrap(tuple(np.asarray(v, float) for v in folds.values()))
            for s, folds in panels.items()
        }
        result = {
            "schema": "BRAZIL_RV_ROUND7_LOSO_V1",
            "status": "complete",
            "candidate": candidate,
            "seeds": SIX_SEEDS,
            "paired_primary_ic": summary,
            "all_omissions_positive": all(
                v["estimate"] is not None and v["estimate"] > 0
                for v in summary.values()
            ),
            "daily_by_fold": panels,
            "source_trajectories": bindings,
            "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
            "ledger_rule": "use completed matched six-seed economics; this registered omission gate concerns forecast IC only",
        }
        write_json_atomic(root / "six_seed_loso.json", result)
        return result
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
