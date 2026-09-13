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
from .data_roots import resolve_external_root
from .evaluate import _primary_daily_metrics, _primary_population_components
from .round7_program import read
from .round7_readouts import aggregate_path, economics_design, ensemble

SIX_SEEDS = (11, 29, 47, 61, 79, 97)


def run(root, *, candidate=None, reference="A0", seeds=SIX_SEEDS, group="six_seed"):
    candidate = candidate or read(root / "confirmation_leader.json")["cell"]
    design = read(root / "frozen_design.json")
    context = rr._open_ledger_replay(economics_design(design))
    panels = {str(s): {} for s in seeds}
    bindings = {}
    sources = read(root / f"{group}_result.json").get("aggregate_paths", {})
    try:
        for fold in DEVELOPMENT_FOLDS:
            ix = context.evaluation[fold]
            reference_path = sources.get(reference, {}).get(fold)
            base = retained(
                context,
                resolve_external_root(reference_path)[0]
                if reference_path
                else aggregate_path(root, reference, fold, seeds),
                fold,
            )
            members, masks = {}, {}
            for cell in dict.fromkeys((reference, candidate)):
                _, mask, records, values = ensemble(
                    root,
                    design,
                    cell,
                    fold,
                    seeds,
                    context.store.dates[ix],
                    context.store.isins,
                )
                members[cell], masks[cell] = values, mask
                bindings[f"{cell}_{fold}"] = records
            common = masks[reference] & masks[candidate]
            for omitted in seeds:
                daily = {}
                for cell in members:
                    scores = rr.rank_average_ensemble(
                        [
                            m
                            for s, m in zip(seeds, members[cell], strict=True)
                            if s != omitted
                        ],
                        masks[cell],
                    )
                    inputs = replace(base.inputs, scores=scores, score_mask=common)
                    _, values, _ = _primary_daily_metrics(
                        *_primary_population_components(
                            inputs, TRADED_PRIMARY_HORIZONS
                        ),
                        inputs.dates,
                        TRADED_PRIMARY_HORIZONS,
                    )
                    daily[cell] = values
                delta = daily[candidate] - daily[reference]
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
            "reference": reference,
            "seeds": seeds,
            "paired_primary_ic": summary,
            "all_omissions_positive": all(
                v["estimate"] is not None and v["estimate"] > 0
                for v in summary.values()
            ),
            "daily_by_fold": panels,
            "source_trajectories": bindings,
            "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
            "ledger_rule": "use completed matched economics; this omission gate concerns forecast IC only",
        }
        write_json_atomic(root / f"{group}_loso.json", result)
        return result
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--reference", default="A0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SIX_SEEDS))
    parser.add_argument("--group", default="six_seed")
    args = parser.parse_args()
    run(
        args.root,
        candidate=args.candidate,
        reference=args.reference,
        seeds=tuple(args.seeds),
        group=args.group,
    )


if __name__ == "__main__":
    main()
