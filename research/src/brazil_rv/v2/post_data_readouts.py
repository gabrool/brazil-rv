"""Matched post-data screen: current-store scores, unchanged ledger, paired dates."""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import candidate_readout, paired_readouts
from .contract import HORIZONS, TRADED_PRIMARY_HORIZONS
from .evaluate import (
    _economics_signal,
    _primary_daily_metrics,
    _primary_population_components,
    _spearman,
)
from .hedge_beta import build_hedge_beta_sidecar
from .post_data_program import CELLS, fit_path, read
from .research_checkpoint import _completed, _context_arguments, _finish_cell
from .round6 import resolve_file
from .round6_readouts import evaluation_design


def prepare_economics(root):
    """Rebind only the store and beta provenance; prove beta arrays unchanged."""
    design = read(root / "frozen_design.json")
    destination = root / "economics/evaluation_design.json"
    if destination.exists():
        saved = read(destination)
        if saved["store"] != design["store"]:
            raise ValueError("economic store differs from the training freeze")
        return evaluation_design({"evaluation_sources": {"design": saved}})
    source = resolve_file(design["economic_source"])
    economic = evaluation_design({"evaluation_sources": {"design": read(source)}})
    old_beta = read(Path(economic["bova11"]["hedge_beta_root"]) / "manifest.json")
    beta = root / "economics/hedge_beta"
    build_hedge_beta_sidecar(
        store_root=Path(design["store"]["root"]),
        expected_store_manifest_sha256=design["store"]["manifest_sha256"],
        bova11_root=Path(economic["bova11"]["root"]),
        expected_bova11_manifest_sha256=economic["bova11"]["manifest_sha256"],
        output_root=beta,
    )
    new_beta = read(beta / "manifest.json")
    if {n: r["sha256"] for n, r in new_beta["arrays"].items()} != {
        n: r["sha256"] for n, r in old_beta["arrays"].items()
    }:
        raise ValueError("economic beta changed after the protected-array repair")
    economic["store"] = design["store"]
    economic["bova11"].update(
        hedge_beta_root=str(beta),
        hedge_beta_manifest_sha256=sha256_file(beta / "manifest.json"),
    )
    economic["post_data_binding"] = {
        "source_evaluation_design_sha256": sha256_file(source),
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
        "beta_arrays_exact": True,
        "execution_policy_changed": False,
    }
    write_json_atomic(destination, economic)
    return economic


def ensemble(root, design, choices, cell, fold, seeds, dates, isins):
    members, reference, records = [], None, []
    recipe = "incumbent" if cell == "S0" else choices["recipes"][cell]
    for seed in seeds:
        directory = fit_path(root, cell, recipe, fold, seed)
        record = read(directory / "run_manifest.json")
        if (
            record["status"] != "completed"
            or record["seed"] != seed
            or record["fold"] != fold
        ):
            raise ValueError("screen fit is incomplete or mismatched")
        score_manifest = read(directory / "scores/score_manifest.json")
        if (
            score_manifest["store"]["manifest_sha256"]
            != design["store"]["manifest_sha256"]
        ):
            raise ValueError("screen scores use another store")
        values, mask = rr._score_artifact(
            directory / "scores",
            require_clean_transfer=True,
            expected_dates=dates,
            expected_isins=isins,
            expected_feature_schema_sha256=design["feature_schema_sha256"],
        )
        if reference is not None and not np.array_equal(reference, mask):
            raise ValueError("seed prediction populations differ")
        members.append(values)
        reference = mask
        records.append(
            {
                "seed": seed,
                "run_manifest_sha256": sha256_file(directory / "run_manifest.json"),
                "score_manifest_sha256": sha256_file(
                    directory / "scores/score_manifest.json"
                ),
            }
        )
    return rr.rank_average_ensemble(members, reference), reference, members, records


def alignment(inputs):
    """Same three-head outcome population for head, composite and target views."""
    horizons = TRADED_PRIMARY_HORIZONS
    indexes = [HORIZONS.index(h) for h in horizons]
    scores, targets, outcome, scored = _primary_population_components(inputs, horizons)
    heads, primary, _ = _primary_daily_metrics(
        scores, targets, outcome, scored, inputs.dates, horizons
    )
    common = outcome & scored
    composite, composite_valid = _economics_signal(inputs)
    series = {"primary_neutral_target_ic": primary}
    for lag in (1, 5):
        persistence = np.full(len(inputs.dates), np.nan)
        for day in range(lag, len(inputs.dates)):
            persistence[day] = _spearman(
                composite[day],
                composite[day - lag],
                composite_valid[day] & composite_valid[day - lag],
            )
        series[f"composite_persistence_{lag}"] = persistence
    for j, horizon in enumerate(horizons):
        series[f"D{horizon}_neutral_ic_common"] = heads[:, j]
        series[f"D{horizon}_composite_neutral_ic_common"] = np.asarray(
            [
                _spearman(
                    composite[d], targets[d, :, j], common[d] & composite_valid[d]
                )
                for d in range(len(inputs.dates))
            ]
        )
    target_views = (
        ("scaled", inputs.scaled_midrank_targets, inputs.scaled_target_mask),
        (
            "shareholder",
            inputs.shareholder_midrank_targets,
            inputs.shareholder_target_mask,
        ),
        ("price", inputs.price_midrank_targets, inputs.price_target_mask),
    )
    view_population = common.copy()
    for _, values, mask in target_views:
        view_population &= np.asarray(mask)[..., indexes].all(-1) & np.isfinite(
            np.asarray(values)[..., indexes]
        ).all(-1)
    for name, values, _ in (
        ("neutral", inputs.neutral_midrank_targets, inputs.neutral_target_mask),
        *target_views,
    ):
        _, daily, _ = _primary_daily_metrics(
            scores,
            np.asarray(values)[..., indexes],
            view_population,
            scored,
            inputs.dates,
            horizons,
        )
        series[f"{name}_ic_same_target_view_population"] = daily
    return {
        "series": {
            key: [float(x) if np.isfinite(x) else None for x in values]
            for key, values in series.items()
        },
        "populations": {
            "active": np.asarray(inputs.active).sum(1).tolist(),
            "individual_head_outcomes": (
                np.asarray(inputs.neutral_target_mask)[..., indexes]
                & np.asarray(inputs.active)[..., None]
            )
            .sum(1)
            .tolist(),
            "common_scored_outcomes": common.sum(1).tolist(),
            "score_available_to_book": composite_valid.sum(1).tolist(),
            "same_target_view_outcomes": view_population.sum(1).tolist(),
        },
    }


def intervals(values):
    arrays = tuple(np.asarray(v, dtype=float) for v in values)
    return {
        str(block): rr._folded_bootstrap(arrays, block_length=block, seed=20260913)
        for block in (20, 60)
    }


def require_full_primary_scores(mask, active):
    indexes = [HORIZONS.index(h) for h in TRADED_PRIMARY_HORIZONS]
    if not np.array_equal(
        mask[..., indexes], np.broadcast_to(active[..., None], (*active.shape, 3))
    ):
        raise ValueError("screen score mask drops a primary-head PIT-active name")


def evaluate_fold(root, design, choices, economic, hashes, fold):
    """Each fold owns its store/context and disjoint output directories."""
    context = rr._open_ledger_replay(economic)
    policy, _ = rr.load_selected_policy(
        Path(economic["execution_policy"]["root"]),
        expected_result_sha256=economic["execution_policy"]["result_sha256"],
    )
    seeds = design["seeds"]
    try:
        ix = context.evaluation[fold]
        active = context.store.read("active", ix)
        for cell in CELLS:
            output = root / "aggregates" / cell / fold
            if _completed(output):
                if (
                    read(output / "score_manifest.json")["metadata"]["source_hashes"]
                    != hashes
                ):
                    raise ValueError("aggregate differs from the frozen comparison")
                require_full_primary_scores(
                    np.load(output / "score_mask.npy", allow_pickle=False), active
                )
                continue
            if output.exists():
                raise RuntimeError(f"partial aggregate requires diagnosis: {output}")
            scores, mask, members, records = ensemble(
                root,
                design,
                choices,
                cell,
                fold,
                seeds,
                context.store.dates[ix],
                context.store.isins,
            )
            require_full_primary_scores(mask, active)
            rr._persist_scores(
                output,
                {"scores": scores, "score_mask": mask},
                {
                    **rr._source_tier_labels(context.store.manifest),
                    "arm": cell,
                    "fold": fold,
                    "seeds": seeds,
                    "evaluation_date_indices": ix.tolist(),
                    "trajectories": records,
                    "source_hashes": hashes,
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
            write_json_atomic(output / "alignment.json", alignment(evaluated.inputs))

            def primary(values):
                local = replace(evaluated.inputs, scores=values)
                daily = _primary_daily_metrics(
                    *_primary_population_components(local, TRADED_PRIMARY_HORIZONS),
                    local.dates,
                    TRADED_PRIMARY_HORIZONS,
                )[1]
                return [float(x) if np.isfinite(x) else None for x in daily]

            seed_readouts = {
                str(s): primary(v) for s, v in zip(seeds, members, strict=True)
            }
            omission = {
                str(s): primary(
                    rr.rank_average_ensemble(
                        [m for i, m in enumerate(members) if seeds[i] != s], mask
                    )
                )
                for s in seeds
            }
            write_json_atomic(
                output / "seed_ic.json",
                {"single_seed": seed_readouts, "omitted_seed": omission},
            )
            _finish_cell(output, evaluated, name=cell, fold=fold)
            print(f"accepted {cell}/{fold}", flush=True)
    finally:
        context.store.close()


def evaluate(root):
    design, choices = (
        read(root / "frozen_design.json"),
        read(root / "calibration_choice.json"),
    )
    economic = prepare_economics(root)
    hashes = {
        "v2_store_manifest": design["store"]["manifest_sha256"],
        "post_data_frozen_design": sha256_file(root / "frozen_design.json"),
        "post_data_calibration_choice": sha256_file(root / "calibration_choice.json"),
    }
    seeds, folds = design["seeds"], design["screen_folds"]
    paths = {
        cell: {fold: root / "aggregates" / cell / fold for fold in folds}
        for cell in CELLS
    }
    # Preserve chronological replay within a book; only independent folds run
    # concurrently. Spawn avoids inheriting initialized numerical thread pools.
    with ProcessPoolExecutor(
        max_workers=min(4, len(folds)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                evaluate_fold, root, design, choices, economic, hashes, fold
            )
            for fold in folds
        ]
        for future in futures:
            future.result()
    context = rr._open_ledger_replay(economic)
    try:
        requested = {(cell, "S0") for cell in CELLS if cell != "S0"}
        requested |= {
            ("TE_slow", "TL_slow"),
            ("TE_slow", "C1_slow"),
            ("C1_slow", "S0_common"),
            ("C1_family", "C1_slow"),
            ("C1_all", "C1_family"),
            ("TE_family", "TE_slow"),
            ("TE_all", "TE_family"),
        }
        pairs, paired_alignment = {}, {}
        alignments = {
            c: {f: read(p / "alignment.json") for f, p in locations.items()}
            for c, locations in paths.items()
        }
        for left, right in sorted(requested):
            key = f"{left}_minus_{right}"
            paired_readouts(
                context, {left: paths[left], right: paths[right]}, root / "paired"
            )
            audits = {
                f: read(root / "paired" / key / f"{f}.json")["population_audit"][f]
                for f in folds
            }
            pairs[key] = {
                metric: intervals(
                    [[r["delta"] for r in audits[f][metric]] for f in folds]
                )
                for metric in audits[folds[0]]
            }
            # Every arm emits all active primary-head scores (checked above).
            # These stored alignments therefore share exact outcome populations.
            paired_alignment[key] = {
                metric: intervals(
                    [
                        np.asarray(alignments[left][f]["series"][metric], float)
                        - np.asarray(alignments[right][f]["series"][metric], float)
                        for f in folds
                    ]
                )
                for metric in alignments[left][folds[0]]["series"]
            }
        result = {
            "status": "completed",
            "source_hashes": hashes,
            "screen_folds": folds,
            "seeds": seeds,
            "readouts": {c: candidate_readout(p) for c, p in paths.items()},
            "paired_registered_intervals": pairs,
            "paired_traded_head_alignment": paired_alignment,
            "auxiliary_readout_intervals": "historical helpers use 20 sessions, 10000 draws, seed20260815; registered paired intervals above use seed20260913 and 20/60 blocks",
            "alignment": alignments,
            "seed_dispersion": {
                c: {f: read(p / "seed_ic.json") for f, p in locations.items()}
                for c, locations in paths.items()
            },
            "aggregate_paths": {
                c: {f: str(p) for f, p in locations.items()}
                for c, locations in paths.items()
            },
            "research_label": "repeatedly studied development history; nominal paired screen; no automatic research promotion",
            "heldout_access": False,
            "forward_capture": False,
        }
        write_json_atomic(root / "screen_result.json", result)
        return result
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare-economics-only", action="store_true")
    args = parser.parse_args()
    prepare_economics(args.root) if args.prepare_economics_only else evaluate(args.root)


if __name__ == "__main__":
    main()
