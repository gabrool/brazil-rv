"""Matched, resumable CPU information screens; no network or policy selection."""

from __future__ import annotations

import argparse
import gc
import json
import time
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .contract import (
    DEVELOPMENT_END,
    FINETUNE_START,
    GBDT_SEEDS,
    HORIZONS,
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
    TRADED_PRIMARY_HORIZONS,
)
from .data import read_scalar_feature_view
from .evaluate import _primary_daily_metrics
from .gbdt import (
    GBDTConfig,
    MultiHorizonGBDT,
    assemble_gbdt_scalar_view,
    gbdt_scalar_feature_names,
)
from .round5_magnitude import FitClip
from .store import open_store_for_dates, peak_rss_bytes
from .validate_pipeline import _window_target_mask

FAMILIES = (
    "events",
    "fundamentals",
    "lending",
    "cross_market",
    "options",
    "oddlot",
    "sector",
    "magnitudes",
    "rebalance",
    "microstructure",
)
MAXIMUM_RSS = 8 * 1024**3


def model_program_identity():
    return {
        name: sha256_file(Path(__file__).with_name(name + ".py"))
        for name in ("round5_screens", "gbdt", "round5_magnitude", "normalization")
    }


def bind(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}


def verify(record):
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"screen artifact identity differs: {path}")
    return path


def _save(root, name, array):
    path = root / f"{name}.npy"
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    return bind(path)


def prepare(store_root: Path, output: Path, amendments: list[Path], *, threads=4):
    """One immutable feature cache, with coverage measured before any fit."""
    code = rr._git_identity()
    acceptance_path = store_root / "round5_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not (acceptance["status"] == "passed" and acceptance["protected_arrays_exact"]):
        raise ValueError("Round-5 screens require accepted store extension")
    if sha256_file(store_root / "manifest.json") != acceptance["manifest_sha256"]:
        raise ValueError("screen store differs from accepted extension")
    axis = np.load(store_root / "date_index.npy", allow_pickle=False)
    if axis[-1] > np.datetime64(DEVELOPMENT_END):
        raise PermissionError("screen cache contains held-out dates")
    fit, selection, evaluation, windows, folds = rr._fold_indices(axis)
    rows = np.flatnonzero(axis >= np.datetime64(FINETUNE_START))
    store, access = open_store_for_dates(store_root, rows, purpose="training")
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "cache"
    cache.mkdir()
    start = int(rows[0])
    arrays, family_records = {}, {}
    try:
        active = store.read("active", rows)
        mask = store.read(REGISTERED_PRIMARY_TARGET_MASK, rows)
        targets = store.read_target(REGISTERED_PRIMARY_TARGET, rows, valid_mask=mask)
        for name, value in (
            ("dates", axis[rows]),
            ("global_indices", rows),
            ("active", active),
            ("targets", targets),
            ("target_mask", mask),
        ):
            arrays[name] = _save(cache, name, value)
        for family in ("a_slow", *FAMILIES):
            source_family = "slow" if family == "a_slow" else f"sidecar_{family}"
            view = read_scalar_feature_view(store, rows, (source_family,))
            encoded = assemble_gbdt_scalar_view(view, label=family)
            present = active & view.valid.any(axis=-1)
            record = {
                "values": _save(cache, family, encoded),
                "presence": _save(cache, f"{family}_presence", present),
                "field_names": list(view.names),
                "encoded_names": list(gbdt_scalar_feature_names(view.names)),
                "coverage": {},
            }
            for fold in evaluation:
                counts = {
                    key: int(present[indices[fold] - start].sum())
                    for key, indices in (
                        ("fit", fit),
                        ("selection", selection),
                        ("evaluation", evaluation),
                    )
                }
                counts["informative"] = counts["fit"] > 0 and counts["evaluation"] > 0
                record["coverage"][fold] = counts
            family_records[family] = record
            del view, encoded, present
            gc.collect()
            if peak_rss_bytes() > MAXIMUM_RSS:
                raise MemoryError("screen cache preparation exceeded8GiB")
        design = {
            "schema": "BRAZIL_RV_ROUND5_CPU_SCREENS_V1",
            "code": code,
            "model_program_sha256": model_program_identity(),
            "library_versions": {
                name: version(name) for name in ("numpy", "lightgbm", "scipy")
            },
            "store": bind(store_root / "manifest.json"),
            "acceptance": bind(acceptance_path),
            "amendments": [bind(path) for path in amendments],
            "config": asdict(GBDTConfig(seeds=GBDT_SEEDS, num_threads=threads)),
            "cache_start_global_index": start,
            "arrays": arrays,
            "families": family_records,
            "fit": {k: v.tolist() for k, v in fit.items()},
            "selection": {k: v.tolist() for k, v in selection.items()},
            "evaluation": {k: v.tolist() for k, v in evaluation.items()},
            "fit_target_windows": {k: v.tolist() for k, v in windows.items()},
            "folds": folds,
            "primary_horizons": list(TRADED_PRIMARY_HORIZONS),
            "tree_shap_names_per_day": 16,
            "network_selection_weight": 0,
            "access": access.payload(),
            "peak_rss_bytes": peak_rss_bytes(),
        }
        write_json_atomic(output / "design.json", design)
        return design
    finally:
        store.close()


def _cache(record):
    # The parent coordinator verifies all immutable files once before dispatch.
    return np.load(record["path"], mmap_mode="r", allow_pickle=False)


def window_targets(targets, source_mask, indices, start, target_window):
    local = np.asarray(indices) - start
    mask = _window_target_mask(
        np.asarray(source_mask[local]),
        np.asarray(indices),
        target_window_indices=np.asarray(target_window),
    )
    return np.where(mask, targets[local], 0), mask


def clip_magnitudes(encoded, active, fit_local, fit_global):
    """A cache may contain later values, but only fit rows estimate bounds."""
    width = encoded.shape[-1] // 2
    raw = np.asarray(encoded[..., :width])
    valid = np.isfinite(raw)
    fitted = FitClip.fit(raw, valid, active, np.asarray(fit_local))
    return replace(fitted, fit_date_indices=tuple(int(x) for x in fit_global))


def _features(parent, sidecar, local, clip):
    base = np.asarray(parent[local])
    if sidecar is None:
        return base
    extra = np.array(sidecar[local], copy=True)
    if clip is not None:
        width = extra.shape[-1] // 2
        # np.clip preserves NaN; unsupported fields remain missing, not zero.
        extra[..., :width] = np.clip(extra[..., :width], clip.lower, clip.upper)
    return np.concatenate((base, extra), axis=-1)


def shap_coordinates(eligible, maximum_per_day=16):
    """Mask-only, evenly spaced per-day names; no target-dependent sampling."""
    coordinates = []
    for day, valid in enumerate(eligible):
        names = np.flatnonzero(valid)
        if len(names):
            chosen = names[
                np.linspace(
                    0, len(names) - 1, min(maximum_per_day, len(names)), dtype=int
                )
            ]
            coordinates.extend((day, int(name)) for name in chosen)
    return np.asarray(coordinates, np.int64).reshape(-1, 2)


def primary_daily(scores, targets, target_mask, active, dates):
    columns = [HORIZONS.index(h) for h in TRADED_PRIMARY_HORIZONS]
    predicted, outcomes = scores[..., columns], targets[..., columns]
    population = active & target_mask[..., columns].all(axis=-1)
    supported = active & np.isfinite(predicted).all(axis=-1)
    return _primary_daily_metrics(
        predicted, outcomes, population, supported, dates, TRADED_PRIMARY_HORIZONS
    )[1]


def run_cell(design_path: str, family: str, fold: str, seed: int):
    started = time.monotonic()
    path = Path(design_path)
    design = json.loads(path.read_text(encoding="utf-8"))
    if seed not in design["config"]["seeds"] or family not in design["families"]:
        raise ValueError("cell is not registered in the frozen screen design")
    root = path.parent / "cells" / family / fold / f"seed_{seed}"
    result_path = root / "result.json"
    design_hash = sha256_file(path)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["design_sha256"] != design_hash:
            raise ValueError("completed cell binds another screen design")
        for record in result["outputs"].values():
            verify(record)
        return {
            "family": family,
            "fold": fold,
            "seed": seed,
            "status": "reused_complete",
        }
    root.mkdir(parents=True, exist_ok=True)
    coverage = design["families"][family]["coverage"][fold]
    if family != "a_slow" and not coverage["fit"]:
        result = {
            "design_sha256": design_hash,
            "family": family,
            "fold": fold,
            "seed": seed,
            "status": "no_family_observation_in_fit_use_parent",
            "coverage": coverage,
            "outputs": {},
        }
        write_json_atomic(result_path, result)
        return result
    arrays = {name: _cache(record) for name, record in design["arrays"].items()}
    start = design["cache_start_global_index"]
    fit, selection, evaluation = (
        np.asarray(design[key][fold], np.int64)
        for key in ("fit", "selection", "evaluation")
    )
    parent = _cache(design["families"]["a_slow"]["values"])
    sidecar = (
        None if family == "a_slow" else _cache(design["families"][family]["values"])
    )
    clip = (
        clip_magnitudes(sidecar, arrays["active"], fit - start, fit)
        if family == "magnitudes"
        else None
    )
    names = tuple(design["families"]["a_slow"]["encoded_names"])
    if sidecar is not None:
        names += tuple(design["families"][family]["encoded_names"])
    train_x = _features(parent, sidecar, fit - start, clip)
    selection_x = _features(parent, sidecar, selection - start, clip)
    train_y, train_mask = window_targets(
        arrays["targets"],
        arrays["target_mask"],
        fit,
        start,
        design["fit_target_windows"][fold],
    )
    selection_y, selection_mask = window_targets(
        arrays["targets"], arrays["target_mask"], selection, start, selection
    )
    config = GBDTConfig(**{**design["config"], "seeds": (seed,)})
    model_root = root / "models"
    model_manifest = model_root / "model_manifest.json"
    if model_manifest.exists():
        model = MultiHorizonGBDT.load(
            model_root, expected_manifest_sha256=sha256_file(model_manifest)
        )
        saved = json.loads(model_manifest.read_text(encoding="utf-8"))
        if (
            saved.get("design_sha256") != design_hash
            or saved.get("fold") != fold
            or saved.get("family") != family
            or saved.get("seed") != seed
        ):
            raise ValueError("partial cell models have a different source binding")
    else:
        model = MultiHorizonGBDT(config, feature_names=names).fit(
            train_x,
            train_y,
            train_mask,
            selection_x,
            selection_y,
            selection_mask,
            train_dates=fit,
            validation_dates=selection,
        )
        model.save(
            model_root,
            metadata={
                "design_sha256": design_hash,
                "family": family,
                "fold": fold,
                "seed": seed,
                "magnitude_clip": None if clip is None else clip.payload(),
            },
        )
    del train_x, selection_x, train_y, train_mask, selection_y, selection_mask
    gc.collect()
    evaluation_x = _features(parent, sidecar, evaluation - start, clip)
    active = np.asarray(arrays["active"][evaluation - start])
    scores = model.predict_ranks(
        evaluation_x, np.repeat(active[..., None], len(HORIZONS), axis=-1)
    )
    targets, target_mask = window_targets(
        arrays["targets"], arrays["target_mask"], evaluation, start, evaluation
    )
    dates = arrays["dates"][evaluation - start].astype(object).tolist()
    daily = primary_daily(scores, targets, target_mask, active, dates)
    presence = _cache(design["families"][family]["presence"])[evaluation - start]
    coordinates = shap_coordinates(active & presence, design["tree_shap_names_per_day"])
    explanation = (
        evaluation_x[coordinates[:, 0], coordinates[:, 1]][:, None, :]
        if len(coordinates)
        else None
    )
    importance = model.feature_importance(explanation)
    # Unsealed diagnostic files may be recreated after a interrupted process;
    # a completed cell is immutable and verified at the beginning of this call.
    scores_path, daily_path = root / "scores.npy", root / "daily_ic.npy"
    np.save(scores_path, scores, allow_pickle=False)
    np.save(daily_path, daily, allow_pickle=False)
    result = {
        "design_sha256": design_hash,
        "family": family,
        "fold": fold,
        "seed": seed,
        "status": "completed",
        "coverage": coverage,
        "outputs": {
            "scores": bind(scores_path),
            "daily_ic": bind(daily_path),
            "models": bind(model_manifest),
        },
        "feature_names": list(names),
        "importance": {k: v.tolist() for k, v in importance.items()},
        "tree_shap_coordinates": coordinates.tolist(),
        "magnitude_clip": None if clip is None else clip.payload(),
        "mean_primary_ic": float(np.nanmean(daily))
        if np.isfinite(daily).any()
        else None,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    if result["peak_rss_bytes"] > MAXIMUM_RSS:
        raise MemoryError("GBDT cell exceeded8GiB")
    write_json_atomic(result_path, result)
    return {
        key: result[key]
        for key in (
            "family",
            "fold",
            "seed",
            "status",
            "elapsed_seconds",
            "peak_rss_bytes",
        )
    }


def run(design_path: Path, *, workers=2):
    design = json.loads(design_path.read_text(encoding="utf-8"))
    code = rr._git_identity()
    if design["model_program_sha256"] != model_program_identity():
        raise ValueError("screen model implementation differs from its frozen design")
    for record in (*design["amendments"], *design["arrays"].values()):
        verify(record)
    for record in design["families"].values():
        verify(record["values"])
        verify(record["presence"])
    write_json_atomic(
        design_path.parent / "execution.json",
        {
            "code": code,
            "workers": workers,
            "design_sha256": sha256_file(design_path),
            "neural_fits": False,
        },
    )
    cells = [
        (str(design_path), family, fold, seed)
        for family in design["families"]
        for fold in design["evaluation"]
        for seed in design["config"]["seeds"]
    ]
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
        pending = {executor.submit(run_cell, *cell): cell for cell in cells}
        done = 0
        for future in as_completed(pending):
            result = future.result()
            done += 1
            print(
                json.dumps({"completed": done, "total": len(cells), **result}),
                flush=True,
            )
    return summarize(design_path)


def summarize(design_path: Path):
    design = json.loads(design_path.read_text(encoding="utf-8"))
    start = design["cache_start_global_index"]
    arrays = {k: _cache(v) for k, v in design["arrays"].items()}
    daily_by_family, by_family = {}, {}
    for family in design["families"]:
        daily_by_family[family], records = {}, {}
        for fold, indices in design["evaluation"].items():
            scores, seed_readouts, importance = [], [], []
            for seed in design["config"]["seeds"]:
                root = design_path.parent / "cells" / family / fold / f"seed_{seed}"
                result = json.loads((root / "result.json").read_text(encoding="utf-8"))
                if result["status"] == "no_family_observation_in_fit_use_parent":
                    root = design_path.parent / "cells/a_slow" / fold / f"seed_{seed}"
                    result = json.loads(
                        (root / "result.json").read_text(encoding="utf-8")
                    )
                for record in result["outputs"].values():
                    verify(record)
                scores.append(
                    np.load(result["outputs"]["scores"]["path"], allow_pickle=False)
                )
                seed_readouts.append(
                    {"seed": seed, "mean_primary_ic": result["mean_primary_ic"]}
                )
                if result["family"] == family:
                    importance.append(result)
            ensemble = np.mean(np.stack(scores), axis=0, dtype=np.float32)
            indices = np.asarray(indices)
            targets, mask = window_targets(
                arrays["targets"], arrays["target_mask"], indices, start, indices
            )
            daily = primary_daily(
                ensemble,
                targets,
                mask,
                arrays["active"][indices - start],
                arrays["dates"][indices - start].astype(object).tolist(),
            )
            daily_by_family[family][fold] = daily
            records[fold] = {
                "coverage": design["families"][family]["coverage"][fold],
                "seeds": seed_readouts,
                "ensemble_mean_primary_ic": float(np.nanmean(daily))
                if np.isfinite(daily).any()
                else None,
                "ensemble_daily_ic": [
                    float(x) if np.isfinite(x) else None for x in daily
                ],
                "feature_importance": {},
            }
            if importance:
                fields = importance[0]["feature_names"]
                for label in importance[0]["importance"]:
                    mean = np.mean([r["importance"][label] for r in importance], axis=0)
                    records[fold]["feature_importance"][label] = dict(
                        zip(fields, mean.tolist(), strict=True)
                    )
        by_family[family] = {"folds": records}
    for family, daily in daily_by_family.items():
        all_folds = tuple(daily)
        informative = tuple(
            f for f in daily if design["families"][family]["coverage"][f]["informative"]
        )
        for label, folds in (
            ("all_folds", all_folds),
            ("informative_folds", informative),
        ):
            if not folds:
                by_family[family][label] = {
                    "folds": [],
                    "status": "no_informative_folds",
                }
                continue
            by_family[family][label] = {
                "folds": list(folds),
                "primary_ic": rr._folded_bootstrap(tuple(daily[f] for f in folds)),
            }
            if family != "a_slow":
                by_family[family][label]["paired_delta_vs_a_slow"] = (
                    rr._folded_bootstrap(
                        tuple(daily[f] - daily_by_family["a_slow"][f] for f in folds)
                    )
                )
    result = {
        "schema": "BRAZIL_RV_ROUND5_CPU_SCREEN_READOUT_V1",
        "design": bind(design_path),
        "families": by_family,
        "seeds": list(GBDT_SEEDS),
        "primary_horizons": list(TRADED_PRIMARY_HORIZONS),
        "network_selection_weight": 0,
        "interpretation": "information readout only; null screens do not demote neural families",
    }
    write_json_atomic(design_path.parent / "readout.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "summarize"))
    parser.add_argument("--store", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, action="append", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.store, args.root, args.amendment, threads=args.threads)
    elif args.mode == "run":
        run(args.root / "design.json", workers=args.workers)
    else:
        summarize(args.root / "design.json")


if __name__ == "__main__":
    main()
