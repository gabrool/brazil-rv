from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .artifacts import inventory, sha256_file, verify_inventory, write_json_atomic
from .baselines import BaselinePanel, build_baselines
from .config import PROJECT_ROOT
from .contract import GBDT_SEEDS, HORIZONS, PRETRAIN_END, PRIMARY_HORIZONS, STORE_START
from .evaluate import EvaluationResult, evaluate_scores
from .gbdt import GBDTConfig, MultiHorizonGBDT
from .splits import development_folds
from .store import V2Store, open_store_for_samples
from .train import (
    block_parity_mask,
    rank_average_ensemble,
    stitch_block_parity_predictions,
)
from .validate_pipeline import (
    _date_indices,
    _evaluation_inputs,
    _load_development_cdi,
    _read_store_header,
    _window_target_mask,
)

ROUND1_SCHEMA = "BRAZIL_RV_V2_RESEARCH_ROUND1_V1"
ROUND2_SCHEMA = "BRAZIL_RV_V2_RESEARCH_ROUND2_V1"
PREREGISTRATION = PROJECT_ROOT / "research" / "preregistrations" / "v2_round1_round2.md"
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_SEED = 20260815
NETWORK_SEEDS = (11, 29, 47)
RUNG_GROUPS: dict[str, tuple[str, ...]] = {
    "a_slow": (),
    "b_intraday": (),
    "c_lending": ("lending",),
    "d_all_sidecars": (
        "lending",
        "oddlot",
        "options",
        "rebalance",
        "events",
        "fundamentals",
    ),
}
RESEARCH_FLAGS = {
    "research_claim": True,
    "official_validation_accessed": False,
    "test_accessed": False,
    "deployment_changed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("v2 research requires a clean commit-bound worktree")
    if len(commit) != 40:
        raise RuntimeError("invalid git commit identity")
    return {"commit": commit, "tracked_worktree_clean": True}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _assert_false_access(payload: Mapping[str, object], *, path: Path) -> None:
    if (
        payload.get("official_validation_accessed") is not False
        or payload.get("test_accessed") is not False
    ):
        raise PermissionError(f"sealed-window access recorded in {path}")


def _array_record(path: Path, values: NDArray[np.generic]) -> dict[str, object]:
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _persist_scores(
    root: Path,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, object],
) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=False)
    records: dict[str, dict[str, object]] = {}
    for label, raw in sorted(arrays.items()):
        values = np.asarray(raw)
        path = root / f"{label}.npy"
        np.save(path, values, allow_pickle=False)
        records[path.name] = _array_record(path, values)
    manifest = root / "score_manifest.json"
    digest = write_json_atomic(
        manifest,
        {
            "schema": "BRAZIL_RV_V2_RESEARCH_SCORE_V1",
            "status": "completed",
            **RESEARCH_FLAGS,
            "metadata": dict(metadata),
            "artifacts": records,
        },
    )
    return manifest, digest


def _fold_indices(
    dates: NDArray[np.datetime64],
) -> tuple[
    dict[str, NDArray[np.int64]], dict[str, NDArray[np.int64]], dict[str, object]
]:
    python_dates = tuple(dates.astype("datetime64[D]").astype(object).tolist())
    folds = development_folds(python_dates)
    fit: dict[str, NDArray[np.int64]] = {}
    selection: dict[str, NDArray[np.int64]] = {}
    payload: dict[str, object] = {}
    for fold in folds:
        fit[fold.name] = _date_indices(dates, fold.fit_dates)
        selection[fold.name] = _date_indices(dates, fold.selection_dates)
        payload[fold.name] = fold.payload()
    if tuple(fit) != ("F1", "F2", "F3"):
        raise ValueError("development fold roster differs from the registration")
    return fit, selection, payload


def _pretrain_indices(dates: NDArray[np.datetime64]) -> NDArray[np.int64]:
    full = np.flatnonzero(
        (dates >= np.datetime64(STORE_START)) & (dates <= np.datetime64(PRETRAIN_END))
    ).astype(np.int64)
    if not full.size or np.any(np.diff(full) != 1):
        raise ValueError("pretrain axis is incomplete")
    return full


def _open_round_store(
    store_root: Path,
    fit: Mapping[str, NDArray[np.int64]],
    selection: Mapping[str, NDArray[np.int64]],
    pretrain: NDArray[np.int64],
) -> tuple[V2Store, dict[str, object]]:
    requested = np.unique(
        np.concatenate(
            (
                pretrain,
                *(fit[name] for name in ("F1", "F2", "F3")),
                *(selection[name] for name in ("F1", "F2", "F3")),
            )
        )
    ).astype(np.int64)
    store, ledger = open_store_for_samples(
        store_root,
        requested,
        purpose="evaluation",
        history_lookbacks=253,
        history_end_offsets=0,
    )
    return store, ledger.payload()


def _evaluate(
    *,
    store: V2Store,
    indices: NDArray[np.int64],
    scores: NDArray[np.floating],
    score_mask: NDArray[np.bool_],
    cdi: NDArray[np.float64],
    source_hashes: Mapping[str, str],
    fold: str,
    output: Path,
    pathwise_scores: tuple[NDArray[np.floating], ...] = (),
) -> EvaluationResult:
    inputs = _evaluation_inputs(
        store,
        indices,
        scores,
        score_mask,
        cdi,
        source_hashes,
        pathwise_scores,
        tuple(score_mask for _ in pathwise_scores),
    )
    result = evaluate_scores(inputs, window_name=fold)
    result.report.update(RESEARCH_FLAGS)
    write_json_atomic(output, result.report)
    return result


def _daily_series(report: Mapping[str, object]) -> dict[str, NDArray[np.float64]]:
    primary = set(PRIMARY_HORIZONS)
    daily_primary = report.get("daily_primary_ic")
    metrics = report.get("daily_metric_table")
    persistence = report.get("persistence_table")
    economics = report.get("economics")
    if (
        not isinstance(daily_primary, list)
        or not isinstance(metrics, list)
        or not isinstance(persistence, list)
        or not isinstance(economics, Mapping)
        or not isinstance(economics.get("daily_table"), list)
    ):
        raise ValueError("evaluation report lacks registered daily readouts")

    def average_rows(
        rows: list[object], value_key: str, *, lag: int | None = None
    ) -> NDArray[np.float64]:
        by_date: dict[str, list[float]] = {}
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("daily readout row is malformed")
            if int(raw.get("horizon_sessions", -1)) not in primary:
                continue
            if lag is not None and int(raw.get("lag_sessions", -1)) != lag:
                continue
            key = str(raw["date"])
            value = raw.get(value_key)
            by_date.setdefault(key, []).append(
                np.nan if value is None else float(value)
            )
        return np.asarray(
            [
                np.nanmean(values) if np.isfinite(values).any() else np.nan
                for values in by_date.values()
            ],
            dtype=np.float64,
        )

    primary_values = np.asarray(
        [
            np.nan
            if not isinstance(row, Mapping)
            or row.get("mean_primary_horizon_ic") is None
            else float(row["mean_primary_horizon_ic"])
            for row in daily_primary
        ],
        dtype=np.float64,
    )
    headline = [
        row
        for row in economics["daily_table"]
        if isinstance(row, Mapping)
        and float(row.get("cost_bps_per_side", -1.0)) == 4.0
        and float(row.get("annual_borrow_rate", -1.0)) == 0.02
    ]
    return {
        "residual_ic": primary_values,
        "raw_rank_ic": average_rows(metrics, "raw_rank_ic"),
        "persistence_1": average_rows(persistence, "spearman", lag=1),
        "persistence_5": average_rows(persistence, "spearman", lag=5),
        "decile_spread_bps_per_holding_session": average_rows(
            metrics, "decile_spread_bps_per_holding_session"
        ),
        "headline_net_excess_bps": np.asarray(
            [
                np.nan
                if row.get("net_excess_all_cash_bps") is None
                else float(row["net_excess_all_cash_bps"])
                for row in headline
            ],
            dtype=np.float64,
        ),
    }


def _folded_bootstrap(
    values: Sequence[NDArray[np.floating]],
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | None]:
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if not arrays or any(
        value.ndim != 1 or len(value) < BOOTSTRAP_BLOCK for value in arrays
    ):
        raise ValueError("each fold needs at least one bootstrap block")
    finite = np.concatenate(arrays)
    finite_observations = int(np.isfinite(finite).sum())
    if finite_observations == 0:
        return {
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "finite_observations": 0,
            "replications": replications,
            "block_length_sessions": BOOTSTRAP_BLOCK,
            "fold_boundary_preserved": True,
        }
    estimate = float(np.nanmean(finite))
    generator = np.random.default_rng(seed)
    sums = np.zeros(replications, dtype=np.float64)
    counts = np.zeros(replications, dtype=np.int64)
    for array in arrays:
        blocks = math.ceil(len(array) / BOOTSTRAP_BLOCK)
        starts = generator.integers(
            0,
            len(array) - BOOTSTRAP_BLOCK + 1,
            size=(replications, blocks),
        )
        sampled = (
            starts[..., None] + np.arange(BOOTSTRAP_BLOCK, dtype=np.int64)
        ).reshape(replications, -1)[:, : len(array)]
        selected = array[sampled]
        sums += np.nansum(selected, axis=1)
        counts += np.isfinite(selected).sum(axis=1)
    draws = np.divide(sums, counts, out=np.full(replications, np.nan), where=counts > 0)
    return {
        "estimate": estimate,
        "lower_95": float(np.nanquantile(draws, 0.025)),
        "upper_95": float(np.nanquantile(draws, 0.975)),
        "finite_observations": finite_observations,
        "replications": replications,
        "block_length_sessions": BOOTSTRAP_BLOCK,
        "fold_boundary_preserved": True,
    }


def _pooled_readouts(reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    if tuple(reports) != ("F1", "F2", "F3"):
        raise ValueError("pooled report roster must be F1/F2/F3")
    series = {fold: _daily_series(report) for fold, report in reports.items()}
    labels = tuple(series["F1"])
    return {
        "folds": {
            fold: {label: _folded_bootstrap((values[label],)) for label in labels}
            for fold, values in series.items()
        },
        "pooled": {
            label: _folded_bootstrap(
                tuple(series[fold][label] for fold in ("F1", "F2", "F3"))
            )
            for label in labels
        },
        "horizons": {fold: reports[fold]["horizon_readouts"] for fold in reports},
        "economics_grid": {
            fold: reports[fold]["economics"]["summaries"] for fold in reports
        },
    }


def _paired_readouts(
    candidate: Mapping[str, Mapping[str, object]],
    baseline: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    candidate_series = {fold: _daily_series(candidate[fold]) for fold in candidate}
    baseline_series = {fold: _daily_series(baseline[fold]) for fold in baseline}
    labels = tuple(candidate_series["F1"])
    deltas: dict[str, dict[str, NDArray[np.float64]]] = {}
    for fold in ("F1", "F2", "F3"):
        left_report = candidate[fold]
        right_report = baseline[fold]
        if left_report.get("input_hashes") != right_report.get("input_hashes"):
            differing = {
                key
                for key in set(left_report.get("input_hashes", {}))
                if left_report["input_hashes"].get(key)
                != right_report["input_hashes"].get(key)
                and key != "scores"
            }
            if differing:
                raise ValueError(
                    f"paired inputs differ for {fold}: {sorted(differing)}"
                )
        deltas[fold] = {}
        for label in labels:
            left = candidate_series[fold][label]
            right = baseline_series[fold][label]
            if left.shape != right.shape:
                raise ValueError(f"paired {label} axes differ for {fold}")
            deltas[fold][label] = left - right
    return {
        "schema": "BRAZIL_RV_V2_POOLED_PAIRED_READOUTS_V1",
        "folds": {
            fold: {label: _folded_bootstrap((deltas[fold][label],)) for label in labels}
            for fold in deltas
        },
        "pooled": {
            label: _folded_bootstrap(
                tuple(deltas[fold][label] for fold in ("F1", "F2", "F3"))
            )
            for label in labels
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _feature_names(store: V2Store, rung: str) -> tuple[str, ...]:
    names = store.manifest.get("feature_names")
    if not isinstance(names, Mapping) or not isinstance(names.get("slow"), list):
        raise ValueError("store lacks ordered feature names")
    output = list(names["slow"])
    if rung != "a_slow":
        output.extend(names["intraday"])
        output.extend(("fast_present", "days_since_last_slow_row"))
    for group in RUNG_GROUPS[rung]:
        output.extend(names[f"sidecar_{group}"])
    if len(output) != len(set(output)):
        raise ValueError("GBDT feature names are not unique")
    return tuple(str(value) for value in output)


def _gbdt_features(
    store: V2Store,
    indices: NDArray[np.int64],
    rung: str,
    *,
    pretrain_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    pretrain = (
        np.zeros(len(indices), dtype=np.bool_)
        if pretrain_mask is None
        else np.asarray(pretrain_mask, dtype=np.bool_)
    )
    if pretrain.shape != (len(indices),):
        raise ValueError("pretrain row mask is misaligned")
    slow_indices = indices - (~pretrain).astype(np.int64)
    if np.any(slow_indices < 0):
        raise ValueError("GBDT slow history precedes the store")
    parts = [np.asarray(store.read("slow_values", slow_indices), dtype=np.float32)]
    if rung != "a_slow":
        intraday = np.asarray(store.read("intraday_values", indices), dtype=np.float32)
        intraday[pretrain] = 0.0
        present = np.asarray(store.read("fast_present", indices), dtype=np.float32)
        present[pretrain] = 0.0
        days = np.ones_like(present, dtype=np.float32)
        days[pretrain] = 0.0
        parts.extend((intraday, present[..., None], days[..., None]))
    for group in RUNG_GROUPS[rung]:
        parts.append(
            np.asarray(
                store.read(f"sidecar_{group}_values", slow_indices),
                dtype=np.float32,
            )
        )
    result = np.concatenate(parts, axis=-1, dtype=np.float32)
    if (
        result.shape[-1] != len(_feature_names(store, rung))
        or not np.isfinite(result).all()
    ):
        raise ValueError("GBDT feature panel violates its frozen contract")
    return result


def _persist_model(
    model: MultiHorizonGBDT,
    root: Path,
    verification_features: NDArray[np.floating],
    verification_mask: NDArray[np.bool_],
) -> dict[str, object]:
    manifest, digest = model.save(
        root, metadata={"status": "completed", **RESEARCH_FLAGS}
    )
    restored = MultiHorizonGBDT.load(root, expected_manifest_sha256=digest)
    if not np.array_equal(
        model.predict_ranks(verification_features, verification_mask),
        restored.predict_ranks(verification_features, verification_mask),
    ):
        raise ValueError("GBDT model round-trip changed registered ranks")
    return {"manifest": str(manifest), "manifest_sha256": digest}


def _run_gbdt_candidate(
    *,
    store: V2Store,
    rung: str,
    fit: Mapping[str, NDArray[np.int64]],
    selection: Mapping[str, NDArray[np.int64]],
    pretrain: NDArray[np.int64] | None,
    decay_half_life: float | None,
    cdi: NDArray[np.float64],
    source_hashes: Mapping[str, str],
    root: Path,
    num_threads: int,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    config = GBDTConfig(seeds=GBDT_SEEDS, num_threads=num_threads)
    feature_names = _feature_names(store, rung)
    reports: dict[str, dict[str, object]] = {}
    records: dict[str, object] = {}
    for fold in ("F1", "F2", "F3"):
        fine = fit[fold]
        train_indices = fine if pretrain is None else np.concatenate((pretrain, fine))
        train_pretrain = np.zeros(len(train_indices), dtype=np.bool_)
        if pretrain is not None:
            train_pretrain[: len(pretrain)] = True
        evaluation_indices = selection[fold]
        train_x = _gbdt_features(
            store, train_indices, rung, pretrain_mask=train_pretrain
        )
        evaluation_x = _gbdt_features(store, evaluation_indices, rung)
        train_y = np.asarray(store.read("target_primary", train_indices))
        train_mask = _window_target_mask(
            store.read("target_valid", train_indices), train_indices
        )
        selection_y = np.asarray(store.read("target_primary", evaluation_indices))
        selection_mask = _window_target_mask(
            store.read("target_valid", evaluation_indices), evaluation_indices
        )
        active = np.asarray(store.read("active", evaluation_indices), dtype=np.bool_)
        score_mask = np.repeat(active[..., None], len(HORIZONS), axis=-1)
        sample_weights = None
        if decay_half_life is not None:
            ages = train_indices[-1] - train_indices
            sample_weights = np.power(0.5, ages / decay_half_life)
        predictions: dict[int, NDArray[np.float32]] = {}
        model_records: dict[str, object] = {}
        importance: dict[str, object] = {}
        for parity in (0, 1):
            selected = block_parity_mask(len(evaluation_indices), parity)
            model = MultiHorizonGBDT(config, feature_names=feature_names)
            model.fit(
                train_x,
                train_y,
                train_mask,
                evaluation_x[selected],
                selection_y[selected],
                selection_mask[selected],
                train_dates=train_indices,
                validation_dates=evaluation_indices[selected],
                sample_weights=sample_weights,
            )
            predictions[parity] = model.predict_ranks(evaluation_x, score_mask)
            label = "even" if parity == 0 else "odd"
            raw_importance = model.feature_importance(evaluation_x)
            importance[f"selected_on_{label}"] = {
                name: values.tolist() for name, values in raw_importance.items()
            }
            model_records[f"selected_on_{label}"] = _persist_model(
                model,
                root / "models" / fold / f"selected_on_{label}",
                evaluation_x,
                score_mask,
            )
        stitched = stitch_block_parity_predictions(predictions[0], predictions[1])
        fold_root = root / fold
        manifest, manifest_sha = _persist_scores(
            fold_root,
            {
                "scores": stitched,
                "score_mask": score_mask,
                "selected_on_even_scores": predictions[0],
                "selected_on_odd_scores": predictions[1],
            },
            {
                "engine": "lightgbm",
                "rung": rung,
                "fold": fold,
                "feature_names": list(feature_names),
                "config": asdict(config),
                "pretrain_included": pretrain is not None,
                "time_decay_half_life_sessions": decay_half_life,
                "fit_date_indices": train_indices.tolist(),
                "selection_date_indices": evaluation_indices.tolist(),
                "models": model_records,
                "importance": importance,
            },
        )
        evaluated = _evaluate(
            store=store,
            indices=evaluation_indices,
            scores=stitched,
            score_mask=score_mask,
            cdi=cdi,
            source_hashes=source_hashes,
            fold=fold,
            output=fold_root / "evaluation.json",
            pathwise_scores=(predictions[0], predictions[1]),
        )
        reports[fold] = evaluated.report
        records[fold] = {
            "score_manifest": str(manifest),
            "score_manifest_sha256": manifest_sha,
            "evaluation": str(fold_root / "evaluation.json"),
            "evaluation_sha256": sha256_file(fold_root / "evaluation.json"),
        }
        del model, train_x, evaluation_x, train_y, selection_y
    return reports, records


def freeze_round1(
    *,
    store_root: Path,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    output_root: Path,
    num_threads: int,
) -> str:
    code = _git_identity()
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    store = store_root.resolve(strict=True)
    store_manifest, dates = _read_store_header(store)
    fit, selection, folds = _fold_indices(dates)
    _load_development_cdi(
        dates=dates,
        cdi_path=cdi_path,
        expected_sha256=cdi_sha256,
        experiment52_cdi_path=experiment52_cdi_path,
        experiment52_expected_sha256=experiment52_cdi_sha256,
    )
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "schema": ROUND1_SCHEMA,
        "status": "frozen_before_score",
        **RESEARCH_FLAGS,
        "frozen_at_utc": _utc_now(),
        "implementation": code,
        "preregistration": {
            "path": str(PREREGISTRATION.resolve(strict=True)),
            "sha256": sha256_file(PREREGISTRATION),
        },
        "store": {
            "root": str(store),
            "manifest_sha256": sha256_file(store / "manifest.json"),
            "schema": store_manifest["schema"],
        },
        "cdi": {
            "development_extension": {
                "path": str(cdi_path.resolve(strict=True)),
                "sha256": cdi_sha256,
            },
            "experiment52_reference": {
                "path": str(experiment52_cdi_path.resolve(strict=True)),
                "sha256": experiment52_cdi_sha256,
            },
        },
        "folds": folds,
        "baseline_roster": [
            "reversal_5",
            "reversal_21",
            "momentum_12_1",
            "reversal_5_momentum_12_1_blend",
        ],
        "gbdt_rungs": {name: list(groups) for name, groups in RUNG_GROUPS.items()},
        "gbdt_seeds": list(GBDT_SEEDS),
        "data_span_arms": ["fine_only", "pretrain_uniform", "pretrain_decay_756"],
        "gbdt_num_threads": num_threads,
        "bootstrap": {
            "replications": BOOTSTRAP_REPLICATIONS,
            "block_length_sessions": BOOTSTRAP_BLOCK,
            "seed": BOOTSTRAP_SEED,
            "fold_boundary_preserved": True,
        },
    }
    return write_json_atomic(output / "frozen_design.json", design)


def run_round1(
    *,
    output_root: Path,
    num_threads: int,
) -> str:
    output = output_root.resolve(strict=True)
    design_path = output / "frozen_design.json"
    design = _read_json(design_path)
    if (
        design.get("schema") != ROUND1_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-1 root is not a frozen registered design")
    code = _git_identity()
    if design.get("implementation") != code:
        raise ValueError("Round-1 implementation differs from the frozen commit")
    if num_threads != design.get("gbdt_num_threads"):
        raise ValueError("Round-1 thread setting differs from the frozen design")
    if (output / "round1_result.json").exists():
        raise FileExistsError(output / "round1_result.json")
    store_root = Path(str(design["store"]["root"]))
    store_manifest, dates = _read_store_header(store_root)
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("Round-1 store manifest hash mismatch")
    fit, selection, _ = _fold_indices(dates)
    pretrain = _pretrain_indices(dates)
    cdi_design = design["cdi"]
    cdi, cdi_provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(str(cdi_design["development_extension"]["path"])),
        expected_sha256=str(cdi_design["development_extension"]["sha256"]),
        experiment52_cdi_path=Path(str(cdi_design["experiment52_reference"]["path"])),
        experiment52_expected_sha256=str(
            cdi_design["experiment52_reference"]["sha256"]
        ),
    )
    store, access = _open_round_store(store_root, fit, selection, pretrain)
    source_hashes = {
        "v2_store_manifest": str(design["store"]["manifest_sha256"]),
        "cdi_development_extension": str(
            cdi_provenance["development_extension"]["sha256"]
        ),
        "cdi_experiment52_reference": str(
            cdi_provenance["experiment52_reference"]["sha256"]
        ),
        "preregistration": str(design["preregistration"]["sha256"]),
    }
    events: list[dict[str, object]] = [
        {"event": "round1_started", "at_utc": _utc_now()}
    ]
    try:
        baseline_start = max(0, min(int(x[0]) for x in selection.values()) - 253)
        baseline_end = max(int(x[-1]) for x in selection.values())
        baseline_axis = np.arange(baseline_start, baseline_end + 1, dtype=np.int64)
        panels = build_baselines(
            store.read("adjusted_close", baseline_axis),
            store.read("observed", baseline_axis),
            store.read("active", baseline_axis),
            store.read("ambiguous_action_mask", baseline_axis),
            slow_lag=1,
        )
        baseline_reports: dict[str, dict[str, dict[str, object]]] = {
            name: {} for name in panels
        }
        baseline_records: dict[str, dict[str, object]] = {name: {} for name in panels}
        for fold in ("F1", "F2", "F3"):
            indices = selection[fold]
            local = indices - baseline_start
            for name, panel in panels.items():
                assert isinstance(panel, BaselinePanel)
                root = output / "baselines" / name / fold
                manifest, manifest_sha = _persist_scores(
                    root,
                    {
                        "scores": panel.scores[local],
                        "score_mask": panel.score_mask[local],
                    },
                    {"engine": "naive_baseline", "name": name, "fold": fold},
                )
                result = _evaluate(
                    store=store,
                    indices=indices,
                    scores=panel.scores[local],
                    score_mask=panel.score_mask[local],
                    cdi=cdi,
                    source_hashes=source_hashes,
                    fold=fold,
                    output=root / "evaluation.json",
                )
                baseline_reports[name][fold] = result.report
                baseline_records[name][fold] = {
                    "score_manifest": str(manifest),
                    "score_manifest_sha256": manifest_sha,
                    "evaluation": str(root / "evaluation.json"),
                    "evaluation_sha256": sha256_file(root / "evaluation.json"),
                }
        baseline_summary = {
            name: _pooled_readouts(reports)
            for name, reports in baseline_reports.items()
        }
        events.append({"event": "baselines_completed", "at_utc": _utc_now()})

        rung_reports: dict[str, dict[str, dict[str, object]]] = {}
        rung_records: dict[str, object] = {}
        rung_summaries: dict[str, object] = {}
        rung_comparisons: dict[str, object] = {}
        kept: list[str] = []
        previous: str | None = None
        for rung in RUNG_GROUPS:
            reports, records = _run_gbdt_candidate(
                store=store,
                rung=rung,
                fit=fit,
                selection=selection,
                pretrain=None,
                decay_half_life=None,
                cdi=cdi,
                source_hashes=source_hashes,
                root=output / "gbdt_ladder" / rung,
                num_threads=num_threads,
            )
            rung_reports[rung] = reports
            rung_records[rung] = records
            rung_summaries[rung] = _pooled_readouts(reports)
            if previous is None:
                kept.append(rung)
            else:
                paired = _paired_readouts(reports, rung_reports[previous])
                rung_comparisons[f"{rung}_minus_{previous}"] = paired
                pooled = paired["pooled"]
                if not (
                    pooled["residual_ic"]["estimate"] < 0.0
                    and pooled["headline_net_excess_bps"]["estimate"] < 0.0
                ):
                    kept.append(rung)
            previous = rung
            events.append({"event": f"{rung}_completed", "at_utc": _utc_now()})
        parent = max(
            kept,
            key=lambda rung: (
                rung_summaries[rung]["pooled"]["residual_ic"]["estimate"],
                rung_summaries[rung]["pooled"]["headline_net_excess_bps"]["estimate"],
                -list(RUNG_GROUPS).index(rung),
            ),
        )

        span_reports: dict[str, dict[str, dict[str, object]]] = {
            "fine_only": rung_reports[parent]
        }
        span_records: dict[str, object] = {"fine_only": rung_records[parent]}
        for arm, decay in (("pretrain_uniform", None), ("pretrain_decay_756", 756.0)):
            reports, records = _run_gbdt_candidate(
                store=store,
                rung=parent,
                fit=fit,
                selection=selection,
                pretrain=pretrain,
                decay_half_life=decay,
                cdi=cdi,
                source_hashes=source_hashes,
                root=output / "gbdt_data_span" / arm,
                num_threads=num_threads,
            )
            span_reports[arm] = reports
            span_records[arm] = records
            events.append({"event": f"{arm}_completed", "at_utc": _utc_now()})
        span_summaries = {
            arm: _pooled_readouts(reports) for arm, reports in span_reports.items()
        }
        span_comparisons = {
            f"{arm}_minus_fine_only": _paired_readouts(
                reports, span_reports["fine_only"]
            )
            for arm, reports in span_reports.items()
            if arm != "fine_only"
        }
        events.append({"event": "round1_completed", "at_utc": _utc_now()})
        result = {
            "schema": ROUND1_SCHEMA,
            "status": "completed",
            **RESEARCH_FLAGS,
            "completed_at_utc": _utc_now(),
            "frozen_design": {
                "path": str(design_path),
                "sha256": sha256_file(design_path),
            },
            "implementation": code,
            "store_access": access,
            "sources": source_hashes,
            "baselines": {"artifacts": baseline_records, "readouts": baseline_summary},
            "gbdt_ladder": {
                "artifacts": rung_records,
                "readouts": rung_summaries,
                "paired_deltas": rung_comparisons,
                "kept_rungs": kept,
                "parent_rung": parent,
                "preference_rule": (
                    "keep a rung if pooled-IC delta is positive with an interval mostly "
                    "above zero OR headline net excess improves; drop a rung that worsens "
                    "both; keep ambiguous rungs; parent is best kept pooled IC with "
                    "economics as tie-break"
                ),
            },
            "gbdt_data_span_preview": {
                "artifacts": span_records,
                "readouts": span_summaries,
                "paired_deltas": span_comparisons,
                "decision_weight": "informational_only",
            },
            "operational_events": events,
        }
        return write_json_atomic(output / "round1_result.json", result)
    finally:
        store.close()


def _existing_score_and_evaluation_record(root: Path) -> dict[str, object]:
    manifest_path = root / "score_manifest.json"
    evaluation_path = root / "evaluation.json"
    manifest = _read_json(manifest_path)
    _assert_false_access(manifest, path=manifest_path)
    if (
        manifest.get("schema") != "BRAZIL_RV_V2_RESEARCH_SCORE_V1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError(f"incomplete registered score artifact: {root}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"score manifest lacks artifacts: {manifest_path}")
    for name, raw_record in artifacts.items():
        if not isinstance(name, str) or not isinstance(raw_record, Mapping):
            raise ValueError(f"malformed score artifact record: {manifest_path}")
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != int(raw_record["bytes"])
            or sha256_file(path) != raw_record["sha256"]
        ):
            raise ValueError(f"registered score artifact hash mismatch: {path}")
    _evaluation_from_path(evaluation_path)
    return {
        "score_manifest": str(manifest_path),
        "score_manifest_sha256": sha256_file(manifest_path),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
    }


def recover_round1_result(*, output_root: Path) -> str:
    """Serialize completed Round-1 artifacts after a reporting-only failure."""
    output = output_root.resolve(strict=True)
    result_path = output / "round1_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    design_path = output / "frozen_design.json"
    design = _read_json(design_path)
    if (
        design.get("schema") != ROUND1_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-1 root is not a frozen registered design")
    recovery_code = _git_identity()

    baseline_reports: dict[str, dict[str, dict[str, object]]] = {}
    baseline_records: dict[str, dict[str, object]] = {}
    for raw_name in design["baseline_roster"]:
        name = str(raw_name)
        baseline_reports[name] = {}
        baseline_records[name] = {}
        for fold in ("F1", "F2", "F3"):
            root = output / "baselines" / name / fold
            baseline_reports[name][fold] = _evaluation_from_path(
                root / "evaluation.json"
            )
            baseline_records[name][fold] = _existing_score_and_evaluation_record(root)
    baseline_summary = {
        name: _pooled_readouts(reports) for name, reports in baseline_reports.items()
    }

    rung_reports: dict[str, dict[str, dict[str, object]]] = {}
    rung_records: dict[str, object] = {}
    rung_summaries: dict[str, object] = {}
    rung_comparisons: dict[str, object] = {}
    kept: list[str] = []
    previous: str | None = None
    for rung in RUNG_GROUPS:
        reports: dict[str, dict[str, object]] = {}
        records: dict[str, object] = {}
        for fold in ("F1", "F2", "F3"):
            root = output / "gbdt_ladder" / rung / fold
            reports[fold] = _evaluation_from_path(root / "evaluation.json")
            records[fold] = _existing_score_and_evaluation_record(root)
        rung_reports[rung] = reports
        rung_records[rung] = records
        rung_summaries[rung] = _pooled_readouts(reports)
        if previous is None:
            kept.append(rung)
        else:
            paired = _paired_readouts(reports, rung_reports[previous])
            rung_comparisons[f"{rung}_minus_{previous}"] = paired
            pooled = paired["pooled"]
            if not (
                float(pooled["residual_ic"]["estimate"]) < 0.0
                and float(pooled["headline_net_excess_bps"]["estimate"]) < 0.0
            ):
                kept.append(rung)
        previous = rung
    parent = max(
        kept,
        key=lambda rung: (
            float(rung_summaries[rung]["pooled"]["residual_ic"]["estimate"]),
            float(
                rung_summaries[rung]["pooled"]["headline_net_excess_bps"]["estimate"]
            ),
            -list(RUNG_GROUPS).index(rung),
        ),
    )

    span_reports: dict[str, dict[str, dict[str, object]]] = {
        "fine_only": rung_reports[parent]
    }
    span_records: dict[str, object] = {"fine_only": rung_records[parent]}
    for arm in ("pretrain_uniform", "pretrain_decay_756"):
        reports = {}
        records = {}
        for fold in ("F1", "F2", "F3"):
            root = output / "gbdt_data_span" / arm / fold
            reports[fold] = _evaluation_from_path(root / "evaluation.json")
            records[fold] = _existing_score_and_evaluation_record(root)
        span_reports[arm] = reports
        span_records[arm] = records
    span_summaries = {
        arm: _pooled_readouts(reports) for arm, reports in span_reports.items()
    }
    span_comparisons = {
        f"{arm}_minus_fine_only": _paired_readouts(reports, span_reports["fine_only"])
        for arm, reports in span_reports.items()
        if arm != "fine_only"
    }

    store_root = Path(str(design["store"]["root"]))
    _, dates = _read_store_header(store_root)
    fit, selection, _ = _fold_indices(dates)
    store, access = _open_round_store(
        store_root, fit, selection, _pretrain_indices(dates)
    )
    store.close()
    source_hashes = {
        "v2_store_manifest": str(design["store"]["manifest_sha256"]),
        "cdi_development_extension": str(
            design["cdi"]["development_extension"]["sha256"]
        ),
        "cdi_experiment52_reference": str(
            design["cdi"]["experiment52_reference"]["sha256"]
        ),
        "preregistration": str(design["preregistration"]["sha256"]),
    }
    result = {
        "schema": ROUND1_SCHEMA,
        "status": "completed",
        **RESEARCH_FLAGS,
        "completed_at_utc": _utc_now(),
        "frozen_design": {
            "path": str(design_path),
            "sha256": sha256_file(design_path),
        },
        "implementation": recovery_code,
        "score_implementation": design["implementation"],
        "reporting_recovery": {
            "scope": (
                "serialize undefined zero-support readouts as JSON null; all score, "
                "model, and evaluation artifacts were hash-verified and reused without "
                "recomputation"
            ),
            "implementation": recovery_code,
        },
        "store_access": access,
        "sources": source_hashes,
        "baselines": {"artifacts": baseline_records, "readouts": baseline_summary},
        "gbdt_ladder": {
            "artifacts": rung_records,
            "readouts": rung_summaries,
            "paired_deltas": rung_comparisons,
            "kept_rungs": kept,
            "parent_rung": parent,
            "preference_rule": (
                "keep a rung if pooled-IC delta is positive with an interval mostly "
                "above zero OR headline net excess improves; drop a rung that worsens "
                "both; keep ambiguous rungs; parent is best kept pooled IC with "
                "economics as tie-break"
            ),
        },
        "gbdt_data_span_preview": {
            "artifacts": span_records,
            "readouts": span_summaries,
            "paired_deltas": span_comparisons,
            "decision_weight": "informational_only",
        },
        "operational_events": [
            {
                "event": "round1_reporting_recovered_from_completed_artifacts",
                "at_utc": _utc_now(),
            }
        ],
    }
    return write_json_atomic(result_path, result)


def _verify_sealed_root(root: Path, *, expected_schema: str) -> dict[str, object]:
    source = root.resolve(strict=True)
    inventory_path = source / "artifact_inventory.json"
    inventory_payload = _read_json(inventory_path)
    if (
        inventory_payload.get("schema") != "BRAZIL_RV_V2_RESEARCH_INVENTORY_V1"
        or inventory_payload.get("status") != "passed"
    ):
        raise ValueError("source research root lacks its passing inventory")
    rows = inventory_payload.get("files")
    excluded = inventory_payload.get("excluded_self")
    if (
        not isinstance(rows, list)
        or not isinstance(excluded, list)
        or any(not isinstance(row, dict) for row in rows)
        or any(not isinstance(value, str) for value in excluded)
    ):
        raise ValueError("source research inventory is malformed")
    if inventory(source, exclude=set(excluded)) != rows:
        raise ValueError("source research inventory no longer matches its files")
    result_name = (
        "round1_result.json"
        if expected_schema == ROUND1_SCHEMA
        else "round2_result.json"
    )
    result = _read_json(source / result_name)
    if result.get("schema") != expected_schema or result.get("status") != "completed":
        raise ValueError("source research result is incomplete")
    _assert_false_access(result, path=source / result_name)
    return result


def freeze_round2(
    *,
    round1_root: Path,
    store_root: Path,
    cdi_path: Path,
    cdi_sha256: str,
    experiment52_cdi_path: Path,
    experiment52_cdi_sha256: str,
    output_root: Path,
    fast_checkpoint: Path | None,
    fast_checkpoint_sha256: str | None,
    max_parallel: int,
) -> str:
    code = _git_identity()
    if not 4 <= max_parallel <= 6:
        raise ValueError("Round 2 requires four to six concurrent trajectories")
    round1_path = round1_root.resolve(strict=True)
    round1 = _verify_sealed_root(round1_path, expected_schema=ROUND1_SCHEMA)
    if round1.get("implementation") != code:
        raise ValueError("Round 1 and Round 2 must use the same frozen implementation")
    parent = round1["gbdt_ladder"]["parent_rung"]
    if parent not in RUNG_GROUPS:
        raise ValueError("Round-1 GBDT parent is not a registered rung")
    output = output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    store = store_root.resolve(strict=True)
    store_manifest, dates = _read_store_header(store)
    _fold_indices(dates)
    _load_development_cdi(
        dates=dates,
        cdi_path=cdi_path,
        expected_sha256=cdi_sha256,
        experiment52_cdi_path=experiment52_cdi_path,
        experiment52_expected_sha256=experiment52_cdi_sha256,
    )
    if (fast_checkpoint is None) != (fast_checkpoint_sha256 is None):
        raise ValueError("v1 fast checkpoint and SHA-256 must be set together")
    fast: dict[str, object]
    if fast_checkpoint is None:
        fast = {"used": False, "reason": "no compatible checkpoint bound before freeze"}
    else:
        checkpoint = fast_checkpoint.resolve(strict=True)
        actual = sha256_file(checkpoint)
        if actual != fast_checkpoint_sha256:
            raise ValueError("v1 fast checkpoint SHA-256 mismatch")
        fast = {"used": True, "path": str(checkpoint), "sha256": actual}
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "schema": ROUND2_SCHEMA,
        "status": "frozen_before_score",
        **RESEARCH_FLAGS,
        "frozen_at_utc": _utc_now(),
        "implementation": code,
        "preregistration": {
            "path": str(PREREGISTRATION.resolve(strict=True)),
            "sha256": sha256_file(PREREGISTRATION),
        },
        "round1": {
            "root": str(round1_path),
            "result_sha256": sha256_file(round1_path / "round1_result.json"),
            "inventory_sha256": sha256_file(round1_path / "artifact_inventory.json"),
            "gbdt_parent_rung": parent,
        },
        "store": {
            "root": str(store),
            "manifest_sha256": sha256_file(store / "manifest.json"),
            "schema": store_manifest["schema"],
        },
        "cdi": {
            "development_extension": {
                "path": str(cdi_path.resolve(strict=True)),
                "sha256": cdi_sha256,
            },
            "experiment52_reference": {
                "path": str(experiment52_cdi_path.resolve(strict=True)),
                "sha256": experiment52_cdi_sha256,
            },
        },
        "enabled_sidecars": list(RUNG_GROUPS[parent]),
        "v1_fast_initialization": fast,
        "network": {
            "seeds": list(NETWORK_SEEDS),
            "folds": ["F1", "F2", "F3"],
            "arms": ["A_fine_only", "B_pretrain_finetune", "C_joint_decay_756"],
            "maximum_epochs": 20,
            "patience": 3,
            "lambda_persistence": 0.0,
            "lookback_sessions": 60,
            "pairs_per_batch": 8,
            "pretrained_learning_rate_multiplier": 0.3,
            "joint_time_decay_half_life_sessions": 756.0,
        },
        "max_parallel_trajectories": max_parallel,
        "bootstrap": {
            "replications": BOOTSTRAP_REPLICATIONS,
            "block_length_sessions": BOOTSTRAP_BLOCK,
            "seed": BOOTSTRAP_SEED,
            "fold_boundary_preserved": True,
        },
    }
    return write_json_atomic(output / "frozen_design.json", design)


def _training_command(
    *,
    design: Mapping[str, object],
    output_dir: Path,
    stage: str,
    seed: int,
    fold: str | None = None,
    parity: int | None = None,
    pretrain_checkpoint: Path | None = None,
    pretrain_sha256: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "brazil_rv.v2.train",
        "--store",
        str(design["store"]["root"]),
        "--output-dir",
        str(output_dir),
        "--score-output-dir",
        str(output_dir / "scores"),
        "--stage",
        stage,
        "--seed",
        str(seed),
        "--maximum-epochs",
        "20",
        "--patience",
        "3",
        "--lookback",
        "60",
        "--pairs-per-batch",
        "8",
        "--selection-batch-size",
        "1",
        "--num-workers",
        "0",
        "--lambda-persistence",
        "0.0",
        "--device",
        "cuda",
    ]
    if fold is not None:
        command.extend(("--fold", fold))
    if parity is not None:
        command.extend(("--selection-parity", str(parity)))
    for group in design["enabled_sidecars"]:
        command.extend(("--sidecar", str(group)))
    fast = design["v1_fast_initialization"]
    if fast["used"]:
        command.extend(
            (
                "--fast-pretrained-checkpoint",
                str(fast["path"]),
                "--fast-pretrained-sha256",
                str(fast["sha256"]),
            )
        )
    if pretrain_checkpoint is not None:
        if pretrain_sha256 is None:
            raise ValueError("stage-P handoff hash is required")
        command.extend(
            (
                "--pretrain-checkpoint",
                str(pretrain_checkpoint),
                "--pretrain-sha256",
                pretrain_sha256,
            )
        )
    return command


def _plan_job(
    *,
    name: str,
    seed: int,
    fold: str,
    run_dir: Path,
    command: Sequence[str],
    stage: str,
    parity: int | None,
) -> dict[str, object]:
    return {
        "name": name,
        "seed": seed,
        "fold": fold,
        "run_dir": str(run_dir),
        "cwd": str(PROJECT_ROOT),
        "command": list(command),
        "expected_manifest": {
            "stage": stage,
            "selection_parity": parity,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    }


def write_round2_plan_p(*, output_root: Path) -> str:
    root = output_root.resolve(strict=True)
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != ROUND2_SCHEMA
        or design.get("status") != "frozen_before_score"
    ):
        raise ValueError("Round-2 root is not frozen")
    jobs = []
    for seed in NETWORK_SEEDS:
        run_dir = root / "trajectories" / "arm_B" / "stage_P" / f"seed_{seed}"
        jobs.append(
            _plan_job(
                name=f"arm_B_stage_P_seed_{seed}",
                seed=seed,
                fold="pretrain_internal",
                run_dir=run_dir,
                command=_training_command(
                    design=design, output_dir=run_dir, stage="P", seed=seed
                ),
                stage="P",
                parity=None,
            )
        )
    return write_json_atomic(
        root / "round2_plan_p.json",
        {
            "schema": "BRAZIL_RV_V2_RUN_MANY_PLAN_V1",
            "phase": "stage_P",
            "max_parallel": int(design["max_parallel_trajectories"]),
            "jobs": jobs,
        },
    )


def write_round2_plan_main(*, output_root: Path) -> str:
    root = output_root.resolve(strict=True)
    design = _read_json(root / "frozen_design.json")
    if not (root / "round2_plan_p.json").is_file():
        raise FileNotFoundError("Stage-P plan is absent")
    handoffs: dict[int, tuple[Path, str]] = {}
    for seed in NETWORK_SEEDS:
        p_root = root / "trajectories" / "arm_B" / "stage_P" / f"seed_{seed}"
        manifest = _read_json(p_root / "run_manifest.json")
        _assert_false_access(manifest, path=p_root / "run_manifest.json")
        if (
            manifest.get("status") != "completed"
            or manifest.get("stage") != "P"
            or manifest.get("seed") != seed
        ):
            raise ValueError(f"Stage-P trajectory is incomplete for seed {seed}")
        checkpoint = p_root / "raw_patience.pt"
        digest = sha256_file(checkpoint)
        if manifest.get("artifacts", {}).get("raw_patience.pt") != digest:
            raise ValueError(f"Stage-P checkpoint manifest mismatch for seed {seed}")
        handoffs[seed] = (checkpoint, digest)
    jobs = []
    arms = (
        ("arm_A", "F", False),
        ("arm_B", "F", True),
        ("arm_C", "J", False),
    )
    for arm, stage, uses_handoff in arms:
        for fold in ("F1", "F2", "F3"):
            for seed in NETWORK_SEEDS:
                for parity in (0, 1):
                    label = "even" if parity == 0 else "odd"
                    run_dir = (
                        root
                        / "trajectories"
                        / arm
                        / f"{fold}_seed_{seed}_select_{label}"
                    )
                    handoff, handoff_sha = (
                        handoffs[seed] if uses_handoff else (None, None)
                    )
                    jobs.append(
                        _plan_job(
                            name=f"{arm}_{fold}_seed_{seed}_select_{label}",
                            seed=seed,
                            fold=fold,
                            run_dir=run_dir,
                            command=_training_command(
                                design=design,
                                output_dir=run_dir,
                                stage=stage,
                                seed=seed,
                                fold=fold,
                                parity=parity,
                                pretrain_checkpoint=handoff,
                                pretrain_sha256=handoff_sha,
                            ),
                            stage=stage,
                            parity=parity,
                        )
                    )
    return write_json_atomic(
        root / "round2_plan_main.json",
        {
            "schema": "BRAZIL_RV_V2_RUN_MANY_PLAN_V1",
            "phase": "registered_arms",
            "max_parallel": int(design["max_parallel_trajectories"]),
            "stage_p_handoffs": {
                str(seed): {"path": str(path), "sha256": digest}
                for seed, (path, digest) in handoffs.items()
            },
            "jobs": jobs,
        },
    )


def _score_artifact(root: Path) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    manifest_path = root / "score_manifest.json"
    manifest = _read_json(manifest_path)
    _assert_false_access(manifest, path=manifest_path)
    if (
        manifest.get("schema") != "BRAZIL_RV_V2_SCORE_ARTIFACT_V1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError(f"incomplete score artifact: {root}")
    scores_path = root / "scores.npy"
    mask_path = root / "score_mask.npy"
    for path in (scores_path, mask_path):
        record = manifest["artifacts"][path.name]
        if (
            path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"score artifact hash mismatch: {path}")
    return (
        np.load(scores_path, allow_pickle=False),
        np.load(mask_path, allow_pickle=False),
    )


def _aggregate_network_fold(
    root: Path, arm: str, fold: str
) -> tuple[NDArray[np.float32], NDArray[np.bool_], tuple[NDArray[np.float32], ...]]:
    stitched = []
    paths = []
    reference_mask: NDArray[np.bool_] | None = None
    for seed in NETWORK_SEEDS:
        parity_scores = {}
        for parity, label in ((0, "even"), (1, "odd")):
            run = root / "trajectories" / arm / f"{fold}_seed_{seed}_select_{label}"
            manifest = _read_json(run / "run_manifest.json")
            _assert_false_access(manifest, path=run / "run_manifest.json")
            if (
                manifest.get("status") != "completed"
                or manifest.get("selection_parity") != parity
            ):
                raise ValueError(f"trajectory is incomplete: {run}")
            scores, mask = _score_artifact(run / "scores")
            if reference_mask is None:
                reference_mask = mask
            elif not np.array_equal(reference_mask, mask):
                raise ValueError("network member score masks differ")
            parity_scores[parity] = scores
            paths.append(scores)
        stitched.append(
            stitch_block_parity_predictions(parity_scores[0], parity_scores[1])
        )
    assert reference_mask is not None
    ensemble = rank_average_ensemble(stitched, reference_mask)
    return ensemble, reference_mask, tuple(paths)


def _load_round1_parent_fold(
    round1_root: Path, parent: str, fold: str
) -> tuple[NDArray[np.float32], NDArray[np.bool_], tuple[NDArray[np.float32], ...]]:
    root = round1_root / "gbdt_ladder" / parent / fold
    manifest = _read_json(root / "score_manifest.json")
    _assert_false_access(manifest, path=root / "score_manifest.json")
    arrays = {}
    for label in (
        "scores",
        "score_mask",
        "selected_on_even_scores",
        "selected_on_odd_scores",
    ):
        path = root / f"{label}.npy"
        record = manifest["artifacts"][path.name]
        if (
            path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"Round-1 parent score hash mismatch: {path}")
        arrays[label] = np.load(path, allow_pickle=False)
    return (
        arrays["scores"],
        arrays["score_mask"],
        (arrays["selected_on_even_scores"], arrays["selected_on_odd_scores"]),
    )


def _evaluation_from_path(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    _assert_false_access(payload, path=path)
    if payload.get("schema") != "BRAZIL_RV_V2_EVALUATION_V1":
        raise ValueError(f"not a v2 evaluation: {path}")
    return payload


def finalize_round2(*, output_root: Path) -> str:
    root = output_root.resolve(strict=True)
    design = _read_json(root / "frozen_design.json")
    if (
        design.get("schema") != ROUND2_SCHEMA
        or design.get("implementation") != _git_identity()
    ):
        raise ValueError("Round-2 design differs from the clean implementation")
    if (root / "round2_result.json").exists():
        raise FileExistsError(root / "round2_result.json")
    round1_root = Path(str(design["round1"]["root"]))
    _verify_sealed_root(round1_root, expected_schema=ROUND1_SCHEMA)
    parent = str(design["round1"]["gbdt_parent_rung"])
    store_root = Path(str(design["store"]["root"]))
    store_manifest, dates = _read_store_header(store_root)
    fit, selection, _ = _fold_indices(dates)
    pretrain = _pretrain_indices(dates)
    store, access = _open_round_store(store_root, fit, selection, pretrain)
    cdi_design = design["cdi"]
    cdi, provenance = _load_development_cdi(
        dates=dates,
        cdi_path=Path(str(cdi_design["development_extension"]["path"])),
        expected_sha256=str(cdi_design["development_extension"]["sha256"]),
        experiment52_cdi_path=Path(str(cdi_design["experiment52_reference"]["path"])),
        experiment52_expected_sha256=str(
            cdi_design["experiment52_reference"]["sha256"]
        ),
    )
    source_hashes = {
        "v2_store_manifest": str(design["store"]["manifest_sha256"]),
        "cdi_development_extension": str(provenance["development_extension"]["sha256"]),
        "cdi_experiment52_reference": str(
            provenance["experiment52_reference"]["sha256"]
        ),
        "preregistration": str(design["preregistration"]["sha256"]),
        "round1_result": str(design["round1"]["result_sha256"]),
    }
    arm_reports: dict[str, dict[str, dict[str, object]]] = {}
    arm_artifacts: dict[str, object] = {}
    try:
        for arm in ("arm_A", "arm_B", "arm_C"):
            arm_reports[arm] = {}
            arm_artifacts[arm] = {}
            for fold in ("F1", "F2", "F3"):
                scores, mask, paths = _aggregate_network_fold(root, arm, fold)
                aggregate = root / "aggregates" / arm / fold
                manifest, digest = _persist_scores(
                    aggregate,
                    {"scores": scores, "score_mask": mask},
                    {
                        "engine": "starter_network",
                        "arm": arm,
                        "fold": fold,
                        "seeds": list(NETWORK_SEEDS),
                        "seed_aggregation": "opposite-parity stitch then tie-aware rank average",
                    },
                )
                evaluated = _evaluate(
                    store=store,
                    indices=selection[fold],
                    scores=scores,
                    score_mask=mask,
                    cdi=cdi,
                    source_hashes=source_hashes,
                    fold=fold,
                    output=aggregate / "evaluation.json",
                    pathwise_scores=paths,
                )
                arm_reports[arm][fold] = evaluated.report
                arm_artifacts[arm][fold] = {
                    "score_manifest": str(manifest),
                    "score_manifest_sha256": digest,
                    "evaluation": str(aggregate / "evaluation.json"),
                    "evaluation_sha256": sha256_file(aggregate / "evaluation.json"),
                }
        arm_readouts = {
            arm: _pooled_readouts(reports) for arm, reports in arm_reports.items()
        }
        arm_deltas = {
            "B_minus_A": _paired_readouts(arm_reports["arm_B"], arm_reports["arm_A"]),
            "C_minus_A": _paired_readouts(arm_reports["arm_C"], arm_reports["arm_A"]),
        }
        eligible = ["arm_A"]
        a_economics = arm_readouts["arm_A"]["pooled"]["headline_net_excess_bps"][
            "estimate"
        ]
        for arm in ("arm_B", "arm_C"):
            if (
                arm_readouts[arm]["pooled"]["headline_net_excess_bps"]["estimate"]
                >= a_economics
            ):
                eligible.append(arm)
        long_small_and_uncertain = all(
            arm_deltas[label]["pooled"]["residual_ic"]["estimate"] <= 0.002
            and arm_deltas[label]["pooled"]["residual_ic"]["lower_95"]
            <= 0.0
            <= arm_deltas[label]["pooled"]["residual_ic"]["upper_95"]
            for label in ("B_minus_A", "C_minus_A")
        )
        arm_order = {"arm_A": 2, "arm_B": 1, "arm_C": 0}
        chosen_arm = (
            "arm_A"
            if long_small_and_uncertain
            else max(
                eligible,
                key=lambda arm: (
                    arm_readouts[arm]["pooled"]["residual_ic"]["estimate"],
                    arm_readouts[arm]["pooled"]["headline_net_excess_bps"]["estimate"],
                    arm_order[arm],
                ),
            )
        )

        comparator_reports: dict[str, dict[str, dict[str, object]]] = {
            "network": arm_reports[chosen_arm],
            "gbdt": {},
            "ensemble": {},
        }
        comparator_artifacts: dict[str, object] = {
            "network": arm_artifacts[chosen_arm],
            "gbdt": {},
            "ensemble": {},
        }
        for fold in ("F1", "F2", "F3"):
            network_scores = np.load(
                root / "aggregates" / chosen_arm / fold / "scores.npy",
                allow_pickle=False,
            )
            network_mask = np.load(
                root / "aggregates" / chosen_arm / fold / "score_mask.npy",
                allow_pickle=False,
            )
            _, _, network_paths = _aggregate_network_fold(root, chosen_arm, fold)
            gbdt_scores, gbdt_mask, gbdt_paths = _load_round1_parent_fold(
                round1_root, parent, fold
            )
            if not np.array_equal(network_mask, gbdt_mask):
                raise ValueError("network and GBDT score masks differ")
            gbdt_root = root / "comparators" / "gbdt" / fold
            g_manifest, g_digest = _persist_scores(
                gbdt_root,
                {"scores": gbdt_scores, "score_mask": gbdt_mask},
                {"engine": "round1_gbdt_parent", "parent_rung": parent, "fold": fold},
            )
            g_eval = _evaluate(
                store=store,
                indices=selection[fold],
                scores=gbdt_scores,
                score_mask=gbdt_mask,
                cdi=cdi,
                source_hashes=source_hashes,
                fold=fold,
                output=gbdt_root / "evaluation.json",
                pathwise_scores=gbdt_paths,
            )
            comparator_reports["gbdt"][fold] = g_eval.report
            comparator_artifacts["gbdt"][fold] = {
                "score_manifest": str(g_manifest),
                "score_manifest_sha256": g_digest,
                "evaluation": str(gbdt_root / "evaluation.json"),
                "evaluation_sha256": sha256_file(gbdt_root / "evaluation.json"),
            }

            ensemble_scores = rank_average_ensemble(
                (network_scores, gbdt_scores), network_mask
            )
            parity_ensembles = tuple(
                rank_average_ensemble(
                    (network_path, gbdt_paths[index % 2]), network_mask
                )
                for index, network_path in enumerate(network_paths)
            )
            ensemble_root = root / "comparators" / "ensemble" / fold
            e_manifest, e_digest = _persist_scores(
                ensemble_root,
                {"scores": ensemble_scores, "score_mask": network_mask},
                {
                    "engine": "fixed_equal_weight_rank_average",
                    "members": ["network", "gbdt"],
                    "weights": [0.5, 0.5],
                    "fold": fold,
                },
            )
            e_eval = _evaluate(
                store=store,
                indices=selection[fold],
                scores=ensemble_scores,
                score_mask=network_mask,
                cdi=cdi,
                source_hashes=source_hashes,
                fold=fold,
                output=ensemble_root / "evaluation.json",
                pathwise_scores=parity_ensembles,
            )
            comparator_reports["ensemble"][fold] = e_eval.report
            comparator_artifacts["ensemble"][fold] = {
                "score_manifest": str(e_manifest),
                "score_manifest_sha256": e_digest,
                "evaluation": str(ensemble_root / "evaluation.json"),
                "evaluation_sha256": sha256_file(ensemble_root / "evaluation.json"),
            }
        comparator_readouts = {
            name: _pooled_readouts(reports)
            for name, reports in comparator_reports.items()
        }
        comparator_deltas = {
            "network_minus_gbdt": _paired_readouts(
                comparator_reports["network"], comparator_reports["gbdt"]
            ),
            "ensemble_minus_gbdt": _paired_readouts(
                comparator_reports["ensemble"], comparator_reports["gbdt"]
            ),
            "ensemble_minus_network": _paired_readouts(
                comparator_reports["ensemble"], comparator_reports["network"]
            ),
        }
        parent_order = {"network": 2, "gbdt": 1, "ensemble": 0}
        v2_parent = max(
            comparator_readouts,
            key=lambda name: (
                comparator_readouts[name]["pooled"]["residual_ic"]["estimate"],
                comparator_readouts[name]["pooled"]["headline_net_excess_bps"][
                    "estimate"
                ],
                parent_order[name],
            ),
        )
        stage_p = {}
        for seed in NETWORK_SEEDS:
            p_root = root / "trajectories" / "arm_B" / "stage_P" / f"seed_{seed}"
            history = _read_json(p_root / "history.json")
            manifest = _read_json(p_root / "run_manifest.json")
            _assert_false_access(manifest, path=p_root / "run_manifest.json")
            stage_p[str(seed)] = {
                "history": str(p_root / "history.json"),
                "history_sha256": sha256_file(p_root / "history.json"),
                "raw_patience_checkpoint_sha256": sha256_file(
                    p_root / "raw_patience.pt"
                ),
                "final_ema_checkpoint_sha256": sha256_file(p_root / "final_ema.pt"),
                "selected_epoch": manifest.get("selected_epoch"),
                "stopped_epoch": manifest.get("stopped_epoch"),
                "holdout_history": history,
            }
        result = {
            "schema": ROUND2_SCHEMA,
            "status": "completed",
            **RESEARCH_FLAGS,
            "completed_at_utc": _utc_now(),
            "frozen_design": {
                "path": str(root / "frozen_design.json"),
                "sha256": sha256_file(root / "frozen_design.json"),
            },
            "implementation": _git_identity(),
            "store_access": access,
            "sources": source_hashes,
            "stage_p_holdout": stage_p,
            "data_span_arms": {
                "artifacts": arm_artifacts,
                "readouts": arm_readouts,
                "paired_deltas": arm_deltas,
                "eligible_by_economics": eligible,
                "long_history_small_and_uncertain": long_small_and_uncertain,
                "chosen_arm": chosen_arm,
                "preference_rule": (
                    "highest mean pooled IC unless headline economics are worse than Arm A; "
                    "if neither long-history arm beats A by more than 0.002 with intervals "
                    "spanning zero, prefer A"
                ),
            },
            "parent_comparison": {
                "artifacts": comparator_artifacts,
                "readouts": comparator_readouts,
                "paired_deltas": comparator_deltas,
                "v2_parent": v2_parent,
                "preference_rule": "best pooled IC with economics as tie-break",
            },
        }
        return write_json_atomic(root / "round2_result.json", result)
    finally:
        store.close()


def seal_root(
    *, root: Path, stdout_log: Path | None = None, stderr_log: Path | None = None
) -> str:
    output = root.resolve(strict=True)
    for source, label in ((stdout_log, "stdout.log"), (stderr_log, "stderr.log")):
        if source is not None:
            destination = output / "operational_logs" / label
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.copyfile(source.resolve(strict=True), destination)
    json_paths = [
        path
        for path in output.rglob("*.json")
        if path.name not in {"artifact_inventory.json", "access_audit.json"}
    ]
    flagged = []
    for path in json_paths:
        payload = _read_json(path)
        if "official_validation_accessed" in payload or "test_accessed" in payload:
            _assert_false_access(payload, path=path)
            flagged.append(path.relative_to(output).as_posix())
    audit = {
        "schema": "BRAZIL_RV_V2_RESEARCH_ACCESS_AUDIT_V1",
        "status": "passed",
        **RESEARCH_FLAGS,
        "json_artifacts_with_access_flags": flagged,
        "json_artifact_count": len(json_paths),
        "audited_at_utc": _utc_now(),
    }
    write_json_atomic(output / "access_audit.json", audit)
    excluded = {"artifact_inventory.json", "artifact_inventory.json.sha256"}
    rows = inventory(output, exclude=excluded)
    inventory_payload = {
        "schema": "BRAZIL_RV_V2_RESEARCH_INVENTORY_V1",
        "status": "passed",
        **RESEARCH_FLAGS,
        "excluded_self": sorted(excluded),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "files": rows,
    }
    digest = write_json_atomic(output / "artifact_inventory.json", inventory_payload)
    if inventory(output, exclude=excluded) != rows:
        raise RuntimeError("research root changed while its inventory was sealed")
    verify_inventory(output, inventory(output))
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run registered v2 research rounds")
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-round1")
    freeze.add_argument("--store", type=Path, required=True)
    freeze.add_argument("--cdi", type=Path, required=True)
    freeze.add_argument("--cdi-sha256", required=True)
    freeze.add_argument("--experiment52-cdi", type=Path, required=True)
    freeze.add_argument("--experiment52-cdi-sha256", required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--num-threads", type=int, default=0)
    run = commands.add_parser("run-round1")
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--num-threads", type=int, default=0)
    recover = commands.add_parser("recover-round1-result")
    recover.add_argument("--output-root", type=Path, required=True)
    seal = commands.add_parser("seal-root")
    seal.add_argument("--root", type=Path, required=True)
    seal.add_argument("--stdout-log", type=Path)
    seal.add_argument("--stderr-log", type=Path)
    freeze2 = commands.add_parser("freeze-round2")
    freeze2.add_argument("--round1-root", type=Path, required=True)
    freeze2.add_argument("--store", type=Path, required=True)
    freeze2.add_argument("--cdi", type=Path, required=True)
    freeze2.add_argument("--cdi-sha256", required=True)
    freeze2.add_argument("--experiment52-cdi", type=Path, required=True)
    freeze2.add_argument("--experiment52-cdi-sha256", required=True)
    freeze2.add_argument("--output-root", type=Path, required=True)
    freeze2.add_argument("--fast-checkpoint", type=Path)
    freeze2.add_argument("--fast-checkpoint-sha256")
    freeze2.add_argument("--max-parallel", type=int, default=4)
    plan_p = commands.add_parser("write-round2-plan-p")
    plan_p.add_argument("--output-root", type=Path, required=True)
    plan_main = commands.add_parser("write-round2-plan-main")
    plan_main.add_argument("--output-root", type=Path, required=True)
    finalize = commands.add_parser("finalize-round2")
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze-round1":
        digest = freeze_round1(
            store_root=arguments.store,
            cdi_path=arguments.cdi,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            output_root=arguments.output_root,
            num_threads=arguments.num_threads,
        )
    elif arguments.command == "run-round1":
        digest = run_round1(
            output_root=arguments.output_root,
            num_threads=arguments.num_threads,
        )
    elif arguments.command == "recover-round1-result":
        digest = recover_round1_result(output_root=arguments.output_root)
    elif arguments.command == "freeze-round2":
        digest = freeze_round2(
            round1_root=arguments.round1_root,
            store_root=arguments.store,
            cdi_path=arguments.cdi,
            cdi_sha256=arguments.cdi_sha256,
            experiment52_cdi_path=arguments.experiment52_cdi,
            experiment52_cdi_sha256=arguments.experiment52_cdi_sha256,
            output_root=arguments.output_root,
            fast_checkpoint=arguments.fast_checkpoint,
            fast_checkpoint_sha256=arguments.fast_checkpoint_sha256,
            max_parallel=arguments.max_parallel,
        )
    elif arguments.command == "write-round2-plan-p":
        digest = write_round2_plan_p(output_root=arguments.output_root)
    elif arguments.command == "write-round2-plan-main":
        digest = write_round2_plan_main(output_root=arguments.output_root)
    elif arguments.command == "finalize-round2":
        digest = finalize_round2(output_root=arguments.output_root)
    else:
        digest = seal_root(
            root=arguments.root,
            stdout_log=arguments.stdout_log,
            stderr_log=arguments.stderr_log,
        )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
