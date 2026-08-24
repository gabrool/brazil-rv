from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brazil_rv.preprocessing.contract import (
    DECISION_EQUITY_INDICES,
    DYNAMIC_CHANNELS,
    SLOW_CHANNELS,
)

from .analyze import compare_observation_ensembles, load_run_observations
from .contract import ALLOWED_SEEDS, GH200_RUNTIME, HORIZONS
from .data import (
    create_evaluation_loader,
    feature_store_identity,
    load_sample_index,
    select_sample_split,
    select_training_window,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_equity_input_ablation_predictions,
    compile_model,
)
from .metrics import (
    daily_horizon_ic,
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .model import build_model
from .provenance import repository_commit
from .three_fold_sidecar_screen import crossfit_patience_observations
from .train import run_training
from .trajectory import load_checkpoint, predictions_for_rule

FOLDS = ("fold_c", "fold_a", "fold_b")
CORRELATION_THRESHOLD = 0.80
REPRESENTATIVE_TEST_THRESHOLD = 0.00025
PREVIEW_MEAN_LIMIT = 0.00025
PREVIEW_FOLD_LIMIT = 0.0005
NONINFERIOR_FOLD_FLOOR = -0.0005
IMPROVEMENT_MEAN = 0.0005
R2_TIE_TOLERANCE = 0.00025
VISIBLE_EQUITY_MINUTES = DECISION_EQUITY_INDICES[-1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_artifact_inventory(root: Path) -> Path:
    path = root / "artifact_inventory.json"
    rows = []
    for artifact in sorted(root.rglob("*")):
        if artifact.is_file() and artifact != path:
            rows.append(
                {
                    "path": artifact.relative_to(root).as_posix(),
                    "bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact),
                }
            )
    _atomic_json(
        path,
        {
            "schema": "FEATURE_REMOVAL_ARTIFACT_INVENTORY_V1",
            "created_at": _now(),
            "object_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "artifacts": rows,
        },
    )
    return path


def _feature_key(kind: str, index: int) -> str:
    return f"{kind}_{index}"


def _feature_rows() -> list[dict[str, object]]:
    rows = []
    for kind, names in (("dynamic", DYNAMIC_CHANNELS), ("slow", SLOW_CHANNELS)):
        for index, name in enumerate(names):
            rows.append(
                {
                    "key": _feature_key(kind, index),
                    "kind": kind,
                    "index": index,
                    "name": name,
                    "global_index": len(rows),
                    "model_input_missing_fraction": 0.0,
                }
            )
    return rows


FEATURES = tuple(_feature_rows())
FEATURE_BY_KEY = {str(row["key"]): row for row in FEATURES}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def _rank_standardize(values: np.ndarray) -> np.ndarray:
    ranked = np.empty(values.shape, dtype=np.float64)
    for column in range(values.shape[1]):
        ranked[:, column] = _average_ranks(values[:, column])
    ranked -= ranked.mean(axis=0)
    norms = np.sqrt(np.square(ranked).sum(axis=0))
    valid = norms > 0
    ranked[:, valid] /= norms[valid]
    ranked[:, ~valid] = np.nan
    return ranked


def _accumulate(total: np.ndarray, count: np.ndarray, values: np.ndarray) -> None:
    finite = np.isfinite(values)
    total[finite] += values[finite]
    count[finite] += 1


def _mean_matrix(total: np.ndarray, count: np.ndarray) -> np.ndarray:
    output = np.zeros_like(total)
    np.divide(total, count, out=output, where=count > 0)
    return output


def _semantic_groups() -> list[dict[str, object]]:
    return [
        {
            "set_id": "semantic_beta_family",
            "members": [f"slow_{index}" for index in range(20, 26)],
        },
        {
            "set_id": "semantic_realized_volatility_family",
            "members": [
                "dynamic_10",
                "dynamic_11",
                "dynamic_12",
                "slow_9",
                "slow_10",
                "slow_11",
            ],
        },
        {
            "set_id": "semantic_observed_fraction_family",
            "members": ["dynamic_15", "slow_15", "slow_16"],
        },
        {
            "set_id": "semantic_calendar_family",
            "members": [f"slow_{index}" for index in range(26, 30)],
        },
        {
            "set_id": "semantic_market_aggregate_family",
            "members": [f"dynamic_{index}" for index in range(16, 22)],
        },
        {
            "set_id": "semantic_dollar_real_volume_family",
            "members": ["dynamic_24", "slow_12", "slow_13", "slow_14", "slow_18"],
        },
    ]


def _components(matrix: np.ndarray) -> list[list[int]]:
    parent = list(range(matrix.shape[0]))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(matrix.shape[0]):
        for right in range(left + 1, matrix.shape[0]):
            if abs(float(matrix[left, right])) >= CORRELATION_THRESHOLD:
                union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(matrix.shape[0]):
        grouped.setdefault(find(index), []).append(index)
    return sorted(grouped.values(), key=lambda values: (values[0], len(values)))


def build_stage_a(
    *,
    store: Path,
    output_dir: Path,
    feature_keys: Sequence[str] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_keys = tuple(FEATURE_BY_KEY if feature_keys is None else feature_keys)
    if not selected_keys or len(set(selected_keys)) != len(selected_keys):
        raise ValueError("Stage A feature keys must be nonempty and unique")
    unknown = sorted(set(selected_keys).difference(FEATURE_BY_KEY))
    if unknown:
        raise ValueError(f"Unknown Stage A feature keys: {unknown}")
    selected_features = tuple(FEATURE_BY_KEY[key] for key in selected_keys)
    dynamic_indices = tuple(
        int(row["index"]) for row in selected_features if row["kind"] == "dynamic"
    )
    slow_indices = tuple(
        int(row["index"]) for row in selected_features if row["kind"] == "slow"
    )
    ordered_keys = tuple(
        _feature_key("dynamic", index) for index in dynamic_indices
    ) + tuple(_feature_key("slow", index) for index in slow_indices)
    if ordered_keys != selected_keys:
        raise ValueError("Stage A feature keys must preserve canonical feature order")
    sample_index = load_sample_index(store)
    training = select_sample_split(sample_index, "train")
    date_indices = np.unique(training.get_column("date_idx").to_numpy())
    if date_indices.size != 716:
        raise ValueError("Stage A requires exactly 716 training dates")
    dynamic = np.load(store / "equity_features.npy", mmap_mode="r", allow_pickle=False)
    slow = np.load(store / "equity_slow.npy", mmap_mode="r", allow_pickle=False)
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False)
    dynamic_count = len(dynamic_indices)
    slow_count_value = len(slow_indices)
    dynamic_cross_total = np.zeros(
        (dynamic_count, dynamic_count), dtype=np.float64
    )
    dynamic_cross_count = np.zeros(
        (dynamic_count, dynamic_count), dtype=np.int64
    )
    slow_total = np.zeros((slow_count_value, slow_count_value), dtype=np.float64)
    slow_count = np.zeros((slow_count_value, slow_count_value), dtype=np.int64)
    cross_total = np.zeros((dynamic_count, slow_count_value), dtype=np.float64)
    cross_count = np.zeros((dynamic_count, slow_count_value), dtype=np.int64)
    decision_minutes = np.asarray(DECISION_EQUITY_INDICES) - 1
    active_by_date: dict[int, np.ndarray] = {}
    for position, date_index in enumerate(date_indices):
        active = np.asarray(membership[date_index] & ready[date_index], dtype=bool)
        active_by_date[int(date_index)] = active
        slow_ranked = _rank_standardize(
            np.asarray(slow[date_index, active], dtype=np.float64)[
                :, slow_indices
            ]
        )
        _accumulate(slow_total, slow_count, slow_ranked.T @ slow_ranked)
        for minute in decision_minutes:
            dynamic_ranked = _rank_standardize(
                np.asarray(dynamic[date_index, active, minute], dtype=np.float64)[
                    :, dynamic_indices
                ]
            )
            _accumulate(
                dynamic_cross_total,
                dynamic_cross_count,
                dynamic_ranked.T @ dynamic_ranked,
            )
            _accumulate(
                cross_total,
                cross_count,
                dynamic_ranked.T @ slow_ranked,
            )
        if (position + 1) % 50 == 0:
            print(f"stage_a_cross_section_dates={position + 1}/716", flush=True)
    temporal_total = np.zeros((dynamic_count, dynamic_count), dtype=np.float64)
    temporal_count = np.zeros((dynamic_count, dynamic_count), dtype=np.int64)
    for equity in range(dynamic.shape[1]):
        active_dates = np.asarray(
            [index for index in date_indices if active_by_date[int(index)][equity]],
            dtype=np.int64,
        )
        values = np.asarray(
            dynamic[active_dates, equity, :VISIBLE_EQUITY_MINUTES],
            dtype=np.float64,
        )[..., dynamic_indices].reshape(-1, dynamic_count)
        ranked = _rank_standardize(values)
        _accumulate(temporal_total, temporal_count, ranked.T @ ranked)
        if (equity + 1) % 20 == 0:
            print(
                f"stage_a_time_series_equities={equity + 1}/{dynamic.shape[1]}",
                flush=True,
            )
    dynamic_cross = _mean_matrix(dynamic_cross_total, dynamic_cross_count)
    dynamic_temporal = _mean_matrix(temporal_total, temporal_count)
    slow_matrix = _mean_matrix(slow_total, slow_count)
    cross_matrix = _mean_matrix(cross_total, cross_count)
    feature_count = len(selected_features)
    matrix = np.eye(feature_count, dtype=np.float64)
    choose_temporal = np.abs(dynamic_temporal) > np.abs(dynamic_cross)
    matrix[:dynamic_count, :dynamic_count] = np.where(
        choose_temporal, dynamic_temporal, dynamic_cross
    )
    matrix[dynamic_count:, dynamic_count:] = slow_matrix
    matrix[:dynamic_count, dynamic_count:] = cross_matrix
    matrix[dynamic_count:, :dynamic_count] = cross_matrix.T
    np.fill_diagonal(matrix, 1.0)
    components = _components(matrix)
    clusters = []
    for number, indices in enumerate(components, 1):
        internal = matrix[np.ix_(indices, indices)]
        off_diagonal = np.abs(internal[~np.eye(len(indices), dtype=bool)])
        clusters.append(
            {
                "set_id": f"correlation_cluster_{number:02d}",
                "members": [str(selected_features[index]["key"]) for index in indices],
                "member_names": [
                    str(selected_features[index]["name"]) for index in indices
                ],
                "max_internal_absolute_rho": (
                    0.0 if not off_diagonal.size else float(off_diagonal.max())
                ),
            }
        )
    pair_rows = []
    for left in range(feature_count):
        for right in range(left + 1, feature_count):
            pair_rows.append(
                {
                    "left": str(selected_features[left]["key"]),
                    "right": str(selected_features[right]["key"]),
                    "rho": float(matrix[left, right]),
                    "absolute_rho": abs(float(matrix[left, right])),
                    "edge": abs(float(matrix[left, right])) >= CORRELATION_THRESHOLD,
                }
            )
    pair_rows.sort(key=lambda row: -float(row["absolute_rho"]))
    cluster_table = {
        "schema": (
            "FEATURE_REMOVAL_CLUSTER_TABLE_V1"
            if feature_count == len(FEATURES)
            else "FEATURE_REMOVAL_CLUSTER_TABLE_V2"
        ),
        "created_at": _now(),
        "feature_store_identity": feature_store_identity(store),
        "training_date_count": int(date_indices.size),
        "correlation_contract": {
            "threshold": CORRELATION_THRESHOLD,
            "slow_slow": "mean per-date active-equity cross-sectional Spearman",
            "dynamic_dynamic": (
                "larger-absolute signed value of mean per-(date,decision) active-equity "
                "cross-sectional Spearman and mean per-equity time-series Spearman"
            ),
            "dynamic_slow": (
                "mean per-(date,decision) active-equity cross-sectional Spearman"
            ),
            "dynamic_time_series_minutes": f"model-visible minutes 0:{VISIBLE_EQUITY_MINUTES}",
            "ties": "average ranks",
            "future_data_used": False,
        },
        "features": list(selected_features),
        "correlation_matrix": matrix.tolist(),
        "pairs": pair_rows,
        "clusters": clusters,
        "semantic_groups": [
            {**group, "members": members}
            for group in _semantic_groups()
            if len(
                members := [
                    key for key in group["members"] if key in selected_keys
                ]
            )
            >= 2
        ],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    path = output_dir / "cluster_table.json"
    _atomic_json(path, cluster_table)
    _atomic_json(
        output_dir / "stage_a_manifest.json",
        {
            "schema": "FEATURE_REMOVAL_STAGE_A_V1",
            "status": "completed",
            "created_at": _now(),
            "cluster_table": str(path.resolve()),
            "cluster_table_sha256": _sha256(path),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return path


def _ensemble(
    members: Mapping[str, EvaluationObservations],
) -> EvaluationObservations:
    reference = next(iter(members.values()))
    for member in members.values():
        assert_observations_aligned(reference, member)
    return replace(
        reference,
        predictions=rank_average_predictions(
            [member.predictions for member in members.values()], reference.label_mask
        ),
    )


def _comparison(
    parent: EvaluationObservations, candidate: EvaluationObservations
) -> dict[str, object]:
    assert_observations_aligned(parent, candidate)
    parent_ic = sample_level_spearman_ic(
        parent.predictions, parent.targets, parent.label_mask
    )
    candidate_ic = sample_level_spearman_ic(
        candidate.predictions, candidate.targets, candidate.label_mask
    )
    dates, parent_daily = per_date_primary_ic(parent_ic, parent.date_idx)
    _, candidate_daily = per_date_primary_ic(candidate_ic, candidate.date_idx)
    _, parent_horizon = daily_horizon_ic(parent_ic, parent.date_idx)
    _, candidate_horizon = daily_horizon_ic(candidate_ic, candidate.date_idx)
    daily_drop = parent_daily - candidate_daily
    interval = moving_block_bootstrap(
        daily_drop,
        replications=10_000,
        block_length=10,
        seed=20260833,
    )
    return {
        "parent_minus_ablated_ic": finite_mean(daily_drop),
        "per_horizon_parent_minus_ablated_ic": {
            str(minutes): finite_mean(
                parent_horizon[:, index] - candidate_horizon[:, index]
            )
            for index, minutes in enumerate(HORIZONS)
        },
        "date_count": int(dates.size),
        "block10_interval": {
            key: np.asarray(value).tolist() for key, value in interval.items()
        },
    }


def _date_parities(date_idx: np.ndarray) -> dict[str, np.ndarray]:
    dates = np.unique(date_idx)
    return {"odd": dates[0::2], "even": dates[1::2]}


def _slice_observations(
    observations: EvaluationObservations, mask: np.ndarray
) -> EvaluationObservations:
    return replace(
        observations,
        predictions=observations.predictions[mask],
        targets=observations.targets[mask],
        raw_returns=observations.raw_returns[mask],
        label_mask=observations.label_mask[mask],
        sample_id=observations.sample_id[mask],
        date_idx=observations.date_idx[mask],
        decision_idx=observations.decision_idx[mask],
    )


def _fold_c_replays(report: Path) -> dict[str, list[dict[str, object]]]:
    value = _read_json(report)
    replays = value.get("comparison_metadata", {}).get("parent_patience_replays")
    if not isinstance(replays, dict) or set(replays) != {
        f"seed_{seed}" for seed in ALLOWED_SEEDS
    }:
        raise ValueError("Fold-C replay report lacks three frozen parent seeds")
    return replays


def _fold_ab_replays(
    report: Path,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    value = _read_json(report)
    if (
        value.get("official_validation_accessed") is not False
        or value.get("test_accessed") is not False
        or value.get("matched_seed_contract", {}).get("replay_epochs_exact") is not True
    ):
        raise ValueError("A/B replay report is not the sealed exact source assembly")
    parent = value.get("parent")
    if not isinstance(parent, dict) or set(parent) != {"fold_a", "fold_b"}:
        raise ValueError("A/B replay report lacks both discovery folds")
    output: dict[str, dict[str, list[dict[str, object]]]] = {}
    directions = (("odd", "even"), ("even", "odd"))
    for fold in ("fold_a", "fold_b"):
        members = parent[fold].get("members")
        if not isinstance(members, dict) or set(members) != {
            f"seed_{seed}" for seed in ALLOWED_SEEDS
        }:
            raise ValueError(f"A/B replay report lacks three seeds for {fold}")
        output[fold] = {}
        for seed in ALLOWED_SEEDS:
            member = members[f"seed_{seed}"]
            epochs = member.get("historical_replay_epochs")
            reproduced = member.get("reproduced_replay_epochs")
            if epochs != reproduced or not isinstance(epochs, list) or len(epochs) != 2:
                raise ValueError(
                    f"Historical/reproduced replays differ for {fold} seed {seed}"
                )
            replays = []
            for (selection, evaluation), pair in zip(directions, epochs, strict=True):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f"Malformed replay pair for {fold} seed {seed}")
                selected, stopped = (int(value) for value in pair)
                if not 1 <= selected <= stopped <= 20:
                    raise ValueError(f"Invalid replay epoch for {fold} seed {seed}")
                replays.append(
                    {
                        "selection_parity": selection,
                        "evaluation_parity": evaluation,
                        "selected_epoch": selected,
                        "stopped_epoch": stopped,
                        "source": "validated_exact_historical_replay",
                    }
                )
            output[fold][f"seed_{seed}"] = replays
    return output


def _parent_paths(
    *,
    fold: str,
    seed: int,
    parent_reference_campaign: Path,
    parent_checkpoint_campaign: Path,
    fold_c_parent: Path,
) -> tuple[Path, Path]:
    if fold == "fold_c":
        run = fold_c_parent / "fold_c" / f"seed_{seed}"
        return run, run
    return (
        parent_reference_campaign / fold / f"seed_{seed}",
        parent_checkpoint_campaign / fold / f"seed_{seed}",
    )


def evaluate_ablation_sets(
    *,
    store: Path,
    definitions: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]],
    parent_reference_campaign: Path,
    parent_checkpoint_campaign: Path,
    parent_ab_replay_report: Path,
    fold_c_parent: Path,
    fold_c_parent_replay_report: Path,
    folds: Sequence[str] = FOLDS,
) -> dict[str, dict[str, object]]:
    if not definitions:
        return {}
    sample_index = load_sample_index(store)
    frozen_c = _fold_c_replays(fold_c_parent_replay_report)
    frozen_ab = _fold_ab_replays(parent_ab_replay_report)
    model = build_model().cuda()
    compiled = compile_model(model)
    results: dict[str, dict[str, object]] = {name: {} for name in definitions}
    if not folds or any(fold not in FOLDS for fold in folds):
        raise ValueError("Ablation folds must be a nonempty subset of C/A/B")
    for fold in folds:
        _, selection_rows, _ = select_training_window(sample_index, fold)
        parent_members: dict[str, EvaluationObservations] = {}
        ablated_members: dict[str, dict[str, EvaluationObservations]] = {
            name: {} for name in definitions
        }
        for seed in ALLOWED_SEEDS:
            reference_run, checkpoint_run = _parent_paths(
                fold=fold,
                seed=seed,
                parent_reference_campaign=parent_reference_campaign,
                parent_checkpoint_campaign=parent_checkpoint_campaign,
                fold_c_parent=fold_c_parent,
            )
            parent, directions = crossfit_patience_observations(
                reference_run,
                (
                    frozen_c[f"seed_{seed}"]
                    if fold == "fold_c"
                    else frozen_ab[fold][f"seed_{seed}"]
                ),
            )
            parent_members[f"seed_{seed}"] = parent
            per_ablation = {
                name: np.zeros_like(parent.predictions) for name in definitions
            }
            parities = _date_parities(parent.date_idx)
            for direction in directions:
                evaluation_parity = str(direction["evaluation_parity"])
                evaluation_dates = parities[evaluation_parity]
                rows = selection_rows.filter(
                    selection_rows.get_column("date_idx").is_in(evaluation_dates)
                )
                loader = create_evaluation_loader(store, rows, GH200_RUNTIME, seed)
                checkpoint = load_checkpoint(
                    checkpoint_run, int(direction["selected_epoch"])
                )
                model.load_state_dict(checkpoint["model_state_dict"])
                reference, predictions = collect_equity_input_ablation_predictions(
                    compiled,
                    loader,
                    definitions,
                )
                expected = np.isin(parent.date_idx, evaluation_dates)
                assert_observations_aligned(
                    _slice_observations(parent, expected), reference
                )
                for name, values in predictions.items():
                    per_ablation[name][expected] = values
            for name, predictions in per_ablation.items():
                ablated_members[name][f"seed_{seed}"] = replace(
                    parent, predictions=predictions
                )
        parent_ensemble = _ensemble(parent_members)
        for name in definitions:
            results[name][fold] = _comparison(
                parent_ensemble, _ensemble(ablated_members[name])
            )
    return results


def _definition(members: Sequence[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    dynamic = sorted(
        int(FEATURE_BY_KEY[key]["index"])
        for key in members
        if FEATURE_BY_KEY[key]["kind"] == "dynamic"
    )
    slow = sorted(
        int(FEATURE_BY_KEY[key]["index"])
        for key in members
        if FEATURE_BY_KEY[key]["kind"] == "slow"
    )
    return tuple(dynamic), tuple(slow)


def _mean_drop(folds: Mapping[str, Mapping[str, object]]) -> float:
    return float(np.mean([folds[fold]["parent_minus_ablated_ic"] for fold in FOLDS]))


def _preview_passes(folds: Mapping[str, Mapping[str, object]]) -> bool:
    values = [float(folds[fold]["parent_minus_ablated_ic"]) for fold in FOLDS]
    return float(np.mean(values)) <= PREVIEW_MEAN_LIMIT and all(
        value <= PREVIEW_FOLD_LIMIT for value in values
    )


def _zero_preview() -> dict[str, object]:
    folds = {
        fold: {
            "parent_minus_ablated_ic": 0.0,
            "per_horizon_parent_minus_ablated_ic": {
                str(value): 0.0 for value in HORIZONS
            },
            "date_count": 0,
            "block10_interval": {},
        }
        for fold in FOLDS
    }
    return {"folds": folds, "mean_drop": 0.0, "gate_passed": True}


def _historical_single_fold(value: Mapping[str, object]) -> dict[str, object]:
    bootstrap = value.get("moving_block_bootstrap")
    if not isinstance(bootstrap, dict) or not isinstance(bootstrap.get("10"), dict):
        raise ValueError("Historical P0.3 fold lacks its block-10 interval")
    return {
        "parent_minus_ablated_ic": float(value["parent_minus_zeroed_ic"]),
        "per_horizon_parent_minus_ablated_ic": {
            str(key): float(item)
            for key, item in value["per_horizon_parent_minus_zeroed_ic"].items()
        },
        "date_count": int(value["date_count"]),
        "block10_interval": bootstrap["10"],
        "source": "imported_unchanged_from_experiment_39_p0_3",
    }


def run_stage_b(
    *,
    store: Path,
    cluster_table: Path,
    p0_attribution_report: Path,
    parent_reference_campaign: Path,
    parent_checkpoint_campaign: Path,
    parent_ab_replay_report: Path,
    fold_c_parent: Path,
    fold_c_parent_replay_report: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    stage_a = _read_json(cluster_table)
    p0 = _read_json(p0_attribution_report)
    historical = {
        _feature_key(str(row["kind"]), int(row["index"])): row
        for row in p0["ranked_features"]
    }
    single_definitions = {key: _definition([key]) for key in FEATURE_BY_KEY}
    fold_c_singles = evaluate_ablation_sets(
        store=store,
        definitions=single_definitions,
        parent_reference_campaign=parent_reference_campaign,
        parent_checkpoint_campaign=parent_checkpoint_campaign,
        parent_ab_replay_report=parent_ab_replay_report,
        fold_c_parent=fold_c_parent,
        fold_c_parent_replay_report=fold_c_parent_replay_report,
        folds=("fold_c",),
    )
    singles: dict[str, dict[str, object]] = {}
    for key in FEATURE_BY_KEY:
        folds = {
            "fold_a": _historical_single_fold(historical[key]["folds"]["fold_a"]),
            "fold_b": _historical_single_fold(historical[key]["folds"]["fold_b"]),
            "fold_c": fold_c_singles[key]["fold_c"],
        }
        singles[key] = {
            "field": FEATURE_BY_KEY[key],
            "folds": folds,
            "three_fold_mean_drop": _mean_drop(folds),
            "p0_ab_classification": historical[key]["classification"],
        }
    sets = [
        {**row, "set_kind": "correlation_cluster"}
        for row in stage_a["clusters"]
        if len(row["members"]) > 1
    ]
    sets.extend(
        {**row, "set_kind": "semantic_group"} for row in stage_a["semantic_groups"]
    )
    signature_to_eval: dict[tuple[str, ...], str] = {}
    group_definitions = {}
    for row in sets:
        signature = tuple(sorted(row["members"]))
        evaluation = signature_to_eval.setdefault(
            signature, f"group_signature_{len(signature_to_eval):02d}"
        )
        row["evaluation_id"] = evaluation
        group_definitions[evaluation] = _definition(signature)
    group_evaluations = evaluate_ablation_sets(
        store=store,
        definitions=group_definitions,
        parent_reference_campaign=parent_reference_campaign,
        parent_checkpoint_campaign=parent_checkpoint_campaign,
        parent_ab_replay_report=parent_ab_replay_report,
        fold_c_parent=fold_c_parent,
        fold_c_parent_replay_report=fold_c_parent_replay_report,
    )
    representative_definitions = {}
    groups: dict[str, dict[str, object]] = {}
    for row in sets:
        folds = group_evaluations[row["evaluation_id"]]
        drops = [float(folds[fold]["parent_minus_ablated_ic"]) for fold in FOLDS]
        group_dead = (
            float(np.mean(drops)) <= 0 and sum(value > 0 for value in drops) <= 1
        )
        representative = min(
            row["members"],
            key=lambda key: (
                -float(singles[key]["three_fold_mean_drop"]),
                float(FEATURE_BY_KEY[key]["model_input_missing_fraction"]),
                len(str(FEATURE_BY_KEY[key]["name"])),
                int(FEATURE_BY_KEY[key]["global_index"]),
            ),
        )
        materially_positive = float(np.mean(drops)) >= REPRESENTATIVE_TEST_THRESHOLD
        if materially_positive and not group_dead:
            remaining = [key for key in row["members"] if key != representative]
            representative_definitions[f"representative_{row['set_id']}"] = _definition(
                remaining
            )
        groups[row["set_id"]] = {
            **row,
            "joint_folds": folds,
            "joint_mean_drop": float(np.mean(drops)),
            "group_dead": group_dead,
            "representative": representative,
            "representative_test_required": materially_positive and not group_dead,
        }
    representative_evaluations = evaluate_ablation_sets(
        store=store,
        definitions=representative_definitions,
        parent_reference_campaign=parent_reference_campaign,
        parent_checkpoint_campaign=parent_checkpoint_campaign,
        parent_ab_replay_report=parent_ab_replay_report,
        fold_c_parent=fold_c_parent,
        fold_c_parent_replay_report=fold_c_parent_replay_report,
    )
    for set_id, row in groups.items():
        if row["group_dead"]:
            row["classification"] = "group_dead"
        elif row["representative_test_required"]:
            folds = representative_evaluations[f"representative_{set_id}"]
            residual = _mean_drop(folds)
            row["minus_representative_folds"] = folds
            row["minus_representative_mean_drop"] = residual
            row["classification"] = (
                "representative_sufficient"
                if residual <= REPRESENTATIVE_TEST_THRESHOLD
                else "keep_all"
            )
        else:
            row["classification"] = "keep_all"
    singleton_keys = {
        row["members"][0] for row in stage_a["clusters"] if len(row["members"]) == 1
    }
    r1 = {
        key
        for row in groups.values()
        if row["classification"] == "group_dead"
        for key in row["members"]
    }
    r1.update(
        key
        for key in singleton_keys
        if singles[key]["p0_ab_classification"] == "dead"
        and float(singles[key]["folds"]["fold_c"]["parent_minus_ablated_ic"]) <= 0
    )
    r2_extras = {
        key
        for row in groups.values()
        if row["classification"] == "representative_sufficient"
        for key in row["members"]
        if key != row["representative"]
    }
    proposal_reasons: dict[str, list[str]] = {key: [] for key in FEATURE_BY_KEY}
    for row in groups.values():
        if row["classification"] == "group_dead":
            for key in row["members"]:
                proposal_reasons[key].append(f"group_dead:{row['set_id']}")
        elif row["classification"] == "representative_sufficient":
            for key in row["members"]:
                if key != row["representative"]:
                    proposal_reasons[key].append(f"non_representative:{row['set_id']}")
    for key in singleton_keys:
        if (
            singles[key]["p0_ab_classification"] == "dead"
            and float(singles[key]["folds"]["fold_c"]["parent_minus_ablated_ic"]) <= 0
        ):
            proposal_reasons[key].append("dead_singleton_all_three_folds")

    preview_history: list[dict[str, object]] = []

    def preview(name: str, members: set[str]) -> dict[str, object]:
        if not members:
            result = _zero_preview()
        else:
            folds = evaluate_ablation_sets(
                store=store,
                definitions={name: _definition(sorted(members))},
                parent_reference_campaign=parent_reference_campaign,
                parent_checkpoint_campaign=parent_checkpoint_campaign,
                parent_ab_replay_report=parent_ab_replay_report,
                fold_c_parent=fold_c_parent,
                fold_c_parent_replay_report=fold_c_parent_replay_report,
            )[name]
            result = {
                "folds": folds,
                "mean_drop": _mean_drop(folds),
                "gate_passed": _preview_passes(folds),
            }
        preview_history.append({"name": name, "members": sorted(members), **result})
        return result

    r1_current = set(r1)
    r1_preview = preview("r1_initial", r1_current)
    r1_final_preview = r1_preview
    if not r1_preview["gate_passed"]:
        r1_current = set()
        r1_final_preview = _zero_preview()
        for key in sorted(
            r1, key=lambda value: float(singles[value]["three_fold_mean_drop"])
        ):
            trial = r1_current | {key}
            r1_preview = preview(f"r1_walkback_add_{key}", trial)
            if not r1_preview["gate_passed"]:
                break
            r1_current = trial
            r1_final_preview = r1_preview
    r2_current = r1_current | r2_extras
    r2_preview = preview("r2_initial", r2_current)
    r2_final_preview = r2_preview
    if not r2_preview["gate_passed"]:
        r2_current = set(r1_current)
        r2_final_preview = r1_final_preview
        for key in sorted(
            r2_extras - r1_current,
            key=lambda value: float(singles[value]["three_fold_mean_drop"]),
        ):
            trial = r2_current | {key}
            r2_preview = preview(f"r2_walkback_add_{key}", trial)
            if not r2_preview["gate_passed"]:
                break
            r2_current = trial
            r2_final_preview = r2_preview
    if not r1_current.issubset(r2_current):
        raise AssertionError("Walked-back R1/R2 sets are not nested")
    frozen = {
        "schema": "FEATURE_REMOVAL_FROZEN_SETS_V1",
        "created_at": _now(),
        "cluster_table_sha256": _sha256(cluster_table),
        "initial_r1": sorted(r1),
        "initial_r2": sorted(r1 | r2_extras),
        "r1": sorted(r1_current),
        "r2": sorted(r2_current),
        "r1_definition": _definition(sorted(r1_current)),
        "r2_definition": _definition(sorted(r2_current)),
        "r1_final_preview": r1_final_preview,
        "r2_final_preview": r2_final_preview,
        "proposal_reasons": {
            key: reasons for key, reasons in proposal_reasons.items() if reasons
        },
        "preview_history": preview_history,
        "preview_gate": {
            "mean_cost_max": PREVIEW_MEAN_LIMIT,
            "single_fold_cost_max": PREVIEW_FOLD_LIMIT,
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    frozen_path = output_dir / "frozen_removal_sets.json"
    _atomic_json(frozen_path, frozen)
    report = {
        "schema": "FEATURE_REMOVAL_STAGE_B_V1",
        "created_at": _now(),
        "cluster_table": str(cluster_table.resolve()),
        "cluster_table_sha256": _sha256(cluster_table),
        "p0_attribution_report": str(p0_attribution_report.resolve()),
        "p0_attribution_sha256": _sha256(p0_attribution_report),
        "parent_ab_replay_report": str(parent_ab_replay_report.resolve()),
        "parent_ab_replay_report_sha256": _sha256(parent_ab_replay_report),
        "singles": singles,
        "groups": groups,
        "frozen_sets": str(frozen_path.resolve()),
        "frozen_sets_sha256": _sha256(frozen_path),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    report_path = output_dir / "stage_b_report.json"
    _atomic_json(report_path, report)
    return frozen_path


def _run_stage_b_isolated(**kwargs: object) -> Path:
    """Release compiled Stage-B CUDA state before Stage-C workers start."""
    with ProcessPoolExecutor(
        max_workers=1, mp_context=mp.get_context("spawn")
    ) as executor:
        return executor.submit(run_stage_b, **kwargs).result()


def _run_training_job(
    store: Path,
    run_dir: Path,
    seed: int,
    fold: str,
    dynamic: tuple[int, ...],
    slow: tuple[int, ...],
) -> str:
    run_training(
        store=store,
        seed=seed,
        selection_window=fold,
        run_dir=run_dir,
        zero_dynamic_channels=dynamic,
        zero_slow_fields=slow,
    )
    return str(run_dir)


def _candidate_run(output_dir: Path, candidate: str, fold: str, seed: int) -> Path:
    return output_dir / "candidates" / candidate / fold / f"seed_{seed}"


def _readout_members(
    paths: Mapping[str, Path], readout: str
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for name, path in paths.items():
        if readout == "patience3_raw":
            observations, replay = crossfit_patience_observations(path)
        elif readout == "final_ema_0995":
            reference = load_run_observations(path, "final_raw")
            observations = replace(
                reference, predictions=predictions_for_rule(path, readout)
            )
            replay = []
        else:
            raise ValueError(f"Unknown readout: {readout}")
        members[name] = observations
        replays[name] = replay
    return members, replays


def _parent_members(
    *,
    fold: str,
    readout: str,
    parent_reference_campaign: Path,
    fold_c_parent: Path,
    frozen_c: Mapping[str, list[dict[str, object]]],
    frozen_ab: Mapping[str, Mapping[str, list[dict[str, object]]]],
) -> tuple[dict[str, EvaluationObservations], dict[str, object]]:
    members = {}
    replays = {}
    for seed in ALLOWED_SEEDS:
        path = (
            fold_c_parent / "fold_c" / f"seed_{seed}"
            if fold == "fold_c"
            else parent_reference_campaign / fold / f"seed_{seed}"
        )
        if readout == "patience3_raw":
            observations, replay = crossfit_patience_observations(
                path,
                (
                    frozen_c[f"seed_{seed}"]
                    if fold == "fold_c"
                    else frozen_ab[fold][f"seed_{seed}"]
                ),
            )
        else:
            reference = load_run_observations(path, "final_raw")
            observations = replace(
                reference, predictions=predictions_for_rule(path, "final_ema_0995")
            )
            replay = []
        members[f"seed_{seed}"] = observations
        replays[f"seed_{seed}"] = replay
    return members, replays


def _extract_analysis(path: Path) -> dict[str, object]:
    value = _read_json(path / "analysis.json")
    return {
        "candidate_ensemble_ic": value["candidate"]["ensemble_ic"],
        "parent_ensemble_ic": value["parent"]["ensemble_ic"],
        "candidate_minus_parent_ic": value["candidate_minus_parent_primary_ic"],
        "per_date_delta_bootstrap": value["per_date_delta_bootstrap"],
        "horizon_guardrails": value["horizon_guardrails"],
        "time_of_day_guardrails": value["time_of_day_guardrails"],
        "analysis": str((path / "analysis.json").resolve()),
    }


def run_stage_c(
    *,
    store: Path,
    frozen_sets: Path,
    parent_reference_campaign: Path,
    parent_ab_replay_report: Path,
    fold_c_parent: Path,
    fold_c_parent_replay_report: Path,
    output_dir: Path,
    parallel_processes: int = 2,
) -> Path:
    if parallel_processes != 2:
        raise ValueError("Stage C is frozen to exactly two GH200 processes")
    output_dir.mkdir(parents=True, exist_ok=False)
    frozen = _read_json(frozen_sets)
    definitions = {
        "prune_r1": tuple(
            tuple(int(value) for value in part) for part in frozen["r1_definition"]
        ),
        "prune_r2": tuple(
            tuple(int(value) for value in part) for part in frozen["r2_definition"]
        ),
    }
    jobs = [
        (
            store,
            _candidate_run(output_dir, candidate, fold, seed),
            seed,
            fold,
            definition[0],
            definition[1],
        )
        for candidate, definition in definitions.items()
        for fold in FOLDS
        for seed in ALLOWED_SEEDS
    ]
    with ProcessPoolExecutor(
        max_workers=parallel_processes, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_run_training_job, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    frozen_c = _fold_c_replays(fold_c_parent_replay_report)
    frozen_ab = _fold_ab_replays(parent_ab_replay_report)
    candidates: dict[str, object] = {}
    for candidate in definitions:
        readouts: dict[str, object] = {}
        for readout in ("patience3_raw", "final_ema_0995"):
            folds: dict[str, object] = {}
            for fold in FOLDS:
                candidate_members, candidate_replays = _readout_members(
                    {
                        f"seed_{seed}": _candidate_run(
                            output_dir, candidate, fold, seed
                        )
                        for seed in ALLOWED_SEEDS
                    },
                    readout,
                )
                parent_members, parent_replays = _parent_members(
                    fold=fold,
                    readout=readout,
                    parent_reference_campaign=parent_reference_campaign,
                    fold_c_parent=fold_c_parent,
                    frozen_c=frozen_c,
                    frozen_ab=frozen_ab,
                )
                standalone = compare_observation_ensembles(
                    candidate_members,
                    parent_members,
                    candidate_rule=readout,
                    parent_rule=readout,
                    output_dir=output_dir
                    / "analysis"
                    / candidate
                    / readout
                    / fold
                    / "standalone",
                    comparison_metadata={
                        "fold": fold,
                        "replacement_decision": readout == "patience3_raw",
                        "candidate_patience_replays": candidate_replays,
                        "parent_patience_replays": parent_replays,
                        "official_validation_accessed": False,
                        "test_accessed": False,
                    },
                )
                ensemble = compare_observation_ensembles(
                    {
                        **{
                            f"parent_{key}": value
                            for key, value in parent_members.items()
                        },
                        **{
                            f"candidate_{key}": value
                            for key, value in candidate_members.items()
                        },
                    },
                    parent_members,
                    candidate_rule=f"parent_plus_{candidate}_{readout}_uniform_6",
                    parent_rule=readout,
                    output_dir=output_dir
                    / "analysis"
                    / candidate
                    / readout
                    / fold
                    / "parent_plus_candidate",
                    comparison_metadata={
                        "fold": fold,
                        "informational_only": True,
                        "official_validation_accessed": False,
                        "test_accessed": False,
                    },
                )
                folds[fold] = {
                    "standalone": _extract_analysis(standalone),
                    "parent_plus_candidate": _extract_analysis(ensemble),
                }
            readouts[readout] = folds
        primary = readouts["patience3_raw"]
        deltas = [
            float(primary[fold]["standalone"]["candidate_minus_parent_ic"])
            for fold in FOLDS
        ]
        candidates[candidate] = {
            "removed_fields": frozen["r1" if candidate == "prune_r1" else "r2"],
            "readouts": readouts,
            "primary_fold_deltas": dict(zip(FOLDS, deltas, strict=True)),
            "primary_mean_delta": float(np.mean(deltas)),
            "noninferior": float(np.mean(deltas)) >= 0
            and all(value >= NONINFERIOR_FOLD_FLOOR for value in deltas),
            "improvement": float(np.mean(deltas)) >= IMPROVEMENT_MEAN
            and all(value >= 0 for value in deltas),
        }
    r1, r2 = candidates["prune_r1"], candidates["prune_r2"]
    winner = None
    if r1["noninferior"] and r2["noninferior"]:
        winner = (
            "prune_r2"
            if float(r2["primary_mean_delta"])
            >= float(r1["primary_mean_delta"]) - R2_TIE_TOLERANCE
            else "prune_r1"
        )
    elif r1["noninferior"]:
        winner = "prune_r1"
    elif r2["noninferior"]:
        winner = "prune_r2"
    removed = set() if winner is None else set(candidates[winner]["removed_fields"])
    verdicts = [
        {
            **row,
            "verdict": "remove" if row["key"] in removed else "keep",
            "stage_b_proposal_reasons": frozen["proposal_reasons"].get(row["key"], []),
            "rule": (
                "selected_retrained_noninferior_removal_set"
                if row["key"] in removed
                else (
                    "proposed_but_not_in_selected_retrained_removal_set"
                    if frozen["proposal_reasons"].get(row["key"])
                    else "never_qualified_as_removal_candidate"
                )
            ),
        }
        for row in FEATURES
    ]
    summary = {
        "schema": "FEATURE_REMOVAL_STAGE_C_SUMMARY_V1",
        "created_at": _now(),
        "frozen_sets": str(frozen_sets.resolve()),
        "frozen_sets_sha256": _sha256(frozen_sets),
        "candidates": candidates,
        "winner": winner,
        "outcome": (
            "store_v2_feature_specification_selected"
            if winner is not None
            else "redundant_but_load_bearing_under_retraining_keep_all"
        ),
        "field_verdicts": verdicts,
        "canonical_recipe_changed": False,
        "official_read_lineup_changed": False,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    summary_path = output_dir / "stage_c_summary.json"
    _atomic_json(summary_path, summary)
    _atomic_json(
        output_dir / "store_v2_feature_specification.json",
        {
            "schema": "STORE_V2_FEATURE_SPECIFICATION_V1",
            "created_at": _now(),
            "status": "selected" if winner is not None else "not_selected",
            "winner": winner,
            "removed_fields": sorted(removed),
            "retained_fields": sorted(set(FEATURE_BY_KEY) - removed),
            "source_summary_sha256": _sha256(summary_path),
            "store_rebuild_performed": False,
            "canonical_recipe_changed": False,
        },
    )
    return summary_path


def run_program(
    *,
    store: Path,
    p0_attribution_report: Path,
    parent_reference_campaign: Path,
    parent_checkpoint_campaign: Path,
    parent_ab_replay_report: Path,
    fold_c_parent: Path,
    fold_c_parent_replay_report: Path,
    output_dir: Path,
    parallel_processes: int = 2,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "program_manifest.json"
    manifest: dict[str, object] = {
        "schema": "FEATURE_REMOVAL_PROGRAM_V1",
        "status": "running",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "feature_store": str(store.resolve()),
        "feature_store_identity": feature_store_identity(store),
        "stages": {name: {"status": "pending"} for name in ("a", "b", "c")},
        "candidate_order": ["prune_r1", "prune_r2"],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, manifest)
    try:
        manifest["stages"]["a"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        cluster_table = build_stage_a(store=store, output_dir=output_dir / "stage_a")
        manifest["stages"]["a"] = {
            "status": "completed",
            "cluster_table": str(cluster_table.resolve()),
            "sha256": _sha256(cluster_table),
        }
        _atomic_json(manifest_path, manifest)
        manifest["stages"]["b"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        frozen = _run_stage_b_isolated(
            store=store,
            cluster_table=cluster_table,
            p0_attribution_report=p0_attribution_report,
            parent_reference_campaign=parent_reference_campaign,
            parent_checkpoint_campaign=parent_checkpoint_campaign,
            parent_ab_replay_report=parent_ab_replay_report,
            fold_c_parent=fold_c_parent,
            fold_c_parent_replay_report=fold_c_parent_replay_report,
            output_dir=output_dir / "stage_b",
        )
        manifest["stages"]["b"] = {
            "status": "completed",
            "frozen_sets": str(frozen.resolve()),
            "sha256": _sha256(frozen),
        }
        _atomic_json(manifest_path, manifest)
        manifest["stages"]["c"] = {"status": "running"}
        _atomic_json(manifest_path, manifest)
        stage_c = run_stage_c(
            store=store,
            frozen_sets=frozen,
            parent_reference_campaign=parent_reference_campaign,
            parent_ab_replay_report=parent_ab_replay_report,
            fold_c_parent=fold_c_parent,
            fold_c_parent_replay_report=fold_c_parent_replay_report,
            output_dir=output_dir / "stage_c",
            parallel_processes=parallel_processes,
        )
        manifest["stages"]["c"] = {
            "status": "completed",
            "summary": str(stage_c.resolve()),
            "sha256": _sha256(stage_c),
        }
        manifest.update({"status": "completed", "completed_at": _now()})
        _atomic_json(manifest_path, manifest)
        inventory = _write_artifact_inventory(output_dir)
        print(
            f"artifact_inventory={inventory} sha256={_sha256(inventory)}",
            flush=True,
        )
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": _now(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered incumbent feature-removal program"
    )
    for name in (
        "store",
        "p0_attribution_report",
        "parent_reference_campaign",
        "parent_checkpoint_campaign",
        "parent_ab_replay_report",
        "fold_c_parent",
        "fold_c_parent_replay_report",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--parallel-processes", type=int, default=2)
    print(run_program(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
