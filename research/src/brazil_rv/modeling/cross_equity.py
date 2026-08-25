from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brazil_rv.preprocessing.contract import EQUITY_SLOW_CHANNELS

from .contract import ALLOWED_SEEDS, HORIZONS
from .engine import EvaluationObservations, assert_observations_aligned
from .metrics import (
    per_date_primary_ic,
    primary_validation_score,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .three_fold_sidecar_screen import FOLDS, crossfit_patience_observations

LAMBDAS = (0.25, 0.5, 0.75, 1.0)
GROUPING_ORDER = ("clusters", "liquidity", "beta")
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCK = 10
BOOTSTRAP_SEED = 46


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parent_run(
    parent_campaign: Path, fold_c_parent: Path, fold: str, seed: int
) -> Path:
    if fold == "fold_c":
        return fold_c_parent / "fold_c" / f"seed_{seed}"
    return parent_campaign / fold / f"seed_{seed}"


def _load_parent_ensemble(
    *,
    fold: str,
    parent_campaign: Path,
    fold_c_parent: Path,
    fold_c_replays: dict[str, list[dict[str, object]]],
) -> EvaluationObservations:
    members = []
    for seed in ALLOWED_SEEDS:
        run = _parent_run(parent_campaign, fold_c_parent, fold, seed)
        observations, _ = crossfit_patience_observations(
            run,
            fold_c_replays[f"seed_{seed}"] if fold == "fold_c" else None,
        )
        members.append(observations)
    for candidate in members[1:]:
        assert_observations_aligned(members[0], candidate)
    predictions = rank_average_predictions(
        [member.predictions for member in members], members[0].label_mask
    )
    return replace(members[0], predictions=predictions)


def _expand_monthly(
    values: np.ndarray, month_dates: np.ndarray, date_count: int
) -> np.ndarray:
    output = np.full((date_count, *values.shape[1:]), -1, dtype=values.dtype)
    for day in range(date_count):
        month = int(np.searchsorted(month_dates, day, side="right") - 1)
        if month >= 0:
            output[day] = values[month]
    return output


def _liquidity_terciles(
    slow: np.ndarray,
    active: np.ndarray,
    month_dates: np.ndarray,
) -> np.ndarray:
    adv_index = EQUITY_SLOW_CHANNELS.index("median_daily_dollar_volume_20d_log_scale")
    monthly = np.full((len(month_dates), slow.shape[1]), -1, dtype=np.int16)
    for month, day in enumerate(month_dates):
        valid = np.asarray(active[day])
        slots = np.flatnonzero(valid)
        ordered = slots[np.argsort(slow[day, slots, adv_index], kind="stable")]
        for label, group in enumerate(np.array_split(ordered, 3)):
            monthly[month, group] = label
    return _expand_monthly(monthly, month_dates, slow.shape[0])


def _discrete_components(
    scores: np.ndarray,
    mask: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    component = np.zeros_like(scores, dtype=np.float32)
    for sample in range(scores.shape[0]):
        sample_groups = groups[sample]
        for horizon in range(scores.shape[2]):
            valid = mask[sample, :, horizon] & (sample_groups >= 0)
            for group in np.unique(sample_groups[valid]):
                selected = valid & (sample_groups == group)
                component[sample, selected, horizon] = np.mean(
                    scores[sample, selected, horizon], dtype=np.float64
                )
    component[~mask] = 0
    within = np.where(mask, scores - component, 0).astype(np.float32)
    return component, within


def _beta_components(
    scores: np.ndarray,
    mask: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    component = np.zeros_like(scores, dtype=np.float32)
    for sample in range(scores.shape[0]):
        exposure = beta[sample]
        for horizon in range(scores.shape[2]):
            valid = mask[sample, :, horizon] & np.isfinite(exposure)
            if valid.sum() < 3 or np.std(exposure[valid]) == 0:
                continue
            design = np.column_stack((np.ones(valid.sum()), exposure[valid]))
            coefficients = np.linalg.lstsq(
                design, scores[sample, valid, horizon], rcond=None
            )[0]
            component[sample, valid, horizon] = design @ coefficients
    component[~mask] = 0
    within = np.where(mask, scores - component, 0).astype(np.float32)
    return component, within


def _daily_ic(
    observations: EvaluationObservations, predictions: np.ndarray
) -> np.ndarray:
    sample_ic = sample_level_spearman_ic(
        predictions.astype(np.float32), observations.targets, observations.label_mask
    )
    _, daily = per_date_primary_ic(sample_ic, observations.date_idx)
    return daily


def _score(observations: EvaluationObservations, predictions: np.ndarray) -> float:
    return primary_validation_score(
        predictions.astype(np.float32),
        observations.targets,
        observations.label_mask,
        observations.date_idx,
    )


def _moving_block_indices(
    length: int, replications: int, block: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    block_count = math.ceil(length / block)
    starts = generator.integers(0, length - block + 1, size=(replications, block_count))
    return (starts[..., None] + np.arange(block)).reshape(replications, -1)[:, :length]


def _interval(values: np.ndarray, *, seed: int) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size < BOOTSTRAP_BLOCK:
        raise ValueError("Block bootstrap has too few dates")
    sampled = finite[
        _moving_block_indices(
            finite.size, BOOTSTRAP_REPLICATIONS, BOOTSTRAP_BLOCK, seed
        )
    ]
    means = sampled.mean(axis=1)
    return {
        "estimate": float(finite.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _stability_interval(
    parent: np.ndarray, candidate: np.ndarray, *, seed: int
) -> dict[str, float]:
    valid = np.isfinite(parent) & np.isfinite(candidate)
    parent, candidate = parent[valid], candidate[valid]
    indices = _moving_block_indices(
        parent.size, BOOTSTRAP_REPLICATIONS, BOOTSTRAP_BLOCK, seed
    )
    values = np.std(parent[indices], axis=1, ddof=1) - np.std(
        candidate[indices], axis=1, ddof=1
    )
    return {
        "estimate": float(np.std(parent, ddof=1) - np.std(candidate, ddof=1)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _horizon_scores(
    observations: EvaluationObservations, predictions: np.ndarray
) -> list[float]:
    rows = []
    for horizon in range(len(HORIZONS)):
        mask = np.zeros_like(observations.label_mask)
        mask[..., horizon] = observations.label_mask[..., horizon]
        rows.append(
            primary_validation_score(
                predictions.astype(np.float32),
                observations.targets,
                mask,
                observations.date_idx,
            )
        )
    return rows


def _variance_share(
    scores: np.ndarray, component: np.ndarray, mask: np.ndarray
) -> list[float | None]:
    rows = []
    for horizon in range(scores.shape[2]):
        ratios = []
        for sample in range(scores.shape[0]):
            valid = mask[sample, :, horizon]
            if valid.sum() >= 3 and np.var(scores[sample, valid, horizon]) > 0:
                ratios.append(
                    np.var(component[sample, valid, horizon])
                    / np.var(scores[sample, valid, horizon])
                )
        rows.append(float(np.mean(ratios)) if ratios else None)
    return rows


def _small_group_ic(
    observations: EvaluationObservations,
    predictions: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float | None]:
    result = {}
    for group in np.unique(groups[groups >= 0]):
        values = []
        for sample in range(predictions.shape[0]):
            for horizon in range(predictions.shape[2]):
                valid = observations.label_mask[sample, :, horizon] & (
                    groups[sample] == group
                )
                if valid.sum() < 3:
                    continue
                left = predictions[sample, valid, horizon]
                right = observations.targets[sample, valid, horizon]
                left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable")
                right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable")
                values.append(float(np.corrcoef(left_rank, right_rank)[0, 1]))
        result[str(int(group))] = float(np.mean(values)) if values else None
    return result


def _fold_analysis(
    observations: EvaluationObservations,
    groups_by_date: dict[str, np.ndarray],
    beta_by_date: np.ndarray,
    adv_by_date: np.ndarray,
) -> dict[str, object]:
    parent = observations.predictions
    parent_daily = _daily_ic(observations, parent)
    decomposition = {}
    sweep = {}
    for grouping in GROUPING_ORDER:
        if grouping == "beta":
            component, within = _beta_components(
                parent, observations.label_mask, beta_by_date[observations.date_idx]
            )
            grouping_values = beta_by_date[observations.date_idx]
        else:
            grouping_values = groups_by_date[grouping][observations.date_idx]
            component, within = _discrete_components(
                parent, observations.label_mask, grouping_values
            )
        decomposition[grouping] = {
            "cross_group_variance_share_by_horizon": _variance_share(
                parent, component, observations.label_mask
            ),
            "cross_group_ic_by_horizon": _horizon_scores(observations, component),
            "within_group_ic_by_horizon": _horizon_scores(observations, within),
            "per_group_parent_ic": (
                None
                if grouping == "beta"
                else _small_group_ic(observations, parent, grouping_values)
            ),
        }
        if grouping == "liquidity":
            adv = adv_by_date[observations.date_idx]
            adv_predictions = np.repeat(adv[..., None], parent.shape[2], axis=2)
            decomposition[grouping]["score_vs_adv_rank_ic"] = primary_validation_score(
                parent,
                adv_predictions.astype(np.float32),
                observations.label_mask & np.isfinite(adv_predictions),
                observations.date_idx,
            )
        for value in LAMBDAS:
            candidate = np.where(
                observations.label_mask, parent - value * component, 0
            ).astype(np.float32)
            candidate_daily = _daily_ic(observations, candidate)
            delta = candidate_daily - parent_daily
            key = f"{grouping}_lambda_{value:g}"
            sweep[key] = {
                "grouping": grouping,
                "lambda": value,
                "parent_ic": _score(observations, parent),
                "candidate_ic": _score(observations, candidate),
                "paired_delta": float(np.mean(delta)),
                "delta_block10": _interval(delta, seed=BOOTSTRAP_SEED),
                "parent_daily_ic_sd": float(np.std(parent_daily, ddof=1)),
                "candidate_daily_ic_sd": float(np.std(candidate_daily, ddof=1)),
                "stability_improvement": float(
                    np.std(parent_daily, ddof=1) - np.std(candidate_daily, ddof=1)
                ),
                "stability_block10": _stability_interval(
                    parent_daily, candidate_daily, seed=BOOTSTRAP_SEED
                ),
                "daily_parent": parent_daily.tolist(),
                "daily_candidate": candidate_daily.tolist(),
                "per_group_candidate_ic": (
                    None
                    if grouping == "beta"
                    else _small_group_ic(observations, candidate, grouping_values)
                ),
            }
    return {"decomposition": decomposition, "sweep": sweep}


def _rotated_screen(folds: dict[str, dict[str, object]]) -> dict[str, object]:
    selected = {}
    heldout_deltas = []
    heldout_parent = []
    heldout_candidate = []
    for heldout in FOLDS:
        training = [fold for fold in FOLDS if fold != heldout]
        keys = list(folds[heldout]["sweep"])
        eligible = []
        for key in keys:
            rows = [folds[fold]["sweep"][key] for fold in training]
            if np.mean([row["paired_delta"] for row in rows]) >= -0.0005:
                eligible.append(key)
        if not eligible:
            selected[heldout] = {
                "candidate": None,
                "reason": "no training-fold noninferior candidate",
            }
            continue
        key = min(
            eligible,
            key=lambda candidate: (
                -np.mean(
                    [
                        folds[fold]["sweep"][candidate]["stability_improvement"]
                        for fold in training
                    ]
                ),
                -np.mean(
                    [
                        folds[fold]["sweep"][candidate]["paired_delta"]
                        for fold in training
                    ]
                ),
                GROUPING_ORDER.index(folds[heldout]["sweep"][candidate]["grouping"]),
                folds[heldout]["sweep"][candidate]["lambda"],
            ),
        )
        evaluation = folds[heldout]["sweep"][key]
        parent = np.asarray(evaluation["daily_parent"], dtype=np.float64)
        candidate = np.asarray(evaluation["daily_candidate"], dtype=np.float64)
        heldout_deltas.append(candidate - parent)
        heldout_parent.append(parent)
        heldout_candidate.append(candidate)
        selected[heldout] = {
            "candidate": key,
            "training_folds": training,
            "heldout_paired_delta": evaluation["paired_delta"],
            "heldout_stability_improvement": evaluation["stability_improvement"],
        }
    if len(heldout_deltas) != len(FOLDS):
        return {
            "rotations": selected,
            "supported": False,
            "reason": "one or more rotations had no candidate",
        }
    deltas = np.concatenate(heldout_deltas)
    parent = np.concatenate(heldout_parent)
    candidate = np.concatenate(heldout_candidate)
    delta_interval = _interval(deltas, seed=BOOTSTRAP_SEED)
    stability = _stability_interval(parent, candidate, seed=BOOTSTRAP_SEED)
    supported = (
        float(np.mean(deltas)) >= -0.0005
        and delta_interval["lower_95"] >= -0.001
        and stability["lower_95"] > 0.0
    )
    return {
        "rotations": selected,
        "heldout_mean_paired_delta": float(np.mean(deltas)),
        "heldout_delta_block10": delta_interval,
        "heldout_stability_block10": stability,
        "supported": supported,
        "support_rule": (
            "mean paired delta >= -0.0005; block10 lower >= -0.001; "
            "stability-improvement block10 lower > 0"
        ),
    }


def run_n0(
    *,
    store: Path,
    graph_dir: Path,
    parent_campaign: Path,
    fold_c_parent: Path,
    fold_c_parent_replay_report: Path,
    output_dir: Path,
) -> Path:
    paths = [
        store,
        graph_dir,
        parent_campaign,
        fold_c_parent,
        fold_c_parent_replay_report,
        output_dir,
    ]
    store, graph_dir, parent_campaign, fold_c_parent, replay_path, output_dir = map(
        Path.resolve, paths
    )
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    replay_report = json.loads(replay_path.read_text(encoding="utf-8"))
    replays = replay_report.get("comparison_metadata", {}).get(
        "parent_patience_replays"
    )
    if not isinstance(replays, dict):
        raise ValueError("Fold-C replay report lacks parent_patience_replays")
    slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    active = np.load(store / "equity_membership.npy", mmap_mode="r") & np.load(
        store / "equity_data_ready.npy", mmap_mode="r"
    )
    with np.load(graph_dir / "monthly_graphs.npz") as graph:
        month_dates = graph["month_date_idx"].copy()
        clusters = _expand_monthly(graph["clusters"], month_dates, slow.shape[0])
    liquidity = _liquidity_terciles(slow, active, month_dates)
    beta_index = EQUITY_SLOW_CHANNELS.index("beta_to_WIN")
    beta = np.asarray(slow[..., beta_index], dtype=np.float32)
    beta[~active] = np.nan
    adv_index = EQUITY_SLOW_CHANNELS.index("median_daily_dollar_volume_20d_log_scale")
    adv = np.asarray(slow[..., adv_index], dtype=np.float32)
    adv[~active] = np.nan
    folds = {}
    for fold in FOLDS:
        observations = _load_parent_ensemble(
            fold=fold,
            parent_campaign=parent_campaign,
            fold_c_parent=fold_c_parent,
            fold_c_replays=replays,
        )
        folds[fold] = _fold_analysis(
            observations,
            {"clusters": clusters, "liquidity": liquidity},
            beta,
            adv,
        )
    rotation = _rotated_screen(folds)
    cluster_nonpositive = sum(
        float(
            np.nanmean(
                folds[fold]["decomposition"]["clusters"]["cross_group_ic_by_horizon"]
            )
        )
        <= 0
        for fold in FOLDS
    )
    result = {
        "schema": "EXPERIMENT46_N0_ANALYSIS_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "store": str(store),
            "graph_manifest_sha256": _sha256(graph_dir / "manifest.json"),
            "fold_c_parent_replay_report": str(replay_path),
            "fold_c_parent_replay_report_sha256": _sha256(replay_path),
        },
        "groupings": {
            "clusters": "primary evidential",
            "liquidity": "evidential",
            "beta": "evidential continuous exposure",
            "sector": {
                "status": "unavailable",
                "reason": "no canonical present-day sector mapping covers the security axis",
                "pit_violation_if_used": True,
                "adoption_evidence": False,
            },
        },
        "folds": folds,
        "rotated_adoption_screen": rotation,
        "deployment_changed": False,
        "future_deployment_transform_arm_registered": bool(rotation["supported"]),
        "t_peer_registration": {
            "condition": "cluster cross-group component IC <= 0 on at least two folds",
            "nonpositive_fold_count": cluster_nonpositive,
            "registered": cluster_nonpositive >= 2,
            "executed": False,
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    result_path = output_dir / "n0_analysis.json"
    _atomic_json(result_path, result)
    _atomic_json(
        output_dir / "manifest.json",
        {
            "schema": "EXPERIMENT46_N0_MANIFEST_V1",
            "analysis": result_path.name,
            "analysis_sha256": _sha256(result_path),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment-46 N0 analysis")
    for name in (
        "store",
        "graph_dir",
        "parent_campaign",
        "fold_c_parent",
        "fold_c_parent_replay_report",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    print(run_n0(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
