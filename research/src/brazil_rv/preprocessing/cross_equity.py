from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.modeling.data import (
    EXTERNAL_SIDECAR_SCHEMA,
    feature_store_axis_identity,
    feature_store_identity,
)
from brazil_rv.modeling.metrics import primary_validation_score

from .contract import (
    DECISION_EQUITY_INDICES,
    DYNAMIC_CHANNELS,
    EQUITY_SLOW_CHANNELS,
    HORIZONS,
)

PEER_FEATURES = (
    "peer_mean_return_15m",
    "peer_mean_return_60m",
    "peer_relative_return_60m",
    "peer_dispersion_60m",
    "peer_breadth_15m",
    "peer_relative_volume_surprise",
    "peer_mean_return_1d",
    "peer_relative_return_1d",
)
GRAPH_LOOKBACK = 126
GRAPH_MIN_OBSERVED = 101
PEER_COUNT = 8
PEER_MINIMUM = 3
PEER_MIN_RHO = 0.15
CLUSTER_COUNT = 12
CLUSTER_MINIMUM = 3
F2_END_DATE = "2023-03-31"
F2_MIN_HALF_ABS_IC = 0.001
F2_MAX_ABS_CORRELATION = 0.80


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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman_matrix(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    eligible = valid.sum(axis=0) >= GRAPH_MIN_OBSERVED
    slots = np.flatnonzero(eligible)
    result = np.full((values.shape[1], values.shape[1]), np.nan, dtype=np.float64)
    np.fill_diagonal(result, 1.0)
    for left_position, left in enumerate(slots):
        for right in slots[left_position + 1 :]:
            common = valid[:, left] & valid[:, right]
            if common.sum() < GRAPH_MIN_OBSERVED:
                continue
            left_rank = _average_ranks(values[common, left])
            right_rank = _average_ranks(values[common, right])
            rho = float(np.corrcoef(left_rank, right_rank)[0, 1])
            result[left, right] = result[right, left] = rho
    return result


def average_linkage_clusters(
    correlations: np.ndarray,
    eligible: np.ndarray,
    *,
    cluster_count: int = CLUSTER_COUNT,
    minimum_size: int = CLUSTER_MINIMUM,
) -> np.ndarray:
    """Fixed average-linkage clustering without an external clustering dependency."""
    slots = np.flatnonzero(eligible)
    labels = np.full(correlations.shape[0], -1, dtype=np.int16)
    if slots.size < cluster_count:
        return labels
    distance = np.clip(1.0 - correlations[np.ix_(slots, slots)], 0.0, 2.0)
    distance[~np.isfinite(distance)] = 2.0
    clusters = [np.array([index], dtype=np.int16) for index in range(slots.size)]
    while len(clusters) > cluster_count:
        best = None
        for left in range(len(clusters) - 1):
            for right in range(left + 1, len(clusters)):
                value = float(
                    np.mean(distance[np.ix_(clusters[left], clusters[right])])
                )
                candidate = (value, left, right)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, left, right = best
        clusters[left] = np.concatenate((clusters[left], clusters[right]))
        del clusters[right]
    while len(clusters) > 1 and min(map(len, clusters)) < minimum_size:
        source = min(
            range(len(clusters)), key=lambda index: (len(clusters[index]), index)
        )
        destination = min(
            (index for index in range(len(clusters)) if index != source),
            key=lambda index: (
                float(np.mean(distance[np.ix_(clusters[source], clusters[index])])),
                index,
            ),
        )
        clusters[destination] = np.concatenate(
            (clusters[destination], clusters[source])
        )
        del clusters[source]
    clusters.sort(key=lambda value: int(slots[value].min()))
    for label, members in enumerate(clusters):
        labels[slots[members]] = label
    return labels


def _adjusted_rand_index(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = (left >= 0) & (right >= 0)
    left, right = left[valid], right[valid]
    if left.size < 2:
        return None
    table = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(table, (left, right), 1)

    def choose2(values: np.ndarray) -> np.int64:
        return np.sum(values * (values - 1) // 2)

    cells = float(choose2(table))
    rows = float(choose2(table.sum(axis=1)))
    columns = float(choose2(table.sum(axis=0)))
    total = float(left.size * (left.size - 1) // 2)
    expected = rows * columns / total if total else 0.0
    maximum = 0.5 * (rows + columns)
    return 1.0 if maximum == expected else (cells - expected) / (maximum - expected)


def build_peer_graph(store: Path, output_dir: Path) -> Path:
    store, output_dir = store.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    dates = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
    equities = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r")
    membership = np.load(store / "equity_membership.npy", mmap_mode="r")
    return_index = EQUITY_SLOW_CHANNELS.index(
        "previous_close_to_close_return_normalized"
    )
    returns = slow[..., return_index]
    trade_dates = dates.get_column("trade_date").to_list()
    month_starts = [
        index
        for index, value in enumerate(trade_dates)
        if index == 0
        or (value.year, value.month)
        != (
            trade_dates[index - 1].year,
            trade_dates[index - 1].month,
        )
    ]
    month_starts = [index for index in month_starts if index >= GRAPH_LOOKBACK - 1]
    month_count, equity_count = len(month_starts), returns.shape[1]
    peers = np.full((month_count, equity_count, PEER_COUNT), -1, dtype=np.int16)
    peer_rho = np.full(
        (month_count, equity_count, PEER_COUNT), np.nan, dtype=np.float32
    )
    clusters = np.full((month_count, equity_count), -1, dtype=np.int16)
    eligible = np.zeros((month_count, equity_count), dtype=bool)
    audit_rows: list[dict[str, object]] = []
    previous_clusters = None
    ticker = (
        equities.get_column("ticker").to_list() if "ticker" in equities.columns else []
    )
    ticker_to_slot = {str(value): index for index, value in enumerate(ticker)}
    for month, date_index in enumerate(month_starts):
        begin = date_index - GRAPH_LOOKBACK + 1
        window = np.asarray(returns[begin : date_index + 1], dtype=np.float64)
        valid = np.asarray(
            ready[begin : date_index + 1] & membership[begin : date_index + 1]
        )
        for day in range(window.shape[0]):
            day_valid = valid[day]
            if day_valid.any():
                window[day, day_valid] -= np.median(window[day, day_valid])
        correlations = _spearman_matrix(window, valid)
        eligible[month] = valid.sum(axis=0) >= GRAPH_MIN_OBSERVED
        for slot in np.flatnonzero(eligible[month]):
            candidates = np.flatnonzero(
                eligible[month]
                & (np.arange(equity_count) != slot)
                & (correlations[slot] >= PEER_MIN_RHO)
            )
            ordered = candidates[
                np.lexsort((candidates, -correlations[slot, candidates]))
            ][:PEER_COUNT]
            peers[month, slot, : ordered.size] = ordered
            peer_rho[month, slot, : ordered.size] = correlations[slot, ordered]
        clusters[month] = average_linkage_clusters(correlations, eligible[month])
        snapshot = output_dir / f"snapshot_{trade_dates[date_index]}.npz"
        np.savez(
            snapshot,
            date_idx=np.int32(date_index),
            peers=peers[month],
            peer_rho=peer_rho[month],
            clusters=clusters[month],
            eligible=eligible[month],
        )
        pair_audit = {}
        for left, right in (("PETR3", "PETR4"), ("ELET3", "ELET6")):
            if left in ticker_to_slot and right in ticker_to_slot:
                a, b = ticker_to_slot[left], ticker_to_slot[right]
                pair_audit[f"{left}_{right}"] = {
                    "left_top1": int(peers[month, a, 0]) == b,
                    "right_top1": int(peers[month, b, 0]) == a,
                }
        sizes = np.bincount(clusters[month][clusters[month] >= 0])
        audit_rows.append(
            {
                "date_idx": date_index,
                "trade_date": str(trade_dates[date_index]),
                "eligible_count": int(eligible[month].sum()),
                "cluster_sizes": sizes.tolist(),
                "adjusted_rand_vs_previous": (
                    None
                    if previous_clusters is None
                    else _adjusted_rand_index(previous_clusters, clusters[month])
                ),
                "share_class_top1": pair_audit,
                "snapshot": snapshot.name,
                "snapshot_sha256": _sha256(snapshot),
            }
        )
        previous_clusters = clusters[month].copy()
    arrays = output_dir / "monthly_graphs.npz"
    np.savez(
        arrays,
        month_date_idx=np.asarray(month_starts, dtype=np.int32),
        peers=peers,
        peer_rho=peer_rho,
        clusters=clusters,
        eligible=eligible,
    )
    _atomic_json(
        output_dir / "audit.json",
        {
            "schema": "EXPERIMENT46_PEER_GRAPH_AUDIT_V1",
            "monthly_snapshots": audit_rows,
            "gate": None,
        },
    )
    _atomic_json(
        output_dir / "manifest.json",
        {
            "schema": "EXPERIMENT46_PIT_PEER_GRAPH_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_store_identity": feature_store_identity(store),
            "axes": feature_store_axis_identity(store),
            "parameters": {
                "return_field": EQUITY_SLOW_CHANNELS[return_index],
                "market_residual": "per-date median removed",
                "lookback_sessions": GRAPH_LOOKBACK,
                "minimum_observed_sessions": GRAPH_MIN_OBSERVED,
                "correlation": "pairwise Spearman",
                "peer_count": PEER_COUNT,
                "peer_minimum": PEER_MINIMUM,
                "peer_min_rho": PEER_MIN_RHO,
                "linkage": "average",
                "cluster_count_before_small_merge": CLUSTER_COUNT,
                "cluster_minimum": CLUSTER_MINIMUM,
            },
            "arrays": {
                "monthly_graphs.npz": {
                    "sha256": _sha256(arrays),
                    "month_count": month_count,
                }
            },
            "audit_sha256": _sha256(output_dir / "audit.json"),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return output_dir


def _peer_stat(
    measurement: np.ndarray,
    valid: np.ndarray,
    peers: np.ndarray,
    statistic: str,
) -> tuple[np.ndarray, np.ndarray]:
    safe = np.maximum(peers, 0)
    peer_valid = (peers >= 0) & valid[safe]
    count = peer_valid.sum(axis=1)
    values = measurement[safe]
    masked = np.where(peer_valid, values, 0.0)
    if statistic == "mean":
        result = masked.sum(axis=1) / np.maximum(count, 1)
    elif statistic == "std":
        mean = masked.sum(axis=1) / np.maximum(count, 1)
        result = np.sqrt(
            np.where(peer_valid, (values - mean[:, None]) ** 2, 0.0).sum(axis=1)
            / np.maximum(count, 1)
        )
    elif statistic == "breadth":
        result = (peer_valid & (values > 0)).sum(axis=1) / np.maximum(count, 1)
    else:
        raise ValueError(statistic)
    return result, count >= PEER_MINIMUM


def build_peer_feature_library(store: Path, graph_dir: Path, output_dir: Path) -> Path:
    store, graph_dir, output_dir = map(Path.resolve, (store, graph_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    with np.load(graph_dir / "monthly_graphs.npz") as graph:
        month_dates = graph["month_date_idx"].copy()
        monthly_peers = graph["peers"].copy()
    dynamic = np.load(store / "equity_features.npy", mmap_mode="r")
    slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    membership = np.load(store / "equity_membership.npy", mmap_mode="r")
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r")
    shape = (
        dynamic.shape[0],
        dynamic.shape[1],
        len(DECISION_EQUITY_INDICES),
        len(PEER_FEATURES),
    )
    values_path, mask_path = output_dir / "values.npy", output_dir / "mask.npy"
    values = np.lib.format.open_memmap(
        values_path, mode="w+", dtype=np.float32, shape=shape
    )
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.bool_, shape=shape)
    values[...] = 0
    masks[...] = False
    observed_index = DYNAMIC_CHANNELS.index("observed")
    ret15_index = DYNAMIC_CHANNELS.index("return_15m_normalized")
    ret60_index = DYNAMIC_CHANNELS.index("return_60m_normalized")
    volume_index = DYNAMIC_CHANNELS.index("volume_surprise")
    daily_index = EQUITY_SLOW_CHANNELS.index(
        "previous_close_to_close_return_normalized"
    )
    for day in range(dynamic.shape[0]):
        month = int(np.searchsorted(month_dates, day, side="right") - 1)
        if month < 0:
            continue
        peers = monthly_peers[month]
        active = np.asarray(membership[day] & ready[day])
        daily = np.asarray(slow[day, :, daily_index], dtype=np.float64)
        peer_daily, peer_daily_valid = _peer_stat(daily, active, peers, "mean")
        for decision, cutoff in enumerate(DECISION_EQUITY_INDICES):
            observed = np.asarray(dynamic[day, :, :, observed_index] > 0.5)
            valid15 = active & (observed[:, cutoff - 15 : cutoff].sum(axis=1) >= 12)
            valid60 = active & (observed[:, cutoff - 60 : cutoff].sum(axis=1) >= 48)
            valid_volume = active & observed[:, cutoff - 1]
            ret15 = np.asarray(
                dynamic[day, :, cutoff - 1, ret15_index], dtype=np.float64
            )
            ret60 = np.asarray(
                dynamic[day, :, cutoff - 1, ret60_index], dtype=np.float64
            )
            volume = np.asarray(
                dynamic[day, :, cutoff - 1, volume_index], dtype=np.float64
            )
            peer15, peer15_valid = _peer_stat(ret15, valid15, peers, "mean")
            peer60, peer60_valid = _peer_stat(ret60, valid60, peers, "mean")
            dispersion, dispersion_valid = _peer_stat(ret60, valid60, peers, "std")
            breadth, breadth_valid = _peer_stat(ret15, valid15, peers, "breadth")
            peer_volume, peer_volume_valid = _peer_stat(
                volume, valid_volume, peers, "mean"
            )
            columns = (
                (peer15, active & peer15_valid),
                (peer60, active & peer60_valid),
                (ret60 - peer60, valid60 & peer60_valid),
                (dispersion, active & dispersion_valid),
                (breadth, active & breadth_valid),
                (volume - peer_volume, valid_volume & peer_volume_valid),
                (peer_daily, active & peer_daily_valid),
                (daily - peer_daily, active & peer_daily_valid),
            )
            for feature, (column, valid_column) in enumerate(columns):
                values[day, valid_column, decision, feature] = column[valid_column]
                masks[day, valid_column, decision, feature] = True
    values.flush()
    masks.flush()
    del values, masks
    graph_manifest = graph_dir / "manifest.json"
    _atomic_json(
        output_dir / "manifest.json",
        {
            "schema": EXTERNAL_SIDECAR_SCHEMA,
            "candidate_schema": "EXPERIMENT46_PEER_FEATURE_LIBRARY_V1",
            "cadence": "intraday",
            "feature_names": list(PEER_FEATURES),
            "feature_store_identity": feature_store_identity(store),
            "axes": feature_store_axis_identity(store),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": {
                "peer_graph": str(graph_dir),
                "peer_graph_manifest_sha256": _sha256(graph_manifest),
                "last_predecision_minute": True,
                "minimum_peer_measurements": PEER_MINIMUM,
            },
            "arrays": {
                "values.npy": {
                    "shape": list(shape),
                    "dtype": "float32",
                    "sha256": _sha256(values_path),
                },
                "mask.npy": {
                    "shape": list(shape),
                    "dtype": "bool",
                    "sha256": _sha256(mask_path),
                },
            },
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return output_dir


def _feature_ic(
    values: np.ndarray,
    masks: np.ndarray,
    targets: np.ndarray,
    label_mask: np.ndarray,
    dates: np.ndarray,
) -> float:
    predictions = np.repeat(values[..., None], len(HORIZONS), axis=-1)
    return primary_validation_score(
        predictions.astype(np.float32),
        targets,
        label_mask & masks[..., None],
        dates,
    )


def _pooled_standardized_correlation(
    left: np.ndarray,
    left_mask: np.ndarray,
    right: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    mask = left_mask & right_mask
    counts = mask.sum(axis=1)
    valid_rows = counts >= 30
    if not valid_rows.any():
        return float("nan")
    safe = np.maximum(counts, 1)
    left_centered = np.where(
        mask, left - np.where(mask, left, 0).sum(axis=1)[:, None] / safe[:, None], 0
    )
    right_centered = np.where(
        mask, right - np.where(mask, right, 0).sum(axis=1)[:, None] / safe[:, None], 0
    )
    left_centered[~valid_rows] = 0
    right_centered[~valid_rows] = 0
    denominator = np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    return (
        float(np.sum(left_centered * right_centered) / denominator)
        if denominator
        else float("nan")
    )


def screen_peer_features(
    store: Path,
    library_dir: Path,
    output_dir: Path,
    *,
    zero_dynamic_channels: tuple[int, ...],
    zero_slow_fields: tuple[int, ...],
) -> Path:
    store, library_dir, output_dir = map(Path.resolve, (store, library_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    sample = (
        pl.read_parquet(store / "sample_index.parquet")
        .filter(pl.col("trade_date") <= pl.lit(F2_END_DATE).str.to_date())
        .sort("sample_id")
    )
    date_idx = sample["date_idx"].to_numpy()
    decision_idx = sample["decision_idx"].to_numpy()
    values_all = np.load(library_dir / "values.npy", mmap_mode="r")
    masks_all = np.load(library_dir / "mask.npy", mmap_mode="r")
    features = np.asarray(values_all[date_idx, :, decision_idx])
    feature_masks = np.asarray(masks_all[date_idx, :, decision_idx])
    targets = np.asarray(
        np.load(store / "targets.npy", mmap_mode="r")[date_idx, :, decision_idx]
    )
    label_mask = np.asarray(
        np.load(store / "label_mask.npy", mmap_mode="r")[date_idx, :, decision_idx]
    )
    membership = np.load(store / "equity_membership.npy", mmap_mode="r")
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r")
    active = np.asarray(membership[date_idx] & ready[date_idx])
    unique_dates = np.unique(date_idx)
    split_date = unique_dates[len(unique_dates) // 2]
    halves = (date_idx < split_date, date_idx >= split_date)
    dynamic = np.load(store / "equity_features.npy", mmap_mode="r")
    slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    surviving_dynamic = [
        index
        for index in range(len(DYNAMIC_CHANNELS))
        if index not in zero_dynamic_channels
    ]
    surviving_slow = [
        index
        for index in range(len(EQUITY_SLOW_CHANNELS))
        if index not in zero_slow_fields
    ]
    rows = []
    for feature, name in enumerate(PEER_FEATURES):
        half_ics = [
            _feature_ic(
                features[half, :, feature],
                feature_masks[half, :, feature],
                targets[half],
                label_mask[half],
                date_idx[half],
            )
            for half in halves
        ]
        correlations: list[float] = []
        for channel in surviving_dynamic:
            current = np.empty(features.shape[:2], dtype=np.float32)
            for decision in np.unique(decision_idx):
                selected = decision_idx == decision
                cutoff = DECISION_EQUITY_INDICES[int(decision)]
                current[selected] = dynamic[date_idx[selected], :, cutoff - 1, channel]
            correlations.append(
                _pooled_standardized_correlation(
                    features[:, :, feature],
                    feature_masks[:, :, feature],
                    current,
                    active,
                )
            )
        for channel in surviving_slow:
            current = np.asarray(slow[date_idx, :, channel])
            correlations.append(
                _pooled_standardized_correlation(
                    features[:, :, feature],
                    feature_masks[:, :, feature],
                    current,
                    active,
                )
            )
        finite = [abs(value) for value in correlations if np.isfinite(value)]
        max_existing = max(finite, default=0.0)
        finite_halves = all(np.isfinite(value) for value in half_ics)
        stable = finite_halves and half_ics[0] * half_ics[1] > 0
        minimum = min(map(abs, half_ics)) if finite_halves else 0.0
        eligible = (
            stable
            and minimum >= F2_MIN_HALF_ABS_IC
            and max_existing < F2_MAX_ABS_CORRELATION
        )
        rows.append(
            {
                "feature": name,
                "half_1_ic": half_ics[0] if finite_halves else None,
                "half_2_ic": half_ics[1] if finite_halves else None,
                "fit_mean_ic": float(np.mean(half_ics)) if finite_halves else None,
                "split_half_sign_consistent": stable,
                "minimum_half_abs_ic": minimum,
                "max_abs_correlation_surviving_store_v2": max_existing,
                "incremental_score": minimum * (1.0 - max_existing**2),
                "eligible_before_candidate_redundancy": eligible,
            }
        )
    ordered = sorted(
        rows, key=lambda row: (-float(row["incremental_score"]), str(row["feature"]))
    )
    selected: list[str] = []
    for row in ordered:
        if not row["eligible_before_candidate_redundancy"]:
            row["selected"] = False
            row["exclusion"] = "F2 eligibility rule"
            continue
        index = PEER_FEATURES.index(str(row["feature"]))
        redundant = any(
            abs(
                _pooled_standardized_correlation(
                    features[:, :, index],
                    feature_masks[:, :, index],
                    features[:, :, PEER_FEATURES.index(other)],
                    feature_masks[:, :, PEER_FEATURES.index(other)],
                )
            )
            >= F2_MAX_ABS_CORRELATION
            for other in selected
        )
        row["selected"] = not redundant
        row["exclusion"] = "retained-candidate redundancy" if redundant else None
        if not redundant:
            selected.append(str(row["feature"]))
    selection_path = output_dir / "f2_selection.json"
    _atomic_json(
        selection_path,
        {
            "schema": "EXPERIMENT46_F2_SELECTION_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fit_end_date": F2_END_DATE,
            "selection_rule": {
                "same_sign_chronological_halves": True,
                "minimum_half_abs_ic": F2_MIN_HALF_ABS_IC,
                "maximum_abs_redundancy": F2_MAX_ABS_CORRELATION,
                "incremental_order": "min(abs(half ICs)) * (1 - max_existing_corr^2)",
                "minimum_features_for_f3": 2,
            },
            "store_v2_zero_dynamic_channels": list(zero_dynamic_channels),
            "store_v2_zero_slow_fields": list(zero_slow_fields),
            "features": rows,
            "selected_features": selected,
            "f3_allowed": len(selected) >= 2,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    if len(selected) >= 2:
        sidecar = output_dir / "selected_sidecar"
        sidecar.mkdir()
        indices = [PEER_FEATURES.index(name) for name in selected]
        for filename, dtype in (("values.npy", np.float32), ("mask.npy", np.bool_)):
            source = np.load(library_dir / filename, mmap_mode="r")
            destination = np.lib.format.open_memmap(
                sidecar / filename,
                mode="w+",
                dtype=dtype,
                shape=(*source.shape[:-1], len(indices)),
            )
            destination[...] = source[..., indices]
            destination.flush()
            del destination
        source_manifest = json.loads(
            (library_dir / "manifest.json").read_text(encoding="utf-8")
        )
        _atomic_json(
            sidecar / "manifest.json",
            {
                **source_manifest,
                "candidate_schema": "EXPERIMENT46_F3_COMBINED_PEER_SIDECAR_V1",
                "feature_names": selected,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "provenance": {
                    **source_manifest["provenance"],
                    "source_library": str(library_dir),
                    "f2_selection": str(selection_path),
                    "f2_selection_sha256": _sha256(selection_path),
                },
                "arrays": {
                    filename: {
                        "shape": list(np.load(sidecar / filename, mmap_mode="r").shape),
                        "dtype": dtype,
                        "sha256": _sha256(sidecar / filename),
                    }
                    for filename, dtype in (
                        ("values.npy", "float32"),
                        ("mask.npy", "bool"),
                    )
                },
            },
        )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment-46 PIT peer graph and F2")
    subparsers = parser.add_subparsers(dest="command", required=True)
    graph = subparsers.add_parser("graph")
    graph.add_argument("--store", type=Path, required=True)
    graph.add_argument("--output-dir", type=Path, required=True)
    library = subparsers.add_parser("library")
    library.add_argument("--store", type=Path, required=True)
    library.add_argument("--graph-dir", type=Path, required=True)
    library.add_argument("--output-dir", type=Path, required=True)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--store", type=Path, required=True)
    screen.add_argument("--library-dir", type=Path, required=True)
    screen.add_argument("--output-dir", type=Path, required=True)
    screen.add_argument("--zero-dynamic-channels", type=int, nargs="*", default=[])
    screen.add_argument("--zero-slow-fields", type=int, nargs="*", default=[])
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "graph":
        print(build_peer_graph(**args))
    elif command == "library":
        print(build_peer_feature_library(**args))
    else:
        args["zero_dynamic_channels"] = tuple(args["zero_dynamic_channels"])
        args["zero_slow_fields"] = tuple(args["zero_slow_fields"])
        print(screen_peer_features(**args))


if __name__ == "__main__":
    main()
