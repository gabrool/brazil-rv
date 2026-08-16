from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .analyze_preprocessing import (
    AUDIT_SEED,
    BASE_CHANNELS,
    DERIVED_CHANNELS,
    PAIR_THRESHOLD,
    PCA_FAMILIES,
    VOLUME_STATE_CHANNELS,
    VOLATILITY_STATE_CHANNELS,
    AuditArrays,
    AuditDates,
    DistributionAccumulator,
    pca_summary,
    redundancy_tables,
    shift_metrics,
)
from .audit import DYNAMIC_BOUNDS, SLOW_BOUNDS
from .contract import (
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    DECISION_GLOBAL_INDICES,
    DYNAMIC_CHANNELS,
    FIXED_RATE_CONTEXT_SYMBOLS,
    GLOBAL_CONTEXT_SYMBOLS,
    GLOBAL_SLOW_CHANNELS,
    GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
    LOCAL_CONTEXT_SYMBOLS,
    MIN_ACTIVE_EQUITIES,
    SLOW_CHANNELS,
)

LOCAL_INAPPLICABLE_SLOW = frozenset((*range(13, 15), *range(17, 26)))
DI1N_INAPPLICABLE_SLOW = frozenset((1, 2, *range(13, 15), *range(17, 26), 30, 31))


@dataclass(frozen=True)
class SampleKeys:
    date_idx: NDArray[np.int64]
    entity_idx: NDArray[np.int64]
    decision_idx: NDArray[np.int64]
    full_row_count: int

    def __len__(self) -> int:
        return int(self.date_idx.size)


@dataclass(frozen=True)
class FeatureSample:
    values: NDArray[np.float64]
    valid: NDArray[np.bool_]
    feature_names: tuple[str, ...]
    feature_kinds: tuple[str, ...]
    date_idx: NDArray[np.int64]
    entity_idx: NDArray[np.int64]
    decision_idx: NDArray[np.int64]
    full_row_count: int


def deterministic_stratified_keys(
    dates: NDArray[np.integer],
    applicable_entities: NDArray[np.bool_],
    *,
    decision_count: int,
    decisions_per_date: int,
    entities_per_decision: int,
    seed: int = AUDIT_SEED,
) -> SampleKeys:
    """Balance bounded samples over dates, decisions, and entities."""
    date_indices = np.asarray(dates, dtype=np.int64)
    applicable = np.asarray(applicable_entities, dtype=bool)
    if applicable.ndim != 2 or applicable.shape[0] != date_indices.size:
        raise ValueError("Applicability must align to dates")
    if decision_count <= 0 or decisions_per_date <= 0 or entities_per_decision <= 0:
        raise ValueError("Sampling dimensions must be positive")
    output: list[tuple[int, int, int]] = []
    full_count = 0
    chosen_decisions = min(decisions_per_date, decision_count)
    stride = max(1, decision_count // chosen_decisions)
    for local_date, date_idx in enumerate(date_indices):
        entities = np.flatnonzero(applicable[local_date])
        full_count += int(entities.size) * decision_count
        if not entities.size:
            continue
        offset = (seed + int(date_idx) * 17) % decision_count
        decisions = np.unique(
            (offset + stride * np.arange(chosen_decisions)) % decision_count
        )
        for decision in decisions:
            start = (
                seed + int(date_idx) * 131 + int(decision) * 37 + local_date
            ) % entities.size
            count = min(entities_per_decision, entities.size)
            selected = entities[(start + np.arange(count)) % entities.size]
            output.extend(
                (int(date_idx), int(entity), int(decision)) for entity in selected
            )
    if not output:
        return SampleKeys(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            full_count,
        )
    rows = np.asarray(output, dtype=np.int64)
    return SampleKeys(rows[:, 0], rows[:, 1], rows[:, 2], full_count)


def _historical_observed_count(
    features: NDArray[np.generic], keys: SampleKeys, minutes: NDArray[np.int64]
) -> NDArray[np.int64]:
    offsets = np.arange(1, 21, dtype=np.int64)
    history = keys.date_idx[:, None] - offsets
    available = history >= 0
    history = np.maximum(history, 0)
    observed = np.asarray(
        features[history, keys.entity_idx[:, None], minutes[:, None], 5], dtype=bool
    )
    return (observed & available).sum(axis=1, dtype=np.int64)


def _endpoint_valid(
    features: NDArray[np.generic],
    keys: SampleKeys,
    minutes: NDArray[np.int64],
    window: int,
    mapping_changed: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    endpoint = minutes - window
    valid = endpoint >= 0
    endpoint = np.maximum(endpoint, 0)
    valid &= np.asarray(
        features[keys.date_idx, keys.entity_idx, endpoint, 5], dtype=bool
    )
    if mapping_changed is not None:
        for row in np.flatnonzero(valid):
            valid[row] &= not mapping_changed[
                keys.date_idx[row],
                keys.entity_idx[row],
                endpoint[row] + 1 : minutes[row] + 1,
            ].any()
    return valid


def _realized_valid(
    features: NDArray[np.generic],
    keys: SampleKeys,
    minutes: NDArray[np.int64],
    window: int,
    mapping_changed: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    valid = minutes >= window
    minimum = int(math.ceil(0.8 * window))
    for row in np.flatnonzero(valid):
        start = int(minutes[row] - window)
        stop = int(minutes[row])
        observed = np.asarray(
            features[keys.date_idx[row], keys.entity_idx[row], start : stop + 1, 5],
            dtype=bool,
        )
        adjacent = observed[1:] & observed[:-1]
        if mapping_changed is not None:
            adjacent &= ~mapping_changed[
                keys.date_idx[row], keys.entity_idx[row], start + 1 : stop + 1
            ]
        valid[row] = int(adjacent.sum()) >= minimum
    return valid


def _dynamic_validity(
    features: NDArray[np.generic],
    keys: SampleKeys,
    minutes: NDArray[np.int64],
    ready: NDArray[np.bool_],
    *,
    entity_kind: str,
    mapping_changed: NDArray[np.bool_] | None,
) -> NDArray[np.bool_]:
    valid = np.zeros((len(keys), len(DYNAMIC_CHANNELS)), dtype=bool)
    observed = np.asarray(
        features[keys.date_idx, keys.entity_idx, minutes, 5], dtype=bool
    )
    move_valid = observed.copy()
    if mapping_changed is not None:
        move_valid &= ~mapping_changed[keys.date_idx, keys.entity_idx, minutes]
    valid[:, :4] = move_valid[:, None]
    valid[:, 4] = observed & (_historical_observed_count(features, keys, minutes) >= 10)
    valid[:, 5] = ready
    valid[:, 6] = observed
    for channel, window in zip((7, 8, 9), (15, 30, 60), strict=True):
        valid[:, channel] = observed & _endpoint_valid(
            features, keys, minutes, window, mapping_changed
        )
    for channel, window in zip((10, 11, 12), (15, 30, 60), strict=True):
        valid[:, channel] = observed & _realized_valid(
            features, keys, minutes, window, mapping_changed
        )
    valid[:, 13] = observed
    valid[:, 14] = observed
    valid[:, 15] = ready
    if mapping_changed is not None:
        for row in np.flatnonzero(valid[:, 6]):
            history = np.asarray(
                features[
                    keys.date_idx[row],
                    keys.entity_idx[row],
                    : minutes[row] + 1,
                    5,
                ],
                dtype=bool,
            )
            positions = np.flatnonzero(history)
            if (
                positions.size
                and mapping_changed[
                    keys.date_idx[row],
                    keys.entity_idx[row],
                    positions[0] + 1 : minutes[row] + 1,
                ].any()
            ):
                valid[row, 6] = False
    if entity_kind == "equity":
        for channel in (16, 18, 20, 22):
            valid[:, channel] = valid[:, 7]
        for channel in (17, 19, 21, 23):
            valid[:, channel] = valid[:, 9]
        valid[:, 24] = valid[:, 4]
        valid[:, 25] = valid[:, 11]
    valid &= ready[:, None]
    return valid


def _apply_equity_cross_section_validity(
    features: NDArray[np.generic],
    membership: NDArray[np.generic],
    readiness: NDArray[np.generic],
    keys: SampleKeys,
    minutes: NDArray[np.int64],
    valid: NDArray[np.bool_],
) -> None:
    """Mask neutral channels that the builder leaves outside usable cross-sections."""
    groups = np.unique(np.column_stack((keys.date_idx, minutes)), axis=0)
    cached: tuple[int, NDArray[np.bool_], NDArray[np.bool_]] | None = None
    for date_idx, minute in groups:
        date_idx, minute = int(date_idx), int(minute)
        rows = (keys.date_idx == date_idx) & (minutes == minute)
        if cached is None or cached[0] != date_idx:
            cached = (
                date_idx,
                np.asarray(membership[date_idx] & readiness[date_idx], dtype=bool),
                np.asarray(features[date_idx, :, :, 5], dtype=bool),
            )
        _, active, observed = cached
        current = observed[:, minute]

        def endpoint(window: int) -> NDArray[np.bool_]:
            if minute < window:
                return np.zeros(current.shape, dtype=bool)
            return current & observed[:, minute - window]

        return_15 = active & endpoint(15)
        return_60 = active & endpoint(60)
        history_start = max(0, date_idx - 20)
        volume = (
            active
            & current
            & (
                np.asarray(
                    features[history_start:date_idx, :, minute, 5], dtype=bool
                ).sum(axis=0)
                >= 10
            )
        )
        realized_30 = np.zeros(current.shape, dtype=bool)
        if minute >= 30:
            adjacent = (
                observed[:, minute - 29 : minute + 1]
                & observed[:, minute - 30 : minute]
            )
            realized_30 = active & current & (adjacent.sum(axis=1) >= 24)

        sampled = keys.entity_idx[rows]
        for source_valid, aggregate_channels, rank_channel in (
            (return_15, (16, 18, 20), 22),
            (return_60, (17, 19, 21), 23),
        ):
            usable = int(source_valid.sum()) >= MIN_ACTIVE_EQUITIES
            for channel in aggregate_channels:
                valid[rows, channel] = usable & active[sampled]
            valid[rows, rank_channel] = usable & source_valid[sampled]
        volume_usable = int(volume.sum()) >= MIN_ACTIVE_EQUITIES
        realized_usable = int(realized_30.sum()) >= MIN_ACTIVE_EQUITIES
        valid[rows, 24] = volume_usable & volume[sampled]
        valid[rows, 25] = realized_usable & realized_30[sampled]


def slow_applicability(entity_kind: str, entity_idx: int) -> NDArray[np.bool_]:
    width = len(GLOBAL_SLOW_CHANNELS) if entity_kind == "global" else len(SLOW_CHANNELS)
    result = np.ones(width, dtype=bool)
    if entity_kind == "equity":
        result[30:32] = False
    elif entity_kind == "local":
        result[list(LOCAL_INAPPLICABLE_SLOW)] = False
        symbol = LOCAL_CONTEXT_SYMBOLS[entity_idx]
        if symbol not in FIXED_RATE_CONTEXT_SYMBOLS:
            result[30:32] = False
        if symbol == "DI1$N":
            result[list(DI1N_INAPPLICABLE_SLOW)] = False
    elif entity_kind == "global":
        result[list(GLOBAL_UNUSED_SLOW_CHANNEL_INDICES)] = False
    else:
        raise ValueError(f"Unknown entity kind: {entity_kind}")
    return result


def extract_feature_sample(
    arrays: AuditArrays,
    keys: SampleKeys,
    *,
    entity_kind: str,
    mapping_changed: NDArray[np.bool_] | None = None,
) -> FeatureSample:
    if entity_kind == "equity":
        dynamic_name, slow_name = "equity_features.npy", "equity_slow.npy"
        decisions = np.asarray(DECISION_EQUITY_INDICES)
        membership = arrays.array("equity_membership.npy")
        readiness = arrays.array("equity_data_ready.npy")
        applicable = np.asarray(membership[keys.date_idx, keys.entity_idx], dtype=bool)
        ready = applicable & np.asarray(
            readiness[keys.date_idx, keys.entity_idx], dtype=bool
        )
    elif entity_kind == "local":
        dynamic_name, slow_name = "context_features.npy", "context_slow.npy"
        decisions = np.asarray(DECISION_CONTEXT_INDICES)
        readiness = arrays.array("context_data_ready.npy")
        ready = np.asarray(readiness[keys.date_idx, keys.entity_idx], dtype=bool)
    elif entity_kind == "global":
        dynamic_name, slow_name = "global_features.npy", "global_slow.npy"
        decisions = np.asarray(DECISION_GLOBAL_INDICES)
        readiness = arrays.array("global_data_ready.npy")
        ready = np.asarray(
            readiness[keys.date_idx, keys.entity_idx, keys.decision_idx], dtype=bool
        )
    else:
        raise ValueError(f"Unknown entity kind: {entity_kind}")
    dynamic = arrays.array(dynamic_name)
    slow = arrays.array(slow_name)
    minutes = decisions[keys.decision_idx] - 1
    dynamic_values = np.asarray(
        dynamic[keys.date_idx, keys.entity_idx, minutes], dtype=np.float64
    )
    dynamic_valid = _dynamic_validity(
        dynamic,
        keys,
        minutes,
        ready,
        entity_kind=entity_kind,
        mapping_changed=mapping_changed,
    )
    if entity_kind == "equity":
        _apply_equity_cross_section_validity(
            dynamic,
            membership,
            readiness,
            keys,
            minutes,
            dynamic_valid,
        )
    if entity_kind == "global":
        slow_values = np.asarray(
            slow[keys.date_idx, keys.entity_idx, keys.decision_idx], dtype=np.float64
        )
    else:
        slow_values = np.asarray(slow[keys.date_idx, keys.entity_idx], dtype=np.float64)
    slow_valid = np.stack(
        [slow_applicability(entity_kind, int(slot)) for slot in keys.entity_idx]
    )
    slow_valid &= ready[:, None]
    slow_names = GLOBAL_SLOW_CHANNELS if entity_kind == "global" else SLOW_CHANNELS
    return FeatureSample(
        np.concatenate((dynamic_values, slow_values), axis=1),
        np.concatenate((dynamic_valid, slow_valid), axis=1),
        tuple((*DYNAMIC_CHANNELS, *slow_names)),
        tuple((*("dynamic" for _ in DYNAMIC_CHANNELS), *("slow" for _ in slow_names))),
        keys.date_idx,
        keys.entity_idx,
        keys.decision_idx,
        keys.full_row_count,
    )


def _stats(
    registry: dict[tuple[str, str, str, str], DistributionAccumulator],
    key: tuple[str, str, str, str],
    bounds: tuple[float, float] | None,
) -> DistributionAccumulator:
    if key not in registry:
        registry[key] = DistributionAccumulator(
            ":".join(key),
            None if bounds is None else bounds[0],
            None if bounds is None else bounds[1],
        )
    return registry[key]


def _rows(
    registry: dict[tuple[str, str, str, str], DistributionAccumulator],
) -> list[dict[str, object]]:
    return [
        {
            "entity_kind": entity,
            "scope_kind": scope,
            "scope_value": value,
            "feature": feature,
            **stats.row(),
        }
        for (entity, scope, value, feature), stats in sorted(registry.items())
    ]


def run_normalization_audit(
    arrays: AuditArrays,
    dates: AuditDates,
    equity_index: pl.DataFrame,
    *,
    global_mapping_changed: NDArray[np.bool_] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Audit unique base minutes plus decision-cutoff derived and slow state."""
    registry: dict[tuple[str, str, str, str], DistributionAccumulator] = {}
    security = {
        (slot, channel): DistributionAccumulator(
            f"security:{slot}:{DYNAMIC_CHANNELS[channel]}",
            DYNAMIC_BOUNDS[channel][0],
            DYNAMIC_BOUNDS[channel][1],
        )
        for slot in range(equity_index.height)
        for channel in BASE_CHANNELS
    }
    membership = arrays.array("equity_membership.npy")
    specs = (
        (
            "equity",
            arrays.array("equity_features.npy"),
            arrays.array("equity_data_ready.npy"),
            max(DECISION_EQUITY_INDICES),
            tuple(str(slot) for slot in range(equity_index.height)),
        ),
        (
            "local",
            arrays.array("context_features.npy"),
            arrays.array("context_data_ready.npy"),
            max(DECISION_CONTEXT_INDICES),
            LOCAL_CONTEXT_SYMBOLS,
        ),
        (
            "global",
            arrays.array("global_features.npy"),
            None,
            max(DECISION_GLOBAL_INDICES),
            GLOBAL_CONTEXT_SYMBOLS,
        ),
    )
    for entity, features, readiness, visible, labels in specs:
        for date_idx in dates.train:
            date_int = int(date_idx)
            year = dates.trade_dates[date_int].year
            dynamic = np.asarray(features[date_int, :, :visible], dtype=np.float32)
            observed_channel = dynamic[..., 5] > 0.5
            if entity == "equity":
                ready = np.asarray(
                    membership[date_int] & readiness[date_int], dtype=bool
                )
            elif entity == "local":
                ready = np.asarray(readiness[date_int], dtype=bool)
            else:
                ready = observed_channel.any(axis=1)
            applicable = np.broadcast_to(ready[:, None], observed_channel.shape)
            observed = observed_channel & applicable
            move_observed = observed
            if entity == "global" and global_mapping_changed is not None:
                move_observed = (
                    observed & ~global_mapping_changed[date_int, :, :visible]
                )
            history_start = max(0, date_int - 20)
            history = np.asarray(
                features[history_start:date_int, :, :visible, 5], dtype=bool
            )
            volume_valid = observed & (history.sum(axis=0) >= 10)
            for minute_start in range(0, visible, 30):
                stop = min(minute_start + 30, visible)
                for slot, label in enumerate(labels):
                    possible = int(applicable[slot, minute_start:stop].sum())
                    for channel in BASE_CHANNELS:
                        valid = (
                            volume_valid[slot, minute_start:stop]
                            if channel == 4
                            else move_observed[slot, minute_start:stop]
                        )
                        values = dynamic[slot, minute_start:stop, channel][valid]
                        bounds = DYNAMIC_BOUNDS[channel]
                        scopes = [
                            ("overall", "train"),
                            ("year", str(year)),
                            ("time_bin_30m", str(minute_start // 30)),
                        ]
                        if entity != "equity":
                            scopes.append(("context_symbol", label))
                        for scope, value in scopes:
                            _stats(
                                registry,
                                (entity, scope, value, DYNAMIC_CHANNELS[channel]),
                                bounds,
                            ).update(values, possible_count=possible)
                        if entity == "equity":
                            security[slot, channel].update(
                                values, possible_count=possible
                            )

    # Derived channels use exactly the last stored state before each decision.
    for entity, _, _, _, labels in specs:
        if entity == "equity":
            applicable = np.asarray(membership[dates.train], dtype=bool)
            decisions_per_date, entities_per_decision = 11, 8
        else:
            applicable = np.ones((dates.train.size, len(labels)), dtype=bool)
            decisions_per_date, entities_per_decision = 11, len(labels)
        keys = deterministic_stratified_keys(
            dates.train,
            applicable,
            decision_count=len(DECISION_EQUITY_INDICES),
            decisions_per_date=decisions_per_date,
            entities_per_decision=entities_per_decision,
        )
        sample = extract_feature_sample(
            arrays,
            keys,
            entity_kind=entity,
            mapping_changed=(global_mapping_changed if entity == "global" else None),
        )
        for row in range(len(keys)):
            year = dates.trade_dates[int(keys.date_idx[row])].year
            label = labels[int(keys.entity_idx[row])]
            for channel in DERIVED_CHANNELS:
                if not sample.valid[row, channel]:
                    continue
                bounds = DYNAMIC_BOUNDS[channel]
                for scope, value in (
                    ("decision_overall", "train"),
                    ("decision_year", str(year)),
                ):
                    _stats(
                        registry,
                        (entity, scope, value, DYNAMIC_CHANNELS[channel]),
                        bounds,
                    ).update(
                        np.asarray([sample.values[row, channel]]), possible_count=1
                    )
                if entity != "equity":
                    _stats(
                        registry,
                        (
                            entity,
                            "decision_context_symbol",
                            label,
                            DYNAMIC_CHANNELS[channel],
                        ),
                        bounds,
                    ).update(
                        np.asarray([sample.values[row, channel]]), possible_count=1
                    )
            offset = len(DYNAMIC_CHANNELS)
            slow_names = GLOBAL_SLOW_CHANNELS if entity == "global" else SLOW_CHANNELS
            for channel in (*VOLATILITY_STATE_CHANNELS, *VOLUME_STATE_CHANNELS):
                position = offset + channel
                if position >= sample.values.shape[1]:
                    continue
                bounds = SLOW_BOUNDS[channel] if channel < len(SLOW_BOUNDS) else None
                if entity == "global" and channel in GLOBAL_UNUSED_SLOW_CHANNEL_INDICES:
                    continue
                if not slow_applicability(entity, int(keys.entity_idx[row]))[channel]:
                    continue
                feature = slow_names[channel]
                values = (
                    np.asarray([sample.values[row, position]])
                    if sample.valid[row, position]
                    else np.empty(0, dtype=np.float64)
                )
                for scope, value in (
                    ("slow_overall", "train"),
                    ("slow_year", str(year)),
                ):
                    _stats(registry, (entity, scope, value, feature), bounds).update(
                        values, possible_count=1
                    )
                if entity != "equity":
                    _stats(
                        registry,
                        (entity, "slow_context_symbol", label, feature),
                        bounds,
                    ).update(values, possible_count=1)

    metadata = equity_index.sort("equity_slot").to_dicts()
    security_rows: list[dict[str, object]] = []
    for slot, row in enumerate(metadata):
        for channel in BASE_CHANNELS:
            security_rows.append(
                {
                    "row_type": "security",
                    "equity_slot": slot,
                    "security_id": row["security_id"],
                    "latest_ticker": row["latest_ticker"],
                    "feature": DYNAMIC_CHANNELS[channel],
                    **security[slot, channel].row(),
                }
            )
    for channel in BASE_CHANNELS:
        selected = [
            row
            for row in security_rows
            if row["feature"] == DYNAMIC_CHANNELS[channel] and row["valid_count"]
        ]
        for metric in ("mean", "std", "median", "mad", "upper_clipping_fraction"):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            if not values.size:
                continue
            for percentile, result in zip(
                ("p10", "median", "p90"),
                np.quantile(values, (0.1, 0.5, 0.9)),
                strict=True,
            ):
                security_rows.append(
                    {
                        "row_type": "cross_security_summary",
                        "feature": DYNAMIC_CHANNELS[channel],
                        "summary_metric": metric,
                        "summary_percentile": percentile,
                        "summary_value": float(result),
                    }
                )
    return _rows(registry), security_rows


def build_samples(
    arrays: AuditArrays,
    dates: AuditDates,
    *,
    global_mapping_changed: NDArray[np.bool_] | None = None,
) -> tuple[dict[str, FeatureSample], dict[str, FeatureSample], dict[str, object]]:
    membership = arrays.array("equity_membership.npy")
    samples: dict[str, FeatureSample] = {}
    validation_samples: dict[str, FeatureSample] = {}
    manifest: dict[str, object] = {}
    for split_name, indices, destination in (
        ("train", dates.train, samples),
        ("validation", dates.validation, validation_samples),
    ):
        specs = (
            (
                "equity",
                np.asarray(membership[indices], dtype=bool),
                8,
                4,
            ),
            (
                "local",
                np.ones((indices.size, len(LOCAL_CONTEXT_SYMBOLS)), dtype=bool),
                6,
                len(LOCAL_CONTEXT_SYMBOLS),
            ),
            (
                "global",
                np.ones((indices.size, len(GLOBAL_CONTEXT_SYMBOLS)), dtype=bool),
                6,
                len(GLOBAL_CONTEXT_SYMBOLS),
            ),
        )
        for entity, applicable, decisions_per_date, entities_per_decision in specs:
            keys = deterministic_stratified_keys(
                indices,
                applicable,
                decision_count=len(DECISION_EQUITY_INDICES),
                decisions_per_date=decisions_per_date,
                entities_per_decision=entities_per_decision,
            )
            sample = extract_feature_sample(
                arrays,
                keys,
                entity_kind=entity,
                mapping_changed=(
                    global_mapping_changed if entity == "global" else None
                ),
            )
            destination[entity] = sample
            manifest[f"{split_name}_{entity}"] = {
                "sampled_row_count": len(keys),
                "full_applicable_row_count": keys.full_row_count,
                "date_count": int(indices.size),
                "decisions_per_date": decisions_per_date,
                "entities_per_decision": entities_per_decision,
            }
    return samples, validation_samples, manifest


def run_redundancy_and_shift(
    training: dict[str, FeatureSample], validation: dict[str, FeatureSample]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    pairwise_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    pca_rows: list[dict[str, object]] = []
    shift_rows: list[dict[str, object]] = []
    for entity in ("equity", "local", "global"):
        train = training[entity]
        valid = validation[entity]
        for kind in ("dynamic", "slow"):
            positions = np.asarray(
                [
                    index
                    for index, value in enumerate(train.feature_kinds)
                    if value == kind
                ],
                dtype=np.int64,
            )
            names = tuple(train.feature_names[index] for index in positions)
            pairwise, summaries = redundancy_tables(
                train.values[:, positions],
                train.valid[:, positions],
                names,
                minimum_absolute_correlation=0.0,
            )
            for row in pairwise:
                row.update({"entity_kind": entity, "feature_kind": kind})
            for row in summaries:
                row.update({"entity_kind": entity, "feature_kind": kind})
            pairwise_rows.extend(pairwise)
            pair_rows.extend(
                row
                for row in pairwise
                if float(row["absolute_spearman_rho"]) >= PAIR_THRESHOLD
            )
            feature_rows.extend(summaries)
        for family, family_positions in PCA_FAMILIES.items():
            kind = family.split("_", 1)[0]
            offset = 0 if kind == "dynamic" else len(DYNAMIC_CHANNELS)
            positions = np.asarray(
                [
                    offset + value
                    for value in family_positions
                    if offset + value < train.values.shape[1]
                ],
                dtype=np.int64,
            )
            if positions.size < 2:
                continue
            result = pca_summary(train.values[:, positions], train.valid[:, positions])
            pca_rows.append(
                {
                    "entity_kind": entity,
                    "feature_kind": kind,
                    "semantic_family": family,
                    "features": "|".join(
                        train.feature_names[index] for index in positions
                    ),
                    **result,
                }
            )
        for position, name in enumerate(train.feature_names):
            train_values = train.values[train.valid[:, position], position]
            validation_values = valid.values[valid.valid[:, position], position]
            kind = train.feature_kinds[position]
            bounds: tuple[float, float] | None
            if kind == "dynamic":
                bounds = DYNAMIC_BOUNDS[position]
            else:
                slow_position = position - len(DYNAMIC_CHANNELS)
                bounds = (
                    SLOW_BOUNDS[slow_position]
                    if slow_position < len(SLOW_BOUNDS)
                    else None
                )
            if position < len(DYNAMIC_CHANNELS):
                train_possible = (
                    0
                    if entity != "equity" and position >= 16
                    else train.values.shape[0]
                )
                validation_possible = (
                    0
                    if entity != "equity" and position >= 16
                    else valid.values.shape[0]
                )
            else:
                slow_position = position - len(DYNAMIC_CHANNELS)
                train_possible = sum(
                    slow_applicability(entity, int(slot))[slow_position]
                    for slot in train.entity_idx
                )
                validation_possible = sum(
                    slow_applicability(entity, int(slot))[slow_position]
                    for slot in valid.entity_idx
                )
            metrics = shift_metrics(
                train_values,
                validation_values,
                training_possible=int(train_possible),
                validation_possible=int(validation_possible),
                lower_clip=None if bounds is None else bounds[0],
                upper_clip=None if bounds is None else bounds[1],
            )
            shift_rows.append(
                {
                    "entity_kind": entity,
                    "feature_kind": kind,
                    "feature": name,
                    **metrics,
                }
            )
    pairwise_rows.sort(
        key=lambda row: (
            -float(row["absolute_spearman_rho"]),
            str(row["entity_kind"]),
            str(row["feature_left"]),
        )
    )
    return pairwise_rows, pair_rows, feature_rows, pca_rows, shift_rows


def largest_material_shifts(
    rows: list[dict[str, object]], count: int = 15
) -> list[dict[str, object]]:
    def score(row: dict[str, object]) -> float:
        values = (
            row.get("absolute_standardized_mean_difference"),
            row.get("absolute_robust_median_shift"),
            row.get("ks_statistic"),
            row.get("validation_outside_training_p01_p99_fraction"),
            abs(float(row.get("clipping_fraction_change") or 0.0)),
            abs(float(row.get("observed_fraction_change") or 0.0)),
        )
        return max(float(value) for value in values if value is not None)

    return sorted(rows, key=score, reverse=True)[:count]
