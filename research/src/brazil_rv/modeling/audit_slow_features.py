from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from brazil_rv.preprocessing.contract import (
    GLOBAL_SLOW_CHANNELS,
    GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
    SLOW_CHANNELS,
)

from .context_ablation import resolve_context_ablation_for_store
from .contract import (
    GLOBAL_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_SYMBOLS,
    TRAIN_END,
    TRAIN_START,
)
from .data import resolve_feature_store, select_sample_split, validate_feature_store
from .feature_ablation import (
    ResolvedFeatureAblation,
    resolve_feature_ablation_for_store,
)
from .stage2_context_ablation import _feature_store_identity

AUDIT_NAME = "stage4_training_slow_feature_redundancy"
AUDIT_VERSION = 2
FROZEN_CONTEXT_ABLATION = "drop_win_and_global_non_rates"
EXPECTED_RETAINED_CONTEXTS = (
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
    "ZT.v.0",
    "ZN.v.0",
)
EQUITY_LOCAL_AXIS = "equity_local_slow"
GLOBAL_AXIS = "global_slow"
AUDIT_JSON = "slow_feature_audit.json"
STATS_CSV = "slow_feature_stats.csv"
PEARSON_CSV = "slow_feature_pearson.csv"
SPEARMAN_CSV = "slow_feature_spearman.csv"
FOCUSED_CSV = "slow_feature_focused_correlations.csv"


@dataclass(frozen=True)
class SlowAuditGroupInput:
    group_id: str
    group_kind: str
    symbol: str | None
    axis_identity: str
    channel_names: tuple[str, ...]
    values: np.ndarray
    applicable: np.ndarray
    context_disabled_indices: tuple[int, ...] = ()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_csv(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.write_csv(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _nullable_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in matrix
    ]


def _channel_statuses(
    width: int,
    *,
    applicable: np.ndarray,
    context_disabled_indices: tuple[int, ...],
    removed_indices: tuple[int, ...],
) -> tuple[str, ...]:
    if applicable.shape != (width,) or applicable.dtype != np.bool_:
        raise ValueError("Slow-audit applicability mask has the wrong contract")
    disabled = set(context_disabled_indices)
    removed = set(removed_indices)
    if not disabled <= set(range(width)) or not removed <= set(range(width)):
        raise ValueError("Slow-audit status index is outside the canonical axis")
    return tuple(
        "inapplicable"
        if not applicable[index]
        else "context_disabled"
        if index in disabled
        else "removed"
        if index in removed
        else "active"
        for index in range(width)
    )


def _focused_correlations(
    names: tuple[str, ...],
    statuses: tuple[str, ...],
    pearson: np.ndarray,
    spearman: np.ndarray,
    pair_counts: np.ndarray,
) -> list[dict[str, object]]:
    active = tuple(index for index, status in enumerate(statuses) if status == "active")
    focused: list[dict[str, object]] = []
    for removed, status in enumerate(statuses):
        if status != "removed":
            continue
        for method, matrix in (("pearson", pearson), ("spearman", spearman)):
            candidates = [
                (retained, float(matrix[removed, retained]))
                for retained in active
                if np.isfinite(matrix[removed, retained])
            ]
            candidates.sort(key=lambda item: (-abs(item[1]), item[0]))
            for rank, (retained, correlation) in enumerate(candidates[:5], start=1):
                focused.append(
                    {
                        "removed_index": removed,
                        "removed_name": names[removed],
                        "method": method,
                        "rank": rank,
                        "retained_index": retained,
                        "retained_name": names[retained],
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation),
                        "paired_valid_observation_count": int(
                            pair_counts[removed, retained]
                        ),
                    }
                )
    return focused


def compute_slow_statistics(
    values: np.ndarray,
    *,
    slow_names: tuple[str, ...] = SLOW_CHANNELS,
    removed_indices: tuple[int, ...],
    applicable: np.ndarray | None = None,
    context_disabled_indices: tuple[int, ...] = (),
    axis_identity: str = EQUITY_LOCAL_AXIS,
) -> dict[str, object]:
    """Summarize one canonical slow axis; inapplicable inputs remain missing."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(slow_names):
        raise ValueError("Slow-audit values have the wrong shape")
    expected_names = (
        SLOW_CHANNELS if axis_identity == EQUITY_LOCAL_AXIS else GLOBAL_SLOW_CHANNELS
    )
    if axis_identity not in {EQUITY_LOCAL_AXIS, GLOBAL_AXIS}:
        raise ValueError("Slow-audit axis identity is unknown")
    if len(slow_names) != len(set(slow_names)) or slow_names != expected_names:
        raise ValueError("Slow-audit channel axis is not canonical")
    if applicable is None:
        applicable = np.ones(len(slow_names), dtype=bool)
    else:
        applicable = np.asarray(applicable, dtype=bool)
    statuses = _channel_statuses(
        len(slow_names),
        applicable=applicable,
        context_disabled_indices=context_disabled_indices,
        removed_indices=removed_indices,
    )
    effective = array.copy()
    for index, status in enumerate(statuses):
        if status in {"inapplicable", "context_disabled"}:
            effective[:, index] = np.nan

    finite = np.isfinite(effective)
    statistics: list[dict[str, object]] = []
    for index, (name, status) in enumerate(zip(slow_names, statuses, strict=True)):
        valid = effective[finite[:, index], index]
        statistics.append(
            {
                "index": index,
                "name": name,
                "status": status,
                "is_applicable": status != "inapplicable",
                "is_context_disabled": status == "context_disabled",
                "is_removed_position": index in removed_indices,
                "valid_observation_count": int(valid.size),
                "finite_fraction": float(valid.size / array.shape[0]),
                "exact_zero_fraction": (
                    float(np.mean(valid == 0.0)) if valid.size else None
                ),
                "mean": float(np.mean(valid)) if valid.size else None,
                "standard_deviation": (
                    float(np.std(valid, ddof=0)) if valid.size else None
                ),
            }
        )

    frame = pd.DataFrame(effective, columns=slow_names)
    pearson = frame.corr(method="pearson", min_periods=2).to_numpy(dtype=np.float64)
    spearman = frame.corr(method="spearman", min_periods=2).to_numpy(dtype=np.float64)
    pair_counts = finite.astype(np.int64).T @ finite.astype(np.int64)
    return {
        "statistics": statistics,
        "pearson": _nullable_matrix(pearson),
        "spearman": _nullable_matrix(spearman),
        "paired_valid_observation_counts": pair_counts.tolist(),
        "focused": _focused_correlations(
            slow_names, statuses, pearson, spearman, pair_counts
        ),
    }


def _indices(value: object, field: str) -> set[int]:
    if not isinstance(value, list) or any(
        not isinstance(index, int) or isinstance(index, bool) for index in value
    ):
        raise ValueError(f"Feature schema has invalid applicability field: {field}")
    result = set(value)
    if len(result) != len(value):
        raise ValueError(f"Feature schema has duplicate applicability index: {field}")
    return result


def _applicable(width: int, inapplicable: set[int]) -> np.ndarray:
    if not inapplicable <= set(range(width)):
        raise ValueError("Feature schema has an out-of-range inapplicable channel")
    result = np.ones(width, dtype=bool)
    result[list(sorted(inapplicable))] = False
    return result


def _select_training_samples(
    sample_index: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, object]]:
    training = select_sample_split(sample_index, "train").sort("sample_id")
    if (
        training.is_empty()
        or training["trade_date"].min() != TRAIN_START
        or training["trade_date"].max() != TRAIN_END
    ):
        raise ValueError("Slow audit did not resolve the exact training period")
    sample_mask = np.column_stack(
        [
            training[column].to_numpy().astype("<i8", copy=False)
            for column in ("sample_id", "date_idx", "decision_idx")
        ]
    )
    return training, {
        "split": "train",
        "training_start": str(TRAIN_START),
        "training_end": str(TRAIN_END),
        "training_date_count": int(training["trade_date"].n_unique()),
        "training_sample_count": training.height,
        "training_sample_mask_sha256": hashlib.sha256(
            sample_mask.tobytes()
        ).hexdigest(),
        "training_sample_mask_fields": ["sample_id", "date_idx", "decision_idx"],
        "excluded_splits": ["embargo_1", "validation", "embargo_2", "test"],
        "sampling": {"used": False, "seed": None},
    }


def _training_groups(
    store: Path,
) -> tuple[tuple[SlowAuditGroupInput, ...], dict[str, object]]:
    training, sample_contract = _select_training_samples(validate_feature_store(store))
    date_indices = np.unique(training["date_idx"].to_numpy()).astype(np.int64)

    context = resolve_context_ablation_for_store(store, FROZEN_CONTEXT_ABLATION)
    retained_local = tuple(
        symbol
        for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS)
        if slot not in context.local_slots
    )
    retained_global = tuple(
        symbol
        for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS)
        if slot not in context.global_slots
    )
    if (
        *retained_local,
        *retained_global,
    ) != EXPECTED_RETAINED_CONTEXTS or context.equity_slow_indices != (
        SLOW_CHANNELS.index("beta_to_WIN"),
    ):
        raise ValueError(
            "Frozen context ablation does not resolve to the expected core"
        )

    schema = json.loads((store / "feature_schema.json").read_text(encoding="utf-8"))
    family = schema.get("family_inapplicable_zero_fields")
    if not isinstance(family, dict):
        raise ValueError("Feature schema is missing family applicability")
    all_context = _indices(family.get("all_context"), "all_context")
    applicability: dict[str, np.ndarray] = {
        "equity": _applicable(32, _indices(family.get("equity"), "equity")),
        "global": _applicable(32, _indices(schema.get("global_slow"), "global_slow")),
    }
    if tuple(sorted(_indices(schema.get("global_slow"), "global_slow"))) != (
        GLOBAL_UNUSED_SLOW_CHANNEL_INDICES
    ):
        raise ValueError("Feature schema global structural positions are not canonical")
    for symbol in retained_local:
        inapplicable = set(all_context)
        if symbol in {"WIN$", "WDO$"}:
            inapplicable |= _indices(family.get("WIN_WDO"), "WIN_WDO")
        elif symbol in {"DI1F27", "DI1F28", "DI1F29", "DI1F31"}:
            inapplicable |= _indices(
                family.get("fixed_maturity_DI"), "fixed_maturity_DI"
            )
        elif symbol == "DI1$N":
            inapplicable |= _indices(family.get("DI1$N"), "DI1$N")
        applicability[symbol] = _applicable(32, inapplicable)

    equity_slow = np.load(store / "equity_slow.npy", mmap_mode="r")
    membership = np.load(store / "equity_membership.npy", mmap_mode="r")
    equity_ready = np.load(store / "equity_data_ready.npy", mmap_mode="r")
    equity_values = np.asarray(equity_slow[date_indices], dtype=np.float32)
    active_equities = np.asarray(
        membership[date_indices] & equity_ready[date_indices], dtype=bool
    )
    groups: list[SlowAuditGroupInput] = [
        SlowAuditGroupInput(
            "equity",
            "equity",
            None,
            EQUITY_LOCAL_AXIS,
            SLOW_CHANNELS,
            equity_values[active_equities].copy(),
            applicability["equity"],
            context.equity_slow_indices,
        )
    ]

    local_slow = np.load(store / "context_slow.npy", mmap_mode="r")
    local_ready = np.load(store / "context_data_ready.npy", mmap_mode="r")
    for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS):
        if symbol not in retained_local:
            continue
        ready = np.asarray(local_ready[date_indices, slot], dtype=bool)
        rows = np.asarray(local_slow[date_indices, slot], dtype=np.float32)[
            ready
        ].copy()
        groups.append(
            SlowAuditGroupInput(
                f"local:{symbol}",
                "local_context",
                symbol,
                EQUITY_LOCAL_AXIS,
                SLOW_CHANNELS,
                rows,
                applicability[symbol],
            )
        )

    global_slow = np.load(store / "global_slow.npy", mmap_mode="r")
    global_ready = np.load(store / "global_data_ready.npy", mmap_mode="r")
    train_dates = training["date_idx"].to_numpy().astype(np.int64, copy=False)
    decisions = training["decision_idx"].to_numpy().astype(np.int64, copy=False)
    for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS):
        if symbol not in retained_global:
            continue
        ready = np.asarray(global_ready[train_dates, slot, decisions], dtype=bool)
        rows = np.asarray(global_slow[train_dates, slot, decisions], dtype=np.float32)[
            ready
        ].copy()
        groups.append(
            SlowAuditGroupInput(
                f"global:{symbol}",
                "global_context",
                symbol,
                GLOBAL_AXIS,
                GLOBAL_SLOW_CHANNELS,
                rows,
                applicability["global"],
            )
        )

    if tuple(group.group_id for group in groups) != (
        "equity",
        "local:WDO$",
        "local:DI1F27",
        "local:DI1F28",
        "local:DI1F29",
        "local:DI1F31",
        "local:DI1$N",
        "global:ZT.v.0",
        "global:ZN.v.0",
    ):
        raise ValueError("Slow-audit group order does not match the frozen context")
    for group in groups:
        effective = group.values.copy()
        effective[:, ~group.applicable] = np.nan
        if group.context_disabled_indices:
            effective[:, group.context_disabled_indices] = np.nan
        if not np.isfinite(effective).any(axis=1).all():
            raise ValueError(
                f"Slow audit retained an empty applicable row: {group.group_id}"
            )

    sample_contract.update(
        {
            "observation_rule": (
                "Partitioned canonical-axis groups: ready membership-and-data-ready "
                "equities per training date; ready retained-local tokens per training "
                "date; and ready retained-global tokens per training sample decision. "
                "Family-inapplicable and context-disabled cells are excluded as NaN."
            ),
            "group_order": [group.group_id for group in groups],
            "row_group_counts": {
                group.group_id: int(group.values.shape[0]) for group in groups
            },
            "total_observation_count": int(
                sum(group.values.shape[0] for group in groups)
            ),
            "retained_context_symbols": list(EXPECTED_RETAINED_CONTEXTS),
        }
    )
    return tuple(groups), sample_contract


def _group_position_mapping(
    group: SlowAuditGroupInput, ablation: ResolvedFeatureAblation
) -> list[dict[str, object]]:
    mapping = []
    for position in ablation.position_mapping:
        group_name = (
            position.global_name
            if group.axis_identity == GLOBAL_AXIS
            else position.equity_local_name
        )
        mapping.append(
            {
                **position.metadata(),
                "group_axis_identity": group.axis_identity,
                "group_axis_name": group_name,
                "group_structurally_inapplicable": not bool(
                    group.applicable[position.index]
                ),
            }
        )
    return mapping


def _analyze_group(
    group: SlowAuditGroupInput, ablation: ResolvedFeatureAblation
) -> dict[str, object]:
    analysis = compute_slow_statistics(
        group.values,
        slow_names=group.channel_names,
        removed_indices=ablation.slow_indices,
        applicable=group.applicable,
        context_disabled_indices=group.context_disabled_indices,
        axis_identity=group.axis_identity,
    )
    return {
        "group_id": group.group_id,
        "group_kind": group.group_kind,
        "symbol": group.symbol,
        "axis_identity": group.axis_identity,
        "observation_count": int(group.values.shape[0]),
        "removed_position_mapping": _group_position_mapping(group, ablation),
        "channels": analysis["statistics"],
        "pearson_correlation_matrix": analysis["pearson"],
        "spearman_correlation_matrix": analysis["spearman"],
        "paired_valid_observation_counts": analysis["paired_valid_observation_counts"],
        "focused_correlations": analysis["focused"],
    }


def _group_prefix(group: dict[str, object]) -> dict[str, object]:
    return {
        "group_id": group["group_id"],
        "group_kind": group["group_kind"],
        "symbol": group["symbol"],
        "axis_identity": group["axis_identity"],
    }


def _stats_frame(groups: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {**_group_prefix(group), **channel}
            for group in groups
            for channel in group["channels"]
        ]
    )


def _correlation_frame(groups: list[dict[str, object]], method: str) -> pl.DataFrame:
    matrix_field = f"{method}_correlation_matrix"
    rows = []
    for group in groups:
        channels = group["channels"]
        matrix = group[matrix_field]
        counts = group["paired_valid_observation_counts"]
        for left, left_channel in enumerate(channels):
            for right, right_channel in enumerate(channels):
                rows.append(
                    {
                        **_group_prefix(group),
                        "left_index": left,
                        "left_name": left_channel["name"],
                        "right_index": right,
                        "right_name": right_channel["name"],
                        "correlation": matrix[left][right],
                        "paired_valid_observation_count": counts[left][right],
                    }
                )
    return pl.DataFrame(rows, infer_schema_length=None)


def _focused_frame(groups: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {**_group_prefix(group), **row}
            for group in groups
            for row in group["focused_correlations"]
        ],
        infer_schema_length=None,
    )


def run_training_slow_audit(store: Path, output_dir: Path) -> Path:
    feature_identity = _feature_store_identity(store)
    feature_ablation = resolve_feature_ablation_for_store(store, "drop_slow_low_prior")
    inputs, sample_contract = _training_groups(store)
    groups = [_analyze_group(group, feature_ablation) for group in inputs]

    output_dir.mkdir(parents=True, exist_ok=False)
    stats_path = output_dir / STATS_CSV
    pearson_path = output_dir / PEARSON_CSV
    spearman_path = output_dir / SPEARMAN_CSV
    focused_path = output_dir / FOCUSED_CSV
    _atomic_write_csv(stats_path, _stats_frame(groups))
    _atomic_write_csv(pearson_path, _correlation_frame(groups, "pearson"))
    _atomic_write_csv(spearman_path, _correlation_frame(groups, "spearman"))
    _atomic_write_csv(focused_path, _focused_frame(groups))

    audit = {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        "validation_only_experiment": True,
        "test_metrics_accessed": False,
        "feature_store": feature_identity,
        "feature_manifest_sha256": feature_identity["manifest_sha256"],
        "context_ablation": FROZEN_CONTEXT_ABLATION,
        "context_ablation_metadata": resolve_context_ablation_for_store(
            store, FROZEN_CONTEXT_ABLATION
        ).metadata(),
        "feature_ablation": feature_ablation.metadata(),
        "canonical_axes": {
            EQUITY_LOCAL_AXIS: [
                {"index": index, "name": name}
                for index, name in enumerate(SLOW_CHANNELS)
            ],
            GLOBAL_AXIS: [
                {"index": index, "name": name}
                for index, name in enumerate(GLOBAL_SLOW_CHANNELS)
            ],
        },
        "group_order": [group["group_id"] for group in groups],
        "sample_contract": sample_contract,
        "groups": groups,
        "output_sha256": {
            STATS_CSV: _sha256(stats_path),
            PEARSON_CSV: _sha256(pearson_path),
            SPEARMAN_CSV: _sha256(spearman_path),
            FOCUSED_CSV: _sha256(focused_path),
        },
    }
    audit_path = output_dir / AUDIT_JSON
    _atomic_write_json(audit_path, audit)
    return audit_path


def _portable_feature_identity_matches(
    recorded: object, expected: dict[str, object]
) -> bool:
    if not isinstance(recorded, dict):
        return False
    left = dict(recorded)
    right = dict(expected)
    left.pop("resolved_path", None)
    right.pop("resolved_path", None)
    return left == right


def _reject_test_metadata(payload: object) -> None:
    if isinstance(payload, dict):
        for raw_key, value in payload.items():
            key = str(raw_key).lower().replace("-", "_")
            if key not in {
                "test_start",
                "test_end",
                "test_metrics_accessed",
            } and "test" in key.split("_"):
                raise ValueError(
                    f"Training-only slow-feature audit contains forbidden field: {raw_key}"
                )
            _reject_test_metadata(value)
    elif isinstance(payload, list):
        for value in payload:
            _reject_test_metadata(value)


def _validate_matrix(value: object, field: str) -> list[list[object]]:
    if (
        not isinstance(value, list)
        or len(value) != 32
        or any(not isinstance(row, list) or len(row) != 32 for row in value)
    ):
        raise ValueError(f"Training-only slow-feature audit has malformed {field}")
    return value


def _validate_group(
    recorded: object,
    expected: SlowAuditGroupInput,
    ablation: ResolvedFeatureAblation,
) -> None:
    if not isinstance(recorded, dict):
        raise ValueError("Training-only slow-feature audit group is malformed")
    expected_prefix = {
        "group_id": expected.group_id,
        "group_kind": expected.group_kind,
        "symbol": expected.symbol,
        "axis_identity": expected.axis_identity,
        "observation_count": int(expected.values.shape[0]),
        "removed_position_mapping": _group_position_mapping(expected, ablation),
    }
    if any(recorded.get(field) != value for field, value in expected_prefix.items()):
        raise ValueError(
            f"Training-only slow-feature audit group drifted: {expected.group_id}"
        )
    channels = recorded.get("channels")
    if not isinstance(channels, list) or len(channels) != 32:
        raise ValueError("Training-only slow-feature audit channels are malformed")
    statuses = _channel_statuses(
        32,
        applicable=expected.applicable,
        context_disabled_indices=expected.context_disabled_indices,
        removed_indices=ablation.slow_indices,
    )
    for index, (channel, name, status) in enumerate(
        zip(channels, expected.channel_names, statuses, strict=True)
    ):
        expected_static = {
            "index": index,
            "name": name,
            "status": status,
            "is_applicable": status != "inapplicable",
            "is_context_disabled": status == "context_disabled",
            "is_removed_position": index in ablation.slow_indices,
        }
        if not isinstance(channel, dict) or any(
            channel.get(field) != value for field, value in expected_static.items()
        ):
            raise ValueError(
                f"Training-only slow-feature audit active set drifted: {expected.group_id}/{index}"
            )
        count = channel.get("valid_observation_count")
        fraction = channel.get("finite_fraction")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count > expected.values.shape[0]
            or not isinstance(fraction, (int, float))
            or not np.isclose(
                float(fraction),
                count / expected.values.shape[0],
                rtol=0.0,
                atol=1e-15,
            )
            or (status in {"inapplicable", "context_disabled"} and count != 0)
        ):
            raise ValueError(
                f"Training-only slow-feature audit counts are invalid: {expected.group_id}/{index}"
            )

    pearson = _validate_matrix(
        recorded.get("pearson_correlation_matrix"), "Pearson matrix"
    )
    spearman = _validate_matrix(
        recorded.get("spearman_correlation_matrix"), "Spearman matrix"
    )
    counts = _validate_matrix(
        recorded.get("paired_valid_observation_counts"), "pair-count matrix"
    )
    for index, channel in enumerate(channels):
        if counts[index][index] != channel["valid_observation_count"]:
            raise ValueError("Training-only slow-feature audit pair counts drifted")
    expected_focused = _focused_correlations(
        expected.channel_names,
        statuses,
        np.asarray(
            [[np.nan if value is None else value for value in row] for row in pearson],
            dtype=np.float64,
        ),
        np.asarray(
            [[np.nan if value is None else value for value in row] for row in spearman],
            dtype=np.float64,
        ),
        np.asarray(counts, dtype=np.int64),
    )
    if recorded.get("focused_correlations") != expected_focused:
        raise ValueError(
            f"Training-only slow-feature audit focused candidates drifted: {expected.group_id}"
        )


def validate_training_slow_audit(
    audit_path: Path, feature_identity: dict[str, object]
) -> dict[str, object]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    _reject_test_metadata(audit)
    store = Path(str(feature_identity["resolved_path"]))
    expected_ablation = resolve_feature_ablation_for_store(store, "drop_slow_low_prior")
    expected_context = resolve_context_ablation_for_store(
        store, FROZEN_CONTEXT_ABLATION
    ).metadata()
    expected_axes = {
        EQUITY_LOCAL_AXIS: [
            {"index": index, "name": name} for index, name in enumerate(SLOW_CHANNELS)
        ],
        GLOBAL_AXIS: [
            {"index": index, "name": name}
            for index, name in enumerate(GLOBAL_SLOW_CHANNELS)
        ],
    }
    if (
        audit.get("audit_name") != AUDIT_NAME
        or audit.get("audit_version") != AUDIT_VERSION
        or audit.get("test_metrics_accessed") is not False
        or audit.get("validation_only_experiment") is not True
        or audit.get("feature_manifest_sha256")
        != feature_identity.get("manifest_sha256")
        or not _portable_feature_identity_matches(
            audit.get("feature_store"), feature_identity
        )
        or audit.get("context_ablation") != FROZEN_CONTEXT_ABLATION
        or audit.get("context_ablation_metadata") != expected_context
        or audit.get("feature_ablation") != expected_ablation.metadata()
        or audit.get("canonical_axes") != expected_axes
    ):
        raise ValueError("Training-only slow-feature audit identity is incompatible")

    expected_groups, expected_sample = _training_groups(store)
    groups = audit.get("groups")
    expected_order = [group.group_id for group in expected_groups]
    if (
        not isinstance(groups, list)
        or audit.get("group_order") != expected_order
        or [group.get("group_id") for group in groups if isinstance(group, dict)]
        != expected_order
        or audit.get("sample_contract") != expected_sample
    ):
        raise ValueError(
            "Training-only slow-feature audit used the wrong sample or groups"
        )
    for recorded, expected in zip(groups, expected_groups, strict=True):
        _validate_group(recorded, expected, expected_ablation)

    output_hashes = audit.get("output_sha256")
    if not isinstance(output_hashes, dict) or tuple(output_hashes) != (
        STATS_CSV,
        PEARSON_CSV,
        SPEARMAN_CSV,
        FOCUSED_CSV,
    ):
        raise ValueError("Training-only slow-feature audit lacks output hashes")
    for name in (STATS_CSV, PEARSON_CSV, SPEARMAN_CSV, FOCUSED_CSV):
        path = audit_path.parent / name
        if not path.is_file() or output_hashes.get(name) != _sha256(path):
            raise ValueError(f"Training-only slow-feature audit output changed: {name}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = resolve_feature_store().resolve()
    path = run_training_slow_audit(store, args.output_dir.resolve())
    print(f"Wrote training-only slow-feature audit: {path}")


if __name__ == "__main__":
    main()
