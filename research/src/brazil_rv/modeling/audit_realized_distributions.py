from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.preprocessing.audit import DYNAMIC_BOUNDS, SLOW_BOUNDS
from brazil_rv.preprocessing.contract import (
    DECISION_CONTEXT_INDICES,
    DECISION_EQUITY_INDICES,
    DECISION_GLOBAL_INDICES,
    DYNAMIC_CHANNELS,
    EQUITY_PEER_CHANNELS,
    EQUITY_PEER_VALID_CHANNELS,
    GLOBAL_SLOW_CHANNELS,
    SLOW_CHANNELS,
)

from .context_ablation import resolve_context_ablation_for_store
from .contract import (
    GLOBAL_CONTEXT_SYMBOLS,
    GLOBAL_WINDOW_MINUTES,
    HORIZONS,
    LOCAL_CONTEXT_SYMBOLS,
    TRAIN_END,
    TRAIN_START,
)
from .data import resolve_feature_store, select_sample_split, validate_feature_store
from .feature_ablation import resolve_feature_ablation_for_store
from .stage2_context_ablation import _feature_store_identity

AUDIT_NAME = "realized_feature_target_distributions"
AUDIT_VERSION = 1
AUDIT_JSON = "realized_distribution_audit.json"
FEATURE_PARQUET = "feature_distributions.parquet"
TARGET_PARQUET = "target_distributions.parquet"
FROZEN_CONTEXT_ABLATION = "drop_win_and_global_non_rates"
FROZEN_FEATURE_ABLATION = "none"
RETAINED_MACRO_SYMBOLS = (
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
    "ZT.v.0",
    "ZN.v.0",
)
QUANTILE_PROBABILITIES = (0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999)
DEFAULT_SAMPLE_CAPACITY = 16_384
DATE_CHUNK = 4


@dataclass
class DeterministicSystematicSample:
    capacity: int
    seen: int = 0
    stride: int = 1
    positions: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    values: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).ravel()
        if not flat.size:
            return
        positions = np.arange(self.seen, self.seen + flat.size, dtype=np.int64)
        selected = positions % self.stride == 0
        if selected.any():
            self.positions = np.concatenate((self.positions, positions[selected]))
            self.values = np.concatenate((self.values, flat[selected]))
        self.seen += int(flat.size)
        while self.values.size > self.capacity:
            self.stride *= 2
            keep = self.positions % self.stride == 0
            self.positions = self.positions[keep]
            self.values = self.values[keep]


@dataclass
class StreamingDistribution:
    sample_capacity: int = DEFAULT_SAMPLE_CAPACITY
    valid_count: int = 0
    finite_count: int = 0
    nonfinite_count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    zero_count: int = 0
    lower_boundary_count: int = 0
    upper_boundary_count: int = 0
    sample: DeterministicSystematicSample = field(init=False)

    def __post_init__(self) -> None:
        if self.sample_capacity <= 0:
            raise ValueError("Quantile sample capacity must be positive")
        self.sample = DeterministicSystematicSample(self.sample_capacity)

    def update(
        self,
        values: np.ndarray,
        valid: np.ndarray | None = None,
        clip_bounds: tuple[float, float] | None = None,
    ) -> None:
        array = np.asarray(values)
        if valid is None:
            selected = array.ravel()
        else:
            mask = np.broadcast_to(np.asarray(valid, dtype=bool), array.shape)
            selected = array[mask]
        self.valid_count += int(selected.size)
        finite_mask = np.isfinite(selected)
        self.nonfinite_count += int((~finite_mask).sum())
        finite = np.asarray(selected[finite_mask], dtype=np.float64)
        if not finite.size:
            return
        batch_count = int(finite.size)
        batch_mean = float(finite.mean(dtype=np.float64))
        centered = finite - batch_mean
        batch_m2 = float(np.dot(centered, centered))
        if self.finite_count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            combined = self.finite_count + batch_count
            self.mean += delta * batch_count / combined
            self.m2 += (
                batch_m2 + delta * delta * self.finite_count * batch_count / combined
            )
        self.finite_count += batch_count
        self.minimum = min(self.minimum, float(finite.min()))
        self.maximum = max(self.maximum, float(finite.max()))
        self.zero_count += int(np.count_nonzero(finite == 0.0))
        if clip_bounds is not None:
            lower, upper = clip_bounds
            self.lower_boundary_count += int(np.count_nonzero(finite == lower))
            self.upper_boundary_count += int(np.count_nonzero(finite == upper))
        self.sample.update(finite)

    def row(
        self,
        *,
        table: str,
        family: str,
        symbol: str | None,
        channel_index: int | None,
        channel_name: str,
        status: str,
        clip_bounds: tuple[float, float] | None,
    ) -> dict[str, object]:
        sampled = self.sample.values
        quantiles = (
            np.quantile(sampled, QUANTILE_PROBABILITIES, method="linear")
            if sampled.size
            else np.full(len(QUANTILE_PROBABILITIES), np.nan)
        )
        median = float(quantiles[4]) if sampled.size else None
        mad = (
            float(np.median(np.abs(sampled - median)))
            if sampled.size and median is not None
            else None
        )
        denominator = self.finite_count
        result: dict[str, object] = {
            "table": table,
            "family": family,
            "symbol": symbol,
            "channel_index": channel_index,
            "channel_name": channel_name,
            "status": status,
            "valid_count": self.valid_count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "mean": self.mean if denominator else None,
            "standard_deviation": (
                math.sqrt(max(self.m2 / denominator, 0.0)) if denominator else None
            ),
            "median": median,
            "mad": mad,
            "minimum": self.minimum if denominator else None,
            "maximum": self.maximum if denominator else None,
            "zero_fraction": self.zero_count / denominator if denominator else None,
            "clip_lower": clip_bounds[0] if clip_bounds is not None else None,
            "clip_upper": clip_bounds[1] if clip_bounds is not None else None,
            "lower_boundary_fraction": (
                self.lower_boundary_count / denominator
                if denominator and clip_bounds is not None
                else None
            ),
            "upper_boundary_fraction": (
                self.upper_boundary_count / denominator
                if denominator and clip_bounds is not None
                else None
            ),
            "clipping_boundary_fraction": (
                (self.lower_boundary_count + self.upper_boundary_count) / denominator
                if denominator and clip_bounds is not None
                else None
            ),
            "quantile_sample_count": int(sampled.size),
            "quantile_sample_capacity": self.sample_capacity,
            "quantile_sample_stride": self.sample.stride,
            "quantile_sample_method": "deterministic_systematic_valid_ordinal_v1",
        }
        for probability, value in zip(QUANTILE_PROBABILITIES, quantiles, strict=True):
            suffix = str(probability).replace(".", "_")
            result[f"quantile_{suffix}"] = float(value) if sampled.size else None
        return result


@dataclass
class AuditedChannel:
    table: str
    family: str
    symbol: str | None
    channel_index: int | None
    channel_name: str
    status: str
    clip_bounds: tuple[float, float] | None
    distribution: StreamingDistribution

    def update(self, values: np.ndarray, valid: np.ndarray | None = None) -> None:
        self.distribution.update(values, valid, self.clip_bounds)

    def row(self) -> dict[str, object]:
        return self.distribution.row(
            table=self.table,
            family=self.family,
            symbol=self.symbol,
            channel_index=self.channel_index,
            channel_name=self.channel_name,
            status=self.status,
            clip_bounds=self.clip_bounds,
        )


@dataclass
class ViolationLog:
    counts: dict[tuple[str, str, str], int] = field(default_factory=dict)

    def add(self, code: str, location: str, message: str, count: int = 1) -> None:
        if count > 0:
            key = (code, location, message)
            self.counts[key] = self.counts.get(key, 0) + int(count)

    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "code": code,
                "location": location,
                "count": count,
                "message": message,
            }
            for (code, location, message), count in sorted(self.counts.items())
        ]


@dataclass
class TargetStructureDiagnostics:
    label_count: dict[int, int] = field(
        default_factory=lambda: {horizon: 0 for horizon in HORIZONS}
    )
    eligible_count: dict[int, int] = field(
        default_factory=lambda: {horizon: 0 for horizon in HORIZONS}
    )
    cross_section_count: dict[int, int] = field(
        default_factory=lambda: {horizon: 0 for horizon in HORIZONS}
    )
    cross_sections_with_ties: dict[int, int] = field(
        default_factory=lambda: {horizon: 0 for horizon in HORIZONS}
    )
    labels_in_ties: dict[int, int] = field(
        default_factory=lambda: {horizon: 0 for horizon in HORIZONS}
    )
    extreme_labels: dict[int, int] = field(
        default_factory=lambda: {horizon: 0 for horizon in HORIZONS}
    )

    def update(
        self,
        targets: np.ndarray,
        label_mask: np.ndarray,
        eligible: np.ndarray,
        cross_mean: dict[int, StreamingDistribution],
        cross_std: dict[int, StreamingDistribution],
        violations: ViolationLog,
    ) -> None:
        if targets.shape != label_mask.shape or targets.ndim != 4:
            raise ValueError(
                "Target diagnostic arrays must have [date,equity,decision,horizon]"
            )
        if eligible.shape != targets.shape[:3]:
            raise ValueError("Target eligible mask has the wrong axes")
        for horizon_index, horizon in enumerate(HORIZONS):
            mask = label_mask[..., horizon_index]
            self.label_count[horizon] += int(mask.sum())
            self.eligible_count[horizon] += int(eligible.sum())
            for date_index in range(targets.shape[0]):
                for decision_index in range(targets.shape[2]):
                    valid = mask[date_index, :, decision_index]
                    if not valid.any():
                        continue
                    values = np.asarray(
                        targets[date_index, valid, decision_index, horizon_index],
                        dtype=np.float64,
                    )
                    self.cross_section_count[horizon] += 1
                    finite = values[np.isfinite(values)]
                    if finite.size != values.size:
                        continue
                    cross_mean[horizon].update(np.asarray([finite.mean()]))
                    cross_std[horizon].update(np.asarray([finite.std(ddof=0)]))
                    outside = (finite <= -1.0) | (finite >= 1.0)
                    violations.add(
                        "target_outside_open_interval",
                        f"target:{horizon}m",
                        "Valid targets must lie strictly inside (-1, 1).",
                        int(outside.sum()),
                    )
                    if abs(float(finite.mean())) > 2e-6:
                        violations.add(
                            "target_cross_section_not_centered",
                            f"target:{horizon}m",
                            "A valid target cross-section is not centered at zero.",
                        )
                    _, counts = np.unique(finite, return_counts=True)
                    tied = counts[counts > 1]
                    if tied.size:
                        self.cross_sections_with_ties[horizon] += 1
                        self.labels_in_ties[horizon] += int(tied.sum())
                    count = finite.size
                    lower = np.float32(1.0 / count - 1.0)
                    upper = np.float32(1.0 - 1.0 / count)
                    self.extreme_labels[horizon] += int(
                        np.count_nonzero(
                            (finite == float(lower)) | (finite == float(upper))
                        )
                    )

    def rows(self) -> dict[str, dict[str, object]]:
        result = {}
        for horizon in HORIZONS:
            labels = self.label_count[horizon]
            cross_sections = self.cross_section_count[horizon]
            eligible = self.eligible_count[horizon]
            result[f"{horizon}m"] = {
                "label_count": labels,
                "label_coverage": labels / eligible if eligible else None,
                "cross_section_count": cross_sections,
                "cross_sections_with_ties": self.cross_sections_with_ties[horizon],
                "cross_section_tie_prevalence": (
                    self.cross_sections_with_ties[horizon] / cross_sections
                    if cross_sections
                    else None
                ),
                "labels_in_ties": self.labels_in_ties[horizon],
                "label_tie_prevalence": (
                    self.labels_in_ties[horizon] / labels if labels else None
                ),
                "extreme_attainable_rank_count": self.extreme_labels[horizon],
                "extreme_attainable_rank_fraction": (
                    self.extreme_labels[horizon] / labels if labels else None
                ),
            }
        return result


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


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _indices(value: object, field_name: str) -> set[int]:
    if not isinstance(value, list) or any(
        not isinstance(index, int) or isinstance(index, bool) for index in value
    ):
        raise ValueError(f"Feature schema has invalid index list: {field_name}")
    result = set(value)
    if len(result) != len(value):
        raise ValueError(f"Feature schema has duplicate indices: {field_name}")
    return result


def _local_slow_structural_indices(symbol: str, schema: dict[str, object]) -> set[int]:
    family = schema.get("family_inapplicable_zero_fields")
    if not isinstance(family, dict):
        raise ValueError("Feature schema is missing family applicability")
    result = _indices(family.get("all_context"), "all_context")
    if symbol == "WDO$":
        result |= _indices(family.get("WIN_WDO"), "WIN_WDO")
    elif symbol in {"DI1F27", "DI1F28", "DI1F29", "DI1F31"}:
        result |= _indices(family.get("fixed_maturity_DI"), "fixed_maturity_DI")
    elif symbol == "DI1$N":
        result |= _indices(family.get("DI1$N"), "DI1$N")
    else:
        raise ValueError(f"Unknown retained local macro symbol: {symbol}")
    return result


def _channel(
    output: list[AuditedChannel],
    *,
    table: str,
    family: str,
    symbol: str | None,
    channel_index: int | None,
    channel_name: str,
    status: str,
    clip_bounds: tuple[float, float] | None,
    sample_capacity: int,
) -> AuditedChannel:
    channel = AuditedChannel(
        table,
        family,
        symbol,
        channel_index,
        channel_name,
        status,
        clip_bounds,
        StreamingDistribution(sample_capacity),
    )
    output.append(channel)
    return channel


def _add_structural_violation(
    violations: ViolationLog,
    values: np.ndarray,
    valid: np.ndarray,
    location: str,
) -> None:
    selected = np.asarray(values)[np.broadcast_to(valid, np.asarray(values).shape)]
    violations.add(
        "unexpected_nonzero_structural_channel",
        location,
        "A schema-declared structural channel contains a nonzero value.",
        int(np.count_nonzero(selected != 0.0)),
    )


def _add_nonbinary_violation(
    violations: ViolationLog,
    values: np.ndarray,
    valid: np.ndarray,
    location: str,
) -> None:
    selected = np.asarray(values)[np.broadcast_to(valid, np.asarray(values).shape)]
    violations.add(
        "nonbinary_mask_channel",
        location,
        "A stored mask/observed channel is not exactly zero or one.",
        int(np.count_nonzero((selected != 0.0) & (selected != 1.0))),
    )


def _global_consumed_mask(
    readiness: np.ndarray,
) -> np.ndarray:
    consumed = np.zeros((readiness.shape[0], max(DECISION_GLOBAL_INDICES)), dtype=bool)
    for decision_index, cutoff in enumerate(DECISION_GLOBAL_INDICES):
        ready = readiness[:, decision_index]
        consumed[ready, cutoff - GLOBAL_WINDOW_MINUTES : cutoff] = True
    return consumed


def _warnings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    warnings: list[dict[str, object]] = []
    for row in rows:
        if row["status"] not in {"observed_numerical", "valid_numerical"}:
            continue
        location = f"{row['family']}:{row['symbol']}:{row['channel_name']}"
        boundary = row.get("clipping_boundary_fraction")
        zero = row.get("zero_fraction")
        std = row.get("standard_deviation")
        if isinstance(boundary, (int, float)) and boundary >= 0.01:
            warnings.append(
                {
                    "code": "material_clipping_boundary_fraction",
                    "location": location,
                    "value": boundary,
                    "message": "At least one percent of finite values are at an explicit clip boundary.",
                }
            )
        if isinstance(zero, (int, float)) and zero >= 0.99:
            warnings.append(
                {
                    "code": "very_high_observed_zero_fraction",
                    "location": location,
                    "value": zero,
                    "message": "At least 99 percent of validity-conditioned values are numerical zero.",
                }
            )
        if std == 0.0 and int(row["finite_count"]) > 0:
            warnings.append(
                {
                    "code": "constant_valid_channel",
                    "location": location,
                    "value": 0.0,
                    "message": "The validity-conditioned channel is constant.",
                }
            )
        elif isinstance(std, (int, float)) and 0.0 < std <= 1e-4:
            warnings.append(
                {
                    "code": "very_low_valid_dispersion",
                    "location": location,
                    "value": std,
                    "message": (
                        "Validity-conditioned standard deviation is at most 1e-4."
                    ),
                }
            )
        elif isinstance(std, (int, float)) and std >= 10.0:
            warnings.append(
                {
                    "code": "large_valid_dispersion",
                    "location": location,
                    "value": std,
                    "message": (
                        "Validity-conditioned standard deviation is at least 10."
                    ),
                }
            )
    return warnings


def _split_identity(training: pl.DataFrame) -> dict[str, object]:
    fields = np.column_stack(
        [
            training[column].to_numpy().astype("<i8", copy=False)
            for column in ("sample_id", "date_idx", "decision_idx")
        ]
    )
    return {
        "split": "train",
        "start": str(TRAIN_START),
        "end": str(TRAIN_END),
        "date_count": int(training["trade_date"].n_unique()),
        "sample_count": training.height,
        "sample_identity_fields": ["sample_id", "date_idx", "decision_idx"],
        "sample_identity_sha256": hashlib.sha256(fields.tobytes()).hexdigest(),
        "excluded_splits": ["embargo_1", "validation", "embargo_2", "test"],
    }


def run_realized_distribution_audit(
    store: Path,
    output_dir: Path,
    *,
    sample_capacity: int = DEFAULT_SAMPLE_CAPACITY,
) -> Path:
    store = store.resolve()
    output_dir = output_dir.resolve()
    if sample_capacity <= 0:
        raise ValueError("Quantile sample capacity must be positive")
    if output_dir == store or output_dir.is_relative_to(store):
        raise ValueError("Audit outputs must be outside the immutable feature store")
    sample_index = validate_feature_store(store)
    training = select_sample_split(sample_index, "train").sort("sample_id")
    if (
        training.is_empty()
        or training["trade_date"].min() != TRAIN_START
        or training["trade_date"].max() != TRAIN_END
    ):
        raise ValueError(
            "Realized-distribution audit did not resolve the training split"
        )
    date_indices = np.unique(training["date_idx"].to_numpy()).astype(np.int64)
    context_ablation = resolve_context_ablation_for_store(
        store, FROZEN_CONTEXT_ABLATION
    )
    feature_ablation = resolve_feature_ablation_for_store(
        store, FROZEN_FEATURE_ABLATION
    )
    retained_local = tuple(
        symbol
        for slot, symbol in enumerate(LOCAL_CONTEXT_SYMBOLS)
        if slot not in context_ablation.local_slots
    )
    retained_global = tuple(
        symbol
        for slot, symbol in enumerate(GLOBAL_CONTEXT_SYMBOLS)
        if slot not in context_ablation.global_slots
    )
    if (*retained_local, *retained_global) != RETAINED_MACRO_SYMBOLS:
        raise ValueError("Effective context ablation does not match the incumbent core")
    schema = json.loads((store / "feature_schema.json").read_text(encoding="utf-8"))
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))

    arrays = {
        name: np.load(store / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "equity_features.npy",
            "equity_slow.npy",
            "equity_membership.npy",
            "equity_data_ready.npy",
            "context_features.npy",
            "context_slow.npy",
            "context_data_ready.npy",
            "global_features.npy",
            "global_slow.npy",
            "global_data_ready.npy",
            "equity_peer_features.npy",
            "equity_peer_valid.npy",
            "targets.npy",
            "raw_returns.npy",
            "label_mask.npy",
            "horizon_mask.npy",
        )
    }
    feature_channels: list[AuditedChannel] = []
    target_channels: list[AuditedChannel] = []
    violations = ViolationLog()

    equity_dynamic = [
        _channel(
            feature_channels,
            table="feature",
            family="equity_dynamic",
            symbol=None,
            channel_index=index,
            channel_name=name,
            status="mask" if index == 5 else "observed_numerical",
            clip_bounds=DYNAMIC_BOUNDS[index],
            sample_capacity=sample_capacity,
        )
        for index, name in enumerate(DYNAMIC_CHANNELS)
    ]
    local_dynamic = {
        symbol: [
            _channel(
                feature_channels,
                table="feature",
                family="macro_dynamic",
                symbol=symbol,
                channel_index=index,
                channel_name=name,
                status=(
                    "structural_zero"
                    if index >= 16
                    else "mask"
                    if index == 5
                    else "observed_numerical"
                ),
                clip_bounds=DYNAMIC_BOUNDS[index],
                sample_capacity=sample_capacity,
            )
            for index, name in enumerate(DYNAMIC_CHANNELS)
        ]
        for symbol in retained_local
    }
    global_dynamic = {
        symbol: [
            _channel(
                feature_channels,
                table="feature",
                family="macro_dynamic",
                symbol=symbol,
                channel_index=index,
                channel_name=name,
                status=(
                    "structural_zero"
                    if index >= 16
                    else "mask"
                    if index == 5
                    else "observed_numerical"
                ),
                clip_bounds=DYNAMIC_BOUNDS[index],
                sample_capacity=sample_capacity,
            )
            for index, name in enumerate(DYNAMIC_CHANNELS)
        ]
        for symbol in retained_global
    }

    equity_structural = _indices(
        schema["family_inapplicable_zero_fields"]["equity"], "equity"
    )
    equity_slow = [
        _channel(
            feature_channels,
            table="feature",
            family="equity_slow",
            symbol=None,
            channel_index=index,
            channel_name=name,
            status=(
                "context_disabled"
                if index in context_ablation.equity_slow_indices
                else "structural_zero"
                if index in equity_structural
                else "valid_numerical"
            ),
            clip_bounds=SLOW_BOUNDS[index],
            sample_capacity=sample_capacity,
        )
        for index, name in enumerate(SLOW_CHANNELS)
    ]
    local_slow: dict[str, list[AuditedChannel]] = {}
    for symbol in retained_local:
        structural = _local_slow_structural_indices(symbol, schema)
        local_slow[symbol] = [
            _channel(
                feature_channels,
                table="feature",
                family="macro_slow",
                symbol=symbol,
                channel_index=index,
                channel_name=name,
                status="structural_zero" if index in structural else "valid_numerical",
                clip_bounds=SLOW_BOUNDS[index],
                sample_capacity=sample_capacity,
            )
            for index, name in enumerate(SLOW_CHANNELS)
        ]
    global_structural = _indices(schema.get("global_slow"), "global_slow")
    global_slow = {
        symbol: [
            _channel(
                feature_channels,
                table="feature",
                family="macro_slow",
                symbol=symbol,
                channel_index=index,
                channel_name=name,
                status=(
                    "structural_zero"
                    if index in global_structural
                    else "valid_numerical"
                ),
                clip_bounds=SLOW_BOUNDS[index],
                sample_capacity=sample_capacity,
            )
            for index, name in enumerate(GLOBAL_SLOW_CHANNELS)
        ]
        for symbol in retained_global
    }

    mask_channels = {
        name: _channel(
            feature_channels,
            table="feature",
            family="mask",
            symbol=symbol,
            channel_index=None,
            channel_name=name,
            status="mask",
            clip_bounds=(0.0, 1.0),
            sample_capacity=sample_capacity,
        )
        for name, symbol in (
            ("equity_membership", None),
            ("equity_data_ready", None),
            ("equity_active", None),
            ("equity_observed", None),
            *((f"{symbol}_data_ready", symbol) for symbol in retained_local),
            *((f"{symbol}_data_ready", symbol) for symbol in retained_global),
            *((f"{symbol}_observed", symbol) for symbol in RETAINED_MACRO_SYMBOLS),
        )
    }

    peer_numeric_bounds = (
        DYNAMIC_BOUNDS[7],
        DYNAMIC_BOUNDS[9],
        (-1.0, 1.0),
        (-1.0, 1.0),
        DYNAMIC_BOUNDS[7],
        DYNAMIC_BOUNDS[9],
    )
    peer_numeric = [
        _channel(
            feature_channels,
            table="feature",
            family=(
                "selected_peer_numerical" if index < 4 else "issuer_peer_numerical"
            ),
            symbol=None,
            channel_index=index,
            channel_name=name,
            status="valid_numerical",
            clip_bounds=peer_numeric_bounds[index],
            sample_capacity=sample_capacity,
        )
        for index, name in enumerate(EQUITY_PEER_CHANNELS)
    ]
    peer_validity = [
        _channel(
            feature_channels,
            table="feature",
            family=("selected_peer_validity" if index < 2 else "issuer_peer_validity"),
            symbol=None,
            channel_index=index,
            channel_name=name,
            status="mask",
            clip_bounds=(0.0, 1.0),
            sample_capacity=sample_capacity,
        )
        for index, name in enumerate(EQUITY_PEER_VALID_CHANNELS)
    ]

    target_values = {
        horizon: _channel(
            target_channels,
            table="target",
            family="target",
            symbol=None,
            channel_index=index,
            channel_name=f"target_{horizon}m",
            status="valid_numerical",
            clip_bounds=None,
            sample_capacity=sample_capacity,
        )
        for index, horizon in enumerate(HORIZONS)
    }
    raw_values = {
        horizon: _channel(
            target_channels,
            table="target",
            family="raw_return",
            symbol=None,
            channel_index=index,
            channel_name=f"raw_return_{horizon}m",
            status="valid_numerical",
            clip_bounds=None,
            sample_capacity=sample_capacity,
        )
        for index, horizon in enumerate(HORIZONS)
    }
    label_channels = {
        horizon: _channel(
            target_channels,
            table="target",
            family="mask",
            symbol=None,
            channel_index=index,
            channel_name=f"label_mask_{horizon}m",
            status="mask",
            clip_bounds=(0.0, 1.0),
            sample_capacity=sample_capacity,
        )
        for index, horizon in enumerate(HORIZONS)
    }
    horizon_channels = {
        horizon: _channel(
            target_channels,
            table="target",
            family="mask",
            symbol=None,
            channel_index=index,
            channel_name=f"horizon_mask_{horizon}m",
            status="mask",
            clip_bounds=(0.0, 1.0),
            sample_capacity=sample_capacity,
        )
        for index, horizon in enumerate(HORIZONS)
    }
    cross_mean = {
        horizon: StreamingDistribution(sample_capacity) for horizon in HORIZONS
    }
    cross_std = {
        horizon: StreamingDistribution(sample_capacity) for horizon in HORIZONS
    }
    target_structure = TargetStructureDiagnostics()

    local_slots = {
        symbol: LOCAL_CONTEXT_SYMBOLS.index(symbol) for symbol in retained_local
    }
    global_slots = {
        symbol: GLOBAL_CONTEXT_SYMBOLS.index(symbol) for symbol in retained_global
    }
    feature_visible = max(DECISION_EQUITY_INDICES)
    context_visible = max(DECISION_CONTEXT_INDICES)
    feature_to_valid = (0, 1, 0, 1, 2, 3)

    for chunk_start in range(0, date_indices.size, DATE_CHUNK):
        dates = date_indices[chunk_start : chunk_start + DATE_CHUNK]
        membership = np.asarray(arrays["equity_membership.npy"][dates], dtype=bool)
        ready = np.asarray(arrays["equity_data_ready.npy"][dates], dtype=bool)
        active = membership & ready
        mask_channels["equity_membership"].update(membership)
        mask_channels["equity_data_ready"].update(ready)
        mask_channels["equity_active"].update(active)

        dynamic = np.asarray(
            arrays["equity_features.npy"][dates, :, :feature_visible, :],
            dtype=np.float32,
        )
        active_grid = np.broadcast_to(active[:, :, None], dynamic.shape[:3])
        observed = dynamic[..., 5]
        _add_nonbinary_violation(
            violations, observed, active_grid, "equity_dynamic:observed"
        )
        observed_mask = active_grid & (observed == 1.0)
        mask_channels["equity_observed"].update(observed, active_grid)
        for index, channel in enumerate(equity_dynamic):
            channel.update(
                dynamic[..., index],
                active_grid if index == 5 else observed_mask,
            )

        slow = np.asarray(arrays["equity_slow.npy"][dates], dtype=np.float32)
        for index, channel in enumerate(equity_slow):
            if index in context_ablation.equity_slow_indices:
                channel.update(np.zeros_like(slow[..., index]), active)
            else:
                channel.update(slow[..., index], active)
            if index in equity_structural:
                _add_structural_violation(
                    violations,
                    slow[..., index],
                    active,
                    f"equity_slow:{SLOW_CHANNELS[index]}",
                )

        context_ready = np.asarray(arrays["context_data_ready.npy"][dates], dtype=bool)
        context_values = np.asarray(
            arrays["context_features.npy"][dates, :, :context_visible, :],
            dtype=np.float32,
        )
        context_slow_values = np.asarray(
            arrays["context_slow.npy"][dates], dtype=np.float32
        )
        for symbol, slot in local_slots.items():
            source_ready = context_ready[:, slot]
            mask_channels[f"{symbol}_data_ready"].update(source_ready)
            grid = np.broadcast_to(
                source_ready[:, None], context_values[:, slot].shape[:2]
            )
            source = context_values[:, slot]
            observed = source[..., 5]
            _add_nonbinary_violation(
                violations, observed, grid, f"macro_dynamic:{symbol}:observed"
            )
            observed_mask = grid & (observed == 1.0)
            mask_channels[f"{symbol}_observed"].update(observed, grid)
            for index, channel in enumerate(local_dynamic[symbol]):
                channel.update(
                    source[..., index],
                    grid if index == 5 or index >= 16 else observed_mask,
                )
                if index >= 16:
                    _add_structural_violation(
                        violations,
                        source[..., index],
                        grid,
                        f"macro_dynamic:{symbol}:{DYNAMIC_CHANNELS[index]}",
                    )
            structural = _local_slow_structural_indices(symbol, schema)
            for index, channel in enumerate(local_slow[symbol]):
                channel.update(context_slow_values[:, slot, index], source_ready)
                if index in structural:
                    _add_structural_violation(
                        violations,
                        context_slow_values[:, slot, index],
                        source_ready,
                        f"macro_slow:{symbol}:{SLOW_CHANNELS[index]}",
                    )

        global_ready = np.asarray(arrays["global_data_ready.npy"][dates], dtype=bool)
        global_values_chunk = np.asarray(
            arrays["global_features.npy"][dates], dtype=np.float32
        )
        global_slow_chunk = np.asarray(
            arrays["global_slow.npy"][dates], dtype=np.float32
        )
        for symbol, slot in global_slots.items():
            source_ready = global_ready[:, slot]
            mask_channels[f"{symbol}_data_ready"].update(source_ready)
            consumed = _global_consumed_mask(source_ready)
            source = global_values_chunk[:, slot]
            observed = source[..., 5]
            _add_nonbinary_violation(
                violations, observed, consumed, f"macro_dynamic:{symbol}:observed"
            )
            observed_mask = consumed & (observed == 1.0)
            mask_channels[f"{symbol}_observed"].update(observed, consumed)
            for index, channel in enumerate(global_dynamic[symbol]):
                channel.update(
                    source[..., index],
                    consumed if index == 5 or index >= 16 else observed_mask,
                )
                if index >= 16:
                    _add_structural_violation(
                        violations,
                        source[..., index],
                        consumed,
                        f"macro_dynamic:{symbol}:{DYNAMIC_CHANNELS[index]}",
                    )
            for index, channel in enumerate(global_slow[symbol]):
                channel.update(global_slow_chunk[:, slot, :, index], source_ready)
                if index in global_structural:
                    _add_structural_violation(
                        violations,
                        global_slow_chunk[:, slot, :, index],
                        source_ready,
                        f"macro_slow:{symbol}:{GLOBAL_SLOW_CHANNELS[index]}",
                    )

        peer = np.asarray(
            arrays["equity_peer_features.npy"][dates, :, :feature_visible, :],
            dtype=np.float32,
        )
        peer_valid = np.asarray(
            arrays["equity_peer_valid.npy"][dates, :, :feature_visible, :],
            dtype=bool,
        )
        peer_active = np.broadcast_to(active[:, :, None, None], peer_valid.shape)
        violations.add(
            "peer_validity_on_inactive_equity",
            "peer_validity",
            "Peer validity is true for an inactive equity.",
            int(np.count_nonzero(peer_valid & ~peer_active)),
        )
        for index, channel in enumerate(peer_validity):
            channel.update(peer_valid[..., index], peer_active[..., index])
        for index, channel in enumerate(peer_numeric):
            validity = peer_valid[..., feature_to_valid[index]]
            channel.update(peer[..., index], validity)
            invalid = ~validity
            violations.add(
                "nonzero_invalid_peer_value",
                f"peer:{EQUITY_PEER_CHANNELS[index]}",
                "A peer numerical value is nonzero under its false validity mask.",
                int(np.count_nonzero(peer[..., index][invalid] != 0.0)),
            )

        targets = np.asarray(arrays["targets.npy"][dates], dtype=np.float32)
        raw = np.asarray(arrays["raw_returns.npy"][dates], dtype=np.float32)
        labels = np.asarray(arrays["label_mask.npy"][dates], dtype=bool)
        horizons = np.asarray(arrays["horizon_mask.npy"][dates], dtype=bool)
        violations.add(
            "nonzero_invalid_target",
            "targets",
            "A target is nonzero under a false label mask.",
            int(np.count_nonzero(targets[~labels] != 0.0)),
        )
        violations.add(
            "nonzero_invalid_raw_return",
            "raw_returns",
            "A raw return is nonzero under a false label mask.",
            int(np.count_nonzero(raw[~labels] != 0.0)),
        )
        full_observed = np.asarray(
            arrays["equity_features.npy"][dates, :, :, 5], dtype=bool
        )
        entry = full_observed[:, :, np.asarray(DECISION_EQUITY_INDICES)]
        exits = np.stack(
            [
                full_observed[
                    :,
                    :,
                    np.asarray(DECISION_EQUITY_INDICES) + horizon - 1,
                ]
                for horizon in HORIZONS
            ],
            axis=3,
        )
        required = (
            active[:, :, None, None]
            & entry[:, :, :, None]
            & exits
            & horizons[:, None, :, :]
        )
        violations.add(
            "label_mask_readiness_inconsistency",
            "label_mask",
            "A label is valid without membership, readiness, exact endpoints, or horizon readiness.",
            int(np.count_nonzero(labels & ~required)),
        )
        counts = labels.sum(axis=1)
        violations.add(
            "horizon_mask_count_inconsistency",
            "horizon_mask",
            "Horizon readiness disagrees with the minimum 30-label rule.",
            int(np.count_nonzero(horizons != (counts >= 30))),
        )
        eligible = np.broadcast_to(active[:, :, None], targets.shape[:3])
        for index, horizon in enumerate(HORIZONS):
            target_values[horizon].update(targets[..., index], labels[..., index])
            raw_values[horizon].update(raw[..., index], labels[..., index])
            label_channels[horizon].update(labels[..., index], eligible)
            horizon_channels[horizon].update(horizons[..., index])
        target_structure.update(
            targets,
            labels,
            eligible,
            cross_mean,
            cross_std,
            violations,
        )

    for horizon in HORIZONS:
        target_channels.append(
            AuditedChannel(
                "target",
                "target_cross_section",
                None,
                HORIZONS.index(horizon),
                f"cross_section_mean_{horizon}m",
                "valid_numerical",
                None,
                cross_mean[horizon],
            )
        )
        target_channels.append(
            AuditedChannel(
                "target",
                "target_cross_section",
                None,
                HORIZONS.index(horizon),
                f"cross_section_standard_deviation_{horizon}m",
                "valid_numerical",
                None,
                cross_std[horizon],
            )
        )

    feature_rows = [channel.row() for channel in feature_channels]
    target_rows = [channel.row() for channel in target_channels]
    for row in (*feature_rows, *target_rows):
        if int(row["nonfinite_count"]) > 0:
            violations.add(
                "nonfinite_inside_valid_observation",
                f"{row['family']}:{row['symbol']}:{row['channel_name']}",
                "A validity-conditioned numerical channel contains nonfinite values.",
                int(row["nonfinite_count"]),
            )
    target_by_horizon = target_structure.rows()
    for row in target_rows:
        if row["family"] == "target":
            horizon = row["channel_name"].removeprefix("target_")
            row.update(target_by_horizon[horizon])

    all_rows = [*feature_rows, *target_rows]
    warnings = _warnings(all_rows)
    hard_failures = violations.rows()
    output_dir.mkdir(parents=True, exist_ok=False)
    feature_path = output_dir / FEATURE_PARQUET
    target_path = output_dir / TARGET_PARQUET
    _atomic_write_parquet(
        feature_path, pl.DataFrame(feature_rows, infer_schema_length=None)
    )
    _atomic_write_parquet(
        target_path, pl.DataFrame(target_rows, infer_schema_length=None)
    )
    split_identity = _split_identity(training)
    store_identity = {
        **_feature_store_identity(store),
        "feature_schema_sha256": _sha256(store / "feature_schema.json"),
        "sample_index_sha256": _sha256(store / "sample_index.parquet"),
        "canonical_pointer_path": str(store.parent / "m1_features_canonical_path.txt"),
        "manifest_declared_outputs": manifest.get("outputs"),
    }
    audit = {
        "audit_name": AUDIT_NAME,
        "audit_version": AUDIT_VERSION,
        "summary_verdict": "fail" if hard_failures else "pass",
        "hard_failures": hard_failures,
        "warnings": warnings,
        "test_metrics_accessed": False,
        "training_only": True,
        "feature_store": store_identity,
        "split_identity": split_identity,
        "applied_ablations": {
            "context": context_ablation.metadata(),
            "feature": feature_ablation.metadata(),
        },
        "retained_macro_symbols": list(RETAINED_MACRO_SYMBOLS),
        "methodology": {
            "feature_occurrence_weighting": (
                "unique canonical training-date cells within the maximum causal "
                "decision-visible feature window; global cells are included once "
                "when consumed by at least one ready training decision"
            ),
            "numerical_conditioning": (
                "membership and data readiness for equities; source readiness for "
                "macro histories; observed==1 for ordinary dynamic numerical channels; "
                "governing validity masks for peer values; label masks for targets and raw returns"
            ),
            "structural_zero_policy": (
                "schema-declared structural channels are reported separately and "
                "any nonzero value is a hard failure"
            ),
            "exact_statistics": [
                "valid_count",
                "finite_count",
                "nonfinite_count",
                "mean",
                "standard_deviation",
                "minimum",
                "maximum",
                "zero_fraction",
                "explicit_clipping_boundary_fraction",
            ],
            "sampled_statistics": [
                "median",
                "mad",
                "quantiles",
            ],
            "quantile_sample": {
                "method": "deterministic_systematic_valid_ordinal_v1",
                "capacity_per_channel": sample_capacity,
                "probabilities": list(QUANTILE_PROBABILITIES),
                "reproducible": True,
            },
            "warning_thresholds": {
                "very_low_valid_standard_deviation": 1e-4,
                "large_valid_standard_deviation": 10.0,
            },
        },
        "target_diagnostics_by_horizon": target_by_horizon,
        "row_counts": {
            FEATURE_PARQUET: len(feature_rows),
            TARGET_PARQUET: len(target_rows),
        },
        "output_sha256": {
            FEATURE_PARQUET: _sha256(feature_path),
            TARGET_PARQUET: _sha256(target_path),
        },
    }
    audit_path = output_dir / AUDIT_JSON
    _atomic_write_json(audit_path, audit)
    return audit_path


def validate_realized_distribution_audit(
    audit_path: Path,
    expected_feature_store: dict[str, object] | None = None,
    *,
    require_pass: bool = True,
) -> dict[str, object]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("audit_name") != AUDIT_NAME
        or audit.get("audit_version") != AUDIT_VERSION
        or audit.get("training_only") is not True
        or audit.get("test_metrics_accessed") is not False
        or audit.get("retained_macro_symbols") != list(RETAINED_MACRO_SYMBOLS)
        or audit.get("applied_ablations", {}).get("context", {}).get("key")
        != FROZEN_CONTEXT_ABLATION
        or audit.get("applied_ablations", {}).get("feature", {}).get("key")
        != FROZEN_FEATURE_ABLATION
    ):
        raise ValueError("Realized-distribution audit identity is incompatible")
    split = audit.get("split_identity")
    if (
        not isinstance(split, dict)
        or split.get("split") != "train"
        or split.get("start") != str(TRAIN_START)
        or split.get("end") != str(TRAIN_END)
    ):
        raise ValueError("Realized-distribution audit used the wrong split")
    feature_store = audit.get("feature_store")
    if not isinstance(feature_store, dict):
        raise ValueError("Realized-distribution audit lacks feature-store identity")
    if expected_feature_store is not None:
        left = dict(feature_store)
        right = dict(expected_feature_store)
        for value in (left, right):
            value.pop("resolved_path", None)
            value.pop("feature_schema_sha256", None)
            value.pop("sample_index_sha256", None)
            value.pop("canonical_pointer_path", None)
            value.pop("manifest_declared_outputs", None)
        if left != right:
            raise ValueError("Realized-distribution audit identifies another store")
    hashes = audit.get("output_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("Realized-distribution audit lacks output hashes")
    row_counts = audit.get("row_counts")
    if not isinstance(row_counts, dict):
        raise ValueError("Realized-distribution audit lacks output row counts")
    required_columns = {
        "table",
        "family",
        "channel_name",
        "status",
        "valid_count",
        "finite_count",
        "nonfinite_count",
        "mean",
        "standard_deviation",
        "median",
        "mad",
        "minimum",
        "maximum",
        "zero_fraction",
        "clipping_boundary_fraction",
        "quantile_sample_method",
    }
    for name in (FEATURE_PARQUET, TARGET_PARQUET):
        path = audit_path.parent / name
        if not path.is_file() or hashes.get(name) != _sha256(path):
            raise ValueError(f"Realized-distribution audit output changed: {name}")
        frame = pl.read_parquet(path)
        if row_counts.get(name) != frame.height or not required_columns <= set(
            frame.columns
        ):
            raise ValueError(
                f"Realized-distribution audit output schema changed: {name}"
            )
    hard_failures = audit.get("hard_failures")
    warnings = audit.get("warnings")
    if not isinstance(hard_failures, list) or not isinstance(warnings, list):
        raise ValueError("Realized-distribution audit findings are malformed")
    expected_verdict = "fail" if hard_failures else "pass"
    if audit.get("summary_verdict") != expected_verdict:
        raise ValueError("Realized-distribution audit verdict is inconsistent")
    if require_pass and expected_verdict != "pass":
        raise ValueError("Realized-distribution audit contains hard failures")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--quantile-sample-capacity",
        type=int,
        default=DEFAULT_SAMPLE_CAPACITY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = resolve_feature_store().resolve()
    audit_path = run_realized_distribution_audit(
        store,
        args.output_dir.resolve(),
        sample_capacity=args.quantile_sample_capacity,
    )
    audit = validate_realized_distribution_audit(audit_path, require_pass=False)
    print(
        "Wrote training-only realized-distribution audit "
        f"({audit['summary_verdict']}): {audit_path}"
    )


if __name__ == "__main__":
    main()
