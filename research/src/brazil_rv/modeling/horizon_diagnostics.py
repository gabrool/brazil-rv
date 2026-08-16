from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray

from brazil_rv.preprocessing.contract import DYNAMIC_CHANNELS, EQUITY_SLOW_CHANNELS

from .contract import (
    ABSOLUTE_PATCH_COUNT,
    CONTEXT_COUNT,
    DYNAMIC_CHANNEL_COUNT,
    EQUITY_COUNT,
    HORIZONS,
    LOCAL_CONTEXT_COUNT,
    PATCH_MINUTES,
    PEER_STATE_WIDTH,
    TRAIN_END,
    VALIDATION_END,
    context_family_slots,
)
from .metrics import average_ranks


RIDGE_PENALTIES = tuple(float(value) for value in np.logspace(-6, 2, 9))
ANALYSIS_SEED = 20260815
BOOTSTRAP_SEED = 20260815
PERMUTATION_SEED = 20260815
FIXED_TARGET_BASIS = (
    np.asarray(
        (
            (1.0, 1.0, 1.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, -2.0),
        ),
        dtype=np.float64,
    )
    / np.asarray((math.sqrt(3.0), math.sqrt(2.0), math.sqrt(6.0)))[:, None]
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, Path)):
        return str(value)
    return value


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def atomic_csv(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_csv(temporary)
    os.replace(temporary, path)


def assert_analysis_rows(rows: pl.DataFrame, *, allow_validation: bool = False) -> None:
    if rows.is_empty():
        raise ValueError("Analysis window is empty")
    maximum = rows.get_column("trade_date").max()
    boundary = VALIDATION_END if allow_validation else TRAIN_END
    if maximum is None or maximum > boundary:
        raise ValueError(f"Analysis window ends after {boundary}: {maximum}")


@dataclass
class RidgeSufficientStatistics:
    feature_dim: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum_x = np.zeros(self.feature_dim, dtype=np.float64)
        self.sum_y = 0.0
        self.xtx = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
        self.xty = np.zeros(self.feature_dim, dtype=np.float64)
        self.yty = 0.0

    def update(
        self,
        features: NDArray[np.floating],
        targets: NDArray[np.floating],
        valid: NDArray[np.bool_],
    ) -> None:
        mask = np.asarray(valid, dtype=bool)
        x = np.asarray(features, dtype=np.float64)[mask]
        y = np.asarray(targets, dtype=np.float64)[mask]
        finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
        x = x[finite]
        y = y[finite]
        if not y.size:
            return
        self.count += y.size
        self.sum_x += x.sum(axis=0)
        self.sum_y += float(y.sum())
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.yty += float(y @ y)

    def merge(self, other: RidgeSufficientStatistics) -> None:
        if other.feature_dim != self.feature_dim:
            raise ValueError("Ridge statistic dimensions differ")
        self.count += other.count
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.xtx += other.xtx
        self.xty += other.xty
        self.yty += other.yty

    def solve(self, penalty: float) -> tuple[NDArray[np.float64], float]:
        if self.count <= self.feature_dim:
            raise ValueError("Insufficient ridge observations")
        mean_x = self.sum_x / self.count
        mean_y = self.sum_y / self.count
        covariance = self.xtx / self.count - np.outer(mean_x, mean_x)
        cross = self.xty / self.count - mean_x * mean_y
        coefficients = np.linalg.solve(
            covariance + penalty * np.eye(self.feature_dim),
            cross,
        )
        intercept = float(mean_y - mean_x @ coefficients)
        return coefficients, intercept


@dataclass
class PairMoments:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def update(
        self,
        left: NDArray[np.floating],
        right: NDArray[np.floating],
        valid: NDArray[np.bool_],
    ) -> None:
        mask = np.asarray(valid, dtype=bool)
        x = np.asarray(left, dtype=np.float64)[mask]
        y = np.asarray(right, dtype=np.float64)[mask]
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        self.count += x.size
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_xx += float(x @ x)
        self.sum_yy += float(y @ y)
        self.sum_xy += float(x @ y)

    def covariance(self) -> float:
        if self.count < 2:
            return float("nan")
        return self.sum_xy / self.count - self.sum_x * self.sum_y / self.count**2

    def correlation(self) -> float:
        if self.count < 2:
            return float("nan")
        var_x = self.sum_xx / self.count - (self.sum_x / self.count) ** 2
        var_y = self.sum_yy / self.count - (self.sum_y / self.count) ** 2
        denominator = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
        return self.covariance() / denominator if denominator > 0.0 else float("nan")


@dataclass
class VectorMoments:
    width: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(self.width, dtype=np.float64)
        self.outer = np.zeros((self.width, self.width), dtype=np.float64)

    def update(self, values: NDArray[np.floating]) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        matrix = matrix[np.isfinite(matrix).all(axis=1)]
        self.count += matrix.shape[0]
        self.sum += matrix.sum(axis=0)
        self.outer += matrix.T @ matrix

    def covariance(self) -> NDArray[np.float64]:
        mean = self.sum / self.count
        return self.outer / self.count - np.outer(mean, mean)


def _pair_row(
    scope: str,
    left: int,
    right: int,
    moments: PairMoments,
) -> dict[str, object]:
    return {
        "scope": scope,
        "left_horizon": HORIZONS[left],
        "right_horizon": HORIZONS[right],
        "valid_count": moments.count,
        "covariance": moments.covariance(),
        "correlation": moments.correlation(),
    }


def _mean_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def run_target_basis_audit(
    store: Path,
    train_rows: pl.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    assert_analysis_rows(train_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    raw_returns = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    label_mask = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    date_rows = train_rows.select("date_idx", "trade_date").unique().sort("trade_date")
    dates = date_rows.get_column("date_idx").to_numpy()
    trade_dates = date_rows.get_column("trade_date").to_list()
    pairs = tuple((left, right) for left in range(3) for right in range(left, 3))
    pooled = {pair: PairMoments() for pair in pairs}
    raw_pooled = {pair: PairMoments() for pair in pairs}
    by_decision = {
        (decision, *pair): PairMoments() for decision in range(55) for pair in pairs
    }
    complete = VectorMoments(3)
    raw_complete = VectorMoments(3)
    per_date_rows: list[dict[str, object]] = []
    coverage = np.zeros(3, dtype=np.int64)
    total_slots = 0

    for date_idx, trade_date in zip(dates, trade_dates, strict=True):
        values = np.asarray(targets[int(date_idx)], dtype=np.float64)
        raw = np.asarray(raw_returns[int(date_idx)], dtype=np.float64)
        mask = np.asarray(label_mask[int(date_idx)], dtype=bool)
        coverage += mask.sum(axis=(0, 1))
        total_slots += mask.shape[0] * mask.shape[1]
        flat = values.reshape(-1, 3)
        flat_raw = raw.reshape(-1, 3)
        flat_mask = mask.reshape(-1, 3)
        all_valid = flat_mask.all(axis=1)
        complete.update(flat[all_valid])
        raw_complete.update(flat_raw[all_valid])
        for left, right in pairs:
            valid = flat_mask[:, left] & flat_mask[:, right]
            pooled[left, right].update(flat[:, left], flat[:, right], valid)
            raw_pooled[left, right].update(flat_raw[:, left], flat_raw[:, right], valid)
            local = PairMoments()
            local.update(flat[:, left], flat[:, right], valid)
            per_date_rows.append(
                {
                    "trade_date": trade_date,
                    **_pair_row("date", left, right, local),
                }
            )
        for decision in range(55):
            decision_values = values[:, decision]
            decision_mask = mask[:, decision]
            for left, right in pairs:
                valid = decision_mask[:, left] & decision_mask[:, right]
                by_decision[decision, left, right].update(
                    decision_values[:, left],
                    decision_values[:, right],
                    valid,
                )

    decision_rows = [
        {
            "decision_idx": decision,
            **_pair_row("decision", left, right, moments),
        }
        for (decision, left, right), moments in by_decision.items()
    ]
    pairwise_rows = [
        _pair_row("pooled", left, right, pooled[left, right]) for left, right in pairs
    ]
    for scope, rows in (
        ("equal_date", per_date_rows),
        ("equal_decision", decision_rows),
    ):
        for left, right in pairs:
            selected = [
                row
                for row in rows
                if row["left_horizon"] == HORIZONS[left]
                and row["right_horizon"] == HORIZONS[right]
            ]
            pairwise_rows.append(
                {
                    "scope": scope,
                    "left_horizon": HORIZONS[left],
                    "right_horizon": HORIZONS[right],
                    "valid_count": int(
                        sum(int(row["valid_count"]) for row in selected)
                    ),
                    "covariance": _mean_finite(
                        [float(row["covariance"]) for row in selected]
                    ),
                    "correlation": _mean_finite(
                        [float(row["correlation"]) for row in selected]
                    ),
                }
            )

    correlation = np.eye(3, dtype=np.float64)
    raw_correlation = np.eye(3, dtype=np.float64)
    for left, right in pairs:
        correlation[left, right] = correlation[right, left] = pooled[
            left, right
        ].correlation()
        raw_correlation[left, right] = raw_correlation[right, left] = raw_pooled[
            left, right
        ].correlation()
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    complete_covariance = complete.covariance()
    basis_variance = np.diag(
        FIXED_TARGET_BASIS @ complete_covariance @ FIXED_TARGET_BASIS.T
    )
    complete_scale = np.sqrt(np.diag(complete_covariance))
    complete_correlation = complete_covariance / np.outer(
        complete_scale, complete_scale
    )
    summary: dict[str, object] = {
        "train_end": str(TRAIN_END),
        "date_count": len(dates),
        "valid_counts_by_horizon": dict(zip(HORIZONS, coverage.tolist(), strict=True)),
        "coverage_fraction_by_horizon": dict(
            zip(HORIZONS, (coverage / total_slots).tolist(), strict=True)
        ),
        "complete_case_count": complete.count,
        "pooled_target_correlation": correlation,
        "complete_case_target_covariance": complete_covariance,
        "complete_case_correlation_sensitivity": complete_correlation,
        "eigenvalues": eigenvalues,
        "eigenvectors_columns": eigenvectors,
        "variance_shares": eigenvalues / eigenvalues.sum(),
        "fixed_basis_rows": FIXED_TARGET_BASIS,
        "fixed_basis_variance": basis_variance,
        "raw_return_headline_correlation": raw_correlation,
        "raw_complete_case_count": raw_complete.count,
    }
    atomic_csv(pl.DataFrame(pairwise_rows), output_dir / "target_pairwise.csv")
    atomic_parquet(
        pl.DataFrame(per_date_rows), output_dir / "target_basis_by_date.parquet"
    )
    atomic_csv(pl.DataFrame(decision_rows), output_dir / "target_basis_by_decision.csv")
    atomic_json(output_dir / "target_basis_summary.json", summary)
    return summary


def build_oof_plan(
    train_rows: pl.DataFrame,
) -> tuple[dict[str, pl.DataFrame], dict[str, object]]:
    assert_analysis_rows(train_rows)
    dates = train_rows.get_column("trade_date").unique().sort().to_list()
    blocks = np.array_split(np.asarray(dates, dtype=object), 4)
    if any(not block.size for block in blocks):
        raise ValueError("OOF plan requires four non-empty date blocks")
    windows = {
        f"B{index}": train_rows.filter(pl.col("trade_date").is_in(block.tolist()))
        for index, block in enumerate(blocks)
    }
    for index in range(3):
        if blocks[index][-1] >= blocks[index + 1][0]:
            raise ValueError("OOF blocks are not strictly chronological")
    plan = {
        "train_end": str(TRAIN_END),
        "blocks": {
            name: {
                "start": str(rows.get_column("trade_date").min()),
                "end": str(rows.get_column("trade_date").max()),
                "date_count": rows.get_column("trade_date").n_unique(),
                "sample_count": rows.height,
            }
            for name, rows in windows.items()
        },
        "folds": {
            "fold_1": {"fit": ["B0"], "selection": "B1", "prediction": "B2"},
            "fold_2": {
                "fit": ["B0", "B1"],
                "selection": "B2",
                "prediction": "B3",
            },
        },
    }
    return windows, plan


def context_permutation(
    rows: pl.DataFrame,
    seed: int = PERMUTATION_SEED,
) -> tuple[dict[int, int], dict[str, object]]:
    assert_analysis_rows(rows, allow_validation=True)
    date_rows = rows.select("date_idx", "trade_date").unique().sort("trade_date")
    generator = np.random.default_rng(seed)
    mapping: dict[int, int] = {}
    group_sizes: dict[str, int] = {}
    keyed: dict[tuple[int, int], list[tuple[int, date]]] = {}
    for date_idx, trade_date in date_rows.iter_rows():
        key = (trade_date.year, (trade_date.month - 1) // 3 + 1)
        keyed.setdefault(key, []).append((int(date_idx), trade_date))
    for key, values in keyed.items():
        date_indices = np.asarray([value[0] for value in values], dtype=np.int64)
        order = generator.permutation(date_indices)
        donors = np.roll(order, 1) if order.size > 1 else order
        mapping.update(
            {int(recipient): int(donor) for recipient, donor in zip(order, donors)}
        )
        group_sizes[f"{key[0]}Q{key[1]}"] = int(order.size)
    serialized = json.dumps(sorted(mapping.items()), separators=(",", ":")).encode()
    self_maps = sum(recipient == donor for recipient, donor in mapping.items())
    return mapping, {
        "seed": seed,
        "mapping_sha256": hashlib.sha256(serialized).hexdigest(),
        "group_sizes": group_sizes,
        "self_map_count": self_maps,
    }


def _family_instrument_indices(family: str, equity_count: int) -> tuple[int, ...]:
    return tuple(equity_count + slot for slot in context_family_slots(family))


def mask_context_family_batch(
    batch: dict[str, torch.Tensor], family: str
) -> dict[str, torch.Tensor]:
    result = dict(batch)
    equity_count = batch["instrument_mask"].shape[1] - CONTEXT_COUNT
    slots = _family_instrument_indices(family, equity_count)
    for name in ("patches", "history_patch_mask", "instrument_mask", "slow_features"):
        values = batch[name].clone()
        values[:, slots] = 0
        result[name] = values
    return result


def permute_context_family_batch(
    batch: dict[str, torch.Tensor],
    family: str,
    mapping: dict[int, int],
) -> dict[str, torch.Tensor]:
    decisions = batch["decision_idx"][batch["sample_valid_mask"]]
    if decisions.unique().numel() != 1:
        raise ValueError("Context permutation batch crosses decision indices")
    date_indices = batch["date_idx"].tolist()
    positions = {int(value): index for index, value in enumerate(date_indices)}
    donor_positions = []
    for value in date_indices:
        donor = mapping[int(value)]
        if donor not in positions:
            raise ValueError("Permutation donor is absent from the decision batch")
        donor_positions.append(positions[donor])
    donors = torch.as_tensor(donor_positions, dtype=torch.long)
    result = dict(batch)
    equity_count = batch["instrument_mask"].shape[1] - CONTEXT_COUNT
    slots = _family_instrument_indices(family, equity_count)
    for name in ("patches", "history_patch_mask", "instrument_mask", "slow_features"):
        values = batch[name].clone()
        source = batch[name].index_select(0, donors)
        values[:, slots] = source[:, slots]
        result[name] = values
    return result


def gradient_cosine(
    left: torch.Tensor | None, right: torch.Tensor | None
) -> tuple[float | None, str | None]:
    if left is None or right is None:
        return None, "absent"
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        return None, "zero_norm"
    return float(torch.dot(left, right) / (left_norm * right_norm)), None


def rank_standardize_predictions(
    predictions: NDArray[np.floating],
    label_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    values = np.asarray(predictions, dtype=np.float64)
    mask = np.asarray(label_mask, dtype=bool)
    result = np.zeros_like(values)
    for sample in range(values.shape[0]):
        for horizon in range(values.shape[2]):
            valid = mask[sample, :, horizon]
            if valid.sum() < 2:
                continue
            ranks = average_ranks(values[sample, valid, horizon])
            centered = ranks - ranks.mean()
            scale = math.sqrt(float(np.mean(centered**2)))
            if scale > 0.0:
                result[sample, valid, horizon] = centered / scale
    return result


def _center_equity(values: torch.Tensor, equity_mask: torch.Tensor) -> torch.Tensor:
    weight = equity_mask[..., None].to(values.dtype)
    count = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * weight).sum(dim=1, keepdim=True) / count
    return (values - mean) * weight


def _last_factor_channels(
    batch: dict[str, torch.Tensor],
    instrument_index: int,
    channel_indices: tuple[int, ...],
    *,
    global_instrument: bool,
) -> torch.Tensor:
    positions = (
        torch.full_like(batch["state_position"], ABSOLUTE_PATCH_COUNT - 1)
        if global_instrument
        else batch["state_position"] - 1
    )
    patch = batch["patches"][
        torch.arange(positions.shape[0]), instrument_index, positions
    ].reshape(positions.shape[0], PATCH_MINUTES, DYNAMIC_CHANNEL_COUNT)
    return patch[:, -1, channel_indices]


def residual_source_designs(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    equity_mask = batch["instrument_mask"][:, :EQUITY_COUNT]
    slow = _center_equity(batch["slow_features"][:, :EQUITY_COUNT], equity_mask)
    if "peer_state" not in batch or batch["peer_state"].shape[-1] != PEER_STATE_WIDTH:
        raise ValueError("Selected-peer state is required for residual probes")
    peer = batch["peer_state"] * equity_mask[..., None]
    factor_names = (
        "return_since_open_normalized",
        "return_15m_normalized",
        "return_30m_normalized",
        "return_60m_normalized",
        "realized_vol_30m_log_ratio",
        "cumulative_volume_surprise",
    )
    channels = tuple(DYNAMIC_CHANNELS.index(name) for name in factor_names)
    beta_symbols = ("WDO$", "DI1F27", "DI1F28", "DI1F29", "DI1F31")
    beta_names = tuple(f"beta_to_{symbol.rstrip('$')}" for symbol in beta_symbols)
    beta_indices = tuple(EQUITY_SLOW_CHANNELS.index(name) for name in beta_names)
    local_slots = (1, 2, 3, 4, 5)
    interactions: list[torch.Tensor] = []
    readiness: list[torch.Tensor] = []
    for beta_index, local_slot in zip(beta_indices, local_slots, strict=True):
        factor = _last_factor_channels(
            batch, EQUITY_COUNT + local_slot, channels, global_instrument=False
        )
        interactions.append(slow[..., beta_index, None] * factor[:, None, :])
        ready = batch["instrument_mask"][:, EQUITY_COUNT + local_slot].to(slow.dtype)
        readiness.append(slow[..., beta_index, None] * ready[:, None, None])
    characteristic_names = (
        "overnight_gap_cross_section_rank",
        "dollar_volume_cross_section_rank",
        "realized_vol_cross_section_rank",
    )
    characteristic_indices = tuple(
        EQUITY_SLOW_CHANNELS.index(name) for name in characteristic_names
    )
    global_start = EQUITY_COUNT + LOCAL_CONTEXT_COUNT
    for global_slot in (2, 3):
        factor = _last_factor_channels(
            batch,
            global_start + global_slot,
            channels,
            global_instrument=True,
        )
        ready = batch["instrument_mask"][:, global_start + global_slot].to(slow.dtype)
        for characteristic_index in characteristic_indices:
            characteristic = slow[..., characteristic_index, None]
            interactions.append(characteristic * factor[:, None, :])
            readiness.append(characteristic * ready[:, None, None])
    macro = torch.cat((*interactions, *readiness), dim=-1)
    return {
        "slow": slow,
        "selected_peer": peer,
        "macro_interaction": macro,
        "combined": torch.cat((slow, peer, macro), dim=-1),
    }
