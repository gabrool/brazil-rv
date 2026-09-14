"""Magnitude-preserving auxiliary labels, fit scaling and one-pass shared heads."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .artifacts import sha256_file, write_json_atomic
from .contract import HORIZONS
from .decision_program import benchmark_excess_returns
from .portfolio_program import read


def attach_economic_head(model):
    """Zero alpha initially; preserve the matched control's random state."""
    width = (
        model.config.width
        if hasattr(model.config, "width")
        else model.config.fusion_width
    )
    with torch.random.fork_rng(devices=[]):
        head = nn.Linear(width, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    model.economic_head = head.to(next(model.parameters()).device)


def economic_loss(prediction, target, valid):
    """Equal date/member weight; Huber in fit-scaled daily residual-return units."""
    mask = valid[..., None]
    error = torch.where(mask, prediction.float() - target.float()[..., None], 0.0)
    absolute = error.abs()
    loss = torch.where(absolute <= 1, 0.5 * error.square(), absolute - 0.5)
    count = mask.sum(1).clamp_min(1)
    per_date = loss.sum(1) / count
    admitted = valid.any(1)
    return (per_date * admitted[:, None]).sum() / (
        admitted.sum().clamp_min(1) * prediction.shape[-1]
    )


def prepare_targets(root, data, binding, store_dates):
    """Reuse sealed economic outcomes; never place them in predictive features."""
    output = root / "phase3/economic_targets.npz"
    market = benchmark_excess_returns(data)
    count = len(market)
    cash = np.full(count, np.nan)
    for day in range(count - 5):
        cash[day] = np.prod(1 + data.inputs.cdi_returns[day + 1 : day + 6]) - 1
    values = (
        data.inputs.shareholder_simple_returns[..., HORIZONS.index(5)]
        - cash[:, None]
        - data.beta * market[:, None]
    ) / 5
    valid = (
        data.inputs.shareholder_target_mask[..., HORIZONS.index(5)]
        & data.inputs.active
        & np.isfinite(values)
    )
    indices = np.searchsorted(
        store_dates, np.asarray(data.inputs.dates, dtype="datetime64[D]")
    )
    if not np.array_equal(
        store_dates[indices], np.asarray(data.inputs.dates, dtype="datetime64[D]")
    ):
        raise ValueError("economic target dates do not match the store")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(
            handle,
            indices=indices,
            values=np.where(valid, values, 0).astype(np.float32),
            valid=valid,
            isins=np.asarray(data.inputs.security_ids),
        )
    write_json_atomic(
        output.with_suffix(".json"),
        {
            "sha256": sha256_file(output),
            "policy_data_sha256": binding,
            "target": "(shareholder_H5 - CDI_H5 - decision_beta * BOVA_excess_H5) / 5",
            "horizon": 5,
            "valid_stock_days": int(valid.sum()),
            "first_date": str(data.inputs.dates[0]),
            "last_date": str(data.inputs.dates[-1]),
        },
    )


class EconomicCollator:
    """Add labels after unchanged PIT compaction; statistics consume only fit."""

    def __init__(self, collate, source, fit, fit_window, isins):
        record = read(source.with_suffix(".json"))
        if sha256_file(source) != record["sha256"]:
            raise ValueError("economic label source changed")
        with np.load(source, allow_pickle=False) as arrays:
            if tuple(arrays["isins"].tolist()) != tuple(isins):
                raise ValueError("economic target security identity differs")
            indices, values, valid = (
                arrays[k].copy() for k in ("indices", "values", "valid")
            )
        allowed = set(int(i) for i in fit_window)
        admitted = np.asarray(
            [i in fit and all(int(i) + h in allowed for h in range(6)) for i in indices]
        )
        valid &= admitted[:, None]
        date_count = valid.sum(1)
        if not np.any(date_count):
            raise ValueError("economic auxiliary has no permitted fit labels")
        daily_second = np.where(valid, values.astype(np.float64) ** 2, 0).sum(
            1
        ) / np.maximum(date_count, 1)
        scale = float(np.sqrt(daily_second[date_count > 0].mean()))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("economic fit target scale is undefined")
        self.collate = collate
        self.positions = {int(i): j for j, i in enumerate(indices)}
        self.values = np.where(valid, values / scale, 0).astype(np.float32)
        self.valid = valid
        self.contract = {
            "source": str(source),
            "source_sha256": record["sha256"],
            "scale": scale,
            "fit_date_indices": [int(i) for i in indices[admitted]],
            "fit_target_window": [int(i) for i in fit_window],
            "valid_stock_days": int(valid.sum()),
            "loss": "equal-date Huber delta=1; zero-centered fit RMS scale; no target ranking or clipping",
            "head": "zero-initialized linear head on shared final trunk; matched RNG preserved",
        }

    def __call__(self, samples):
        batch = self.collate(samples)
        rows = np.asarray([self.positions[int(i)] for i in batch["date_index"]])
        names = batch["name_index"].numpy()
        valid = self.valid[rows[:, None], names.clip(min=0)] & (names >= 0)
        batch["economic_target"] = torch.from_numpy(
            np.where(valid, self.values[rows[:, None], names.clip(min=0)], 0)
        )
        batch["economic_mask"] = torch.from_numpy(valid)
        return batch
