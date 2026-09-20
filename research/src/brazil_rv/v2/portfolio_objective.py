"""Neural daily-return preferences and chronological portfolio utility gradients."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import fields

import numpy as np
import torch
from torch import nn

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.portfolio_account import PortfolioAccount, tensor
from brazil_rv.execution.portfolio_policy import account_decision, portfolio_variance
from .contract import HORIZONS
from .economic_objective import attach_economic_head
from .round7_training import autocast_dtype, forward, member_loss


def clone_account(account):
    """Independent TBPTT/SAM start, including Python state and unpaid cash claims."""
    values = {}
    for field in fields(account):
        value = getattr(account, field.name)
        if isinstance(value, torch.Tensor):
            value = value.detach().clone()
        elif isinstance(value, np.ndarray):
            value = value.copy()
        elif field.name in {"payments", "proceeds_releases", "bonus_proceeds"}:
            value = [(day, amount.detach().clone()) for day, amount in value]
        elif field.name == "settlements":
            value = [
                (day, free.detach().clone(), restricted.detach().clone())
                for day, free, restricted in value
            ]
        elif field.name in {"loans", "custody", "custody_fees"}:
            value = value.detached_copy()
        elif isinstance(value, (dict, set)):
            value = deepcopy(value)
        values[field.name] = value
    return PortfolioAccount(**values)


def smooth_ranks(scores, active, temperature=0.1):
    """Symmetric normalized midranks; only same-date eligible peers participate."""
    x = scores.float().transpose(1, 2)
    mask = active[:, None, :]
    count = mask.sum(-1, keepdim=True).clamp_min(1)
    mean = (x * mask).sum(-1, keepdim=True) / count
    centered = (x - mean) * mask
    deviation = (centered.square().sum(-1, keepdim=True) / count + 1e-8).sqrt()
    z = centered / deviation
    pairs = torch.sigmoid((z[..., :, None] - z[..., None, :]) / temperature)
    ranks = 2 * (pairs * mask[..., None, :]).sum(-1) / count - 1
    return (ranks * mask).transpose(1, 2)


class NeuralPreference(nn.Module):
    """Rank anchor plus an unranked cardinal correction from the shared encoder."""

    def __init__(self, model, *, characteristic, horizons, calibration):
        super().__init__()
        attach_economic_head(model)
        self.model = model
        self.characteristic = characteristic
        self.head_indices = [HORIZONS.index(h) for h in horizons]
        self.primary_indices = [list(horizons).index(h) for h in (3, 5, 10)]
        self.amp_dtype = autocast_dtype(next(model.parameters()).device)
        self.cuda = next(model.parameters()).is_cuda
        # The mapping is frozen from the previous prior-only OOS calibration.
        # Buffers migrate with the network but stay in FP32 for preference maths.
        self.register_buffer("rank_mean", torch.tensor(float(calibration.mean[0])))
        self.register_buffer("rank_scale", torch.tensor(float(calibration.scale[0])))
        self.register_buffer(
            "coefficient", torch.tensor(float(calibration.coefficient[0]))
        )
        self.register_buffer("intercept", torch.tensor(float(calibration.intercept)))

    def forward(self, batch):
        with torch.autocast(
            device_type="cuda" if self.cuda else "cpu",
            dtype=self.amp_dtype,
            enabled=self.cuda,
        ):
            scores, hidden = forward(
                self.model,
                batch,
                characteristic=self.characteristic,
                return_hidden=True,
            )
            cardinal = self.model.economic_head(hidden).squeeze(-1).float().mean(2)
        scores = scores.float()
        primary = scores.mean(2)[..., self.primary_indices]
        ranks = smooth_ranks(primary, batch["active_mask"])
        base = (
            ranks.mean(-1) - self.rank_mean
        ) / self.rank_scale * self.coefficient + self.intercept
        # One network output unit = one daily basis point. No tanh/clip/reranking.
        preference = (base + cardinal * 1e-4) * batch["active_mask"]
        ranking = member_loss(
            scores,
            batch["targets"][..., self.head_indices],
            batch["target_mask"][..., self.head_indices]
            & batch["active_mask"][..., None],
        )
        return primary, preference, ranking


class TensorPreference:
    """Direct preference interface; never consult the cache's old score features."""

    def __init__(self, values, first=0):
        self.values, self.first = values, first

    def preference_for(self, data, day, names, state):
        return self.values[day - self.first, names]

    def market_return_for(self, day):
        return 0.0

    def forecast_uncertainty(self, ranks):
        return None


def full_preferences(compact, names, security_count):
    """Restore permanent identity without losing the neural autograd connection."""
    mask = names >= 0
    return (
        compact.new_zeros((len(compact), security_count))
        .scatter_add(
            1,
            names.clamp_min(0),
            compact * mask,
        )
        .to(device="cpu", dtype=torch.float64)
    )


def utility_path(
    data, preference, account, rows, *, allocation=AllocationConfig(), terminal=False
):
    """Actual one-session accounting; final liquidation only at a real boundary."""
    policy = TensorPreference(preference, int(rows[0]))
    utilities, targets, nav = [], [], []
    for offset, day in enumerate(rows):
        day = int(day)
        risk = portfolio_variance(data, day, account.market_weights)
        final = terminal and offset == len(rows) - 1
        target = (
            tensor(np.zeros(preference.shape[1] + 1))
            if final
            else account_decision(
                data,
                policy,
                account,
                day,
                allocation=allocation,
            )
        )
        outcome = data.step(account, target, day, terminal=final)
        utilities.append(1e4 * (outcome["net_excess"] - 2.5 * risk))
        targets.append(target.detach().numpy())
        nav.append(float(outcome["nav"].detach()))
    return -torch.stack(utilities).mean(), account, np.asarray(targets), np.asarray(nav)


def active_view(data, rows, active):
    """No cached old forecast controls a newly produced preference's eligibility."""
    result = copy(data)
    result.valid = np.zeros_like(data.valid)
    result.valid[rows] = active
    return result
