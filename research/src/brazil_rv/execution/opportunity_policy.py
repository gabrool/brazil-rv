"""Small conditional calibration and stateful residual alternatives."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import CalibratedPolicy
from brazil_rv.v2.round7_preprocessing import RobustScaler


class OpportunityPolicy(CalibratedPolicy):
    """Both alternatives initialize exactly at coherent deterministic allocation.

    Common state is held once per date and broadcast only for the selected
    compact name set. All scaling is fit-only; missingness remains explicit.
    """

    def __init__(self, data, calibration, fit_rows, *, kind):
        super().__init__(calibration)
        self.kind = kind
        context = data.context
        self.common_scaler = RobustScaler.fit(
            context.common[fit_rows],
            context.valid[fit_rows],
            fit_rows,
            passthrough=context.passthrough,
        )
        static = data.static[fit_rows][data.valid[fit_rows]]
        self.static_scaler = RobustScaler.fit(
            static,
            np.ones_like(static, bool),
            fit_rows,
            passthrough=tuple(i in (6, 7, 12) for i in range(static.shape[-1])),
        )
        for label, scaler in (
            ("common", self.common_scaler),
            ("static", self.static_scaler),
        ):
            self.register_buffer(
                label + "_center", torch.as_tensor(scaler.center, dtype=torch.float32)
            )
            self.register_buffer(
                label + "_scale", torch.as_tensor(scaler.scale, dtype=torch.float32)
            )
            self.register_buffer(
                label + "_passthrough", torch.as_tensor(scaler.passthrough)
            )
        common_size = context.common.shape[-1] + 4
        self.calibration_network = nn.Linear(common_size, 3)
        nn.init.zeros_(self.calibration_network.weight)
        nn.init.zeros_(self.calibration_network.bias)
        if kind == "stateful":
            self.network = nn.Sequential(
                nn.Linear(common_size + static.shape[-1] + 10 + 3, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            output = self.network[-1]
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        elif kind != "reliability":
            raise ValueError("unknown controller comparison")

    def conditioned_common(self, data, day):
        raw = torch.from_numpy(data.context.common[day])
        valid = torch.from_numpy(data.context.valid[day])
        common = torch.where(
            valid,
            torch.where(
                self.common_passthrough,
                raw,
                torch.asinh((raw - self.common_center) / self.common_scale),
            ),
            0,
        )
        ranks = tensor(data.ranks[day, data.valid[day]])
        with torch.no_grad():
            expected = super().forward(None, None, ranks)
            # Values are daily bps, not normalized rank-model logit spreads.
            opportunity = (
                torch.stack(
                    (
                        expected.mean(),
                        expected.std(correction=0),
                        expected.max(),
                        expected.min(),
                    )
                )
                if len(expected)
                else tensor([0.0, 0.0, 0.0, 0.0])
            )
        return torch.cat((common, torch.asinh(1e4 * opportunity).float()))

    def preference_for(self, data, day, names, state):
        ranks = tensor(data.ranks[day, names])
        base = super().forward(None, None, ranks)
        common = self.conditioned_common(data, day)
        adjustment = self.calibration_network(common).double()
        preference = (
            base * (1 + adjustment[0].tanh())
            + 0.0003 * ranks.mean(-1) * adjustment[1]
            + 0.0001 * adjustment[2]
        )
        if self.kind == "reliability":
            return preference
        raw = torch.from_numpy(data.static[day, names])
        static = torch.where(
            self.static_passthrough,
            raw,
            torch.asinh((raw - self.static_center) / self.static_scale),
        )
        extra = torch.stack(
            (
                torch.from_numpy(data.context.disagreement[day, names]).float(),
                ranks.std(-1, correction=0).float(),
                torch.asinh(base * 1e4).float(),
            ),
            -1,
        )
        features = torch.cat(
            (static, state.float(), extra, common.expand(len(names), -1)), -1
        )
        # A state correction starts in one-daily-bp units around the explicit
        # conditional rank/alpha path. The output is unbounded, not clipped.
        return preference + 0.0001 * self.network(features).squeeze(-1).double()

    def preprocessing_payload(self):
        return {
            "common": self.common_scaler.payload(),
            "static": self.static_scaler.payload(),
        }
