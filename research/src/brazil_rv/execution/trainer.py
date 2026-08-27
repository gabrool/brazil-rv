from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint

from .config import ExecutionConfig
from .simulator import MarketReplay, simulate


@dataclass(frozen=True)
class PolicyBatch:
    market: MarketReplay
    ranks: Tensor
    rank_valid: Tensor
    refresh_mask: Tensor
    sigma: Tensor


@dataclass(frozen=True)
class PolicyTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    risk_aversion: float = 0.0
    gradient_clip_norm: float = 1.0
    seed: int = 0
    use_sam: bool = False
    sam_rho: float = 0.05
    gradient_checkpointing: bool = False
    patience: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative")
        if not math.isfinite(self.risk_aversion) or self.risk_aversion < 0:
            raise ValueError("Risk aversion must be finite and nonnegative")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("Gradient clip norm must be finite and positive")
        if not math.isfinite(self.sam_rho) or self.sam_rho <= 0:
            raise ValueError("SAM rho must be finite and positive")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("Patience must be positive when configured")


@dataclass(frozen=True)
class TrainStepMetrics:
    objective_brl: float
    mean_excess_pnl_brl: float
    daily_net_pnl_variance_brl2: float
    gradient_norm: float


def policy_objective(
    net_pnl_brl: Tensor,
    daily_cdi_rate: Tensor,
    nav_brl: float,
    risk_aversion: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return objective, daily excess PnL, and population PnL variance.

    ``risk_aversion`` has units BRL^-1 because the objective is intentionally in
    BRL, matching the registered policy contract.
    """
    all_cash_cdi = daily_cdi_rate.to(net_pnl_brl) * nav_brl
    excess = net_pnl_brl - all_cash_cdi
    variance = net_pnl_brl.var(correction=0)
    objective = excess.mean() - risk_aversion * variance
    return objective, excess, variance


class PolicyTrainer:
    """Direct differentiable policy optimizer; experiment selection stays outside."""

    def __init__(
        self,
        policy: torch.nn.Module,
        execution_config: ExecutionConfig,
        config: PolicyTrainerConfig = PolicyTrainerConfig(),
    ) -> None:
        self.policy = policy
        self.execution_config = execution_config
        self.config = config
        policy_seed = getattr(policy, "seed", config.seed)
        if int(policy_seed) != config.seed:
            raise ValueError("Trainer seed differs from the policy initialization seed")
        self.optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.best_monitor = -math.inf
        self.monitor_bad_steps = 0
        self.best_policy_state: dict[str, Tensor] | None = None

    def _net_pnl(self, batch: PolicyBatch) -> Tensor:
        def replay(input_ranks: Tensor) -> Tensor:
            return simulate(
                batch.market,
                input_ranks,
                batch.rank_valid,
                batch.refresh_mask,
                batch.sigma,
                self.policy,
                self.execution_config,
            ).net_pnl_brl

        if self.config.gradient_checkpointing:
            return checkpoint(replay, batch.ranks, use_reentrant=False)
        return replay(batch.ranks)

    def _objective(self, batch: PolicyBatch) -> tuple[Tensor, Tensor, Tensor]:
        return policy_objective(
            self._net_pnl(batch),
            batch.market.daily_cdi_rate,
            self.execution_config.nav_brl,
            self.config.risk_aversion,
        )

    def _sam_step(self, batch: PolicyBatch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        objective, excess, variance = self._objective(batch)
        (-objective).backward()
        parameters = [
            parameter
            for parameter in self.policy.parameters()
            if parameter.grad is not None
        ]
        gradient_norm = torch.linalg.vector_norm(
            torch.stack([parameter.grad.norm(2) for parameter in parameters]), 2
        )
        scale = self.config.sam_rho / gradient_norm.clamp_min(
            torch.finfo(gradient_norm.dtype).tiny
        )
        perturbations: list[Tensor] = []
        with torch.no_grad():
            for parameter in parameters:
                perturbation = parameter.grad * scale
                parameter.add_(perturbation)
                perturbations.append(perturbation)
        try:
            self.optimizer.zero_grad(set_to_none=True)
            second_objective, _, _ = self._objective(batch)
            (-second_objective).backward()
        finally:
            with torch.no_grad():
                for parameter, perturbation in zip(
                    parameters, perturbations, strict=True
                ):
                    parameter.sub_(perturbation)
        return objective, excess, variance, gradient_norm

    def train_step(self, batch: PolicyBatch) -> TrainStepMetrics:
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.config.use_sam:
            objective, excess, variance, preclip_norm = self._sam_step(batch)
        else:
            objective, excess, variance = self._objective(batch)
            (-objective).backward()
            preclip_norm = clip_grad_norm_(
                self.policy.parameters(), self.config.gradient_clip_norm
            )
        if self.config.use_sam:
            preclip_norm = clip_grad_norm_(
                self.policy.parameters(), self.config.gradient_clip_norm
            )
        self.optimizer.step()
        return TrainStepMetrics(
            objective_brl=float(objective.detach()),
            mean_excess_pnl_brl=float(excess.detach().mean()),
            daily_net_pnl_variance_brl2=float(variance.detach()),
            gradient_norm=float(preclip_norm.detach()),
        )

    def update_monitor(self, value: float) -> bool:
        """Record a caller-computed monitor; return whether patience is exhausted."""
        if not math.isfinite(value):
            raise ValueError("Monitor value must be finite")
        if value > self.best_monitor:
            self.best_monitor = value
            self.monitor_bad_steps = 0
            self.best_policy_state = copy.deepcopy(self.policy.state_dict())
        else:
            self.monitor_bad_steps += 1
        return (
            self.config.patience is not None
            and self.monitor_bad_steps >= self.config.patience
        )

    def restore_best(self) -> None:
        if self.best_policy_state is None:
            raise RuntimeError("No caller monitor has selected a policy state")
        self.policy.load_state_dict(self.best_policy_state)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "schema": "BRAZIL_RV_POLICY_TRAINER_CHECKPOINT_V1",
                "trainer_config": asdict(self.config),
                "execution_config_sha256": self.execution_config.sha256,
                "policy_contract": getattr(self.policy, "contract_metadata", None),
                "seed": self.config.seed,
                "policy_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_monitor": self.best_monitor,
                "monitor_bad_steps": self.monitor_bad_steps,
                "best_policy_state": self.best_policy_state,
            },
            temporary,
        )
        temporary.replace(path)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != "BRAZIL_RV_POLICY_TRAINER_CHECKPOINT_V1"
            or payload.get("trainer_config") != asdict(self.config)
            or payload.get("execution_config_sha256") != self.execution_config.sha256
            or payload.get("policy_contract")
            != getattr(self.policy, "contract_metadata", None)
            or payload.get("seed") != self.config.seed
        ):
            raise ValueError("Policy trainer checkpoint contract differs")
        self.policy.load_state_dict(payload["policy_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.best_monitor = float(payload["best_monitor"])
        self.monitor_bad_steps = int(payload["monitor_bad_steps"])
        self.best_policy_state = payload["best_policy_state"]
