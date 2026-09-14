"""Joint stock/hedge allocation with cash and differentiable sparse QP solving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import torch
from scipy import sparse
from torch import Tensor


@dataclass(frozen=True)
class AllocationConfig:
    gross_cap: float = 2.25
    net_cap: float = 0.05
    beta_cap: float = 0.05
    stock_cap: float = 0.05
    hedge_cap: float = 0.60
    risk_aversion: float = 5.0
    planning_sessions: int = 5
    cost_bps: float = 4.0


class _SparseQP(torch.autograd.Function):
    """Only preferences/bounds are learned; use OSQP's native vector adjoint.

    A solve owns its workspace until backward. Reusing one mutable workspace
    across sequential autograd nodes would differentiate the wrong day's QP.
    Avoid the upstream torch wrapper's per-call full-machine thread pool and
    unnecessary matrix derivatives for our fixed risk/constraint coefficients.
    """

    @staticmethod
    def forward(ctx, q, lower, upper, p, a, warm_start):
        solver = osqp.OSQP(algebra="builtin")
        solver.setup(
            P=p,
            q=q.detach().numpy(),
            A=a,
            l=lower.detach().numpy(),
            u=upper.detach().numpy(),
            verbose=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=20000,
            polishing=True,
        )
        solver.warm_start(x=warm_start)
        result = solver.solve(raise_error=True)
        ctx.solver = solver
        return torch.from_numpy(result.x)

    @staticmethod
    def backward(ctx, gradient):
        solver = ctx.solver
        solver.adjoint_derivative_compute(dx=gradient.contiguous().numpy())
        dq, dl, du = solver.adjoint_derivative_get_vec()
        values = tuple(torch.from_numpy(x) for x in (dq, dl, du))
        if not all(torch.isfinite(x).all() for x in values):
            raise FloatingPointError("allocation adjoint is non-finite")
        return *values, None, None, None


def allocate(
    preference: Tensor,
    previous: Tensor,
    *,
    beta: np.ndarray,
    idiosyncratic_variance: np.ndarray,
    market_variance: float,
    daily_borrow: np.ndarray,
    lower: Tensor,
    upper: Tensor,
    config: AllocationConfig = AllocationConfig(),
) -> Tensor:
    """Allocate a compact stock-plus-hedge vector; residual capital is cash.

    Preferences/risks/borrow are daily returns, variances and holding costs.
    Five-period expected utility pays today's linear transaction cost once.
    The last coordinate is the optional hedge. Historical prices never enter
    this solve; the caller supplies only information available at the decision.
    OSQP uses FP64 CPU sparse algebra; neural preferences retain their gradient
    through any dtype conversion. Fixed diagonal+market covariance is PSD.
    """
    preference, previous, lower, upper = (
        x.to(device="cpu", dtype=torch.float64)
        for x in (preference, previous, lower, upper)
    )
    n = preference.numel()
    beta = np.asarray(beta, dtype=np.float64)
    diagonal = np.asarray(idiosyncratic_variance, dtype=np.float64)
    borrow = torch.as_tensor(daily_borrow, dtype=torch.float64)
    if (diagonal <= 0).any() or market_variance < 0:
        raise ValueError("allocation risk must be positive diagonal plus PSD factor")
    identity = sparse.eye(n, format="csc")
    zero = sparse.csc_matrix((n, n))
    column = sparse.csc_matrix((n, 1))
    # x = (weights, absolute trade, absolute weight, market exposure).
    a = sparse.bmat(
        [
            [identity, zero, zero, column],
            [identity, -identity, zero, column],
            [-identity, -identity, zero, column],
            [identity, zero, -identity, column],
            [-identity, zero, -identity, column],
            [np.ones((1, n)), None, None, sparse.csc_matrix((1, 1))],
            [beta.reshape(1, n), None, None, sparse.csc_matrix((1, 1))],
            [None, None, np.ones((1, n)), sparse.csc_matrix((1, 1))],
            [-beta.reshape(1, n), None, None, np.ones((1, 1))],
        ],
        format="csc",
    )
    # Work in basis points to avoid absolute tolerance dwarfing daily utility.
    horizon = config.planning_sessions
    p = sparse.diags(
        np.r_[
            1e4 * horizon * config.risk_aversion * diagonal,
            np.zeros(2 * n),
            1e4 * horizon * config.risk_aversion * market_variance,
        ],
        format="csc",
    )
    q = torch.cat(
        (
            -1e4 * horizon * (preference + borrow / 2),
            torch.full((n,), config.cost_bps, dtype=torch.float64),
            1e4 * horizon * borrow / 2,
            torch.zeros(1, dtype=torch.float64),
        )
    )
    bounds_lower = torch.cat(
        (
            lower,
            torch.full((4 * n,), -torch.inf, dtype=torch.float64),
            torch.tensor(
                [-config.net_cap, -config.beta_cap, -torch.inf, 0.0],
                dtype=torch.float64,
            ),
        )
    )
    bounds_upper = torch.cat(
        (
            upper,
            previous,
            -previous,
            torch.zeros(2 * n, dtype=torch.float64),
            torch.tensor(
                [config.net_cap, config.beta_cap, config.gross_cap, 0.0],
                dtype=torch.float64,
            ),
        )
    )
    prior = previous.detach().numpy()
    warm = np.r_[prior, np.zeros(n), np.abs(prior), beta @ prior]
    # Percent-NAV solver coordinates avoid an ill-scaled epigraph: stock weights
    # are a few percent while objective slopes are several basis points. This
    # is an exact change of units, including adjoints and all bound gradients.
    scale = 100.0
    solution = (
        _SparseQP.apply(
            q / scale,
            bounds_lower * scale,
            bounds_upper * scale,
            p / scale**2,
            a,
            warm * scale,
        )[:n]
        / scale
    )
    weights = solution.detach().numpy()
    residual = max(
        np.abs(weights).sum() - config.gross_cap,
        abs(weights.sum()) - config.net_cap,
        abs(beta @ weights) - config.beta_cap,
        float((lower.detach().numpy() - weights).max()),
        float((weights - upper.detach().numpy()).max()),
    )
    if not np.isfinite(weights).all() or residual > 2e-6:
        raise FloatingPointError(f"allocation violates planned constraints: {residual}")
    return solution
