"""Joint stock/hedge allocation with cash and differentiable sparse QP solving."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import clarabel
import torch
from scipy import linalg, sparse
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
    sector_net_cap: float | None = None


def _solve_primal_dual(p, q, a, lower, upper):
    """Solve the identical sparse QP and map cone duals to interval duals."""
    equal = np.isfinite(lower) & (lower == upper)
    high = np.isfinite(upper) & ~equal
    low = np.isfinite(lower) & ~equal
    constraints = sparse.vstack((a[equal], a[high], -a[low]), format="csc")
    bounds = np.r_[upper[equal], upper[high], -lower[low]]
    n_equal, n_high = int(equal.sum()), int(high.sum())
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.tol_gap_abs = settings.tol_gap_rel = 1e-12
    settings.tol_feas = 1e-10
    result = clarabel.DefaultSolver(
        p,
        q,
        constraints,
        bounds,
        [
            clarabel.ZeroConeT(n_equal),
            clarabel.NonnegativeConeT(n_high + int(low.sum())),
        ],
        settings,
    ).solve()
    if result.status != clarabel.SolverStatus.Solved:
        raise FloatingPointError(f"allocation interior-point solve: {result.status}")
    z = np.asarray(result.z)
    dual = np.zeros(len(lower))
    dual[equal] = z[:n_equal]
    dual[high] += z[n_equal : n_equal + n_high]
    dual[low] -= z[n_equal + n_high :]
    return np.asarray(result.x), dual


class _SparseQP(torch.autograd.Function):
    """Accurate forward QP with its reduced active-face implicit derivative."""

    @staticmethod
    def forward(ctx, q, lower, upper, p, a):
        q_np, lower_np, upper_np = (
            value.detach().numpy() for value in (q, lower, upper)
        )
        x, dual = _solve_primal_dual(p, q_np, a, lower_np, upper_np)
        if any(ctx.needs_input_grad[:3]):
            ctx.problem = (x, dual, p.diagonal(), a, lower_np, upper_np)
        return torch.from_numpy(x)

    @staticmethod
    def backward(ctx, gradient):
        x, dual, diagonal, a, lower, upper = ctx.problem
        n = (len(x) - 1) // 3
        w = x[:n]
        beta = np.asarray(a[5 * n + 1, :n].toarray()).ravel()
        # Epigraphs reduce exactly to |w-previous|, |w| and beta'w. At a
        # strict kink the corresponding weight is fixed; elsewhere their
        # slopes are constant. The remaining Hessian is positive diagonal
        # plus one factor, so only a tiny exposure-constraint system remains.
        tolerance = 1e-6  # percent NAV, below the accepted primal tolerance
        at_lower = (np.abs(w - lower[:n]) < tolerance) & (dual[:n] < -1e-9)
        at_upper = (np.abs(w - upper[:n]) < tolerance) & (dual[:n] > 1e-9)
        equal = lower[:n] == upper[:n]
        at_lower |= equal
        at_upper &= ~equal
        zero = (
            (np.abs(w) < tolerance)
            & (dual[3 * n : 4 * n] > 1e-9)
            & (dual[4 * n : 5 * n] > 1e-9)
        )
        no_trade = (
            (np.abs(w - upper[n : 2 * n]) < tolerance)
            & (dual[n : 2 * n] > 1e-9)
            & (dual[2 * n : 3 * n] > 1e-9)
        )
        fixed = at_lower | at_upper | zero | no_trade
        free = ~fixed
        rows, sides, exposures = [], [], []
        for row in [5 * n, 5 * n + 1, 5 * n + 2, *range(5 * n + 4, len(dual))]:
            if lower[row] == upper[row] or abs(dual[row]) > 1e-9:
                rows.append(row)
                sides.append(dual[row] < 0)
                exposures.append(
                    np.sign(w) if row == 5 * n + 2 else a[row, :n].toarray().ravel()
                )
        c = np.asarray(exposures).reshape(-1, n)
        cf = c[:, free]
        d, b = diagonal[:n][free], beta[free]
        factor = diagonal[-1]

        def inverse_hessian(v):
            out = v / d[:, None]
            return (
                out
                - (b / d)[:, None]
                * (factor * (b @ out) / (1 + factor * np.sum(b * b / d)))[None, :]
            )

        g = gradient.numpy()[:n]
        z = np.zeros(n)
        multiplier = np.zeros(len(c))
        if free.any():
            inverse_g = inverse_hessian(g[free, None])[:, 0]
            if len(c):
                inverse_c = inverse_hessian(cf.T)
                gram = cf @ inverse_c
                # Redundant exposures (e.g. identical net and beta constraints)
                # share a minimum-norm dual; the primal derivative is unique.
                multiplier = linalg.pinvh(gram, rtol=1e-12) @ (cf @ inverse_g)
                z[free] = inverse_g - inverse_c @ multiplier
            else:
                z[free] = inverse_g
        fixed_gradient = g - factor * beta * (beta @ z) - c.T @ multiplier
        residual = max(
            np.max(np.abs(c @ z), initial=0),
            np.max(np.abs((fixed_gradient - diagonal[:n] * z)[free]), initial=0),
        )
        if residual > 1e-7 * max(1.0, float(np.max(np.abs(g)))):
            raise FloatingPointError(f"allocation derivative residual: {residual:g}")
        dq, dl, du = np.zeros(len(x)), np.zeros(len(lower)), np.zeros(len(upper))
        dq[:n] = -z
        dq[n : 2 * n] = -z * np.sign(w - upper[n : 2 * n])
        dq[2 * n : 3 * n] = -z * np.sign(w)
        dq[-1] = -beta @ z
        for i in np.flatnonzero(fixed):
            if at_lower[i]:
                dl[i] = fixed_gradient[i]
            elif at_upper[i]:
                du[i] = fixed_gradient[i]
            elif not zero[i]:
                du[n + i], du[2 * n + i] = (
                    0.5 * fixed_gradient[i],
                    -0.5 * fixed_gradient[i],
                )
        for row, low, value in zip(rows, sides, multiplier):
            (dl if low else du)[row] = value
        values = tuple(torch.from_numpy(x) for x in (dq, dl, du))
        if not all(torch.isfinite(x).all() for x in values):
            raise FloatingPointError("allocation adjoint is non-finite")
        return *values, None, None


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
    forecast_uncertainty: Tensor | None = None,
    sector_exposure: np.ndarray | None = None,
) -> Tensor:
    """Allocate a compact stock-plus-hedge vector; residual capital is cash.

    Preferences/risks/borrow are daily returns, variances and holding costs.
    Five-period expected utility pays today's linear transaction cost once.
    The last coordinate is the optional hedge. Historical prices never enter
    this solve; the caller supplies only information available at the decision.
    The solver uses FP64 CPU sparse algebra; neural preferences retain their gradient
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
    if forecast_uncertainty is not None:
        uncertainty = forecast_uncertainty.to(device="cpu", dtype=torch.float64)
        # Worst-case mean-return adjustment; distinct from return covariance.
        q = q + torch.cat(
            (
                torch.zeros(2 * n, dtype=torch.float64),
                1e4 * horizon * uncertainty,
                torch.zeros(1, dtype=torch.float64),
            )
        )
    if not torch.isfinite(q).all():
        raise FloatingPointError("allocation preference or borrow cost is non-finite")
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
    if config.sector_net_cap is not None:
        if sector_exposure is None:
            raise ValueError("sector allocation requires dated classifications")
        groups = np.asarray(sector_exposure, dtype=np.float64)
        a = sparse.vstack(
            (a, sparse.hstack((groups, sparse.csc_matrix((len(groups), 2 * n + 1))))),
            format="csc",
        )
        caps = torch.full((len(groups),), config.sector_net_cap, dtype=torch.float64)
        bounds_lower = torch.cat((bounds_lower, -caps))
        bounds_upper = torch.cat((bounds_upper, caps))
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
    if config.sector_net_cap is not None and len(groups):
        residual = max(residual, np.abs(groups @ weights).max() - config.sector_net_cap)
    if not np.isfinite(weights).all() or residual > 2e-6:
        raise FloatingPointError(f"allocation violates planned constraints: {residual}")
    return solution
