"""Registered ledger-only signals and attribution of the original filled trades."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from brazil_rv.execution.stateful_ledger import (
    LedgerConfig,
    StatefulLedgerResult,
    equity_borrow_registration_fee,
)
from brazil_rv.modeling.metrics import average_ranks
from brazil_rv.v2.corporate_actions import AlignedActionTerms
from brazil_rv.v2.artifacts import inventory, sha256_file

if TYPE_CHECKING:
    from brazil_rv.v2.evaluate import EvaluationInputs


@dataclass(frozen=True)
class ExecutionPolicy:
    theta: float = 1.0
    horizons: tuple[int, ...] = (1, 2, 3, 5)
    inverse_volatility: bool = False
    buffer_per_quintile: int = 6

    def ledger_config(self) -> LedgerConfig:
        return replace(LedgerConfig(), buffer_per_side=5 * self.buffer_per_quintile)

    @property
    def key(self) -> str:
        heads = "_".join(str(value) for value in self.horizons)
        sizing = "inverse_sigma" if self.inverse_volatility else "equal"
        return f"theta{self.theta:g}_d{heads}_{sizing}_b{self.buffer_per_quintile}"

    @property
    def changed_dimensions(self) -> int:
        return sum(
            (
                self.theta != 1,
                self.horizons != (1, 2, 3, 5),
                self.inverse_volatility,
                self.buffer_per_quintile != 6,
            )
        )


def policy_grid() -> tuple[ExecutionPolicy, ...]:
    return tuple(
        ExecutionPolicy(theta, horizons, inverse)
        for theta, horizons, inverse in product(
            (1.0, 0.5, 0.25), ((1, 2, 3, 5), (3, 5, 10), (5, 10)), (False, True)
        )
    )


def load_selected_policy(root: Path, *, expected_result_sha256: str | None = None):
    """Bind the complete sealed sweep before applying its choice to new scores."""
    root = root.resolve(strict=True)
    inventory_path = root / "artifact_inventory.json"
    sealed = json.loads(inventory_path.read_text())
    if (
        sealed["status"] != "passed"
        or inventory(root, exclude=set(sealed["excluded_self"])) != sealed["files"]
    ):
        raise ValueError("execution sweep inventory differs from its sealed files")
    result_path = root / "execution_sweep_result.json"
    result_sha = sha256_file(result_path)
    if expected_result_sha256 is not None and result_sha != expected_result_sha256:
        raise ValueError("execution sweep result differs from its frozen binding")
    result = json.loads(result_path.read_text())
    if result["status"] != "complete" or not result["protected_inputs_exact"]:
        raise ValueError("execution policy requires a completed, exact-input sweep")
    if result["official_validation_accessed"] or result["test_accessed"]:
        raise PermissionError("execution policy source accessed protected outcomes")
    raw = result["decision"]["selected_policy"]
    policy = ExecutionPolicy(**{**raw, "horizons": tuple(raw["horizons"])})
    return policy, {
        "root": str(root),
        "result_sha256": result_sha,
        "inventory_sha256": sha256_file(inventory_path),
        "policy": {**asdict(policy), "horizons": list(policy.horizons)},
        "label": result["decision"]["label"],
    }


def ledger_gate_failures(summary, *, headline):
    failed = [key for key, count in summary["entry_defect_signatures"].items() if count]
    if headline:
        if not 1.5 <= summary["mean_gross_fraction_nav"] <= 2.25:
            failed.append("mean_gross_outside_1.5_to_2.25")
        if summary["mean_unresolved_stale_inventory_fraction_nav"] >= 0.02:
            failed.append("mean_unresolved_stale_inventory_at_least_0.02")
        if summary["insolvent"]:
            failed.append("insolvent")
        for side in ("long", "short"):
            if summary[f"mean_absolute_volatility_occupancy_deviation_{side}"] > 2:
                failed.append(f"mean_quintile_occupancy_deviation_{side}_above_two")
    return failed


def smooth_and_rank(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    theta: float,
    carry_sessions: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Consume only current/prior composites; reset after five missing sessions."""
    result = np.full(values.shape, np.nan, dtype=np.float64)
    supported = np.zeros(values.shape, dtype=np.bool_)
    state = np.full(values.shape[1], np.nan, dtype=np.float64)
    missing = np.zeros(values.shape[1], dtype=np.int64)
    for day in range(len(values)):
        observed = valid[day] & np.isfinite(values[day])
        continuing = observed & np.isfinite(state)
        entering = observed & ~np.isfinite(state)
        state[continuing] = (1.0 - theta) * state[continuing] + theta * values[
            day, continuing
        ]
        state[entering] = values[day, entering]
        missing[observed] = 0
        missing[~observed] += 1
        state[missing > carry_sessions] = np.nan
        supported[day] = np.isfinite(state)
        count = int(supported[day].sum())
        if count:
            result[day, supported[day]] = (
                2.0 * (average_ranks(state[supported[day]]) + 0.5) / count - 1.0
            )
    return result, supported


def traded_signal(inputs: EvaluationInputs, policy: ExecutionPolicy):
    from brazil_rv.v2.evaluate import _economics_signal

    composite, valid = _economics_signal(inputs, horizons=policy.horizons)
    return smooth_and_rank(
        composite,
        valid,
        theta=policy.theta,
        carry_sessions=0 if policy.theta == 1.0 else 5,
    )


def traded_readouts(inputs: EvaluationInputs, scores, mask) -> dict[str, NDArray]:
    # The broadcast is a temporary diagnostic view, never a replacement model panel.
    from brazil_rv.v2.research_rounds import (
        _single_persistence_series,
        _single_spread_series,
    )
    from brazil_rv.v2.evaluate import (
        _primary_daily_metrics,
        _primary_population_components,
    )

    view = replace(
        inputs,
        scores=np.broadcast_to(scores[..., None], inputs.scores.shape),
        score_mask=np.broadcast_to(mask[..., None], inputs.score_mask.shape),
    )
    primary = _primary_population_components(view)
    _, ic, _ = _primary_daily_metrics(*primary, inputs.dates)
    return {
        "neutral_ic": ic,
        "persistence_1": _single_persistence_series(view, lag=1),
        "persistence_5": _single_persistence_series(view, lag=5),
        "spread_bps_per_holding_session": _single_spread_series(view),
    }


def prior_median_volume(volume, valid, indices) -> NDArray[np.float64]:
    """Twenty completed valid daily observations; no imputation or current bar."""
    result = np.full((len(indices), volume.shape[1]), np.nan, dtype=np.float64)
    for row, day in enumerate(indices):
        if day < 20:
            continue
        window = np.asarray(volume[day - 20 : day], dtype=np.float64)
        supported = np.asarray(valid[day - 20 : day], dtype=bool).all(axis=0)
        supported &= np.isfinite(window).all(axis=0) & (window >= 0).all(axis=0)
        result[row, supported] = np.median(window[:, supported], axis=0)
    return result


def original_trade_attribution(
    result: StatefulLedgerResult,
    *,
    entry_liquid: NDArray[np.bool_],
    entry_known: NDArray[np.bool_],
    action_terms: AlignedActionTerms,
    annual_borrow_rate_by_name: NDArray[np.floating] | None,
    config: LedgerConfig,
) -> dict[str, object]:
    """Partition the existing fills; contribution bps use the full book's NAV.

    Liquid status belongs to the entry intention, including later partial fills.
    Shared hedge/funding effects use opening equity gross shares (entry turnover
    when opening gross is zero). Cash-only effects remain explicitly unallocated.
    """
    days, names = result.signed_shares.shape
    # Row 0 reconstructs the whole equity book as an independent reconciliation;
    # row 1 follows only the qualifying entry cohorts.
    shares = np.zeros((2, names), dtype=np.float64)
    marks = np.full(names, np.nan, dtype=np.float64)
    orders = {order.order_id: order for order in result.intended_orders}
    fills = defaultdict(list)
    for fill in result.fills:
        if fill.purpose != "hedge":
            fills[fill.fill_session].append(fill)
    fields = (
        "equity_gross_pnl_bps",
        "equity_trading_cost_bps",
        "equity_borrow_raw_bps",
        "equity_borrow_fee_bps",
        "shared_hedge_gross_pnl_bps",
        "shared_hedge_cost_bps",
        "shared_hedge_borrow_bps",
        "shared_funding_less_cdi_bps",
        "net_excess_contribution_bps",
        "other_trades_net_excess_contribution_bps",
        "unallocated_cash_bps",
        "liquid_gross_fraction_nav",
        "shared_allocation_fraction",
    )
    daily = {key: np.zeros(days, dtype=np.float64) for key in fields}
    entry_notional = np.zeros(3, dtype=np.float64)  # all, qualifying, unknown
    largest_pnl_error = 0.0
    largest_share_error = 0.0
    for day in range(days):
        opening_value = np.nansum(shares * marks, axis=1)
        opening_gross = np.nansum(np.abs(shares * marks), axis=1)
        claims = np.zeros(2)
        for name in np.flatnonzero(
            action_terms.has_action[day] & action_terms.session_resolved[day]
        ):
            q = float(action_terms.shares_per_prior_share[day, name])
            d = float(action_terms.cash_per_prior_share[day, name])
            successor = int(action_terms.successor_index[day, name])
            old = shares[:, name].copy()
            claims += old * d
            converted = (marks[name] - d) / q if old[0] != 0 and q > 0 else np.nan
            shares[:, name] = 0
            marks[name] = np.nan
            shares[:, successor] = old * q
            marks[successor] = converted
        shorts = shares < 0
        short_value = np.where(shorts, np.abs(shares * marks), 0.0)
        rates = (
            np.full(names, config.annual_borrow_rate)
            if config.borrow_source == "uniform"
            else np.asarray(annual_borrow_rate_by_name[day], dtype=np.float64)
        )
        fees = equity_borrow_registration_fee(rates, config=config)
        borrowed = np.nansum(
            short_value * np.expm1(np.log1p(rates) / config.annual_sessions), axis=1
        )
        borrow_fee = np.nansum(
            short_value * np.expm1(np.log1p(fees) / config.annual_sessions), axis=1
        )
        cash = np.zeros(2)
        costs = np.zeros(2)
        entries = np.zeros(2)
        for fill in fills[day]:
            name = fill.security_index
            if fill.purpose == "entry":
                order = orders[fill.order_id]
                fraction = float(
                    entry_liquid[order.decision_session, order.security_index]
                )
                unknown = not entry_known[order.decision_session, order.security_index]
                entry_notional += fill.gross_notional * np.asarray(
                    [1, fraction, unknown]
                )
                entries += fill.gross_notional * np.asarray([1, fraction])
            else:
                fraction = shares[1, name] / shares[0, name] if shares[0, name] else 0.0
            allocation = np.asarray([1.0, fraction])
            signed_quantity = fill.quantity * (1 if fill.side == "buy" else -1)
            shares[:, name] += allocation * signed_quantity
            if shares[0, name] == 0.0:
                shares[:, name] = 0.0
            cash -= allocation * signed_quantity * fill.price
            costs += allocation * fill.cost
        marks = result.mark_price[day].copy()
        pnl = np.nansum(shares * marks, axis=1) - opening_value + cash + claims
        scale = 1e4 / result.start_nav[day]
        largest_pnl_error = max(
            largest_pnl_error, abs(pnl[0] * scale - result.equity_gross_pnl_bps[day])
        )
        largest_share_error = max(
            largest_share_error,
            float(np.max(np.abs(shares[0] - result.signed_shares[day]))),
        )
        np.testing.assert_allclose(
            [
                pnl[0] * scale,
                borrowed[0] * scale,
                borrow_fee[0] * scale,
                costs[0] * scale,
            ],
            [
                result.equity_gross_pnl_bps[day],
                result.equity_borrow_raw_bps[day],
                result.equity_borrow_fee_bps[day],
                result.cost_bps[day] - result.hedge_cost_bps[day],
            ],
            rtol=1e-9,
            atol=1e-7,
            err_msg=f"original-trade attribution does not reconcile on {result.dates[day]}",
        )
        np.testing.assert_allclose(
            shares[0], result.signed_shares[day], rtol=1e-10, atol=1e-9
        )
        base = opening_gross if opening_gross[0] > 0 else entries
        fraction = base[1] / base[0] if base[0] > 0 else 0.0
        shared = (
            result.hedge_gross_pnl_bps[day]
            - result.hedge_cost_bps[day]
            - result.hedge_borrow_bps[day]
            + result.interest_bps[day]
            - result.cdi_benchmark_bps[day]
        )
        daily["equity_gross_pnl_bps"][day] = pnl[1] * scale
        daily["equity_trading_cost_bps"][day] = costs[1] * scale
        daily["equity_borrow_raw_bps"][day] = borrowed[1] * scale
        daily["equity_borrow_fee_bps"][day] = borrow_fee[1] * scale
        daily["shared_hedge_gross_pnl_bps"][day] = (
            fraction * result.hedge_gross_pnl_bps[day]
        )
        daily["shared_hedge_cost_bps"][day] = fraction * result.hedge_cost_bps[day]
        daily["shared_hedge_borrow_bps"][day] = fraction * result.hedge_borrow_bps[day]
        daily["shared_funding_less_cdi_bps"][day] = fraction * (
            result.interest_bps[day] - result.cdi_benchmark_bps[day]
        )
        contribution = (
            pnl[1] - costs[1] - borrowed[1] - borrow_fee[1]
        ) * scale + fraction * shared
        daily["net_excess_contribution_bps"][day] = contribution
        daily["unallocated_cash_bps"][day] = shared if base[0] == 0 else 0.0
        daily["other_trades_net_excess_contribution_bps"][day] = (
            pnl[0]
            - pnl[1]
            - costs[0]
            + costs[1]
            - borrowed[0]
            + borrowed[1]
            - borrow_fee[0]
            + borrow_fee[1]
        ) * scale + ((1 - fraction) * shared if base[0] > 0 else 0.0)
        daily["liquid_gross_fraction_nav"][day] = (
            np.nansum(np.abs(shares[1] * marks)) / result.nav[day]
        )
        daily["shared_allocation_fraction"][day] = fraction
    np.testing.assert_allclose(
        daily["net_excess_contribution_bps"]
        + daily["other_trades_net_excess_contribution_bps"]
        + daily["unallocated_cash_bps"],
        result.net_excess_all_cash_bps,
        rtol=1e-9,
        atol=1e-7,
    )
    return {
        "daily": {key: value.tolist() for key, value in daily.items()},
        "mean": {key: float(value.mean()) for key, value in daily.items()},
        "entry_notional_all_liquid_unknown": entry_notional.tolist(),
        "max_equity_pnl_reconciliation_error_bps": largest_pnl_error,
        "max_share_reconciliation_error": largest_share_error,
        "basis": "original_trade_contribution_full_book_NAV_no_replacement_no_rescaling",
    }
