"""Small inventory-aware preference model and shared constrained decision path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from .allocation import AllocationConfig, allocate
from .portfolio_account import PortfolioAccount, tensor
from .action_settlement import slice_action_settlements
from .share_distributions import slice_distributions, basket_betas, basket_prices
from .loan_contracts import slice_loan_settlements
from .stateful_ledger import (
    LedgerConfig,
    PortfolioTarget,
    daily_borrow_cost,
    simulate_stateful_ledger,
)
from brazil_rv.v2.evaluate import _ledger_inputs
from brazil_rv.v2.portfolio_inputs import HEADS, normalized_ranks


def policy_ledger_config(**changes):
    defaults = dict(
        volatility_balanced_entries=False,
        planned_absolute_net_cap=0.05,
    )
    return replace(LedgerConfig(), **(defaults | changes))


def daily_borrow(rates, dates, config, *, hedge_rates=None):
    dates = np.asarray(dates, dtype="datetime64[D]")
    if config.borrow_source == "uniform":
        rates = np.full_like(rates, config.annual_borrow_rate, dtype=np.float64)
    equity = daily_borrow_cost(rates, dates[..., None], config=config)
    hedge_rates = (
        config.hedge_annual_borrow_rate
        if hedge_rates is None
        else np.where(
            np.isnan(hedge_rates), config.hedge_annual_borrow_rate, hedge_rates
        )
    )
    hedge = np.broadcast_to(
        daily_borrow_cost(hedge_rates, dates, config=config), equity.shape[:-1]
    )
    return np.concatenate((equity, hedge[..., None]), axis=-1)


@dataclass
class PolicyData:
    inputs: object
    beta: np.ndarray
    diagonal: np.ndarray
    factor: np.ndarray
    prior_cdi: np.ndarray
    references: np.ndarray

    def __post_init__(self):
        inputs = self.inputs
        self.ranks = normalized_ranks(
            inputs.scores[..., HEADS], inputs.score_mask[..., HEADS]
        )
        self.valid = inputs.active & inputs.score_mask[..., HEADS].all(-1)
        borrow = daily_borrow(
            inputs.annual_borrow_rate_by_name,
            inputs.dates,
            policy_ledger_config(),
            hedge_rates=inputs.hedge_annual_borrow_rate,
        )
        self.volatility = np.sqrt(self.diagonal + self.beta**2 * self.factor[:, None])
        self.shortable = inputs.shortable_by_borrow_source["borrow_balance"]
        cdi_valid = np.isfinite(self.prior_cdi)
        prior_cdi = np.where(cdi_valid, self.prior_cdi, 0.0)
        prior_valid = np.concatenate((np.zeros_like(self.valid[:1]), self.valid[:-1]))
        difference = np.zeros_like(self.ranks)
        difference[1:] = self.ranks[1:] - self.ranks[:-1]
        difference *= (self.valid & prior_valid)[..., None]
        self.static = np.concatenate(
            (
                self.ranks,
                difference,
                self.valid[..., None],
                prior_valid[..., None],
                np.log(self.volatility)[..., None],
                self.beta[..., None],
                np.arcsinh(borrow[..., :-1] / 0.0001)[..., None],
                np.broadcast_to(
                    prior_cdi[:, None, None] / 0.0001, (*self.valid.shape, 1)
                ),
                np.broadcast_to(cdi_valid[:, None, None], (*self.valid.shape, 1)),
            ),
            axis=-1,
        ).astype(np.float32)
        if not np.isfinite(self.static[self.valid]).all():
            raise ValueError("policy observed features contain a non-finite value")

    def borrow_rates(self, day, config):
        # Realized financing uses the account's cost scenario; the decision path
        # below keeps its primary estimate fixed across cost stress replays.
        rates = self.inputs.hedge_annual_borrow_rate
        return daily_borrow(
            self.inputs.annual_borrow_rate_by_name[day],
            self.inputs.dates[day],
            config,
            hedge_rates=None if rates is None else rates[day],
        )

    def initial_account(self, start, config):
        hedge = (
            self.inputs.initial_hedge_reference_price
            if start == 0
            else self.inputs.bova11_close[start - 1]
        )
        account = PortfolioAccount.empty(
            np.r_[self.references[start], hedge], config=config
        )
        account.retired_sources.update(
            event.source_index
            for event in self.inputs.share_distributions
            if event.effective_session < start
        )
        return account

    def prior_unresolved(self, day):
        return (
            self.inputs.initial_unresolved_action
            if day == 0
            else ~self.inputs.action_session_resolved[day - 1]
        )

    def step(self, account, target, day, *, terminal=False):
        inputs = self.inputs
        n = len(inputs.security_ids)
        config = account.config
        rates = (
            np.full(n, config.annual_borrow_rate)
            if config.borrow_source == "uniform"
            else inputs.annual_borrow_rate_by_name[day]
        )
        hedge = config.hedge_annual_borrow_rate
        if inputs.hedge_annual_borrow_rate is not None and np.isfinite(
            inputs.hedge_annual_borrow_rate[day]
        ):
            hedge = inputs.hedge_annual_borrow_rate[day]
        return account.step(
            target,
            day=day,
            close=np.r_[inputs.raw_close[day], inputs.bova11_close[day]],
            cdi=float(inputs.cdi_returns[day]),
            session_date=inputs.dates[day],
            annual_borrow=np.r_[rates, hedge],
            loan_reference=np.full(n + 1, np.nan)
            if inputs.loan_reference_prices is None
            else inputs.loan_reference_prices[day],
            action_q=np.r_[inputs.action_shares_per_prior_share[day], 1.0],
            action_d=np.r_[inputs.action_cash_per_prior_share[day], 0.0],
            action_resolved=np.r_[inputs.action_session_resolved[day], True],
            successor=np.r_[inputs.action_successor_index[day], n],
            payment_session=np.r_[inputs.action_payment_session[day], -1],
            action_settlements=inputs.action_settlements,
            share_distributions=inputs.share_distributions,
            loan_cash_settlements=inputs.loan_cash_settlements,
            entry_fill_allowed=None
            if inputs.entry_fill_allowed is None
            else np.r_[inputs.entry_fill_allowed[day], True],
            terminal=terminal,
        )


class CalibratedPolicy(nn.Module):
    """Fit-only economic mapping; no unused neural computation at inference."""

    def __init__(self, calibration, *, uncertainty_scale=0.0):
        super().__init__()
        self.register_buffer("rank_mean", tensor(calibration.mean))
        self.register_buffer("rank_scale", tensor(calibration.scale))
        self.register_buffer("coefficient", tensor(calibration.coefficient))
        self.register_buffer("intercept", tensor(calibration.intercept))
        self.equal_rank = len(calibration.coefficient) == 1
        self.uncertainty_scale = uncertainty_scale
        self.register_buffer(
            "coefficient_covariance",
            tensor(calibration.covariance)
            if calibration.covariance is not None
            else torch.zeros((len(calibration.mean) + 1,) * 2, dtype=torch.float64),
            persistent=False,
        )

    def forward(self, static, state, ranks):
        if self.equal_rank:
            ranks = ranks.mean(-1, keepdim=True)
        return (
            ranks - self.rank_mean
        ) / self.rank_scale @ self.coefficient + self.intercept

    def forecast_uncertainty(self, ranks):
        if not self.uncertainty_scale:
            return None
        if self.equal_rank:
            ranks = ranks.mean(-1, keepdim=True)
        features = torch.cat(
            (
                torch.ones_like(ranks[..., :1]),
                (ranks - self.rank_mean) / self.rank_scale,
            ),
            -1,
        )
        variance = (features @ self.coefficient_covariance * features).sum(-1)
        return self.uncertainty_scale * variance.clamp_min(0).sqrt()

    def preference_for(self, data, day, names, state):
        return self(
            torch.from_numpy(data.static[day, names]),
            state,
            tensor(data.ranks[day, names]),
        )

    def market_return_for(self, day):
        """Zero unless an independently fitted, causal market forecast is supplied."""
        return 0.0


class PreferenceModel(CalibratedPolicy):
    """Zero residual starts exactly at the fit-only calibrated optimizer."""

    def __init__(self, data, calibration, fit_rows, *, uncertainty_scale=0.0):
        super().__init__(calibration, uncertainty_scale=uncertainty_scale)
        values = data.static[fit_rows][data.valid[fit_rows]]
        mean = values.mean(0, dtype=np.float64)
        scale = np.maximum(values.std(0, dtype=np.float64), 1e-4)
        # Eligible fitting rows make validity constant. Keep 0/1 units so a
        # held name's first missing score does not become a -10,000 input.
        mean[[6, 7, 12]], scale[[6, 7, 12]] = 0.0, 1.0
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(len(mean) + 10, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, static, state, ranks):
        base = super().forward(static, state, ranks)
        features = torch.cat(((static - self.mean) / self.scale, state.float()), -1)
        return base + 0.001 * self.network(features).squeeze(-1).double()


def state_features(weights, cash, restricted, age, adverse, pending, beta, volatility):
    """Only marked prior inventory; smooth scaling, no clipping of adverse moves."""
    stock = weights[:-1]
    common = torch.stack(
        (
            cash,
            restricted,
            weights.abs().sum(),
            weights.sum(),
            weights @ tensor(np.r_[beta, 1.0]),
        )
    )
    return torch.cat(
        (
            torch.stack(
                (
                    stock / 0.05,
                    torch.log1p(age / 20),
                    torch.asinh(adverse / (0.05 * tensor(volatility))),
                    pending / 0.05,
                    (stock != 0).double(),
                ),
                -1,
            ),
            common.expand(len(stock), -1),
        ),
        -1,
    )


def decide(
    data,
    model,
    day,
    weights,
    cash,
    restricted,
    age,
    adverse,
    pending,
    allowed,
    required,
    *,
    allocation=AllocationConfig(),
    locked=None,
    effective_beta=None,
    claim_exposure=None,
    loan_short_blocked=None,
    reserved_weights=None,
):
    """Compact exactly to available or held names, preserving all real inventory."""
    if model is None:
        # Cash fallback requests liquidation through the same actual-fill ledger.
        result = torch.zeros_like(weights)
        if locked is not None:
            result[:-1] = torch.where(torch.as_tensor(locked), weights[:-1], 0.0)
        if reserved_weights is not None:
            result[:-1] += reserved_weights[:-1]
        return result
    stock = weights[:-1]
    beta = data.beta[day] if effective_beta is None else effective_beta
    names = np.flatnonzero(allowed | (stock.detach().numpy() != 0))
    ids = np.r_[names, len(stock)]
    previous = weights[ids]
    lower = torch.maximum(previous.clamp_max(0), tensor(-allocation.stock_cap))
    upper = torch.minimum(previous.clamp_min(0), tensor(allocation.stock_cap))
    openings = torch.as_tensor(np.r_[allowed[names], True])
    shorts = torch.as_tensor(np.r_[data.shortable[day, names], True])
    for event in data.inputs.loan_cash_settlements:
        if event.prohibit_new_borrow and event.effective_session <= day:
            shorts[:-1] &= torch.as_tensor(names != event.security_index)
    lower = torch.where(openings & shorts, -allocation.stock_cap, lower)
    upper = torch.where(openings, allocation.stock_cap, upper)
    exits = torch.as_tensor(np.r_[required[names], False])
    lower, upper = torch.where(exits, 0, lower), torch.where(exits, 0, upper)
    lower = torch.cat((lower[:-1], tensor([-allocation.hedge_cap])))
    upper = torch.cat((upper[:-1], tensor([allocation.hedge_cap])))
    if claim_exposure is not None:
        exposure = torch.cat((claim_exposure[names], previous.new_zeros(1)))
        lower = torch.maximum(
            lower, torch.minimum(previous, -allocation.stock_cap - exposure)
        )
        upper = torch.minimum(
            upper, torch.maximum(previous, allocation.stock_cap - exposure)
        )
        lower[-1], upper[-1] = -allocation.hedge_cap, allocation.hedge_cap
    if loan_short_blocked is not None:
        lower = torch.where(torch.as_tensor(loan_short_blocked[ids]), 0.0, lower)
        upper = torch.maximum(upper, lower)
    if locked is not None:
        fixed = torch.as_tensor(np.r_[locked[names], False])
        lower, upper = (
            torch.where(fixed, previous, lower),
            torch.where(fixed, previous, upper),
        )
    if reserved_weights is not None:
        reserve = reserved_weights[ids]
        lower = torch.where(reserve > 0, torch.maximum(lower, reserve), lower)
        upper = torch.where(reserve < 0, torch.minimum(upper, reserve), upper)
        upper = torch.where(reserve > 0, torch.maximum(upper, lower), upper)
        lower = torch.where(reserve < 0, torch.minimum(lower, upper), lower)
    features = state_features(
        weights,
        cash,
        restricted,
        age,
        adverse,
        pending,
        beta,
        data.volatility[day],
    )[names]
    pref = model.preference_for(data, day, names, features)
    if claim_exposure is not None:
        # The basket is fixed until delivery. Its idiosyncratic covariance with
        # the same already-held underlying is an exact linear objective term.
        pref = (
            pref
            - allocation.risk_aversion
            * tensor(data.diagonal[day, names])
            * claim_exposure[names]
        )
    market = pref.new_tensor(model.market_return_for(day))
    # Stock preferences are benchmark residuals. A directional forecast must
    # enter stocks and the hedge coherently, once, through the same market beta.
    preference = torch.cat((pref + tensor(beta[names]) * market, market[None]))
    groups = None
    if allocation.sector_net_cap is not None:
        labels = data.sectors[day, names]
        baskets = []
        sector_labels = [labels]
        for event in data.inputs.share_distributions:
            if event.effective_session >= day or event.source_index not in names:
                continue
            legs = [
                leg
                for leg in event.legs
                if leg.delivery_session is None or leg.delivery_session > day
            ]
            if legs and stock[event.source_index].detach().item() != 0:
                sectors = data.sectors[day, [leg.successor_index for leg in legs]]
                baskets.append((event.source_index, legs, sectors))
                sector_labels.append(sectors)
        observed_labels = np.concatenate(sector_labels)
        known = np.unique(observed_labels[observed_labels != ""])
        groups = np.column_stack(
            (known[:, None] == labels[None, :], np.zeros(len(known)))
        ).astype(float)
        for source, legs, sectors in baskets:
            # A single leg has unit sector weight even before its first quote.
            values = (
                np.ones(1)
                if len(legs) == 1
                else basket_prices(legs, data.references[day])
                * [leg.shares_per_prior_share for leg in legs]
            )
            groups[:, np.flatnonzero(names == source)[0]] = (
                known[:, None] == sectors
            ) @ (values / values.sum())
    uncertainty = model.forecast_uncertainty(tensor(data.ranks[day, names]))
    chosen = allocate(
        preference,
        previous,
        beta=np.r_[beta[names], 1.0],
        idiosyncratic_variance=np.r_[data.diagonal[day, names], 1e-8],
        market_variance=float(data.factor[day]),
        daily_borrow=data.borrow_rates(day, policy_ledger_config())[ids],
        lower=lower,
        upper=upper,
        config=allocation,
        sector_exposure=groups,
        forecast_uncertainty=None
        if uncertainty is None
        else torch.cat((uncertainty, uncertainty.new_zeros(1))),
    )
    return torch.zeros_like(weights).scatter(0, torch.as_tensor(ids), chosen)


def account_decision(data, model, account, day, *, allocation=AllocationConfig()):
    account.prepare_day(day)
    locked = np.zeros(len(data.inputs.security_ids), bool)
    if account.distributions:
        locked[list(account.distributions)] = True
    claim_exposure = account.market_weights[:-1] - account.weights[:-1]
    claim_exposure = torch.where(torch.as_tensor(locked), 0.0, claim_exposure)
    allowed, required = account.eligibility(
        np.r_[data.valid[day], True], np.r_[data.prior_unresolved(day), False]
    )
    held = account.shares[:-1].detach().numpy() != 0
    age = tensor(np.where(held, day - account.entry_day[:-1], 0))
    adverse = (
        torch.where(
            torch.as_tensor(held),
            account.weights[:-1].sign()
            * (
                (account.shares[:-1] * account.marks[:-1]).abs()
                - account.cost_basis[:-1]
            ),
            0,
        )
        / account.nav
    )
    return decide(
        data,
        model,
        day,
        account.weights,
        account.cash / account.nav,
        account.restricted.sum() / account.nav,
        age,
        adverse,
        account.pending_exit[:-1] * account.weights[:-1],
        allowed[:-1],
        required[:-1],
        allocation=allocation,
        locked=locked,
        reserved_weights=account.reserved_weights,
        effective_beta=basket_betas(
            account.distributions, account.marks.detach().numpy(), data.beta[day]
        ),
        claim_exposure=claim_exposure,
        loan_short_blocked=account.loans.short_blocked(
            day, account.config.loan_recalls
        ),
    )


def portfolio_variance(data, day, weights):
    beta = tensor(np.r_[data.beta[day], 1.0])
    diagonal = tensor(np.r_[data.diagonal[day], 1e-8])
    return (weights.square() * diagonal).sum() + (
        weights @ beta
    ).square() * data.factor[day]


def ledger_arguments(data, start, stop):
    """Slice a causal accounting window with its pre-window reference state."""
    inputs = data.inputs
    score = data.ranks[start:stop].mean(-1)
    arguments = _ledger_inputs(inputs, score, data.valid[start:stop])
    # Slice only economic arrays; target payloads never enter this decision path.
    for key in (
        "active",
        "raw_close",
        "cdi_returns",
        "annual_borrow_rate_by_name",
        "borrow_rate_imputed",
        "borrow_rate_placeholder",
        "selection_volatility",
        "entry_fill_allowed",
        "hedge_beta",
        "hedge_beta_valid",
        "hedge_close",
        "hedge_annual_borrow_rate",
        "loan_reference_prices",
    ):
        if arguments[key] is not None:
            arguments[key] = arguments[key][start:stop]
    arguments["dates"] = inputs.dates[start:stop]
    action = arguments["action_terms"]
    arguments["action_terms"] = type(action)(
        *(
            getattr(action, name)[start:stop]
            for name in (
                "shares_per_prior_share",
                "cash_per_prior_share",
                "session_resolved",
                "has_action",
                "successor_index",
            )
        )
    )
    payments = inputs.action_payment_session[start:stop].copy()
    payments[payments >= 0] -= start
    arguments["action_payment_session"] = payments
    arguments["action_settlements"] = slice_action_settlements(
        inputs.action_settlements, start, stop
    )
    arguments["share_distributions"] = slice_distributions(
        inputs.share_distributions, start, stop
    )
    arguments["loan_cash_settlements"] = slice_loan_settlements(
        inputs.loan_cash_settlements, start, stop
    )
    arguments["initial_reference_price"] = data.references[start]
    arguments["initial_unresolved_action"] = data.prior_unresolved(start)
    arguments["initial_hedge_reference_price"] = (
        inputs.initial_hedge_reference_price
        if start == 0
        else inputs.bova11_close[start - 1]
    )
    # Betas here have already been resolved with their complete causal history.
    arguments["hedge_beta"] = data.beta[start:stop]
    arguments["hedge_beta_valid"] = np.ones_like(data.valid[start:stop])
    arguments["hedge_beta_history"] = None
    return arguments


def exact_replay(
    data,
    model,
    start,
    stop,
    *,
    config=None,
    targets=None,
    allocation=AllocationConfig(),
):
    """Independent ledger, same decision function; final boundary liquidates once."""
    inputs = data.inputs
    config = policy_ledger_config() if config is None else config
    # Stress scenarios change realized costs, not the frozen policy's estimate.
    arguments = ledger_arguments(data, start, stop)
    chosen = []
    prior_weights = []

    def callback(state):
        day = start + state.day
        current_model = model[day] if isinstance(model, Mapping) else model
        weights = tensor(np.r_[state.weights, state.hedge_weight])
        with torch.no_grad():
            target = (
                tensor(targets[state.day])
                if targets is not None
                else decide(
                    data,
                    current_model,
                    day,
                    weights,
                    tensor(state.free_cash_fraction),
                    tensor(state.restricted_cash_fraction),
                    tensor(state.holding_sessions),
                    tensor(state.marked_pnl_fraction),
                    tensor(
                        state.pending_entry_weights
                        + state.pending_exit_fractions * state.weights
                    ),
                    state.entry_allowed,
                    state.required_exit,
                    allocation=allocation,
                    locked=state.locked,
                    reserved_weights=tensor(np.r_[state.reserved_weights, 0.0]),
                    effective_beta=state.effective_beta,
                    claim_exposure=tensor(state.claim_exposure),
                    loan_short_blocked=state.loan_short_blocked,
                )
            )
        chosen.append(target.numpy())
        market_weights = weights.numpy().copy()
        market_weights[:-1][state.locked] = 0.0
        market_weights[:-1] += state.claim_exposure
        prior_weights.append(market_weights)
        return PortfolioTarget(target[:-1].numpy(), target[-1].item())

    result = simulate_stateful_ledger(
        **arguments,
        config=replace(
            config,
            loan_recalls=tuple(
                replace(
                    event,
                    notice_session=event.notice_session - start,
                    return_session=event.return_session - start,
                )
                for event in config.loan_recalls
                if event.notice_session < stop
            ),
        ),
        shortable=data.shortable[start:stop],
        portfolio_policy=callback,
    )
    # The final day is a forced boundary rather than a controller observation.
    chosen.append(np.zeros(len(inputs.security_ids) + 1))
    stock = (
        result.equity_market_weights[-2]
        if stop - start > 1
        else np.zeros(len(inputs.security_ids))
    )
    hedge = (
        result.hedge_signed_shares[-2] * result.hedge_mark_price[-2] / result.nav[-2]
        if stop - start > 1
        else 0.0
    )
    prior_weights.append(np.r_[stock, hedge])
    return result, np.asarray(chosen), np.asarray(prior_weights)
