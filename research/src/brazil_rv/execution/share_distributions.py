"""Sparse, source-bound corporate distributions on a replay's session/name axes.

Between recognition and delivery the predecessor represents a non-tradable basket
of signed share entitlements. It remains in NAV and exposure; custody netting and
release of short proceeds happen only when each leg is delivered.
"""

from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class FractionAuction:
    available_session: int
    cash_per_share: float
    payment_session: int
    provision_loan_fractions: bool = False
    zero_quantity_rent_through_payment: bool = False

    def __post_init__(self):
        if self.payment_session < self.available_session:
            raise ValueError("fraction payment cannot precede known auction terms")
        if not math.isfinite(self.cash_per_share) or self.cash_per_share <= 0:
            raise ValueError("fraction auction requires a positive sourced price")


@dataclass(frozen=True)
class ShareDelivery:
    successor_index: int
    shares_per_prior_share: float
    delivery_session: int | None
    fractional_auction: FractionAuction | None = None
    loan_principal_fraction: float = 1.0


@dataclass(frozen=True)
class ShareClaimPosition:
    session: int
    source_index: int
    successor_index: int
    signed_quantity: float
    mark: float
    delivery_session: int | None


@dataclass(frozen=True)
class ShareDistribution:
    source_index: int
    effective_session: int
    available_session: int
    legs: tuple[ShareDelivery, ...]
    source: str
    cash_per_prior_share: float = 0.0
    payment_session: int | None = None

    def __post_init__(self):
        if not self.legs or not self.source:
            raise ValueError("share distributions require legs and source evidence")
        if any(
            not math.isfinite(leg.loan_principal_fraction)
            or not 0 <= leg.loan_principal_fraction <= 1
            for leg in self.legs
        ) or not math.isclose(
            sum(leg.loan_principal_fraction for leg in self.legs), 1, abs_tol=1e-12
        ):
            raise ValueError("loan principal fractions must be explicit and sum to one")
        if self.available_session > self.effective_session:
            raise ValueError("later-known distribution terms cannot be backdated")
        if (
            not math.isfinite(self.cash_per_prior_share)
            or self.cash_per_prior_share < 0
        ):
            raise ValueError("distribution cash must be finite and nonnegative")
        if (
            self.payment_session is not None
            and self.payment_session < self.effective_session
        ):
            raise ValueError("cash payment cannot precede economic succession")
        for leg in self.legs:
            if leg.fractional_auction is not None and (
                len(self.legs) != 1
                or leg.delivery_session is None
                or leg.fractional_auction.available_session < leg.delivery_session
            ):
                raise ValueError(
                    "fraction auction requires a prior single-leg delivery"
                )
            if (
                leg.successor_index == self.source_index
                or not math.isfinite(leg.shares_per_prior_share)
                or leg.shares_per_prior_share <= 0
            ):
                raise ValueError(
                    "distribution legs require a distinct successor and positive ratio"
                )
            if (
                leg.delivery_session is not None
                and leg.delivery_session < self.effective_session
            ):
                raise ValueError("share delivery cannot precede economic succession")


def slice_distributions(distributions, start, stop):
    """Rebase a flat-start replay, retaining prior cancellation identities.

    Negative effective sessions retire the predecessor, without inventing opening
    inventory or historical entitlements. Future delivery/payment dates stay intact.
    """
    return tuple(
        replace(
            event,
            effective_session=event.effective_session - start,
            available_session=event.available_session - start,
            payment_session=None
            if event.payment_session is None
            else event.payment_session - start,
            legs=tuple(
                replace(
                    leg,
                    delivery_session=None
                    if leg.delivery_session is None
                    else leg.delivery_session - start,
                    fractional_auction=None
                    if leg.fractional_auction is None
                    else replace(
                        leg.fractional_auction,
                        available_session=leg.fractional_auction.available_session
                        - start,
                        payment_session=leg.fractional_auction.payment_session - start,
                    ),
                )
                for leg in event.legs
            ),
        )
        for event in distributions
        if event.effective_session < stop
    )


def basket_prices(legs, references):
    """Prices available before this realization; never search ahead for a quote.

    Supported existing-listed-successor cases require causal references.
    A newly listed/unpriced leg needs an evidenced valuation contract before replay;
    guessing its share of the predecessor value would fabricate a price.
    """
    prices = np.array([references[leg.successor_index] for leg in legs], dtype=float)
    if not np.all(np.isfinite(prices) & (prices > 0)):
        raise ValueError("unpriced distribution leg requires an evidenced valuation")
    return prices


def basket_betas(pending, references, beta):
    result = np.asarray(beta, dtype=float).copy()
    for name, legs in pending.items():
        values = basket_prices(legs, references) * [
            leg.shares_per_prior_share for leg in legs
        ]
        result[name] = (
            values
            @ np.array([beta[leg.successor_index] for leg in legs])
            / values.sum()
        )
    return result
