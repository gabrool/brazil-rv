"""Explicit cash/custody terms supplementing gross same-security actions.

Gross price adjustment and model labels do not depend on account withholding.
Short compensation is a separately declared fraction of the gross distribution.
Bonus shares are economic inventory at effect, but cannot be disposed of before
credit. Loan quantity changes at effect, with all same-name returns deferred to
credit: an explicit conservative contract hypothesis, not a recovered B3 rule.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ActionSettlement:
    security_index: int
    effective_session: int
    available_session: int
    source: str
    bonus_delivery_session: int | None = None
    withholding_rate: float = 0.0
    short_cash_fraction: float = 1.0

    def __post_init__(self):
        if self.security_index < 0 or not self.source:
            raise ValueError("action settlement requires an identity and source")
        if self.available_session > self.effective_session:
            raise ValueError("action settlement cannot precede knowledge")
        if self.bonus_delivery_session is not None and (
            self.bonus_delivery_session < self.effective_session
        ):
            raise ValueError("bonus credit cannot precede economic effect")
        if (
            not 0 <= self.withholding_rate <= 1
            or not 0 <= self.short_cash_fraction <= 1
        ):
            raise ValueError("cash settlement fractions must lie in [0, 1]")

    def validate_action(self, q, cash, successor):
        if successor != self.security_index or cash < 0:
            raise ValueError("settlement terms require a same-security positive action")
        if self.bonus_delivery_session is not None and (q <= 1 or cash != 0):
            raise ValueError("delayed bonus requires additional shares and no cash")


def slice_action_settlements(events, start, stop):
    return tuple(
        replace(
            event,
            effective_session=event.effective_session - start,
            available_session=event.available_session - start,
            bonus_delivery_session=None
            if event.bonus_delivery_session is None
            else event.bonus_delivery_session - start,
        )
        for event in events
        if start <= event.effective_session < stop
    )
