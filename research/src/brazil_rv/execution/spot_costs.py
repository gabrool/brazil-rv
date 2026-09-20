"""Dated ordinary-CNPJ spot attribution; amounts remain unrounded research costs."""

import numpy as np
from dataclasses import dataclass


EXECUTION_COMPONENTS = (
    "bundled",
    "b3_trading",
    "b3_clearing",
    "brokerage",
    "shortfall",
)


@dataclass(frozen=True)
class MonthlySpotTariff:
    valid_from: str
    valid_to: str
    available_date: str
    trading_bps: float
    source_sha256: str

    def __post_init__(self):
        if not self.valid_from <= self.valid_to < "2021-02-02":
            raise ValueError(
                "monthly global-market tariff requires its old validity interval"
            )
        if not np.isfinite(self.trading_bps) or not 0.2 <= self.trading_bps <= 0.5:
            raise ValueError("monthly rate lies outside the sourced progressive tariff")


def execution_bps(config, session_date):
    """Stock/ETF rows, component columns. No fund or client-ADTV discount.

    018/2013 and 061/2013 use previous-month GLOBAL-market ADTV and 2.75bp
    clearing. Explicit recovered monthly rows keep validity and availability
    separate. Missing inputs require a labelled bound, never account turnover.
    B3 177/2020 starts on 2021-02-02; 017/2023 consolidates the same rates.
    This admission ends with the development calendar, before later tariff changes.
    The caller must reject opposite same-security/day fills in this bounded mode.
    """
    if config.spot_cost_model == "bundled":
        return np.array(
            [
                [config.cost_bps_per_side, 0, 0, 0, 0],
                [config.hedge_cost_bps_per_side, 0, 0, 0, 0],
            ],
            dtype=float,
        )
    date = np.datetime64(session_date, "D")
    if not np.datetime64("2016-07-18") <= date <= np.datetime64("2024-12-30"):
        raise ValueError(
            "dated spot tariff is admitted only for 2016-07-18 through 2024-12-30"
        )
    trading = 0.5
    if date < np.datetime64("2021-02-02") and config.spot_execution_phase == "regular":
        selected = [
            t
            for t in config.monthly_spot_tariffs
            if max(t.valid_from, t.available_date) <= str(date) <= t.valid_to
        ]
        trading = (
            selected[0].trading_bps if selected else config.unrecovered_spot_trading_bps
        )
        if trading is None:
            raise ValueError(
                "missing historical market tariff requires an explicit sourced bound"
            )
    row = [
        0,
        0.7 if config.spot_execution_phase == "auction" else trading,
        2.75 if date < np.datetime64("2021-02-02") else 2.5,
        config.execution_brokerage_bps,
        config.execution_shortfall_bps,
    ]
    return np.array([row, row], dtype=float)
