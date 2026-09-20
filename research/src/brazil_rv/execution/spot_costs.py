"""Dated ordinary-CNPJ spot attribution; amounts remain unrounded research costs."""

import numpy as np


EXECUTION_COMPONENTS = (
    "bundled",
    "b3_trading",
    "b3_clearing",
    "brokerage",
    "shortfall",
)


def execution_bps(config, session_date):
    """Stock/ETF rows, component columns. No fund, ADTV or day-trade discount.

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
    if not np.datetime64("2021-02-02") <= date <= np.datetime64("2024-12-30"):
        raise ValueError(
            "dated spot tariff is admitted only for 2021-02-02 through 2024-12-30"
        )
    row = [
        0,
        0.7 if config.spot_execution_phase == "auction" else 0.5,
        2.5,
        config.execution_brokerage_bps,
        config.execution_shortfall_bps,
    ]
    return np.array([row, row], dtype=float)
