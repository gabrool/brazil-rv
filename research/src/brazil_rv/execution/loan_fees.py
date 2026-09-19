"""Dated B3 loan tariff components, in annual decimal rates.

Sources: B3's archived pre-platform guide, 125/2020-PRE and 081/2022-PRE.
These are the exchange charges, not loan rent or broker intermediation.
The historical R$10 contract minimum requires contract-level settlement; this
rate function deliberately does not pretend it is another annual spread.
"""

from typing import Literal

import numpy as np


LoanModality = Literal["normal", "direct", "otc", "compulsory"]
LOAN_FEE_CONVENTION = "B3_dated_components_20201026_20221114"


def loan_fee_rates(annual_rate, dates, *, modality: LoanModality = "normal"):
    """Return [..., trading/post-trading] rates for each accrual date.

    Before the electronic platform, ordinary loans use the voluntary 25 bp
    annual rate (50 bp for automatic/compulsory). From 2020-10-26 each
    component clips and rounds independently. The 2022 change applies to
    accrual from 2022-11-14, including existing contracts, not retrospectively
    to their earlier accrual. Inputs must already be causal contract rates.
    """
    rate, day = np.broadcast_arrays(
        np.asarray(annual_rate, dtype=np.float64),
        np.asarray(dates, dtype="datetime64[D]"),
    )
    # B3 specifies six decimals in decimal-rate notation, both before and
    # after applying each component's multiplier/floor/cap (081/2022 p. II).
    rounded = np.floor(rate * 1e6 + 0.5) / 1e6
    if modality == "normal":
        alpha, floor, old_cap, cap = (
            (0.02, 0.18),
            (0.000025, 0.000225),
            (0.001, 0.009),
            (0.0007, 0.0063),
        )
    elif modality == "direct":
        alpha, floor, old_cap, cap = (
            (0.025, 0.18),
            (0.00006, 0.00044),
            (0.0015, 0.011),
            (0.001, 0.0085),
        )
    elif modality == "otc":
        alpha, floor, old_cap, cap = (
            (0.0, 0.30),
            (0.0, 0.0005),
            (0.0, 0.015),
            (0.0, 0.012),
        )
    elif modality == "compulsory":
        alpha, floor, old_cap, cap = (
            (0.04, 0.36),
            (0.0002, 0.0018),
            (0.0025, 0.0225),
            (0.0025, 0.0225),
        )
    else:
        raise ValueError(f"unknown B3 loan modality: {modality}")
    caps = np.where((day >= np.datetime64("2022-11-14"))[..., None], cap, old_cap)
    fees = np.minimum(np.maximum(rounded[..., None] * alpha, floor), caps)
    fees = np.floor(fees * 1e6 + 0.5) / 1e6
    fixed = (0.0, 0.005 if modality == "compulsory" else 0.0025)
    return np.where((day < np.datetime64("2020-10-26"))[..., None], fixed, fees)
