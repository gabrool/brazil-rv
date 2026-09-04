from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

COTAHIST_YEARS: Final[tuple[int, ...]] = tuple(range(2009, 2027))
STORE_START: Final[date] = date(2010, 1, 4)
PRETRAIN_END: Final[date] = date(2021, 7, 30)
FINETUNE_START: Final[date] = date(2021, 8, 16)
DEVELOPMENT_END: Final[date] = date(2024, 12, 30)
OFFICIAL_START: Final[date] = date(2025, 1, 2)
OFFICIAL_END: Final[date] = date(2025, 12, 30)
FALLBACK_TEST_START: Final[date] = date(2026, 1, 2)
ACCUMULATED_TEST_AFTER: Final[date] = date(2026, 7, 17)

HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
PRIMARY_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5)
ALLOWED_LOOKBACKS: Final[tuple[int, ...]] = (20, 60, 120)
DEFAULT_LOOKBACK: Final[int] = 60
ALLOWED_SEEDS: Final[tuple[int, ...]] = (11, 29, 47)
V1_READ_SEEDS: Final[tuple[int, ...]] = (11, 29, 47, 61, 79, 97, 113, 131, 149, 167)
GBDT_SEEDS: Final[tuple[int, ...]] = (11, 29, 47, 61, 79)

# The M1 grid starts at 10:00. Index 345 is the 15:45 bar. Features consume
# indices [0, 345); entry is the open of index 345 (minute 346 in one-based
# terminology). This sample is synthesized by v2; v1 has no cutoff-345 row.
DECISION_MINUTE_INDEX: Final[int] = 345
FAST_PATCH_MINUTES: Final[int] = 5
FAST_REAL_PATCHES: Final[int] = DECISION_MINUTE_INDEX // FAST_PATCH_MINUTES

# Experiment 48's deployed store-v2 checkpoint was trained with these legacy
# v1 slow channels neutralized.  Any reuse of that encoder must apply the same
# mask to ``equity_slow.npy`` before the tensor reaches the model.
V1_STORE_V2_ZERO_SLOW_FIELDS: Final[tuple[int, ...]] = (
    1,
    2,
    3,
    12,
    13,
    14,
    15,
    16,
    18,
    20,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
)

UNIVERSE_PRIOR_SESSIONS: Final[int] = 20
UNIVERSE_MIN_TRADED: Final[int] = 15
UNIVERSE_MIN_MEDIAN_VOLUME_BRL: Final[float] = 2_000_000.0
UNIVERSE_MIN_PRIOR_CLOSE_BRL: Final[float] = 1.0
UNIVERSE_MIN_HISTORY: Final[int] = 60
SELECTION_EMBARGO_SESSIONS: Final[int] = 75
TARGETED_FUSION_GATE_BIAS: Final[float] = -2.0
SOFT_RANK_TEMPERATURE: Final[float] = 0.5

SLOW_FEATURES: Final[tuple[str, ...]] = (
    "log_return_1",
    "log_return_5",
    "log_return_21",
    "log_return_63",
    "log_return_126",
    "log_return_252",
    "momentum_12_1",
    "yang_zhang_vol_5",
    "yang_zhang_vol_20",
    "yang_zhang_vol_60",
    "vol_of_vol_60",
    "realized_skew_60",
    "realized_kurtosis_60",
    "max_return_21",
    "distance_52_week_high",
    "beta_60",
    "idiosyncratic_vol_60",
    "log_volume_mean_20",
    "volume_zscore_20",
    "amihud_20",
    "trade_count_zscore_20",
    "turnover_proxy_20",
    "high_low_range_1",
    "high_low_range_5",
    "close_location_value",
    "log_adjusted_close",
    "log_listing_age",
    "cluster_mean_return_5",
    "cluster_mean_return_21",
    "name_minus_cluster_return_5",
    "name_minus_cluster_return_21",
    "cluster_dispersion",
)

INTRADAY_DAILY_FEATURES: Final[tuple[str, ...]] = (
    "overnight_return",
    "intraday_return_1545",
    "overnight_return_sum_5",
    "overnight_return_sum_20",
    "intraday_return_sum_5",
    "intraday_return_sum_20",
    "overnight_minus_intraday",
    "overnight_minus_intraday_mean_20",
    "last_30_minute_return_share_lag1",
    "last_hour_volume_share_lag1",
    "close_vwap_deviation_lag1",
    "vwap_deviation_1545",
    "realized_vol_5m_1",
    "realized_vol_5m_5",
    "realized_vol_5m_20",
    "realized_skew_5m_20",
    "roll_spread_20",
    "corwin_schultz_spread_20",
    "intraday_range_1545",
    "volume_1545_relative_median_20",
)

SIDECAR_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "lending": (
        "loan_balance_to_volume_20",
        "loan_balance_change_1",
        "loan_balance_change_5",
        "loan_rate",
        "loan_rate_change_5",
    ),
    "events": (
        "sessions_until_announced_earnings",
        "sessions_since_earnings",
        "standardized_unexpected_earnings",
    ),
    "options": (
        "put_call_oi_ratio",
        "delta_oi_to_volume_1",
        "atm_iv_to_median_20",
        "put_skew",
    ),
    "oddlot": ("oddlot_volume_share", "oddlot_volume_share_change_5"),
    # Preserve Experiment 33's audited state semantics verbatim.  Generic
    # event/count/direction aliases would silently change the meaning of its
    # transformed preview and effective-date fields.
    "rebalance": tuple(
        f"{index}_{suffix}"
        for index in ("ibov", "ibxx", "smll")
        for suffix in (
            "current_weight_sqrt",
            "preview_delta_signed_sqrt",
            "preview_add",
            "preview_delete",
            "preview_pressure",
            "pre_effective_ramp",
            "post_effective_reversal",
        )
    ),
    "fundamentals": (
        "log_market_cap",
        "book_to_market",
        "gross_profitability",
        "leverage",
    ),
}


@dataclass(frozen=True)
class StoreSchema:
    slow_features: tuple[str, ...] = SLOW_FEATURES
    intraday_daily_features: tuple[str, ...] = INTRADAY_DAILY_FEATURES
    horizons: tuple[int, ...] = HORIZONS
    decision_minute_index: int = DECISION_MINUTE_INDEX

    def __post_init__(self) -> None:
        if len(self.slow_features) != 32:
            raise ValueError("The v2 slow feature contract must contain exactly 32 fields")
        if len(self.intraday_daily_features) != 20:
            raise ValueError("The v2 intraday-derived contract must contain exactly 20 fields")
