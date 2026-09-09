from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

PROTOCOL_SCHEMA: Final[str] = "BRAZIL_RV_V2_PROTOCOL_V2"
MODEL_INPUT_SCHEMA: Final[str] = "BRAZIL_RV_V2_MODEL_INPUT_V2"
CHECKPOINT_INPUT_SCHEMA: Final[str] = "BRAZIL_RV_V2_CHECKPOINT_INPUT_V2"
RAW_PATIENCE_SCHEMA: Final[str] = "BRAZIL_RV_V2_RAW_PATIENCE_V2"
FINAL_EMA_SCHEMA: Final[str] = "BRAZIL_RV_V2_FINAL_EMA_0995_V2"
TRAINING_STAGE_SCHEMA: Final[str] = "BRAZIL_RV_V2_TRAINING_STAGE_V2"
SCORE_ARTIFACT_SCHEMA: Final[str] = "BRAZIL_RV_V2_SCORE_ARTIFACT_V2"
GBDT_MODELS_SCHEMA: Final[str] = "BRAZIL_RV_V2_GBDT_MODELS_V2"
RUN_MANY_PLAN_SCHEMA: Final[str] = "BRAZIL_RV_V2_RUN_MANY_PLAN_V2"
RUN_MANY_RESULT_SCHEMA: Final[str] = "BRAZIL_RV_V2_RUN_MANY_V2"
DECISION_SAMPLE_SCHEMA: Final[str] = "BRAZIL_RV_V2_DECISION_SAMPLE_V2"
DECISION_FEATURE_ALIGNMENT: Final[str] = "canonical_decision_snapshot_t"
DECISION_FEATURE_CONTRACT: Final[dict[str, object]] = {
    "all_stages": DECISION_FEATURE_ALIGNMENT,
    "daily_market_source": "through_t_minus_1",
    "decision_available_publications": "through_decision_t",
    "consumer_side_shift": False,
}
FEATURE_AGE_CONTRACT: Final[dict[str, object]] = {
    "unit": "exchange_sessions",
    "definition": "sessions since the most recent usable source observation",
    "current_observation": 0.0,
    "unknown_or_left_censored_sentinel": -1.0,
    "model_transform": "log1p(min(age,252))/log1p(252)",
    "age_known_is_independent_of_feature_validity": True,
}

COTAHIST_YEARS: Final[tuple[int, ...]] = tuple(range(2009, 2027))
STORE_START: Final[date] = date(2010, 1, 4)
PRETRAIN_END: Final[date] = date(2016, 6, 30)
FINETUNE_START: Final[date] = date(2016, 7, 18)
DEVELOPMENT_END: Final[date] = date(2024, 12, 30)
OFFICIAL_START: Final[date] = date(2025, 1, 2)
OFFICIAL_END: Final[date] = date(2025, 12, 30)
FALLBACK_TEST_START: Final[date] = date(2026, 1, 2)
ACCUMULATED_TEST_AFTER: Final[date] = date(2026, 7, 17)

HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10)
PRIMARY_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5)
TRADED_PRIMARY_HORIZONS: Final[tuple[int, ...]] = (3, 5, 10)
DEVELOPMENT_FOLDS: Final[tuple[str, ...]] = tuple(f"F{x}" for x in range(1, 15))
CONFIRMATION_SEEDS: Final[tuple[int, ...]] = (61, 79, 97)
DEFAULT_HORIZON_LOSS_WEIGHTS: Final[tuple[float, ...]] = (0.2,) * 5
ALLOWED_LOOKBACKS: Final[tuple[int, ...]] = (20, 60, 120)
DEFAULT_LOOKBACK: Final[int] = 60
ALLOWED_SEEDS: Final[tuple[int, ...]] = (11, 29, 47)
V1_READ_SEEDS: Final[tuple[int, ...]] = (11, 29, 47, 61, 79, 97, 113, 131, 149, 167)
GBDT_SEEDS: Final[tuple[int, ...]] = (11, 29, 47, 61, 79)

# Rev-4 research binds every train/selection consumer to this virtual store
# view.  The physical ``target_primary`` array remains available only as the
# explicitly reported legacy scaled-target readout.
REGISTERED_PRIMARY_TARGET: Final[str] = "target_primary_neutral"
REGISTERED_PRIMARY_TARGET_MASK: Final[str] = "target_primary_neutral_valid"
TARGET_NEUTRALIZATION_FEATURES: Final[tuple[str, ...]] = (
    "yang_zhang_vol_20",
    "beta_60",
    "log_volume_mean_20",
)
TARGET_NEUTRALIZATION_VOL_GROUPS: Final[int] = 10
TARGET_NEUTRALIZATION_BETA_GROUPS: Final[int] = 5
TARGET_NEUTRALIZATION_MIN_NONLINEAR_NAMES: Final[int] = 40

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
    "observed_history_age_sessions",
    "observed_history_left_censored",
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
INTRADAY_PRIOR_SESSION_FEATURES: Final[tuple[str, ...]] = (
    "last_30_minute_return_share_lag1",
    "last_hour_volume_share_lag1",
    "close_vwap_deviation_lag1",
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
        "sessions_since_financial_filing",
        "standardized_unexpected_earnings",
    ),
    "options": (
        "put_call_log_oi_ratio",
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
        "liabilities_to_assets",
    ),
}


@dataclass(frozen=True)
class StoreSchema:
    slow_features: tuple[str, ...] = SLOW_FEATURES
    intraday_daily_features: tuple[str, ...] = INTRADAY_DAILY_FEATURES
    horizons: tuple[int, ...] = HORIZONS
    decision_minute_index: int = DECISION_MINUTE_INDEX

    def __post_init__(self) -> None:
        if not self.slow_features or not self.intraday_daily_features:
            raise ValueError("store feature families must be nonempty")
        if len(set(self.slow_features)) != len(self.slow_features) or len(
            set(self.intraday_daily_features)
        ) != len(self.intraday_daily_features):
            raise ValueError("store feature names must be unique within each family")
        if self.decision_minute_index <= 0:
            raise ValueError("decision minute index must be positive")
