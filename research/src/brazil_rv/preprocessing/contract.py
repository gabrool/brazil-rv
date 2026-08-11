from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

import numpy as np

CONTRACT_VERSION = "M1_FEATURES_INTRADAY_DI_MASKED_CONTEXT_HUMAN_PRIORS_V4"
LOCAL_CONTEXT_AVAILABILITY_RULE = (
    "Local instruments never gate B3 samples; unavailable instruments are masked "
    "by context_data_ready."
)
SAMPLE_ELIGIBILITY_RULE = (
    "At least MIN_ACTIVE_EQUITIES satisfy membership and equity feature readiness; "
    "local and global context readiness never gate samples."
)

PROJECT_ROOT = Path(r"C:\Brazil-RV")
UNIVERSE_POINTER = (
    PROJECT_ROOT / "quant-data/b3/interim/universe/pit_v1_canonical_path.txt"
)
ASSIGNMENTS_POINTER = (
    PROJECT_ROOT / "quant-data/b3/interim/m1_reconciliation/accepted/"
    "xp_accepted_assignments_v1_path.txt"
)
COTAHIST_POINTER = (
    PROJECT_ROOT / "quant-data/b3/interim/b3/cotahist/parsed_canonical_path.txt"
)
CONTEXT_POINTER = PROJECT_ROOT / "quant-data/b3/raw/xp/milestone3_long_history_path.txt"
GLOBAL_SOURCE_POINTER = (
    PROJECT_ROOT
    / "quant-data/b3/interim/global_context/global_context_canonical_path.txt"
)
HUMAN_PRIORS_POINTER = (
    PROJECT_ROOT / "quant-data/b3/interim/b3/human_priors_v1_canonical_path.txt"
)
CATALOGUE_PATH = (
    PROJECT_ROOT / "quant-data/b3/raw/xp/catalogue_canonical/symbol_catalogue.parquet"
)
OUTPUT_BASE = PROJECT_ROOT / "quant-data/b3/processed/features"
CANONICAL_OUTPUT_POINTER = OUTPUT_BASE / "m1_features_canonical_path.txt"

EXPECTED_EQUITIES = 158
LOCAL_CONTEXT_SYMBOLS = (
    "WIN$",
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
    "DI1$N",
)
LOCAL_CONTEXT_FAMILIES = (
    "EQUITY_FUTURE",
    "FX_FUTURE",
    "RATE_FUTURE",
    "RATE_FUTURE",
    "RATE_FUTURE",
    "RATE_FUTURE",
    "RATE_FUTURE_LIQUIDITY_SELECTED_UNADJUSTED",
)
FIXED_RATE_CONTEXT_SYMBOLS = ("DI1F27", "DI1F28", "DI1F29", "DI1F31")
LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL = "DI1$N"
RATE_CONTEXT_SYMBOLS = frozenset(
    (*FIXED_RATE_CONTEXT_SYMBOLS, LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL)
)
EXPOSURE_BETA_CONTEXT_SYMBOLS = (
    "WIN$",
    "WDO$",
    "DI1F27",
    "DI1F28",
    "DI1F29",
    "DI1F31",
)
GLOBAL_CONTEXT_SYMBOLS = (
    "ES.v.0",
    "NQ.v.0",
    "ZT.v.0",
    "ZN.v.0",
    "CL.v.0",
    "HG.v.0",
    "6E.v.0",
    "6M.v.0",
)
GLOBAL_CONTEXT_FAMILIES = (
    "US_EQUITY_ES",
    "US_EQUITY_NQ",
    "US_RATES_ZT",
    "US_RATES_ZN",
    "COMMODITY_CL",
    "COMMODITY_HG",
    "FX_6E",
    "FX_6M",
)
GLOBAL_QUOTE_DIRECTIONS = (
    "USD_PER_INDEX_POINT",
    "USD_PER_INDEX_POINT",
    "USD_PER_TREASURY_POINT",
    "USD_PER_TREASURY_POINT",
    "USD_PER_BARREL",
    "USD_PER_POUND",
    "USD_PER_EUR",
    "USD_PER_MXN",
)
GLOBAL_PROVIDER = "Databento"
GLOBAL_DATASET = "GLBX.MDP3"
GLOBAL_SCHEMA = "ohlcv-1m"
GLOBAL_DATABENTO_VERSION = "0.81.0"
GLOBAL_CONTINUOUS_ROLL_RULE = "highest_prior_day_volume"
GLOBAL_AVAILABILITY_RULE = "bar_start_utc + 1 minute <= decision_time_utc"

EQUITY_SESSION_START_MINUTE = 10 * 60
EQUITY_SESSION_MINUTES = 405
CONTEXT_SESSION_START_MINUTE = 9 * 60
CONTEXT_SESSION_MINUTES = 465
GLOBAL_SESSION_START_MINUTE = 4 * 60 + 30
GLOBAL_SESSION_END_MINUTE = 14 * 60 + 45
GLOBAL_SESSION_MINUTES = GLOBAL_SESSION_END_MINUTE - GLOBAL_SESSION_START_MINUTE
DYNAMIC_CHANNELS = (
    "open_move_normalized",
    "high_move_normalized",
    "low_move_normalized",
    "close_move_normalized",
    "volume_surprise",
    "observed",
    "return_since_open_normalized",
    "return_15m_normalized",
    "return_30m_normalized",
    "return_60m_normalized",
    "realized_vol_15m_log_ratio",
    "realized_vol_30m_log_ratio",
    "realized_vol_60m_log_ratio",
    "cumulative_volume_surprise",
    "session_range_position",
    "observed_fraction_30m",
    "market_median_return_15m",
    "market_median_return_60m",
    "market_breadth_15m",
    "market_breadth_60m",
    "market_dispersion_15m",
    "market_dispersion_60m",
    "cross_section_return_rank_15m",
    "cross_section_return_rank_60m",
    "cross_section_volume_rank",
    "cross_section_volatility_rank_30m",
)
EQUITY_PEER_CHANNELS = (
    "return_vs_selected_peer_median_15m",
    "return_vs_selected_peer_median_60m",
    "selected_peer_return_rank_15m",
    "selected_peer_return_rank_60m",
    "return_vs_issuer_peer_median_15m",
    "return_vs_issuer_peer_median_60m",
)
EQUITY_PEER_VALID_CHANNELS = (
    "selected_peer_15m_valid",
    "selected_peer_60m_valid",
    "issuer_peer_15m_valid",
    "issuer_peer_60m_valid",
)
SLOW_CHANNELS = (
    "vol_regime",
    "overnight_gap_normalized",
    "previous_close_to_close_return_normalized",
    "previous_open_to_close_return_normalized",
    "previous_last_60m_return_normalized",
    "previous_realized_vol_log_ratio",
    "previous_volume_log_ratio",
    "return_5d_normalized",
    "return_20d_normalized",
    "realized_vol_5d_log_ratio",
    "realized_vol_20d_log_ratio",
    "vol_of_vol_20d",
    "median_daily_real_volume_20d_log_scale",
    "median_daily_dollar_volume_20d_log_scale",
    "daily_dollar_volume_regime_20d",
    "observed_fraction_5d",
    "observed_fraction_20d",
    "overnight_gap_cross_section_rank",
    "dollar_volume_cross_section_rank",
    "realized_vol_cross_section_rank",
    "beta_to_WIN",
    "beta_to_WDO",
    "beta_to_DI1F27",
    "beta_to_DI1F28",
    "beta_to_DI1F29",
    "beta_to_DI1F31",
    "weekday_sin",
    "weekday_cos",
    "month_end_proximity",
    "quarter_end_proximity",
    "prior_rate_level_scaled",
    "time_to_expiry_scaled",
)
EQUITY_SLOW_CHANNELS = SLOW_CHANNELS
CONTEXT_SLOW_CHANNELS = SLOW_CHANNELS
GLOBAL_SLOW_CHANNELS = (
    "vol_regime",
    "previous_b3_close_to_decision_return_normalized",
    "previous_futures_session_close_to_close_return_normalized",
    "previous_futures_session_open_to_close_return_normalized",
    "previous_futures_session_last_60m_return_normalized",
    "previous_futures_session_realized_vol_log_ratio",
    "previous_futures_session_volume_log_ratio",
    "return_5_sessions_normalized",
    "return_20_sessions_normalized",
    "realized_vol_5_sessions_log_ratio",
    "realized_vol_20_sessions_log_ratio",
    "vol_of_vol_20_sessions",
    "median_daily_volume_20_sessions_log_scale",
    "unused_equity_liquidity_13",
    "unused_equity_liquidity_14",
    "unused_equity_liquidity_15",
    "observed_session_fraction_5_sessions",
    "unused_equity_context_17",
    "unused_equity_context_18",
    "unused_equity_context_19",
    "unused_equity_context_20",
    "unused_equity_context_21",
    "unused_equity_context_22",
    "unused_equity_context_23",
    "unused_equity_context_24",
    "unused_equity_context_25",
    "weekday_sin",
    "weekday_cos",
    "month_end_proximity",
    "quarter_end_proximity",
    "unused_local_rate_level",
    "time_to_expiry_scaled",
)
GLOBAL_UNUSED_SLOW_CHANNEL_INDICES = (
    *range(13, 16),
    *range(17, 26),
    30,
)
DYNAMIC_CHANNEL_COUNT = len(DYNAMIC_CHANNELS)
SLOW_CHANNEL_COUNT = len(SLOW_CHANNELS)
EQUITY_SLOW_COUNT = SLOW_CHANNEL_COUNT
CONTEXT_SLOW_COUNT = SLOW_CHANNEL_COUNT
GLOBAL_SLOW_COUNT = len(GLOBAL_SLOW_CHANNELS)
PATCH_INPUT_WIDTH = 5 * DYNAMIC_CHANNEL_COUNT
if DYNAMIC_CHANNEL_COUNT != 26 or SLOW_CHANNEL_COUNT != 32 or GLOBAL_SLOW_COUNT != 32:
    raise ValueError("Feature channel contract has the wrong width")

EXPECTED_DATE_COUNT = 1248
EXPECTED_ELIGIBLE_DATE_COUNT = 1_228
EXPECTED_SAMPLE_COUNT = 67_540
EXPECTED_FIRST_ELIGIBLE_DATE = date(2021, 8, 16)
EXPECTED_LAST_ELIGIBLE_DATE = date(2026, 7, 17)
EXPECTED_ELIGIBLE_DATES_WITH_UNAVAILABLE_LOCAL_CONTEXT = 145

DECISION_EQUITY_INDICES = tuple(15 + 5 * index for index in range(55))
DECISION_CONTEXT_INDICES = tuple(75 + 5 * index for index in range(55))
DECISION_GLOBAL_INDICES = tuple(345 + 5 * index for index in range(55))
DECISION_TIMES = tuple(
    time((EQUITY_SESSION_START_MINUTE + index) // 60, index % 60)
    for index in DECISION_EQUITY_INDICES
)
HORIZONS = (30, 60, 120)

VOL_WARMUP_VALID_DAYS = 20
VOL_EWMA_HALF_LIFE_DAYS = 20
MIN_ADJACENT_RETURNS_PER_DAY = 30
PRICE_VOL_FLOOR = 1e-5
RATE_VOL_FLOOR_BP = 0.01
PRICE_VOL_REFERENCE = 1e-4
RATE_VOL_REFERENCE_BP = 0.1
RATE_PERCENT_MIN = 1.0
RATE_PERCENT_MAX = 50.0
VOLUME_LOOKBACK_SESSIONS = 20
VOLUME_MIN_OBSERVATIONS = 10
MAD_NORMALIZATION = 1.4826
VOLUME_MAD_FLOOR = 0.1
PRICE_FEATURE_CLIP = 10.0
VOLUME_FEATURE_CLIP = 6.0
VOL_REGIME_CLIP = 4.0
MIN_ACTIVE_EQUITIES = 30
VOL_EWMA_ALPHA = 1 - 2 ** (-1 / VOL_EWMA_HALF_LIFE_DAYS)
RETURN_WINDOWS = (15, 30, 60)
REALIZED_VOL_MIN_FRACTION = 0.80
REALIZED_VOL_LOG_FLOOR = 1e-6
REALIZED_VOL_LOG_CLIP = 4.0
OBSERVED_FRACTION_WINDOW = 30
SLOW_SHORT_WINDOW = 5
SLOW_LONG_WINDOW = 20
SLOW_SHORT_MIN_VALID = 4
SLOW_LONG_MIN_VALID = 15
BETA_EWMA_HALF_LIFE_DAYS = 20
BETA_EWMA_ALPHA = 1 - 2 ** (-1 / BETA_EWMA_HALF_LIFE_DAYS)
BETA_MIN_PAIRED_SESSIONS = 20
BETA_VARIANCE_FLOOR = 1e-12
BETA_CLIP = 5.0
REAL_VOLUME_LOG_CENTER = 12.0
REAL_VOLUME_LOG_SCALE = 4.0
DOLLAR_VOLUME_LOG_CENTER = 18.0
DOLLAR_VOLUME_LOG_SCALE = 4.0
LIQUIDITY_SELECTED_RATE_ZERO_SLOW_CHANNEL_INDICES = (
    1,
    2,
    *range(13, 15),
    *range(17, 26),
    30,
    31,
)


@dataclass(frozen=True)
class OutputArraySpec:
    dtype: np.dtype
    shape: tuple[int, ...]


def output_array_specs(date_count: int) -> dict[str, OutputArraySpec]:
    d = date_count
    n = EXPECTED_EQUITIES
    c = len(LOCAL_CONTEXT_SYMBOLS)
    g = len(GLOBAL_CONTEXT_SYMBOLS)
    q = len(DECISION_EQUITY_INDICES)
    h = len(HORIZONS)
    f = len(DYNAMIC_CHANNELS)
    return {
        "equity_features.npy": OutputArraySpec(
            np.dtype(np.float32), (d, n, EQUITY_SESSION_MINUTES, f)
        ),
        "equity_slow.npy": OutputArraySpec(
            np.dtype(np.float32), (d, n, len(EQUITY_SLOW_CHANNELS))
        ),
        "equity_membership.npy": OutputArraySpec(np.dtype(bool), (d, n)),
        "equity_data_ready.npy": OutputArraySpec(np.dtype(bool), (d, n)),
        "equity_peer_features.npy": OutputArraySpec(
            np.dtype(np.float32),
            (d, n, EQUITY_SESSION_MINUTES, len(EQUITY_PEER_CHANNELS)),
        ),
        "equity_peer_valid.npy": OutputArraySpec(
            np.dtype(bool),
            (d, n, EQUITY_SESSION_MINUTES, len(EQUITY_PEER_VALID_CHANNELS)),
        ),
        "context_features.npy": OutputArraySpec(
            np.dtype(np.float32), (d, c, CONTEXT_SESSION_MINUTES, f)
        ),
        "context_slow.npy": OutputArraySpec(
            np.dtype(np.float32), (d, c, len(CONTEXT_SLOW_CHANNELS))
        ),
        "context_data_ready.npy": OutputArraySpec(np.dtype(bool), (d, c)),
        "raw_returns.npy": OutputArraySpec(np.dtype(np.float32), (d, n, q, h)),
        "global_features.npy": OutputArraySpec(
            np.dtype(np.float32), (d, g, GLOBAL_SESSION_MINUTES, f)
        ),
        "global_slow.npy": OutputArraySpec(
            np.dtype(np.float32), (d, g, q, len(GLOBAL_SLOW_CHANNELS))
        ),
        "global_data_ready.npy": OutputArraySpec(np.dtype(bool), (d, g, q)),
        "targets.npy": OutputArraySpec(np.dtype(np.float32), (d, n, q, h)),
        "label_mask.npy": OutputArraySpec(np.dtype(bool), (d, n, q, h)),
        "cross_section_median.npy": OutputArraySpec(np.dtype(np.float32), (d, q, h)),
        "horizon_mask.npy": OutputArraySpec(np.dtype(bool), (d, q, h)),
    }


def manifest_constants() -> dict[str, object]:
    return {
        "expected_equities": EXPECTED_EQUITIES,
        "local_context_symbols": list(LOCAL_CONTEXT_SYMBOLS),
        "local_context_families": list(LOCAL_CONTEXT_FAMILIES),
        "fixed_rate_context_symbols": list(FIXED_RATE_CONTEXT_SYMBOLS),
        "liquidity_selected_rate_context_symbol": (
            LIQUIDITY_SELECTED_RATE_CONTEXT_SYMBOL
        ),
        "exposure_beta_context_symbols": list(EXPOSURE_BETA_CONTEXT_SYMBOLS),
        "rate_quote_unit": "annual_percentage_rate",
        "rate_price_change_unit": "basis_points",
        "rate_percentage_range": [RATE_PERCENT_MIN, RATE_PERCENT_MAX],
        "rate_expiry_distance_basis": (
            "calendar_days_to_authoritative_contract_expiry"
        ),
        "global_context_symbols": list(GLOBAL_CONTEXT_SYMBOLS),
        "global_context_families": list(GLOBAL_CONTEXT_FAMILIES),
        "global_quote_directions": list(GLOBAL_QUOTE_DIRECTIONS),
        "global_provider": GLOBAL_PROVIDER,
        "global_dataset": GLOBAL_DATASET,
        "global_schema": GLOBAL_SCHEMA,
        "global_databento_version": GLOBAL_DATABENTO_VERSION,
        "global_continuous_roll_rule": GLOBAL_CONTINUOUS_ROLL_RULE,
        "global_availability_rule": GLOBAL_AVAILABILITY_RULE,
        "expected_date_count": EXPECTED_DATE_COUNT,
        "expected_eligible_date_count": EXPECTED_ELIGIBLE_DATE_COUNT,
        "expected_sample_count": EXPECTED_SAMPLE_COUNT,
        "expected_first_eligible_date": str(EXPECTED_FIRST_ELIGIBLE_DATE),
        "expected_last_eligible_date": str(EXPECTED_LAST_ELIGIBLE_DATE),
        "expected_eligible_dates_with_unavailable_local_context": (
            EXPECTED_ELIGIBLE_DATES_WITH_UNAVAILABLE_LOCAL_CONTEXT
        ),
        "sample_eligibility_rule": SAMPLE_ELIGIBILITY_RULE,
        "local_context_availability_rule": LOCAL_CONTEXT_AVAILABILITY_RULE,
        "equity_session_start_minute": EQUITY_SESSION_START_MINUTE,
        "equity_session_minutes": EQUITY_SESSION_MINUTES,
        "context_session_start_minute": CONTEXT_SESSION_START_MINUTE,
        "context_session_minutes": CONTEXT_SESSION_MINUTES,
        "dynamic_channels": list(DYNAMIC_CHANNELS),
        "equity_slow_channels": list(SLOW_CHANNELS),
        "equity_peer_channels": list(EQUITY_PEER_CHANNELS),
        "equity_peer_valid_channels": list(EQUITY_PEER_VALID_CHANNELS),
        "global_session_start_minute": GLOBAL_SESSION_START_MINUTE,
        "global_session_end_minute": GLOBAL_SESSION_END_MINUTE,
        "global_session_minutes": GLOBAL_SESSION_MINUTES,
        "context_slow_channels": list(SLOW_CHANNELS),
        "decision_equity_indices": list(DECISION_EQUITY_INDICES),
        "decision_context_indices": list(DECISION_CONTEXT_INDICES),
        "horizons": list(HORIZONS),
        "global_slow_channels": list(GLOBAL_SLOW_CHANNELS),
        "vol_warmup_valid_days": VOL_WARMUP_VALID_DAYS,
        "vol_ewma_half_life_days": VOL_EWMA_HALF_LIFE_DAYS,
        "decision_global_indices": list(DECISION_GLOBAL_INDICES),
        "vol_ewma_alpha": VOL_EWMA_ALPHA,
        "min_adjacent_returns_per_day": MIN_ADJACENT_RETURNS_PER_DAY,
        "price_vol_floor": PRICE_VOL_FLOOR,
        "rate_vol_floor_bp": RATE_VOL_FLOOR_BP,
        "price_vol_reference": PRICE_VOL_REFERENCE,
        "rate_vol_reference_bp": RATE_VOL_REFERENCE_BP,
        "volume_lookback_sessions": VOLUME_LOOKBACK_SESSIONS,
        "volume_min_observations": VOLUME_MIN_OBSERVATIONS,
        "mad_normalization": MAD_NORMALIZATION,
        "volume_mad_floor": VOLUME_MAD_FLOOR,
        "price_feature_clip": PRICE_FEATURE_CLIP,
        "volume_feature_clip": VOLUME_FEATURE_CLIP,
        "vol_regime_clip": VOL_REGIME_CLIP,
        "min_active_equities": MIN_ACTIVE_EQUITIES,
        "return_windows": list(RETURN_WINDOWS),
        "realized_vol_min_fraction": REALIZED_VOL_MIN_FRACTION,
        "realized_vol_log_floor": REALIZED_VOL_LOG_FLOOR,
        "realized_vol_log_clip": REALIZED_VOL_LOG_CLIP,
        "observed_fraction_window": OBSERVED_FRACTION_WINDOW,
        "slow_short_window": SLOW_SHORT_WINDOW,
        "slow_long_window": SLOW_LONG_WINDOW,
        "slow_short_min_valid": SLOW_SHORT_MIN_VALID,
        "slow_long_min_valid": SLOW_LONG_MIN_VALID,
        "beta_ewma_half_life_days": BETA_EWMA_HALF_LIFE_DAYS,
        "beta_ewma_alpha": BETA_EWMA_ALPHA,
        "beta_min_paired_sessions": BETA_MIN_PAIRED_SESSIONS,
        "beta_variance_floor": BETA_VARIANCE_FLOOR,
        "beta_clip": BETA_CLIP,
        "real_volume_log_affine": {
            "center": REAL_VOLUME_LOG_CENTER,
            "scale": REAL_VOLUME_LOG_SCALE,
        },
        "dollar_volume_log_affine": {
            "center": DOLLAR_VOLUME_LOG_CENTER,
            "scale": DOLLAR_VOLUME_LOG_SCALE,
        },
    }
