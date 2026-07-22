from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np

CONTRACT_VERSION = "M1_FEATURES_V1"

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
CATALOGUE_PATH = (
    PROJECT_ROOT / "quant-data/b3/raw/xp/catalogue_canonical/symbol_catalogue.parquet"
)
OUTPUT_BASE = PROJECT_ROOT / "quant-data/b3/processed/features"
CANONICAL_OUTPUT_POINTER = OUTPUT_BASE / "m1_features_v1_canonical_path.txt"

EXPECTED_EQUITIES = 158
CONTEXT_SYMBOLS = ("WIN$", "WDO$", "DI1F27", "DI1F28", "DI1F29", "DI1F31")
CONTEXT_FAMILIES = (
    "EQUITY_FUTURE",
    "FX_FUTURE",
    "RATE_FUTURE",
    "RATE_FUTURE",
    "RATE_FUTURE",
    "RATE_FUTURE",
)
RATE_CONTEXT_SYMBOLS = frozenset(CONTEXT_SYMBOLS[2:])

EQUITY_SESSION_START_MINUTE = 10 * 60
EQUITY_SESSION_MINUTES = 405
CONTEXT_SESSION_START_MINUTE = 9 * 60
CONTEXT_SESSION_MINUTES = 465
DYNAMIC_CHANNELS = (
    "open_move_normalized",
    "high_move_normalized",
    "low_move_normalized",
    "close_move_normalized",
    "volume_surprise",
    "observed",
)
EQUITY_SLOW_CHANNELS = ("vol_regime",)
CONTEXT_SLOW_CHANNELS = (
    "vol_regime",
    "prior_rate_level_scaled",
    "time_to_expiry_scaled",
)

DECISION_EQUITY_INDICES = tuple(15 + 5 * index for index in range(55))
DECISION_CONTEXT_INDICES = tuple(75 + 5 * index for index in range(55))
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
VOLUME_LOOKBACK_SESSIONS = 20
VOLUME_MIN_OBSERVATIONS = 10
MAD_NORMALIZATION = 1.4826
VOLUME_MAD_FLOOR = 0.1
PRICE_FEATURE_CLIP = 10.0
VOLUME_FEATURE_CLIP = 6.0
VOL_REGIME_CLIP = 4.0
MIN_ACTIVE_EQUITIES = 30
VOL_EWMA_ALPHA = 1 - 2 ** (-1 / VOL_EWMA_HALF_LIFE_DAYS)


@dataclass(frozen=True)
class OutputArraySpec:
    dtype: np.dtype
    shape: tuple[int, ...]


def output_array_specs(date_count: int) -> dict[str, OutputArraySpec]:
    d = date_count
    n = EXPECTED_EQUITIES
    c = len(CONTEXT_SYMBOLS)
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
        "context_features.npy": OutputArraySpec(
            np.dtype(np.float32), (d, c, CONTEXT_SESSION_MINUTES, f)
        ),
        "context_slow.npy": OutputArraySpec(
            np.dtype(np.float32), (d, c, len(CONTEXT_SLOW_CHANNELS))
        ),
        "context_data_ready.npy": OutputArraySpec(np.dtype(bool), (d, c)),
        "raw_returns.npy": OutputArraySpec(np.dtype(np.float32), (d, n, q, h)),
        "targets.npy": OutputArraySpec(np.dtype(np.float32), (d, n, q, h)),
        "label_mask.npy": OutputArraySpec(np.dtype(bool), (d, n, q, h)),
        "cross_section_median.npy": OutputArraySpec(np.dtype(np.float32), (d, q, h)),
        "horizon_mask.npy": OutputArraySpec(np.dtype(bool), (d, q, h)),
    }


def manifest_constants() -> dict[str, object]:
    return {
        "expected_equities": EXPECTED_EQUITIES,
        "context_symbols": list(CONTEXT_SYMBOLS),
        "context_families": list(CONTEXT_FAMILIES),
        "equity_session_start_minute": EQUITY_SESSION_START_MINUTE,
        "equity_session_minutes": EQUITY_SESSION_MINUTES,
        "context_session_start_minute": CONTEXT_SESSION_START_MINUTE,
        "context_session_minutes": CONTEXT_SESSION_MINUTES,
        "dynamic_channels": list(DYNAMIC_CHANNELS),
        "equity_slow_channels": list(EQUITY_SLOW_CHANNELS),
        "context_slow_channels": list(CONTEXT_SLOW_CHANNELS),
        "decision_equity_indices": list(DECISION_EQUITY_INDICES),
        "decision_context_indices": list(DECISION_CONTEXT_INDICES),
        "horizons": list(HORIZONS),
        "vol_warmup_valid_days": VOL_WARMUP_VALID_DAYS,
        "vol_ewma_half_life_days": VOL_EWMA_HALF_LIFE_DAYS,
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
    }
