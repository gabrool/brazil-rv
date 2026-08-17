from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.contract import (
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from brazil_rv.preprocessing.contract import EQUITY_SESSION_MINUTES
from brazil_rv.preprocessing.intraday_normalization import (
    PROFILE_BIN_COUNT,
    PROFILE_BIN_MINUTES,
    PROFILE_SCHEMA,
    ProfileConfig,
    estimate_causal_profile,
    repository_commit,
    sha256_file,
    validate_equity_tod_profile,
)


def _valid_profile(path: Path) -> None:
    dates = (TRAIN_START, TRAIN_END, VALIDATION_START, VALIDATION_END)
    daily_variance = np.ones((len(dates), PROFILE_BIN_COUNT), dtype=np.float64)
    daily_variance[:2] *= np.linspace(0.8, 1.2, PROFILE_BIN_COUNT)
    daily_count = np.full(daily_variance.shape, 50, dtype=np.int64)
    profile = estimate_causal_profile(daily_variance, daily_count, dates)
    q_path = path / "equity_tod_profile.npy"
    np.save(q_path, profile.relative_variance, allow_pickle=False)
    rows: list[dict[str, object]] = []
    for date_idx, trade_date in enumerate(dates):
        split = "train" if trade_date <= TRAIN_END else "validation"
        for bin_idx in range(PROFILE_BIN_COUNT):
            start = bin_idx * PROFILE_BIN_MINUTES
            q = profile.relative_variance[date_idx, bin_idx]
            rows.append(
                {
                    "date_idx": date_idx,
                    "trade_date": trade_date,
                    "split": split,
                    "bin_idx": bin_idx,
                    "session_minute_start": start,
                    "session_minute_end_exclusive": min(
                        start + PROFILE_BIN_MINUTES, EQUITY_SESSION_MINUTES
                    ),
                    "relative_variance": q,
                    "standard_deviation_multiplier": math.sqrt(q),
                    "effective_historical_profile_days": int(
                        profile.historical_profile_days[date_idx, bin_idx]
                    ),
                    "shrinkage_weight": profile.shrinkage_weight[date_idx, bin_idx],
                    "historical_observation_count": int(
                        profile.historical_observation_count[date_idx, bin_idx]
                    ),
                    "current_daily_variance_estimate": profile.daily_variance[
                        date_idx, bin_idx
                    ],
                    "current_daily_observation_count": int(
                        profile.daily_observation_count[date_idx, bin_idx]
                    ),
                }
            )
    csv_path = path / "equity_tod_profile.csv"
    pl.DataFrame(rows).write_csv(csv_path)
    manifest = {
        "schema": PROFILE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit(),
        "configuration": asdict(ProfileConfig()),
        "training_window": [str(TRAIN_START), str(TRAIN_END)],
        "validation_window": [str(VALIDATION_START), str(VALIDATION_END)],
        "training_profile_freeze_date": str(TRAIN_END),
        "profile_input": "unclipped_legacy_normalized_equity_close_moves",
        "historical_count_unit": "valid_session_bin_estimates",
        "current_date_update_rule": "emit_then_update",
        "validation_update_rule": "frozen_training_end_profile",
        "test_accessed": False,
        "date_count": len(dates),
        "bin_count": PROFILE_BIN_COUNT,
        "hash_scope": {
            "kind": "development_only",
            "end_date": str(VALIDATION_END),
            "date_count": len(dates),
        },
        "artifacts": {
            q_path.name: sha256_file(q_path),
            csv_path.name: sha256_file(csv_path),
        },
    }
    (path / "equity_tod_profile.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_profile_validator_reconstructs_semantics_after_hash_validation(
    tmp_path: Path,
) -> None:
    _valid_profile(tmp_path)
    validate_equity_tod_profile(tmp_path)
    csv_path = tmp_path / "equity_tod_profile.csv"
    frame = pl.read_csv(csv_path, try_parse_dates=True).with_columns(
        pl.when((pl.col("date_idx") == 0) & (pl.col("bin_idx") == 0))
        .then(pl.col("standard_deviation_multiplier") + 0.1)
        .otherwise(pl.col("standard_deviation_multiplier"))
        .alias("standard_deviation_multiplier")
    )
    frame.write_csv(csv_path)
    manifest_path = tmp_path / "equity_tod_profile.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][csv_path.name] = sha256_file(csv_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="multiplier does not reconstruct"):
        validate_equity_tod_profile(tmp_path)
