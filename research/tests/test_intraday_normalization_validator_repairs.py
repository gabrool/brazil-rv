from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import brazil_rv.preprocessing.intraday_normalization_diagnostics as diagnostics
from brazil_rv.preprocessing.intraday_normalization import ARMS, sha256_file


def _diagnostic_store(path: Path) -> None:
    dates = (
        date(2024, 6, 27),
        date(2024, 6, 28),
        date(2025, 6, 27),
        date(2025, 6, 30),
    )
    security_count = 10
    minute_count = 30
    pl.DataFrame({"date_idx": range(len(dates)), "trade_date": dates}).write_parquet(
        path / "date_index.parquet"
    )
    pl.DataFrame(
        {
            "equity_slot": range(security_count),
            "security_id": [f"security-{slot}" for slot in range(security_count)],
        }
    ).write_parquet(path / "equity_index.parquet")
    np.save(
        path / "equity_membership.npy",
        np.ones((len(dates), security_count), dtype=bool),
        allow_pickle=False,
    )
    np.save(
        path / "equity_data_ready.npy",
        np.ones((len(dates), security_count), dtype=bool),
        allow_pickle=False,
    )
    features = np.zeros(
        (len(dates), security_count, minute_count, 26), dtype=np.float32
    )
    grid = np.arange(features[..., 3].size, dtype=np.float32).reshape(
        features[..., 3].shape
    )
    features[..., 3] = np.sin(grid / 17.0) + 0.01 * grid
    features[..., 5] = 1.0
    np.save(path / "equity_features.npy", features, allow_pickle=False)


def _diagnostic_profile(path: Path) -> None:
    dates = (
        date(2024, 6, 27),
        date(2024, 6, 28),
        date(2025, 6, 27),
        date(2025, 6, 30),
    )
    (path / "equity_tod_profile.json").write_text("{}", encoding="utf-8")
    pl.DataFrame(
        {
            "date_idx": range(len(dates)),
            "trade_date": dates,
            "bin_idx": [0] * len(dates),
            "relative_variance": [1.0] * len(dates),
            "effective_historical_profile_days": [20] * len(dates),
            "shrinkage_weight": [0.5] * len(dates),
        }
    ).write_csv(path / "equity_tod_profile.csv")


def test_diagnostic_validator_rejects_self_consistently_rehashed_scientific_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "store"
    profile = tmp_path / "profile"
    output = tmp_path / "diagnostics"
    store.mkdir()
    profile.mkdir()
    _diagnostic_store(store)
    _diagnostic_profile(profile)
    monkeypatch.setattr(diagnostics, "CHANNELS", {3: "close_move_normalized"})
    monkeypatch.setattr(diagnostics, "RETURN_CHANNEL_WINDOWS", {3: 15})
    monkeypatch.setattr(diagnostics, "REALIZED_VOL_CHANNEL_WINDOWS", {})
    monkeypatch.setattr(diagnostics, "VISIBLE_EQUITY_MINUTES", 30)
    monkeypatch.setattr(diagnostics, "DECISION_FEATURE_MINUTES", tuple(range(30)))
    monkeypatch.setattr(diagnostics, "OPENING_BIN", 0)
    monkeypatch.setattr(diagnostics, "MIDDAY_BIN", 0)
    monkeypatch.setattr(
        diagnostics,
        "load_equity_tod_profile",
        lambda _path: ({}, np.ones((4, 1), dtype=np.float64)),
    )
    stores = {arm: store for arm in ARMS}

    diagnostics.run_heteroskedasticity_diagnostics(stores, profile, output)

    effect_path = output / "heteroskedasticity_effect_sizes.csv"
    effects = pl.read_csv(effect_path)
    baseline_effect = effect_path.read_bytes()
    effects = effects.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("weighted_log_std_rmse") + 0.25)
        .otherwise(pl.col("weighted_log_std_rmse"))
        .alias("weighted_log_std_rmse")
    )
    effects.write_csv(effect_path)
    manifest_path = output / "diagnostics_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_manifest = manifest_path.read_bytes()
    manifest["output_sha256"][effect_path.name] = sha256_file(effect_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="effect sizes.*reconstruct"):
        diagnostics.validate_heteroskedasticity_diagnostics(output)

    effect_path.write_bytes(baseline_effect)
    manifest_path.write_bytes(baseline_manifest)
    summary_path = output / "heteroskedasticity_summary.json"
    baseline_summary = summary_path.read_bytes()
    summary = json.loads(baseline_summary.decode("utf-8"))
    summary["row_counts"]["heteroskedasticity_by_bin.csv"] += 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest = json.loads(baseline_manifest.decode("utf-8"))
    manifest["output_sha256"][summary_path.name] = sha256_file(summary_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="summary.*reconstruct"):
        diagnostics.validate_heteroskedasticity_diagnostics(output)

    summary_path.write_bytes(baseline_summary)
    manifest_path.write_bytes(baseline_manifest)
    markdown_path = output / "heteroskedasticity_summary.md"
    markdown_path.write_bytes(b"scientifically incorrect\n")
    manifest = json.loads(baseline_manifest.decode("utf-8"))
    manifest["output_sha256"][markdown_path.name] = sha256_file(markdown_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Markdown does not reconstruct"):
        diagnostics.validate_heteroskedasticity_diagnostics(output)
