from __future__ import annotations

import inspect
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling.contract import (
    BASELINE_TCN_SETTINGS,
    EQUITY_COUNT,
    GLOBAL_CONTEXT_COUNT,
    LOCAL_CONTEXT_COUNT,
    SLOW_FEATURE_COUNT,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_START,
    architecture_for_model,
)
from brazil_rv.modeling.data import _build_patch_batch
from brazil_rv.modeling.feature_variant import OverlayArray, load_variant_manifest
from brazil_rv.modeling.model import build_neural_model, count_trainable_parameters
from brazil_rv.modeling.run_intraday_normalization_stage import Stage
from brazil_rv.preprocessing.contract import EQUITY_SESSION_MINUTES
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    PROFILE_BIN_COUNT,
    build_seasonal_dynamic_features,
    estimate_causal_profile,
    load_equity_tod_profile,
    sha256_file,
)
from brazil_rv.preprocessing.intraday_normalization_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    validate_heteroskedasticity_diagnostics,
)
from brazil_rv.preprocessing.transforms import build_dynamic_features


def _raw_path(increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    date_count, minute_count = increments.shape
    close = np.exp(np.cumsum(increments, axis=1))
    previous = np.concatenate((np.ones((date_count, 1)), close[:, :-1]), axis=1)
    raw = np.zeros((date_count, minute_count, 5), dtype=np.float64)
    raw[..., 0] = previous
    raw[..., 1] = np.maximum(previous, close) * np.exp(0.0002)
    raw[..., 2] = np.minimum(previous, close) * np.exp(-0.0002)
    raw[..., 3] = close
    raw[..., 4] = 100.0
    return raw, np.ones((date_count, minute_count), dtype=bool)


def test_gamma_zero_is_bitwise_legacy() -> None:
    generator = np.random.default_rng(4)
    raw, observed = _raw_path(
        generator.normal(0.0, 0.001, size=(3, EQUITY_SESSION_MINUTES))
    )
    ready = np.ones(3, dtype=bool)
    sigma = np.full(3, 0.002)
    q = np.linspace(0.5, 1.5, PROFILE_BIN_COUNT)[None].repeat(3, axis=0)
    legacy = build_dynamic_features(
        raw,
        observed,
        ready,
        sigma,
        is_rate=False,
        first_observed_open=True,
    )
    candidate = build_seasonal_dynamic_features(raw, observed, ready, sigma, q, 0.0)
    assert np.array_equal(candidate[0], legacy[0])
    assert np.array_equal(candidate[1], legacy[1])


def test_integrated_window_and_corrected_realized_volatility_math() -> None:
    increments = np.full((1, EQUITY_SESSION_MINUTES), 0.1)
    raw, observed = _raw_path(increments)
    ready = np.ones(1, dtype=bool)
    sigma = np.ones(1)
    q = np.ones((1, PROFILE_BIN_COUNT), dtype=np.float64)
    q[:, 0] = 4.0
    features, _ = build_seasonal_dynamic_features(raw, observed, ready, sigma, q, 1.0)

    integrated_15m = 14.0 * 4.0 + 1.0
    assert features[0, 30, 7] == pytest.approx(1.5 / np.sqrt(integrated_15m), rel=1e-6)
    corrected_rms = np.sqrt((29.0 * (0.1 / 2.0) ** 2 + 0.1**2) / 30.0)
    assert features[0, 30, 11] == pytest.approx(np.log(corrected_rms), rel=1e-6)


def test_ohlc_channels_share_one_seasonal_multiplier() -> None:
    raw, observed = _raw_path(np.full((1, EQUITY_SESSION_MINUTES), 0.001))
    raw[0, 10, 0] *= np.exp(0.0001)
    ready = np.ones(1, dtype=bool)
    sigma = np.full(1, 0.01)
    q = np.ones((1, PROFILE_BIN_COUNT))
    q[:, 0] = 4.0
    legacy, _ = build_seasonal_dynamic_features(raw, observed, ready, sigma, q, 0.0)
    full, _ = build_seasonal_dynamic_features(raw, observed, ready, sigma, q, 1.0)
    ratios = full[0, 10, :4] / legacy[0, 10, :4]
    assert np.allclose(ratios, 0.5, rtol=1e-6, atol=1e-7)


def test_half_and_full_remove_a_synthetic_u_shape_without_inversion() -> None:
    generator = np.random.default_rng(20260817)
    q = np.ones(PROFILE_BIN_COUNT)
    q[0] = 4.0
    q[4] = 0.5
    minute_q = q[np.arange(EQUITY_SESSION_MINUTES) // 30]
    increments = generator.normal(
        0.0,
        0.001 * np.sqrt(minute_q),
        size=(300, EQUITY_SESSION_MINUTES),
    )
    raw, observed = _raw_path(increments)
    ready = np.ones(300, dtype=bool)
    sigma = np.full(300, 0.001)
    q_by_day = np.broadcast_to(q, (300, PROFILE_BIN_COUNT)).copy()

    ratios: list[float] = []
    opening_to_midday: list[float] = []
    for gamma in (0.0, 0.5, 1.0):
        features, _ = build_seasonal_dynamic_features(
            raw, observed, ready, sigma, q_by_day, gamma
        )
        close = features[..., 3]
        standard_deviations = np.asarray(
            [
                np.std(close[:, start : start + 30], ddof=1)
                for start in range(0, EQUITY_SESSION_MINUTES, 30)
            ]
        )
        ratios.append(float(standard_deviations.max() / standard_deviations.min()))
        opening_to_midday.append(float(standard_deviations[0] / standard_deviations[4]))
    assert ratios[2] < ratios[1] < ratios[0]
    assert ratios[2] < 1.12
    assert 0.9 < opening_to_midday[2] < 1.1


def _profile_inputs():
    dates = tuple(TRAIN_START - timedelta(days=30 - index) for index in range(60))
    variance = np.ones((len(dates), PROFILE_BIN_COUNT), dtype=np.float64)
    variance[:, 0] = 4.0
    variance[:, 4] = 0.5
    count = np.full(variance.shape, 100, dtype=np.int64)
    return dates, variance, count


def test_profile_is_u_shaped_shrunk_and_uses_session_bin_day_counts() -> None:
    dates, variance, count = _profile_inputs()
    profile = estimate_causal_profile(variance, count, dates)
    final = profile.relative_variance[-1]
    assert final[0] > final[4]
    assert profile.historical_profile_days[-1, 0] == len(dates) - 1
    assert profile.shrinkage_weight[-1, 0] == pytest.approx(
        (len(dates) - 1) / (len(dates) - 1 + 20)
    )
    assert np.average(final, weights=profile.historical_observation_count[-1]) == (
        pytest.approx(1.0, abs=1e-12)
    )


def test_profile_is_emit_then_update_and_future_mutation_is_causal() -> None:
    dates, variance, count = _profile_inputs()
    baseline = estimate_causal_profile(variance, count, dates)
    changed = variance.copy()
    changed[25, 0] *= 100.0
    mutated = estimate_causal_profile(changed, count, dates)
    assert np.array_equal(
        baseline.relative_variance[:26], mutated.relative_variance[:26]
    )
    assert not np.array_equal(
        baseline.relative_variance[26:], mutated.relative_variance[26:]
    )


def test_validation_profile_is_frozen_and_ignores_validation_observations() -> None:
    dates = tuple(TRAIN_END - timedelta(days=9 - index) for index in range(10)) + (
        VALIDATION_START,
        VALIDATION_START + timedelta(days=1),
    )
    variance = np.ones((len(dates), PROFILE_BIN_COUNT), dtype=np.float64)
    count = np.full(variance.shape, 50, dtype=np.int64)
    baseline = estimate_causal_profile(variance, count, dates)
    variance[-2:] *= np.arange(1, PROFILE_BIN_COUNT + 1)
    mutated = estimate_causal_profile(variance, count, dates)
    assert np.array_equal(baseline.relative_variance, mutated.relative_variance)
    assert np.array_equal(mutated.relative_variance[-2], mutated.relative_variance[-1])


def test_missing_bin_does_not_update_as_zero() -> None:
    dates, variance, count = _profile_inputs()
    count[:, 7] = 0
    variance[:, 7] = 0.0
    profile = estimate_causal_profile(variance, count, dates)
    assert np.all(profile.historical_profile_days[:, 7] == 0)
    assert np.all(profile.shrinkage_weight[:, 7] == 0.0)


def test_profile_estimator_has_no_target_or_label_input() -> None:
    parameters = set(inspect.signature(estimate_causal_profile).parameters)
    assert parameters == {
        "daily_variance",
        "daily_observation_count",
        "trade_dates",
        "config",
    }


def test_overlay_replaces_only_declared_channels_and_rejects_held_out_dates() -> None:
    parent = np.arange(3 * 2 * 6 * 5, dtype=np.float32).reshape(3, 2, 6, 5)
    overlay = np.full((2, 2, 6, 2), -7.0, dtype=np.float32)
    values = OverlayArray(parent, overlay, (0, 2))
    selected = values[np.asarray([0, 1]), :, :4, :]
    assert selected.shape == (2, 2, 4, 5)
    assert np.all(selected[..., 0] == -7.0)
    assert np.all(selected[..., 2] == -7.0)
    assert np.array_equal(selected[..., 1], parent[:2, :, :4, 1])
    with pytest.raises(ValueError, match="held-out"):
        values[np.asarray([2]), :, :4, 0]


def test_peer_overlay_maps_only_decision_minutes() -> None:
    parent = np.zeros((2, 2, 10, 6), dtype=np.float32)
    overlay = np.ones((2, 2, 2, 2), dtype=np.float32)
    values = OverlayArray(parent, overlay, (0, 4), minute_positions=(3, 8))
    assert np.all(values[np.asarray([0]), :, 3, :5][..., (0, 4)] == 1.0)
    with pytest.raises(ValueError, match="decision minutes"):
        values[np.asarray([0]), :, 4, 0]


def test_variant_manifest_rejects_wrong_gamma(tmp_path: Path) -> None:
    manifest = {
        "schema": "EQUITY_INTRADAY_NORMALIZATION_OVERLAY_V1",
        "test_accessed": False,
        "test_rows_present": False,
        "arm": "equity_tod_half",
        "gamma": 1.0,
        "allowed_date_end": "2025-06-30",
    }
    (tmp_path / "intraday_normalization_variant.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="arm/gamma"):
        load_variant_manifest(tmp_path)


def test_profile_loader_rejects_corrupt_profile_hash(tmp_path: Path) -> None:
    q_path = tmp_path / "equity_tod_profile.npy"
    csv_path = tmp_path / "equity_tod_profile.csv"
    np.save(q_path, np.ones((1, PROFILE_BIN_COUNT)), allow_pickle=False)
    csv_path.write_text("date_idx,bin_idx\n0,0\n", encoding="utf-8")
    manifest = {
        "schema": "EQUITY_INTRADAY_VARIANCE_PROFILE_V1",
        "test_accessed": False,
        "date_count": 1,
        "bin_count": PROFILE_BIN_COUNT,
        "artifacts": {
            q_path.name: sha256_file(q_path),
            csv_path.name: sha256_file(csv_path),
        },
    }
    (tmp_path / "equity_tod_profile.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    q_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_equity_tod_profile(tmp_path)


def test_candidate_batch_preserves_shapes_masks_context_and_parameter_count() -> None:
    equity = np.zeros((1, EQUITY_COUNT, 405, 26), dtype=np.float32)
    equity[..., 3] = 1.0
    equity[..., 5] = 1.0
    context = np.zeros((1, LOCAL_CONTEXT_COUNT, 465, 26), dtype=np.float32)
    context[..., 5] = 1.0
    global_features = np.zeros((1, GLOBAL_CONTEXT_COUNT, 615, 26), dtype=np.float32)
    global_features[..., 5] = 1.0
    arrays = {
        "equity_features.npy": equity,
        "equity_slow.npy": np.ones(
            (1, EQUITY_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "context_features.npy": context,
        "context_slow.npy": np.ones(
            (1, LOCAL_CONTEXT_COUNT, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "context_data_ready.npy": np.ones((1, LOCAL_CONTEXT_COUNT), dtype=bool),
        "global_features.npy": global_features,
        "global_slow.npy": np.ones(
            (1, GLOBAL_CONTEXT_COUNT, 55, SLOW_FEATURE_COUNT), dtype=np.float32
        ),
        "global_data_ready.npy": np.ones((1, GLOBAL_CONTEXT_COUNT, 55), dtype=bool),
    }
    overlay = np.zeros(
        (1, EQUITY_COUNT, 285, len(AFFECTED_DYNAMIC_CHANNELS)), dtype=np.float32
    )
    overlay[..., AFFECTED_DYNAMIC_CHANNELS.index(3)] = 2.0
    candidate_arrays = {
        **arrays,
        "equity_features.npy": OverlayArray(equity, overlay, AFFECTED_DYNAMIC_CHANNELS),
    }
    arguments = (
        np.asarray([0]),
        np.asarray([15]),
        np.asarray([0]),
        np.asarray([75]),
        np.ones((1, EQUITY_COUNT), dtype=bool),
        "enabled",
    )
    legacy = _build_patch_batch(arrays, *arguments)
    candidate = _build_patch_batch(candidate_arrays, *arguments)
    assert candidate["patches"].shape == legacy["patches"].shape
    assert candidate["patches"].dtype == legacy["patches"].dtype == np.float32
    assert np.array_equal(candidate["history_patch_mask"], legacy["history_patch_mask"])
    assert np.array_equal(candidate["instrument_mask"], legacy["instrument_mask"])
    assert np.array_equal(candidate["slow_features"], legacy["slow_features"])
    assert not np.array_equal(candidate["patches"], legacy["patches"])

    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    counts = {
        count_trainable_parameters(build_neural_model("tcn", architecture, "selected"))
        for _ in (0.0, 0.5, 1.0)
    }
    assert len(counts) == 1


def test_corrupt_diagnostic_output_is_rejected(tmp_path: Path) -> None:
    profile = tmp_path / "profile"
    output = tmp_path / "diagnostics"
    profile.mkdir()
    output.mkdir()
    profile_manifest = profile / "equity_tod_profile.json"
    profile_manifest.write_text("{}", encoding="utf-8")
    counts = {
        "heteroskedasticity_by_bin.csv": 96,
        "heteroskedasticity_by_year.csv": 3,
        "heteroskedasticity_by_security.csv": 3,
        "return_windows_by_decision_bin.csv": 3,
        "heteroskedasticity_effect_sizes.csv": 96,
        "cross_security_dispersion.csv": 96,
    }
    arms = ("legacy_daily_vol", "equity_tod_half", "equity_tod_full")
    for name, count in counts.items():
        pl.DataFrame(
            {"arm": [arms[index % 3] for index in range(count)], "value": range(count)}
        ).write_csv(output / name)
    summary = {
        "schema": DIAGNOSTIC_SCHEMA,
        "test_accessed": False,
        "row_counts": counts,
        "primary_results": [{"arm": arm} for arm in arms],
    }
    (output / "heteroskedasticity_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (output / "heteroskedasticity_summary.md").write_text("valid\n", encoding="utf-8")
    output_names = (
        *counts,
        "heteroskedasticity_summary.json",
        "heteroskedasticity_summary.md",
    )
    lineage = {
        "schema": DIAGNOSTIC_SCHEMA,
        "test_accessed": False,
        "profile": {
            "path": str(profile),
            "manifest_sha256": sha256_file(profile_manifest),
        },
        "stores": {arm: str(tmp_path / arm) for arm in arms},
        "output_sha256": {name: sha256_file(output / name) for name in output_names},
    }
    (output / "diagnostics_manifest.json").write_text(
        json.dumps(lineage), encoding="utf-8"
    )
    (output / "heteroskedasticity_by_bin.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="output hash mismatch"):
        validate_heteroskedasticity_diagnostics(output)


def _identity_store(path: Path) -> None:
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    (path / "feature_schema.json").write_text(
        json.dumps({"contract_version": "fixture"}), encoding="utf-8"
    )
    (path / "sample_index.parquet").write_bytes(b"fixture")


def test_stage_completed_step_resumes_and_failed_step_archives(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    _identity_store(parent)
    stage = Stage(tmp_path / "stage", parent)
    artifact = stage.output_dir / "artifact"
    calls = 0

    def action() -> None:
        nonlocal calls
        calls += 1
        artifact.mkdir()
        (artifact / "value").write_text("valid", encoding="utf-8")

    def validate() -> None:
        assert (artifact / "value").read_text(encoding="utf-8") == "valid"

    stage.step("complete", {"value": 1}, (artifact,), action, validate)
    stage.step("complete", {"value": 1}, (artifact,), action, validate)
    assert calls == 1

    failed = stage.output_dir / "failed"
    partial = failed.with_name(f"{failed.name}.partial")

    def fail() -> None:
        partial.mkdir()
        raise RuntimeError("expected")

    with pytest.raises(RuntimeError, match="expected"):
        stage.step("retry", {}, (failed,), fail, lambda: None)

    def retry() -> None:
        failed.mkdir()
        (failed / "done").write_text("yes", encoding="utf-8")

    stage.step("retry", {}, (failed,), retry, lambda: None)
    assert len(stage.manifest["archive_history"]) == 1
    assert stage.manifest["archive_history"][0]["source"] == str(partial)
    assert Path(stage.manifest["archive_history"][0]["archive"]).is_dir()
