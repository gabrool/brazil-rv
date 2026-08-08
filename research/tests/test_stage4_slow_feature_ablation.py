from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling import (
    analyze_stage4_slow_feature_ablation,
    stage4_slow_feature_ablation,
    train,
)
from brazil_rv.modeling.analyze_stage4_slow_feature_ablation import (
    _period_delta,
    _three_seed_summary,
)
from brazil_rv.modeling.audit_slow_features import (
    AUDIT_VERSION,
    compute_slow_statistics,
)
from brazil_rv.modeling.context_ablation import (
    get_context_ablation,
    resolve_context_ablation,
)
from brazil_rv.modeling.contract import (
    DYNAMIC_CHANNEL_COUNT,
    EQUITY_COUNT,
    GLOBAL_CONTEXT_COUNT,
    GLOBAL_CONTEXT_SYMBOLS,
    LOCAL_CONTEXT_COUNT,
    LOCAL_CONTEXT_SYMBOLS,
    SLOW_FEATURE_COUNT,
)
from brazil_rv.modeling.data import BatchRequest, VectorizedFeatureDataset
from brazil_rv.modeling.evaluate import _normalize_feature_ablation_identity
from brazil_rv.modeling.feature_ablation import (
    FEATURE_ABLATION_KEYS,
    apply_feature_ablation_to_slow_features,
    get_feature_ablation,
    resolve_feature_ablation,
    resolve_feature_ablation_for_store,
)
from brazil_rv.modeling.stage3_context_addition import stage3_jobs
from brazil_rv.preprocessing.contract import (
    GLOBAL_SLOW_CHANNELS,
    GLOBAL_UNUSED_SLOW_CHANNEL_INDICES,
    SLOW_CHANNELS,
)

REMOVED_NAMES = (
    "vol_regime",
    "realized_vol_20d_log_ratio",
    "vol_of_vol_20d",
    "median_daily_real_volume_20d_log_scale",
    "daily_dollar_volume_regime_20d",
    "observed_fraction_5d",
    "observed_fraction_20d",
    "weekday_sin",
    "weekday_cos",
    "month_end_proximity",
    "quarter_end_proximity",
)
REMOVED_INDICES = (0, 10, 11, 12, 14, 15, 16, 26, 27, 28, 29)


REMOVED_GLOBAL_NAMES = tuple(GLOBAL_SLOW_CHANNELS[index] for index in REMOVED_INDICES)


def _resolved_feature(key: str):
    return resolve_feature_ablation(
        get_feature_ablation(key), slow_features=SLOW_CHANNELS
    )


def _resolved_context():
    return resolve_context_ablation(
        get_context_ablation("drop_win_and_global_non_rates"),
        local_symbols=LOCAL_CONTEXT_SYMBOLS,
        global_symbols=GLOBAL_CONTEXT_SYMBOLS,
        equity_slow_features=SLOW_CHANNELS,
    )


def _schema(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "contract_version": "M1_FEATURES_INTRADAY_DI_MASKED_CONTEXT",
                "slow_channels": [
                    {"index": index, "name": name}
                    for index, name in enumerate(SLOW_CHANNELS)
                ],
                "global_slow_channels": [
                    {"index": index, "name": name}
                    for index, name in enumerate(GLOBAL_SLOW_CHANNELS)
                ],
                "global_slow": list(GLOBAL_UNUSED_SLOW_CHANNEL_INDICES),
            }
        ),
        encoding="utf-8",
    )


def _synthetic_store(path: Path) -> tuple[Path, pl.DataFrame, dict[str, np.ndarray]]:
    generator = np.random.default_rng(404)
    arrays = {
        "equity_features.npy": generator.normal(
            size=(1, EQUITY_COUNT, 405, DYNAMIC_CHANNEL_COUNT)
        ).astype(np.float32),
        "equity_slow.npy": generator.normal(
            size=(1, EQUITY_COUNT, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "equity_membership.npy": np.zeros((1, EQUITY_COUNT), dtype=bool),
        "equity_data_ready.npy": np.zeros((1, EQUITY_COUNT), dtype=bool),
        "context_features.npy": generator.normal(
            size=(1, LOCAL_CONTEXT_COUNT, 465, DYNAMIC_CHANNEL_COUNT)
        ).astype(np.float32),
        "context_slow.npy": generator.normal(
            size=(1, LOCAL_CONTEXT_COUNT, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "context_data_ready.npy": np.ones((1, LOCAL_CONTEXT_COUNT), dtype=bool),
        "global_features.npy": generator.normal(
            size=(1, GLOBAL_CONTEXT_COUNT, 615, DYNAMIC_CHANNEL_COUNT)
        ).astype(np.float32),
        "global_slow.npy": generator.normal(
            size=(1, GLOBAL_CONTEXT_COUNT, 55, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "global_data_ready.npy": np.ones((1, GLOBAL_CONTEXT_COUNT, 55), dtype=bool),
        "targets.npy": generator.normal(size=(1, EQUITY_COUNT, 55, 3)).astype(
            np.float32
        ),
        "label_mask.npy": np.zeros((1, EQUITY_COUNT, 55, 3), dtype=bool),
        "raw_returns.npy": generator.normal(size=(1, EQUITY_COUNT, 55, 3)).astype(
            np.float32
        ),
    }
    arrays["equity_membership.npy"][:, :3] = True
    arrays["equity_data_ready.npy"][:, :3] = True
    arrays["label_mask.npy"][:, :3] = True
    arrays["equity_features.npy"][..., 5] = 1.0
    arrays["context_features.npy"][..., 5] = 1.0
    arrays["global_features.npy"][..., 5] = 1.0
    for filename, array in arrays.items():
        np.save(path / filename, array)
    rows = pl.DataFrame(
        {
            "sample_id": [0],
            "date_idx": [0],
            "decision_idx": [54],
            "equity_cutoff_index": [285],
            "context_cutoff_index": [345],
        }
    )
    return path, rows, arrays


def _batch(store: Path, rows: pl.DataFrame, feature_key: str):
    return VectorizedFeatureDataset(
        store,
        rows,
        "context_pooled",
        "enabled",
        context_ablation=_resolved_context(),
        feature_ablation=_resolved_feature(feature_key),
    )[BatchRequest((0,), 1)]


def test_registry_and_schema_resolution_are_exact(tmp_path: Path) -> None:
    assert FEATURE_ABLATION_KEYS == ("none", "drop_slow_low_prior")
    resolved = _resolved_feature("drop_slow_low_prior")
    assert resolved.specification.removed_slow_features == REMOVED_NAMES
    assert resolved.slow_indices == REMOVED_INDICES
    metadata = resolved.metadata()
    assert metadata["shared_position_count"] == 11
    assert metadata["resolved_position_mapping"] == [
        {
            "index": index,
            "equity_local_name": equity_name,
            "global_name": global_name,
            "global_structurally_unused": index in {14, 15},
        }
        for index, equity_name, global_name in zip(
            REMOVED_INDICES, REMOVED_NAMES, REMOVED_GLOBAL_NAMES, strict=True
        )
    ]
    assert len(metadata["resolved_identity_sha256"]) == 64
    _schema(tmp_path / "feature_schema.json")
    assert (
        resolve_feature_ablation_for_store(tmp_path, "drop_slow_low_prior") == resolved
    )
    schema = json.loads((tmp_path / "feature_schema.json").read_text())
    schema["slow_channels"][0], schema["slow_channels"][1] = (
        schema["slow_channels"][1],
        schema["slow_channels"][0],
    )
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="indices are not contiguous"):
        resolve_feature_ablation_for_store(tmp_path, "drop_slow_low_prior")


def test_registry_rejects_changed_global_axis_with_canonical_local_axis(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "feature_schema.json"
    _schema(schema_path)
    schema = json.loads(schema_path.read_text())
    (
        schema["global_slow_channels"][0]["name"],
        schema["global_slow_channels"][1]["name"],
    ) = (
        schema["global_slow_channels"][1]["name"],
        schema["global_slow_channels"][0]["name"],
    )
    schema_path.write_text(json.dumps(schema))
    with pytest.raises(ValueError, match="global slow-feature axis is not canonical"):
        resolve_feature_ablation_for_store(tmp_path, "drop_slow_low_prior")


def test_none_is_noop_and_treatment_zeros_only_registered_positions() -> None:
    values = np.arange(2 * 3 * 32, dtype=np.float32).reshape(2, 3, 32) + 1.0
    baseline = apply_feature_ablation_to_slow_features(
        values, _resolved_feature("none")
    )
    assert baseline is values
    treatment = apply_feature_ablation_to_slow_features(
        values, _resolved_feature("drop_slow_low_prior")
    )
    assert treatment.dtype == values.dtype
    assert treatment.shape == values.shape
    assert not treatment[..., REMOVED_INDICES].any()
    retained = [index for index in range(32) if index not in REMOVED_INDICES]
    np.testing.assert_array_equal(treatment[..., retained], values[..., retained])
    np.testing.assert_array_equal(values, baseline)


def test_shared_batch_mask_changes_only_slow_positions_and_composes_with_context(
    tmp_path: Path,
) -> None:
    store, rows, original = _synthetic_store(tmp_path)
    control = _batch(store, rows, "none")
    treatment = _batch(store, rows, "drop_slow_low_prior")
    assert not treatment["slow_features"][..., REMOVED_INDICES].any()
    retained = [index for index in range(32) if index not in REMOVED_INDICES]
    np.testing.assert_array_equal(
        treatment["slow_features"][..., retained],
        control["slow_features"][..., retained],
    )
    assert not treatment["slow_features"][:, :EQUITY_COUNT, 20].any()
    for field in (
        "patches",
        "history_patch_mask",
        "instrument_mask",
        "targets",
        "raw_returns",
        "label_mask",
        "sample_valid_mask",
        "sample_id",
        "date_idx",
        "decision_idx",
    ):
        np.testing.assert_array_equal(treatment[field], control[field])
    for filename, expected in original.items():
        np.testing.assert_array_equal(np.load(store / filename), expected)


def _training_args(*extra: str) -> list[str]:
    return [
        "--model",
        "tcn",
        "--tcn-fusion",
        "context_pooled",
        "--tcn-width",
        "64",
        "--tcn-receptive-field",
        "full",
        "--tcn-block",
        "swiglu",
        "--optimizer",
        "sam_adamw",
        "--objective",
        "soft_spearman",
        "--soft-rank-temperature",
        "0.50",
        "--sam-rho",
        "0.125",
        "--global-context",
        "enabled",
        "--context-ablation",
        "drop_win_and_global_non_rates",
        "--seed",
        "11",
        *extra,
    ]


def test_cli_default_and_output_identity_preserve_control_and_separate_treatment() -> (
    None
):
    assert train.parse_args(_training_args()).feature_ablation == "none"
    assert (
        train.parse_args(
            _training_args("--feature-ablation", "drop_slow_low_prior")
        ).feature_ablation
        == "drop_slow_low_prior"
    )
    created = datetime(2026, 8, 7, tzinfo=timezone.utc)
    common = (
        "tcn",
        train.TCNSettings("context_pooled", 64, "full", "swiglu"),
        "sam_adamw",
        "soft_spearman",
        0.5,
        0.125,
        "enabled",
        11,
        created,
        "drop_win_and_global_non_rates",
    )
    control = train._run_directory_name(*common, "none")
    treatment = train._run_directory_name(*common, "drop_slow_low_prior")
    assert control != treatment
    assert "_feature-" not in control
    assert "_feature-drop_slow_low_prior_" in treatment


def test_feature_identity_legacy_rule_and_checkpoint_mismatch(tmp_path: Path) -> None:
    none = _resolved_feature("none").metadata()
    treatment = _resolved_feature("drop_slow_low_prior").metadata()
    legacy = _normalize_feature_ablation_identity({}, {})
    assert legacy.metadata == none
    assert legacy.source == "legacy_implicit_none"
    with pytest.raises(ValueError, match="presence"):
        _normalize_feature_ablation_identity({"feature_ablation": treatment}, {})
    with pytest.raises(ValueError, match="identity mismatch"):
        _normalize_feature_ablation_identity(
            {"feature_ablation": treatment}, {"feature_ablation": none}
        )
    with pytest.raises(ValueError, match="named run"):
        _normalize_feature_ablation_identity(
            {}, {}, run_dir=tmp_path / "run_feature-drop_slow_low_prior_seed11"
        )


def test_training_audit_excludes_structural_zeros_and_is_deterministic() -> None:
    values = np.full((6, 32), np.nan, dtype=np.float64)
    values[:3, 0] = [1.0, 2.0, 3.0]
    values[:3, 1] = [2.0, 4.0, 6.0]
    values[3:, 1] = 0.0
    first = compute_slow_statistics(values, removed_indices=(0,))
    second = compute_slow_statistics(values, removed_indices=(0,))
    assert first == second
    assert first["pearson"][0][1] == pytest.approx(1.0)
    assert first["spearman"][0][1] == pytest.approx(1.0)
    stat = first["statistics"][0]
    assert stat["valid_observation_count"] == 3
    assert stat["exact_zero_fraction"] == 0.0
    focused = [
        row
        for row in first["focused"]
        if row["removed_index"] == 0 and row["retained_index"] == 1
    ]
    assert {row["paired_valid_observation_count"] for row in focused} == {3}


def test_stage4_matrix_order_commands_context_and_preflight_are_exact() -> None:
    jobs = stage4_slow_feature_ablation.stage4_jobs()
    assert [(job["logical_configuration"], job["seed"]) for job in jobs] == [
        (logical, seed)
        for logical in ("full_slow", "drop_slow_low_prior")
        for seed in (11, 29, 47)
    ]
    assert all(
        job["context_ablation"] == "drop_win_and_global_non_rates" for job in jobs
    )
    assert [job["feature_ablation"] for job in jobs] == [
        "none",
        "none",
        "none",
        "drop_slow_low_prior",
        "drop_slow_low_prior",
        "drop_slow_low_prior",
    ]
    assert len({job["job_specification_sha256"] for job in jobs}) == 6
    payload = {
        "logical_job_count": 6,
        "adopted_control_count": 3,
        "pending_training_job_count": 3,
    }
    assert stage4_slow_feature_ablation.format_dry_run_preflight(payload) == (
        "logical jobs: 6\n"
        "adopted controls: 3\n"
        "new training jobs: 3\n"
        "seeds: 11,29,47\n"
        "control: full_slow\n"
        "treatment: drop_slow_low_prior\n"
        "context: drop_win_and_global_non_rates\n"
        "test metrics accessed: no"
    )


def _stage3_source_state(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    jobs = []
    for base in stage3_jobs():
        logical = str(base["logical_configuration"])
        seed = int(base["seed"])
        jobs.append(
            {
                **base,
                "status": "completed",
                "result_origin": "trained_stage3",
                "run_dir": str(path.parent / f"{logical}_{seed}"),
                "run_manifest_sha256": hashlib.sha256(
                    f"{logical}/{seed}".encode()
                ).hexdigest(),
                "producing_git_commit_sha": "stage3-commit",
                "source_stage2_state": None,
                "source_stage2_job": None,
            }
        )
    state = {
        "state_version": 1,
        "sweep_name": "stage3_context_addition_matched_seeds",
        "status": "completed",
        "configuration": {"orchestrator_git_commit_sha": "stage3-commit"},
        "jobs": jobs,
    }
    path.write_text(json.dumps(state), encoding="utf-8")
    configuration = {
        "source_stage3_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_stage3_producing_commit": "stage3-commit",
    }
    return state, configuration


def test_stage3_adoption_accepts_only_exact_core_and_legacy_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "stage3.json"
    state, configuration = _stage3_source_state(state_path)
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_validate_stage3_source_configuration",
        lambda source, current: None,
    )

    def completed(job, source):
        run_dir = Path(str(job["run_dir"]))
        return run_dir, float(job["seed"]) / 1000.0, job["run_manifest_sha256"]

    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_stage3_completed_job_artifacts",
        completed,
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "validate_stage4_completed_run",
        lambda run_dir, config, feature, seed, commit, allow_legacy_none=False: (
            seed / 1000.0,
            {"run_manifest.json": hashlib.sha256(f"core/{seed}".encode()).hexdigest()},
            "legacy_implicit_none",
        ),
    )
    adopted = stage4_slow_feature_ablation._validated_stage3_adoptions(
        state_path, configuration
    )
    assert tuple(adopted) == (11, 29, 47)
    assert all(values[-1] == "legacy_implicit_none" for values in adopted.values())

    corrupted = copy.deepcopy(state)
    core = next(
        job for job in corrupted["jobs"] if job["logical_configuration"] == "core"
    )
    core["context_ablation"] = "drop_global_non_rates"
    state_path.write_text(json.dumps(corrupted), encoding="utf-8")
    configuration["source_stage3_state_sha256"] = hashlib.sha256(
        state_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="canonical matrix"):
        stage4_slow_feature_ablation._validated_stage3_adoptions(
            state_path, configuration
        )


def test_matched_delta_is_treatment_minus_control_and_requires_three_seeds() -> None:
    control = {11: 0.01, 29: 0.02, 47: 0.03}
    treatment = {11: 0.02, 29: 0.01, 47: 0.03}
    summary = _three_seed_summary(control, treatment)
    assert summary["delta_by_seed"] == {"11": 0.01, "29": -0.01, "47": 0.0}
    assert summary["mean_delta"] == pytest.approx(0.0)
    assert summary["nonnegative_seed_delta_count"] == 2
    with pytest.raises(ValueError, match="exact matched seeds"):
        _three_seed_summary({11: 0.0}, {11: 0.1})

    period = {
        "start": "2024-07-08",
        "end": "2025-06-30",
        "date_count": 244,
        "primary_ic": 0.2,
        "mean_gross_top_minus_bottom": 0.01,
        "mean_one_way_turnover": 0.3,
        "horizons": {
            f"{horizon}m": {
                "spearman_ic": 0.2,
                "gross_top_minus_bottom": 0.01,
                "one_way_turnover": 0.3,
            }
            for horizon in (30, 60, 120)
        },
    }
    treatment_period = copy.deepcopy(period)
    treatment_period["primary_ic"] = 0.25
    assert _period_delta(treatment_period, period)["delta_ic"] == pytest.approx(0.05)


def test_dry_run_builds_three_adopted_controls_and_three_pending_treatments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = {
        "source_stage3_state_sha256": "source-sha",
        "source_stage3_producing_commit": "stage3-commit",
    }
    adopted = {
        seed: (
            {
                "run_dir": str(tmp_path / f"source_{seed}"),
                "result_origin": "trained_stage3",
                "producing_git_commit_sha": "stage3-commit",
                "run_manifest_sha256": f"manifest-{seed}",
            },
            tmp_path / f"resolved_{seed}",
            seed / 1000.0,
            f"manifest-{seed}",
            {"run_manifest.json": f"manifest-{seed}"},
            "legacy_implicit_none",
        )
        for seed in (11, 29, 47)
    }
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_prepare",
        lambda **kwargs: ("current", True, tmp_path, configuration),
    )
    monkeypatch.setattr(
        stage4_slow_feature_ablation,
        "_validated_stage3_adoptions",
        lambda path, config: adopted,
    )
    payload = stage4_slow_feature_ablation.dry_run_payload(
        tmp_path / "stage3.json", tmp_path / "audit.json"
    )
    assert payload["logical_job_count"] == 6
    assert payload["adopted_control_count"] == 3
    assert payload["pending_training_job_count"] == 3
    assert [job["status"] for job in payload["jobs"]] == [
        "completed",
        "completed",
        "completed",
        "pending",
        "pending",
        "pending",
    ]


def _write_validation_run(
    run_dir: Path, score: float, manifest: dict[str, object]
) -> None:

    run_dir.mkdir()
    dates = []
    current = date(2024, 7, 8)
    while len(dates) < 243:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    dates.append(date(2025, 6, 30))
    rows = [
        {
            "trade_date": trade_date,
            "date_idx": date_index,
            "horizon_minutes": horizon,
            "spearman_ic": score + horizon / 1_000_000.0,
            "top_return": 0.02,
            "bottom_return": -0.01,
            "top_minus_bottom": 0.03 + score / 10.0,
            "long_only_top": 0.02,
            "one_way_turnover": 0.40 - score / 10.0,
        }
        for date_index, trade_date in enumerate(dates)
        for horizon in (30, 60, 120)
    ]
    daily = pl.DataFrame(rows)
    daily.write_parquet(run_dir / "validation_daily_metrics.parquet")
    horizon_rows = []
    for horizon in (30, 60, 120):
        selected = daily.filter(pl.col("horizon_minutes") == horizon)
        horizon_rows.append(
            {
                "horizon_minutes": horizon,
                "mean_daily_spearman_ic": float(selected["spearman_ic"].mean()),
                "mean_top_minus_bottom": float(selected["top_minus_bottom"].mean()),
                "mean_one_way_turnover": float(selected["one_way_turnover"].mean()),
            }
        )
    primary = float(np.mean([row["mean_daily_spearman_ic"] for row in horizon_rows]))
    (run_dir / "validation_metrics.json").write_text(
        json.dumps({"primary_score": primary, "horizons": horizon_rows})
    )
    pl.DataFrame(
        {"epoch": [1], "optimizer_steps": [77], "epoch_seconds": [2.0]}
    ).write_csv(run_dir / "history.csv")
    manifest.update(
        {
            "best_epoch": 1,
            "stopped_epoch": 1,
            "successful_optimizer_updates": 77,
            "training_duration_seconds": 2.0,
            "best_validation_primary_score": primary,
        }
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))


def test_validation_only_analyzer_requires_six_matched_runs_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import asdict

    from brazil_rv.modeling.contract import SplitBoundaries
    from brazil_rv.modeling.stage2_context_ablation import _training_semantics

    source = tmp_path / "stage3.json"
    source.write_text("{}")
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    feature_metadata = {
        key: _resolved_feature(key).metadata()
        for key in ("none", "drop_slow_low_prior")
    }
    configuration = {
        "orchestrator_git_commit_sha": "stage4-commit",
        "feature_store": {
            "resolved_path": str(tmp_path / "store"),
            "manifest_sha256": stage4_slow_feature_ablation.PACKAGED_FEATURE_MANIFEST_SHA256,
        },
        "feature_contract": "M1_FEATURES_INTRADAY_DI_MASKED_CONTEXT",
        "retained_context_symbols": list(
            stage4_slow_feature_ablation.EXPECTED_RETAINED_CONTEXTS
        ),
        "training_semantics": _training_semantics(),
        "split_boundaries": {
            key: str(value) for key, value in asdict(SplitBoundaries()).items()
        },
        "logical_configuration_order": ["full_slow", "drop_slow_low_prior"],
        "feature_ablation_by_logical_configuration": {
            "full_slow": "none",
            "drop_slow_low_prior": "drop_slow_low_prior",
        },
        "feature_ablation_metadata_by_key": feature_metadata,
        "context_ablation": "drop_win_and_global_non_rates",
        "context_ablation_metadata": get_context_ablation(
            "drop_win_and_global_non_rates"
        ).metadata(),
        "seeds": [11, 29, 47],
        "logical_job_count": 6,
        "adopted_stage3_job_count": 3,
        "new_training_job_count": 3,
        "source_stage3_state": str(source),
        "source_stage3_state_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_stage3_producing_commit": "stage3-commit",
        "required_feature_manifest_sha256": stage4_slow_feature_ablation.PACKAGED_FEATURE_MANIFEST_SHA256,
        "training_slow_audit": {
            "path": str(audit),
            "sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
            "audit_name": "stage4_training_slow_feature_redundancy",
            "audit_version": AUDIT_VERSION,
        },
    }
    jobs = []
    run_lookup = {}
    effects = {11: 0.002, 29: -0.001, 47: 0.0}
    for base in stage4_slow_feature_ablation.stage4_jobs():
        logical = str(base["logical_configuration"])
        seed = int(base["seed"])
        score = 0.05 + seed / 10000.0
        if logical == "drop_slow_low_prior":
            score += effects[seed]
        run_dir = tmp_path / f"{logical}_{seed}"
        manifest = {
            "context_ablation": get_context_ablation(
                "drop_win_and_global_non_rates"
            ).metadata(),
            "git_commit_sha": (
                "stage3-commit" if logical == "full_slow" else "stage4-commit"
            ),
        }
        if logical != "full_slow":
            manifest["feature_ablation"] = feature_metadata["drop_slow_low_prior"]
        _write_validation_run(run_dir, score, manifest)
        manifest_sha = hashlib.sha256(
            (run_dir / "run_manifest.json").read_bytes()
        ).hexdigest()
        job = {
            **base,
            "status": "completed",
            "result_origin": (
                "adopted_stage3" if logical == "full_slow" else "trained_stage4"
            ),
            "run_dir": str(run_dir),
            "run_manifest_sha256": manifest_sha,
            "output_sha256": {"run_manifest.json": manifest_sha},
            "feature_ablation_identity_source": (
                "legacy_implicit_none"
                if logical == "full_slow"
                else "explicit_registry_metadata"
            ),
            "producing_git_commit_sha": (
                "stage3-commit" if logical == "full_slow" else "stage4-commit"
            ),
            "primary_validation_ic": json.loads(
                (run_dir / "validation_metrics.json").read_text()
            )["primary_score"],
        }
        jobs.append(job)
        run_lookup[(logical, seed)] = run_dir
    state = {
        "state_version": 1,
        "sweep_name": "stage4_slow_low_prior_ablation_matched_seeds",
        "status": "completed",
        "configuration": configuration,
        "jobs": jobs,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))

    monkeypatch.setattr(
        analyze_stage4_slow_feature_ablation,
        "validate_training_slow_audit",
        lambda path, identity: {},
    )
    monkeypatch.setattr(
        analyze_stage4_slow_feature_ablation,
        "_validated_stage3_adoptions",
        lambda path, config: {},
    )

    def completed(job, config):
        path = Path(str(job["run_dir"]))
        return (
            path,
            float(job["primary_validation_ic"]),
            str(job["run_manifest_sha256"]),
            dict(job["output_sha256"]),
            str(job["feature_ablation_identity_source"]),
        )

    monkeypatch.setattr(
        analyze_stage4_slow_feature_ablation,
        "_completed_job_artifacts",
        completed,
    )
    monkeypatch.setattr(
        analyze_stage4_slow_feature_ablation,
        "validate_stage4_completed_run",
        lambda run_dir, config, feature, seed, commit, allow_legacy_none=False: (
            json.loads((run_dir / "validation_metrics.json").read_text())[
                "primary_score"
            ],
            {
                "run_manifest.json": hashlib.sha256(
                    (run_dir / "run_manifest.json").read_bytes()
                ).hexdigest()
            },
            (
                "legacy_implicit_none"
                if feature == "none"
                else "explicit_registry_metadata"
            ),
        ),
    )
    output = tmp_path / "analysis"
    paths = analyze_stage4_slow_feature_ablation.analyze_sweep(
        state_path, output, stage3_state_path=source, slow_audit_path=audit
    )
    summary = json.loads(paths[0].read_text())
    assert summary["three_seed_summary"]["delta_by_seed"] == {
        "11": pytest.approx(0.002),
        "29": pytest.approx(-0.001),
        "47": pytest.approx(0.0),
    }
    assert summary["three_seed_summary"]["nonnegative_seed_delta_count"] == 2
    assert summary["test_metrics_accessed"] is False
    assert summary["selection"] is None
    assert summary["automatic_winner_selection"] is False
    first_bytes = tuple(path.read_bytes() for path in paths)
    second_paths = analyze_stage4_slow_feature_ablation.analyze_sweep(
        state_path, output, stage3_state_path=source, slow_audit_path=audit
    )
    assert first_bytes == tuple(path.read_bytes() for path in second_paths)

    incomplete = copy.deepcopy(state)
    incomplete["jobs"][-1]["status"] = "failed"
    state_path.write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="completed six-job matrix"):
        analyze_stage4_slow_feature_ablation.analyze_sweep(
            state_path, output, stage3_state_path=source, slow_audit_path=audit
        )


def test_final_test_metadata_is_rejected() -> None:
    with pytest.raises(ValueError, match="forbidden test-derived field"):
        stage4_slow_feature_ablation._reject_test_derived_metadata(
            {"final_test_primary_ic": 0.1}, "synthetic Stage-4 state"
        )
