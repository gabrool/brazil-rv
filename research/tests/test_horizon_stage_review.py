from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from brazil_rv.modeling.contract import (
    BASELINE_TCN_SETTINGS,
    HORIZONS,
    TEST_START,
    architecture_for_model,
)
from brazil_rv.modeling.data import feature_store_identity
from brazil_rv.modeling.engine import checkpoint_payload
from brazil_rv.modeling.evaluate import load_current_neural_run
from brazil_rv.modeling.model import build_neural_model
from brazil_rv.modeling.run_horizon_multiscale_stage import (
    Arm,
    Stage,
    _comparison,
    _expected_run_config,
    _fingerprint,
    _run_arm,
)
from brazil_rv.modeling.stage_conclusions import build_hypothesis_summary
from brazil_rv.modeling.stage_validation import (
    read_json_object,
    validate_completed_run,
    validate_frozen_probes,
    validate_gradient_audit,
    validate_oof_plan,
    validate_split_contract,
)
from brazil_rv.preprocessing.contract import CONTRACT_VERSION


def _comparison_values(
    *, trained_positive: bool = False, score_delta: float = 0.002
) -> list[dict[str, object]]:
    def row(
        name: str,
        horizon: int,
        delta: float,
        lower: float | None = None,
        upper: float | None = None,
    ) -> dict[str, object]:
        return {
            "comparison": name,
            "horizon_minutes": horizon,
            "delta_ic": delta,
            "delta_lower_95": delta - 0.002 if lower is None else lower,
            "delta_upper_95": delta + 0.002 if upper is None else upper,
        }

    trained_delta = 0.01 if trained_positive else -0.001
    trained_lower = 0.004 if trained_positive else -0.004
    values = [
        row("shared_multiscale_vs_final_seed29", 0, 0.003),
        row("horizon_multiscale_vs_shared_seed29", 0, 0.004),
        row("final_score_mlp_vs_final_seed29", 0, score_delta),
        row("horizon_multiscale_vs_final_seed29", 0, 0.007),
    ]
    for horizon in (*HORIZONS, 0):
        values.append(
            row(
                "horizon_multiscale_vs_final_three_seed",
                horizon,
                trained_delta,
                trained_lower,
                trained_delta + 0.004,
            )
        )
    return values


def _hypothesis_inputs(
    *,
    probe_positive: bool,
    trained_positive: bool,
    score_delta: float = 0.002,
    single_delta: float = -0.001,
) -> dict[str, object]:
    flags = {str(horizon): False for horizon in HORIZONS}
    if probe_positive:
        flags["30"] = True
    return {
        "comparisons": _comparison_values(
            trained_positive=trained_positive, score_delta=score_delta
        ),
        "frozen_summary": {
            "best_tap_by_horizon": {
                "30": "block_2",
                "60": "block_4",
                "120": "block_6",
            },
            "earlier_tap_beats_final_post_fusion_by_horizon": flags,
            "concatenated_beats_final_post_fusion_by_horizon": {
                str(horizon): False for horizon in HORIZONS
            },
        },
        "gradient_summary": {
            "sample_count": 20,
            "by_group_and_horizon_pair": [{"fraction_negative": 0.5}],
        },
        "single_horizon_rows": [
            {"training_horizon": str(horizon), "delta_from_control": single_delta}
            for horizon in HORIZONS
        ],
        "context_summary": {"inference": {"baseline": {}}},
        "context_rows": [
            {
                "context_family": family,
                "horizon_minutes": 0,
                "delta_ic": 0.0,
                "delta_lower_95": -0.001,
                "delta_upper_95": 0.001,
            }
            for family in ("wdo", "br_rates", "us_rates")
        ],
        "oof_summary": {
            "results": [
                {
                    "probe": "slow",
                    "horizon_minutes": 30,
                    "delta_from_base": 0.001,
                }
            ]
        },
        "target_summary": {
            "pooled_target_correlation": np.eye(3).tolist(),
            "eigenvalues": [1.5, 1.0, 0.5],
            "variance_shares": [0.5, 1 / 3, 1 / 6],
            "fixed_basis_variance": [1.0, 0.5, 0.25],
        },
    }


def test_probe_positive_but_trained_negative_keeps_diagnostic_support() -> None:
    result = build_hypothesis_summary(
        **_hypothesis_inputs(probe_positive=True, trained_positive=False)
    )
    assert result["hypotheses"]["representation_information_loss"][
        "supported_diagnostically"
    ]
    assert not result["hypotheses"]["trained_multiscale_result"]["supported"]
    assert "did not exploit it consistently" in result["bottleneck_interpretation"]


def test_positive_trained_multiscale_result_is_supported() -> None:
    result = build_hypothesis_summary(
        **_hypothesis_inputs(probe_positive=False, trained_positive=True)
    )
    assert result["hypotheses"]["trained_multiscale_result"]["supported"]
    assert (
        "trained horizon-multiscale readout is supported"
        in result["bottleneck_interpretation"]
    )


def test_matching_score_mlp_is_identified_as_competing_explanation() -> None:
    result = build_hypothesis_summary(
        **_hypothesis_inputs(
            probe_positive=False,
            trained_positive=False,
            score_delta=0.007,
        )
    )
    assert result["hypotheses"]["score_capacity_control"]["competing_explanation"]


def test_neither_probes_nor_training_supports_bottleneck() -> None:
    result = build_hypothesis_summary(
        **_hypothesis_inputs(probe_positive=False, trained_positive=False)
    )
    assert (
        "Neither frozen representation probes nor the trained"
        in result["bottleneck_interpretation"]
    )


def test_negative_transfer_evidence_stays_separate_from_bottleneck() -> None:
    result = build_hypothesis_summary(
        **_hypothesis_inputs(
            probe_positive=False,
            trained_positive=False,
            single_delta=0.01,
        )
    )
    assert (
        len(result["hypotheses"]["horizon_conflict"]["single_horizon_improvements"])
        == 3
    )
    assert not result["hypotheses"]["representation_information_loss"][
        "supported_diagnostically"
    ]
    assert "Neither frozen representation probes" in result["bottleneck_interpretation"]


def _canonical_rows(split: str) -> pl.DataFrame:
    if split == "train":
        start, end, count = date(2021, 8, 16), date(2024, 6, 28), 716
    else:
        start, end, count = date(2024, 7, 8), date(2025, 6, 30), 244
    offsets = np.linspace(0, (end - start).days, count, dtype=np.int64)
    dates = [start + timedelta(days=int(offset)) for offset in offsets]
    rows = [
        {
            "sample_id": date_index * 55 + decision,
            "trade_date": trade_date,
            "date_idx": date_index,
            "decision_idx": decision,
        }
        for date_index, trade_date in enumerate(dates)
        for decision in range(55)
    ]
    return pl.DataFrame(rows)


def test_complete_preflight_split_contract() -> None:
    train = validate_split_contract(_canonical_rows("train"), "train")
    validation = validate_split_contract(_canonical_rows("validation"), "validation")
    assert (train["date_count"], train["sample_count"]) == (716, 716 * 55)
    assert (validation["date_count"], validation["sample_count"]) == (244, 244 * 55)


def test_split_contract_rejects_missing_final_date() -> None:
    rows = _canonical_rows("train").filter(pl.col("trade_date") < date(2024, 6, 28))
    with pytest.raises(ValueError, match="train end"):
        validate_split_contract(rows, "train")


def test_split_contract_rejects_incorrect_counts() -> None:
    rows = _canonical_rows("validation")
    removed = rows["trade_date"].unique().sort()[10]
    rows = rows.filter(pl.col("trade_date") != removed)
    with pytest.raises(ValueError, match="date count"):
        validate_split_contract(rows, "validation")


def test_split_contract_rejects_injected_test_row() -> None:
    rows = _canonical_rows("validation")
    injected = rows.head(1).with_columns(pl.lit(TEST_START).alias("trade_date"))
    with pytest.raises(ValueError, match="held-out test"):
        validate_split_contract(pl.concat((rows, injected)), "validation")


def test_stage_reuses_valid_completed_artifact_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"valid": true}', encoding="utf-8")
    stage = object.__new__(Stage)
    stage.commit = "abc"
    stage.store_identity = {"id": "store"}
    config = {"mode": "audit"}
    stage.manifest = {
        "steps": {
            "audit": {
                "status": "completed",
                "fingerprint": _fingerprint(
                    {
                        "commit": stage.commit,
                        "feature_store": stage.store_identity,
                        "config": config,
                    }
                ),
            }
        }
    }
    stage.logger = logging.getLogger("review-resume")
    stage.write_manifest = lambda: None
    called = False

    def action() -> None:
        nonlocal called
        called = True

    stage.step(
        "audit",
        config,
        (artifact,),
        action,
        lambda: read_json_object(artifact),
    )
    assert not called
    artifact.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        stage.step(
            "audit",
            config,
            (artifact,),
            action,
            lambda: read_json_object(artifact),
        )


def test_failed_step_retries_only_its_action(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    stage = object.__new__(Stage)
    stage.commit = "abc"
    stage.store_identity = {"id": "store"}
    stage.manifest = {"steps": {"audit": {"status": "failed"}}}
    stage.logger = logging.getLogger("review-failed-retry")
    stage.write_manifest = lambda: None
    calls = 0

    def action() -> None:
        nonlocal calls
        calls += 1
        artifact.write_text('{"valid": true}', encoding="utf-8")

    stage.step(
        "audit",
        {"mode": "audit"},
        (artifact,),
        action,
        lambda: read_json_object(artifact),
    )
    assert calls == 1
    assert stage.manifest["steps"]["audit"]["status"] == "completed"


def test_incomplete_training_directory_is_archived_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    (run_dir / "partial.txt").write_text("preserve", encoding="utf-8")
    rows = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "sample_id": [0],
            "date_idx": [0],
            "decision_idx": [0],
        }
    )

    def fake_run(*_: object, **__: object) -> None:
        (run_dir / "new.txt").write_text("retry", encoding="utf-8")

    monkeypatch.setattr(
        "brazil_rv.modeling.run_horizon_multiscale_stage._run_neural", fake_run
    )
    archive_events: list[dict[str, object]] = []
    result = _run_arm(
        run_dir,
        Arm("retry"),
        tmp_path,
        rows,
        rows,
        fit_name="B0",
        selection_name="B1",
        expected={"test": True},
        step_identity="training_retry",
        record_archive=archive_events.append,
    )
    assert result is None
    assert len(archive_events) == 1
    archived = Path(str(archive_events[0]["path"]))
    assert archived is not None and ".incomplete." in archived.name
    assert (archived / "partial.txt").read_text(encoding="utf-8") == "preserve"
    assert (run_dir / "new.txt").read_text(encoding="utf-8") == "retry"
    assert read_json_object(run_dir / "run_manifest.json")["status"] == "running"


def test_corrupt_and_incomplete_artifact_schemas_are_rejected(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        validate_oof_plan(plan, {})

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "frozen_block_probes.csv").write_text(
        "candidate\nblock_1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing columns"):
        validate_frozen_probes(frozen)

    gradient = tmp_path / "gradient"
    gradient.mkdir()
    (gradient / "horizon_gradient_audit.parquet").write_text(
        "not parquet", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid Parquet"):
        validate_gradient_audit(gradient)


def _feature_store(path: Path) -> Path:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"contract_version": CONTRACT_VERSION}), encoding="utf-8"
    )
    (path / "feature_schema.json").write_text(
        json.dumps({"contract_version": CONTRACT_VERSION}), encoding="utf-8"
    )
    pl.DataFrame({"sample_id": [0]}).write_parquet(path / "sample_index.parquet")
    return path


def _small_window(name: str, start: date) -> tuple[pl.DataFrame, dict[str, object]]:
    rows = pl.DataFrame(
        {
            "trade_date": [start, start + timedelta(days=1)],
            "sample_id": [0, 1],
            "date_idx": [0, 1],
            "decision_idx": [0, 0],
        }
    )
    return rows, {
        "name": name,
        "start": str(start),
        "end": str(start + timedelta(days=1)),
        "date_count": 2,
        "sample_count": 2,
    }


def _completed_run(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    store = _feature_store(tmp_path / "store")
    fit_rows, _ = _small_window("B0", date(2022, 1, 3))
    selection_rows, _ = _small_window("B1", date(2022, 2, 1))
    arm = Arm("control", training_horizon="60", context_ablation="wdo")
    expected = _expected_run_config(
        arm,
        store,
        fit_rows,
        selection_rows,
        "B0",
        "B1",
        True,
        repository_commit_value="test-commit",
    )
    settings = BASELINE_TCN_SETTINGS
    architecture = architecture_for_model("tcn", settings)
    torch.manual_seed(29)
    model = build_neural_model("tcn", architecture, "selected")
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        "tcn",
        architecture,
        settings,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
        29,
        1,
        0.1,
        store,
        "enabled",
        "selected",
        "60",
        "wdo",
        run_provenance=expected,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(checkpoint, run_dir / "best_checkpoint.pt")
    manifest = {
        "status": "completed",
        "repository_commit": expected["repository_commit"],
        "run_provenance": expected,
        "feature_store": str(store),
        "feature_store_identity": feature_store_identity(store),
        "split": {
            "training": "B0",
            "selection": "B1",
            "fit_window": expected["fit_window"],
            "selection_window": expected["selection_window"],
            "test_accessed": False,
        },
        "seed": 29,
        "global_context": "enabled",
        "training_horizon": "60",
        "selection_horizon": "60",
        "context_family_ablation": "wdo",
        "model": expected["model"],
        "parameter_count": expected["parameter_count"],
        "objective": expected["objective"],
        "optimizer": expected["optimizer"],
        "sam": expected["sam"],
        "training": expected["training"],
        "best_epoch": 1,
        "best_validation_score": 0.1,
        "epochs_completed": 1,
        "total_run_seconds": 3.0,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "primary_score": 0.1,
                "horizons": [
                    {
                        "horizon_minutes": horizon,
                        "mean_daily_spearman_ic": 0.1,
                    }
                    for horizon in HORIZONS
                ],
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        [
            {
                "date_idx": date_idx,
                "horizon_minutes": horizon,
                "spearman_ic": 0.1,
            }
            for date_idx in range(2)
            for horizon in HORIZONS
        ]
    ).write_parquet(run_dir / "validation_daily_metrics.parquet")
    pl.DataFrame(
        {
            "epoch": [1],
            "train_objective_loss": [0.2],
            "validation_objective_loss": [0.2],
            "validation_primary_ic": [0.1],
            "optimizer_steps": [1],
        }
    ).write_csv(run_dir / "history.csv")
    return run_dir, store, expected


def test_completed_run_validates_full_provenance_and_real_reconstruction(
    tmp_path: Path,
) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    validate_completed_run(run_dir, store, expected)
    _, checkpoint, restored_store = load_current_neural_run(run_dir)
    assert restored_store == store
    assert checkpoint["training_horizon"] == "60"
    assert checkpoint["selection_horizon"] == "60"
    assert checkpoint["context_family_ablation"] == "wdo"


def test_completed_run_rejects_one_material_setting(tmp_path: Path) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "run_manifest.json"
    manifest = read_json_object(path)
    manifest["global_context"] = "masked"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="global_context differs"):
        validate_completed_run(run_dir, store, expected)


def test_completed_run_rejects_wrong_fit_window(tmp_path: Path) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "run_manifest.json"
    manifest = read_json_object(path)
    manifest["split"]["fit_window"]["end"] = "2022-01-31"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fit window differs"):
        validate_completed_run(run_dir, store, expected)


def test_context_ablation_comparison_is_paired_and_deterministic(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    control.mkdir()
    candidate.mkdir()
    for path, offset in ((control, 0.0), (candidate, 0.01)):
        (path / "validation_metrics.json").write_text(
            json.dumps(
                {
                    "primary_score": 0.1 + offset,
                    "horizons": [
                        {
                            "horizon_minutes": horizon,
                            "mean_daily_spearman_ic": 0.1 + offset,
                        }
                        for horizon in HORIZONS
                    ],
                }
            ),
            encoding="utf-8",
        )
        pl.DataFrame(
            [
                {
                    "date_idx": date_idx,
                    "horizon_minutes": horizon,
                    "spearman_ic": 0.1 + offset + date_idx * 0.0001,
                }
                for date_idx in range(12)
                for horizon in HORIZONS
            ]
        ).write_parquet(path / "validation_daily_metrics.parquet")
    first, _ = _comparison("without_wdo_vs_final_seed29", candidate, control, 29)
    second, _ = _comparison("without_wdo_vs_final_seed29", candidate, control, 29)
    assert {int(row["horizon_minutes"]) for row in first} == {*HORIZONS, 0}
    for left, right in zip(first, second, strict=True):
        assert left == right
        assert np.isclose(float(left["delta_ic"]), 0.01)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_completed_run_rejects_wrong_repository_commit(tmp_path: Path) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "run_manifest.json"
    manifest = read_json_object(path)
    manifest["repository_commit"] = "wrong-commit"
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="repository commit differs"):
        validate_completed_run(run_dir, store, expected)


@pytest.mark.parametrize(
    "field",
    ("effective_batch_size", "early_stop_patience", "warmup_steps"),
)
def test_completed_run_rejects_changed_training_contract(
    tmp_path: Path, field: str
) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "run_manifest.json"
    manifest = read_json_object(path)
    manifest["run_provenance"]["training"][field] += 1
    manifest["training"][field] += 1
    _write_json(path, manifest)
    with pytest.raises(ValueError, match="run provenance differs"):
        validate_completed_run(run_dir, store, expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("epoch", 2, "checkpoint best epoch differs"),
        ("validation_score", 0.2, "checkpoint validation score differs"),
    ),
)
def test_completed_run_rejects_checkpoint_result_mismatch(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "best_checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint[field] = value
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=message):
        validate_completed_run(run_dir, store, expected)


def test_completed_run_rejects_history_best_score_mismatch(tmp_path: Path) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "history.csv"
    history = pl.read_csv(path).with_columns(pl.lit(0.2).alias("validation_primary_ic"))
    history.write_csv(path)
    with pytest.raises(ValueError, match="history best validation score differs"):
        validate_completed_run(run_dir, store, expected)


def test_completed_run_rejects_metrics_that_disagree_with_daily(
    tmp_path: Path,
) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "validation_metrics.json"
    metrics = read_json_object(path)
    metrics["horizons"][0]["mean_daily_spearman_ic"] = 0.2
    _write_json(path, metrics)
    with pytest.raises(ValueError, match="validation horizon IC 30 differs"):
        validate_completed_run(run_dir, store, expected)


@pytest.mark.parametrize("mutation", ("duplicate", "missing"))
def test_completed_run_rejects_duplicate_or_missing_daily_keys(
    tmp_path: Path, mutation: str
) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    path = run_dir / "validation_daily_metrics.parquet"
    daily = pl.read_parquet(path)
    if mutation == "duplicate":
        daily = pl.concat((daily.slice(0, daily.height - 1), daily.head(1)))
        message = "duplicate keys"
    else:
        daily = daily.slice(0, daily.height - 1)
        message = "validation daily rows differs"
    daily.write_parquet(path)
    with pytest.raises(ValueError, match=message):
        validate_completed_run(run_dir, store, expected)


def test_completed_run_rejects_same_config_cross_artifact_result_mismatch(
    tmp_path: Path,
) -> None:
    run_dir, store, expected = _completed_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json_object(manifest_path)
    manifest["best_validation_score"] = 0.2
    _write_json(manifest_path, manifest)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["validation_score"] = 0.2
    torch.save(checkpoint, checkpoint_path)
    history_path = run_dir / "history.csv"
    pl.read_csv(history_path).with_columns(
        pl.lit(0.2).alias("validation_primary_ic")
    ).write_csv(history_path)
    with pytest.raises(ValueError, match="selected validation score differs"):
        validate_completed_run(run_dir, store, expected)


def _retry_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2)],
            "sample_id": [0],
            "date_idx": [0],
            "decision_idx": [0],
        }
    )


def test_archive_record_is_persisted_before_retry_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_manifest.json", {"status": "failed"})
    stage = object.__new__(Stage)
    stage.manifest_path = tmp_path / "stage_manifest.json"
    stage.manifest = {"archived_incomplete_runs": []}

    def fail_retry(*_: object, **__: object) -> None:
        persisted = read_json_object(stage.manifest_path)["archived_incomplete_runs"]
        assert len(persisted) == 1
        assert Path(str(persisted[0]["path"])).exists()
        raise RuntimeError("retry failed")

    monkeypatch.setattr(
        "brazil_rv.modeling.run_horizon_multiscale_stage._run_neural", fail_retry
    )
    with pytest.raises(RuntimeError, match="retry failed"):
        _run_arm(
            run_dir,
            Arm("retry"),
            tmp_path,
            _retry_rows(),
            _retry_rows(),
            expected={"test": True},
            step_identity="train_retry",
            record_archive=stage.record_incomplete_archive,
        )
    persisted = read_json_object(stage.manifest_path)["archived_incomplete_runs"]
    assert persisted[0]["prior_status"] == "failed"


def test_repeated_failed_retries_append_archive_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_json(run_dir / "run_manifest.json", {"status": "running"})
    stage = object.__new__(Stage)
    stage.manifest_path = tmp_path / "stage_manifest.json"
    stage.manifest = {"archived_incomplete_runs": []}

    def fail_retry(*_: object, **__: object) -> None:
        raise RuntimeError("retry failed")

    monkeypatch.setattr(
        "brazil_rv.modeling.run_horizon_multiscale_stage._run_neural", fail_retry
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="retry failed"):
            _run_arm(
                run_dir,
                Arm("retry"),
                tmp_path,
                _retry_rows(),
                _retry_rows(),
                expected={"test": True},
                step_identity="train_retry",
                record_archive=stage.record_incomplete_archive,
            )
    persisted = read_json_object(stage.manifest_path)["archived_incomplete_runs"]
    assert len(persisted) == 2
    assert len({row["path"] for row in persisted}) == 2
    assert all(row["prior_status"] == "running" for row in persisted)


@pytest.mark.parametrize(
    ("manifest", "message"),
    (
        (None, "no manifest"),
        ("{", "Invalid JSON"),
        ({"status": "mystery"}, "Ambiguous run status"),
    ),
)
def test_ambiguous_or_corrupt_run_directories_are_refused(
    tmp_path: Path, manifest: object, message: str
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    if isinstance(manifest, dict):
        _write_json(run_dir / "run_manifest.json", manifest)
    elif isinstance(manifest, str):
        (run_dir / "run_manifest.json").write_text(manifest, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _run_arm(
            run_dir,
            Arm("retry"),
            tmp_path,
            _retry_rows(),
            _retry_rows(),
            expected={"test": True},
            step_identity="train_retry",
            record_archive=lambda _: None,
        )
