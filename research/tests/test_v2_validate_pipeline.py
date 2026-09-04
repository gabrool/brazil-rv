from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import FINETUNE_START, HORIZONS
from brazil_rv.v2.score import ScoreArtifact
from brazil_rv.v2.store import V2Store, open_store_for_dates, write_store
from brazil_rv.v2.train import StageTrainingResult
from brazil_rv.v2 import validate_pipeline as pipeline


def _development_store(tmp_path: Path) -> tuple[Path, Path, str, Path, str]:
    date_axis = np.arange(
        np.datetime64("2010-01-04"),
        np.datetime64("2024-12-31"),
        dtype="datetime64[D]",
    )
    date_axis = date_axis[np.is_busday(date_axis)]
    dates = date_axis.astype(object).tolist()
    day_count = len(dates)
    name_count = 4
    slow = np.zeros((day_count, name_count, 32), dtype=np.float32)
    intraday = np.zeros((day_count, name_count, 20), dtype=np.float32)
    name_axis = np.arange(name_count, dtype=np.float32)[None, :, None]
    horizon_axis = np.arange(len(HORIZONS), dtype=np.float32)[None, None, :]
    targets = np.broadcast_to(
        name_axis + 0.01 * horizon_axis,
        (day_count, name_count, len(HORIZONS)),
    ).copy()
    close = (
        100.0
        + np.arange(day_count, dtype=np.float64)[:, None] * 0.01
        + np.arange(name_count, dtype=np.float64)[None, :]
    )
    store_root = write_store(
        tmp_path / "store",
        dates=dates,
        isins=[f"BRTEST{index:02d}NOR1" for index in range(name_count)],
        arrays={
            "active": np.ones((day_count, name_count), dtype=np.bool_),
            "observed": np.ones((day_count, name_count), dtype=np.bool_),
            "target_exclusion_event_mask": np.zeros(
                (day_count, name_count), dtype=np.bool_
            ),
            "ambiguous_action_mask": np.zeros(
                (day_count, name_count), dtype=np.bool_
            ),
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "intraday_values": intraday,
            "intraday_valid": np.ones_like(intraday, dtype=np.bool_),
            "fast_present": np.zeros((day_count, name_count), dtype=np.bool_),
            "target_primary": targets,
            "target_valid": np.ones_like(targets, dtype=np.bool_),
            "target_raw_midrank": targets.copy(),
            "target_raw_valid": np.ones_like(targets, dtype=np.bool_),
            "target_raw_log_return": targets.astype(np.float64) * 0.0001,
            "adjusted_close": close,
        },
        feature_names={
            "slow": [f"slow_{index}" for index in range(32)],
            "intraday": [f"intraday_{index}" for index in range(20)],
        },
        sources=[{"path": "fixture", "sha256": "a" * 64}],
        metadata={
            "v1_isin_subset_verified": True,
            "v1_calendar_verified": True,
            "implementation_git_commit": "1" * 40,
        },
    )
    development_dates = [value for value in dates if value >= FINETUNE_START]
    cdi_path = tmp_path / "daily_cdi.parquet"
    pl.DataFrame(
        {
            "trade_date": development_dates,
            "daily_cdi_rate": [0.0004] * len(development_dates),
        }
    ).write_parquet(cdi_path)
    experiment52_cdi_path = tmp_path / "experiment52_daily_cdi.parquet"
    pl.DataFrame(
        {
            "trade_date": development_dates[:-20],
            "daily_cdi_rate": [0.0004] * (len(development_dates) - 20),
        }
    ).write_parquet(experiment52_cdi_path)
    return (
        store_root,
        cdi_path,
        sha256_file(cdi_path),
        experiment52_cdi_path,
        sha256_file(experiment52_cdi_path),
    )


class _FakeGBDT:
    fit_calls: list[tuple[int, int]] = []
    saved: dict[Path, _FakeGBDT] = {}

    def __init__(self, config, *, feature_names) -> None:
        self.config = config
        self.feature_names = tuple(feature_names)
        self.models = {}

    def fit(
        self,
        train_features,
        train_targets,
        train_mask,
        validation_features,
        validation_targets,
        validation_mask,
        **kwargs,
    ):
        del train_targets, train_mask, validation_targets, validation_mask, kwargs
        self.fit_calls.append((len(train_features), len(validation_features)))
        self.models = {
            head: [_FakeBooster(head, seed) for seed in self.config.seeds]
            for head in range(len(HORIZONS))
        }
        return self

    def predict_ranks(self, features, score_mask):
        name_rank = np.arange(features.shape[1], dtype=np.float32)[None, :, None]
        result = np.broadcast_to(name_rank, score_mask.shape).copy()
        return np.where(score_mask, result, 0.0).astype(np.float32)

    def predict_raw(self, features):
        head = np.arange(len(HORIZONS), dtype=np.float32)[None, None, :]
        return np.broadcast_to(
            head, (features.shape[0], features.shape[1], len(HORIZONS))
        ).copy()

    def save(self, root, *, metadata=None):
        output = Path(root).resolve()
        output.mkdir(parents=True)
        for head, models in self.models.items():
            for seed, booster in zip(self.config.seeds, models, strict=True):
                booster.save_model(str(output / f"head_{head}_seed_{seed}.txt"))
        manifest = output / "model_manifest.json"
        manifest_sha = write_json_atomic(
            manifest,
            {
                "schema": "BRAZIL_RV_V2_GBDT_MODELS_V1",
                **dict(metadata or {}),
            },
        )
        self.saved[output] = self
        return manifest, manifest_sha

    @classmethod
    def load(cls, root, *, expected_manifest_sha256):
        output = Path(root).resolve()
        if sha256_file(output / "model_manifest.json") != expected_manifest_sha256:
            raise ValueError("manifest mismatch")
        return cls.saved[output]

    def feature_importance(self, features):
        width = features.shape[-1]
        return {
            "gain": np.arange(width, dtype=np.float64),
            "split": np.zeros(width, dtype=np.float64),
        }


class _FakeBooster:
    def __init__(self, head: int, seed: int) -> None:
        self.head = head
        self.seed = seed

    def save_model(self, path: str) -> None:
        Path(path).write_text(
            f"fake-lightgbm head={self.head} seed={self.seed}\n", encoding="ascii"
        )


def test_pipeline_rejects_store_built_by_a_different_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_root, cdi_path, cdi_sha, reference_path, reference_sha = (
        _development_store(tmp_path)
    )
    monkeypatch.setattr(
        pipeline,
        "_git_identity",
        lambda: {"commit": "2" * 40, "tracked_worktree_clean": True},
    )
    with pytest.raises(ValueError, match="store implementation commit differs"):
        pipeline.run_pipeline_validation(
            store_root=store_root,
            cdi_path=cdi_path,
            cdi_sha256=cdi_sha,
            experiment52_cdi_path=reference_path,
            experiment52_cdi_sha256=reference_sha,
            output_root=tmp_path / "validation",
        )


def test_development_pipeline_orchestrates_and_seals_every_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        store_root,
        cdi_path,
        cdi_sha,
        experiment52_cdi_path,
        experiment52_cdi_sha,
    ) = _development_store(tmp_path)
    training_calls: list[dict[str, object]] = []

    def fake_train_stage(**kwargs) -> StageTrainingResult:
        training_calls.append(kwargs)
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        raw = output / "raw_patience.pt"
        ema = output / "final_ema.pt"
        raw.write_bytes(f"raw-{len(training_calls)}".encode())
        ema.write_bytes(f"ema-{len(training_calls)}".encode())
        history = output / "history.json"
        write_json_atomic(history, [{"epoch": 1, "selection_score": 0.0}])
        pretrain = kwargs.get("pretrain_checkpoint")
        manifest = output / "run_manifest.json"
        write_json_atomic(
            manifest,
            {
                "schema": "BRAZIL_RV_V2_TRAINING_STAGE_V1",
                "status": "completed",
                "stage": kwargs["stage"],
                "seed": kwargs["seed"],
                "fold": kwargs["fold"],
                "pretrain_checkpoint_sha256": (
                    None if pretrain is None else sha256_file(Path(pretrain))
                ),
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        return StageTrainingResult(
            stage=kwargs["stage"],
            seed=kwargs["seed"],
            fold=kwargs["fold"],
            epochs_completed=1,
            raw_patience_checkpoint=raw,
            final_ema_checkpoint=ema,
            history_path=history,
            manifest_path=manifest,
            selected_epoch=1,
            stopped_epoch=1,
            selection_parity=kwargs["selection_parity"],
            evaluation_parity=(
                None
                if kwargs["selection_parity"] is None
                else 1 - kwargs["selection_parity"]
            ),
        )

    def fake_score_checkpoint_artifact(**kwargs) -> ScoreArtifact:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True)
        dataset = kwargs["loader"].dataset
        indices = np.asarray(dataset.date_indices, dtype=np.int64)
        active = np.asarray(dataset.store.read("active", indices), dtype=np.bool_)
        score_mask = np.repeat(active[..., None], len(HORIZONS), axis=-1)
        name_rank = np.arange(active.shape[1], dtype=np.float32)[None, :, None]
        scores = np.where(
            score_mask, np.broadcast_to(name_rank, score_mask.shape), 0.0
        ).astype(np.float32)
        scores_path = output / "scores.npy"
        mask_path = output / "score_mask.npy"
        dates_path = output / "date_index.npy"
        isins_path = output / "isin_index.npy"
        np.save(scores_path, scores, allow_pickle=False)
        np.save(mask_path, score_mask, allow_pickle=False)
        np.save(dates_path, dataset.store.dates[indices], allow_pickle=False)
        np.save(isins_path, np.asarray(dataset.store.isins), allow_pickle=False)
        manifest = output / "score_manifest.json"
        manifest_sha = write_json_atomic(
            manifest,
            {
                "schema": "BRAZIL_RV_V2_SCORE_ARTIFACT_V1",
                "status": "completed",
                "official_validation_accessed": False,
                "test_accessed": False,
            },
        )
        checkpoint = Path(kwargs["checkpoint"])
        return ScoreArtifact(
            root=output,
            scores_path=scores_path,
            score_mask_path=mask_path,
            date_index_path=dates_path,
            isin_index_path=isins_path,
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            checkpoint_sha256=sha256_file(checkpoint),
        )

    _FakeGBDT.fit_calls.clear()
    monkeypatch.setattr(
        pipeline,
        "_git_identity",
        lambda: {"commit": "1" * 40, "tracked_worktree_clean": True},
    )
    monkeypatch.setattr(pipeline, "MultiHorizonGBDT", _FakeGBDT)
    monkeypatch.setattr(pipeline, "train_stage", fake_train_stage)
    monkeypatch.setattr(
        pipeline, "score_checkpoint_artifact", fake_score_checkpoint_artifact
    )
    runtime = pipeline.ValidationRuntime(
        fine_epochs=1,
        handoff_epochs=1,
        gbdt_maximum_rounds=2,
        gbdt_early_stopping_rounds=1,
        max_fit_sessions=12,
        max_selection_sessions=12,
        max_pretrain_fit_sessions=12,
        max_pretrain_selection_sessions=12,
        slow_lookback=20,
        pairs_per_batch=8,
        compile_forward=False,
        device="cpu",
    )
    result = pipeline.run_pipeline_validation(
        store_root=store_root,
        cdi_path=cdi_path,
        cdi_sha256=cdi_sha,
        experiment52_cdi_path=experiment52_cdi_path,
        experiment52_cdi_sha256=experiment52_cdi_sha,
        output_root=tmp_path / "validation",
        runtime=runtime,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["pipeline_validation"] is True
    assert manifest["research_claim"] is False
    assert manifest["official_validation_accessed"] is False
    assert manifest["test_accessed"] is False
    assert manifest["date_contract"]["maximum_date"] <= "2024-12-30"
    cdi_source = manifest["sources"]["cdi"]
    assert cdi_source["development_extension"] == {
        "path": str(cdi_path.resolve()),
        "sha256": cdi_sha,
    }
    assert cdi_source["experiment52_reference"] == {
        "path": str(experiment52_cdi_path.resolve()),
        "sha256": experiment52_cdi_sha,
    }
    assert cdi_source["equality_proof"] == {
        "comparison_columns": ["trade_date", "daily_cdi_rate"],
        "reference_fully_contained": True,
        "overlap_count": pl.read_parquet(experiment52_cdi_path).height,
        "overlap_date_range": {
            "start": str(pl.read_parquet(experiment52_cdi_path).item(0, "trade_date")),
            "end": str(pl.read_parquet(experiment52_cdi_path).item(-1, "trade_date")),
        },
        "max_abs_daily_cdi_rate": 0.0,
        "exact_byte_match": True,
    }
    assert len(manifest["results"]["baselines"]) == 12
    assert len(manifest["results"]["gbdt_triage"]) == 2
    assert len(_FakeGBDT.fit_calls) == 4
    assert all(
        record["seeds"] == [11, 29, 47, 61, 79]
        for record in manifest["results"]["gbdt_triage"]
    )
    assert len(list((result.root / "gbdt_triage" / "models").rglob("*.txt"))) == 100
    assert len(training_calls) == 6
    assert all(
        call["train_loader"].batch_sampler.pairs_per_batch == 8
        and call["train_loader"].batch_sampler.drop_last is True
        for call in training_calls
    )
    assert [call["stage"] for call in training_calls].count("P") == 1
    persistence = [
        call
        for call in training_calls
        if call["model_config"].lambda_persistence == 0.1
    ]
    assert len(persistence) == 2
    assert all(call["maximum_epochs"] == 1 for call in persistence)
    handoff = next(
        call for call in training_calls if call.get("pretrain_checkpoint") is not None
    )
    assert handoff["stage"] == "F"
    assert handoff["expected_pretrain_sha256"] == sha256_file(
        Path(handoff["pretrain_checkpoint"])
    )

    for path in result.root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            assert payload["pipeline_validation"] is True, path
            assert payload["research_claim"] is False, path
            assert payload["official_validation_accessed"] is False, path
            assert payload["test_accessed"] is False, path
    inventory_payload = json.loads(result.inventory_path.read_text(encoding="utf-8"))
    for row in inventory_payload["files"]:
        artifact = result.root / row["path"]
        assert artifact.stat().st_size == row["bytes"]
        assert sha256_file(artifact) == row["sha256"]
    assert result.manifest_sha256 == sha256_file(result.manifest_path)
    assert result.inventory_sha256 == sha256_file(result.inventory_path)

    with pytest.raises(FileExistsError):
        pipeline.run_pipeline_validation(
            store_root=store_root,
            cdi_path=cdi_path,
            cdi_sha256=cdi_sha,
            experiment52_cdi_path=experiment52_cdi_path,
            experiment52_cdi_sha256=experiment52_cdi_sha,
            output_root=result.root,
            runtime=runtime,
        )


def test_development_cdi_requires_exact_experiment52_overlap(tmp_path: Path) -> None:
    (
        store_root,
        cdi_path,
        cdi_sha,
        experiment52_cdi_path,
        _,
    ) = _development_store(tmp_path)
    reference = pl.read_parquet(experiment52_cdi_path)
    reference = reference.with_columns(
        pl.when(pl.col("trade_date") == reference.item(0, "trade_date"))
        .then(pl.col("daily_cdi_rate") + 1e-12)
        .otherwise(pl.col("daily_cdi_rate"))
        .alias("daily_cdi_rate")
    )
    reference.write_parquet(experiment52_cdi_path)
    dates = np.load(store_root / "date_index.npy", allow_pickle=False)

    with pytest.raises(ValueError, match="differs from the Experiment-52 reference"):
        pipeline._load_development_cdi(
            dates=dates,
            cdi_path=cdi_path,
            expected_sha256=cdi_sha,
            experiment52_cdi_path=experiment52_cdi_path,
            experiment52_expected_sha256=sha256_file(experiment52_cdi_path),
        )


def test_development_cdi_requires_full_experiment52_span(tmp_path: Path) -> None:
    (
        store_root,
        cdi_path,
        cdi_sha,
        experiment52_cdi_path,
        _,
    ) = _development_store(tmp_path)
    reference = pl.read_parquet(experiment52_cdi_path)
    reference = pl.concat(
        [
            pl.DataFrame(
                {
                    "trade_date": [FINETUNE_START.replace(day=13)],
                    "daily_cdi_rate": [0.0004],
                }
            ),
            reference,
        ]
    )
    reference.write_parquet(experiment52_cdi_path)
    dates = np.load(store_root / "date_index.npy", allow_pickle=False)

    with pytest.raises(ValueError, match="span is not fully contained"):
        pipeline._load_development_cdi(
            dates=dates,
            cdi_path=cdi_path,
            expected_sha256=cdi_sha,
            experiment52_cdi_path=experiment52_cdi_path,
            experiment52_expected_sha256=sha256_file(experiment52_cdi_path),
        )


def test_development_cdi_hash_checks_experiment52_reference(tmp_path: Path) -> None:
    (
        store_root,
        cdi_path,
        cdi_sha,
        experiment52_cdi_path,
        _,
    ) = _development_store(tmp_path)
    dates = np.load(store_root / "date_index.npy", allow_pickle=False)

    with pytest.raises(ValueError, match="Experiment-52 CDI series SHA256 mismatch"):
        pipeline._load_development_cdi(
            dates=dates,
            cdi_path=cdi_path,
            expected_sha256=cdi_sha,
            experiment52_cdi_path=experiment52_cdi_path,
            experiment52_expected_sha256="0" * 64,
        )


def test_parser_requires_and_documents_both_cdi_sources() -> None:
    parser = pipeline._parser()
    required = {action.dest for action in parser._actions if action.required}
    assert {
        "cdi_path",
        "cdi_sha256",
        "experiment52_cdi_path",
        "experiment52_cdi_sha256",
    }.issubset(required)
    help_text = parser.format_help()
    assert "--experiment52-cdi-path" in help_text
    assert "Exact Experiment-52 reference daily_cdi.parquet" in help_text


def test_runtime_caps_and_dataset_refuse_sealed_dates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one and three"):
        pipeline.ValidationRuntime(fine_epochs=4)
    with pytest.raises(ValueError, match="exactly 8 date pairs"):
        pipeline.ValidationRuntime(pairs_per_batch=7)
    store = V2Store(
        root=tmp_path,
        manifest={},
        dates=np.asarray(["2025-01-02"], dtype="datetime64[D]"),
        isins=("BRTESTACNOR1",),
        _arrays={},
    )
    with pytest.raises(PermissionError, match="2025/2026"):
        pipeline._dataset(
            store,
            np.asarray([0], dtype=np.int64),
            stage="evaluation",
            purpose="evaluation",
            lookback=20,
            sidecars=(),
        )


def test_evaluation_inputs_zero_targets_outside_the_exact_window(
    tmp_path: Path,
) -> None:
    store_root, _, _, _, _ = _development_store(tmp_path)
    dates = np.load(store_root / "date_index.npy", allow_pickle=False)
    authorized = np.arange(len(dates) - 3, len(dates), dtype=np.int64)
    store, _ = open_store_for_dates(
        store_root,
        authorized,
        purpose="evaluation",
    )
    indices = authorized[:2]
    score_shape = (len(indices), len(store.isins), len(HORIZONS))
    try:
        inputs = pipeline._evaluation_inputs(
            store,
            indices,
            np.zeros(score_shape, dtype=np.float32),
            np.ones(score_shape, dtype=np.bool_),
            np.full(len(dates), 0.0004, dtype=np.float64),
            {},
        )
    finally:
        store.close()

    assert not inputs.target_mask[-1].any()
    assert not inputs.raw_target_mask[-1].any()
    assert np.all(inputs.residual_midrank_targets[~inputs.target_mask] == 0.0)
    assert np.all(inputs.raw_midrank_targets[~inputs.raw_target_mask] == 0.0)
    assert np.all(inputs.raw_log_returns[~inputs.raw_target_mask] == 0.0)


def test_git_identity_refuses_untracked_files(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(("1" * 40 + "\n", "?? untracked.py\n"))
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        return SimpleNamespace(stdout=next(responses))

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="clean"):
        pipeline._git_identity()
    assert calls[1][-1] == "--untracked-files=all"
