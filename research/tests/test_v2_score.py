from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.contract import (
    DECISION_FEATURE_CONTRACT,
    FEATURE_AGE_CONTRACT,
    INTRADAY_DAILY_FEATURES,
)
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.round5_magnitude import FEATURE_NAMES, FitClip
from brazil_rv.v2.score import score_checkpoint_artifact
from v2_store_fixtures import write_fixture_store as write_store
from brazil_rv.v2.train import (
    _canonical_payload_sha256,
    build_checkpoint_input_contract,
    load_pretrain_handoff,
    model_config_contract,
    stage_p_model_config,
)


def _omit_absent_fast(rows):
    batch = default_collate(rows)
    if not torch.any(batch["fast_present"].bool()):
        for name in (
            "fast_patches",
            "fast_patch_mask",
            "fast_state_position",
            "v1_equity_slow",
        ):
            batch.pop(name, None)
    return batch


def _scoring_fixture(
    tmp_path,
    *,
    slow_names: tuple[str, str] = ("slow_0", "slow_1"),
    include_official: bool = False,
    include_magnitudes: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    name_count = 4
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(25)]
    if include_official:
        dates.extend((date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)))
    day_count = len(dates)
    generator = np.random.default_rng(17)
    slow = generator.standard_normal((day_count, name_count, 2)).astype(np.float32)
    intraday = generator.standard_normal(
        (day_count, name_count, len(INTRADAY_DAILY_FEATURES))
    ).astype(np.float32)
    active = np.ones((day_count, name_count), dtype=np.bool_)
    active[21, 2] = False
    magnitude_arrays = {}
    if include_magnitudes:
        values = np.arange(day_count * name_count * 4, dtype=np.float32).reshape(
            day_count, name_count, 4
        )
        valid = np.ones_like(values, bool)
        valid[20:22, :, 3] = False
        valid[20, 0, 0] = False
        values[20, 0, 0] = -1e9
        values[21, 2] = 1e9
        values[22:] *= 1e6
        magnitude_arrays = {
            "sidecar_magnitudes_values": values,
            "sidecar_magnitudes_valid": valid,
            "sidecar_magnitudes_age_sessions": np.where(valid, 1, -1).astype(
                np.float32
            ),
        }
    store = write_store(
        tmp_path / "store",
        dates=dates,
        isins=[f"BRTEST{index:02d}NOR1" for index in range(name_count)],
        arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "slow_age_sessions": np.zeros_like(slow, dtype=np.float32),
            "slow_timestep_valid": np.ones((day_count, name_count), dtype=np.bool_),
            "intraday_values": intraday,
            "intraday_valid": np.ones_like(intraday, dtype=np.bool_),
            "intraday_age_sessions": np.zeros_like(intraday, dtype=np.float32),
            "active": active,
            **magnitude_arrays,
        },
        feature_names={
            "slow": list(slow_names),
            "intraday": list(INTRADAY_DAILY_FEATURES),
            **(
                {"sidecar_magnitudes": list(FEATURE_NAMES)}
                if include_magnitudes
                else {}
            ),
        },
        metadata={
            "feature_age_contract": dict(FEATURE_AGE_CONTRACT),
            "slow_entry_alignment": dict(DECISION_FEATURE_CONTRACT),
        },
    )
    score_indices = [22, 23, 24] if include_magnitudes else [20, 21, 22]
    dataset = V2DailyDataset(
        store,
        score_indices,
        stage="evaluation",
        lookback=20,
        purpose="evaluation",
        enabled_sidecars=("magnitudes",) if include_magnitudes else (),
    )
    config = ModelConfig(
        slow_feature_count=2,
        slow_lookback=20,
        dropout=0.1,
        compile_forward=False,
        sidecar_feature_counts=(("magnitudes", 4),) if include_magnitudes else (),
    )
    torch.manual_seed(29)
    model = DailyMultiHorizonModel(config)
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=_omit_absent_fast
    )
    training_loader = loader
    if include_magnitudes:
        training_loader = DataLoader(
            V2DailyDataset(
                store,
                [20, 21],
                stage="finetune",
                lookback=20,
                purpose="training",
                enabled_sidecars=("magnitudes",),
            ),
            batch_size=2,
        )
    input_contract = build_checkpoint_input_contract(config, training_loader, loader)
    checkpoint = tmp_path / "raw_patience.pt"
    torch.save(
        {
            "schema": "BRAZIL_RV_V2_RAW_PATIENCE_V2",
            "stage": "F",
            "seed": 29,
            "fold": "F1",
            "model_state_dict": model.state_dict(),
            "input_contract": input_contract,
            "transfer_chronology_clean": True,
            "feature_schema_sha256": dataset.store.manifest["feature_schema_sha256"],
            "action_terms_source": dataset.store.manifest["metadata"][
                "action_terms_source"
            ],
            "schedule_source": dataset.store.manifest["metadata"]["schedule_source"],
        },
        checkpoint,
    )
    return dataset, config, checkpoint, active[score_indices]


def test_magnitude_scoring_restores_training_bounds_without_refitting(
    tmp_path, monkeypatch
):
    dataset, config, checkpoint, _ = _scoring_fixture(tmp_path, include_magnitudes=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    inputs = payload["input_contract"]
    frozen = inputs["training"]["features"]["magnitude_clip"]
    assert frozen == inputs["selection"]["features"]["magnitude_clip"]
    assert frozen["fit_date_indices"] == [20, 21]
    assert frozen["lower"][0] > 0 and frozen["upper"][0] < 1000
    assert frozen["lower"][3] is None and frozen["upper"][3] is None
    dataset.magnitude_clip = None
    with pytest.raises(ValueError, match="frozen fit clipping"):
        dataset[0]

    def no_fit(*args, **kwargs):
        raise AssertionError("scoring must never estimate clipping bounds")

    monkeypatch.setattr(FitClip, "fit", no_fit)
    artifact = score_checkpoint_artifact(
        checkpoint=checkpoint,
        model_config=config,
        loader=DataLoader(dataset, batch_size=2, collate_fn=_omit_absent_fast),
        output_dir=tmp_path / "magnitude_scores",
        device=torch.device("cpu"),
    )
    assert dataset.magnitude_clip.payload() == frozen
    sample = dataset[0]
    assert (sample["sidecar_magnitudes_values"][:, 0] <= frozen["upper"][0]).all()
    assert (sample["sidecar_magnitudes_values"][:, 3] > 1e6).all()
    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf8"))
    assert manifest["scoring_input"]["features"]["magnitude_clip"] == frozen


def test_magnitude_handoff_allows_stage_fit_bounds_but_keeps_feature_identity(tmp_path):
    _, config, checkpoint, _ = _scoring_fixture(tmp_path, include_magnitudes=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    fine_contract = deepcopy(payload["input_contract"])
    pretrain_config = stage_p_model_config(config)
    payload["stage"] = "P"
    payload["model_state_dict"] = DailyMultiHorizonModel(pretrain_config).state_dict()
    contract = payload["input_contract"]
    contract["model_config"] = model_config_contract(pretrain_config)
    for subset in ("training", "selection"):
        clip = contract[subset]["features"]["magnitude_clip"]
        clip["fit_date_indices"] = [0, 1]
        clip["lower"] = [x / 2 if x is not None else None for x in clip["lower"]]
        clip["upper"] = [x / 2 if x is not None else None for x in clip["upper"]]
    contract.pop("sha256")
    contract["sha256"] = _canonical_payload_sha256(contract)
    pretrain_path = tmp_path / "magnitude_pretrain.pt"
    torch.save(payload, pretrain_path)
    transferred = load_pretrain_handoff(
        DailyMultiHorizonModel(config),
        pretrain_path,
        expected_sha256=sha256_file(pretrain_path),
        fine_tune_input_contract=fine_contract,
    )
    assert "sidecar_projections.magnitudes" in transferred
    fine_contract["training"]["features"]["ordered_slow_names"] = ["different"]
    with pytest.raises(ValueError, match="store/feature identities differ"):
        load_pretrain_handoff(
            DailyMultiHorizonModel(config),
            pretrain_path,
            expected_sha256=sha256_file(pretrain_path),
            fine_tune_input_contract=fine_contract,
        )


def test_scoring_is_repeat_bit_identical_and_provenance_bound(tmp_path) -> None:
    dataset, config, checkpoint, active = _scoring_fixture(tmp_path)
    first = score_checkpoint_artifact(
        checkpoint=checkpoint,
        model_config=config,
        loader=DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=_omit_absent_fast,
        ),
        output_dir=tmp_path / "scores_first",
        device=torch.device("cpu"),
    )
    second = score_checkpoint_artifact(
        checkpoint=checkpoint,
        model_config=config,
        loader=DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=_omit_absent_fast,
        ),
        output_dir=tmp_path / "scores_second",
        expected_checkpoint_sha256=first.checkpoint_sha256,
        device=torch.device("cpu"),
        record_branch_diagnostics=True,
    )
    scores = np.load(first.scores_path, allow_pickle=False)
    repeated = np.load(second.scores_path, allow_pickle=False)
    score_mask = np.load(first.score_mask_path, allow_pickle=False)
    assert scores.shape == (3, 4, 5)
    assert scores.dtype == np.float32
    assert np.array_equal(scores, repeated)
    assert first.scores_path.read_bytes() == second.scores_path.read_bytes()
    second_manifest = json.loads(second.manifest_path.read_text())
    gates = second_manifest["gate_diagnostics"]
    assert gates["sha256"] == sha256_file(second.root / gates["path"])
    gate_report = json.loads((second.root / gates["path"]).read_text())
    assert sum(gate_report["effective_fast_present_counts"]) == int(active.sum())
    assert np.array_equal(score_mask, np.repeat(active[..., None], 5, axis=-1))
    assert np.all(scores[~score_mask] == 0.0)

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert "magnitude_clip" not in manifest["scoring_input"]["features"]
    assert manifest["checkpoint"]["kind"] == "BRAZIL_RV_V2_RAW_PATIENCE_V2"
    assert manifest["checkpoint"]["seed"] == 29
    assert manifest["access_ledger"]["purpose"] == "evaluation"
    assert manifest["fast_initialization_provenance"]["mode"] == "fresh"
    assert manifest["fast_initialization_provenance"]["contaminated"] is False
    assert manifest["transfer_chronology_clean"] is True
    assert manifest["scoring_input"]["dates"]["first_date"] == "2024-01-21"
    assert len(manifest["scoring_input_sha256"]) == 64
    assert manifest["official_validation_accessed"] is False
    assert manifest["test_accessed"] is False
    assert manifest["store"]["manifest_sha256"] == sha256_file(
        dataset.store.root / "manifest.json"
    )
    assert manifest["artifacts"]["scores.npy"]["sha256"] == sha256_file(
        first.scores_path
    )
    assert first.manifest_sha256 == sha256_file(first.manifest_path)


def test_scoring_accepts_ema_and_rejects_nonchronological_loader(tmp_path) -> None:
    dataset, config, raw_checkpoint, _ = _scoring_fixture(tmp_path)
    payload = torch.load(raw_checkpoint, map_location="cpu", weights_only=False)
    payload["schema"] = "BRAZIL_RV_V2_FINAL_EMA_0995_V2"
    ema_checkpoint = tmp_path / "final_ema.pt"
    torch.save(payload, ema_checkpoint)
    result = score_checkpoint_artifact(
        checkpoint=ema_checkpoint,
        model_config=config,
        loader=DataLoader(
            dataset,
            batch_size=3,
            shuffle=False,
            collate_fn=_omit_absent_fast,
        ),
        output_dir=tmp_path / "ema_scores",
        device=torch.device("cpu"),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint"]["kind"] == "BRAZIL_RV_V2_FINAL_EMA_0995_V2"

    reverse_loader = DataLoader(
        dataset,
        batch_size=3,
        sampler=[2, 1, 0],
        collate_fn=_omit_absent_fast,
    )
    with pytest.raises(ValueError, match="chronological order"):
        score_checkpoint_artifact(
            checkpoint=raw_checkpoint,
            model_config=config,
            loader=reverse_loader,
            output_dir=tmp_path / "invalid_order",
            device=torch.device("cpu"),
        )
    assert not (tmp_path / "invalid_order").exists()


def test_scoring_rejects_hash_mismatch_before_creating_artifact(tmp_path) -> None:
    dataset, config, checkpoint, _ = _scoring_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        score_checkpoint_artifact(
            checkpoint=checkpoint,
            model_config=config,
            loader=DataLoader(dataset, batch_size=2, collate_fn=_omit_absent_fast),
            output_dir=tmp_path / "bad_hash",
            expected_checkpoint_sha256="0" * 64,
            device=torch.device("cpu"),
        )
    assert not (tmp_path / "bad_hash").exists()


def test_scoring_rejects_non_evaluation_access_ledger(tmp_path) -> None:
    _, config, checkpoint, _ = _scoring_fixture(tmp_path / "fixture")
    selected = V2DailyDataset(
        tmp_path / "fixture" / "store",
        [20, 21, 22],
        stage="evaluation",
        lookback=20,
        purpose="selection",
    )
    output = tmp_path / "selection_scores"
    with pytest.raises(ValueError, match="evaluation-purpose"):
        score_checkpoint_artifact(
            checkpoint=checkpoint,
            model_config=config,
            loader=DataLoader(selected, batch_size=2, collate_fn=_omit_absent_fast),
            output_dir=output,
            device=torch.device("cpu"),
        )
    assert not output.exists()


def test_scoring_rejects_swapped_ordered_features_before_output(tmp_path) -> None:
    _, config, checkpoint, _ = _scoring_fixture(tmp_path / "original")
    swapped, _, _, _ = _scoring_fixture(
        tmp_path / "swapped", slow_names=("slow_1", "slow_0")
    )
    output = tmp_path / "swapped_scores"
    with pytest.raises(ValueError, match="feature schema differs"):
        score_checkpoint_artifact(
            checkpoint=checkpoint,
            model_config=config,
            loader=DataLoader(
                swapped,
                batch_size=2,
                shuffle=False,
                collate_fn=_omit_absent_fast,
            ),
            output_dir=output,
            device=torch.device("cpu"),
        )
    assert not output.exists()


def test_scoring_restores_checkpoint_after_initializer_is_deleted(tmp_path) -> None:
    dataset, _, _, _ = _scoring_fixture(tmp_path / "fixture")
    legacy = {
        "fast_encoder_mode": "legacy_v1_contaminated",
        "allow_contaminated_v1_initialization": True,
    }
    source = DailyMultiHorizonModel(ModelConfig(slow_feature_count=2, **legacy))
    initializer = tmp_path / "v1_initializer.pt"
    torch.save({"model_state_dict": source.fast_encoder.state_dict()}, initializer)
    initializer_sha256 = hashlib.sha256(initializer.read_bytes()).hexdigest()
    config = ModelConfig(
        slow_feature_count=2,
        slow_lookback=20,
        **legacy,
        fast_pretrained=True,
        fast_pretrained_checkpoint=initializer,
        fast_pretrained_sha256=initializer_sha256,
        compile_forward=False,
    )
    model = DailyMultiHorizonModel(config)
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=_omit_absent_fast
    )
    checkpoint = tmp_path / "initialized_stage.pt"
    torch.save(
        {
            "schema": "BRAZIL_RV_V2_RAW_PATIENCE_V2",
            "stage": "F",
            "seed": 29,
            "fold": "F1",
            "model_state_dict": model.state_dict(),
            "input_contract": build_checkpoint_input_contract(config, loader, loader),
            "transfer_chronology_clean": False,
            "feature_schema_sha256": dataset.store.manifest["feature_schema_sha256"],
            "action_terms_source": dataset.store.manifest["metadata"][
                "action_terms_source"
            ],
            "schedule_source": dataset.store.manifest["metadata"]["schedule_source"],
        },
        checkpoint,
    )
    initializer.unlink()
    result = score_checkpoint_artifact(
        checkpoint=checkpoint,
        model_config=config,
        loader=loader,
        output_dir=tmp_path / "restored_scores",
        device=torch.device("cpu"),
    )
    assert result.scores_path.is_file()


def test_official_scoring_rejects_unclean_transfer_before_output(tmp_path) -> None:
    development, config, checkpoint, _ = _scoring_fixture(
        tmp_path / "fixture", include_official=True
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["transfer_chronology_clean"] = False
    unclean = tmp_path / "unclean.pt"
    torch.save(payload, unclean)
    registration_root = tmp_path / "research" / "preregistrations"
    registration_root.mkdir(parents=True)
    registration = registration_root / "official_read.md"
    registration.write_text("frozen official read\n", encoding="utf-8")
    official = V2DailyDataset(
        development.store.root,
        [25, 26, 27],
        stage="evaluation",
        lookback=20,
        purpose="evaluation",
        registration_path=registration,
        preregistration_root=registration_root,
    )
    output = tmp_path / "unclean_official_scores"
    with pytest.raises(PermissionError, match="chronology-contaminated"):
        score_checkpoint_artifact(
            checkpoint=unclean,
            model_config=config,
            loader=DataLoader(
                official,
                batch_size=2,
                shuffle=False,
                collate_fn=_omit_absent_fast,
            ),
            output_dir=output,
            device=torch.device("cpu"),
        )
    assert not output.exists()


def test_registered_official_dates_are_independently_scored_and_recorded(
    tmp_path,
) -> None:
    development, config, checkpoint, _ = _scoring_fixture(
        tmp_path / "fixture", include_official=True
    )
    registration_root = tmp_path / "research" / "preregistrations"
    registration_root.mkdir(parents=True)
    registration = registration_root / "official_read.md"
    registration.write_text("frozen official read\n", encoding="utf-8")
    official = V2DailyDataset(
        development.store.root,
        [25, 26, 27],
        stage="evaluation",
        lookback=20,
        purpose="evaluation",
        registration_path=registration,
        preregistration_root=registration_root,
    )

    result = score_checkpoint_artifact(
        checkpoint=checkpoint,
        model_config=config,
        loader=DataLoader(
            official,
            batch_size=2,
            shuffle=False,
            collate_fn=_omit_absent_fast,
        ),
        output_dir=tmp_path / "official_scores",
        device=torch.device("cpu"),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["official_validation_accessed"] is True
    assert manifest["test_accessed"] is False
    assert manifest["access_ledger"]["registration"] == {
        "path": str(registration.resolve()),
        "sha256": sha256_file(registration),
    }
    assert manifest["scoring_input"]["dates"]["first_date"] == "2025-01-02"
    assert manifest["scoring_input"]["dates"]["last_date"] == "2025-01-06"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert (
        manifest["scoring_input"]["dates"]
        != checkpoint_payload["input_contract"]["selection"]["dates"]
    )
