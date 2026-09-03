from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

from brazil_rv.v2.artifacts import sha256_file
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.score import score_checkpoint_artifact
from brazil_rv.v2.store import write_store
from brazil_rv.v2.train import build_checkpoint_input_contract


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
    tmp_path, *, slow_names: tuple[str, str] = ("slow_0", "slow_1")
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    day_count, name_count = 25, 4
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(day_count)]
    generator = np.random.default_rng(17)
    slow = generator.standard_normal((day_count, name_count, 2)).astype(np.float32)
    active = np.ones((day_count, name_count), dtype=np.bool_)
    active[21, 2] = False
    store = write_store(
        tmp_path / "store",
        dates=dates,
        isins=[f"BRTEST{index:02d}NOR1" for index in range(name_count)],
        arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "active": active,
        },
        feature_names={"slow": list(slow_names), "intraday": []},
    )
    dataset = V2DailyDataset(
        store,
        [20, 21, 22],
        stage="evaluation",
        lookback=20,
        purpose="evaluation",
    )
    config = ModelConfig(
        slow_feature_count=2,
        slow_lookback=20,
        dropout=0.1,
        compile_forward=False,
    )
    torch.manual_seed(29)
    model = DailyMultiHorizonModel(config)
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=_omit_absent_fast
    )
    input_contract = build_checkpoint_input_contract(config, loader, loader)
    checkpoint = tmp_path / "raw_patience.pt"
    torch.save(
        {
            "schema": "V2_RAW_PATIENCE",
            "stage": "F",
            "seed": 29,
            "fold": "F1",
            "model_state_dict": model.state_dict(),
            "input_contract": input_contract,
        },
        checkpoint,
    )
    return dataset, config, checkpoint, active[20:23]


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
    )
    scores = np.load(first.scores_path, allow_pickle=False)
    repeated = np.load(second.scores_path, allow_pickle=False)
    score_mask = np.load(first.score_mask_path, allow_pickle=False)
    assert scores.shape == (3, 4, 5)
    assert scores.dtype == np.float32
    assert np.array_equal(scores, repeated)
    assert first.scores_path.read_bytes() == second.scores_path.read_bytes()
    assert np.array_equal(score_mask, np.repeat(active[..., None], 5, axis=-1))
    assert np.all(scores[~score_mask] == 0.0)

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint"]["kind"] == "V2_RAW_PATIENCE"
    assert manifest["checkpoint"]["seed"] == 29
    assert manifest["access_ledger"]["purpose"] == "evaluation"
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
    payload["schema"] = "V2_FINAL_EMA_0995"
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
    assert manifest["checkpoint"]["kind"] == "V2_FINAL_EMA_0995"

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
            loader=DataLoader(
                dataset, batch_size=2, collate_fn=_omit_absent_fast
            ),
            output_dir=tmp_path / "bad_hash",
            expected_checkpoint_sha256="0" * 64,
            device=torch.device("cpu"),
        )
    assert not (tmp_path / "bad_hash").exists()


def test_scoring_rejects_swapped_ordered_features_before_output(tmp_path) -> None:
    _, config, checkpoint, _ = _scoring_fixture(tmp_path / "original")
    swapped, _, _, _ = _scoring_fixture(
        tmp_path / "swapped", slow_names=("slow_1", "slow_0")
    )
    output = tmp_path / "swapped_scores"
    with pytest.raises(ValueError, match="dataset differs"):
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
    source = DailyMultiHorizonModel(ModelConfig(slow_feature_count=2))
    initializer = tmp_path / "v1_initializer.pt"
    torch.save({"model_state_dict": source.fast_encoder.state_dict()}, initializer)
    initializer_sha256 = hashlib.sha256(initializer.read_bytes()).hexdigest()
    config = ModelConfig(
        slow_feature_count=2,
        slow_lookback=20,
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
            "schema": "V2_RAW_PATIENCE",
            "stage": "F",
            "seed": 29,
            "fold": "F1",
            "model_state_dict": model.state_dict(),
            "input_contract": build_checkpoint_input_contract(
                config, loader, loader
            ),
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
