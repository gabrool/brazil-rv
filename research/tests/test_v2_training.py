from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from brazil_rv.modeling.engine import soft_spearman_loss
from brazil_rv.modeling.trajectory import ModelEMA
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.contract import PRETRAIN_END, STORE_START
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.losses import (
    multi_horizon_loss_components,
    score_persistence_penalty,
)
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import (
    DatePairBatchSampler,
    PatienceTracker,
    build_optimizer,
    compile_forward,
    load_pretrain_handoff,
    load_stage_checkpoint,
    pretrain_internal_split,
    rank_average_ensemble,
    reshape_date_pair_batch,
    sam_step,
    stitch_block_parity_predictions,
    train_stage,
    _input_static_identity,
    _model_input_segments,
    _require_production_pair_sampler,
    _validate_tracked_stage_inputs,
)
from brazil_rv.v2.store import write_store


def test_external_location_is_not_part_of_model_input_identity() -> None:
    first = {
        "store": {
            "manifest_sha256": "a" * 64,
            "external_artifact_resolutions": [
                {
                    "recorded_path": r"D:\quant-data\sealed.npy",
                    "resolved_path": r"D:\quant-data\sealed.npy",
                    "bytes": 123,
                    "sha256": "b" * 64,
                    "override_file": None,
                }
            ],
        },
        "features": {"ordered": ["x"]},
        "lookback_sessions": 60,
    }
    relocated = json.loads(json.dumps(first))
    resolution = relocated["store"]["external_artifact_resolutions"][0]
    resolution["resolved_path"] = "/lambda/nfs/quant-data/sealed.npy"
    resolution["override_file"] = "/run-config/data_roots.json"

    assert _input_static_identity(first) == _input_static_identity(relocated)
    resolution["sha256"] = "c" * 64
    assert _input_static_identity(first) != _input_static_identity(relocated)


def test_multihead_objective_averages_each_horizon_separately() -> None:
    torch.manual_seed(3)
    scores = torch.randn(2, 5, 6)
    targets = torch.randn(2, 5, 6)
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[..., 5] = False
    components = multi_horizon_loss_components(scores, targets, mask)
    expected = torch.stack(
        [
            soft_spearman_loss(
                scores[..., head : head + 1],
                targets[..., head : head + 1],
                mask[..., head : head + 1],
            )
            for head in range(5)
        ]
    ).mean()
    assert torch.allclose(components["horizon"], expected)
    assert components["to_close"] == 0
    assert torch.allclose(components["total"], expected)


def test_persistence_uses_population_zscores_and_score_mask() -> None:
    scores = torch.zeros(1, 2, 4, 6)
    scores[0, 0, :3, 0] = torch.tensor([-1.0, 0.0, 1.0])
    scores[0, 1, :3, 0] = torch.tensor([1.0, 0.0, -1.0])
    mask = torch.zeros(1, 2, 4, 5, dtype=torch.bool)
    mask[..., 0] = torch.tensor([True, True, True, False])
    penalty = score_persistence_penalty(scores, mask)
    assert penalty.item() == pytest.approx(4.0, abs=1e-4)
    scores[0, :, 3, 0] = 1e9
    assert score_persistence_penalty(scores, mask).item() == pytest.approx(
        penalty.item()
    )


def test_persistence_is_forced_to_float32_outside_autocast() -> None:
    scores = torch.randn(1, 2, 4, 6, dtype=torch.bfloat16)
    mask = torch.ones(1, 2, 4, 5, dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        penalty = score_persistence_penalty(scores, mask)
    assert penalty.dtype == torch.float32


def test_date_pair_sampler_keeps_adjacent_rows_and_is_deterministic() -> None:
    sampler = DatePairBatchSampler(range(7), pairs_per_batch=2, seed=11)
    first = list(sampler)
    second = list(sampler)
    assert first == second
    assert len(first) == 3
    for batch in first:
        for offset in range(0, len(batch), 2):
            assert batch[offset + 1] == batch[offset] + 1
    tensor = torch.arange(12).reshape(6, 2)
    assert reshape_date_pair_batch(tensor).shape == (3, 2, 2)


def test_production_pair_sampler_emits_only_exact_eight_pair_batches() -> None:
    sampler = DatePairBatchSampler(
        range(18), pairs_per_batch=8, seed=11, drop_last=True
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert all(len(batch) == 16 for batch in batches)
    assert all(
        batch[offset + 1] == batch[offset] + 1
        for batch in batches
        for offset in range(0, len(batch), 2)
    )
    _require_production_pair_sampler(SimpleNamespace(batch_sampler=sampler))
    with pytest.raises(ValueError, match="exactly 8 pairs and drop_last=True"):
        _require_production_pair_sampler(
            SimpleNamespace(
                batch_sampler=DatePairBatchSampler(
                    range(18), pairs_per_batch=7, drop_last=True
                )
            )
        )
    with pytest.raises(ValueError, match="exactly 8 pairs and drop_last=True"):
        _require_production_pair_sampler(
            SimpleNamespace(
                batch_sampler=DatePairBatchSampler(
                    range(18), pairs_per_batch=8, drop_last=False
                )
            )
        )


def test_time_decay_sampler_is_epoch_deterministic() -> None:
    left = DatePairBatchSampler(
        range(20), seed=47, time_decay_half_life=756.0, pairs_per_batch=4
    )
    right = DatePairBatchSampler(
        range(20), seed=47, time_decay_half_life=756.0, pairs_per_batch=4
    )
    left.set_epoch(2)
    right.set_epoch(2)
    assert list(left) == list(right)


def test_stage_j_requires_the_frozen_time_decay_and_other_stages_reject_it(
    tmp_path,
) -> None:
    base = ModelConfig(slow_feature_count=32, compile_forward=False)
    with pytest.raises(
        ValueError, match="stage J requires time_decay_half_life_sessions=756.0"
    ):
        train_stage(
            stage="J",
            seed=11,
            fold="F1",
            train_loader=[],
            selection_loader=[],
            output_dir=tmp_path / "j_without_decay",
            model_config=base,
            selection_parity=0,
            maximum_epochs=1,
        )
    with pytest.raises(
        ValueError, match="stage F requires time_decay_half_life_sessions=None"
    ):
        train_stage(
            stage="F",
            seed=11,
            fold="F1",
            train_loader=[],
            selection_loader=[],
            output_dir=tmp_path / "f_with_decay",
            model_config=ModelConfig(
                slow_feature_count=32,
                compile_forward=False,
                time_decay_half_life_sessions=756.0,
            ),
            selection_parity=0,
            maximum_epochs=1,
        )


def test_date_pair_sampler_never_crosses_a_window_gap() -> None:
    sampler = DatePairBatchSampler(
        (4, 5, 9, 10), session_indices=(20, 21, 40, 41), pairs_per_batch=8
    )
    assert list(sampler) == [[0, 1, 2, 3]]
    assert list(DatePairBatchSampler((20, 21, 40, 41), pairs_per_batch=8)) == [
        [0, 1, 2, 3]
    ]


def test_pretrain_internal_holdout_is_last_ten_percent_after_embargo() -> None:
    fit, embargo, selection = pretrain_internal_split(np.arange(1_000))
    assert fit.tolist() == list(range(830))
    assert embargo.tolist() == list(range(830, 900))
    assert selection.tolist() == list(range(900, 1_000))


def test_block_parity_stitch_preserves_full_contiguous_axis() -> None:
    selected_on_even = np.full((13, 2), 10.0)
    selected_on_odd = np.full((13, 2), 20.0)
    stitched = stitch_block_parity_predictions(selected_on_even, selected_on_odd)
    assert np.array_equal(stitched[:5], selected_on_odd[:5])
    assert np.array_equal(stitched[5:10], selected_on_even[5:10])
    assert np.array_equal(stitched[10:], selected_on_odd[10:])


def test_rank_average_ensemble_is_per_group_and_tie_aware() -> None:
    left = np.array([[[3.0], [1.0], [1.0], [9.0]]])
    right = np.array([[[0.0], [4.0], [2.0], [8.0]]])
    mask = np.array([[True, True, True, False]])
    actual = rank_average_ensemble((left, right), mask)
    assert actual[0, :, 0].tolist() == [1.0, 1.25, 0.75, 0.0]


def test_optimizer_routes_pretrained_parameters_at_point_three_lr() -> None:
    model = DailyMultiHorizonModel(ModelConfig(slow_feature_count=32))
    pretrained = frozenset(
        f"fast_encoder.{name}" for name, _ in model.fast_encoder.named_parameters()
    )
    optimizer = build_optimizer(model, pretrained_parameter_names=pretrained)
    grouped = {
        (group["pretrained"], group["weight_decay"]): group["lr"]
        for group in optimizer.param_groups
    }
    assert grouped[(True, 0.01)] == pytest.approx(9e-5)
    assert grouped[(False, 0.01)] == pytest.approx(3e-4)
    decay_ids = {
        id(parameter)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.01
        for parameter in group["params"]
    }
    assert id(model.slow_input_norm.weight) in decay_ids
    assert id(model.absent_state) not in decay_ids


def test_actual_patience_restores_best_raw_state() -> None:
    model = nn.Linear(1, 1, bias=False)
    tracker = PatienceTracker(patience=3, maximum_epochs=20)
    stopped = False
    for epoch, score in enumerate((1.0, 0.9, 0.8, 0.7), start=1):
        model.weight.data.fill_(epoch)
        stopped = tracker.update(epoch, score, model)
    assert stopped
    assert tracker.selected_epoch == 1
    assert tracker.stopped_epoch == 4
    tracker.restore(model)
    assert model.weight.item() == 1.0


def test_sam_updates_model_and_ema() -> None:
    torch.manual_seed(13)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ema = ModelEMA(model, 0.995)
    inputs = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    targets = torch.tensor([[1.0], [-0.5]])
    before = model.weight.detach().clone()

    def closure() -> torch.Tensor:
        return (model(inputs) - targets).square().mean()

    result = sam_step(model, optimizer, closure, ema=ema)
    assert np.isfinite(result.first_gradient_norm)
    assert not torch.equal(before, model.weight)
    assert not torch.equal(ema.shadow["weight"], before)


def test_fullgraph_compile_captures_gru_forward() -> None:
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    model = DailyMultiHorizonModel(config).eval()
    compiled = compile_forward(model, backend="eager", mode=None)
    scores = compiled(
        torch.randn(2, 3, 20, 32),
        torch.ones(2, 3, 20, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool),
    )
    assert scores.shape == (2, 3, 6)


def _tracked_pretrain_loaders(tmp_path):
    calendar = np.arange(
        np.datetime64(STORE_START),
        np.datetime64(PRETRAIN_END) + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    dates = calendar[np.is_busday(calendar)]
    fit, _, selection = pretrain_internal_split(np.arange(dates.size))
    generator = np.random.default_rng(19)
    name_count = 4
    slow = generator.standard_normal((dates.size, name_count, 32)).astype(
        np.float32
    )
    targets = generator.standard_normal((dates.size, name_count, 5)).astype(
        np.float32
    )
    store = write_store(
        tmp_path / "tracked_store",
        dates=dates,
        isins=[f"BRTEST{index:02d}NOR1" for index in range(name_count)],
        arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "active": np.ones((dates.size, name_count), dtype=np.bool_),
            "target_primary": targets,
            "target_valid": np.ones_like(targets, dtype=np.bool_),
        },
        feature_names={
            "slow": [f"slow_{index}" for index in range(32)],
            "intraday": [],
        },
    )
    train_dataset = V2DailyDataset(
        store,
        fit[-9:],
        stage="pretrain",
        lookback=20,
        purpose="training",
    )
    selection_dataset = V2DailyDataset(
        store,
        selection[:6],
        stage="pretrain",
        lookback=20,
        purpose="selection",
    )

    def train_loader():
        return DataLoader(
            train_dataset,
            batch_sampler=DatePairBatchSampler(
                train_dataset.date_indices,
                pairs_per_batch=8,
                seed=19,
                drop_last=True,
            ),
            num_workers=0,
        )

    selection_loader = DataLoader(
        selection_dataset,
        batch_size=len(selection_dataset),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, selection_loader


def test_stage_runner_archives_patience_ema_and_handoff(tmp_path) -> None:
    config = ModelConfig(
        slow_feature_count=32,
        slow_lookback=20,
        compile_forward=False,
    )
    train_loader_factory, selection_loader = _tracked_pretrain_loaders(tmp_path)
    train_loader = train_loader_factory()
    output_dir = tmp_path / "stage_p"
    output_dir.mkdir()
    (output_dir / "launcher.stdout.log").write_text("", encoding="utf-8")
    (output_dir / "launcher.stderr.log").write_text("", encoding="utf-8")
    result = train_stage(
        stage="P",
        seed=29,
        fold="pretrain_internal",
        train_loader=train_loader,
        selection_loader=selection_loader,
        output_dir=output_dir,
        model_config=config,
        maximum_epochs=1,
        patience=1,
        device=torch.device("cpu"),
    )
    assert result.raw_patience_checkpoint.is_file()
    assert result.final_ema_checkpoint.is_file()
    assert result.history_path.is_file()
    assert result.manifest_path.is_file()
    assert train_loader.batch_sampler.epoch == 0
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 29
    assert manifest["fold"] == "pretrain_internal"
    assert manifest["official_validation_accessed"] is False
    assert manifest["test_accessed"] is False
    assert "allow_untracked_test_loaders" not in manifest
    assert manifest["checkpoint_input_contract"]["training"] is not None
    assert manifest["checkpoint_input_contract"]["selection"] is not None

    repeat_loader = train_loader_factory()
    repeated = train_stage(
        stage="P",
        seed=29,
        fold="pretrain_internal",
        train_loader=repeat_loader,
        selection_loader=selection_loader,
        output_dir=tmp_path / "stage_p_repeat",
        model_config=config,
        maximum_epochs=1,
        patience=1,
        device=torch.device("cpu"),
    )
    assert result.history_path.read_bytes() == repeated.history_path.read_bytes()
    first_state = torch.load(
        result.raw_patience_checkpoint, map_location="cpu", weights_only=False
    )["model_state_dict"]
    repeated_state = torch.load(
        repeated.raw_patience_checkpoint, map_location="cpu", weights_only=False
    )["model_state_dict"]
    assert first_state.keys() == repeated_state.keys()
    assert all(
        torch.equal(first_state[name], repeated_state[name]) for name in first_state
    )

    loaded = DailyMultiHorizonModel(config)
    checkpoint_sha256 = load_stage_checkpoint(loaded, result.raw_patience_checkpoint)
    assert checkpoint_sha256 == manifest["artifacts"]["raw_patience.pt"]
    assert all(
        torch.equal(loaded.state_dict()[name], first_state[name])
        for name in first_state
    )

    fine_tune = DailyMultiHorizonModel(config)
    fast_before = {
        name: value.clone()
        for name, value in fine_tune.fast_encoder.state_dict().items()
    }
    initialized = load_pretrain_handoff(
        fine_tune,
        result.raw_patience_checkpoint,
        expected_sha256=manifest["artifacts"]["raw_patience.pt"],
        expected_seed=29,
    )
    assert initialized
    assert all(
        torch.equal(value, fast_before[name])
        for name, value in fine_tune.fast_encoder.state_dict().items()
    )

    base_payload = torch.load(
        result.raw_patience_checkpoint, map_location="cpu", weights_only=False
    )
    for index, (field, value) in enumerate(
        (
            ("schema", "V2_FINAL_EMA_0995"),
            ("stage", "F"),
            ("seed", 47),
            ("fold", ""),
        )
    ):
        invalid = dict(base_payload)
        invalid[field] = value
        path = tmp_path / f"invalid_handoff_{index}.pt"
        torch.save(invalid, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(
            ValueError, match="handoff schema, stage, optional seed, or fold"
        ):
            load_pretrain_handoff(
                DailyMultiHorizonModel(config),
                path,
                expected_sha256=digest,
                expected_seed=29,
            )
    invalid_contract = dict(base_payload)
    invalid_contract["input_contract"] = dict(base_payload["input_contract"])
    invalid_contract["input_contract"]["schema"] = "tampered"
    invalid_contract_path = tmp_path / "invalid_handoff_contract.pt"
    torch.save(invalid_contract, invalid_contract_path)
    with pytest.raises(ValueError, match="input contract hash"):
        load_pretrain_handoff(
            DailyMultiHorizonModel(config),
            invalid_contract_path,
            expected_sha256=hashlib.sha256(
                invalid_contract_path.read_bytes()
            ).hexdigest(),
            expected_seed=29,
        )


def test_stage_runner_rejects_untracked_production_loaders(tmp_path) -> None:
    with pytest.raises(ValueError, match="authorized access ledgers"):
        train_stage(
            stage="P",
            seed=29,
            fold="pretrain_internal",
            train_loader=[],
            selection_loader=[],
            output_dir=tmp_path / "untracked",
            model_config=ModelConfig(
                slow_feature_count=32,
                slow_lookback=20,
                compile_forward=False,
            ),
            maximum_epochs=1,
            patience=1,
            device=torch.device("cpu"),
        )


def _tracked_input(
    *,
    first_index: int,
    last_index: int,
    first_date: str,
    last_date: str,
    alignment: str,
    canonical_splits: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "BRAZIL_RV_V2_MODEL_INPUT_V1",
        "store": {
            "schema": "BRAZIL_RV_V2_STORE_V1",
            "manifest_sha256": "a" * 64,
            "axes": {"date_count": 1_000, "isin_count": 4},
            "fast_identity": {},
        },
        "features": {
            "ordered_slow_and_sidecar_names": [f"slow_{i}" for i in range(32)],
            "enabled_sidecar_groups": [],
            "ordered_sidecar_names": {},
            "ordered_intraday_names": [],
        },
        "lookback_sessions": 20,
        "entry_alignment": alignment,
        "canonical_splits": canonical_splits,
        "dates": {
            "first_index": first_index,
            "last_index": last_index,
            "first_date": first_date,
            "last_date": last_date,
        },
    }


def test_stage_input_contract_rejects_overlap_and_wrong_f_embargo() -> None:
    canonical = {
        "F1": {
            "fit": {"first_index": 0, "last_index": 9},
            "selection": {"first_index": 85, "last_index": 120},
            "embargo_sessions": 75,
        }
    }
    training = _tracked_input(
        first_index=0,
        last_index=9,
        first_date="2021-08-16",
        last_date="2023-03-17",
        alignment="through_t_minus_1",
        canonical_splits=canonical,
    )
    valid_selection = _tracked_input(
        first_index=85,
        last_index=90,
        first_date="2023-07-03",
        last_date="2023-07-10",
        alignment="through_t_minus_1",
        canonical_splits=canonical,
    )
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    _validate_tracked_stage_inputs(
        "F", "F1_select_even", config, training, valid_selection
    )
    overlap = dict(valid_selection)
    overlap["dates"] = {
        **valid_selection["dates"],
        "first_index": 9,
    }
    with pytest.raises(ValueError, match="ordered and disjoint"):
        _validate_tracked_stage_inputs(
            "F", "F1_select_even", config, training, overlap
        )
    short_embargo = dict(valid_selection)
    short_embargo["dates"] = {
        **valid_selection["dates"],
        "first_index": 84,
    }
    with pytest.raises(ValueError, match="75-session embargo"):
        _validate_tracked_stage_inputs(
            "F", "F1_select_even", config, training, short_embargo
        )


def test_stage_input_contract_rejects_wrong_p_embargo_and_boundaries() -> None:
    canonical = {
        "P": {
            "fit": {"first_index": 0, "last_index": 99},
            "selection": {"first_index": 170, "last_index": 199},
            "embargo_sessions": 70,
            "selection_fraction": 0.10,
        }
    }
    training = _tracked_input(
        first_index=0,
        last_index=99,
        first_date="2010-01-04",
        last_date="2020-12-31",
        alignment="through_t",
        canonical_splits=canonical,
    )
    selection = _tracked_input(
        first_index=169,
        last_index=199,
        first_date="2021-01-04",
        last_date="2021-07-30",
        alignment="through_t",
        canonical_splits=canonical,
    )
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    with pytest.raises(ValueError, match="70-session embargo"):
        _validate_tracked_stage_inputs(
            "P", "pretrain_internal", config, training, selection
        )
    selection["dates"] = {
        **selection["dates"],
        "first_index": 169,
    }
    training["dates"] = {
        **training["dates"],
        "last_index": 98,
    }
    with pytest.raises(ValueError, match="canonical P boundaries"):
        _validate_tracked_stage_inputs(
            "P", "pretrain_internal", config, training, selection
        )


def _joint_segment(
    name: str,
    alignment: str,
    first_index: int,
    last_index: int,
    first_date: str,
    last_date: str,
) -> dict[str, object]:
    return {
        "name": name,
        "entry_alignment": alignment,
        "indices_sha256": "1" * 64,
        "identity_sha256": "2" * 64,
        "count": last_index - first_index + 1,
        "first_index": first_index,
        "last_index": last_index,
        "first_date": first_date,
        "last_date": last_date,
    }


def test_joint_input_contract_records_and_enforces_ordered_p_f_segments() -> None:
    axis = np.arange(
        np.datetime64("2010-01-04"),
        np.datetime64("2021-08-20"),
        dtype="datetime64[D]",
    )
    pretrain = np.flatnonzero(axis <= np.datetime64("2021-07-30"))
    finetune = np.flatnonzero(axis >= np.datetime64("2021-08-16"))
    segments = _model_input_segments(
        axis, np.concatenate((pretrain, finetune)), "joint"
    )
    assert [segment["name"] for segment in segments] == ["P", "F"]
    assert [segment["entry_alignment"] for segment in segments] == [
        "through_t",
        "through_t_minus_1",
    ]
    assert segments[0]["indices_sha256"] != segments[1]["indices_sha256"]

    canonical = {
        "P": {
            "fit": {"first_index": 0, "last_index": 69},
            "selection": {"first_index": 90, "last_index": 99},
            "embargo_sessions": 70,
            "selection_fraction": 0.10,
        },
        "F1": {
            "fit": {
                "first_index": 110,
                "last_index": 119,
                "first_date": "2021-08-16",
                "last_date": "2023-03-17",
                "count": 10,
            },
            "selection": {
                "first_index": 195,
                "last_index": 200,
                "first_date": "2023-07-03",
                "last_date": "2023-12-29",
                "count": 6,
            },
            "embargo_sessions": 75,
        }
    }
    training = _tracked_input(
        first_index=0,
        last_index=119,
        first_date="2010-01-04",
        last_date="2023-03-17",
        alignment="per_segment",
        canonical_splits=canonical,
    )
    training["segments"] = [
        _joint_segment(
            "P", "through_t", 0, 99, "2010-01-04", "2021-07-30"
        ),
        _joint_segment(
            "F",
            "through_t_minus_1",
            110,
            119,
            "2021-08-16",
            "2023-03-17",
        ),
    ]
    selection = _tracked_input(
        first_index=195,
        last_index=200,
        first_date="2023-07-03",
        last_date="2023-12-29",
        alignment="through_t_minus_1",
        canonical_splits=canonical,
    )
    selection["segments"] = [
        _joint_segment(
            "F",
            "through_t_minus_1",
            195,
            200,
            "2023-07-03",
            "2023-12-29",
        )
    ]
    config = ModelConfig(slow_feature_count=32, slow_lookback=20)
    _validate_tracked_stage_inputs(
        "J", "F1_joint", config, training, selection
    )

    training["segments"] = list(reversed(training["segments"]))
    with pytest.raises(ValueError, match="ordered P/F training"):
        _validate_tracked_stage_inputs(
            "J", "F1_joint", config, training, selection
        )
