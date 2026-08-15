from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from brazil_rv.modeling.baselines import SharedCausalTCN
from brazil_rv.modeling.contract import (
    ABSOLUTE_PATCH_COUNT,
    BASELINE_TCN_SETTINGS,
    CONTEXT_COUNT,
    GH200_RUNTIME,
    HORIZONS,
    PATCH_INPUT_WIDTH,
    PEER_STATE_WIDTH,
    SLOW_FEATURE_COUNT,
    TCNSettings,
    architecture_for_model,
    context_family_slots,
    tcn_tap_receptive_field_minutes,
)
from brazil_rv.modeling.data import (
    BatchRequest,
    SingleDecisionBatchSampler,
)
from brazil_rv.modeling.engine import select_training_label_mask
from brazil_rv.modeling.horizon_diagnostics import (
    FIXED_TARGET_BASIS,
    PairMoments,
    RidgeSufficientStatistics,
    assert_analysis_rows,
    build_oof_plan,
    context_permutation,
    gradient_cosine,
    mask_context_family_batch,
    permute_context_family_batch,
)
from brazil_rv.modeling.metrics import moving_block_bootstrap
from brazil_rv.modeling.model import build_neural_model, count_trainable_parameters
from brazil_rv.modeling.run_horizon_multiscale_stage import (
    ARMS,
    ARM_BY_NAME,
    TRAINING_RUN_COUNT,
    Stage,
    _fingerprint,
    parse_args as parse_stage_args,
)
from brazil_rv.modeling.train import _run_directory_name, parse_args


EQUITIES = 4
INSTRUMENTS = EQUITIES + CONTEXT_COUNT


def _settings(readout: str = "final", fusion: str = "context_pooled") -> TCNSettings:
    return replace(BASELINE_TCN_SETTINGS, readout=readout, fusion=fusion)


def _model(readout: str = "final", fusion: str = "context_pooled") -> SharedCausalTCN:
    architecture = architecture_for_model("tcn", _settings(readout, fusion))
    model = build_neural_model("tcn", architecture, "selected", equity_count=EQUITIES)
    assert isinstance(model, SharedCausalTCN)
    return model


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(91)
    patches = torch.randn(
        batch_size,
        INSTRUMENTS,
        ABSOLUTE_PATCH_COUNT,
        PATCH_INPUT_WIDTH,
        generator=generator,
    )
    history = torch.ones(
        batch_size, INSTRUMENTS, ABSOLUTE_PATCH_COUNT, dtype=torch.bool
    )
    instrument = torch.ones(batch_size, INSTRUMENTS, dtype=torch.bool)
    instrument[:, 1] = False
    slow = torch.randn(batch_size, INSTRUMENTS, SLOW_FEATURE_COUNT, generator=generator)
    state = torch.full((batch_size,), 20, dtype=torch.long)
    peer = torch.randn(batch_size, EQUITIES, PEER_STATE_WIDTH, generator=generator)
    return patches, history, instrument, slow, state, peer


def test_readout_defaults_and_receptive_fields() -> None:
    assert BASELINE_TCN_SETTINGS.readout == "final"
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    assert tcn_tap_receptive_field_minutes(architecture) == (
        15,
        35,
        75,
        155,
        315,
        635,
    )


@pytest.mark.parametrize(
    "readout",
    ("final", "shared_multiscale", "horizon_multiscale", "final_score_mlp"),
)
def test_all_readouts_preserve_shape_and_equity_mask(readout: str) -> None:
    torch.manual_seed(29)
    model = _model(readout).eval()
    result = model(*_inputs())
    assert result.shape == (2, EQUITIES, 3)
    assert torch.count_nonzero(result[:, 1]) == 0
    assert torch.isfinite(result).all()


def test_added_parameter_counts_are_exact() -> None:
    torch.manual_seed(29)
    final = count_trainable_parameters(_model("final"))
    torch.manual_seed(29)
    shared = count_trainable_parameters(_model("shared_multiscale"))
    torch.manual_seed(29)
    horizon = count_trainable_parameters(_model("horizon_multiscale"))
    torch.manual_seed(29)
    score = count_trainable_parameters(_model("final_score_mlp"))
    assert shared - final == 6
    assert horizon - final == 18
    assert score - final == 17


def test_final_score_mlp_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(29)
    final = _model("final").eval()
    torch.manual_seed(29)
    controlled = _model("final_score_mlp").eval()
    shared = {
        name: value
        for name, value in final.state_dict().items()
        if name in controlled.state_dict()
    }
    controlled.load_state_dict(shared, strict=False)
    torch.testing.assert_close(
        final(*_inputs()), controlled(*_inputs()), atol=0, rtol=0
    )


def test_old_final_settings_strict_load_with_default_readout() -> None:
    torch.manual_seed(29)
    original = _model("final")
    settings = asdict(BASELINE_TCN_SETTINGS)
    settings.pop("readout")
    restored_settings = TCNSettings(**settings)
    restored = build_neural_model(
        "tcn",
        architecture_for_model("tcn", restored_settings),
        "selected",
        equity_count=EQUITIES,
    )
    restored.load_state_dict(original.state_dict(), strict=True)


def test_forced_final_tap_shared_mixture_matches_final() -> None:
    torch.manual_seed(29)
    final = _model("final").eval()
    torch.manual_seed(29)
    shared = _model("shared_multiscale").eval()
    shared.load_state_dict(final.state_dict(), strict=False)
    with torch.no_grad():
        assert shared.scale_logits is not None
        shared.scale_logits.fill_(-torch.inf)
        shared.scale_logits[-1] = 0
    torch.testing.assert_close(
        final(*_inputs()), shared(*_inputs()), atol=1e-7, rtol=1e-6
    )


def test_scale_weights_contract_and_gradients() -> None:
    torch.manual_seed(29)
    shared = _model("shared_multiscale").eval()
    weights = shared.scale_weights()
    assert weights is not None and weights.shape == (6,)
    assert torch.all(weights >= 0) and torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
    horizon = _model("horizon_multiscale").eval()
    with torch.no_grad():
        assert horizon.scale_logits is not None
        horizon.scale_logits[0, 0] = 2
        horizon.scale_logits[2, -1] = 3
    horizon_weights = horizon.scale_weights()
    assert horizon_weights is not None and horizon_weights.shape == (3, 6)
    assert not torch.equal(horizon_weights[0], horizon_weights[2])
    loss = horizon(*_inputs()).square().sum()
    loss.backward()
    assert horizon.scale_logits.grad is not None
    assert torch.isfinite(horizon.scale_logits.grad).all()
    assert horizon.blocks[0].convolution.weight.grad is not None


def test_final_path_matches_legacy_computation_exactly() -> None:
    torch.manual_seed(29)
    model = _model("final").eval()
    inputs = _inputs()
    with torch.no_grad():
        states, taps = model._instrument_states(*inputs[:5])
        assert taps == ()
        equity = model._add_peer(states[:, :EQUITIES], inputs[5])
        reference = model._fuse_equity(
            equity,
            states[:, EQUITIES:],
            inputs[2][:, :EQUITIES],
            inputs[2][:, EQUITIES:],
        )
        reference = model.prediction_head(reference)
        reference *= inputs[2][:, :EQUITIES, None]
        actual = model(*inputs)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_causality_after_gathered_equity_state() -> None:
    torch.manual_seed(29)
    model = _model("final").eval()
    inputs = list(_inputs())
    baseline = model(*inputs)
    changed = inputs[0].clone()
    changed[:, :EQUITIES, 20:] = torch.randn_like(changed[:, :EQUITIES, 20:])
    inputs[0] = changed
    torch.testing.assert_close(baseline, model(*inputs), atol=0, rtol=0)


@pytest.mark.parametrize(
    "fusion", ("none", "context_only", "pooled_market", "context_pooled")
)
def test_all_tcn_fusions_work_with_horizon_readout(fusion: str) -> None:
    torch.manual_seed(29)
    model = _model("horizon_multiscale", fusion).eval()
    inputs = _inputs()
    if fusion in ("none", "pooled_market"):
        inputs = (
            inputs[0][:, :EQUITIES],
            inputs[1][:, :EQUITIES],
            inputs[2][:, :EQUITIES],
            inputs[3][:, :EQUITIES],
            inputs[4],
            inputs[5],
        )
    assert model(*inputs).shape == (2, EQUITIES, 3)


def test_training_controls_round_trip_through_cli_and_name() -> None:
    args = parse_args(
        [
            "--tcn-readout",
            "horizon_multiscale",
            "--training-horizon",
            "60",
            "--context-family-ablation",
            "wdo",
        ]
    )
    assert args.tcn_readout == "horizon_multiscale"
    assert args.training_horizon == "60"
    assert args.context_family_ablation == "wdo"
    name = _run_directory_name(args, __import__("datetime").datetime(2026, 8, 15))
    assert "readout-horizon_multiscale" in name
    assert "horizon-60" in name
    assert "without-wdo" in name


def test_single_horizon_mask_is_exact() -> None:
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    selected = select_training_label_mask(mask, "60")
    assert not selected[..., 0].any()
    assert selected[..., 1].all()
    assert not selected[..., 2].any()
    assert select_training_label_mask(mask, "all") is mask


def _analysis_batch() -> dict[str, torch.Tensor]:
    batch_size = 4
    patches, history, instruments, slow, state, peer = _inputs(batch_size)
    return {
        "patches": patches,
        "history_patch_mask": history,
        "instrument_mask": instruments,
        "slow_features": slow,
        "state_position": state,
        "peer_state": peer,
        "targets": torch.randn(batch_size, EQUITIES, 3),
        "label_mask": torch.ones(batch_size, EQUITIES, 3, dtype=torch.bool),
        "date_idx": torch.tensor([10, 11, 12, 13]),
        "decision_idx": torch.full((batch_size,), 18),
        "sample_valid_mask": torch.ones(batch_size, dtype=torch.bool),
    }


@pytest.mark.parametrize("family", ("wdo", "br_rates", "us_rates"))
def test_context_family_mask_zeros_all_associated_tensors(family: str) -> None:
    batch = _analysis_batch()
    masked = mask_context_family_batch(batch, family)
    slots = tuple(EQUITIES + slot for slot in context_family_slots(family))
    for name in ("patches", "history_patch_mask", "instrument_mask", "slow_features"):
        assert torch.count_nonzero(masked[name][:, slots]) == 0
    torch.testing.assert_close(masked["targets"], batch["targets"])


def test_context_permutation_is_joint_deterministic_and_label_safe() -> None:
    batch = _analysis_batch()
    mapping = {10: 11, 11: 10, 12: 13, 13: 12}
    first = permute_context_family_batch(batch, "wdo", mapping)
    second = permute_context_family_batch(batch, "wdo", mapping)
    slot = EQUITIES + context_family_slots("wdo")[0]
    torch.testing.assert_close(first["patches"][:, slot], second["patches"][:, slot])
    torch.testing.assert_close(first["patches"][0, slot], batch["patches"][1, slot])
    torch.testing.assert_close(first["targets"], batch["targets"])
    torch.testing.assert_close(first["peer_state"], batch["peer_state"])


def test_single_decision_sampler_never_crosses_decisions() -> None:
    frame = pl.DataFrame(
        {
            "sample_id": np.arange(10),
            "decision_idx": [0] * 5 + [1] * 5,
        }
    )
    sampler = SingleDecisionBatchSampler(frame, 3)
    for request in sampler:
        assert isinstance(request, BatchRequest)
        decisions = frame.get_column("decision_idx").to_numpy()[list(request.indices)]
        assert np.unique(decisions).size == 1


def test_ridge_statistics_match_direct_centered_solution() -> None:
    generator = np.random.default_rng(9)
    x = generator.normal(size=(50, 4))
    y = x @ np.asarray([0.4, -0.2, 0.1, 0.7]) + 0.3
    stats = RidgeSufficientStatistics(4)
    stats.update(x, y, np.ones(50, dtype=bool))
    penalty = 0.05
    coefficient, intercept = stats.solve(penalty)
    centered = x - x.mean(axis=0)
    expected = np.linalg.solve(
        centered.T @ centered / len(x) + penalty * np.eye(4),
        centered.T @ (y - y.mean()) / len(x),
    )
    np.testing.assert_allclose(coefficient, expected)
    assert np.isclose(intercept, y.mean() - x.mean(axis=0) @ expected)


def test_pairwise_masks_and_fixed_basis() -> None:
    moments = PairMoments()
    left = np.asarray([1.0, 2.0, 100.0])
    right = np.asarray([2.0, 4.0, -100.0])
    moments.update(left, right, np.asarray([True, True, False]))
    assert moments.count == 2
    assert np.isclose(moments.correlation(), 1.0)
    np.testing.assert_allclose(
        FIXED_TARGET_BASIS @ FIXED_TARGET_BASIS.T, np.eye(3), atol=1e-12
    )


def _train_rows() -> pl.DataFrame:
    dates = pl.date_range(
        date(2021, 8, 16), date(2021, 8, 31), interval="1d", eager=True
    )
    return pl.DataFrame(
        {
            "sample_id": np.arange(len(dates)),
            "trade_date": dates,
            "date_idx": np.arange(len(dates)),
            "decision_idx": np.zeros(len(dates), dtype=np.int64),
        }
    )


def test_oof_plan_is_chronological_and_train_bounded() -> None:
    windows, plan = build_oof_plan(_train_rows())
    assert list(windows) == ["B0", "B1", "B2", "B3"]
    for index in range(3):
        assert (
            windows[f"B{index}"].get_column("trade_date").max()
            < windows[f"B{index + 1}"].get_column("trade_date").min()
        )
    assert plan["folds"]["fold_2"]["prediction"] == "B3"


def test_analysis_assertion_rejects_test_period() -> None:
    rows = pl.DataFrame({"trade_date": [date(2025, 7, 7)]})
    with pytest.raises(ValueError, match="after"):
        assert_analysis_rows(rows, allow_validation=True)


def test_context_permutation_preserves_quarters_and_has_no_self_maps() -> None:
    dates = [
        date(2024, 7, 8),
        date(2024, 7, 9),
        date(2024, 10, 1),
        date(2024, 10, 2),
    ]
    rows = pl.DataFrame(
        {
            "date_idx": [1, 2, 3, 4],
            "trade_date": dates,
            "decision_idx": [0, 0, 0, 0],
        }
    )
    first, manifest = context_permutation(rows, seed=7)
    second, _ = context_permutation(rows, seed=7)
    assert first == second
    assert manifest["self_map_count"] == 0
    for recipient, donor in first.items():
        left = dates[recipient - 1]
        right = dates[donor - 1]
        assert (left.year, (left.month - 1) // 3) == (
            right.year,
            (right.month - 1) // 3,
        )


def test_gradient_cosine_handles_values_and_zero_norms() -> None:
    cosine, reason = gradient_cosine(
        torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    )
    assert cosine == -1.0 and reason is None
    cosine, reason = gradient_cosine(torch.zeros(2), torch.ones(2))
    assert cosine is None and reason == "zero_norm"


def test_moving_block_bootstrap_is_deterministic() -> None:
    values = np.arange(20, dtype=np.float64)
    first = moving_block_bootstrap(values, replications=100, seed=8)
    second = moving_block_bootstrap(values, replications=100, seed=8)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_stage_surface_has_no_test_option_and_exact_run_count() -> None:
    args = parse_stage_args(["--output-dir", "stage"])
    assert args.output_dir == Path("stage")
    with pytest.raises(SystemExit):
        parse_stage_args(["--output-dir", "stage", "--split", "test"])
    assert len(ARMS) == TRAINING_RUN_COUNT == 16
    assert GH200_RUNTIME.compile_fullgraph
    assert not GH200_RUNTIME.compile_dynamic
    assert tuple(HORIZONS) == (30, 60, 120)


def test_stage_resume_skips_only_valid_completed_step(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    stage = object.__new__(Stage)
    stage.commit = "abc"
    stage.store_identity = {"id": "store"}
    config = {"mode": "audit"}
    fingerprint = _fingerprint(
        {
            "commit": stage.commit,
            "feature_store": stage.store_identity,
            "config": config,
        }
    )
    stage.manifest = {
        "steps": {
            "audit": {
                "status": "completed",
                "fingerprint": fingerprint,
            }
        }
    }
    stage.logger = __import__("logging").getLogger("stage-resume-test")
    stage.write_manifest = lambda: None
    called = False

    def action() -> None:
        nonlocal called
        called = True

    stage.step("audit", config, (artifact,), action, lambda: None)
    assert not called

    def invalid() -> None:
        raise ValueError("artifact validation failed")

    with pytest.raises(ValueError, match="artifact validation"):
        stage.step("audit", config, (artifact,), action, invalid)


def test_consolidation_emits_three_seed_and_control_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from brazil_rv.modeling import run_horizon_multiscale_stage as stage_module

    run_dirs: dict[str, Path] = {}
    for arm_index, arm in enumerate(ARMS):
        run_dir = tmp_path / "runs" / arm.name
        run_dir.mkdir(parents=True)
        run_dirs[arm.name] = run_dir
        offset = 0.001 * arm_index
        metrics = {
            "primary_score": 0.1 + offset,
            "horizons": [
                {
                    "horizon_minutes": minutes,
                    "mean_daily_spearman_ic": 0.1 + offset + horizon * 0.001,
                }
                for horizon, minutes in enumerate(HORIZONS)
            ],
        }
        (run_dir / "validation_metrics.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "best_epoch": 2,
                    "epochs_completed": 4,
                    "parameter_count": 100 + arm_index,
                    "total_run_seconds": 10.0,
                }
            ),
            encoding="utf-8",
        )
        pl.DataFrame(
            [
                {
                    "date_idx": date_idx,
                    "horizon_minutes": minutes,
                    "spearman_ic": 0.1 + offset + horizon * 0.001,
                }
                for date_idx in range(8)
                for horizon, minutes in enumerate(HORIZONS)
            ]
        ).write_parquet(run_dir / "validation_daily_metrics.parquet")
    gradient_dir = tmp_path / "audits" / "gradient"
    context_dir = tmp_path / "audits" / "context"
    frozen_dir = tmp_path / "audits" / "frozen_block"
    oof_dir = tmp_path / "audits" / "oof"
    target_dir = tmp_path / "audits" / "target_basis"
    for directory in (gradient_dir, context_dir, frozen_dir, oof_dir, target_dir):
        directory.mkdir(parents=True)
    (gradient_dir / "horizon_gradient_summary.json").write_text(
        json.dumps(
            {
                "sample_count": 20,
                "by_group_and_horizon_pair": [{"fraction_negative": 0.25}],
                "single_horizon_controls": None,
            }
        ),
        encoding="utf-8",
    )
    (context_dir / "context_family_summary.json").write_text(
        json.dumps({"inference": {}}), encoding="utf-8"
    )
    (frozen_dir / "frozen_block_probe_summary.json").write_text(
        json.dumps(
            {
                "best_tap_by_horizon": {
                    "30": "block_2",
                    "60": "block_4",
                    "120": "block_6",
                },
                "earlier_tap_beats_final_post_fusion_by_horizon": {
                    "30": True,
                    "60": False,
                    "120": False,
                },
                "concatenated_beats_final_post_fusion_by_horizon": {
                    "30": False,
                    "60": False,
                    "120": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (oof_dir / "oof_residual_probe_summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "probe": "slow",
                        "horizon_minutes": 30,
                        "delta_from_base": 0.001,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (target_dir / "target_basis_summary.json").write_text(
        json.dumps(
            {
                "pooled_target_correlation": np.eye(3).tolist(),
                "eigenvalues": [1.5, 1.0, 0.5],
                "variance_shares": [0.5, 1 / 3, 1 / 6],
                "fixed_basis_variance": [1.0, 0.5, 0.25],
            }
        ),
        encoding="utf-8",
    )

    class FakeModel:
        def __init__(self, readout: str) -> None:
            self.architecture = architecture_for_model(
                "tcn", replace(BASELINE_TCN_SETTINGS, readout=readout)
            )
            self.readout = readout

        def scale_weights(self) -> torch.Tensor:
            if self.readout == "shared_multiscale":
                return torch.full((6,), 1 / 6)
            return torch.full((3, 6), 1 / 6)

    def fake_load(run_dir: Path):
        return FakeModel(ARM_BY_NAME[run_dir.name].readout), {}, tmp_path

    monkeypatch.setattr(stage_module, "load_current_neural_run", fake_load)
    stage_module._consolidate(tmp_path, run_dirs)
    stage_module.validate_consolidated(tmp_path, tuple(ARM_BY_NAME))
    comparison = pl.read_csv(tmp_path / "multiscale_comparison.csv")
    assert (
        comparison.filter(
            (pl.col("comparison") == "horizon_multiscale_vs_final_three_seed")
            & (pl.col("horizon_minutes") == 0)
        ).height
        == 1
    )
    gates = pl.read_csv(tmp_path / "multiscale_gate_weights.csv")
    assert gates.filter(pl.col("seed") == 0).height == 18
    summary = json.loads((tmp_path / "stage_summary.json").read_text(encoding="utf-8"))
    assert set(summary["hypotheses"]) == {
        "representation_information_loss",
        "shared_scale_aggregation",
        "horizon_scale_specialization",
        "trained_multiscale_result",
        "score_capacity_control",
        "horizon_conflict",
        "context_source_information",
        "target_structure",
    }
    assert summary["promotion"] == "none"
    context = pl.read_csv(context_dir / "context_training_ablations.csv")
    assert context.height == 12

    comparison_path = tmp_path / "multiscale_comparison.csv"
    original_comparisons = pl.read_csv(comparison_path)
    pl.concat((original_comparisons, original_comparisons.head(1))).write_csv(
        comparison_path
    )
    with pytest.raises(ValueError, match="comparison row count"):
        stage_module.validate_consolidated(tmp_path, tuple(ARM_BY_NAME))

    original_comparisons.write_csv(comparison_path)
    gates.filter(pl.col("arm") != "shared_multiscale_seed29").write_csv(
        tmp_path / "multiscale_gate_weights.csv"
    )
    with pytest.raises(ValueError, match="gate runs"):
        stage_module.validate_consolidated(tmp_path, tuple(ARM_BY_NAME))
