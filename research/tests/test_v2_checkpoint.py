from dataclasses import replace

import numpy as np
import pytest
import torch

from brazil_rv.v2.config import TRIAGE_PROTOCOL
from brazil_rv.v2.contract import TRADED_PRIMARY_HORIZONS
from brazil_rv.v2.evaluate import (
    evaluate_scores,
    _primary_population_components,
    _primary_daily_metrics,
)
from brazil_rv.v2.losses import multi_horizon_loss_components
from brazil_rv.modeling.engine import soft_spearman_loss
from brazil_rv.v2.train import _common_primary_selection_score
from test_v2_evaluate import _fixture


def test_settlement_replay_pools_no_print_fold_and_keeps_nonledger_identical():
    from brazil_rv.v2 import research_rounds as rr
    from brazil_rv.v2.round4 import settlement_replay_projection

    inputs = _fixture()
    prices = inputs.raw_close.copy()
    prices[-1] = np.nan
    inputs = replace(inputs, raw_close=prices)
    old = evaluate_scores(inputs, window_name="F1", protocol=TRIAGE_PROTOCOL)
    new = evaluate_scores(
        inputs,
        window_name="F1",
        protocol=TRIAGE_PROTOCOL,
        settle_terminal_residuals=True,
    )
    assert settlement_replay_projection(old.report) == settlement_replay_projection(
        new.report
    )
    assert new.report["economics"]["headline"]["economics_unresolved"]
    assert (
        new.report["economics"]["headline"][
            "terminal_unresolved_inventory_fraction_nav"
        ]
        > 0
    )
    old_series = rr._daily_series(rr._ResearchEvaluation(old, inputs))
    new_series = rr._daily_series(rr._ResearchEvaluation(new, inputs))
    assert np.isnan(old_series["headline_net_excess_bps"]).all()
    assert np.isfinite(new_series["headline_net_excess_bps"]).all()
    np.testing.assert_array_equal(
        new.headline_net_excess_bps[:-1], old.headline_net_excess_bps[:-1]
    )
    pair = rr._paired_readouts(
        {"F1": rr._ResearchEvaluation(new, inputs)},
        {"F1": rr._ResearchEvaluation(new, inputs)},
    )
    assert pair["pooled"]["headline_net_excess_bps"]["estimate"] == 0.0
    assert pair["pooled"]["headline_net_excess_bps"]["finite_observations"] == len(
        inputs.dates
    )


def test_resolved_fold_secondary_reproduces_prior_economics_exactly(tmp_path):
    from brazil_rv.v2 import research_rounds as rr
    from brazil_rv.v2.artifacts import write_json_atomic
    from brazil_rv.v2.checkpoint_readouts import candidate_readout
    from dataclasses import fields

    # Expand the fixture universe so its normal buffered book has resolved economics.
    inputs = _fixture()
    changes = {}
    for field in fields(inputs):
        value = getattr(inputs, field.name)
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[1] == 60:
            changes[field.name] = np.repeat(value, 2, axis=1)
        elif isinstance(value, dict):
            changes[field.name] = {
                k: np.repeat(v, 2, axis=1)
                if isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[1] == 60
                else v
                for k, v in value.items()
            }
    changes["security_ids"] = tuple(f"SEC-{i}" for i in range(120))
    changes["action_successor_index"] = np.broadcast_to(
        np.arange(120), (25, 120)
    ).copy()
    inputs = replace(inputs, **changes)
    old = evaluate_scores(inputs, window_name="F1", protocol=TRIAGE_PROTOCOL)
    new = evaluate_scores(
        inputs,
        window_name="F1",
        protocol=TRIAGE_PROTOCOL,
        settle_terminal_residuals=True,
    )
    assert not old.report["economics"]["headline"]["economics_unresolved"]
    np.testing.assert_array_equal(
        old.headline_net_excess_bps, new.headline_net_excess_bps
    )
    outputs = {}
    for label, result in (("old", old), ("new", new)):
        path = tmp_path / label
        write_json_atomic(path / "evaluation.json", result.report)
        fields = rr._daily_series(rr._ResearchEvaluation(result, inputs))
        write_json_atomic(
            path / "daily_readouts.json",
            {
                "dates": [d.isoformat() for d in inputs.dates],
                "series": {
                    k: [float(x) if np.isfinite(x) else None for x in v]
                    for k, v in fields.items()
                },
            },
        )
        outputs[label] = candidate_readout({"F1": path})
    assert (
        outputs["new"]["pooled"]["resolved_fold_only_net_excess_bps"]
        == outputs["old"]["pooled"]["headline_net_excess_bps"]
    )


def test_cpu_seal_discloses_absent_transfer_without_weakening_access(tmp_path):
    import json
    from brazil_rv.v2.research_rounds import seal_root
    from brazil_rv.v2.research_checkpoint import SCHEMA
    from brazil_rv.v2.artifacts import write_json_atomic

    path = tmp_path / "frozen_design.json"
    payload = {
        "schema": SCHEMA,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    write_json_atomic(path, payload)
    before = path.read_bytes()
    seal_root(root=tmp_path)
    assert path.read_bytes() == before
    audit = json.loads((tmp_path / "access_audit.json").read_text())
    assert audit["data_only_artifacts_without_neural_transfer"] == [
        "frozen_design.json"
    ]
    write_json_atomic(path, {**payload, "official_validation_accessed": True})
    with pytest.raises(PermissionError, match="sealed-window access"):
        seal_root(root=tmp_path)


def test_default_loss_preserves_mean_value_and_gradient_bitwise():
    torch.manual_seed(19)
    scores = torch.randn(2, 30, 6, requires_grad=True)
    targets = torch.randn_like(scores)
    mask = torch.rand_like(scores) > 0.15
    expected_heads = torch.stack(
        [
            soft_spearman_loss(
                scores[..., h : h + 1], targets[..., h : h + 1], mask[..., h : h + 1]
            )
            for h in range(5)
        ]
    )
    expected = expected_heads.mean()
    actual = multi_horizon_loss_components(scores, targets, mask)["horizon"]
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual, scores, retain_graph=True)[0],
        torch.autograd.grad(expected, scores, retain_graph=True)[0],
    )
    weights = tuple(x / 3.5 for x in (0.25, 0.25, 1.0, 1.0, 1.0))
    weighted = multi_horizon_loss_components(
        scores, targets, mask, horizon_loss_weights=weights
    )["horizon"]
    assert torch.equal(
        weighted, (expected_heads * expected_heads.new_tensor(weights)).sum()
    )


def test_traded_primary_is_one_for_target_ranks_and_ignores_d1_d2_support():
    inputs = _fixture()
    target_mask = inputs.neutral_target_mask.copy()
    target_mask[..., :2] = False
    inputs = replace(
        inputs,
        scores=inputs.neutral_midrank_targets.copy(),
        neutral_target_mask=target_mask,
    )
    components = _primary_population_components(inputs, TRADED_PRIMARY_HORIZONS)
    _, daily, _ = _primary_daily_metrics(
        *components, inputs.dates, TRADED_PRIMARY_HORIZONS
    )
    np.testing.assert_allclose(daily[np.isfinite(daily)], 1.0)
    scores, targets, outcome, support = components
    assert _common_primary_selection_score(
        scores,
        targets,
        np.repeat(outcome[..., None], 3, axis=-1),
        inputs.active,
        horizons=TRADED_PRIMARY_HORIZONS,
    ) == pytest.approx(1.0)
    assert (outcome & support).sum() > 0
    result = evaluate_scores(inputs, window_name="F1", protocol=TRIAGE_PROTOCOL)
    assert result.report["primary_ic"]["estimate"] == pytest.approx(1.0)
    assert result.report["legacy_primary_ic_1235"]["estimate"] is None


def test_d10_missingness_restricts_every_traded_head():
    inputs = _fixture()
    mask = inputs.neutral_target_mask.copy()
    mask[0, 22:, 4] = False
    scores, targets, outcome, support = _primary_population_components(
        replace(inputs, neutral_target_mask=mask), TRADED_PRIMARY_HORIZONS
    )
    _, _, rows = _primary_daily_metrics(
        scores, targets, outcome, support, inputs.dates, TRADED_PRIMARY_HORIZONS
    )
    assert rows[0]["common_score_and_outcome_name_count"] == int(
        (outcome[0] & support[0]).sum()
    )
    assert not outcome[0, 22:].any()
