from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from brazil_rv.modeling.contract import HORIZON_COUNT, VALIDATION_END, RuntimeSettings
from brazil_rv.modeling.data import (
    AUXILIARY_IDENTITY_FILES,
    AUXILIARY_TARGET_SCHEMA,
    auxiliary_target_identity,
)
from brazil_rv.modeling.engine import (
    eager_training_objective,
    objective_metadata,
    run_effective_batch_update,
)
from brazil_rv.modeling.model import (
    auxiliary_prediction_slices,
    build_auxiliary_model,
    build_model,
)
from brazil_rv.modeling.provenance import model_metadata
from brazil_rv.preprocessing.auxiliary_targets import (
    exact_win_returns,
    paired_beta_readiness,
)
from brazil_rv.preprocessing.contract import (
    BETA_MIN_PAIRED_SESSIONS,
    CONTEXT_SESSION_MINUTES,
    DECISION_CONTEXT_INDICES,
    HORIZONS,
)


def test_exact_win_returns_use_only_entry_and_exact_exit() -> None:
    raw = np.zeros((1, CONTEXT_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(raw.shape[:2], dtype=bool)
    entry = DECISION_CONTEXT_INDICES[0]
    raw[0, entry, 0] = 100.0
    observed[0, entry] = True
    for horizon, close in zip(HORIZONS, (101.0, 102.0, 103.0), strict=True):
        exit_index = entry + horizon - 1
        raw[0, exit_index, 3] = close
        observed[0, exit_index] = True

    returns, mask = exact_win_returns(raw, observed)
    np.testing.assert_allclose(
        returns[0, 0],
        np.log(np.asarray([101.0, 102.0, 103.0]) / 100.0),
        rtol=0.0,
        atol=1e-7,
    )
    assert mask[0, 0].all()

    after_exit = raw.copy()
    after_exit[0, entry + HORIZONS[0], 3] = 99_999.0
    mutated, _ = exact_win_returns(after_exit, observed)
    assert mutated[0, 0, 0] == returns[0, 0, 0]

    missing = observed.copy()
    missing[0, entry + HORIZONS[1] - 1] = False
    missing_returns, missing_mask = exact_win_returns(raw, missing)
    assert not missing_mask[0, 0, 1]
    assert missing_returns[0, 0, 1] == 0.0


def test_beta_readiness_is_causal_and_emits_before_update() -> None:
    dates = BETA_MIN_PAIRED_SESSIONS + 3
    equities = np.ones((dates, 2), dtype=bool)
    factor = np.ones(dates, dtype=bool)
    factor[0] = False
    baseline = paired_beta_readiness(equities, factor)
    assert not baseline[BETA_MIN_PAIRED_SESSIONS - 1].any()
    assert not baseline[BETA_MIN_PAIRED_SESSIONS].any()
    assert baseline[BETA_MIN_PAIRED_SESSIONS + 1].all()

    mutated = factor.copy()
    mutated[BETA_MIN_PAIRED_SESSIONS] = False
    changed = paired_beta_readiness(equities, mutated)
    np.testing.assert_array_equal(
        baseline[: BETA_MIN_PAIRED_SESSIONS + 1],
        changed[: BETA_MIN_PAIRED_SESSIONS + 1],
    )
    assert not changed[BETA_MIN_PAIRED_SESSIONS + 1].all()


def test_auxiliary_heads_preserve_parent_initialization_and_rng() -> None:
    torch.manual_seed(17)
    parent = build_model()
    parent_after = torch.rand(4)
    torch.manual_seed(17)
    candidate = build_auxiliary_model("combined")
    candidate_after = torch.rand(4)

    torch.testing.assert_close(parent_after, candidate_after)
    candidate_state = candidate.state_dict()
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, candidate_state[name])
    assert all(
        torch.count_nonzero(parameter) == 0
        for name, parameter in candidate.named_parameters()
        if name.startswith("auxiliary_heads.")
    )
    assert "auxiliary" not in model_metadata()
    assert model_metadata("combined")["auxiliary"]["name"] == "combined"


def test_all_auxiliary_losses_have_gradient_at_zero_start() -> None:
    batch = 2
    equities = 4
    outputs = torch.zeros(
        batch,
        equities,
        HORIZON_COUNT * 4,
        dtype=torch.float32,
        requires_grad=True,
    )
    rank = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    targets = rank[None, :, None].expand(batch, -1, HORIZON_COUNT).clone()
    residual_targets = targets.flip(1)
    mask = torch.ones_like(targets, dtype=torch.bool)
    sign_targets = (targets > 0).to(torch.float32)
    magnitude_targets = targets.abs() + 0.25

    loss = eager_training_objective("combined")(
        outputs,
        targets,
        mask,
        residual_targets,
        mask,
        sign_targets,
        magnitude_targets,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert outputs.grad is not None
    for selection in auxiliary_prediction_slices("combined").values():
        assert torch.count_nonzero(outputs.grad[..., selection]) > 0
    metadata = objective_metadata("combined")
    assert metadata["total_auxiliary_weight"] == 0.5
    assert metadata["bundle_aggregation"] == "equal_mean_with_fixed_total_weight"


class _TinyAuxiliaryRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(1, 2, bias=False)
        self.auxiliary = nn.Linear(2, HORIZON_COUNT * 3, bias=False)
        nn.init.zeros_(self.auxiliary.weight)

    def forward(
        self,
        patches: torch.Tensor,
        history_patch_mask: torch.Tensor,
        instrument_mask: torch.Tensor,
        slow_features: torch.Tensor,
        state_position: torch.Tensor,
    ) -> torch.Tensor:
        del history_patch_mask, instrument_mask, slow_features, state_position
        features = self.trunk(patches[:, :4, 0, :1])
        main = features[..., :1].expand(-1, -1, HORIZON_COUNT) * 0.0
        return torch.cat((main, self.auxiliary(features)), dim=-1)


def test_ten_step_auxiliary_smoke_wakes_head_and_upstream_trunk() -> None:
    runtime = RuntimeSettings(
        effective_batch_size=2,
        loader_batch_size=2,
        microbatch_size=2,
        evaluation_batch_size=2,
        num_workers=0,
        compile_backend="eager",
    )
    model = _TinyAuxiliaryRanker()
    initial_trunk = model.trunk.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    rank = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    targets = rank[None, :, None].expand(2, -1, HORIZON_COUNT).clone()
    mask = torch.ones_like(targets, dtype=torch.bool)
    batch = {
        "patches": torch.tensor(
            [[[[0.1]], [[0.2]], [[0.4]], [[0.3]]]] * 2,
            dtype=torch.float32,
        ),
        "history_patch_mask": torch.ones(2, 4, 1, dtype=torch.bool),
        "instrument_mask": torch.ones(2, 4, dtype=torch.bool),
        "slow_features": torch.zeros(2, 4, 1),
        "state_position": torch.ones(2, dtype=torch.long),
        "targets": targets,
        "label_mask": mask,
        "residual_targets": targets.flip(1),
        "residual_mask": mask,
        "sign_targets": targets > 0,
        "magnitude_targets": targets.abs() + 0.25,
    }
    for _ in range(10):
        run_effective_batch_update(
            model,
            [batch],
            optimizer,
            None,
            runtime,
            training_objective=eager_training_objective("combined"),
            auxiliary_variant="combined",
        )
    assert torch.count_nonzero(model.auxiliary.weight) > 0
    assert not torch.equal(model.trunk.weight, initial_trunk)


def test_auxiliary_sidecar_identity_rejects_mutation(tmp_path) -> None:
    feature_identity = {
        "path": str(tmp_path / "feature_store"),
        "contract_version": "M1_FEATURES_PIT_CAUSAL_TOD",
        "metadata_sha256": "feature",
    }
    hashes = {}
    for index, name in enumerate(AUXILIARY_IDENTITY_FILES):
        path = tmp_path / name
        np.save(path, np.asarray([index], dtype=np.int64), allow_pickle=False)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    audit_hash = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": AUXILIARY_TARGET_SCHEMA,
                "source_feature_store": feature_identity,
                "through": VALIDATION_END.isoformat(),
                "test_accessed": False,
                "array_sha256": hashes,
                "audit_file_sha256": audit_hash,
            }
        ),
        encoding="utf-8",
    )
    identity = auxiliary_target_identity(tmp_path, feature_identity)
    assert identity["array_sha256"] == hashes
    assert identity["audit_file_sha256"] == audit_hash

    audit_path.write_text('{"mutation": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="recorded contract"):
        auxiliary_target_identity(tmp_path, feature_identity)
    audit_path.write_text("{}", encoding="utf-8")

    with (tmp_path / AUXILIARY_IDENTITY_FILES[0]).open("ab") as output:
        output.write(b"mutation")
    with pytest.raises(ValueError, match="recorded contract"):
        auxiliary_target_identity(tmp_path, feature_identity)
