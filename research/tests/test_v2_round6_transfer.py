import copy
from dataclasses import replace

import pytest
import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import CHECKPOINT_INPUT_SCHEMA, RAW_PATIENCE_SCHEMA
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.train import (
    _canonical_payload_sha256,
    _model_forward,
    load_pretrain_handoff,
    model_config_contract,
)
from test_v2_residual_sidecars import _models_and_batch


def _handoff(tmp_path):
    parent, child, batch = _models_and_batch()
    source_root, target_root = tmp_path / "source", tmp_path / "target"
    manifest = {
        "axes": {"dates": "unchanged", "isins": "unchanged"},
        "tables": {},
        "arrays": {
            "slow_values": {"sha256": "a" * 64},
            "target_primary": {"sha256": "b" * 64},
        },
        "metadata": {
            "feature_schema": {"specifications": [{"family": "slow", "name": "x"}]}
        },
    }
    write_json_atomic(source_root / "manifest.json", manifest)
    destination = copy.deepcopy(manifest)
    destination["arrays"]["sidecar_events_values"] = {"sha256": "c" * 64}
    write_json_atomic(target_root / "manifest.json", destination)
    features = {
        "ordered_slow_and_sidecar_names": ["x", "y"],
        "enabled_sidecar_groups": [],
        "ordered_sidecar_names": {},
    }
    training = {
        "store": {
            "manifest_sha256": sha256_file(source_root / "manifest.json"),
            "feature_schema_sha256": "source",
        },
        "features": features,
        "lookback_sessions": 20,
        "target": {"value_array": "target_primary"},
    }
    config = model_config_contract(parent.config)
    config.pop("slow_encoder_kind")
    config.pop("sidecar_feature_counts")
    contract = {
        "schema": CHECKPOINT_INPUT_SCHEMA,
        "model_config": config,
        "training": training,
    }
    contract["sha256"] = _canonical_payload_sha256(contract)
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        {
            "schema": RAW_PATIENCE_SCHEMA,
            "stage": "P",
            "seed": 11,
            "fold": "pretrain_internal",
            "transfer_chronology_clean": True,
            "input_contract": contract,
            "model_state_dict": parent.state_dict(),
        },
        checkpoint,
    )
    fine = copy.deepcopy(contract)
    fine["model_config"] = model_config_contract(child.config)
    fine["training"]["store"] = {
        "manifest_sha256": sha256_file(target_root / "manifest.json"),
        "feature_schema_sha256": "target",
    }
    fine["training"]["features"] = {
        "ordered_slow_names": ["x", "y"],
        "enabled_sidecar_groups": ["events"],
        "ordered_sidecar_names": {"events": ["a", "b"]},
        "sidecar_encoding": "masked_zero_initialized_residual_projection",
    }
    options = dict(
        expected_sha256=sha256_file(checkpoint),
        expected_seed=11,
        fine_tune_input_contract=fine,
        parent_store_roots=(source_root, target_root),
    )
    return parent, child, batch, checkpoint, options, destination


def test_registered_additive_transfer_preserves_parent_and_reports_new_keys(tmp_path):
    parent, child, batch, checkpoint, options, _ = _handoff(tmp_path)
    loaded = load_pretrain_handoff(child, checkpoint, **options)
    assert loaded == frozenset(dict(parent.named_parameters()))
    assert child.pretrain_transfer_audit[
        "missing_projection_keys_initialized_zero"
    ] == ["sidecar_projections.events"]
    parent.eval()
    child.eval()
    extended = dict(
        batch,
        sidecar_events_values=torch.ones(2, 3, 2),
        sidecar_events_valid=torch.ones(2, 3, 2, dtype=torch.bool),
        sidecar_events_age_sessions=torch.zeros(2, 3, 2),
    )
    assert torch.equal(_model_forward(parent, batch), _model_forward(child, extended))


@pytest.mark.parametrize(
    "mutation", ["target", "names", "nonzero_projection", "architecture"]
)
def test_additive_transfer_rejects_changes_to_parent_contract(tmp_path, mutation):
    _, child, _, checkpoint, options, destination = _handoff(tmp_path)
    if mutation == "target":
        destination["arrays"]["target_primary"]["sha256"] = "d" * 64
        target = options["parent_store_roots"][1] / "manifest.json"
        write_json_atomic(target, destination)
        options["fine_tune_input_contract"]["training"]["store"]["manifest_sha256"] = (
            sha256_file(target)
        )
    elif mutation == "names":
        options["fine_tune_input_contract"]["training"]["features"][
            "ordered_slow_names"
        ].reverse()
    elif mutation == "nonzero_projection":
        with torch.no_grad():
            child.sidecar_projections["events"].fill_(1)
    else:
        child = DailyMultiHorizonModel(replace(child.config, hidden_width=32))
        options["fine_tune_input_contract"]["model_config"] = model_config_contract(
            child.config
        )
    with pytest.raises(
        ValueError, match="protected|semantics|zero initialized|model contract"
    ):
        load_pretrain_handoff(child, checkpoint, **options)
