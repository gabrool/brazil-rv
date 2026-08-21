from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from brazil_rv.modeling.contract import (
    CONTEXT_COUNT,
    EQUITY_COUNT,
    EXPECTED_DECISIONS_PER_DATE,
    RuntimeSettings,
    TCNArchitecture,
)
from brazil_rv.modeling.data import (
    EXTERNAL_SIDECAR_SCHEMA,
    _build_sidecar_batch,
    feature_store_axis_identity,
    feature_store_identity,
    load_external_sidecar,
)
from brazil_rv.modeling.engine import checkpoint_payload, soft_spearman_loss
from brazil_rv.modeling.model import SharedCausalTCN
from brazil_rv.modeling.provenance import build_run_provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_store(tmp_path: Path, date_count: int = 2) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    (store / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "feature_schema.json").write_text(
        json.dumps({"contract_version": "M1_FEATURES_PIT_CAUSAL_TOD"}),
        encoding="utf-8",
    )
    (store / "sample_index.parquet").write_bytes(b"sample-index")
    pl.DataFrame(
        {
            "date_idx": np.arange(date_count, dtype=np.int32),
            "trade_date": [
                date(2024, 1, 2) + timedelta(days=i) for i in range(date_count)
            ],
        }
    ).write_parquet(store / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": np.arange(EQUITY_COUNT, dtype=np.int16),
            "security_id": [f"SECURITY_{i:03d}" for i in range(EQUITY_COUNT)],
        }
    ).write_parquet(store / "equity_index.parquet")
    return store


def _write_sidecar(
    path: Path,
    store: Path,
    values: np.ndarray,
    mask: np.ndarray,
    cadence: str,
) -> None:
    path.mkdir()
    np.save(path / "values.npy", values, allow_pickle=False)
    np.save(path / "mask.npy", mask, allow_pickle=False)
    feature_count = values.shape[-1]
    manifest = {
        "schema": EXTERNAL_SIDECAR_SCHEMA,
        "cadence": cadence,
        "feature_names": [f"feature_{i}" for i in range(feature_count)],
        "feature_store_identity": feature_store_identity(store),
        "axes": feature_store_axis_identity(store),
        "arrays": {
            "values.npy": {
                "shape": list(values.shape),
                "dtype": "float32",
                "sha256": _sha256(path / "values.npy"),
            },
            "mask.npy": {
                "shape": list(mask.shape),
                "dtype": "bool",
                "sha256": _sha256(path / "mask.npy"),
            },
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_sidecar_rejects_canonical_axis_misalignment(tmp_path: Path) -> None:
    store = _feature_store(tmp_path)
    sidecar = tmp_path / "sidecar"
    values = np.zeros((2, EQUITY_COUNT, 1), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)
    _write_sidecar(sidecar, store, values, mask, "daily")
    manifest_path = sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["axes"]["equity_axis_sha256"] = "misaligned"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="axes are misaligned"):
        load_external_sidecar(sidecar, store)


def test_sidecar_rejects_store_identity_and_nonzero_invalid_values(
    tmp_path: Path,
) -> None:
    store = _feature_store(tmp_path)
    values = np.zeros((2, EQUITY_COUNT, 1), dtype=np.float32)
    mask = np.zeros_like(values, dtype=bool)

    identity_dir = tmp_path / "identity"
    _write_sidecar(identity_dir, store, values, mask, "daily")
    manifest_path = identity_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_store_identity"]["metadata_sha256"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="feature store identity is misaligned"):
        load_external_sidecar(identity_dir, store)

    invalid_dir = tmp_path / "invalid"
    values[0, 0, 0] = 1.0
    _write_sidecar(invalid_dir, store, values, mask, "daily")
    with pytest.raises(ValueError, match="invalid values must be exactly zero"):
        load_external_sidecar(invalid_dir, store)


def test_daily_and_intraday_sidecars_use_the_available_decision_index(
    tmp_path: Path,
) -> None:
    store = _feature_store(tmp_path)
    active = np.zeros((2, EQUITY_COUNT), dtype=bool)
    active[:, :2] = True
    dates = np.array([0, 0], dtype=np.int64)
    decisions = np.array([3, 7], dtype=np.int64)

    daily_values = np.zeros((2, EQUITY_COUNT, 1), dtype=np.float32)
    daily_values[0, :, 0] = np.arange(EQUITY_COUNT)
    daily_mask = np.ones_like(daily_values, dtype=bool)
    daily_dir = tmp_path / "daily"
    _write_sidecar(daily_dir, store, daily_values, daily_mask, "daily")
    daily = load_external_sidecar(daily_dir, store)
    daily_batch = _build_sidecar_batch(
        {
            "values.npy": np.load(daily_dir / "values.npy", mmap_mode="r"),
            "mask.npy": np.load(daily_dir / "mask.npy", mmap_mode="r"),
        },
        daily.cadence,
        dates,
        decisions,
        active,
    )
    np.testing.assert_array_equal(daily_batch[0], daily_batch[1])
    assert daily_batch.shape == (2, EQUITY_COUNT, 2)
    assert daily_batch[0, 0, 1] == daily_batch[0, 1, 1] == 1
    assert not daily_batch[:, 2:].any()

    intraday_values = np.zeros(
        (2, EQUITY_COUNT, EXPECTED_DECISIONS_PER_DATE, 1), dtype=np.float32
    )
    intraday_values[0, :, 3, 0] = 3
    intraday_values[0, :, 7, 0] = 7
    intraday_mask = np.ones_like(intraday_values, dtype=bool)
    intraday_dir = tmp_path / "intraday"
    _write_sidecar(intraday_dir, store, intraday_values, intraday_mask, "intraday")
    intraday = load_external_sidecar(intraday_dir, store)
    intraday_batch = _build_sidecar_batch(
        {
            "values.npy": np.load(intraday_dir / "values.npy", mmap_mode="r"),
            "mask.npy": np.load(intraday_dir / "mask.npy", mmap_mode="r"),
        },
        intraday.cadence,
        dates,
        decisions,
        active,
    )
    np.testing.assert_array_equal(intraday_batch[:, 0, 0], [3, 7])
    np.testing.assert_array_equal(intraday_batch[:, 0, 1], [1, 1])


def _tiny_architecture() -> TCNArchitecture:
    return TCNArchitecture(
        patch_input_width=2,
        width=8,
        swiglu_hidden_width=4,
        residual_blocks=1,
        dilations=(1,),
        slow_width=2,
        fusion_width=16,
        dropout=0.0,
        output_horizons=1,
    )


def _model_inputs(equity_count: int) -> tuple[torch.Tensor, ...]:
    instrument_count = equity_count + CONTEXT_COUNT
    return (
        torch.randn(1, instrument_count, 69, 2),
        torch.ones(1, instrument_count, 69, dtype=torch.bool),
        torch.ones(1, instrument_count, dtype=torch.bool),
        torch.randn(1, instrument_count, 2),
        torch.tensor([69]),
    )


def test_zero_start_sidecar_is_exact_parent_and_adapter_gradient_wakes() -> None:
    torch.manual_seed(41)
    parent = SharedCausalTCN(architecture=_tiny_architecture(), equity_count=4)
    parent_rng = torch.get_rng_state().clone()
    torch.manual_seed(41)
    candidate = SharedCausalTCN(
        architecture=_tiny_architecture(),
        equity_count=4,
        sidecar_feature_count=2,
    )
    candidate_rng = torch.get_rng_state().clone()
    assert torch.equal(parent_rng, candidate_rng)
    parent_state = parent.state_dict()
    for name, value in parent_state.items():
        torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)

    torch.manual_seed(9)
    inputs = _model_inputs(4)
    sidecar = torch.tensor(
        [
            [
                [-1.0, 0.2, 1.0, 1.0],
                [-0.4, 0.5, 1.0, 1.0],
                [0.3, -0.1, 1.0, 1.0],
                [1.2, -0.7, 1.0, 1.0],
            ]
        ]
    )
    parent.eval()
    candidate.eval()
    parent_predictions = parent(*inputs)
    candidate_predictions = candidate(*inputs, sidecar)
    torch.testing.assert_close(
        candidate_predictions, parent_predictions, rtol=0, atol=0
    )

    targets = torch.tensor([[[-0.75], [-0.25], [0.25], [0.75]]])
    label_mask = torch.ones_like(targets, dtype=torch.bool)
    loss = soft_spearman_loss(candidate_predictions, targets, label_mask)
    loss.backward()
    assert candidate.sidecar_adapter is not None
    assert candidate.sidecar_adapter.weight.grad is not None
    assert candidate.sidecar_adapter.weight.grad.norm() > 0


def test_sidecar_adapter_never_changes_an_all_missing_equity() -> None:
    torch.manual_seed(41)
    parent = SharedCausalTCN(architecture=_tiny_architecture(), equity_count=4)
    torch.manual_seed(41)
    candidate = SharedCausalTCN(
        architecture=_tiny_architecture(),
        equity_count=4,
        sidecar_feature_count=2,
    )
    assert candidate.sidecar_adapter is not None
    assert candidate.sidecar_adapter.bias is None
    with torch.no_grad():
        candidate.sidecar_adapter.weight.fill_(0.25)

    equity_states = torch.randn(1, 4, _tiny_architecture().width)
    sidecar = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -0.5, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ]
    )
    injected = candidate._inject_sidecar(equity_states, sidecar)
    torch.testing.assert_close(injected[:, 0], equity_states[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(injected[:, 2], equity_states[:, 2], rtol=0, atol=0)
    assert not torch.equal(injected[:, 1], equity_states[:, 1])
    assert not torch.equal(injected[:, 3], equity_states[:, 3])

    torch.manual_seed(9)
    inputs = _model_inputs(4)
    all_missing = torch.zeros(1, 4, 4)
    parent.eval()
    candidate.eval()
    torch.testing.assert_close(
        candidate(*inputs, all_missing), parent(*inputs), rtol=0, atol=0
    )


def test_sidecar_identity_is_recorded_in_provenance_and_checkpoint(
    tmp_path: Path,
) -> None:
    sidecar_identity = {
        "path": str(tmp_path / "sidecar"),
        "feature_count": 2,
        "manifest_sha256": "manifest",
    }
    model = SharedCausalTCN(sidecar_feature_count=2)
    runtime = RuntimeSettings(
        effective_batch_size=2,
        loader_batch_size=2,
        microbatch_size=2,
        evaluation_batch_size=2,
        num_workers=0,
    )
    provenance = build_run_provenance(
        repository_commit_value="sha",
        feature_store=tmp_path,
        feature_store_metadata={"identity": "store"},
        seed=11,
        fit_window={"name": "fit"},
        selection_window={"name": "selection"},
        selection_note="internal fold",
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        training_sample_count=2,
        date_replacement=False,
        external_sidecar=sidecar_identity,
        runtime=runtime,
    )
    assert provenance["external_sidecar"] == sidecar_identity
    adapter = provenance["model"]["external_sidecar_adapter"]
    assert adapter["input_width"] == 4
    assert adapter["bias"] is False
    assert adapter["all_missing_input_injection"] == "identically_zero"
    state = model.state_dict()
    assert "sidecar_adapter.bias" not in state
    payload = checkpoint_payload(
        model,
        {"ema_098": state, "ema_099": state, "ema_0995": state},
        seed=11,
        epoch=1,
        validation_scores={"raw": 0.01},
        feature_store=tmp_path,
        run_provenance=provenance,
    )
    assert payload["external_sidecar"] == sidecar_identity
    assert payload["model"] == provenance["model"]
