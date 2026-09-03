from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .config import ModelConfig
from .contract import ALLOWED_SEEDS, HORIZONS
from .data import V2DailyDataset
from .model import DailyMultiHorizonModel
from .train import (
    _canonical_payload_sha256,
    _input_static_identity,
    _loader_input_payload,
    _repository_commit_if_available,
    _verified_checkpoint_input_contract,
    compile_forward,
    load_stage_checkpoint,
    model_config_contract,
    set_deterministic_seed,
)


@dataclass(frozen=True)
class ScoreArtifact:
    root: Path
    scores_path: Path
    score_mask_path: Path
    date_index_path: Path
    isin_index_path: Path
    manifest_path: Path
    manifest_sha256: str
    checkpoint_sha256: str


def _authorized_dataset(
    loader: Iterable[Mapping[str, object]],
) -> tuple[V2DailyDataset, dict[str, object]]:
    dataset = getattr(loader, "dataset", None)
    if not isinstance(dataset, V2DailyDataset):
        raise TypeError("scoring requires a DataLoader over V2DailyDataset")
    ledger = dataset.access_ledger
    if ledger is None:
        raise PermissionError("scoring requires a preauthorized dataset access ledger")
    access = ledger.payload()
    if access.get("purpose") != "evaluation":
        raise ValueError("scoring requires an evaluation-purpose access ledger")
    if access.get("test_accessed"):
        raise PermissionError("the v2 test window is sealed")
    indices = np.asarray(dataset.date_indices, dtype=np.int64)
    if indices.ndim != 1 or not indices.size or np.any(np.diff(indices) != 1):
        raise ValueError("scoring requires one contiguous chronological session axis")
    return dataset, access


def _model_batch(
    batch: Mapping[str, object],
    device: torch.device,
    *,
    omit_fast_stream: bool,
) -> dict[str, torch.Tensor]:
    names = {
        "slow_features",
        "slow_history_mask",
        "active_mask",
        "fast_patches",
        "fast_patch_mask",
        "fast_present",
        "days_since_last_slow_row",
        "fast_state_position",
        "v1_equity_slow",
    }
    if omit_fast_stream:
        names -= {
            "fast_patches",
            "fast_patch_mask",
            "fast_state_position",
            "v1_equity_slow",
        }
    result = {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
        if name in names and isinstance(value, torch.Tensor)
    }
    required = {
        "slow_features",
        "slow_history_mask",
        "active_mask",
        "fast_present",
        "days_since_last_slow_row",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"scoring batch is missing model inputs: {sorted(missing)}")
    any_fast_present = torch.any(result["fast_present"].bool())
    if not omit_fast_stream and not any_fast_present:
        for name in (
            "fast_patches",
            "fast_patch_mask",
            "fast_state_position",
            "v1_equity_slow",
        ):
            result.pop(name, None)
    elif not omit_fast_stream:
        if (
            "fast_patches" not in result
            or "fast_patch_mask" not in result
            or "v1_equity_slow" not in result
        ):
            raise ValueError(
                "present fast samples require patches, their mask, and v1 equity slow"
            )
    return result


def _forward(
    model: torch.nn.Module, batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    return model(
        batch["slow_features"],
        batch["slow_history_mask"],
        batch["active_mask"],
        batch.get("fast_patches"),
        batch.get("fast_patch_mask"),
        batch.get("fast_present"),
        batch.get("days_since_last_slow_row"),
        batch.get("fast_state_position"),
        batch.get("v1_equity_slow"),
    )


def _array_record(path: Path, values: np.ndarray) -> dict[str, object]:
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def score_checkpoint_artifact(
    *,
    checkpoint: Path,
    model_config: ModelConfig,
    loader: Iterable[Mapping[str, object]],
    output_dir: Path,
    expected_checkpoint_sha256: str | None = None,
    device: torch.device | None = None,
) -> ScoreArtifact:
    """Score one authorized chronological axis into an immutable artifact root.

    Target tensors are deliberately ignored. The explicit score mask is the
    contemporaneous point-in-time active universe, repeated over the five
    registered horizons; horizon endpoint validity remains evaluator-owned.
    """

    dataset, access = _authorized_dataset(loader)
    checkpoint = Path(checkpoint).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    prechecked_sha256 = sha256_file(checkpoint)
    if (
        expected_checkpoint_sha256 is not None
        and prechecked_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError("stage checkpoint SHA-256 differs from the manifest")
    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("stage checkpoint payload is not a mapping")
    stage = checkpoint_payload.get("stage")
    seed = checkpoint_payload.get("seed")
    fold = checkpoint_payload.get("fold")
    if (
        stage not in {"P", "F", "J"}
        or type(seed) is not int
        or seed not in ALLOWED_SEEDS
        or not isinstance(fold, str)
        or not fold
    ):
        raise ValueError("stage checkpoint is missing stage, seed, or fold provenance")
    checkpoint_contract = _verified_checkpoint_input_contract(checkpoint_payload)
    if checkpoint_contract.get("model_config") != model_config_contract(model_config):
        raise ValueError("scoring model config differs from the checkpoint contract")
    recorded_commit = checkpoint_contract.get("implementation_commit")
    current_commit = _repository_commit_if_available()
    if recorded_commit is not None and current_commit != recorded_commit:
        raise ValueError("scoring implementation commit differs from the checkpoint")
    scoring_input = _loader_input_payload(loader)
    checkpoint_selection = checkpoint_contract.get("selection")
    if not isinstance(scoring_input, Mapping) or not isinstance(
        checkpoint_selection, Mapping
    ):
        raise ValueError("scoring or checkpoint selection provenance is missing")
    if _input_static_identity(scoring_input) != _input_static_identity(
        checkpoint_selection
    ):
        raise ValueError("scoring dataset differs from the checkpoint input identity")
    expected_alignment = "through_t" if stage == "P" else "through_t_minus_1"
    expected_dataset_stage = "pretrain" if stage == "P" else "evaluation"
    if (
        dataset.stage != expected_dataset_stage
        or scoring_input.get("entry_alignment") != expected_alignment
    ):
        raise ValueError("scoring dataset has the wrong stage or entry alignment")
    scoring_input_payload = dict(scoring_input)
    scoring_input_sha256 = _canonical_payload_sha256(scoring_input_payload)
    set_deterministic_seed(seed)
    restore_config = replace(
        model_config,
        fast_pretrained=False,
        fast_pretrained_checkpoint=None,
        fast_pretrained_sha256=None,
    )
    model = DailyMultiHorizonModel(restore_config)
    checkpoint_sha256 = load_stage_checkpoint(
        model,
        checkpoint,
        expected_sha256=prechecked_sha256,
        expected_model_config=model_config,
    )

    target_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(target_device)
    model.eval()
    forward_model = compile_forward(model) if model_config.compile_forward else model
    expected_indices = np.asarray(dataset.date_indices, dtype=np.int64)
    date_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    active_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for cpu_batch in loader:
            date_index = cpu_batch.get("date_index")
            if not isinstance(date_index, torch.Tensor) or date_index.ndim != 1:
                raise ValueError("scoring batch needs a one-dimensional date_index")
            batch = _model_batch(
                cpu_batch,
                target_device,
                omit_fast_stream=dataset.stage == "pretrain",
            )
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.bfloat16,
                enabled=model_config.use_bf16 and target_device.type == "cuda",
            ):
                predictions = _forward(forward_model, batch)[..., : len(HORIZONS)]
            if predictions.shape[:2] != batch["active_mask"].shape:
                raise ValueError("model scores are misaligned with the active universe")
            date_parts.append(date_index.detach().cpu().numpy().astype(np.int64))
            score_parts.append(predictions.float().cpu().numpy())
            active_parts.append(batch["active_mask"].bool().cpu().numpy())
    if not score_parts:
        raise ValueError("scoring loader produced no rows")
    actual_indices = np.concatenate(date_parts)
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError(
            "scoring loader must emit every dataset row exactly once in chronological order"
        )
    scores = np.concatenate(score_parts).astype(np.float32, copy=False)
    active = np.concatenate(active_parts).astype(np.bool_, copy=False)
    expected_shape = (
        expected_indices.size,
        len(dataset.store.isins),
        len(HORIZONS),
    )
    if scores.shape != expected_shape or active.shape != expected_shape[:2]:
        raise ValueError("scoring output differs from the full dataset axes")
    score_mask = np.repeat(active[..., None], len(HORIZONS), axis=-1)
    if not np.isfinite(scores[score_mask]).all():
        raise FloatingPointError("an authorized score is non-finite")
    if np.any(scores[~score_mask] != 0.0):
        raise ValueError("inactive names must have exact zero model scores")

    store_root = dataset.store.root.resolve()
    dates = np.asarray(dataset.store.dates[expected_indices], dtype="datetime64[D]")
    isins = np.asarray(dataset.store.isins, dtype=np.str_)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.scoring-", dir=output.parent)
    )
    try:
        arrays = {
            "scores.npy": scores,
            "score_mask.npy": score_mask,
            "date_index.npy": dates,
            "isin_index.npy": isins,
        }
        records: dict[str, dict[str, object]] = {}
        for name, values in arrays.items():
            path = staging / name
            np.save(path, values, allow_pickle=False)
            records[name] = _array_record(path, values)
        config_payload = model_config_contract(model_config)
        index_bytes = np.asarray(expected_indices, dtype="<i8").tobytes()
        manifest = {
            "schema": "BRAZIL_RV_V2_SCORE_ARTIFACT_V1",
            "status": "completed",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
                "kind": checkpoint_payload["schema"],
                "stage": stage,
                "seed": seed,
                "fold": str(fold),
            },
            "model_config": config_payload,
            "checkpoint_input_contract_sha256": checkpoint_contract["sha256"],
            "scoring_input": scoring_input_payload,
            "scoring_input_sha256": scoring_input_sha256,
            "dataset": {
                "stage": dataset.stage,
                "lookback": dataset.lookback,
                "enabled_sidecars": list(dataset.enabled_sidecars),
                "date_indices_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "date_count": int(expected_indices.size),
                "first_date_index": int(expected_indices[0]),
                "last_date_index": int(expected_indices[-1]),
                "score_mask_rule": "active_universe_repeated_across_horizons",
            },
            "store": {
                "root": str(store_root),
                "manifest_sha256": sha256_file(store_root / "manifest.json"),
                "schema": dataset.store.manifest.get("schema"),
            },
            "access_ledger": access,
            "axes": {
                "date_count": int(scores.shape[0]),
                "isin_count": int(scores.shape[1]),
                "horizons": list(HORIZONS),
            },
            "inference": {
                "device_type": target_device.type,
                "batch_size": getattr(loader, "batch_size", None),
                "compiled": model_config.compile_forward,
                "bf16_autocast": (
                    model_config.use_bf16 and target_device.type == "cuda"
                ),
            },
            "artifacts": records,
            "official_validation_accessed": bool(
                access.get("official_validation_accessed")
            ),
            "test_accessed": bool(access.get("test_accessed")),
        }
        manifest_path = staging / "score_manifest.json"
        manifest_sha256 = write_json_atomic(manifest_path, manifest)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ScoreArtifact(
        root=output,
        scores_path=output / "scores.npy",
        score_mask_path=output / "score_mask.npy",
        date_index_path=output / "date_index.npy",
        isin_index_path=output / "isin_index.npy",
        manifest_path=output / "score_manifest.json",
        manifest_sha256=manifest_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
