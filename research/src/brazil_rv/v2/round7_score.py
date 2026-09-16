"""Export selected characteristic scores on the canonical five-horizon container."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicConfig, CharacteristicModel
from .config import ModelConfig
from .contract import HORIZONS, SCORE_ARTIFACT_SCHEMA
from .data import V2DailyDataset, restore_name_axis, stage_name_count
from .model import DailyMultiHorizonModel
from .round7_preprocessing import Round7Preprocessing
from .round7_training import (
    CHECKPOINT_SCHEMA,
    autocast_dtype,
    forward,
    model_batch,
    sequential_batches,
)
from .score import _array_record, parent_prelude_indices, verify_reused_inference_source
from .train import _cli_stage_indices, compile_forward


def canonical_head_panel(scores, active, horizons):
    """Absent D1/D2 slots are explicitly invalid, never invented predictions."""
    output = np.zeros((*active.shape, len(HORIZONS)), np.float32)
    mask = np.zeros(output.shape, bool)
    for index, horizon in enumerate(horizons):
        destination = HORIZONS.index(horizon)
        output[..., destination] = np.where(active, scores[..., index], 0.0)
        mask[..., destination] = active
    if not np.isfinite(output[mask]).all():
        raise FloatingPointError("valid model score is non-finite")
    return output, mask


def score(
    store_root,
    checkpoint,
    output,
    *,
    expected_sha256,
    compiled=True,
    device=None,
    reusable_models=None,
    fixed_name_count=None,
    parent_prelude=False,
):
    if sha256_file(checkpoint) != expected_sha256:
        raise ValueError("scoring checkpoint differs from its bound hash")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["schema"] != CHECKPOINT_SCHEMA or payload["stage"] != (
        "P" if parent_prelude else "F"
    ):
        raise ValueError(
            "scoring checkpoint stage differs from the requested forecast block"
        )
    contract = payload["contract"]
    auxiliary = contract.get("economic_auxiliary")
    inference_source = verify_reused_inference_source(
        contract["code"]["commit"],
        extra_dependencies=("v2/economic_objective.py",) if auxiliary else (),
    )
    if parent_prelude:
        rows = parent_prelude_indices(store_root)
    else:
        _, _, rows, _ = _cli_stage_indices(store_root, "F", payload["fold"])
    if "selection_ic" not in payload:
        raise ValueError("scoring requires the selection-bound checkpoint")
    if contract["store_manifest_sha256"] != sha256_file(store_root / "manifest.json"):
        raise ValueError("scoring store differs from training")
    if output.exists():
        existing = json.loads(
            (output / "score_manifest.json").read_text(encoding="utf-8")
        )
        if existing["checkpoint"]["sha256"] != expected_sha256:
            raise ValueError("existing score panel binds another checkpoint")
        if existing["dataset"]["date_indices"] != rows.tolist():
            raise ValueError("existing score panel binds another forecast block")
        for name, record in existing["artifacts"].items():
            if sha256_file(output / name) != record["sha256"]:
                raise ValueError("existing score array changed")
        return existing
    characteristic = not contract["pretrain_key"].startswith("s0_")
    config = (CharacteristicConfig if characteristic else ModelConfig)(
        **contract["config"]
    )
    horizons = tuple(config.horizons) if characteristic else HORIZONS
    families = tuple(
        n
        for n, _ in (
            config.family_counts if characteristic else config.sidecar_feature_counts
        )
    )
    dataset = V2DailyDataset(
        store_root,
        rows,
        stage="evaluation",
        lookback=60,
        enabled_sidecars=families,
        include_fast=False,
        include_intraday=False,
        include_common_state=characteristic and bool(families),
        compact_names=True,
        purpose="evaluation",
    )
    try:
        preparation = Round7Preprocessing.from_payload(contract["preprocessing"])
        loader = DataLoader(
            dataset,
            batch_sampler=sequential_batches(len(dataset)),
            collate_fn=partial(
                preparation.collate,
                fixed_name_count=max(fixed_name_count or 0, stage_name_count(dataset)),
            ),
        )
        model = (
            reusable_models[0]
            if reusable_models
            else (
                CharacteristicModel(config)
                if characteristic
                else DailyMultiHorizonModel(config)
            )
        )
        if auxiliary and not reusable_models:
            from .economic_objective import attach_economic_head

            attach_economic_head(model)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        economic_head = model.economic_head if auxiliary else None
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        model = (
            reusable_models[1]
            if reusable_models
            else (compile_forward(model) if compiled else model)
        )
        values, validity, indices, economic_values = [], [], [], []
        with torch.no_grad():
            for cpu in loader:
                batch = model_batch(cpu, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype(device),
                    enabled=device.type == "cuda",
                ):
                    prediction = forward(
                        model,
                        batch,
                        characteristic=characteristic,
                        return_hidden=bool(auxiliary),
                    )
                    if auxiliary:
                        prediction, hidden = prediction
                        economic = (
                            economic_head(hidden).squeeze(-1).float().mean(2)
                            * auxiliary["scale"]
                        )
                    predicted = prediction.float().mean(dim=2).cpu().numpy()
                scores, mask = canonical_head_panel(
                    predicted, batch["active_mask"].cpu().numpy(), horizons
                )
                names = cpu["name_index"].numpy()
                if auxiliary:
                    economic_values.append(
                        restore_name_axis(
                            np.where(
                                batch["active_mask"].cpu().numpy(),
                                economic.cpu().numpy(),
                                0,
                            ),
                            names,
                            len(dataset.store.isins),
                        )
                    )
                values.append(
                    restore_name_axis(scores, names, len(dataset.store.isins))
                )
                validity.append(
                    restore_name_axis(mask, names, len(dataset.store.isins))
                )
                indices.extend(cpu["date_index"].tolist())
        if not np.array_equal(indices, rows):
            raise ValueError("scoring must emit each registered date exactly once")
        arrays = {
            "scores.npy": np.concatenate(values),
            "score_mask.npy": np.concatenate(validity),
            "date_index.npy": dataset.store.dates[rows],
            "isin_index.npy": np.asarray(dataset.store.isins),
        }
        if auxiliary:
            arrays["economic_daily_residual.npy"] = np.concatenate(economic_values)
        source = dataset.store.manifest
        access = dataset.access_ledger.payload()
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".round7-scores-", dir=output.parent
        ) as temporary:
            staging = Path(temporary)
            artifacts = {}
            for name, value in arrays.items():
                np.save(staging / name, value, allow_pickle=False)
                artifacts[name] = _array_record(staging / name, value)
            manifest = {
                "schema": SCORE_ARTIFACT_SCHEMA,
                "status": "completed",
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": expected_sha256,
                    "kind": CHECKPOINT_SCHEMA,
                    "stage": payload["stage"],
                    "fold": payload["fold"],
                    "seed": payload["seed"],
                },
                "round7_contract": contract,
                "inference_source": inference_source,
                "transfer_chronology_clean": True,
                "feature_schema_sha256": source["metadata"]["feature_schema"]["sha256"],
                "action_terms_source": source["metadata"]["action_terms_source"],
                "schedule_source": source["metadata"]["schedule_source"],
                "store": {
                    "root": str(store_root),
                    "manifest_sha256": contract["store_manifest_sha256"],
                },
                "axes": {
                    "date_count": len(rows),
                    "isin_count": len(dataset.store.isins),
                    "horizons": list(HORIZONS),
                },
                "trained_horizons": list(horizons),
                "dataset": {
                    "score_mask_rule": "active universe on trained heads; absent heads invalid",
                    "date_indices": rows.tolist(),
                },
                "inference": {
                    "members": config.members if characteristic else 1,
                    "member_aggregation": "mean raw prediction",
                    "device_type": device.type,
                    "autocast_dtype": str(autocast_dtype(device))
                    if device.type == "cuda"
                    else None,
                    "compiled": compiled,
                    "preprocessing": "checkpoint fit-only statistics",
                },
                "artifacts": artifacts,
                "access_ledger": access,
                "official_validation_accessed": access["official_validation_accessed"],
                "test_accessed": access["test_accessed"],
            }
            write_json_atomic(staging / "score_manifest.json", manifest)
            staging.rename(output)
        return manifest
    finally:
        dataset.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-prelude", action="store_true")
    args = parser.parse_args()
    score(
        args.store,
        args.checkpoint,
        args.output,
        expected_sha256=args.checkpoint_sha256,
        parent_prelude=args.parent_prelude,
    )


if __name__ == "__main__":
    main()
