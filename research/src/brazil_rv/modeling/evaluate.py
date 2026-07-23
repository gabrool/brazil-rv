from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch

from .contract import (
    CLOUD_RUNTIME_CONTRACT_VERSION,
    CONTRACT_VERSION,
    MUON_COMPATIBILITY_CONTRACT_VERSION,
    RUNTIME_PROFILES,
    RUNTIME_PROFILE_NAMES,
)
from .data import (
    create_evaluation_loader,
    select_sample_split,
    validate_feature_store,
    warm_feature_store_cache,
)
from .engine import (
    compile_model,
    evaluate_model,
    validate_runtime_profile,
    warmup_compiled_evaluation,
)
from .model import CrossAssetPatchITransformerV1
from .muon import PYTORCH_MUON_REFERENCE
from .optim import OFFICIAL_MUON_BACKEND, REFERENCE_MUON_BACKEND

_CHECKPOINT_IDENTITY_FIELDS = (
    "contract_version",
    "cloud_runtime_contract_version",
    "muon_compatibility_contract_version",
    "muon_backend",
    "muon_reference",
    "model_variant",
    "optimizer_variant",
    "seed",
    "runtime_profile",
    "resolved_feature_store_path",
    "git_commit_sha",
    "architecture_constants",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=RUNTIME_PROFILE_NAMES)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    return parser.parse_args()


def _validate_muon_identity_values(identity: dict[str, object]) -> None:
    if (
        identity["muon_compatibility_contract_version"]
        != MUON_COMPATIBILITY_CONTRACT_VERSION
    ):
        raise ValueError(
            "Invalid Muon identity field: muon_compatibility_contract_version"
        )
    if identity["muon_reference"] != dict(PYTORCH_MUON_REFERENCE):
        raise ValueError("Invalid Muon identity field: muon_reference")

    optimizer_variant = identity["optimizer_variant"]
    if optimizer_variant not in ("hybrid", "adamw"):
        raise ValueError("Invalid Muon identity field: optimizer_variant")
    muon_backend = identity["muon_backend"]
    if optimizer_variant == "hybrid" and muon_backend not in (
        OFFICIAL_MUON_BACKEND,
        REFERENCE_MUON_BACKEND,
    ):
        raise ValueError("Invalid Muon identity field: muon_backend")
    if optimizer_variant == "adamw" and muon_backend is not None:
        raise ValueError("Invalid Muon identity field: muon_backend")


def _validate_run_checkpoint_identity(
    manifest: dict[str, object],
    checkpoint: dict[str, object],
    feature_store: Path,
) -> None:
    for field in _CHECKPOINT_IDENTITY_FIELDS:
        if field not in manifest or field not in checkpoint:
            raise ValueError(f"Missing run/checkpoint identity field: {field}")
        if manifest[field] != checkpoint[field]:
            raise ValueError(f"Run/checkpoint identity mismatch: {field}")
    _validate_muon_identity_values(manifest)
    manifest_store = Path(str(manifest["resolved_feature_store_path"])).expanduser()
    if manifest_store.resolve() != feature_store:
        raise ValueError("Validated feature store does not match the run identity")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_parquet(path: Path, frame: pl.DataFrame) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    profile = RUNTIME_PROFILES[args.profile]
    hardware = validate_runtime_profile(profile)
    torch.set_float32_matmul_precision("high")

    manifest = json.loads(
        (args.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "completed":
        raise ValueError("Standalone evaluation requires a completed run")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Run manifest model contract version does not match")
    if manifest.get("cloud_runtime_contract_version") != CLOUD_RUNTIME_CONTRACT_VERSION:
        raise ValueError("Run manifest cloud runtime contract version does not match")
    feature_store = (
        Path(str(manifest["resolved_feature_store_path"])).expanduser().resolve()
    )
    sample_index = validate_feature_store(feature_store)
    rows = select_sample_split(sample_index, args.split)
    checkpoint = torch.load(
        args.run_dir / "best.pt", map_location="cpu", weights_only=False
    )
    _validate_run_checkpoint_identity(manifest, checkpoint, feature_store)

    cache_report = warm_feature_store_cache(feature_store)
    loader = create_evaluation_loader(
        feature_store, rows, profile, int(manifest["seed"])
    )
    model = CrossAssetPatchITransformerV1(str(checkpoint["model_variant"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cuda")
    compile_model(model, profile)
    (
        evaluation_pass_seconds,
        evaluation_steady_state_median_seconds,
        compile_peak_allocated,
        compile_peak_reserved,
    ) = warmup_compiled_evaluation(model, next(iter(loader)))
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    summary, daily_rows = evaluate_model(model, loader)
    torch.cuda.synchronize()
    evaluation_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    created_at = datetime.now(timezone.utc)
    evaluation_dir = (
        args.run_dir
        / "evaluations"
        / (f"{args.split}_{profile.name}_{created_at:%Y%m%dT%H%M%S%fZ}")
    )
    if evaluation_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {evaluation_dir}")
    evaluation_dir.mkdir(parents=True)
    _atomic_write_json(evaluation_dir / "metrics.json", summary)
    dates = dict(
        pl.read_parquet(feature_store / "date_index.parquet")
        .select("date_idx", "trade_date")
        .iter_rows()
    )
    daily_metrics = pl.DataFrame(
        [{"trade_date": dates[int(row["date_idx"])], **row} for row in daily_rows]
    )
    _atomic_write_parquet(evaluation_dir / "daily_metrics.parquet", daily_metrics)
    evaluation_manifest = {
        "created_at_utc": created_at.isoformat(),
        "split": args.split,
        "training_runtime_profile": manifest["runtime_profile"],
        "evaluation_runtime_profile": profile.name,
        "hardware": asdict(hardware),
        "feature_cache_warmup": asdict(cache_report),
        "compile": {
            "enabled": True,
            "api": "nn.Module.compile",
            "backend": profile.compile_backend,
            "mode": profile.compile_mode,
            "fullgraph": profile.compile_fullgraph,
            "dynamic": profile.compile_dynamic,
            "backward_pass_autocast": "off",
            "eager_fallback_allowed": False,
            "evaluation_pass_seconds": evaluation_pass_seconds,
            "evaluation_steady_state_median_seconds": (
                evaluation_steady_state_median_seconds
            ),
            "peak_allocated_cuda_memory_bytes": compile_peak_allocated,
            "peak_reserved_cuda_memory_bytes": compile_peak_reserved,
        },
        "evaluation_seconds": evaluation_seconds,
        "peak_allocated_cuda_memory_bytes": peak_allocated,
        "peak_reserved_cuda_memory_bytes": peak_reserved,
    }
    _atomic_write_json(evaluation_dir / "evaluation_manifest.json", evaluation_manifest)
    print(f"Evaluated {args.split}: {evaluation_dir}")


if __name__ == "__main__":
    main()
