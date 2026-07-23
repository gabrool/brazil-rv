from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .contract import (
    CONTRACT_VERSION,
    MUON_COMPATIBILITY_CONTRACT_VERSION,
    PROJECT_ROOT,
    RUNTIME_PROFILES,
    RUNTIME_PROFILE_NAMES,
    RUN_OUTPUT_BASE,
    SANITY_DECISION_INDEX,
    SANITY_MAX_LOSS,
    SANITY_MAX_STEPS,
    SANITY_MIN_SPEARMAN,
    SANITY_SAMPLE_COUNT,
)
from .data import (
    create_training_loaders,
    resolve_feature_store,
    select_sample_split,
    validate_feature_store,
    warm_feature_store_cache,
)
from .engine import (
    _optimizer_update,
    _predict,
    _to_cuda,
    checkpoint_payload,
    compile_model,
    masked_huber_loss,
    validate_runtime_profile,
    warmup_compiled_model,
)
from .metrics import sample_level_ic
from .model import CrossAssetPatchITransformerV1, count_trainable_parameters
from .muon import PYTORCH_MUON_REFERENCE
from .optim import build_optimizers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=RUNTIME_PROFILE_NAMES)
    return parser.parse_args()


def _evaluate_fixed_batch(
    model: CrossAssetPatchITransformerV1,
    cpu_batch: dict[str, torch.Tensor],
) -> tuple[float, float, bool]:
    batch = _to_cuda(cpu_batch)
    valid_count = int(cpu_batch["sample_valid_mask"].sum())
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = _predict(model, batch)
        loss = masked_huber_loss(
            output[:valid_count],
            batch["targets"][:valid_count],
            batch["label_mask"][:valid_count],
        )
    predictions = output[:valid_count].float().cpu().numpy()
    targets = cpu_batch["targets"][:valid_count].numpy()
    masks = cpu_batch["label_mask"][:valid_count].numpy()
    spearman, _ = sample_level_ic(predictions, targets, masks)
    return (
        float(loss),
        float(np.nanmean(spearman)),
        bool(np.isfinite(predictions).all()),
    )


def _validate_compiled_checkpoint_compatibility(
    model: CrossAssetPatchITransformerV1,
    evaluation_batch: dict[str, torch.Tensor],
    muon_backend: str,
    profile_name: str,
    completed_steps: int,
    validation_score: float,
    feature_store: Path,
) -> tuple[bool, float]:
    model.eval()
    cuda_batch = _to_cuda(evaluation_batch)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        compiled_predictions = _predict(model, cuda_batch).float()
    git_commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT / "quant" / "b3-quant",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkpoint = checkpoint_payload(
        model,
        "full",
        "hybrid",
        muon_backend,
        profile_name,
        11,
        completed_steps,
        validation_score,
        feature_store,
        git_commit_sha,
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temporary:
        checkpoint_path = Path(temporary.name)
    try:
        torch.save(checkpoint, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        eager_model = CrossAssetPatchITransformerV1("full")
        eager_model.load_state_dict(loaded["model_state_dict"])
        eager_model.to("cuda").eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            eager_predictions = _predict(eager_model, cuda_batch).float()
    finally:
        checkpoint_path.unlink(missing_ok=True)
    if not bool(torch.isfinite(eager_predictions).all()):
        raise ValueError("Eager checkpoint restoration produced nonfinite predictions")
    maximum_difference = float(
        (compiled_predictions - eager_predictions).abs().max().cpu()
    )
    compatible = bool(
        torch.allclose(
            compiled_predictions,
            eager_predictions,
            atol=5e-3,
            rtol=5e-3,
        )
    )
    return compatible, maximum_difference


def main() -> None:
    args = parse_args()
    profile = RUNTIME_PROFILES[args.profile]
    hardware = validate_runtime_profile(profile)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    np.random.seed(11)

    feature_store = resolve_feature_store()
    sample_index = validate_feature_store(feature_store)
    training = select_sample_split(sample_index, "train")
    fixed_rows = (
        training.filter(training.get_column("decision_idx") == SANITY_DECISION_INDEX)
        .sort("trade_date")
        .head(SANITY_SAMPLE_COUNT)
    )
    if (
        fixed_rows.height != SANITY_SAMPLE_COUNT
        or fixed_rows.get_column("trade_date").n_unique() != SANITY_SAMPLE_COUNT
    ):
        raise ValueError("Sanity samples must cover 32 distinct training dates")
    cache_report = warm_feature_store_cache(feature_store)
    train_loader, evaluation_loader, sampler = create_training_loaders(
        feature_store, fixed_rows, fixed_rows, profile, 11
    )
    sampler.set_epoch(0)
    microbatches = list(train_loader)
    evaluation_batch = next(iter(evaluation_loader))

    model = CrossAssetPatchITransformerV1("full").to("cuda")
    model.eval()
    optimizers, _, muon_backend = build_optimizers(model, "hybrid")
    compile_model(model, profile)
    compile_report = warmup_compiled_model(
        model,
        microbatches[0],
        evaluation_batch,
        training_mode=False,
    )
    torch.cuda.reset_peak_memory_stats()
    all_gradients_finite = True
    all_predictions_finite = True
    final_loss = float("inf")
    final_spearman = float("-inf")
    completed_steps = 0

    for step in range(1, SANITY_MAX_STEPS + 1):
        valid_samples = sum(
            int(batch["label_mask"].any(dim=(1, 2)).sum()) for batch in microbatches
        )
        for cpu_batch in microbatches:
            batch = _to_cuda(cpu_batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = _predict(model, batch)
                sample_loss = masked_huber_loss(
                    predictions, batch["targets"], batch["label_mask"]
                )
            microbatch_valid_samples = int(
                cpu_batch["label_mask"].any(dim=(1, 2)).sum()
            )
            (sample_loss * microbatch_valid_samples / valid_samples).backward()
            all_predictions_finite &= bool(torch.isfinite(predictions).all())
        gradients_finite, _ = _optimizer_update(model, optimizers, {})
        all_gradients_finite &= gradients_finite
        if not gradients_finite:
            break
        completed_steps = step
        if step == 1 or step % 10 == 0:
            final_loss, final_spearman, predictions_finite = _evaluate_fixed_batch(
                model, evaluation_batch
            )
            all_predictions_finite &= predictions_finite
            if (
                final_loss < SANITY_MAX_LOSS
                and final_spearman > SANITY_MIN_SPEARMAN
                and all_predictions_finite
            ):
                break

    if completed_steps % 10:
        final_loss, final_spearman, predictions_finite = _evaluate_fixed_batch(
            model, evaluation_batch
        )
        all_predictions_finite &= predictions_finite
    memorization_succeeded = (
        final_loss < SANITY_MAX_LOSS
        and final_spearman > SANITY_MIN_SPEARMAN
        and all_gradients_finite
        and all_predictions_finite
    )
    compiled_checkpoint_eager_compatible = False
    compiled_checkpoint_max_absolute_difference: float | None = None
    if memorization_succeeded:
        (
            compiled_checkpoint_eager_compatible,
            compiled_checkpoint_max_absolute_difference,
        ) = _validate_compiled_checkpoint_compatibility(
            model,
            evaluation_batch,
            muon_backend,
            profile.name,
            completed_steps,
            final_spearman,
            feature_store,
        )

    peak_allocated = max(
        torch.cuda.max_memory_allocated(),
        compile_report.peak_allocated_cuda_memory_bytes,
    )
    peak_reserved = max(
        torch.cuda.max_memory_reserved(),
        compile_report.peak_reserved_cuda_memory_bytes,
    )
    passed = (
        memorization_succeeded
        and compiled_checkpoint_eager_compatible
        and peak_allocated < 0.8 * hardware.total_vram_bytes
        and peak_reserved < 0.9 * hardware.total_vram_bytes
    )
    created_at = datetime.now(timezone.utc)
    output_dir = RUN_OUTPUT_BASE / (
        "sanity_cross_asset_patch_itransformer_v1_"
        f"{profile.name}_{created_at:%Y%m%dT%H%M%S%fZ}"
    )
    if output_dir.exists():
        raise FileExistsError(f"Sanity output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    report = {
        "contract_version": CONTRACT_VERSION,
        "muon_compatibility_contract_version": (MUON_COMPATIBILITY_CONTRACT_VERSION),
        "muon_backend": muon_backend,
        "muon_reference": dict(PYTORCH_MUON_REFERENCE),
        "created_at_utc": created_at.isoformat(),
        "resolved_feature_store_path": str(feature_store),
        "profile": profile.name,
        "hardware": asdict(hardware),
        "compile_settings": {
            "api": "nn.Module.compile",
            "backend": profile.compile_backend,
            "mode": profile.compile_mode,
            "fullgraph": profile.compile_fullgraph,
            "dynamic": profile.compile_dynamic,
            "backward_pass_autocast": "off",
        },
        "compile_warmup": asdict(compile_report),
        "feature_cache_warmup": asdict(cache_report),
        "physical_microbatch_size": profile.microbatch_size,
        "accumulation_steps": profile.accumulation_steps,
        "sample_count": SANITY_SAMPLE_COUNT,
        "decision_index": SANITY_DECISION_INDEX,
        "optimizer_steps": completed_steps,
        "final_masked_huber_loss": final_loss,
        "mean_valid_sample_spearman_ic": final_spearman,
        "all_gradients_finite": all_gradients_finite,
        "all_predictions_finite": all_predictions_finite,
        "compiled_checkpoint_eager_compatible": (compiled_checkpoint_eager_compatible),
        "compiled_checkpoint_max_absolute_difference": (
            compiled_checkpoint_max_absolute_difference
        ),
        "peak_allocated_cuda_memory_bytes": peak_allocated,
        "peak_reserved_cuda_memory_bytes": peak_reserved,
        "parameter_count": count_trainable_parameters(model),
        "passed": passed,
    }
    (output_dir / "sanity_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    if not passed:
        raise RuntimeError(
            "Cloud sanity criteria failed: "
            f"loss={final_loss}, spearman={final_spearman}, "
            f"finite_gradients={all_gradients_finite}, "
            f"finite_predictions={all_predictions_finite}, "
            f"compiled_checkpoint_eager_compatible="
            f"{compiled_checkpoint_eager_compatible}, "
            f"peak_allocated={peak_allocated}, peak_reserved={peak_reserved}"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
