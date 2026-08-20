from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from .contract import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    EARLY_STOP_PATIENCE,
    EMA_DECAYS,
    FINAL_LR_FACTOR,
    GH200_RUNTIME,
    GRADIENT_CLIP,
    MAX_EPOCHS,
    MIN_IC_IMPROVEMENT,
    SAM_RHO,
    SOFT_RANK_TEMPERATURE,
    TCN_ARCHITECTURE,
    WARMUP_FRACTION,
    RuntimeSettings,
)
from .model import auxiliary_model_metadata
from .optim import scheduler_step_contract

RUN_PROVENANCE_SCHEMA = "PIT_CLEAN_TCN_TRAJECTORY"


def repository_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[4],
    ).stdout.strip()


def model_metadata(auxiliary_variant: str | None = None) -> dict[str, object]:
    metadata = json.loads(
        json.dumps(
            {
                "model_name": "tcn",
                "architecture": asdict(TCN_ARCHITECTURE),
                "cross_equity_attention": False,
            }
        )
    )
    if auxiliary_variant is not None:
        metadata["auxiliary"] = auxiliary_model_metadata(auxiliary_variant)
    return metadata


def training_contract(
    training_sample_count: int,
    date_replacement: bool,
    *,
    runtime: RuntimeSettings = GH200_RUNTIME,
    objective: dict[str, object] | None = None,
) -> dict[str, object]:
    steps_per_epoch, warmup_steps = scheduler_step_contract(
        training_sample_count,
        MAX_EPOCHS,
        runtime.effective_batch_size,
    )
    return {
        "epochs": MAX_EPOCHS,
        "fixed_trajectory": True,
        "raw_checkpoint_each_epoch": True,
        "ema_decays": list(EMA_DECAYS),
        "patience3_raw": {
            "patience": EARLY_STOP_PATIENCE,
            "minimum_ic_improvement": MIN_IC_IMPROVEMENT,
            "restores": "best_raw_checkpoint",
        },
        "effective_batch_size": runtime.effective_batch_size,
        "loader_batch_size": runtime.loader_batch_size,
        "microbatch_size": runtime.microbatch_size,
        "date_replacement": date_replacement,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "scheduler_warmup_fraction": WARMUP_FRACTION,
        "scheduler_final_lr_factor": FINAL_LR_FACTOR,
        "learning_rate": ADAMW_LR,
        "adamw_betas": list(ADAMW_BETAS),
        "adamw_epsilon": ADAMW_EPS,
        "adamw_weight_decay": ADAMW_WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "objective": objective
        or {"name": "soft_spearman", "temperature": SOFT_RANK_TEMPERATURE},
        "sam": {"rho": SAM_RHO, "base_optimizer": "adamw"},
    }


def build_run_provenance(
    *,
    repository_commit_value: str,
    feature_store: Path,
    feature_store_metadata: dict[str, object],
    seed: int,
    fit_window: dict[str, object],
    selection_window: dict[str, object],
    selection_note: str,
    parameter_count: int,
    training_sample_count: int,
    date_replacement: bool,
    runtime: RuntimeSettings = GH200_RUNTIME,
    auxiliary_variant: str | None = None,
    auxiliary_target_identity: dict[str, object] | None = None,
    objective: dict[str, object] | None = None,
) -> dict[str, object]:
    provenance = {
        "schema": RUN_PROVENANCE_SCHEMA,
        "repository_commit": repository_commit_value,
        "feature_store": str(feature_store.resolve()),
        "feature_store_identity": feature_store_metadata,
        "model": model_metadata(auxiliary_variant),
        "seed": seed,
        "fit_window": fit_window,
        "selection_window": selection_window,
        "selection_note": selection_note,
        "test_accessed": False,
        "parameter_count": parameter_count,
        "training": training_contract(
            training_sample_count,
            date_replacement,
            runtime=runtime,
            objective=objective,
        ),
        **(
            {"auxiliary_target_identity": auxiliary_target_identity}
            if auxiliary_target_identity is not None
            else {}
        ),
    }
    return json.loads(json.dumps(provenance))
