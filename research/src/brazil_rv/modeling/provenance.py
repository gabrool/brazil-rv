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
    FINAL_LR_FACTOR,
    GH200_RUNTIME,
    GRADIENT_CLIP,
    MAX_EPOCHS,
    MIN_IC_IMPROVEMENT,
    SAM_RHO,
    SOFT_RANK_TEMPERATURE,
    TCN_ARCHITECTURE,
    TCN_ATTENTION_HEADS,
    WARMUP_FRACTION,
    RuntimeSettings,
)
from .optim import scheduler_step_contract

RUN_PROVENANCE_SCHEMA = "PIT_CLEAN_TCN_RUN"


def repository_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def model_metadata(cross_equity_attention: bool) -> dict[str, object]:
    return {
        "model_name": "tcn",
        "architecture": asdict(TCN_ARCHITECTURE),
        "cross_equity_attention": cross_equity_attention,
        "attention": (
            {
                "position": "final_equity_state_before_context_pooled_fusion",
                "heads": TCN_ATTENTION_HEADS,
                "pre_norm": True,
                "output_projection_zero_initialized": True,
                "security_or_classification_embeddings": False,
            }
            if cross_equity_attention
            else None
        ),
    }


def training_contract(
    training_sample_count: int,
    recency: dict[str, object],
    date_replacement: bool,
    *,
    runtime: RuntimeSettings = GH200_RUNTIME,
) -> dict[str, object]:
    steps_per_epoch, warmup_steps = scheduler_step_contract(
        training_sample_count,
        MAX_EPOCHS,
        runtime.effective_batch_size,
    )
    return {
        "maximum_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "minimum_ic_improvement": MIN_IC_IMPROVEMENT,
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
        "objective": {
            "name": "soft_spearman",
            "temperature": SOFT_RANK_TEMPERATURE,
        },
        "sam": {"rho": SAM_RHO, "base_optimizer": "adamw"},
        "recency": recency,
    }


def build_run_provenance(
    *,
    repository_commit_value: str,
    feature_store: Path,
    feature_store_metadata: dict[str, object],
    cross_equity_attention: bool,
    seed: int,
    recency: dict[str, object],
    fit_window: dict[str, object],
    selection_window: dict[str, object],
    parameter_count: int,
    training_sample_count: int,
    date_replacement: bool,
    runtime: RuntimeSettings = GH200_RUNTIME,
) -> dict[str, object]:
    provenance = {
        "schema": RUN_PROVENANCE_SCHEMA,
        "repository_commit": repository_commit_value,
        "feature_store": str(feature_store.resolve()),
        "feature_store_identity": feature_store_metadata,
        "model": model_metadata(cross_equity_attention),
        "seed": seed,
        "fit_window": fit_window,
        "selection_window": selection_window,
        "test_accessed": False,
        "parameter_count": parameter_count,
        "training": training_contract(
            training_sample_count,
            recency,
            date_replacement,
            runtime=runtime,
        ),
    }
    return json.loads(json.dumps(provenance))
