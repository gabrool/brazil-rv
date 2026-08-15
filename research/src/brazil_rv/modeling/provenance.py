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
    WARMUP_FRACTION,
    NeuralArchitecture,
    RuntimeSettings,
    TCNArchitecture,
    TCNSettings,
    context_routing_metadata,
    peer_feature_metadata,
)
from .optim import scheduler_step_contract


RUN_PROVENANCE_SCHEMA = "STAGE_RUN_PROVENANCE_V1"


def repository_commit() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def model_metadata(
    model_name: str,
    architecture: NeuralArchitecture,
    settings: TCNSettings | None,
    peer_features: str = "none",
) -> dict[str, object]:
    metadata = {
        "model_name": model_name,
        "architecture": asdict(architecture),
        "tcn_settings": None if settings is None else asdict(settings),
        "context_routing": (
            context_routing_metadata(architecture)
            if isinstance(architecture, TCNArchitecture)
            else None
        ),
        "peer_features": peer_feature_metadata(model_name, architecture, peer_features),
        "readout": settings.readout if settings is not None else None,
    }
    return json.loads(json.dumps(metadata))


def training_contract(
    training_sample_count: int,
    allow_date_replacement: bool,
    *,
    maximum_epochs: int = MAX_EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    runtime: RuntimeSettings = GH200_RUNTIME,
) -> dict[str, object]:
    steps_per_epoch, warmup_steps = scheduler_step_contract(
        training_sample_count,
        maximum_epochs,
        runtime.effective_batch_size,
    )
    return {
        "allow_date_replacement": allow_date_replacement,
        "maximum_epochs": maximum_epochs,
        "early_stop_patience": early_stop_patience,
        "minimum_ic_improvement": MIN_IC_IMPROVEMENT,
        "effective_batch_size": runtime.effective_batch_size,
        "loader_batch_size": runtime.loader_batch_size,
        "microbatch_size": runtime.microbatch_size,
        "loader_batches_per_effective_batch": (
            runtime.loader_batches_per_effective_batch
        ),
        "microbatches_per_effective_batch": runtime.microbatches_per_effective_batch,
        "evaluation_batch_size": runtime.evaluation_batch_size,
        "num_workers": runtime.num_workers,
        "prefetch_factor": runtime.prefetch_factor,
        "pin_memory": True,
        "persistent_workers": runtime.num_workers > 0,
        "compile_backend": runtime.compile_backend,
        "compile_mode": runtime.compile_mode,
        "compile_fullgraph": runtime.compile_fullgraph,
        "compile_dynamic": runtime.compile_dynamic,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup_steps,
        "scheduler_warmup_fraction": WARMUP_FRACTION,
        "scheduler_final_lr_factor": FINAL_LR_FACTOR,
        "learning_rate": ADAMW_LR,
        "adamw_betas": list(ADAMW_BETAS),
        "adamw_epsilon": ADAMW_EPS,
        "adamw_weight_decay": ADAMW_WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
    }


def build_run_provenance(
    *,
    repository_commit_value: str,
    feature_store: Path,
    feature_store_metadata: dict[str, object],
    model_name: str,
    architecture: NeuralArchitecture,
    settings: TCNSettings | None,
    peer_features: str,
    global_context: str | None,
    objective: dict[str, object],
    optimizer: str,
    sam: dict[str, object],
    seed: int,
    training_horizon: str,
    selection_horizon: str,
    context_family_ablation: str,
    fit_window: dict[str, object],
    selection_window: dict[str, object],
    allow_date_replacement: bool,
    parameter_count: int,
    training_sample_count: int,
    maximum_epochs: int = MAX_EPOCHS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    runtime: RuntimeSettings = GH200_RUNTIME,
) -> dict[str, object]:
    provenance = {
        "schema": RUN_PROVENANCE_SCHEMA,
        "repository_commit": repository_commit_value,
        "feature_store": str(feature_store.resolve()),
        "feature_store_identity": feature_store_metadata,
        "model": model_metadata(model_name, architecture, settings, peer_features),
        "global_context": global_context,
        "objective": objective,
        "optimizer": optimizer,
        "sam": sam,
        "seed": seed,
        "training_horizon": training_horizon,
        "selection_horizon": selection_horizon,
        "context_family_ablation": context_family_ablation,
        "fit_window": fit_window,
        "selection_window": selection_window,
        "parameter_count": parameter_count,
        "training": training_contract(
            training_sample_count,
            allow_date_replacement,
            maximum_epochs=maximum_epochs,
            early_stop_patience=early_stop_patience,
            runtime=runtime,
        ),
    }
    return json.loads(json.dumps(provenance))
