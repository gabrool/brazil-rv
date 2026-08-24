from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from brazil_rv.modeling.contract import GH200_RUNTIME, MAX_EPOCHS, VALIDATION_END
from brazil_rv.modeling.data import (
    create_training_loaders,
    di_tilt_sidecar_identity,
    feature_store_identity,
    load_sample_index,
    sample_window_metadata,
    select_training_window,
)
from brazil_rv.modeling.engine import (
    checkpoint_payload,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    objective_metadata,
    sam_metadata,
    train_one_epoch,
)
from brazil_rv.modeling.model import (
    RESIDUAL_AUXILIARY_VARIANT,
    build_model,
    count_trainable_parameters,
)
from brazil_rv.modeling.optim import build_optimizer, build_scheduler
from brazil_rv.modeling.provenance import build_run_provenance, repository_commit
from brazil_rv.modeling.trajectory import ModelEMA, temporarily_load_state


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _write_reference(path: Path, observations) -> None:
    _atomic_npz(
        path,
        {
            name: getattr(observations, name)
            for name in observations.__dataclass_fields__
            if name != "predictions"
        },
    )


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_fixed_residual(*, store: Path, sidecar: Path, run_dir: Path, seed: int) -> Path:
    if run_dir.exists():
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("seed") == seed
            and manifest.get("training", {}).get("official_monitoring") is False
            and manifest.get("split", {}).get("test_accessed") is False
        ):
            return run_dir
        raise ValueError(f"Existing residual run differs: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "validation_predictions").mkdir()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    sample_index = load_sample_index(store, through=VALIDATION_END)
    train_rows, validation_rows, selection_note = select_training_window(
        sample_index, "official"
    )
    identity = feature_store_identity(store)
    sidecar_identity = di_tilt_sidecar_identity(
        sidecar, identity, require_residual=True
    )
    train_loader, validation_loader, sampler = create_training_loaders(
        store, train_rows, validation_rows, GH200_RUNTIME, seed, sidecar
    )
    model = build_model(RESIDUAL_AUXILIARY_VARIANT).cuda()
    ema = ModelEMA(model, 0.995)
    objective = objective_metadata(True)
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, train_rows.height, MAX_EPOCHS
    )
    provenance = build_run_provenance(
        repository_commit_value=repository_commit(),
        feature_store=store,
        feature_store_metadata=identity,
        seed=seed,
        fit_window=sample_window_metadata(train_rows, "official_fit"),
        selection_window=sample_window_metadata(validation_rows, "official_read"),
        selection_note=selection_note,
        parameter_count=count_trainable_parameters(model),
        training_sample_count=train_rows.height,
        date_replacement=sampler.replace_dates,
        model_variant=RESIDUAL_AUXILIARY_VARIANT,
        objective=objective,
    )
    if (
        provenance["training"]["steps_per_epoch"],
        provenance["training"]["warmup_steps"],
    ) != (steps_per_epoch, warmup_steps):
        raise RuntimeError("Scheduler and provenance differ")
    worker_path = Path(__file__).resolve()
    manifest = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": provenance["repository_commit"],
        "fixed_fit_worker_sha256": _sha256(worker_path),
        "run_provenance": provenance,
        "feature_store": str(store.resolve()),
        "feature_store_identity": identity,
        "auxiliary_inputs": {
            "next_stage_sidecar": str(sidecar.resolve()),
            "next_stage_sidecar_manifest": sidecar_identity,
        },
        "split": {
            "training": "official",
            "selection": "final_only_official_read",
            "test_accessed": False,
        },
        "seed": seed,
        "model": provenance["model"],
        "objective": objective,
        "training": {
            **provenance["training"],
            "official_monitoring": False,
            "validation_evaluations": 1,
            "readout": "final_epoch_ema_0995",
        },
        "sam": sam_metadata(),
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    compiled_model = compile_model(model)
    compiled_objective = compile_training_objective(residual_auxiliary=True)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_started = time.perf_counter()
            sampler.set_epoch(epoch)
            result = train_one_epoch(
                compiled_model,
                train_loader,
                optimizer,
                scheduler,
                GH200_RUNTIME,
                compiled_objective,
                after_update=lambda: ema.update(model),
                residual_auxiliary=True,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "train_objective_loss": result["objective_loss"],
                    "optimizer_steps": result["optimizer_steps"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
            )
            _write_history(run_dir / "history.csv", rows)
        with temporarily_load_state(model, ema.shadow):
            observations, loss = collect_validation_observations(
                model, validation_loader
            )
        _write_reference(run_dir / "validation_reference.npz", observations)
        _atomic_npz(
            run_dir / "validation_predictions" / "epoch_20.npz",
            {"ema_0995": observations.predictions},
        )
        checkpoint = checkpoint_payload(
            model,
            {"ema_0995": ema.cpu_state_dict()},
            seed=seed,
            epoch=MAX_EPOCHS,
            validation_scores={},
            feature_store=store,
            run_provenance=provenance,
        )
        temporary = run_dir / "checkpoints" / "epoch_20.pt.tmp"
        torch.save(checkpoint, temporary)
        os.replace(temporary, run_dir / "checkpoints" / "epoch_20.pt")
        completed = {
            **manifest,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "epochs_completed": MAX_EPOCHS,
            "final_validation_objective_loss": loss,
            "total_run_seconds": time.perf_counter() - started,
        }
        _atomic_json(run_dir / "run_manifest.json", completed)
    except BaseException:
        _atomic_json(
            run_dir / "run_manifest.json",
            {
                **manifest,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "epochs_completed": len(rows),
                "total_run_seconds": time.perf_counter() - started,
            },
        )
        raise
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        run_fixed_residual(
            store=arguments.store,
            sidecar=arguments.sidecar,
            run_dir=arguments.run_dir,
            seed=arguments.seed,
        )
    )
