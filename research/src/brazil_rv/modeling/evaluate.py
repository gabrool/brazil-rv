from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch

from .contract import (
    GH200_RUNTIME,
    TCNArchitecture,
    TCNSettings,
    architecture_for_model,
)
from .data import create_evaluation_loader, load_sample_index, select_sample_split
from .engine import EvaluationObservations, collect_evaluation_observations
from .model import build_neural_model

REQUIRED_CHECKPOINT_KEYS = {
    "model_name",
    "architecture",
    "tcn_settings",
    "peer_features",
    "optimizer_variant",
    "objective",
    "sam",
    "seed",
    "epoch",
    "validation_score",
    "feature_store",
    "global_context",
    "model_state_dict",
}


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one current completed run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    return parser.parse_args(arguments)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def load_current_neural_run(
    run_dir: Path,
) -> tuple[torch.nn.Module, dict[str, object], Path]:
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    model_name = str(checkpoint["model_name"])
    settings_value = checkpoint["tcn_settings"]
    settings = TCNSettings(**settings_value) if settings_value is not None else None
    architecture = architecture_for_model(model_name, settings)
    checkpoint["tcn_settings"] = None if settings is None else asdict(settings)
    checkpoint["architecture"] = asdict(architecture)
    checkpoint.setdefault("training_horizon", "all")
    checkpoint.setdefault("selection_horizon", checkpoint["training_horizon"])
    checkpoint.setdefault("context_family_ablation", "none")
    peer_value = checkpoint["peer_features"]
    if not isinstance(peer_value, dict) or peer_value.get("mode") not in (
        "none",
        "selected",
    ):
        raise ValueError("Checkpoint has invalid current peer metadata")
    peer_features = str(peer_value["mode"])
    model = build_neural_model(
        model_name,
        architecture if isinstance(architecture, TCNArchitecture) else None,
        peer_features,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    store = Path(str(checkpoint["feature_store"]))
    if not store.is_dir():
        raise FileNotFoundError(store)
    return model, checkpoint, store


def collect_neural_evaluation(
    run_dir: Path,
    split: str,
) -> tuple[
    EvaluationObservations,
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    Path,
]:
    model, checkpoint, store = load_current_neural_run(run_dir)
    model = model.cuda()
    rows = select_sample_split(load_sample_index(store), split)
    settings_value = checkpoint["tcn_settings"]
    settings = TCNSettings(**settings_value) if settings_value is not None else None
    architecture = architecture_for_model(str(checkpoint["model_name"]), settings)
    objective = checkpoint["objective"]
    if not isinstance(objective, dict) or objective.get("name") not in (
        "soft_spearman",
        "rank_huber",
    ):
        raise ValueError("Checkpoint has invalid objective metadata")
    peer = checkpoint["peer_features"]
    loader = create_evaluation_loader(
        store,
        rows,
        str(checkpoint["model_name"]),
        checkpoint["global_context"],
        GH200_RUNTIME,
        int(checkpoint["seed"]),
        architecture if isinstance(architecture, TCNArchitecture) else None,
        str(peer["mode"]),
        str(checkpoint["context_family_ablation"]),
    )
    observations, summary, daily = collect_evaluation_observations(
        model,
        loader,
        str(objective["name"]),
        objective.get("temperature"),
    )
    return observations, summary, daily, checkpoint, store


def evaluate_run(run_dir: Path, split: str) -> Path:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Run is not completed")
    output = (
        run_dir / f"evaluation_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    )
    output.mkdir(exist_ok=False)
    observations, summary, daily, _, _ = collect_neural_evaluation(run_dir, split)
    pl.DataFrame(
        {
            "sample_id": observations.sample_id,
            "date_idx": observations.date_idx,
            "decision_idx": observations.decision_idx,
        }
    ).write_parquet(output / "sample_index.parquet")
    _atomic_json(output / "metrics.json", summary)
    pl.DataFrame(daily).write_parquet(output / "daily_metrics.parquet")
    _atomic_json(
        output / "evaluation_manifest.json",
        {"run_dir": str(run_dir), "split": split, "test_accessed": split == "test"},
    )
    return output


def main() -> None:
    args = parse_args()
    print(evaluate_run(args.run_dir, args.split))


if __name__ == "__main__":
    main()
