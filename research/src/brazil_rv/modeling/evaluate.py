from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import torch

from .data import (
    FeatureStoreIdentityCache,
    create_evaluation_loader,
    load_sample_index,
    select_sample_split,
    validate_feature_store_identity,
)
from .engine import EvaluationObservations, collect_evaluation_observations
from .model import build_model

REQUIRED_CHECKPOINT_KEYS = {
    "model",
    "cross_equity_attention",
    "recency_policy",
    "seed",
    "epoch",
    "validation_score",
    "feature_store",
    "feature_store_identity",
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


def load_current_run(
    run_dir: Path,
    *,
    identity_cache: FeatureStoreIdentityCache | None = None,
) -> tuple[torch.nn.Module, dict[str, object], Path]:
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(
        run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing current keys: {sorted(missing)}")
    store = Path(str(checkpoint["feature_store"])).resolve()
    if not store.is_dir():
        raise FileNotFoundError(store)
    validate_feature_store_identity(
        store,
        checkpoint["feature_store_identity"],
        identity_cache=identity_cache,
    )
    model = build_model(
        cross_equity_attention=bool(checkpoint["cross_equity_attention"])
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint, store


def collect_run_evaluation(
    run_dir: Path,
    split: str,
    *,
    identity_cache: FeatureStoreIdentityCache | None = None,
) -> tuple[
    EvaluationObservations,
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    Path,
]:
    model, checkpoint, store = load_current_run(run_dir, identity_cache=identity_cache)
    model = model.cuda()
    rows = select_sample_split(load_sample_index(store), split)
    loader = create_evaluation_loader(store, rows, seed=int(checkpoint["seed"]))
    observations, summary, daily = collect_evaluation_observations(model, loader)
    return observations, summary, daily, checkpoint, store


def evaluate_run(run_dir: Path, split: str) -> Path:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Run is not completed")
    output = (
        run_dir / f"evaluation_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    )
    output.mkdir(exist_ok=False)
    observations, summary, daily, _, _ = collect_run_evaluation(run_dir, split)
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
