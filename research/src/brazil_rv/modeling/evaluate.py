from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .data import (
    FeatureStoreIdentityCache,
    create_evaluation_loader,
    load_sample_index,
    select_sample_split,
    validate_feature_store_identity,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_validation_observations,
    summarize_evaluation_observations,
)
from .model import build_model
from .trajectory import model_state_dicts_for_rule


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen rule from one completed official run"
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    return parser.parse_args(arguments)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_observations(path: Path, observations: EvaluationObservations) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(
            output,
            **{
                name: getattr(observations, name)
                for name in EvaluationObservations.__dataclass_fields__
            },
        )
    os.replace(temporary, path)


def load_current_run(
    run_dir: Path,
    *,
    identity_cache: FeatureStoreIdentityCache | None = None,
) -> tuple[
    tuple[dict[str, torch.Tensor], ...],
    dict[str, object],
    Path,
    str,
]:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Run is not completed")
    if manifest.get("split", {}).get("training") != "official":
        raise ValueError("Only an official-window run can be evaluated externally")
    frozen = manifest.get("frozen_selection")
    if not isinstance(frozen, dict) or not isinstance(
        frozen.get("selected_rule"), str
    ):
        raise ValueError("Run does not contain an internally frozen selection rule")
    store = Path(str(manifest["feature_store"])).resolve()
    if not store.is_dir():
        raise FileNotFoundError(store)
    validate_feature_store_identity(
        store,
        manifest["feature_store_identity"],
        identity_cache=identity_cache,
    )
    rule = str(frozen["selected_rule"])
    states = model_state_dicts_for_rule(run_dir, rule)
    return states, manifest, store, rule


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
    str,
]:
    torch.set_float32_matmul_precision("high")
    states, manifest, store, rule = load_current_run(
        run_dir, identity_cache=identity_cache
    )
    rows = select_sample_split(load_sample_index(store), split)
    loader = create_evaluation_loader(store, rows, seed=int(manifest["seed"]))
    model = build_model().cuda()
    reference: EvaluationObservations | None = None
    predictions = []
    losses = []
    for state in states:
        model.load_state_dict(state, strict=True)
        observations, loss = collect_validation_observations(model, loader)
        if reference is None:
            reference = observations
        else:
            assert_observations_aligned(reference, observations)
        predictions.append(observations.predictions)
        losses.append(loss)
    assert reference is not None
    combined = replace(
        reference,
        predictions=np.mean(
            np.stack(predictions), axis=0, dtype=np.float64
        ).astype(np.float32),
    )
    summary, daily = summarize_evaluation_observations(
        combined, losses[0] if len(losses) == 1 else None
    )
    return combined, summary, daily, manifest, store, rule


def evaluate_run(run_dir: Path, split: str) -> Path:
    observations, summary, daily, _, _, rule = collect_run_evaluation(run_dir, split)
    output = (
        run_dir / f"evaluation_{split}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    )
    output.mkdir(exist_ok=False)
    _write_observations(output / "observations.npz", observations)
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
        {
            "run_dir": str(run_dir.resolve()),
            "split": split,
            "selected_rule": rule,
            "test_accessed": split == "test",
            "campaign_driver": False,
        },
    )
    return output


def main() -> None:
    args = parse_args()
    print(evaluate_run(args.run_dir, args.split))


if __name__ == "__main__":
    main()
