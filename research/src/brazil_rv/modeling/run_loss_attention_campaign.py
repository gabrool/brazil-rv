from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from brazil_rv.preprocessing.target_scale import build_target_scale_sidecar

from .contract import VALIDATION_END
from .data import (
    feature_store_identity,
    load_sample_index,
    select_sample_split,
    target_scale_identity,
)
from .engine import objective_metadata
from .provenance import repository_commit
from .train import run_training

CAMPAIGN_SCHEMA = "HYBRID_LOSS_RESIDUAL_ATTENTION_CAMPAIGN"
SOURCE_CAMPAIGN_SCHEMA = "PIT_CLEAN_CORE_CAMPAIGN"
SEED = 11


@dataclass(frozen=True)
class RunSpec:
    arm: str
    cross_equity_attention: bool


def expand_campaign_specs() -> tuple[RunSpec, ...]:
    return (
        RunSpec("hybrid_base", False),
        RunSpec("hybrid_residual_attention", True),
    )


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(run_dir: Path) -> float:
    manifest = _load_manifest(run_dir / "run_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"Run is incomplete: {run_dir}")
    return float(manifest["best_validation_score"])


def _source_state(source_campaign_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    manifest = _load_manifest(source_campaign_dir / "campaign_manifest.json")
    if (
        manifest.get("schema") != SOURCE_CAMPAIGN_SCHEMA
        or manifest.get("test_accessed") is not False
        or not isinstance(manifest.get("full_tod_store"), dict)
    ):
        raise ValueError("Source campaign does not contain a valid development store")
    store_identity = manifest["full_tod_store"]
    store = Path(str(store_identity["path"])).resolve()
    if feature_store_identity(store) != store_identity:
        raise ValueError("Source campaign full-TOD store identity no longer matches")
    if not (store / "equity_tod_profile.json").is_file() or any(
        (store / name).exists()
        for name in ("equity_peer_features.npy", "equity_peer_valid.npy")
    ):
        raise ValueError("Source campaign store is not the PIT-clean full-TOD store")

    baseline: Path | None = None
    arm_dir = source_campaign_dir / "runs" / "tod_uniform" / f"seed_{SEED}"
    for attempt in sorted(arm_dir.glob("attempt_*"), reverse=True):
        path = attempt / "run_manifest.json"
        if not path.is_file():
            continue
        run = _load_manifest(path)
        objective = run.get("objective")
        if (
            run.get("status") == "completed"
            and run.get("seed") == SEED
            and run.get("recency_policy") == "uniform"
            and run.get("cross_equity_attention") is False
            and run.get("split", {}).get("test_accessed") is False
            and Path(str(run.get("feature_store"))).resolve() == store
            and isinstance(objective, dict)
            and objective.get("name") == "soft_spearman"
        ):
            baseline = attempt
            break
    if baseline is None:
        raise ValueError("Completed TOD soft-loss seed-11 control was not found")
    return store, baseline, store_identity


def _load_or_create_manifest(
    output_dir: Path,
    source_campaign_dir: Path,
    store_identity: dict[str, object],
    baseline: Path,
) -> dict[str, object]:
    path = output_dir / "campaign_manifest.json"
    if path.exists():
        manifest = _load_manifest(path)
        if (
            manifest.get("schema") != CAMPAIGN_SCHEMA
            or manifest.get("repository_sha") != repository_commit()
            or Path(str(manifest.get("source_campaign_dir"))).resolve()
            != source_campaign_dir.resolve()
            or manifest.get("test_accessed") is not False
        ):
            raise ValueError("Existing campaign manifest does not match this campaign")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CAMPAIGN_SCHEMA,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_sha": repository_commit(),
        "source_campaign_dir": str(source_campaign_dir.resolve()),
        "feature_store": store_identity,
        "target_scale": None,
        "baseline_run": str(baseline.resolve()),
        "seed": SEED,
        "recency_policy": "uniform",
        "objective": objective_metadata(),
        "run_specifications": [asdict(spec) for spec in expand_campaign_specs()],
        "primary_metric": "validation mean daily cross-sectional Spearman IC",
        "test_accessed": False,
    }
    _atomic_json(path, manifest)
    return manifest


def _matching_completed_attempt(
    arm_dir: Path,
    spec: RunSpec,
    store: Path,
    scale_identity: dict[str, object],
) -> Path | None:
    for attempt in sorted(arm_dir.glob("attempt_*"), reverse=True):
        path = attempt / "run_manifest.json"
        if not path.is_file():
            continue
        manifest = _load_manifest(path)
        if (
            manifest.get("status") == "completed"
            and manifest.get("repository_commit") == repository_commit()
            and manifest.get("seed") == SEED
            and manifest.get("recency_policy") == "uniform"
            and manifest.get("cross_equity_attention") is spec.cross_equity_attention
            and manifest.get("objective") == objective_metadata()
            and manifest.get("target_scale_identity") == scale_identity
            and Path(str(manifest.get("feature_store"))).resolve() == store
            and manifest.get("split", {}).get("test_accessed") is False
        ):
            return attempt
    return None


def _run_spec(
    output_dir: Path,
    spec: RunSpec,
    store: Path,
    target_scale_dir: Path,
    scale_identity: dict[str, object],
) -> Path:
    arm_dir = output_dir / "runs" / spec.arm / f"seed_{SEED}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    completed = _matching_completed_attempt(arm_dir, spec, store, scale_identity)
    if completed is not None:
        return completed
    attempts = [
        int(path.name.removeprefix("attempt_"))
        for path in arm_dir.glob("attempt_*")
        if path.name.removeprefix("attempt_").isdigit()
    ]
    attempt = arm_dir / f"attempt_{max(attempts, default=0) + 1:02d}"
    return run_training(
        store=store,
        target_scale_dir=target_scale_dir,
        seed=SEED,
        recency_policy="uniform",
        cross_equity_attention=spec.cross_equity_attention,
        run_dir=attempt,
    )


def _validate_development_only(run_dir: Path, store: Path) -> None:
    manifest = _load_manifest(run_dir / "run_manifest.json")
    if manifest["split"]["test_accessed"] is not False:
        raise ValueError("Run reports test access")
    with np.load(run_dir / "validation_observations.npz", allow_pickle=False) as data:
        observed = np.asarray(data["sample_id"])
    rows = select_sample_split(
        load_sample_index(store, through=VALIDATION_END), "validation"
    )
    if not np.isin(observed, rows["sample_id"].to_numpy()).all():
        raise ValueError("Run observations contain non-validation samples")


def _horizon_scores(run_dir: Path) -> dict[str, float]:
    metrics = _load_manifest(run_dir / "validation_metrics.json")
    return {
        str(row["horizon_minutes"]): float(row["mean_daily_spearman_ic"])
        for row in metrics["horizons"]
    }


def _write_report(
    output_dir: Path,
    baseline: Path,
    base: Path,
    attention: Path,
) -> None:
    baseline_score, base_score, attention_score = map(
        _score, (baseline, base, attention)
    )
    report = {
        "schema": CAMPAIGN_SCHEMA,
        "seed": SEED,
        "test_accessed": False,
        "runs": {
            "soft_spearman_tod_baseline": {
                "run_dir": str(baseline),
                "primary_ic": baseline_score,
                "per_horizon_ic": _horizon_scores(baseline),
            },
            "hybrid_base": {
                "run_dir": str(base),
                "primary_ic": base_score,
                "per_horizon_ic": _horizon_scores(base),
            },
            "hybrid_residual_attention": {
                "run_dir": str(attention),
                "primary_ic": attention_score,
                "per_horizon_ic": _horizon_scores(attention),
            },
        },
        "hybrid_base_minus_soft_baseline": base_score - baseline_score,
        "residual_attention_minus_hybrid_base": attention_score - base_score,
    }
    _atomic_json(output_dir / "campaign_report.json", report)
    (output_dir / "campaign_report.md").write_text(
        "# Hybrid loss and residual attention\n\n"
        "| Arm | Validation IC | Delta from parent |\n"
        "|---|---:|---:|\n"
        f"| Soft-Spearman TOD baseline | {baseline_score:.6f} | — |\n"
        f"| Hybrid base | {base_score:.6f} | {base_score - baseline_score:+.6f} |\n"
        f"| Hybrid residual attention | {attention_score:.6f} | "
        f"{attention_score - base_score:+.6f} |\n\n"
        "All results are validation-only; test_accessed=false.\n",
        encoding="utf-8",
    )


def run_campaign(source_campaign_dir: Path, output_dir: Path) -> Path:
    store, baseline, store_identity = _source_state(source_campaign_dir)
    manifest = _load_or_create_manifest(
        output_dir, source_campaign_dir, store_identity, baseline
    )
    target_scale_dir = build_target_scale_sidecar(
        store, store_identity, output_dir / "target_scale"
    )
    scale_identity = target_scale_identity(target_scale_dir, store_identity)
    if manifest["target_scale"] is None:
        manifest["target_scale"] = scale_identity
        _atomic_json(output_dir / "campaign_manifest.json", manifest)
    elif manifest["target_scale"] != scale_identity:
        raise ValueError("Campaign target-scale identity changed")
    runs = {
        spec.arm: _run_spec(output_dir, spec, store, target_scale_dir, scale_identity)
        for spec in expand_campaign_specs()
    }
    for run_dir in runs.values():
        _validate_development_only(run_dir, store)
    _write_report(
        output_dir, baseline, runs["hybrid_base"], runs["hybrid_residual_attention"]
    )
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_accessed": False,
        }
    )
    _atomic_json(output_dir / "campaign_manifest.json", manifest)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid-loss base and residual-attention experiments"
    )
    parser.add_argument("--source-campaign-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_count": len(expand_campaign_specs()),
                    "seed": SEED,
                    "specifications": [
                        asdict(spec) for spec in expand_campaign_specs()
                    ],
                },
                indent=2,
            )
        )
        return
    print(run_campaign(args.source_campaign_dir, args.output_dir))


if __name__ == "__main__":
    main()
