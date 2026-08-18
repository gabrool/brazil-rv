from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.preprocessing.build import build_feature_store

from .contract import (
    ALLOWED_SEEDS,
    RECENCY_POLICIES,
    RUN_OUTPUT_BASE,
    VALIDATION_END,
)
from .data import (
    feature_store_identity,
    load_sample_index,
    resolve_feature_store,
    select_sample_split,
)
from .metrics import average_ranks, moving_block_bootstrap, primary_validation_score
from .provenance import repository_commit
from .train import run_training

CAMPAIGN_SCHEMA = "PIT_CLEAN_CORE_CAMPAIGN"
CAMPAIGN_DIR = RUN_OUTPUT_BASE / "pit_clean_core_campaign"
RECENCY_CANDIDATES = RECENCY_POLICIES[1:]


@dataclass(frozen=True)
class RunSpec:
    stage: str
    arm: str
    seed: int
    store: str
    recency_policy: str
    cross_equity_attention: bool


def expand_campaign_specs() -> list[RunSpec]:
    specs = [
        RunSpec(
            "pit_clean_control", "legacy_uniform", seed, "control", "uniform", False
        )
        for seed in ALLOWED_SEEDS
    ]
    specs.extend(
        RunSpec("full_tod_control", "tod_uniform", seed, "full_tod", "uniform", False)
        for seed in ALLOWED_SEEDS
    )
    specs.extend(
        RunSpec(
            "recency_matrix",
            f"tod_{policy}",
            seed,
            "full_tod",
            policy,
            False,
        )
        for policy in RECENCY_CANDIDATES
        for seed in ALLOWED_SEEDS
    )
    specs.extend(
        RunSpec(
            "cross_equity_attention",
            "attention_selected_parent",
            seed,
            "full_tod",
            "selected_parent",
            True,
        )
        for seed in ALLOWED_SEEDS
    )
    if len(specs) != 21:
        raise RuntimeError("Campaign expansion must contain exactly 21 runs")
    return specs


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _initial_manifest(campaign_dir: Path) -> dict[str, object]:
    control_store = resolve_feature_store()
    return {
        "schema": CAMPAIGN_SCHEMA,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_sha": repository_commit(),
        "test_accessed": False,
        "control_store": feature_store_identity(control_store),
        "full_tod_store": None,
        "normalization_contract": {
            "profile_input": (
                "unclipped_daily_volatility_normalized_equity_close_moves"
            ),
            "bin_minutes": 30,
            "prior_session_equivalents": 20,
            "relative_variance_bounds": [0.25, 4.0],
            "training_session_update": "emit_then_update",
            "freeze_after": "2024-06-28",
            "strength": "full",
        },
        "run_specifications": [asdict(spec) for spec in expand_campaign_specs()],
        "selection": {
            "primary_metric": (
                "mean daily cross-sectional Spearman IC, equally averaged over "
                "validation dates and 30/60/120-minute horizons"
            ),
            "recency": (
                "highest three-seed mean among policies beating uniform in mean "
                "and at least two matched seeds; otherwise uniform"
            ),
            "attention": (
                "promote only when mean exceeds parent and at least two matched "
                "seeds improve"
            ),
        },
    }


def _load_or_create_manifest(campaign_dir: Path) -> dict[str, object]:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    path = campaign_dir / "campaign_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != CAMPAIGN_SCHEMA
            or manifest.get("repository_sha") != repository_commit()
            or manifest.get("test_accessed") is not False
        ):
            raise ValueError("Existing campaign manifest does not match this campaign")
        return manifest
    manifest = _initial_manifest(campaign_dir)
    _atomic_json(path, manifest)
    return manifest


def _store_from_identity(identity: object) -> Path:
    if not isinstance(identity, dict):
        raise ValueError("Campaign feature-store identity is missing")
    path = Path(str(identity["path"]))
    if not path.is_dir() or feature_store_identity(path) != identity:
        raise ValueError("Campaign feature-store identity no longer matches")
    return path


def _completed_attempt(arm_dir: Path, spec: RunSpec, store: Path) -> Path | None:
    for attempt in sorted(arm_dir.glob("attempt_*"), reverse=True):
        path = attempt / "run_manifest.json"
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") == "completed"
            and manifest.get("seed") == spec.seed
            and manifest.get("recency_policy") == spec.recency_policy
            and manifest.get("cross_equity_attention") == spec.cross_equity_attention
            and Path(str(manifest.get("feature_store"))).resolve() == store.resolve()
            and manifest.get("repository_commit") == repository_commit()
            and manifest.get("split", {}).get("test_accessed") is False
        ):
            return attempt
    return None


def _run_spec(campaign_dir: Path, spec: RunSpec, store: Path) -> Path:
    arm_dir = campaign_dir / "runs" / spec.arm / f"seed_{spec.seed}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_attempt(arm_dir, spec, store)
    if completed is not None:
        return completed
    numbers = [
        int(path.name.removeprefix("attempt_"))
        for path in arm_dir.glob("attempt_*")
        if path.name.removeprefix("attempt_").isdigit()
    ]
    attempt = arm_dir / f"attempt_{max(numbers, default=0) + 1:02d}"
    return run_training(
        store=store,
        seed=spec.seed,
        recency_policy=spec.recency_policy,
        cross_equity_attention=spec.cross_equity_attention,
        run_dir=attempt,
    )


def _run_arm(
    campaign_dir: Path,
    arm: str,
    store: Path,
    recency_policy: str,
    *,
    attention: bool = False,
) -> dict[int, Path]:
    return {
        seed: _run_spec(
            campaign_dir,
            RunSpec(
                "cross_equity_attention" if attention else "training",
                arm,
                seed,
                "full_tod" if "legacy" not in arm else "control",
                recency_policy,
                attention,
            ),
            store,
        )
        for seed in ALLOWED_SEEDS
    }


def _score(run_dir: Path) -> float:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Run is incomplete: {run_dir}")
    return float(manifest["best_validation_score"])


def select_recency_parent(arms: dict[str, dict[int, Path]]) -> str:
    uniform = arms["uniform"]
    uniform_scores = np.asarray([_score(uniform[seed]) for seed in ALLOWED_SEEDS])
    eligible: list[tuple[float, str]] = []
    for policy in RECENCY_CANDIDATES:
        scores = np.asarray([_score(arms[policy][seed]) for seed in ALLOWED_SEEDS])
        if (
            scores.mean() > uniform_scores.mean()
            and int(np.sum(scores > uniform_scores)) >= 2
        ):
            eligible.append((float(scores.mean()), policy))
    return max(eligible)[1] if eligible else "uniform"


def _observations(run_dir: Path) -> dict[str, np.ndarray]:
    with np.load(run_dir / "validation_observations.npz", allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _validate_development_only(run_dir: Path, store: Path) -> None:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["split"]["test_accessed"] is not False:
        raise ValueError("Run reports test access")
    rows = load_sample_index(store, through=VALIDATION_END)
    validation = select_sample_split(rows, "validation")
    permitted = validation["sample_id"].to_numpy()
    observed = _observations(run_dir)["sample_id"]
    if not np.isin(observed, permitted).all():
        raise ValueError("Validation observations contain a non-validation sample")


def _ensemble_score(runs: dict[int, Path]) -> float:
    observations = [_observations(runs[seed]) for seed in ALLOWED_SEEDS]
    reference = observations[0]
    for candidate in observations[1:]:
        for name in ("targets", "label_mask", "sample_id", "date_idx", "decision_idx"):
            if not np.array_equal(candidate[name], reference[name]):
                raise ValueError("Matched-seed validation observations are misaligned")
    mask = reference["label_mask"].astype(bool)
    ensemble = np.zeros(reference["predictions"].shape, dtype=np.float32)
    for observation in observations:
        predictions = observation["predictions"]
        for sample in range(predictions.shape[0]):
            for horizon in range(predictions.shape[2]):
                active = mask[sample, :, horizon]
                ensemble[sample, active, horizon] += average_ranks(
                    predictions[sample, active, horizon]
                ).astype(np.float32)
    return primary_validation_score(
        ensemble,
        reference["targets"],
        mask,
        reference["date_idx"],
    )


def _arm_summary(runs: dict[int, Path]) -> dict[str, object]:
    per_seed: list[dict[str, object]] = []
    scores: list[float] = []
    for seed in ALLOWED_SEEDS:
        run_dir = runs[seed]
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(
            (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
        )
        daily = (
            pl.read_parquet(run_dir / "validation_daily_metrics.parquet")
            .group_by("date_idx")
            .agg(pl.col("spearman_ic").mean())
            .sort("date_idx")
        )
        midpoint = daily.height // 2
        score = float(metrics["primary_score"])
        scores.append(score)
        per_seed.append(
            {
                "seed": seed,
                "primary_ic": score,
                "per_horizon_ic": {
                    str(row["horizon_minutes"]): row["mean_daily_spearman_ic"]
                    for row in metrics["horizons"]
                },
                "first_half_ic": float(daily[:midpoint]["spearman_ic"].mean()),
                "latest_half_ic": float(daily[midpoint:]["spearman_ic"].mean()),
                "selected_epoch": manifest["best_epoch"],
                "run_time_seconds": manifest["total_run_seconds"],
                "run_dir": str(run_dir),
            }
        )
    mean_seed = float(np.mean(scores))
    ensemble = _ensemble_score(runs)
    return {
        "per_seed": per_seed,
        "mean_seed_primary_ic": mean_seed,
        "cross_sectional_rank_ensemble_ic": ensemble,
        "mean_seed_at_least_0_05": mean_seed >= 0.05,
        "mean_seed_at_least_0_06": mean_seed >= 0.06,
        "ensemble_at_least_0_05": ensemble >= 0.05,
        "ensemble_at_least_0_06": ensemble >= 0.06,
    }


def _daily_values(run_dir: Path) -> np.ndarray:
    return (
        pl.read_parquet(run_dir / "validation_daily_metrics.parquet")
        .group_by("date_idx")
        .agg(pl.col("spearman_ic").mean())
        .sort("date_idx")["spearman_ic"]
        .to_numpy()
    )


def _matched_delta(
    candidate: dict[int, Path], parent: dict[int, Path]
) -> dict[str, object]:
    deltas = np.stack(
        [
            _daily_values(candidate[seed]) - _daily_values(parent[seed])
            for seed in ALLOWED_SEEDS
        ]
    ).mean(axis=0)
    interval = moving_block_bootstrap(deltas, block_length=20)
    return {
        "per_seed_primary_ic_delta": {
            str(seed): _score(candidate[seed]) - _score(parent[seed])
            for seed in ALLOWED_SEEDS
        },
        "estimate": float(interval["estimate"][0]),
        "lower_95": float(interval["lower_95"][0]),
        "upper_95": float(interval["upper_95"][0]),
    }


def _write_report(
    campaign_dir: Path,
    manifest: dict[str, object],
    arms: dict[str, dict[int, Path]],
    selected_parent: str,
    attention_promoted: bool,
) -> None:
    summaries = {arm: _arm_summary(runs) for arm, runs in arms.items()}
    comparisons = {
        "tod_uniform_minus_legacy_uniform": _matched_delta(
            arms["tod_uniform"], arms["legacy_uniform"]
        ),
        **{
            f"{policy}_minus_uniform": _matched_delta(
                arms[f"tod_{policy}"], arms["tod_uniform"]
            )
            for policy in RECENCY_CANDIDATES
        },
        "attention_minus_selected_parent": _matched_delta(
            arms["attention"], arms[f"tod_{selected_parent}"]
        ),
    }
    report = {
        "schema": CAMPAIGN_SCHEMA,
        "repository_sha": manifest["repository_sha"],
        "control_store": manifest["control_store"],
        "full_tod_store": manifest["full_tod_store"],
        "test_accessed": False,
        "selected_recency_parent": selected_parent,
        "attention_promoted": attention_promoted,
        "arms": summaries,
        "matched_daily_delta_20_session_moving_block_ci": comparisons,
    }
    _atomic_json(campaign_dir / "campaign_report.json", report)
    lines = [
        "# PIT-clean core campaign",
        "",
        "| Arm | Mean-seed IC | Rank-ensemble IC | >=0.05 mean/ensemble | >=0.06 mean/ensemble |",
        "|---|---:|---:|:---:|:---:|",
    ]
    for arm, summary in summaries.items():
        lines.append(
            f"| {arm} | {summary['mean_seed_primary_ic']:.6f} | "
            f"{summary['cross_sectional_rank_ensemble_ic']:.6f} | "
            f"{summary['mean_seed_at_least_0_05']}/{summary['ensemble_at_least_0_05']} | "
            f"{summary['mean_seed_at_least_0_06']}/{summary['ensemble_at_least_0_06']} |"
        )
    lines.extend(
        [
            "",
            f"Selected recency parent: `{selected_parent}`.",
            f"Attention promoted: `{attention_promoted}`.",
            "All results are validation-only; test_accessed=false.",
            "",
        ]
    )
    (campaign_dir / "campaign_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_campaign(campaign_dir: Path = CAMPAIGN_DIR) -> Path:
    manifest = _load_or_create_manifest(campaign_dir)
    control_store = _store_from_identity(manifest["control_store"])
    arms: dict[str, dict[int, Path]] = {}
    arms["legacy_uniform"] = _run_arm(
        campaign_dir, "legacy_uniform", control_store, "uniform"
    )

    if manifest["full_tod_store"] is None:
        current = resolve_feature_store()
        if current != control_store and (current / "equity_tod_profile.json").is_file():
            full_tod_store = current
        else:
            full_tod_store, audit_dir = build_feature_store()
            manifest["full_tod_audit_dir"] = str(audit_dir)
        if any(
            (full_tod_store / name).exists()
            for name in ("equity_peer_features.npy", "equity_peer_valid.npy")
        ):
            raise ValueError("Full TOD store contains forbidden peer arrays")
        manifest["full_tod_store"] = feature_store_identity(full_tod_store)
        _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    full_tod_store = _store_from_identity(manifest["full_tod_store"])

    arms["tod_uniform"] = _run_arm(
        campaign_dir, "tod_uniform", full_tod_store, "uniform"
    )
    recency_arms = {"uniform": arms["tod_uniform"]}
    for policy in RECENCY_CANDIDATES:
        arm = f"tod_{policy}"
        arms[arm] = _run_arm(campaign_dir, arm, full_tod_store, policy)
        recency_arms[policy] = arms[arm]
    selected_parent = select_recency_parent(recency_arms)
    manifest["selected_recency_parent"] = selected_parent
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)

    attention_arm = f"attention_{selected_parent}"
    arms["attention"] = _run_arm(
        campaign_dir,
        attention_arm,
        full_tod_store,
        selected_parent,
        attention=True,
    )
    parent = arms[f"tod_{selected_parent}"]
    attention_scores = np.asarray(
        [_score(arms["attention"][seed]) for seed in ALLOWED_SEEDS]
    )
    parent_scores = np.asarray([_score(parent[seed]) for seed in ALLOWED_SEEDS])
    attention_promoted = bool(
        attention_scores.mean() > parent_scores.mean()
        and int(np.sum(attention_scores > parent_scores)) >= 2
    )
    for runs in arms.values():
        for run_dir in runs.values():
            _validate_development_only(
                run_dir,
                control_store if "legacy_uniform" in str(run_dir) else full_tod_store,
            )
    _write_report(campaign_dir, manifest, arms, selected_parent, attention_promoted)
    manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "attention_promoted": attention_promoted,
            "test_accessed": False,
        }
    )
    _atomic_json(campaign_dir / "campaign_manifest.json", manifest)
    return campaign_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 21-run PIT-clean campaign")
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        specs = [asdict(spec) for spec in expand_campaign_specs()]
        print(json.dumps({"run_count": len(specs), "specifications": specs}, indent=2))
        return
    print(run_campaign(args.campaign_dir))


if __name__ == "__main__":
    main()
