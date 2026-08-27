from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..preprocessing.contract import DECISION_TIMES
from .contract import EXPECTED_SPLIT_DATE_COUNTS, HORIZONS
from .data import (
    FeatureStoreIdentityCache,
    create_evaluation_loader,
    feature_store_identity,
    load_recorded_external_sidecar,
    load_sample_index,
    select_sample_split,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_validation_observations,
)
from .evaluate import load_current_run
from .hpo_sweep import STORE_V2_DYNAMIC_ZERO, STORE_V2_SLOW_ZERO
from .metrics import (
    daily_horizon_ic,
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    primary_validation_score,
    rank_average_predictions,
    sample_level_spearman_ic,
)
from .model import build_model
from .official_read import ALL_SEEDS, PATIENCE_RULE
from .provenance import repository_commit
from .trajectory import checkpoint_path

SCHEMA = "EXPERIMENT51_TEST_READ_V1"
DEPLOYED_RECIPE_SHA256 = (
    "d4729dc4e614e0edd5118ba5ed5b7bc92f69ca2faceab4a09d0559115e5c4058"
)
TEST_START = date(2025, 7, 7)
TEST_END = date(2026, 7, 17)
TRAIN_END = date(2024, 6, 28)
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_BLOCKS = (5, 10)
BOOTSTRAP_SEED = 20260827
PAIRED_HALF_DATES = 129


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def interpretation_band(test_ic: float) -> str:
    if test_ic >= 0.040:
        return "A"
    if test_ic >= 0.035:
        return "B"
    if test_ic >= 0.030:
        return "C"
    return "D"


def paired_h2_minus_h1(daily: np.ndarray) -> np.ndarray:
    values = np.asarray(daily, dtype=np.float64)
    required = 2 * PAIRED_HALF_DATES + 1
    if values.shape != (required,):
        raise ValueError(f"Paired staleness diagnostic requires {required} dates")
    return (
        values[PAIRED_HALF_DATES : 2 * PAIRED_HALF_DATES] - values[:PAIRED_HALF_DATES]
    )


def _bootstrap(values: np.ndarray, blocks: Sequence[int], seed_offset: int) -> dict:
    return {
        str(block): {
            name: np.asarray(value).tolist()
            for name, value in moving_block_bootstrap(
                values,
                replications=BOOTSTRAP_REPLICATIONS,
                block_length=block,
                seed=BOOTSTRAP_SEED + seed_offset + block,
            ).items()
        }
        for block in blocks
    }


def _selected_epoch(run: Path) -> int:
    diagnostics = _read_json(run / "trajectory_diagnostics.json")
    return int(diagnostics["patience3"]["selected_epoch"])


def _inventory_by_path(deployed: Mapping[str, object]) -> dict[Path, dict]:
    inventory = deployed.get("retained_checkpoint_inventory")
    if not isinstance(inventory, list):
        raise ValueError("Experiment-45 checkpoint inventory is missing")
    result = {Path(str(item["path"])).resolve(): item for item in inventory}
    if len(result) != 20 or len(result) != len(inventory):
        raise ValueError("Experiment-45 checkpoint inventory is not the exact 20 files")
    for path, item in result.items():
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"Experiment-45 checkpoint differs: {path}")
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"Experiment-45 checkpoint hash differs: {path}")
    return result


def verify_experiment45(root: Path, store: Path) -> dict[str, object]:
    deployed_path = root / "deployed_recipe.json"
    if _sha256(deployed_path) != DEPLOYED_RECIPE_SHA256:
        raise ValueError("Experiment-45 deployed recipe hash differs")
    deployed = _read_json(deployed_path)
    expected_jobs = {f"arm1_store_v2__seed_{seed}" for seed in ALL_SEEDS}
    expected_members = {f"store_v2|seed_{seed}|patience3_raw" for seed in ALL_SEEDS}
    if (
        deployed.get("test_accessed") is not False
        or deployed.get("ensemble") != "uniform tie-aware rank average"
        or set(deployed.get("measured_member_job_names", ())) != expected_jobs
        or {item["identity"] for item in deployed.get("members", ())}
        != expected_members
    ):
        raise ValueError("Experiment-45 deployed recipe contract differs")
    inventory = _inventory_by_path(deployed)
    store_identity = feature_store_identity(store)
    runs: dict[str, object] = {}
    for seed in ALL_SEEDS:
        run = (root / "runs" / f"arm1_store_v2__seed_{seed}").resolve()
        manifest_path = run / "run_manifest.json"
        diagnostics_path = run / "trajectory_diagnostics.json"
        manifest = _read_json(manifest_path)
        zeroing = manifest.get("equity_input_zeroing", {})
        if (
            manifest.get("status") != "completed"
            or manifest.get("seed") != seed
            or manifest.get("feature_store_identity") != store_identity
            or Path(str(manifest.get("feature_store"))).resolve() != store.resolve()
            or manifest.get("frozen_selection", {}).get("selected_rule")
            != PATIENCE_RULE
            or manifest.get("split", {}).get("training") != "official"
            or manifest.get("split", {}).get("selection") != "official"
            or manifest.get("split", {}).get("test_accessed") is not False
            or tuple(zeroing.get("dynamic_channels", ())) != STORE_V2_DYNAMIC_ZERO
            or tuple(zeroing.get("slow_fields", ())) != STORE_V2_SLOW_ZERO
        ):
            raise ValueError(f"Experiment-45 member contract differs: seed {seed}")
        epoch = _selected_epoch(run)
        selected = checkpoint_path(run, epoch).resolve()
        final = checkpoint_path(run, 20).resolve()
        if selected not in inventory or final not in inventory:
            raise ValueError(f"Experiment-45 retained checkpoints differ: seed {seed}")
        runs[f"seed_{seed}"] = {
            "run": str(run),
            "run_manifest": _artifact(manifest_path),
            "trajectory_diagnostics": _artifact(diagnostics_path),
            "selected_epoch": epoch,
            "selected_checkpoint": _artifact(selected),
            "final_checkpoint": _artifact(final),
        }
    return {
        "root": str(root.resolve()),
        "deployed_recipe": _artifact(deployed_path),
        "consolidation_read_manifest": _artifact(
            root / "consolidation_read_manifest.json"
        ),
        "frozen_design": _artifact(root / "freeze" / "frozen_design.json"),
        "members": runs,
    }


def _interpretation_statement(preregistration: Path) -> str:
    text = preregistration.read_text(encoding="utf-8")
    start = text.index("## Predeclared expectations and interpretation")
    end = text.index("## Accounting and hygiene", start)
    return text[start:end].rstrip() + "\n"


def freeze_program(
    *,
    store: Path,
    experiment45_root: Path,
    output: Path,
    preregistration: Path,
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    rows = select_sample_split(load_sample_index(store), "test")
    dates = rows.select("trade_date").unique().sort("trade_date")["trade_date"]
    if (
        len(dates) != EXPECTED_SPLIT_DATE_COUNTS["test"]
        or dates[0] != TEST_START
        or dates[-1] != TEST_END
    ):
        raise ValueError("Sealed test metadata differs from the frozen split")
    source = verify_experiment45(experiment45_root, store)
    output.mkdir(parents=True)
    preregistration_copy = output / "preregistration.md"
    preregistration_copy.write_bytes(preregistration.read_bytes())
    (output / "interpretation_statement.md").write_text(
        _interpretation_statement(preregistration), encoding="utf-8"
    )
    design = {
        "schema": SCHEMA,
        "status": "frozen",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "preregistration": _artifact(preregistration_copy),
        "interpretation_statement": _artifact(output / "interpretation_statement.md"),
        "store": {
            "path": str(store.resolve()),
            "identity": feature_store_identity(store),
        },
        "experiment45": source,
        "measured_object": {
            "seeds": list(ALL_SEEDS),
            "selected_rule": PATIENCE_RULE,
            "input_contract": "store-v2 34-field",
            "ensemble": "uniform within-sample/horizon tie-aware rank average",
            "retraining": False,
            "comparator": None,
        },
        "test_metadata": {
            "start": TEST_START.isoformat(),
            "end": TEST_END.isoformat(),
            "date_count": len(dates),
            "sample_count": rows.height,
        },
        "measurement_contract": {
            "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
            "bootstrap_blocks": list(BOOTSTRAP_BLOCKS),
            "bootstrap_base_seed": BOOTSTRAP_SEED,
            "staleness_difference": "H2 minus H1",
            "paired_half_dates": PAIRED_HALF_DATES,
            "paired_dates": (
                "first 129 dates paired by ordinal position with the next 129; "
                "date 259 is excluded only from this paired diagnostic"
            ),
            "difficulty": {
                "cross_sectional_dispersion": (
                    "mean population standard deviation of valid raw label returns "
                    "across equities, over sample-horizon cells in each quarter"
                ),
                "per_name_vol_level": (
                    "median sample standard deviation of valid raw label returns "
                    "over equity-horizon groups in each quarter"
                ),
                "active_universe_size": (
                    "mean valid-equity count over sample-horizon cells in each quarter"
                ),
            },
            "execution_metrics": False,
            "post_score_additions": False,
        },
        "official_validation_accessed": True,
        "test_accessed": False,
    }
    design_path = output / "frozen_design.json"
    _atomic_json(design_path, design)
    _atomic_json(
        output / "test_access_ledger.json",
        {
            "schema": "TEST_ACCESS_LEDGER_EVENT_V1",
            "experiment": 51,
            "status": "frozen",
            "first_and_only_test_event": True,
            "official_validation_accessed": True,
            "test_accessed": False,
            "design_sha256": _sha256(design_path),
        },
    )
    return design_path


def _collect_member(
    run: Path,
    rows: pl.DataFrame,
    identity_cache: FeatureStoreIdentityCache,
) -> EvaluationObservations:
    torch.set_float32_matmul_precision("high")
    states, manifest, store, rule = load_current_run(run, identity_cache=identity_cache)
    if rule != PATIENCE_RULE or len(states) != 1:
        raise ValueError("Experiment-51 member state differs from its frozen rule")
    sidecar = load_recorded_external_sidecar(manifest.get("external_sidecar"), store)
    loader = create_evaluation_loader(
        store, rows, seed=int(manifest["seed"]), sidecar=sidecar
    )
    model = build_model(None if sidecar is None else sidecar.feature_count).cuda()
    model.load_state_dict(states[0], strict=True)
    observations, _ = collect_validation_observations(model, loader)
    return observations


def _date_lookup(rows: pl.DataFrame) -> dict[int, date]:
    return {
        int(row["date_idx"]): row["trade_date"]
        for row in rows.select("date_idx", "trade_date").unique().iter_rows(named=True)
    }


def _period_labels(date_idx: np.ndarray, lookup: Mapping[int, date], period: str):
    dates = [lookup[int(value)] for value in np.unique(date_idx)]
    if period == "quarter":
        return [f"{value.year}Q{(value.month - 1) // 3 + 1}" for value in dates]
    if period == "month":
        return [value.strftime("%Y-%m") for value in dates]
    raise ValueError(period)


def difficulty_context(
    observations: EvaluationObservations,
    lookup: Mapping[int, date],
) -> list[dict[str, object]]:
    labels = _period_labels(observations.date_idx, lookup, "quarter")
    unique_dates = np.unique(observations.date_idx)
    sample_period = np.asarray(
        [
            labels[np.searchsorted(unique_dates, value)]
            for value in observations.date_idx
        ]
    )
    result = []
    for quarter in dict.fromkeys(labels):
        on_period = sample_period == quarter
        dispersions = []
        active = []
        for sample in np.flatnonzero(on_period):
            for horizon in range(observations.raw_returns.shape[2]):
                mask = observations.label_mask[sample, :, horizon]
                values = observations.raw_returns[sample, mask, horizon]
                if values.size:
                    dispersions.append(float(np.std(values, ddof=0)))
                    active.append(int(values.size))
        name_vol = []
        for equity in range(observations.raw_returns.shape[1]):
            for horizon in range(observations.raw_returns.shape[2]):
                mask = observations.label_mask[on_period, equity, horizon]
                values = observations.raw_returns[on_period, equity, horizon][mask]
                if values.size > 1:
                    name_vol.append(float(np.std(values, ddof=1)))
        result.append(
            {
                "quarter": quarter,
                "cross_sectional_dispersion": float(np.mean(dispersions)),
                "per_name_vol_level": float(np.median(name_vol)),
                "active_universe_size": float(np.mean(active)),
            }
        )
    return result


def analyze_measurements(
    members: Mapping[int, EvaluationObservations],
    test_lookup: Mapping[int, date],
    validation_reference: EvaluationObservations,
    validation_lookup: Mapping[int, date],
) -> tuple[dict[str, object], list[dict[str, object]], EvaluationObservations]:
    if tuple(members) != ALL_SEEDS:
        raise ValueError("Experiment-51 requires the exact deployed seed order")
    reference = next(iter(members.values()))
    for member in members.values():
        assert_observations_aligned(reference, member)
    ensemble = replace(
        reference,
        predictions=rank_average_predictions(
            [member.predictions for member in members.values()], reference.label_mask
        ),
    )
    sample_ic = sample_level_spearman_ic(
        ensemble.predictions, ensemble.targets, ensemble.label_mask
    )
    date_indices, daily_primary = per_date_primary_ic(sample_ic, ensemble.date_idx)
    horizon_dates, daily_horizons = daily_horizon_ic(sample_ic, ensemble.date_idx)
    if not np.array_equal(date_indices, horizon_dates):
        raise ValueError("Daily metric date axes differ")
    trade_dates = [test_lookup[int(value)] for value in date_indices]
    if len(trade_dates) != EXPECTED_SPLIT_DATE_COUNTS["test"]:
        raise ValueError("Test analysis date count differs")

    quarters = [f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in trade_dates]
    months = [d.strftime("%Y-%m") for d in trade_dates]
    quarter_ic = {
        value: finite_mean(daily_primary[np.asarray(quarters) == value])
        for value in dict.fromkeys(quarters)
    }
    month_ic = {
        value: finite_mean(daily_primary[np.asarray(months) == value])
        for value in dict.fromkeys(months)
    }
    elapsed = np.asarray([(value - TRAIN_END).days for value in trade_dates])
    finite = np.isfinite(daily_primary)
    slope = float(np.polyfit(elapsed[finite], daily_primary[finite], 1)[0])
    paired = paired_h2_minus_h1(daily_primary)
    paired_interval = _bootstrap(paired, (10,), 300)["10"]
    retrain = bool(float(paired_interval["upper_95"][0]) < 0.0)

    tod = []
    for decision in np.unique(ensemble.decision_idx):
        on_decision = ensemble.decision_idx == decision
        tod.append(
            {
                "decision_idx": int(decision),
                "decision_time": DECISION_TIMES[int(decision)].isoformat(),
                "mean_ic": finite_mean(sample_ic[on_decision].ravel()),
            }
        )
    member_ic = {
        f"seed_{seed}": primary_validation_score(
            member.predictions,
            member.targets,
            member.label_mask,
            member.date_idx,
        )
        for seed, member in members.items()
    }
    correlations = []
    for left, right in combinations(ALL_SEEDS, 2):
        correlations.append(
            {
                "left_seed": left,
                "right_seed": right,
                "prediction_spearman": primary_validation_score(
                    members[left].predictions,
                    members[right].predictions,
                    reference.label_mask,
                    reference.date_idx,
                ),
            }
        )
    score = finite_mean(daily_primary)
    mean_member = float(np.mean(list(member_ic.values())))
    daily_rows = [
        {
            "date_idx": int(date_index),
            "trade_date": trade_date,
            "primary_ic": float(primary),
            **{
                f"ic_{horizon}m": float(horizon_values[index])
                for index, horizon in enumerate(HORIZONS)
            },
        }
        for date_index, trade_date, primary, horizon_values in zip(
            date_indices, trade_dates, daily_primary, daily_horizons, strict=True
        )
    ]
    analysis = {
        "schema": SCHEMA,
        "primary": {
            "test_ic": score,
            "daily_date_count": len(daily_primary),
            "bootstrap": _bootstrap(daily_primary, BOOTSTRAP_BLOCKS, 0),
        },
        "per_horizon": {
            str(horizon): {
                "mean_ic": finite_mean(daily_horizons[:, index]),
                "bootstrap": _bootstrap(
                    daily_horizons[:, index], BOOTSTRAP_BLOCKS, 100 + 10 * index
                ),
            }
            for index, horizon in enumerate(HORIZONS)
        },
        "staleness": {
            "quarterly_ic": quarter_ic,
            "daily_ic_slope_per_day_since_train_end": slope,
            "paired_definition": (
                "H2 minus H1; first 129 dates paired with next 129; date 259 "
                "excluded only from paired diagnostic"
            ),
            "h2_minus_h1": finite_mean(paired),
            "block10_bootstrap": paired_interval,
            "retrain_before_live_indicated": retrain,
        },
        "difficulty": {
            "test": difficulty_context(ensemble, test_lookup),
            "official_validation": difficulty_context(
                validation_reference, validation_lookup
            ),
        },
        "tod_guardrail": tod,
        "monthly_ic": month_ic,
        "members": {
            "member_ic": member_ic,
            "prediction_correlation_pairs": correlations,
            "mean_member_ic": mean_member,
            "ensemble_vs_mean_member_gain": score - mean_member,
        },
        "interpretation": {
            "band": interpretation_band(score),
            "official_validation_ic": 0.043718770472,
            "deployment_or_recipe_change": False,
        },
        "execution_metrics_computed": False,
        "official_validation_accessed": True,
        "test_accessed": True,
    }
    return analysis, daily_rows, ensemble


def _load_validation_reference(run: Path) -> EvaluationObservations:
    with np.load(run / "validation_reference.npz", allow_pickle=False) as values:
        fields = {
            name: values[name].copy()
            for name in EvaluationObservations.__dataclass_fields__
            if name != "predictions"
        }
    return EvaluationObservations(
        predictions=np.zeros_like(fields["targets"], dtype=np.float32), **fields
    )


def _result_text(analysis: Mapping[str, object]) -> str:
    band = analysis["interpretation"]["band"]
    staleness = analysis["staleness"]
    if band == "C":
        return (
            "Band C: material degradation. Attribution is limited to the frozen "
            "staleness and period-difficulty tables in measurements 3–4; no new "
            "held-out analysis is authorized."
        )
    if band == "D":
        return (
            "Band D: the official-period edge did not generalize; reassess the program."
        )
    if staleness["retrain_before_live_indicated"]:
        return f"Band {band}; the frozen staleness rule indicates retraining before live use."
    return f"Band {band}; the frozen staleness rule does not indicate deterioration."


def run_program(*, output: Path, design_path: Path) -> Path:
    design = _read_json(design_path)
    ledger_path = output / "test_access_ledger.json"
    ledger = _read_json(ledger_path)
    if (
        design.get("schema") != SCHEMA
        or design.get("repository_commit") != repository_commit()
        or design.get("test_accessed") is not False
        or ledger.get("test_accessed") is not False
        or ledger.get("design_sha256") != _sha256(design_path)
    ):
        raise ValueError("Experiment-51 frozen contract differs")
    verify_experiment45(
        Path(design["experiment45"]["root"]), Path(design["store"]["path"])
    )
    prediction_dir = output / "member_predictions"
    if prediction_dir.exists():
        raise ValueError(
            "Experiment-51 test predictions already exist; second read refused"
        )
    _atomic_json(
        ledger_path,
        {
            **ledger,
            "status": "running",
            "opened_at": _now(),
            "test_accessed": True,
        },
    )
    store = Path(design["store"]["path"])
    all_rows = load_sample_index(store)
    test_rows = select_sample_split(all_rows, "test")
    validation_rows = select_sample_split(all_rows, "validation")
    cache = FeatureStoreIdentityCache()
    members: dict[int, EvaluationObservations] = {}
    reference: EvaluationObservations | None = None
    for seed in ALL_SEEDS:
        run = Path(design["experiment45"]["members"][f"seed_{seed}"]["run"])
        member = _collect_member(run, test_rows, cache)
        if reference is None:
            reference = member
            _atomic_npz(
                output / "test_reference.npz",
                **{
                    name: getattr(member, name)
                    for name in EvaluationObservations.__dataclass_fields__
                    if name != "predictions"
                },
            )
        else:
            assert_observations_aligned(reference, member)
        _atomic_npz(prediction_dir / f"seed_{seed}.npz", predictions=member.predictions)
        members[seed] = member
    assert reference is not None
    validation_run = Path(design["experiment45"]["members"]["seed_11"]["run"])
    analysis, daily_rows, ensemble = analyze_measurements(
        members,
        _date_lookup(test_rows),
        _load_validation_reference(validation_run),
        _date_lookup(validation_rows),
    )
    _atomic_npz(output / "ensemble_predictions.npz", predictions=ensemble.predictions)
    pl.DataFrame(daily_rows).write_parquet(output / "daily_ic.parquet")
    _atomic_json(output / "analysis.json", analysis)
    result = {
        "schema": SCHEMA,
        "completed_at": _now(),
        "test_ic": analysis["primary"]["test_ic"],
        "band": analysis["interpretation"]["band"],
        "staleness_retrain_before_live_indicated": analysis["staleness"][
            "retrain_before_live_indicated"
        ],
        "interpretation": _result_text(analysis),
        "model_or_recipe_changed": False,
        "official_validation_accessed": True,
        "test_accessed": True,
        "test_spent_forever": True,
    }
    _atomic_json(output / "experiment51_result.json", result)
    _atomic_json(
        ledger_path,
        {
            **_read_json(ledger_path),
            "status": "completed",
            "completed_at": _now(),
            "test_accessed": True,
            "test_spent_forever": True,
        },
    )
    files = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_inventory.json", "program_manifest.json"}
        and not path.name.endswith(".tmp")
    )
    _atomic_json(
        output / "artifact_inventory.json",
        {
            "schema": SCHEMA,
            "artifacts": [_artifact(path) for path in files],
            "artifact_count": len(files),
            "test_accessed": True,
        },
    )
    _atomic_json(
        output / "program_manifest.json",
        {
            "schema": SCHEMA,
            "status": "completed",
            "repository_commit": repository_commit(),
            "design_sha256": _sha256(design_path),
            "artifact_inventory": _artifact(output / "artifact_inventory.json"),
            "result": _artifact(output / "experiment51_result.json"),
            "analysis": _artifact(output / "analysis.json"),
            "test_prediction_archive_count": len(ALL_SEEDS) + 1,
            "execution_metrics_computed": False,
            "model_or_recipe_changed": False,
            "official_validation_accessed": True,
            "test_accessed": True,
            "test_spent_forever": True,
        },
    )
    return output / "experiment51_result.json"


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Experiment 51")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--store", required=True, type=Path)
    freeze.add_argument("--experiment45-root", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--preregistration", required=True, type=Path)
    run = subparsers.add_parser("run")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--design", required=True, type=Path)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.command == "freeze":
        print(
            freeze_program(
                store=args.store,
                experiment45_root=args.experiment45_root,
                output=args.output,
                preregistration=args.preregistration,
            )
        )
    else:
        print(run_program(output=args.output, design_path=args.design))


if __name__ == "__main__":
    main()
