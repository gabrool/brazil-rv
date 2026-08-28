from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..execution.inputs import (
    OOF_PREDICTION_ARCHIVE_SCHEMA,
    load_discovery_prediction_archive,
)
from ..execution.splits import (
    PurgedFold,
    load_purged_training_folds,
    policy_evaluation_slices,
    purged_training_folds,
)
from .contract import GH200_RUNTIME, MAX_EPOCHS, TRAIN_END, TRAINING_SPECIFICATION
from .data import (
    create_training_loaders,
    feature_store_axis_identity,
    feature_store_identity,
    load_nextgen_target_sidecar,
    load_sample_index,
    sample_window_metadata,
)
from .engine import (
    EvaluationObservations,
    assert_observations_aligned,
    collect_validation_observations,
    compile_model,
    compile_training_objective,
    train_one_epoch,
)
from .hpo_sweep import STORE_V2_DYNAMIC_ZERO, STORE_V2_SLOW_ZERO
from .metrics import (
    primary_validation_score,
    rank_average_predictions,
    rank_prediction_similarity,
)
from .model import build_model, count_trainable_parameters
from .optim import build_optimizer, build_scheduler
from .provenance import model_metadata, repository_commit
from .trajectory import ModelEMA, temporarily_load_state

OOF_SEEDS = (11, 29, 47, 61, 79, 97, 113, 131, 149, 167)
OOF_SCHEMA = "BRAZIL_RV_OOF_MANUFACTURE_V1"
MAX_PARALLEL = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _atomic_npz(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _indexed_trade_dates(
    table: pl.DataFrame,
) -> tuple[tuple[date, ...], dict[int, date], dict[date, int]]:
    pairs = [
        (int(index), value)
        for index, value in table.select("date_idx", "trade_date").iter_rows()
    ]
    return (
        tuple(value for _, value in pairs),
        dict(pairs),
        {value: index for index, value in pairs},
    )


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _fold_rows(rows: pl.DataFrame, dates: Sequence[object]) -> pl.DataFrame:
    return rows.filter(pl.col("trade_date").is_in(list(dates)))


def _reference_values(observations: EvaluationObservations) -> dict[str, np.ndarray]:
    return {
        name: getattr(observations, name)
        for name in EvaluationObservations.__dataclass_fields__
        if name != "predictions"
    }


def _run_manifest_valid(
    path: Path,
    fold: PurgedFold,
    seed: int,
    store_identity: Mapping[str, object],
    *,
    to_close: bool = False,
) -> bool:
    if not path.is_file():
        return False
    value = _read_json(path)
    return (
        value.get("schema") == "BRAZIL_RV_MONITOR_FREE_OOF_RUN_V1"
        and value.get("status") == "completed"
        and value.get("seed") == seed
        and value.get("feature_store_identity") == store_identity
        and value.get("source_fold_sha256") == fold.payload()["sha256"]
        and value.get("epochs_completed") == MAX_EPOCHS
        and value.get("monitor") is None
        and value.get("official_validation_accessed") is False
        and value.get("test_accessed") is False
        and value.get("head_mode") == ("four_head_to_close" if to_close else "three_head")
    )


def _train_fold_seed(
    store: Path,
    run_dir: Path,
    fold_payload: dict[str, object],
    seed: int,
    target_sidecar_value: str | None = None,
) -> str:
    fold = PurgedFold(
        str(fold_payload["name"]),
        tuple(datetime.fromisoformat(value).date() for value in fold_payload["fit_dates"]),
        tuple(
            datetime.fromisoformat(value).date()
            for value in fold_payload["heldout_dates"]
        ),
        tuple(
            datetime.fromisoformat(value).date()
            for value in fold_payload["embargo_dates"]
        ),
    )
    store = store.resolve()
    identity = feature_store_identity(store)
    manifest_path = run_dir / "run_manifest.json"
    to_close = target_sidecar_value is not None
    if _run_manifest_valid(
        manifest_path, fold, seed, identity, to_close=to_close
    ):
        return str(run_dir)
    if run_dir.exists():
        raise RuntimeError(f"Incomplete OOF run requires reviewed repair: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "predictions").mkdir()
    (run_dir / "checkpoints").mkdir()
    torch.set_float32_matmul_precision("high")
    from .train import set_seeds

    set_seeds(seed)
    rows = load_sample_index(store, through=TRAIN_END)
    fit_rows = _fold_rows(rows, fold.fit_dates)
    heldout_rows = _fold_rows(rows, fold.heldout_dates)
    target_sidecar = (
        None
        if target_sidecar_value is None
        else load_nextgen_target_sidecar(Path(target_sidecar_value), store)
    )
    specification = TRAINING_SPECIFICATION
    if to_close:
        specification = replace(
            specification,
            architecture=replace(specification.architecture, output_horizons=4),
        )
    train_loader, heldout_loader, sampler = create_training_loaders(
        store,
        fit_rows,
        heldout_rows,
        GH200_RUNTIME,
        seed,
        zero_dynamic_channels=STORE_V2_DYNAMIC_ZERO,
        zero_slow_fields=STORE_V2_SLOW_ZERO,
        target_sidecar=target_sidecar,
    )
    model = build_model(
        architecture=specification.architecture, to_close_head=to_close
    ).cuda()
    ema = ModelEMA(model, 0.995)
    optimizer, _ = build_optimizer(model)
    scheduler, steps_per_epoch, warmup_steps = build_scheduler(
        optimizer, fit_rows.height, MAX_EPOCHS
    )
    base = {
        "schema": "BRAZIL_RV_MONITOR_FREE_OOF_RUN_V1",
        "status": "running",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "feature_store": str(store),
        "feature_store_identity": identity,
        "seed": seed,
        "source_fold": fold.payload(),
        "source_fold_sha256": fold.payload()["sha256"],
        "fit_window": sample_window_metadata(fit_rows, f"{fold.name}_fit"),
        "heldout_window": sample_window_metadata(
            heldout_rows, f"{fold.name}_heldout"
        ),
        "fit_exclusion_proof": {
            "heldout_disjoint_from_fit": True,
            "heldout_disjoint_from_embargo": True,
            "fit_date_identity_sha256": fold.payload()["fit_date_identity_sha256"],
            "heldout_date_identity_sha256": fold.payload()[
                "heldout_date_identity_sha256"
            ],
        },
        "model": model_metadata(
            architecture=specification.architecture, to_close_head=to_close
        ),
        "head_mode": "four_head_to_close" if to_close else "three_head",
        "target_sidecar": None if target_sidecar is None else target_sidecar.identity,
        "parameter_count": count_trainable_parameters(model),
        "training": {
            "epochs": MAX_EPOCHS,
            "fixed_trajectory": True,
            "monitor": None,
            "heldout_evaluations_during_training": 0,
            "final_states": ["epoch20_raw", "epoch20_ema_0995"],
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": warmup_steps,
            "objective": "soft_spearman",
            "temperature": 0.5,
            "sam_rho": 0.125,
            "ema_decay": 0.995,
        },
        "monitor": None,
        "equity_input_zeroing": {
            "dynamic_channels": list(STORE_V2_DYNAMIC_ZERO),
            "slow_fields": list(STORE_V2_SLOW_ZERO),
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(manifest_path, base)
    compiled_model = compile_model(model)
    compiled_objective = compile_training_objective()
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    def update_ema() -> None:
        ema.update(model)

    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_started = time.perf_counter()
            sampler.set_epoch(epoch)
            metrics = train_one_epoch(
                compiled_model,
                train_loader,
                optimizer,
                scheduler,
                GH200_RUNTIME,
                compiled_objective,
                after_update=update_ema,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_objective_loss": metrics["objective_loss"],
                    "optimizer_steps": metrics["optimizer_steps"],
                    "epoch_seconds": time.perf_counter() - epoch_started,
                }
            )
            _write_history(run_dir / "history.csv", history)

        raw, _ = collect_validation_observations(model, heldout_loader)
        with temporarily_load_state(model, ema.shadow):
            ema_observations, _ = collect_validation_observations(
                model, heldout_loader
            )
        assert_observations_aligned(raw, ema_observations)
        _atomic_npz(run_dir / "heldout_reference.npz", _reference_values(raw))
        _atomic_npz(
            run_dir / "predictions" / "epoch_20.npz",
            {"raw": raw.predictions, "ema_0995": ema_observations.predictions},
        )
        checkpoint = run_dir / "checkpoints" / "epoch_20.pt"
        torch.save(
            {
                "epoch": MAX_EPOCHS,
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "ema_0995_state_dict": ema.cpu_state_dict(),
                "source_fold_sha256": fold.payload()["sha256"],
                "feature_store_identity": identity,
                "repository_commit": base["repository_commit"],
            },
            checkpoint,
        )
        completed = {
            **base,
            "status": "completed",
            "completed_at": _now(),
            "epochs_completed": MAX_EPOCHS,
            "prediction_sha256": _sha256(run_dir / "predictions" / "epoch_20.npz"),
            "reference_sha256": _sha256(run_dir / "heldout_reference.npz"),
            "checkpoint_sha256_before_cleanup": _sha256(checkpoint),
            "total_run_seconds": time.perf_counter() - started,
        }
        _atomic_json(manifest_path, completed)
    except BaseException:
        _atomic_json(
            manifest_path,
            {
                **base,
                "status": "failed",
                "failed_at": _now(),
                "epochs_completed": len(history),
            },
        )
        raise
    return str(run_dir)


def _source_fold_payload(value: Mapping[str, object]) -> PurgedFold:
    return PurgedFold(
        str(value["name"]),
        tuple(datetime.fromisoformat(item).date() for item in value["fit_dates"]),
        tuple(datetime.fromisoformat(item).date() for item in value["heldout_dates"]),
        tuple(datetime.fromisoformat(item).date() for item in value["embargo_dates"]),
    )


def freeze_program(
    *, store: Path, experiment54_root: Path, preregistration: Path, output_dir: Path
) -> Path:
    store = store.resolve()
    experiment54_root = experiment54_root.resolve()
    preregistration = preregistration.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    rows = load_sample_index(store, through=TRAIN_END)
    dates = tuple(rows.get_column("trade_date").unique().sort().to_list())
    folds = purged_training_folds(dates)
    source54 = _read_json(experiment54_root / "frozen_design.json")
    source54_result = _read_json(experiment54_root / "experiment54_result.json")
    source54_audit = _read_json(experiment54_root / "final_audit.json")
    if source54.get("official_validation_accessed") is not False or source54.get(
        "test_accessed"
    ) is not False:
        raise ValueError("Experiment 54 source accessed a sealed split")
    if (
        source54_result.get("taker_decision") != "VIABLE"
        or source54_audit.get("status") != "passed"
        or source54_result.get("official_validation_accessed") is not False
        or source54_result.get("test_accessed") is not False
        or source54_audit.get("official_validation_accessed") is not False
        or source54_audit.get("test_accessed") is not False
    ):
        raise ValueError("Experiment 54 source is incomplete or unsealed")
    for fold in ("fold_c", "fold_a", "fold_b"):
        source = source54["fold_sources"][fold]
        load_discovery_prediction_archive(
            Path(source["ensemble_prediction"]["path"]),
            Path(source["prediction_reference"]["path"]),
            Path(source["execution_manifest"]["path"]),
            store,
        )
    output_dir.mkdir(parents=True)
    folds.write(output_dir / "purged_folds.json")
    design: dict[str, object] = {
        "schema": "BRAZIL_RV_OOF_FROZEN_DESIGN_V1",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "store": {"path": str(store), "identity": feature_store_identity(store)},
        "preregistration": {
            "path": str(preregistration),
            "sha256": _sha256(preregistration),
        },
        "experiment54": {
            "root": str(experiment54_root),
            "frozen_design_sha256": _sha256(
                experiment54_root / "frozen_design.json"
            ),
            "result_sha256": _sha256(
                experiment54_root / "experiment54_result.json"
            ),
            "final_audit_sha256": _sha256(
                experiment54_root / "final_audit.json"
            ),
            "fold_sources": source54["fold_sources"],
        },
        "purged_folds": folds.payload(),
        "seeds": list(OOF_SEEDS),
        "recipe": {
            "name": "deployed_store_v2",
            "architecture": asdict(TRAINING_SPECIFICATION.architecture),
            "temperature": 0.5,
            "epochs": 20,
            "monitor": None,
            "final_state": "ema_0995",
            "archived_secondary": "epoch20_raw",
            "zero_dynamic_channels": list(STORE_V2_DYNAMIC_ZERO),
            "zero_slow_fields": list(STORE_V2_SLOW_ZERO),
        },
        "trajectory_count": len(folds.folds) * len(OOF_SEEDS),
        "maximum_parallel_training_processes": MAX_PARALLEL,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    design["sha256"] = _canonical_sha256(design)
    _atomic_json(output_dir / "frozen_design.json", design)
    return output_dir / "frozen_design.json"


def _validate_design(root: Path) -> dict[str, object]:
    design = _read_json(root / "frozen_design.json")
    expected = dict(design)
    digest = expected.pop("sha256", None)
    if digest != _canonical_sha256(expected):
        raise ValueError("OOF frozen-design hash differs")
    if design.get("repository_commit") != repository_commit():
        raise ValueError("Repository commit differs from the OOF frozen design")
    store = Path(str(design["store"]["path"]))
    if feature_store_identity(store) != design["store"]["identity"]:
        raise ValueError("OOF feature-store identity differs")
    rows = load_sample_index(store, through=TRAIN_END)
    dates = tuple(rows.get_column("trade_date").unique().sort().to_list())
    load_purged_training_folds(root / "purged_folds.json", dates)
    return design


def _run_path(root: Path, fold: str, seed: int) -> Path:
    return root / "runs" / fold / f"seed_{seed}"


def _execute_jobs(root: Path, design: Mapping[str, object], parallel: int) -> None:
    store = Path(str(design["store"]["path"])).resolve()
    dates = tuple(
        load_sample_index(store, through=TRAIN_END)
        .get_column("trade_date")
        .unique()
        .sort()
        .to_list()
    )
    folds = load_purged_training_folds(
        root / "purged_folds.json",
        dates,
    )
    jobs = []
    for fold in folds.folds:
        for seed in OOF_SEEDS:
            run = _run_path(root, fold.name, seed)
            if _run_manifest_valid(
                run / "run_manifest.json", fold, seed, design["store"]["identity"]
            ):
                continue
            jobs.append((store, run, fold.payload(), seed))
    if parallel == 1:
        for job in jobs:
            print(_train_fold_seed(*job), flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=parallel, mp_context=mp.get_context("spawn")
    ) as executor:
        futures = [executor.submit(_train_fold_seed, *job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _materialize(root: Path, design: Mapping[str, object]) -> dict[str, object]:
    return _materialize_from_runs(root, design, to_close=False)


def _materialize_from_runs(
    root: Path, design: Mapping[str, object], *, to_close: bool
) -> dict[str, object]:
    store = Path(str(design["store"]["path"])).resolve()
    rows = load_sample_index(store, through=TRAIN_END)
    date_rows = rows.select("date_idx", "trade_date").unique().sort("date_idx")
    dates, date_by_index, _ = _indexed_trade_dates(date_rows)
    folds = load_purged_training_folds(root / "purged_folds.json", dates)
    heldout_by_date = {
        value: index
        for index, fold in enumerate(folds.folds)
        for value in fold.heldout_dates
    }
    members_by_fold: dict[int, list[np.ndarray]] = {}
    reference_by_fold: dict[int, dict[str, np.ndarray]] = {}
    run_bindings: dict[str, object] = {}
    for fold_index, fold in enumerate(folds.folds):
        for seed in OOF_SEEDS:
            run = _run_path(root, fold.name, seed)
            manifest_path = run / "run_manifest.json"
            manifest = _read_json(manifest_path)
            if not _run_manifest_valid(
                manifest_path,
                fold,
                seed,
                design["store"]["identity"],
                to_close=to_close,
            ):
                raise ValueError(f"OOF source run differs: {fold.name}/seed_{seed}")
            prediction_path = run / "predictions" / "epoch_20.npz"
            reference_path = run / "heldout_reference.npz"
            if (
                manifest.get("prediction_sha256") != _sha256(prediction_path)
                or manifest.get("reference_sha256") != _sha256(reference_path)
            ):
                raise ValueError("OOF source archive hash differs")
            with np.load(prediction_path, allow_pickle=False) as values:
                prediction = values["ema_0995"].copy()
            with np.load(reference_path, allow_pickle=False) as values:
                reference = {name: values[name].copy() for name in values.files}
            if fold_index not in reference_by_fold:
                reference_by_fold[fold_index] = reference
            elif any(
                not np.array_equal(reference[name], reference_by_fold[fold_index][name])
                for name in reference
            ):
                raise ValueError("OOF member references differ within a fold")
            members_by_fold.setdefault(fold_index, []).append(prediction)
            run_bindings[f"{fold.name}/seed_{seed}"] = {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "prediction": str(prediction_path),
                "prediction_sha256": _sha256(prediction_path),
                "reference": str(reference_path),
                "reference_sha256": _sha256(reference_path),
            }

    predictions: list[np.ndarray] = []
    references: list[dict[str, np.ndarray]] = []
    source_indices: list[np.ndarray] = []
    for fold_index, fold in enumerate(folds.folds):
        reference = reference_by_fold[fold_index]
        activity = np.broadcast_to(
            np.asarray(
                np.load(store / "equity_membership.npy", mmap_mode="r")[
                    reference["date_idx"]
                ]
                & np.load(store / "equity_data_ready.npy", mmap_mode="r")[
                    reference["date_idx"]
                ],
                dtype=bool,
            )[..., None],
            members_by_fold[fold_index][0].shape,
        )
        predictions.append(
            rank_average_predictions(members_by_fold[fold_index], activity)
        )
        references.append(reference)
        source_indices.append(
            np.full(reference["sample_id"].shape, fold_index, dtype=np.int8)
        )
        emitted_dates = {
            date_by_index[int(value)] for value in np.unique(reference["date_idx"])
        }
        if emitted_dates != set(fold.heldout_dates):
            raise ValueError("OOF emitted dates differ from the held-out fold")
    combined_reference = {
        name: np.concatenate([value[name] for value in references])
        for name in references[0]
    }
    combined_predictions = np.concatenate(predictions)
    combined_source = np.concatenate(source_indices)
    order = np.argsort(combined_reference["sample_id"], kind="stable")
    combined_reference = {name: value[order] for name, value in combined_reference.items()}
    combined_predictions = combined_predictions[order]
    combined_source = combined_source[order]
    if (
        not np.array_equal(combined_reference["sample_id"], rows["sample_id"].to_numpy())
        or set(heldout_by_date) != set(dates)
    ):
        raise ValueError("Canonical OOF materializer does not cover all TRAIN samples")

    archive_dir = root / "archive"
    prediction_path = archive_dir / "oof_predictions.npz"
    reference_path = archive_dir / "oof_reference.npz"
    _atomic_npz(prediction_path, {"ranks": combined_predictions})
    _atomic_npz(
        reference_path,
        {
            "sample_id": combined_reference["sample_id"],
            "date_idx": combined_reference["date_idx"],
            "decision_idx": combined_reference["decision_idx"],
            "source_fold_index": combined_source,
        },
    )
    source_manifest: dict[str, object] = {
        "schema": OOF_SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "feature_store_identity": design["store"]["identity"],
        "purged_folds": folds.payload(),
        "run_bindings": run_bindings,
        "prediction_sha256": _sha256(prediction_path),
        "reference_sha256": _sha256(reference_path),
        "fit_exclusion_proof": {
            "all_716_dates_held_out_once": True,
            "every_sample_source_fold_recorded": True,
            "source_fold_fit_excludes_emitted_date": True,
        },
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    source_manifest_path = archive_dir / "oof_source_manifest.json"
    _atomic_json(source_manifest_path, source_manifest)
    refresh = (
        rows.select("decision_idx", "equity_cutoff_index")
        .unique()
        .sort("decision_idx")["equity_cutoff_index"]
        .to_list()
    )
    wrapper = {
        "schema": OOF_PREDICTION_ARCHIVE_SCHEMA,
        "split": "oof_train",
        "official_validation_accessed": False,
        "test_accessed": False,
        "prediction_sha256": _sha256(prediction_path),
        "reference_sha256": _sha256(reference_path),
        "prediction_key": "ranks",
        "feature_store_identity": design["store"]["identity"],
        "axes": feature_store_axis_identity(store),
        "refresh_minutes": refresh,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
    }
    wrapper_path = archive_dir / "execution_manifest.json"
    _atomic_json(wrapper_path, wrapper)
    archive = load_discovery_prediction_archive(
        prediction_path, reference_path, wrapper_path, store
    )
    if archive.date_idx.size != len(dates):
        raise ValueError("Verified OOF archive does not cover all TRAIN dates")
    return {
        "prediction": str(prediction_path),
        "reference": str(reference_path),
        "execution_manifest": str(wrapper_path),
        "source_manifest": str(source_manifest_path),
        "prediction_sha256": _sha256(prediction_path),
        "reference_sha256": _sha256(reference_path),
        "execution_manifest_sha256": _sha256(wrapper_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
    }


def _calibration(
    root: Path, design: Mapping[str, object], archive_record: Mapping[str, object]
) -> list[dict[str, object]]:
    store = Path(str(design["store"]["path"])).resolve()
    oof = load_discovery_prediction_archive(
        Path(str(archive_record["prediction"])),
        Path(str(archive_record["reference"])),
        Path(str(archive_record["execution_manifest"])),
        store,
    )
    dates_table = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= TRAIN_END)
        .sort("date_idx")
    )
    dates, _, date_lookup = _indexed_trade_dates(dates_table)
    slices = {item.name: item for item in policy_evaluation_slices(dates, dates)}
    target_array = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    mask_array = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    for fold in ("fold_c", "fold_a", "fold_b"):
        source = design["experiment54"]["fold_sources"][fold]
        comparator = load_discovery_prediction_archive(
            Path(source["ensemble_prediction"]["path"]),
            Path(source["prediction_reference"]["path"]),
            Path(source["execution_manifest"]["path"]),
            store,
        )
        expected_dates = set(slices[fold].dates)
        date_indices = np.asarray(sorted(date_lookup[value] for value in expected_dates))
        oof_rows = np.nonzero(np.isin(oof.date_idx, date_indices))[0]
        if not np.array_equal(oof.date_idx[oof_rows], comparator.date_idx):
            raise ValueError(f"OOF calibration dates differ for {fold}")
        targets = np.stack(
            [
                target_array[int(day), :, int(decision)]
                for day in comparator.date_idx
                for decision in comparator.decision_idx
            ]
        )
        masks = np.stack(
            [
                mask_array[int(day), :, int(decision)]
                for day in comparator.date_idx
                for decision in comparator.decision_idx
            ]
        )
        oof_flat = oof.ranks[oof_rows].reshape(targets.shape)
        comparator_flat = comparator.ranks.reshape(targets.shape)
        repeated_dates = np.repeat(comparator.date_idx, comparator.decision_idx.size)
        rows.append(
            {
                "fold": fold,
                "oof_final_ema_ic": primary_validation_score(
                    oof_flat, targets, masks, repeated_dates
                ),
                "experiment41_patience_ic": primary_validation_score(
                    comparator_flat, targets, masks, repeated_dates
                ),
                "rank_correlation": rank_prediction_similarity(
                    oof_flat, comparator_flat, masks, repeated_dates
                ),
            }
        )
    _atomic_json(
        root / "archive" / "protocol_calibration.json",
        {
            "schema": "BRAZIL_RV_OOF_PROTOCOL_CALIBRATION_V1",
            "rows": rows,
            "interpretation": (
                "OOF uses monitor-free final EMA-0.995; comparator uses frozen "
                "Experiment-41 Patience. Policy inputs are causal ranks."
            ),
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return rows


def _cleanup_checkpoints(root: Path) -> dict[str, object]:
    removed = []
    for checkpoint in sorted((root / "runs").glob("fold_*/seed_*/checkpoints/epoch_20.pt")):
        manifest = _read_json(checkpoint.parents[1] / "run_manifest.json")
        if manifest.get("checkpoint_sha256_before_cleanup") != _sha256(checkpoint):
            raise ValueError(f"OOF checkpoint differs before cleanup: {checkpoint}")
        removed.append({"path": str(checkpoint), "sha256": _sha256(checkpoint)})
        checkpoint.unlink()
    inventory = {
        "schema": "BRAZIL_RV_OOF_CHECKPOINT_CLEANUP_V1",
        "removed": removed,
        "removed_count": len(removed),
        "retained_prediction_archives": True,
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(root / "checkpoint_cleanup.json", inventory)
    return inventory


def run_to_close_oof_extension(
    *,
    base_oof_root: Path,
    target_sidecar: Path,
    output_dir: Path,
    parallel: int = MAX_PARALLEL,
) -> Path:
    """Run the predeclared 50-trajectory four-head extension after adoption."""
    base_oof_root = base_oof_root.resolve()
    target_sidecar = target_sidecar.resolve()
    output_dir = output_dir.resolve()
    if parallel not in (1, 2):
        raise ValueError("OOF extension allows one or two training processes")
    final = output_dir / "result.json"
    if final.is_file():
        audit = _read_json(output_dir / "final_audit.json")
        if audit.get("status") != "passed":
            raise ValueError("Existing to-close OOF extension audit differs")
        return final
    design = _validate_design(base_oof_root)
    store = Path(str(design["store"]["path"])).resolve()
    sidecar = load_nextgen_target_sidecar(target_sidecar, store)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    import shutil

    shutil.copyfile(
        base_oof_root / "purged_folds.json", output_dir / "purged_folds.json"
    )
    extension_design = {
        "schema": "BRAZIL_RV_TO_CLOSE_OOF_EXTENSION_FROZEN_V1",
        "created_at": _now(),
        "repository_commit": repository_commit(),
        "base_oof_frozen_design": {
            "path": str(base_oof_root / "frozen_design.json"),
            "sha256": _sha256(base_oof_root / "frozen_design.json"),
        },
        "store": design["store"],
        "target_sidecar": sidecar.identity,
        "purged_folds_sha256": _sha256(output_dir / "purged_folds.json"),
        "seeds": list(OOF_SEEDS),
        "trajectory_count": 50,
        "monitor": None,
        "final_state": "ema_0995",
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(output_dir / "frozen_design.json", extension_design)
    rows = load_sample_index(store, through=TRAIN_END)
    dates = tuple(rows.get_column("trade_date").unique().sort().to_list())
    folds = load_purged_training_folds(output_dir / "purged_folds.json", dates)
    jobs = []
    for fold in folds.folds:
        for seed in OOF_SEEDS:
            run = _run_path(output_dir, fold.name, seed)
            if _run_manifest_valid(
                run / "run_manifest.json",
                fold,
                seed,
                design["store"]["identity"],
                to_close=True,
            ):
                continue
            jobs.append((store, run, fold.payload(), seed, str(target_sidecar)))
    if parallel == 1:
        for job in jobs:
            print(_train_fold_seed(*job), flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=parallel, mp_context=mp.get_context("spawn")
        ) as executor:
            futures = [executor.submit(_train_fold_seed, *job) for job in jobs]
            for future in as_completed(futures):
                print(future.result(), flush=True)
    archive = _materialize_from_runs(output_dir, design, to_close=True)
    cleanup = _cleanup_checkpoints(output_dir)
    result = {
        "schema": "BRAZIL_RV_TO_CLOSE_OOF_EXTENSION_RESULT_V1",
        "status": "completed",
        "created_at": _now(),
        "archive": archive,
        "trajectory_count": 50,
        "checkpoint_cleanup_removed_count": cleanup["removed_count"],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(final, result)
    _atomic_json(
        output_dir / "final_audit.json",
        {
            "schema": "BRAZIL_RV_TO_CLOSE_OOF_EXTENSION_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(final),
            "all_50_runs_completed": True,
            "all_716_training_dates_covered_once": True,
            "fit_exclusion_chain_verified": True,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return final


def run_program(root: Path, parallel: int = MAX_PARALLEL) -> Path:
    root = root.resolve()
    if parallel not in (1, 2):
        raise ValueError("OOF manufacture allows one or two training processes")
    design = _validate_design(root)
    _execute_jobs(root, design, parallel)
    archive = _materialize(root, design)
    calibration = _calibration(root, design, archive)
    cleanup = _cleanup_checkpoints(root)
    result = {
        "schema": "BRAZIL_RV_OOF_RESULT_V1",
        "status": "completed",
        "created_at": _now(),
        "frozen_design_sha256": _sha256(root / "frozen_design.json"),
        "archive": archive,
        "protocol_calibration": calibration,
        "trajectory_count": 50,
        "checkpoint_cleanup_removed_count": cleanup["removed_count"],
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    _atomic_json(root / "result.json", result)
    _atomic_json(
        root / "final_audit.json",
        {
            "schema": "BRAZIL_RV_OOF_FINAL_AUDIT_V1",
            "status": "passed",
            "result_sha256": _sha256(root / "result.json"),
            "source_manifest_sha256": archive["source_manifest_sha256"],
            "execution_manifest_sha256": archive["execution_manifest_sha256"],
            "all_50_runs_completed": True,
            "all_716_training_dates_covered_once": True,
            "fit_exclusion_chain_verified": True,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
    )
    return root / "result.json"


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manufacture honest TRAIN OOF ranks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--store", type=Path, required=True)
    freeze.add_argument("--experiment54-root", type=Path, required=True)
    freeze.add_argument("--preregistration", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--parallel", type=int, default=MAX_PARALLEL)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    if args.command == "freeze":
        print(
            freeze_program(
                store=args.store,
                experiment54_root=args.experiment54_root,
                preregistration=args.preregistration,
                output_dir=args.output_dir,
            )
        )
    else:
        print(run_program(args.root, args.parallel))


if __name__ == "__main__":
    main()
