from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch

from brazil_rv.preprocessing.intraday_normalization import (
    ARMS,
    PROFILE_SCHEMA,
    build_equity_tod_profile,
    equity_source_hashes,
    load_equity_tod_profile,
    load_source_context,
    parent_artifact_hashes,
    parent_identity,
    sha256_file,
    write_canonical_json,
)
from brazil_rv.preprocessing.intraday_normalization_diagnostics import (
    run_heteroskedasticity_diagnostics,
    validate_heteroskedasticity_diagnostics,
)
from brazil_rv.preprocessing.intraday_normalization_variants import (
    build_intraday_normalization_variants,
    validate_intraday_normalization_variant,
)

from .analyze_stock_time_attribution import _cache_path, analyze_runs
from .contract import (
    BASELINE_TCN_SETTINGS,
    FEATURE_STORE_POINTER,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    architecture_for_model,
)
from .data import (
    feature_store_identity,
    load_sample_index,
    sample_window_metadata,
    select_sample_split,
)
from .engine import objective_metadata, sam_metadata
from .intraday_normalization_comparison import (
    SEEDS,
    consolidate_intraday_normalization_stage,
    validate_intraday_normalization_comparison,
)
from .model import build_neural_model, count_trainable_parameters
from .provenance import build_run_provenance, repository_commit
from .stage_validation import validate_completed_run
from .train import _run_neural, parse_args as parse_training_args, set_seeds

STAGE_SCHEMA = "EQUITY_INTRADAY_NORMALIZATION_STAGE_V1"
REMOTE_COMMAND = (
    "cd /home/ubuntu/Brazil-RV/quant/b3-quant/research && "
    "uv run --frozen --no-default-groups python -m "
    "brazil_rv.modeling.run_intraday_normalization_stage --output-dir "
    "/lambda/nfs/brazil-rv-east3/quant-data/b3/processed/model_runs/"
    "intraday_normalization_v1"
)


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the validation-only equity intraday normalization stage"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(arguments)


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Stage:
    def __init__(self, output_dir: Path, parent: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "stage_manifest.json"
        self.commit = repository_commit()
        self.parent_identity = feature_store_identity(parent)
        self.logger = logging.getLogger(f"intraday-normalization-{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (
            logging.StreamHandler(),
            logging.FileHandler(self.output_dir / "launcher.log", encoding="utf-8"),
        ):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if (
                self.manifest.get("schema") != STAGE_SCHEMA
                or self.manifest.get("repository_commit") != self.commit
                or self.manifest.get("parent_feature_store") != self.parent_identity
            ):
                raise ValueError(
                    "Existing stage has incompatible code or parent feature store"
                )
            if not isinstance(self.manifest.get("archive_history"), list):
                raise ValueError("Stage archive history must be an append-only list")
        else:
            self.manifest = {
                "schema": STAGE_SCHEMA,
                "status": "running",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "repository_commit": self.commit,
                "parent_feature_store": self.parent_identity,
                "arms": ARMS,
                "seeds": list(SEEDS),
                "training_run_count": len(ARMS) * len(SEEDS),
                "split_boundaries": {
                    "train": [str(TRAIN_START), str(TRAIN_END)],
                    "validation": [str(VALIDATION_START), str(VALIDATION_END)],
                },
                "test_accessed": False,
                "command": REMOTE_COMMAND,
                "steps": {},
                "archive_history": [],
            }
            self.write()

    def write(self) -> None:
        write_canonical_json(self.manifest_path, self.manifest)

    @staticmethod
    def _artifact_candidates(artifact: Path) -> tuple[Path, ...]:
        candidates = [artifact]
        deterministic_partial = artifact.with_name(f"{artifact.name}.partial")
        if deterministic_partial != artifact:
            candidates.append(deterministic_partial)
        candidates.extend(
            path
            for path in artifact.parent.glob(f"{artifact.name}.*.partial")
            if path not in candidates
        )
        return tuple(candidates)

    def _archive_artifacts(self, name: str, artifacts: tuple[Path, ...]) -> None:
        archived_at = datetime.now(timezone.utc)
        events: list[dict[str, object]] = []
        for artifact in artifacts:
            for candidate in self._artifact_candidates(artifact):
                if not candidate.exists():
                    continue
                base = candidate.with_name(
                    f"{candidate.name}.incomplete.{archived_at:%Y%m%dT%H%M%S%fZ}"
                )
                archived = base
                collision = 1
                while archived.exists():
                    archived = base.with_name(f"{base.name}.{collision}")
                    collision += 1
                candidate.rename(archived)
                events.append(
                    {
                        "step": name,
                        "source": str(candidate),
                        "archive": str(archived),
                        "archived_at_utc": archived_at.isoformat(),
                    }
                )
        self.manifest["archive_history"].extend(events)
        self.write()

    def step(
        self,
        name: str,
        config: dict[str, object],
        artifacts: tuple[Path, ...],
        action: Callable[[], None],
        validator: Callable[[], None],
    ) -> None:
        fingerprint = _fingerprint(
            {
                "repository_commit": self.commit,
                "parent_feature_store": self.parent_identity,
                "config": config,
            }
        )
        existing = self.manifest["steps"].get(name)
        if existing is not None and existing.get("status") == "completed":
            if existing.get("fingerprint") != fingerprint:
                raise ValueError(f"Completed step {name} has incompatible provenance")
            if any(not path.exists() for path in artifacts):
                raise ValueError(f"Completed step {name} is missing an artifact")
            validator()
            self.logger.info("reuse completed step %s", name)
            return
        if existing is not None:
            if existing.get("status") not in ("running", "failed"):
                raise ValueError(f"Ambiguous prior status for step {name}")
            self._archive_artifacts(name, artifacts)
        elif any(
            candidate.exists()
            for artifact in artifacts
            for candidate in self._artifact_candidates(artifact)
        ):
            raise ValueError(f"Untracked artifact makes step {name} ambiguous")
        self.logger.info("start step %s", name)
        started = time.perf_counter()
        self.manifest["steps"][name] = {
            "status": "running",
            "fingerprint": fingerprint,
            "config": config,
            "artifacts": [str(path) for path in artifacts],
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.write()
        try:
            action()
            if any(not path.exists() for path in artifacts):
                raise RuntimeError(f"Step {name} did not emit every artifact")
            validator()
        except BaseException as error:
            self.manifest["steps"][name].update(
                {
                    "status": "failed",
                    "error": repr(error),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            self.write()
            raise
        self.manifest["steps"][name].update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        self.write()
        self.logger.info("completed step %s", name)


def _training_args(seed: int) -> argparse.Namespace:
    return parse_training_args(["--seed", str(seed)])


def _expected_run_provenance(
    seed: int,
    store: Path,
    train_rows: pl.DataFrame,
    validation_rows: pl.DataFrame,
) -> dict[str, object]:
    args = _training_args(seed)
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    with torch.random.fork_rng(devices=[]):
        parameter_count = count_trainable_parameters(
            build_neural_model("tcn", architecture, args.peer_features)
        )
    return build_run_provenance(
        repository_commit_value=repository_commit(),
        feature_store=store,
        feature_store_metadata=feature_store_identity(store),
        model_name="tcn",
        architecture=architecture,
        settings=BASELINE_TCN_SETTINGS,
        peer_features=args.peer_features,
        global_context=args.global_context,
        objective=objective_metadata(args.objective, args.temperature),
        optimizer=args.optimizer,
        sam=sam_metadata(args.optimizer, args.sam_rho),
        seed=seed,
        training_horizon="all",
        selection_horizon="all",
        context_family_ablation="none",
        fit_window=sample_window_metadata(train_rows, "train"),
        selection_window=sample_window_metadata(validation_rows, "validation"),
        allow_date_replacement=False,
        parameter_count=parameter_count,
        training_sample_count=train_rows.height,
    )


def _preflight(parent: Path, output_path: Path) -> None:
    context = load_source_context(parent)
    if context.manifest.get("contract_version") != (
        "M1_FEATURES_INTRADAY_DI_MASKED_CONTEXT_HUMAN_PRIORS_V4"
    ):
        raise ValueError("The normalization stage requires canonical feature V4")
    sample_index = load_sample_index(parent)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    if train_rows.is_empty() or validation_rows.is_empty():
        raise ValueError("Train/validation split is empty")
    if validation_rows["trade_date"].max() != VALIDATION_END:
        raise ValueError("Validation rows do not end at the contract boundary")
    args = _training_args(SEEDS[0])
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    parameter_count = count_trainable_parameters(
        build_neural_model("tcn", architecture, args.peer_features)
    )
    summary = {
        "parent_feature_store": parent_identity(context),
        "parent_contract_version": context.manifest["contract_version"],
        "raw_identity_validation": "passed",
        "allowed_date_count": context.allowed_date_count,
        "allowed_date_end": str(context.market_dates[context.allowed_date_count - 1]),
        "test_accessed": False,
        "train_window": sample_window_metadata(train_rows, "train"),
        "validation_window": sample_window_metadata(validation_rows, "validation"),
        "incumbent": {
            "model": "tcn",
            "settings": {
                "fusion": BASELINE_TCN_SETTINGS.fusion,
                "width": BASELINE_TCN_SETTINGS.width,
                "receptive_field": BASELINE_TCN_SETTINGS.receptive_field,
                "block": BASELINE_TCN_SETTINGS.block,
                "slow_routing": BASELINE_TCN_SETTINGS.slow_routing,
                "macro_temporal_routing": BASELINE_TCN_SETTINGS.macro_temporal_routing,
                "readout": BASELINE_TCN_SETTINGS.readout,
            },
            "peer_features": args.peer_features,
            "global_context": args.global_context,
            "objective": objective_metadata(args.objective, args.temperature),
            "optimizer": args.optimizer,
            "sam": sam_metadata(args.optimizer, args.sam_rho),
            "parameter_count": parameter_count,
        },
    }
    write_canonical_json(output_path, summary)


def _validate_preflight(parent: Path, output_path: Path) -> None:
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    if summary.get("test_accessed") is not False:
        raise ValueError("Preflight is not validation-only")
    if summary.get("parent_feature_store") != feature_store_identity(parent):
        raise ValueError("Preflight parent feature-store identity changed")
    if summary["validation_window"]["end"] != str(VALIDATION_END):
        raise ValueError("Preflight validation boundary changed")


def _validate_profile(parent: Path, profile_dir: Path) -> None:
    manifest, _ = load_equity_tod_profile(profile_dir)
    context = load_source_context(parent)
    if manifest.get("schema") != PROFILE_SCHEMA:
        raise ValueError("Wrong profile schema")
    if manifest.get("repository_commit") != repository_commit():
        raise ValueError("Profile repository commit mismatch")
    if manifest["parent_feature_store"] != parent_identity(context):
        raise ValueError("Profile parent identity mismatch")
    if manifest["parent_artifact_sha256"] != parent_artifact_hashes(context):
        raise ValueError("Profile parent hashes mismatch")
    if manifest["equity_source_sha256"] != equity_source_hashes(context):
        raise ValueError("Profile raw-source hashes mismatch")


def _write_cache_manifest(
    cache_dir: Path, run_dirs: dict[tuple[str, int], Path]
) -> None:
    entries = {}
    for (arm, seed), run_dir in run_dirs.items():
        path = _cache_path(cache_dir, run_dir)
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[f"{arm}_seed{seed}"] = {
            "run_path": str(run_dir.resolve()),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    write_canonical_json(
        cache_dir / "cache_manifest.json",
        {"test_accessed": False, "entries": entries},
    )


def _validate_attribution(
    attribution_dir: Path,
    cache_dir: Path,
    run_dirs: dict[tuple[str, int], Path],
) -> None:
    summary = json.loads((attribution_dir / "summary.json").read_text(encoding="utf-8"))
    expected = [str(path.resolve()) for path in run_dirs.values()]
    if (
        summary.get("split") != "validation"
        or summary.get("test_accessed") is not False
    ):
        raise ValueError("Attribution is not validation-only")
    if summary.get("runs") != expected:
        raise ValueError("Attribution run matrix mismatch")
    cache_manifest = json.loads(
        (cache_dir / "cache_manifest.json").read_text(encoding="utf-8")
    )
    if cache_manifest.get("test_accessed") is not False or set(
        cache_manifest.get("entries", {})
    ) != {f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS}:
        raise ValueError("Validation prediction-cache manifest mismatch")
    for entry in cache_manifest["entries"].values():
        path = Path(entry["path"])
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Validation prediction-cache hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {
                "predictions",
                "targets",
                "raw_returns",
                "label_mask",
                "date_idx",
                "decision_idx",
            }:
                raise ValueError(f"Validation prediction-cache schema mismatch: {path}")
            shape = values["predictions"].shape
            if (
                len(shape) != 3
                or values["targets"].shape != shape
                or values["raw_returns"].shape != shape
                or values["label_mask"].shape != shape
                or values["date_idx"].shape != (shape[0],)
                or values["decision_idx"].shape != (shape[0],)
            ):
                raise ValueError(f"Validation prediction-cache shape mismatch: {path}")
            for name in ("predictions", "targets", "raw_returns"):
                if not np.isfinite(values[name]).all():
                    raise ValueError(
                        f"Non-finite validation prediction cache: {path}/{name}"
                    )
    for name in (
        "time_of_day_5m.csv",
        "time_of_day_30m.csv",
        "horizon_attribution.csv",
        "bootstrap_summary.csv",
    ):
        frame = pl.read_csv(attribution_dir / name)
        if frame.is_empty():
            raise ValueError(f"Empty attribution artifact: {name}")
        for column, dtype in frame.schema.items():
            if dtype.is_numeric() and not np.isfinite(frame[column].to_numpy()).all():
                raise ValueError(f"Non-finite attribution artifact: {name}/{column}")


def _root_summary(comparisons: Path, root_json: Path, root_markdown: Path) -> None:
    shutil.copy2(comparisons / "stage_summary.json", root_json)
    shutil.copy2(comparisons / "stage_summary.md", root_markdown)


def _validate_root_summary(
    comparisons: Path, root_json: Path, root_markdown: Path
) -> None:
    validate_intraday_normalization_comparison(comparisons)
    for source, destination in (
        (comparisons / "stage_summary.json", root_json),
        (comparisons / "stage_summary.md", root_markdown),
    ):
        if source.read_bytes() != destination.read_bytes():
            raise ValueError("Root stage summary differs from validated comparison")


def run_stage(output_dir: Path) -> Path:
    parent = Path(FEATURE_STORE_POINTER.read_text(encoding="utf-8").strip()).resolve()
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    stage = Stage(output_dir, parent)
    preflight_path = stage.output_dir / "preflight.json"
    stage.step(
        "preflight",
        {"parent": feature_store_identity(parent), "arms": ARMS, "seeds": SEEDS},
        (preflight_path,),
        lambda: _preflight(parent, preflight_path),
        lambda: _validate_preflight(parent, preflight_path),
    )

    profile_dir = stage.output_dir / "profiles"
    stage.step(
        "profile",
        {
            "bin_minutes": 30,
            "prior_session_equivalents": 20.0,
            "freeze_date": str(TRAIN_END),
        },
        (profile_dir,),
        lambda: build_equity_tod_profile(parent, profile_dir),
        lambda: _validate_profile(parent, profile_dir),
    )

    variants_base = stage.output_dir / "feature_variants"
    candidate_dirs = {arm: variants_base / arm for arm in tuple(ARMS)[1:]}
    stage.step(
        "feature_variants",
        {
            "parent": feature_store_identity(parent),
            "profile": str(profile_dir.resolve()),
            "arms": {arm: ARMS[arm] for arm in tuple(ARMS)[1:]},
        },
        tuple(candidate_dirs.values()),
        lambda: build_intraday_normalization_variants(
            parent, profile_dir, variants_base
        ),
        lambda: [
            validate_intraday_normalization_variant(path)
            for path in candidate_dirs.values()
        ],
    )
    stores = {"legacy_daily_vol": parent, **candidate_dirs}

    diagnostics_dir = stage.output_dir / "diagnostics"
    stage.step(
        "heteroskedasticity_diagnostics",
        {"stores": {arm: feature_store_identity(path) for arm, path in stores.items()}},
        (diagnostics_dir,),
        lambda: run_heteroskedasticity_diagnostics(
            stores, profile_dir, diagnostics_dir
        ),
        lambda: validate_heteroskedasticity_diagnostics(diagnostics_dir),
    )

    sample_index = load_sample_index(parent)
    train_rows = select_sample_split(sample_index, "train")
    validation_rows = select_sample_split(sample_index, "validation")
    runs_base = stage.output_dir / "runs"
    runs_base.mkdir(exist_ok=True)
    run_dirs: dict[tuple[str, int], Path] = {}
    for arm, store in stores.items():
        for seed in SEEDS:
            run_dir = runs_base / f"{arm}_seed{seed}"
            run_dirs[(arm, seed)] = run_dir
            expected = _expected_run_provenance(
                seed, store, train_rows, validation_rows
            )

            def train(
                run_dir: Path = run_dir,
                store: Path = store,
                seed: int = seed,
            ) -> None:
                run_dir.mkdir(parents=True, exist_ok=False)
                set_seeds(seed)
                _run_neural(
                    _training_args(seed),
                    store,
                    train_rows,
                    validation_rows,
                    run_dir,
                )

            stage.step(
                f"train_{arm}_seed{seed}",
                expected,
                (run_dir,),
                train,
                lambda run_dir=run_dir, store=store, expected=expected: (
                    validate_completed_run(run_dir, store, expected)
                ),
            )

    attribution_dir = stage.output_dir / "validation_attribution"
    cache_dir = stage.output_dir / "validation_prediction_cache"
    ordered_runs = list(run_dirs.values())

    def attribution() -> None:
        analyze_runs(ordered_runs, attribution_dir, cache_dir)
        _write_cache_manifest(cache_dir, run_dirs)

    stage.step(
        "validation_attribution",
        {"runs": [str(path.resolve()) for path in ordered_runs]},
        (attribution_dir, cache_dir),
        attribution,
        lambda: _validate_attribution(attribution_dir, cache_dir, run_dirs),
    )

    comparisons = stage.output_dir / "comparisons"
    root_json = stage.output_dir / "stage_summary.json"
    root_markdown = stage.output_dir / "stage_summary.md"

    def consolidate() -> None:
        consolidate_intraday_normalization_stage(
            run_dirs,
            attribution_dir,
            diagnostics_dir,
            comparisons,
            cache_dir,
        )
        _root_summary(comparisons, root_json, root_markdown)

    stage.step(
        "consolidation",
        {
            "runs": [
                feature_store_identity(stores[arm]) for arm in ARMS for _ in SEEDS
            ],
            "bootstrap": {
                "block_days": 5,
                "replications": 10_000,
                "seed": 20260815,
            },
        },
        (comparisons, root_json, root_markdown),
        consolidate,
        lambda: _validate_root_summary(comparisons, root_json, root_markdown),
    )
    stage.manifest.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_accessed": False,
        }
    )
    stage.write()
    return stage.output_dir


def main() -> None:
    args = parse_args()
    print(run_stage(args.output_dir))


if __name__ == "__main__":
    main()
