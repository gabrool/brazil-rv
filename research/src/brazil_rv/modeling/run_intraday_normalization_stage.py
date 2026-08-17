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
    build_equity_tod_profile,
    load_source_context,
    parent_identity,
    sha256_file,
    validate_equity_tod_profile,
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

from .analyze_stock_time_attribution import (
    _cache_path,
    analyze_runs,
    horizon_attribution,
    load_attribution_inputs,
    primary_time_bins,
    time_of_day_30m_attribution,
    time_of_day_attribution,
)
from .contract import (
    BASELINE_TCN_SETTINGS,
    EQUITY_COUNT,
    EXPECTED_DECISIONS_PER_DATE,
    FEATURE_STORE_POINTER,
    HORIZONS,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    architecture_for_model,
    workspace_path,
)
from .data import (
    feature_store_identity,
    load_sample_index,
    sample_window_metadata,
    select_sample_split,
)
from .engine import objective_metadata, sam_metadata
from .feature_variant import load_variant_manifest
from .intraday_normalization_comparison import (
    SEEDS,
    consolidate_intraday_normalization_stage,
    validate_intraday_normalization_comparison,
)
from .metrics import create_metric_table
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


def _stage_store_identity(store: Path, context) -> dict[str, object]:
    store = store.resolve()
    parent = context.parent.resolve()
    development_parent = parent_identity(context)
    if store == parent:
        return development_parent
    manifest = load_variant_manifest(store)
    if manifest is None:
        raise ValueError("Normalization stage store is neither parent nor candidate")
    if manifest.get("canonical_parent_feature_store") != development_parent:
        raise ValueError("Candidate is not bound to the development parent identity")
    return feature_store_identity(store)


class Stage:
    def __init__(
        self,
        output_dir: Path,
        parent: Path,
        *,
        parent_identity_value: dict[str, object] | None = None,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "stage_manifest.json"
        self.commit = repository_commit()
        self.parent_identity = (
            feature_store_identity(parent)
            if parent_identity_value is None
            else parent_identity_value
        )
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
    store_identity: dict[str, object],
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
        feature_store_metadata=store_identity,
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
    sample_index = load_sample_index(parent, end_date=VALIDATION_END)
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
        "parent_feature_store": _stage_store_identity(parent, context),
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
    context = load_source_context(parent)
    if summary.get("test_accessed") is not False:
        raise ValueError("Preflight is not validation-only")
    if summary.get("parent_feature_store") != _stage_store_identity(parent, context):
        raise ValueError("Preflight parent feature-store identity changed")
    if summary["validation_window"]["end"] != str(VALIDATION_END):
        raise ValueError("Preflight validation boundary changed")


def _validate_profile(parent: Path, profile_dir: Path) -> None:
    context = load_source_context(parent)
    validate_equity_tod_profile(profile_dir, expected_context=context)


def _assert_stage_frame(
    actual: pl.DataFrame,
    expected: pl.DataFrame,
    sort_by: tuple[str, ...],
    label: str,
) -> None:
    if actual.columns != expected.columns or actual.height != expected.height:
        raise ValueError(f"{label} schema or row count mismatch")
    if sort_by:
        actual = actual.sort(list(sort_by))
        expected = expected.sort(list(sort_by))
    for name in actual.columns:
        actual_null = actual[name].is_null().to_numpy()
        expected_null = expected[name].is_null().to_numpy()
        if not np.array_equal(actual_null, expected_null):
            raise ValueError(f"{label} null mismatch: {name}")
        left = actual[name].drop_nulls()
        right = expected[name].drop_nulls()
        if actual.schema[name].is_numeric() and expected.schema[name].is_numeric():
            equal = np.allclose(
                left.to_numpy(),
                right.to_numpy(),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            equal = left.to_list() == right.to_list()
        if not equal:
            raise ValueError(f"{label} value mismatch: {name}")


def _require_metric_summary_close(
    actual: dict[str, object], expected: dict[str, object], label: str
) -> None:
    for name in ("primary_score", "mean_valid_sample_spearman_ic"):
        if not np.isclose(actual[name], expected[name], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{label} metric does not reconstruct: {name}")
    actual_horizons = {int(row["horizon_minutes"]): row for row in actual["horizons"]}
    expected_horizons = {
        int(row["horizon_minutes"]): row for row in expected["horizons"]
    }
    if set(actual_horizons) != set(HORIZONS) or set(expected_horizons) != set(HORIZONS):
        raise ValueError(f"{label} horizon lattice is invalid")
    for horizon, expected_row in expected_horizons.items():
        actual_row = actual_horizons[horizon]
        for name, value in expected_row.items():
            if name == "horizon_minutes":
                continue
            if not np.isclose(actual_row[name], value, rtol=1e-10, atol=1e-12):
                raise ValueError(
                    f"{label} horizon metric does not reconstruct: {horizon}/{name}"
                )


def _write_cache_manifest(
    cache_dir: Path, run_dirs: dict[tuple[str, int], Path]
) -> None:
    entries = {}
    for (arm, seed), run_dir in run_dirs.items():
        path = _cache_path(cache_dir, run_dir)
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[f"{arm}_seed{seed}"] = {
            "arm": arm,
            "seed": seed,
            "run_path": str(run_dir.resolve()),
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    write_canonical_json(
        cache_dir / "cache_manifest.json",
        {
            "schema": "INTRADAY_NORMALIZATION_VALIDATION_CACHE_V1",
            "test_accessed": False,
            "entries": entries,
        },
    )


def _validate_attribution_semantics(
    attribution_dir: Path,
    cache_dir: Path,
    run_dirs: dict[tuple[str, int], Path],
) -> None:
    expected_matrix = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(run_dirs) != expected_matrix:
        raise ValueError("Attribution requires the exact nine-run matrix")
    cache_manifest = json.loads(
        (cache_dir / "cache_manifest.json").read_text(encoding="utf-8")
    )
    if cache_manifest.get("schema") != "INTRADAY_NORMALIZATION_VALIDATION_CACHE_V1":
        raise ValueError("Validation prediction-cache schema mismatch")
    expected_time_5m: list[pl.DataFrame] = []
    expected_time_30m: list[pl.DataFrame] = []
    expected_horizon: list[pl.DataFrame] = []
    cache_fields = {
        "predictions",
        "targets",
        "raw_returns",
        "label_mask",
        "date_idx",
        "decision_idx",
    }
    for arm in ARMS:
        for seed in SEEDS:
            key = f"{arm}_seed{seed}"
            run_dir = run_dirs[(arm, seed)].resolve()
            if run_dir.name != key:
                raise ValueError(
                    f"Attribution run directory has the wrong name: {run_dir}"
                )
            entry = cache_manifest["entries"][key]
            cache_path = _cache_path(cache_dir, run_dir).resolve()
            if set(entry) != {"arm", "seed", "run_path", "path", "sha256"} or (
                entry["arm"] != arm
                or int(entry["seed"]) != seed
                or entry["run_path"] != str(run_dir)
                or entry["path"] != str(cache_path)
            ):
                raise ValueError(f"Validation cache-to-run mapping mismatch: {key}")
            run_manifest = json.loads(
                (run_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            if int(run_manifest.get("seed", -1)) != seed:
                raise ValueError(f"Validation cache seed mismatch: {key}")
            store = Path(run_manifest["feature_store"])
            variant = load_variant_manifest(store)
            if arm == "legacy_daily_vol":
                if variant is not None:
                    raise ValueError("Legacy attribution run uses a candidate store")
            elif variant is None or variant.get("arm") != arm:
                raise ValueError(f"Candidate attribution store mismatch: {key}")
            expected_rows = select_sample_split(
                load_sample_index(store, end_date=VALIDATION_END), "validation"
            ).sort("sample_id")
            with np.load(cache_path, allow_pickle=False) as values:
                if set(values.files) != cache_fields:
                    raise ValueError(
                        f"Validation prediction-cache schema mismatch: {cache_path}"
                    )
                cached = {name: values[name] for name in values.files}
            shape = cached["predictions"].shape
            expected_shape = (expected_rows.height, EQUITY_COUNT, len(HORIZONS))
            if shape != expected_shape or any(
                cached[name].shape != shape
                for name in ("targets", "raw_returns", "label_mask")
            ):
                raise ValueError(
                    f"Validation prediction-cache shape mismatch: {cache_path}"
                )
            for name in ("predictions", "targets", "raw_returns"):
                if (
                    cached[name].dtype != np.float32
                    or not np.isfinite(cached[name]).all()
                ):
                    raise ValueError(
                        f"Invalid validation prediction cache: {cache_path}/{name}"
                    )
            if cached["label_mask"].dtype != np.bool_:
                raise ValueError(f"Validation label-mask dtype mismatch: {cache_path}")
            expected_dates = expected_rows["date_idx"].to_numpy().astype(np.int64)
            expected_decisions = (
                expected_rows["decision_idx"].to_numpy().astype(np.int64)
            )
            if (
                cached["date_idx"].dtype != np.int64
                or cached["decision_idx"].dtype != np.int64
            ):
                raise ValueError(f"Validation cache index dtype mismatch: {cache_path}")
            if not np.array_equal(
                cached["date_idx"], expected_dates
            ) or not np.array_equal(cached["decision_idx"], expected_decisions):
                raise ValueError(
                    f"Validation cache date/decision identity mismatch: {cache_path}"
                )
            if set(np.unique(cached["decision_idx"])) != set(
                range(EXPECTED_DECISIONS_PER_DATE)
            ):
                raise ValueError(
                    f"Validation cache decision coverage mismatch: {cache_path}"
                )
            reconstructed_metrics, reconstructed_daily = create_metric_table(
                cached["predictions"],
                cached["targets"],
                cached["raw_returns"],
                cached["label_mask"],
                cached["date_idx"],
                cached["decision_idx"],
            )
            recorded_metrics = json.loads(
                (run_dir / "validation_metrics.json").read_text(encoding="utf-8")
            )
            _require_metric_summary_close(recorded_metrics, reconstructed_metrics, key)
            _assert_stage_frame(
                pl.read_parquet(run_dir / "validation_daily_metrics.parquet"),
                pl.DataFrame(reconstructed_daily),
                ("date_idx", "horizon_minutes"),
                f"validation daily metrics {key}",
            )
            inputs = load_attribution_inputs(run_dir, cache_dir)
            expected_time_5m.append(time_of_day_attribution(inputs))
            expected_time_30m.append(time_of_day_30m_attribution(inputs))
            expected_horizon.append(horizon_attribution(inputs))

    reconstructed = {
        "time_of_day_5m": pl.concat(expected_time_5m),
        "time_of_day_30m": pl.concat(expected_time_30m),
        "horizon_attribution": pl.concat(expected_horizon),
    }
    keys_by_stem = {
        "time_of_day_5m": ("run", "decision_idx", "horizon_minutes"),
        "time_of_day_30m": ("run", "time_bin_30m", "horizon_minutes"),
        "horizon_attribution": ("run", "horizon_minutes"),
    }
    for stem, expected in reconstructed.items():
        keys = keys_by_stem[stem]
        actual = pl.read_csv(attribution_dir / f"{stem}.csv")
        _assert_stage_frame(actual, expected, keys, stem)
        if actual.select(keys).n_unique() != actual.height:
            raise ValueError(f"Attribution endpoint lattice has duplicates: {stem}")
    if reconstructed["time_of_day_5m"].height != (
        len(run_dirs) * EXPECTED_DECISIONS_PER_DATE * len(HORIZONS)
    ) or reconstructed["time_of_day_30m"].height != (
        len(run_dirs) * len(primary_time_bins()) * len(HORIZONS)
    ):
        raise ValueError("Attribution time-bin coverage is incomplete")

    output_stems = (
        "stock_attribution",
        "time_of_day_5m",
        "time_of_day_30m",
        "horizon_attribution",
        "opening_regimes",
        "opening_context",
        "bootstrap_summary",
    )
    summary = json.loads((attribution_dir / "summary.json").read_text(encoding="utf-8"))
    for stem in output_stems:
        csv = pl.read_csv(attribution_dir / f"{stem}.csv")
        parquet = pl.read_parquet(attribution_dir / f"{stem}.parquet")
        _assert_stage_frame(csv, parquet, (), f"attribution CSV/Parquet {stem}")
        if summary["outputs"].get(stem) != csv.height:
            raise ValueError(f"Attribution summary count mismatch: {stem}")
    if set(summary["outputs"]) != set(output_stems):
        raise ValueError("Attribution summary output inventory mismatch")


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
    _validate_attribution_semantics(attribution_dir, cache_dir, run_dirs)


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
    parent = workspace_path(FEATURE_STORE_POINTER.read_text(encoding="utf-8").strip())
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    context = load_source_context(parent)
    development_parent = _stage_store_identity(parent, context)
    stage = Stage(output_dir, parent, parent_identity_value=development_parent)
    preflight_path = stage.output_dir / "preflight.json"
    stage.step(
        "preflight",
        {"parent": development_parent, "arms": ARMS, "seeds": SEEDS},
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
            "parent": development_parent,
            "profile": str(profile_dir.resolve()),
            "arms": {arm: ARMS[arm] for arm in tuple(ARMS)[1:]},
        },
        tuple(candidate_dirs.values()),
        lambda: build_intraday_normalization_variants(
            parent, profile_dir, variants_base
        ),
        lambda: [
            validate_intraday_normalization_variant(path, arm)
            for arm, path in candidate_dirs.items()
        ],
    )
    stores = {"legacy_daily_vol": parent, **candidate_dirs}
    store_identities = {
        arm: _stage_store_identity(path, context) for arm, path in stores.items()
    }

    diagnostics_dir = stage.output_dir / "diagnostics"
    stage.step(
        "heteroskedasticity_diagnostics",
        {"stores": store_identities},
        (diagnostics_dir,),
        lambda: run_heteroskedasticity_diagnostics(
            stores, profile_dir, diagnostics_dir
        ),
        lambda: validate_heteroskedasticity_diagnostics(diagnostics_dir),
    )

    sample_index = load_sample_index(parent, end_date=VALIDATION_END)
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
                seed, store, store_identities[arm], train_rows, validation_rows
            )

            def train(
                run_dir: Path = run_dir,
                store: Path = store,
                seed: int = seed,
                store_identity: dict[str, object] = store_identities[arm],
            ) -> None:
                run_dir.mkdir(parents=True, exist_ok=False)
                set_seeds(seed)
                _run_neural(
                    _training_args(seed),
                    store,
                    train_rows,
                    validation_rows,
                    run_dir,
                    feature_store_metadata=store_identity,
                )

            stage.step(
                f"train_{arm}_seed{seed}",
                expected,
                (run_dir,),
                train,
                lambda run_dir=run_dir, store=store, expected=expected, store_identity=store_identities[arm]: (
                    validate_completed_run(
                        run_dir,
                        store,
                        expected,
                        feature_store_metadata=store_identity,
                    )
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
            "runs": [store_identities[arm] for arm in ARMS for _ in SEEDS],
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
