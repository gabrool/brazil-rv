from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
from numpy.lib.format import open_memmap

from ..preprocessing.contract import (
    EQUITY_SESSION_MINUTES,
    EQUITY_SESSION_START_MINUTE,
)
from ..preprocessing.io import (
    cotahist_files,
    load_assignments,
    load_market_dates_and_security_dates,
    load_source_file,
    prepare_session_bars,
    read_research_interval,
    resolve_inputs,
    validate_physical_source_identity,
    validate_source_date_isolation,
)
from .analyze import BOOTSTRAP_REPLICATIONS
from .contract import (
    EQUITY_COUNT,
    HORIZONS,
    MIN_IC_EQUITIES,
    RUN_OUTPUT_BASE,
    TRAIN_END,
)
from .data import (
    discovery_folds,
    feature_store_identity,
    load_sample_index,
    resolve_feature_store,
    select_sample_split,
)
from .metrics import (
    average_ranks,
    daily_horizon_ic,
    finite_mean,
    moving_block_bootstrap,
    per_date_primary_ic,
    primary_validation_score,
    rank_average_predictions,
    sample_level_spearman_ic,
)

UPSTREAM_REPOSITORY = "https://github.com/shiyu-coder/Kronos"
UPSTREAM_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
TOKENIZER_REPOSITORY = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MODEL_REVISIONS = {
    "Kronos-small": (
        "NeoQuasar/Kronos-small",
        "901c26c1332695a2a8f243eb2f37243a37bea320",
    ),
    "Kronos-base": (
        "NeoQuasar/Kronos-base",
        "2b554741eca47781b64468546e77fef3e85130e6",
    ),
}
PARENT_CAMPAIGN = RUN_OUTPUT_BASE / "trajectory_discovery_e22dd67_20260819T134332Z"
PARENT_CROSSFIT = RUN_OUTPUT_BASE / "trajectory_crossfit_3054228_20260819T161200Z"

CONTEXT_BARS = 512
SESSION_BARS = EQUITY_SESSION_MINUTES // 5
BAR_FIELDS = ("open", "high", "low", "close", "volume")
DECISIONS = (0, 10, 20, 30, 40, 50)
REDUCED_BASE_DECISIONS = (0, 20, 40)
PREDICTION_BARS = 24
TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = 0
SAMPLE_COUNT = 5
FULL_CONTEXT_OBSERVED = math.ceil(0.80 * CONTEXT_BARS)
RECENT_CONTEXT_OBSERVED = math.ceil(0.95 * 24)
FOLD_RANGES = {
    "fold_a": (date(2023, 9, 1), date(2024, 1, 31)),
    "fold_b": (date(2024, 2, 1), date(2024, 6, 28)),
}
GLOBAL_SEED = 20260822


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez(output, **values)
    os.replace(temporary, path)


def _array_metadata(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _repository_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stable_context_seed(
    model_name: str, trade_date: date, decision_idx: int, security_id: str
) -> int:
    payload = (
        f"{model_name}\0{trade_date.isoformat()}\0{decision_idx}\0{security_id}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def aggregate_five_minute_bars(bars: pl.DataFrame) -> pl.DataFrame:
    """Aggregate sorted, validated M1 rows into canonical [t, t+5) bars."""
    if bars.is_empty():
        return pl.DataFrame(
            schema={
                "date_idx": pl.Int32,
                "bar_idx": pl.Int16,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "observed_minutes": pl.Int16,
            }
        )
    return (
        bars.with_columns((pl.col("minute_idx") // 5).cast(pl.Int16).alias("bar_idx"))
        .group_by("date_idx", "bar_idx", maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("real_volume").sum().cast(pl.Float64).alias("volume"),
            pl.len().cast(pl.Int16).alias("observed_minutes"),
        )
        .sort("date_idx", "bar_idx")
    )


def _scope_rows(store: Path) -> pl.DataFrame:
    samples = load_sample_index(store, through=TRAIN_END)
    training = select_sample_split(samples, "train")
    rows = []
    for fold in discovery_folds(training):
        start, end = FOLD_RANGES[fold.name]
        selected = fold.selection_rows.filter(pl.col("decision_idx").is_in(DECISIONS))
        if (
            selected.get_column("trade_date").min() != start
            or selected.get_column("trade_date").max() != end
            or selected.get_column("trade_date").n_unique() != 102
            or selected.height != 102 * len(DECISIONS)
        ):
            raise ValueError(f"{fold.name} differs from the preregistered scope")
        rows.append(selected.with_columns(pl.lit(fold.name).alias("fold")))
    result = pl.concat(rows).sort("date_idx", "decision_idx")
    if result.get_column("trade_date").max() > TRAIN_END:
        raise ValueError("K0 scope crossed the training cutoff")
    return result.with_row_index("scope_idx").with_columns(
        pl.col("scope_idx").cast(pl.Int32)
    )


def _fill_security_bars(
    output: np.memmap,
    synthetic: np.memmap,
    available: np.memmap,
    *,
    slot: int,
    valid_date_indices: Sequence[int],
    observed: pl.DataFrame,
    sidecar_date_count: int,
) -> None:
    observed_by_key = {
        (int(row["date_idx"]), int(row["bar_idx"])): row
        for row in observed.iter_rows(named=True)
    }
    previous_close: float | None = None
    for date_idx in sorted(valid_date_indices):
        if date_idx >= sidecar_date_count:
            continue
        for bar_idx in range(SESSION_BARS):
            row = observed_by_key.get((date_idx, bar_idx))
            if row is not None:
                values = (
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                )
                output[date_idx, slot, bar_idx] = values
                synthetic[date_idx, slot, bar_idx] = False
                available[date_idx, slot, bar_idx] = True
                previous_close = float(row["close"])
            elif previous_close is not None:
                output[date_idx, slot, bar_idx, :4] = previous_close
                output[date_idx, slot, bar_idx, 4] = 0.0
                synthetic[date_idx, slot, bar_idx] = True
                available[date_idx, slot, bar_idx] = True


def prepare_bar_sidecar(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    store = resolve_feature_store()
    inputs = resolve_inputs()
    assignments = load_assignments(inputs.assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    equity_index = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    if tuple(equity_index.get_column("security_id")) != security_ids:
        raise ValueError("Assignment and feature-store security axes differ")

    research_start, research_end = read_research_interval(inputs.universe_dir)
    market_dates, assignment_dates = load_market_dates_and_security_dates(
        cotahist_files(inputs.cotahist_dir),
        security_ids,
        research_start,
        research_end,
    )
    validate_source_date_isolation(assignments, assignment_dates)
    canonical_dates = pl.read_parquet(store / "date_index.parquet").sort("date_idx")
    if tuple(canonical_dates.get_column("trade_date")) != market_dates:
        raise ValueError("Raw-source and feature-store market-date axes differ")
    train_dates = canonical_dates.filter(pl.col("trade_date") <= TRAIN_END)
    date_count = train_dates.height
    if not np.array_equal(
        train_dates.get_column("date_idx").to_numpy(), np.arange(date_count)
    ):
        raise ValueError("Training sidecar date axis must be a contiguous prefix")

    bars_path = output_dir / "bars.npy"
    synthetic_path = output_dir / "synthetic.npy"
    available_path = output_dir / "available.npy"
    timestamps_path = output_dir / "bar_close_timestamp_ns.npy"
    bars = open_memmap(
        bars_path,
        mode="w+",
        dtype=np.float32,
        shape=(date_count, EQUITY_COUNT, SESSION_BARS, len(BAR_FIELDS)),
    )
    synthetic = open_memmap(
        synthetic_path,
        mode="w+",
        dtype=bool,
        shape=(date_count, EQUITY_COUNT, SESSION_BARS),
    )
    available = open_memmap(
        available_path,
        mode="w+",
        dtype=bool,
        shape=(date_count, EQUITY_COUNT, SESSION_BARS),
    )
    timestamps = open_memmap(
        timestamps_path,
        mode="w+",
        dtype=np.int64,
        shape=(date_count, SESSION_BARS),
    )
    bars[...] = 0.0
    synthetic[...] = True
    available[...] = False
    for date_idx, trade_date in train_dates.select(
        "date_idx", "trade_date"
    ).iter_rows():
        midnight = np.datetime64(trade_date.isoformat(), "ns").astype(np.int64)
        close_minutes = EQUITY_SESSION_START_MINUTE + 5 * np.arange(1, SESSION_BARS + 1)
        timestamps[date_idx] = midnight + close_minutes * 60 * 1_000_000_000

    slot_by_security = {value: slot for slot, value in enumerate(security_ids)}
    source_groups = assignments.partition_by("source_file", maintain_order=True)
    for source_number, group in enumerate(source_groups, start=1):
        source_path = Path(group.item(0, "source_file"))
        source = load_source_file(source_path)
        validate_physical_source_identity(group, source, source_path)
        group_ids = tuple(group.get_column("security_id"))
        allowed_dates = frozenset().union(
            *(assignment_dates[security_id] for security_id in group_ids)
        )
        session = prepare_session_bars(
            source,
            source_path,
            allowed_dates,
            market_dates,
            EQUITY_SESSION_START_MINUTE,
            EQUITY_SESSION_MINUTES,
        ).filter(pl.col("date_idx") < date_count)
        for assignment in group.iter_rows(named=True):
            security_id = assignment["security_id"]
            security_dates = assignment_dates[security_id]
            observed = aggregate_five_minute_bars(
                session.filter(pl.col("trade_date").is_in(tuple(security_dates)))
            )
            valid_indices = [
                index
                for index, trade_date in enumerate(market_dates[:date_count])
                if trade_date in security_dates
            ]
            _fill_security_bars(
                bars,
                synthetic,
                available,
                slot=slot_by_security[security_id],
                valid_date_indices=valid_indices,
                observed=observed,
                sidecar_date_count=date_count,
            )
        if source_number % 20 == 0 or source_number == len(source_groups):
            print(f"Prepared K0 equity sources {source_number}/{len(source_groups)}")

    for array in (bars, synthetic, available, timestamps):
        array.flush()
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
    del bars, synthetic, available, timestamps

    train_dates.write_parquet(output_dir / "date_index.parquet")
    equity_index.write_parquet(output_dir / "equity_index.parquet")
    scope = _scope_rows(store)
    scope.write_parquet(output_dir / "scope_index.parquet")
    synthetic_array = np.load(synthetic_path, mmap_mode="r", allow_pickle=False)
    available_array = np.load(available_path, mmap_mode="r", allow_pickle=False)
    equity_coverage_rows = []
    for slot, security_id in equity_index.select(
        "equity_slot", "security_id"
    ).iter_rows():
        slot_available = np.asarray(available_array[:, slot])
        available_count = int(slot_available.sum())
        synthetic_count = int(
            (slot_available & np.asarray(synthetic_array[:, slot])).sum()
        )
        equity_coverage_rows.append(
            {
                "equity_slot": slot,
                "security_id": security_id,
                "available_bar_count": available_count,
                "observed_bar_count": available_count - synthetic_count,
                "synthetic_bar_count": synthetic_count,
                "synthetic_fraction": (
                    synthetic_count / available_count if available_count else None
                ),
            }
        )
    equity_coverage = pl.DataFrame(equity_coverage_rows)
    equity_coverage.write_parquet(output_dir / "equity_coverage.parquet")
    manifest = {
        "schema": "KRONOS_K0_BAR_SIDECAR",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _repository_commit(),
        "feature_store": feature_store_identity(store),
        "canonical_inputs": inputs.manifest_entries(),
        "date_range": {
            "start": str(train_dates.get_column("trade_date").min()),
            "end": str(train_dates.get_column("trade_date").max()),
            "date_count": date_count,
        },
        "equity_count": EQUITY_COUNT,
        "session": {
            "timezone": "America/Sao_Paulo",
            "start": "10:00",
            "minutes": EQUITY_SESSION_MINUTES,
            "bar_minutes": 5,
            "bar_count": SESSION_BARS,
            "bar_interval": "[t,t+5)",
            "timestamp_role": "naive local bar-close time",
        },
        "aggregation": {
            "ohlc": "first open, maximum high, minimum low, last close",
            "volume": "sum of observed MT5 real_volume",
            "amount": "omitted; upstream predictor synthesizes volume * mean(OHLC)",
            "missing_minutes": "aggregate observed minutes only",
            "empty_bar": "previous available close OHLC, zero volume, synthetic=true",
            "overnight_placeholders": False,
        },
        "bar_counts": {
            "available": int(equity_coverage["available_bar_count"].sum()),
            "observed": int(equity_coverage["observed_bar_count"].sum()),
            "synthetic": int(equity_coverage["synthetic_bar_count"].sum()),
        },
        "identity": (
            "permanent security_id with exact accepted COTAHIST assignment dates; "
            "one physical source may serve disjoint dated security segments"
        ),
        "scope": {
            "folds": {
                name: {"start": str(bounds[0]), "end": str(bounds[1]), "dates": 102}
                for name, bounds in FOLD_RANGES.items()
            },
            "decisions": list(DECISIONS),
            "scope_rows": scope.height,
            "official_validation_accessed": False,
            "test_accessed": False,
        },
        "arrays": {
            path.name: _array_metadata(path)
            for path in (bars_path, synthetic_path, available_path, timestamps_path)
        },
        "indices": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (
                output_dir / "date_index.parquet",
                output_dir / "equity_index.parquet",
                output_dir / "equity_coverage.parquet",
                output_dir / "scope_index.parquet",
            )
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return output_dir


@dataclass(frozen=True)
class Context:
    bars: np.ndarray
    timestamp_ns: np.ndarray
    synthetic: np.ndarray

    @property
    def momentum_60m(self) -> float:
        return float(self.bars[-1, 3] / self.bars[-13, 3] - 1.0)


class BarSidecar:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.bars = np.load(path / "bars.npy", mmap_mode="r", allow_pickle=False)
        self.synthetic = np.load(
            path / "synthetic.npy", mmap_mode="r", allow_pickle=False
        )
        self.available = np.load(
            path / "available.npy", mmap_mode="r", allow_pickle=False
        )
        self.timestamps = np.load(
            path / "bar_close_timestamp_ns.npy", mmap_mode="r", allow_pickle=False
        )
        self._positions: dict[int, np.ndarray] = {}

    def context(
        self, date_idx: int, decision_idx: int, equity_slot: int
    ) -> Context | None:
        bar_idx = 2 + decision_idx
        flat_end = date_idx * SESSION_BARS + bar_idx
        positions = self._positions.get(equity_slot)
        if positions is None:
            positions = np.flatnonzero(self.available[:, equity_slot].reshape(-1))
            self._positions[equity_slot] = positions
        insertion = int(np.searchsorted(positions, flat_end))
        if insertion >= positions.size or positions[insertion] != flat_end:
            return None
        start = insertion - CONTEXT_BARS + 1
        if start < 0:
            return None
        selected = positions[start : insertion + 1]
        date_positions, bar_positions = np.divmod(selected, SESSION_BARS)
        context = Context(
            bars=np.asarray(
                self.bars[date_positions, equity_slot, bar_positions], dtype=np.float32
            ),
            timestamp_ns=np.asarray(
                self.timestamps[date_positions, bar_positions], dtype=np.int64
            ),
            synthetic=np.asarray(
                self.synthetic[date_positions, equity_slot, bar_positions], dtype=bool
            ),
        )
        expected = self.timestamps[date_idx, bar_idx]
        if context.timestamp_ns[-1] != expected:
            raise ValueError("K0 context does not close at the decision minute")
        return context


def _prepare_reference(sidecar_dir: Path, output_path: Path) -> Path:
    store = resolve_feature_store()
    scope = pl.read_parquet(sidecar_dir / "scope_index.parquet").sort("scope_idx")
    sidecar = BarSidecar(sidecar_dir)
    membership = np.load(store / "equity_membership.npy", mmap_mode="r")
    data_ready = np.load(store / "equity_data_ready.npy", mmap_mode="r")
    targets_store = np.load(store / "targets.npy", mmap_mode="r")
    label_store = np.load(store / "label_mask.npy", mmap_mode="r")
    raw_returns_store = np.load(store / "raw_returns.npy", mmap_mode="r")
    shape = (scope.height, EQUITY_COUNT)
    coverage = np.zeros(shape, dtype=bool)
    momentum = np.zeros(shape, dtype=np.float32)
    synthetic_count = np.zeros(shape, dtype=np.uint16)
    recent_synthetic_count = np.zeros(shape, dtype=np.uint8)
    targets = np.zeros((*shape, len(HORIZONS)), dtype=np.float32)
    labels = np.zeros_like(targets, dtype=bool)
    raw_returns = np.zeros_like(targets)

    for row in scope.iter_rows(named=True):
        scope_idx = int(row["scope_idx"])
        date_idx = int(row["date_idx"])
        decision_idx = int(row["decision_idx"])
        active = membership[date_idx] & data_ready[date_idx]
        targets[scope_idx] = targets_store[date_idx, :, decision_idx]
        raw_returns[scope_idx] = raw_returns_store[date_idx, :, decision_idx]
        for slot in np.flatnonzero(active):
            context = sidecar.context(date_idx, decision_idx, int(slot))
            if context is None:
                continue
            observed = int((~context.synthetic).sum())
            recent_observed = int((~context.synthetic[-24:]).sum())
            synthetic_count[scope_idx, slot] = CONTEXT_BARS - observed
            recent_synthetic_count[scope_idx, slot] = 24 - recent_observed
            if (
                observed >= FULL_CONTEXT_OBSERVED
                and recent_observed >= RECENT_CONTEXT_OBSERVED
            ):
                coverage[scope_idx, slot] = True
                momentum[scope_idx, slot] = context.momentum_60m
        labels[scope_idx] = (
            label_store[date_idx, :, decision_idx] & coverage[scope_idx, :, None]
        )
        for horizon in range(len(HORIZONS)):
            if labels[scope_idx, :, horizon].sum() < MIN_IC_EQUITIES:
                labels[scope_idx, :, horizon] = False

    _atomic_npz(
        output_path,
        sample_id=scope.get_column("sample_id").to_numpy().astype(np.int64),
        date_idx=scope.get_column("date_idx").to_numpy().astype(np.int64),
        decision_idx=scope.get_column("decision_idx").to_numpy().astype(np.int64),
        fold=np.asarray(scope.get_column("fold").to_list(), dtype="U6"),
        targets=targets,
        label_mask=labels,
        raw_returns=raw_returns,
        coverage=coverage,
        momentum=momentum,
        synthetic_count=synthetic_count,
        recent_synthetic_count=recent_synthetic_count,
    )
    return output_path


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def _eligible_contexts(
    reference: Mapping[str, np.ndarray], decisions: Sequence[int]
) -> Iterator[tuple[int, int]]:
    allowed = np.isin(reference["decision_idx"], np.asarray(decisions))
    for scope_idx in np.flatnonzero(allowed):
        for slot in np.flatnonzero(reference["coverage"][scope_idx]):
            yield int(scope_idx), int(slot)


def _snapshot_files(path: Path) -> list[dict[str, object]]:
    rows = []
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": str(file.relative_to(path)),
                "sha256": _sha256(file),
                "bytes": file.stat().st_size,
            }
        )
    return rows


def _load_kronos(
    model_name: str, kronos_repo: Path
) -> tuple[object, dict[str, object]]:
    actual_commit = subprocess.run(
        ["git", "-C", str(kronos_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected Kronos {UPSTREAM_COMMIT}, got {actual_commit}")
    forbidden = ("finetune", "finetune_csv", "examples", "webui")
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError("A prohibited upstream package was imported")
    sys.path.insert(0, str(kronos_repo))
    try:
        upstream = importlib.import_module("model")
        hub = importlib.import_module("huggingface_hub")
        tokenizer_path = Path(
            hub.snapshot_download(
                TOKENIZER_REPOSITORY,
                revision=TOKENIZER_REVISION,
            )
        )
        model_repo, model_revision = MODEL_REVISIONS[model_name]
        model_path = Path(hub.snapshot_download(model_repo, revision=model_revision))
        tokenizer = upstream.KronosTokenizer.from_pretrained(tokenizer_path)
        model = upstream.Kronos.from_pretrained(model_path)
        tokenizer.eval()
        model.eval()
        predictor = upstream.KronosPredictor(
            model, tokenizer, device="cuda:0", max_context=CONTEXT_BARS
        )
    finally:
        sys.path.pop(0)
    metadata = {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": actual_commit,
        "upstream_model_internals_modified": False,
        "upstream_fine_token_rope_cross_attention": "accepted_as_shipped",
        "model": {
            "repository": model_repo,
            "revision": model_revision,
            "snapshot": str(model_path),
            "files": _snapshot_files(model_path),
        },
        "tokenizer": {
            "repository": TOKENIZER_REPOSITORY,
            "revision": TOKENIZER_REVISION,
            "snapshot": str(tokenizer_path),
            "files": _snapshot_files(tokenizer_path),
        },
    }
    return predictor, metadata


def _predict_context(
    predictor: object,
    context: Context,
    *,
    seed: int,
    use_bf16: bool,
) -> np.ndarray:
    import pandas as pd

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    frame = pd.DataFrame(context.bars, columns=BAR_FIELDS)
    x_timestamp = pd.Series(pd.to_datetime(context.timestamp_ns, unit="ns"))
    y_timestamp = pd.Series(
        pd.date_range(
            x_timestamp.iloc[-1] + pd.Timedelta(minutes=5),
            periods=PREDICTION_BARS,
            freq="5min",
        )
    )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        prediction = predictor.predict_batch(
            [frame],
            [x_timestamp],
            [y_timestamp],
            pred_len=PREDICTION_BARS,
            T=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=False,
        )[0]
    closes = prediction["close"].to_numpy(dtype=np.float64)
    scores = np.asarray(
        [closes[horizon // 5 - 1] / context.bars[-1, 3] - 1.0 for horizon in HORIZONS],
        dtype=np.float32,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Kronos produced a non-finite score")
    return scores


def _context_identity(
    scope: pl.DataFrame, equities: pl.DataFrame, scope_idx: int, slot: int
) -> tuple[date, int, str]:
    row = scope.row(scope_idx, named=True)
    return (
        row["trade_date"],
        int(row["decision_idx"]),
        equities.item(slot, "security_id"),
    )


def _audit_precision(
    predictor: object,
    model_name: str,
    sidecar: BarSidecar,
    sidecar_dir: Path,
    reference: Mapping[str, np.ndarray],
) -> dict[str, object]:
    scope = pl.read_parquet(sidecar_dir / "scope_index.parquet").sort("scope_idx")
    equities = pl.read_parquet(sidecar_dir / "equity_index.parquet").sort("equity_slot")
    contexts = list(_eligible_contexts(reference, DECISIONS))[:100]
    if len(contexts) != 100:
        raise ValueError("Fewer than 100 eligible contexts for precision audit")
    fp32: dict[tuple[int, int], np.ndarray] = {}
    bf16: dict[tuple[int, int], np.ndarray] = {}
    bf16_error: str | None = None
    for key in contexts:
        scope_idx, slot = key
        trade_date, decision, security_id = _context_identity(
            scope, equities, scope_idx, slot
        )
        context = sidecar.context(int(reference["date_idx"][scope_idx]), decision, slot)
        if context is None:
            raise RuntimeError("Eligible precision context disappeared")
        seed = stable_context_seed(model_name, trade_date, decision, security_id)
        fp32[key] = _predict_context(predictor, context, seed=seed, use_bf16=False)
        if bf16_error is None:
            try:
                bf16[key] = _predict_context(
                    predictor, context, seed=seed, use_bf16=True
                )
            except (RuntimeError, TypeError, ValueError) as error:
                bf16_error = f"{type(error).__name__}: {error}"
    group_deltas = []
    if bf16_error is None:
        for scope_idx in sorted({key[0] for key in contexts}):
            slots = np.asarray(
                [slot for candidate, slot in contexts if candidate == scope_idx],
                dtype=np.int64,
            )
            for horizon in range(len(HORIZONS)):
                valid = reference["label_mask"][scope_idx, slots, horizon]
                if valid.sum() < MIN_IC_EQUITIES:
                    continue
                actual = average_ranks(
                    reference["targets"][scope_idx, slots[valid], horizon]
                )
                fp_rank = average_ranks(
                    np.asarray(
                        [fp32[(scope_idx, slot)][horizon] for slot in slots[valid]]
                    )
                )
                bf_rank = average_ranks(
                    np.asarray(
                        [bf16[(scope_idx, slot)][horizon] for slot in slots[valid]]
                    )
                )
                fp_ic = float(np.corrcoef(fp_rank, actual)[0, 1])
                bf_ic = float(np.corrcoef(bf_rank, actual)[0, 1])
                group_deltas.append(
                    {
                        "scope_idx": scope_idx,
                        "horizon_minutes": HORIZONS[horizon],
                        "eligible_equities": int(valid.sum()),
                        "fp32_ic": fp_ic,
                        "bf16_ic": bf_ic,
                        "absolute_ic_change": abs(bf_ic - fp_ic),
                    }
                )
    maximum = max(
        (float(row["absolute_ic_change"]) for row in group_deltas), default=math.inf
    )
    use_bf16 = bf16_error is None and bool(group_deltas) and maximum < 0.001
    return {
        "context_count": len(contexts),
        "selection": "first eligible contexts in chronological date/decision/security order",
        "per_context_seed_reset": True,
        "bf16_error": bf16_error,
        "group_deltas": group_deltas,
        "maximum_absolute_group_ic_change": maximum if math.isfinite(maximum) else None,
        "threshold": 0.001,
        "adopted_precision": "bf16" if use_bf16 else "fp32",
    }


def _audit_throughput(
    predictor: object,
    model_name: str,
    sidecar: BarSidecar,
    sidecar_dir: Path,
    reference: Mapping[str, np.ndarray],
    *,
    use_bf16: bool,
) -> dict[str, object]:
    scope = pl.read_parquet(sidecar_dir / "scope_index.parquet").sort("scope_idx")
    equities = pl.read_parquet(sidecar_dir / "equity_index.parquet").sort("equity_slot")
    contexts = list(_eligible_contexts(reference, DECISIONS))[:200]
    torch.cuda.synchronize()
    started = time.perf_counter()
    for scope_idx, slot in contexts:
        trade_date, decision, security_id = _context_identity(
            scope, equities, scope_idx, slot
        )
        context = sidecar.context(int(reference["date_idx"][scope_idx]), decision, slot)
        if context is None:
            raise RuntimeError("Eligible throughput context disappeared")
        _predict_context(
            predictor,
            context,
            seed=stable_context_seed(model_name, trade_date, decision, security_id),
            use_bf16=use_bf16,
        )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if len(contexts) != 200:
        raise ValueError("Fewer than 200 eligible contexts for throughput audit")
    return {
        "context_count": len(contexts),
        "seconds": seconds,
        "seconds_per_context": seconds / len(contexts),
        "precision": "bf16" if use_bf16 else "fp32",
    }


def _infer_model(
    predictor: object,
    model_name: str,
    sidecar: BarSidecar,
    sidecar_dir: Path,
    reference: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    decisions: Sequence[int],
    use_bf16: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_partial = output_dir / "scores.partial.npy"
    done_partial = output_dir / "done.partial.npy"
    final_score = output_dir / "scores.npy"
    final_done = output_dir / "done.npy"
    shape = reference["targets"].shape
    if final_score.exists() and final_done.exists():
        return {
            "resumed_completed": True,
            "scores": _array_metadata(final_score),
            "done": _array_metadata(final_done),
        }
    if score_partial.exists() != done_partial.exists():
        raise RuntimeError("Partial Kronos inference files are inconsistent")
    if not score_partial.exists():
        scores = open_memmap(score_partial, mode="w+", dtype=np.float32, shape=shape)
        done = open_memmap(
            done_partial,
            mode="w+",
            dtype=bool,
            shape=reference["coverage"].shape,
        )
        scores[...] = 0.0
        done[...] = False
    else:
        scores = open_memmap(score_partial, mode="r+")
        done = open_memmap(done_partial, mode="r+")
        if scores.shape != shape or done.shape != reference["coverage"].shape:
            raise ValueError("Partial inference arrays have the wrong shape")

    scope = pl.read_parquet(sidecar_dir / "scope_index.parquet").sort("scope_idx")
    equities = pl.read_parquet(sidecar_dir / "equity_index.parquet").sort("equity_slot")
    contexts = list(_eligible_contexts(reference, decisions))
    started = time.perf_counter()
    completed_before = int(done.sum())
    for number, (scope_idx, slot) in enumerate(contexts, start=1):
        if done[scope_idx, slot]:
            continue
        trade_date, decision, security_id = _context_identity(
            scope, equities, scope_idx, slot
        )
        context = sidecar.context(int(reference["date_idx"][scope_idx]), decision, slot)
        if context is None:
            raise RuntimeError("Eligible inference context disappeared")
        seed = stable_context_seed(model_name, trade_date, decision, security_id)
        scores[scope_idx, slot] = _predict_context(
            predictor, context, seed=seed, use_bf16=use_bf16
        )
        done[scope_idx, slot] = True
        if number % 250 == 0:
            scores.flush()
            done.flush()
            elapsed = time.perf_counter() - started
            print(
                f"{model_name}: {number}/{len(contexts)} contexts visited; "
                f"elapsed={elapsed:.1f}s"
            )
    scores.flush()
    done.flush()
    elapsed = time.perf_counter() - started
    mapping = getattr(scores, "_mmap", None)
    if mapping is not None:
        mapping.close()
    mapping = getattr(done, "_mmap", None)
    if mapping is not None:
        mapping.close()
    del scores, done
    os.replace(score_partial, final_score)
    os.replace(done_partial, final_done)

    final_scores = np.load(final_score, mmap_mode="r", allow_pickle=False)
    final_flags = np.load(final_done, mmap_mode="r", allow_pickle=False)
    expected = (
        reference["coverage"]
        & np.isin(reference["decision_idx"], np.asarray(decisions))[:, None]
    )
    if not np.array_equal(final_flags, expected):
        raise ValueError("Completed inference mask differs from the fixed scope")
    if np.any(final_scores[~expected] != 0.0):
        raise ValueError("Out-of-scope Kronos scores must be exactly zero")
    if not np.isfinite(final_scores[expected]).all():
        raise ValueError("Kronos scores contain non-finite values")
    return {
        "resumed_completed": False,
        "elapsed_seconds": elapsed,
        "completed_before_resume": completed_before,
        "context_count": int(expected.sum()),
        "scores": _array_metadata(final_score),
        "done": _array_metadata(final_done),
    }


def _parent_members(
    reference: Mapping[str, np.ndarray], decisions: Sequence[int]
) -> dict[str, dict[str, np.ndarray]]:
    analysis = json.loads(
        (PARENT_CROSSFIT / "crossfit_analysis.json").read_text(encoding="utf-8")
    )
    result: dict[str, dict[str, np.ndarray]] = {}
    for fold in ("fold_a", "fold_b"):
        fold_scope = reference["fold"] == fold
        fold_sample_ids = reference["sample_id"][fold_scope]
        fold_decisions = reference["decision_idx"][fold_scope]
        keep = np.isin(fold_decisions, np.asarray(decisions))
        selected_ids = fold_sample_ids[keep]
        fold_members: dict[str, np.ndarray] = {}
        directions = analysis["folds"][fold]["rule_selection_crossfit"]["directions"]
        for seed in (11, 29, 47):
            run_dir = PARENT_CAMPAIGN / fold / f"seed_{seed}"
            with np.load(
                run_dir / "validation_reference.npz", allow_pickle=False
            ) as values:
                order = np.argsort(values["sample_id"], kind="stable")
                sample_ids = values["sample_id"][order]
                date_idx = values["date_idx"][order]
            combined = np.empty(
                (sample_ids.size, EQUITY_COUNT, len(HORIZONS)), np.float32
            )
            assigned = np.zeros(sample_ids.size, dtype=bool)
            for direction in directions:
                replay = direction["rules"]["patience3_raw"]["member_patience_replay"][
                    f"seed_{seed}"
                ]
                epoch = int(replay["selected_epoch"])
                with np.load(
                    run_dir / "validation_predictions" / f"epoch_{epoch:02d}.npz",
                    allow_pickle=False,
                ) as values:
                    predictions = values["raw"][order]
                on_dates = np.isin(
                    date_idx,
                    np.asarray(direction["evaluation_date_idx"], dtype=np.int64),
                )
                combined[on_dates] = predictions[on_dates]
                assigned[on_dates] = True
            if not assigned.all():
                raise ValueError("Parent cross-fit did not assign every sample")
            positions = np.searchsorted(sample_ids, selected_ids)
            if not np.array_equal(sample_ids[positions], selected_ids):
                raise ValueError("K0 scope does not align with parent observations")
            fold_members[f"seed_{seed}"] = combined[positions]
        result[fold] = fold_members
    return result


def _model_scope(
    reference: Mapping[str, np.ndarray], decisions: Sequence[int]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sample_mask = np.isin(reference["decision_idx"], np.asarray(decisions))
    return sample_mask, {name: value[sample_mask] for name, value in reference.items()}


def _fold_scores(
    predictions: np.ndarray, scoped: Mapping[str, np.ndarray]
) -> dict[str, float]:
    return {
        fold: primary_validation_score(
            predictions[scoped["fold"] == fold],
            scoped["targets"][scoped["fold"] == fold],
            scoped["label_mask"][scoped["fold"] == fold],
            scoped["date_idx"][scoped["fold"] == fold],
        )
        for fold in ("fold_a", "fold_b")
    }


def _bootstrap_daily(delta: np.ndarray) -> dict[str, object]:
    return {
        str(block): {
            name: np.asarray(value).tolist()
            for name, value in moving_block_bootstrap(
                delta,
                replications=BOOTSTRAP_REPLICATIONS,
                block_length=block,
                seed=20260822 + block,
            ).items()
        }
        for block in (5, 10)
    }


def _analyze_model(
    model_name: str,
    model_dir: Path,
    reference: Mapping[str, np.ndarray],
    decisions: Sequence[int],
    parent: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, object]:
    sample_mask, scoped = _model_scope(reference, decisions)
    scores = np.asarray(np.load(model_dir / "scores.npy", mmap_mode="r"))[sample_mask]
    scores[~scoped["label_mask"]] = 0.0
    momentum = np.repeat(scoped["momentum"][..., None], len(HORIZONS), axis=2)
    momentum[~scoped["label_mask"]] = 0.0
    parent_prediction = np.zeros_like(scores)
    stack_prediction = np.zeros_like(scores)
    member_predictions: dict[str, np.ndarray] = {}
    for fold in ("fold_a", "fold_b"):
        fold_mask = scoped["fold"] == fold
        members = parent[fold]
        for name, values in members.items():
            member_predictions.setdefault(name, np.zeros_like(scores))[fold_mask] = (
                values
            )
        parent_prediction[fold_mask] = rank_average_predictions(
            list(members.values()), scoped["label_mask"][fold_mask]
        )
        stack_prediction[fold_mask] = rank_average_predictions(
            [*members.values(), scores[fold_mask]], scoped["label_mask"][fold_mask]
        )

    primary = _fold_scores(scores, scoped)
    momentum_scores = _fold_scores(momentum, scoped)
    parent_scores = _fold_scores(parent_prediction, scoped)
    stack_scores = _fold_scores(stack_prediction, scoped)
    correlation = {
        fold: primary_validation_score(
            scores[scoped["fold"] == fold],
            parent_prediction[scoped["fold"] == fold],
            scoped["label_mask"][scoped["fold"] == fold],
            scoped["date_idx"][scoped["fold"] == fold],
        )
        for fold in ("fold_a", "fold_b")
    }
    breakdown = {"horizon": [], "time_of_day": []}
    kronos_ic = sample_level_spearman_ic(
        scores, scoped["targets"], scoped["label_mask"]
    )
    parent_ic = sample_level_spearman_ic(
        parent_prediction, scoped["targets"], scoped["label_mask"]
    )
    stack_ic = sample_level_spearman_ic(
        stack_prediction, scoped["targets"], scoped["label_mask"]
    )
    for fold in ("fold_a", "fold_b"):
        on_fold = scoped["fold"] == fold
        _, kronos_daily_h = daily_horizon_ic(
            kronos_ic[on_fold], scoped["date_idx"][on_fold]
        )
        _, parent_daily_h = daily_horizon_ic(
            parent_ic[on_fold], scoped["date_idx"][on_fold]
        )
        _, stack_daily_h = daily_horizon_ic(
            stack_ic[on_fold], scoped["date_idx"][on_fold]
        )
        for index, horizon in enumerate(HORIZONS):
            breakdown["horizon"].append(
                {
                    "fold": fold,
                    "horizon_minutes": horizon,
                    "kronos_ic": finite_mean(kronos_daily_h[:, index]),
                    "parent_ic": finite_mean(parent_daily_h[:, index]),
                    "parent_plus_kronos_ic": finite_mean(stack_daily_h[:, index]),
                }
            )
        for decision in decisions:
            on_decision = on_fold & (scoped["decision_idx"] == decision)
            breakdown["time_of_day"].append(
                {
                    "fold": fold,
                    "decision_idx": decision,
                    "kronos_ic": finite_mean(kronos_ic[on_decision].ravel()),
                    "parent_ic": finite_mean(parent_ic[on_decision].ravel()),
                    "parent_plus_kronos_ic": finite_mean(stack_ic[on_decision].ravel()),
                }
            )

    ensemble_bootstrap = {}
    for fold in ("fold_a", "fold_b"):
        on_fold = scoped["fold"] == fold
        dates, stack_daily = per_date_primary_ic(
            stack_ic[on_fold], scoped["date_idx"][on_fold]
        )
        parent_dates, parent_daily = per_date_primary_ic(
            parent_ic[on_fold], scoped["date_idx"][on_fold]
        )
        if not np.array_equal(dates, parent_dates):
            raise ValueError("Ensemble bootstrap date axes differ")
        ensemble_bootstrap[fold] = _bootstrap_daily(stack_daily - parent_daily)
    return {
        "model": model_name,
        "decisions": list(decisions),
        "primary_ic": {**primary, "mean_folds": float(np.mean(list(primary.values())))},
        "momentum_control_ic": {
            **momentum_scores,
            "mean_folds": float(np.mean(list(momentum_scores.values()))),
        },
        "parent_ic": {
            **parent_scores,
            "mean_folds": float(np.mean(list(parent_scores.values()))),
        },
        "score_parent_spearman": {
            **correlation,
            "mean_folds": float(np.mean(list(correlation.values()))),
        },
        "parent_plus_kronos_ic": {
            **stack_scores,
            "mean_folds": float(np.mean(list(stack_scores.values()))),
            "delta_vs_parent_mean_folds": float(
                np.mean(list(stack_scores.values()))
                - np.mean(list(parent_scores.values()))
            ),
            "paired_daily_block_bootstrap": ensemble_bootstrap,
            "recipe": "uniform tie-aware rank average of parent seeds 11/29/47 and Kronos",
            "informational_only": True,
        },
        "breakdowns": breakdown,
    }


def _decision(results: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    best_name = max(
        results,
        key=lambda name: float(results[name]["primary_ic"]["mean_folds"]),
    )
    best = results[best_name]
    primary = float(best["primary_ic"]["mean_folds"])
    momentum = float(best["momentum_control_ic"]["mean_folds"])
    correlation = float(best["score_parent_spearman"]["mean_folds"])
    if primary < 0.015 or primary <= momentum:
        outcome = "kill"
        reason = (
            "best mean-fold IC is below 0.015"
            if primary < 0.015
            else "best mean-fold IC does not exceed matched momentum control"
        )
    elif correlation >= 0.5:
        outcome = "park"
        reason = (
            "best model passes the IC floor but is at least 0.5 correlated with parent"
        )
    else:
        outcome = "eligible_for_separately_preregistered_k1"
        reason = "best model passes the IC and orthogonality kill thresholds"
    return {
        "outcome": outcome,
        "reason": reason,
        "best_model": best_name,
        "IC_best": primary,
        "matched_momentum_control_ic": momentum,
        "mean_score_parent_correlation": correlation,
        "k1_started": False,
        "passing_k0_is_confirmatory_evidence": False,
    }


def analyze_run(run_dir: Path) -> Path:
    reference = _load_reference(run_dir / "scope_reference.npz")
    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    results = {}
    for model_name, metadata in run_manifest["models"].items():
        decisions = tuple(metadata["decisions"])
        parent = _parent_members(reference, decisions)
        results[model_name] = _analyze_model(
            model_name,
            run_dir / model_name.casefold(),
            reference,
            decisions,
            parent,
        )
    decision = _decision(results)
    report = {
        "schema": "KRONOS_K0_RESULT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": _repository_commit(),
        "models": results,
        "decision": decision,
        "official_validation_accessed": False,
        "test_accessed": False,
        "k1_started": False,
    }
    _atomic_json(run_dir / "results.json", report)
    pl.DataFrame(
        [
            {"model": model_name, **row}
            for model_name, result in results.items()
            for row in result["breakdowns"]["horizon"]
        ]
    ).write_parquet(run_dir / "model_fold_horizon_metrics.parquet")
    pl.DataFrame(
        [
            {"model": model_name, **row}
            for model_name, result in results.items()
            for row in result["breakdowns"]["time_of_day"]
        ]
    ).write_parquet(run_dir / "model_fold_time_of_day_metrics.parquet")
    pl.DataFrame(
        [
            {
                "model": model_name,
                "fold": fold,
                "primary_ic": result["primary_ic"][fold],
                "momentum_control_ic": result["momentum_control_ic"][fold],
                "parent_ic": result["parent_ic"][fold],
                "score_parent_spearman": result["score_parent_spearman"][fold],
                "parent_plus_kronos_ic": result["parent_plus_kronos_ic"][fold],
            }
            for model_name, result in results.items()
            for fold in ("fold_a", "fold_b")
        ]
    ).write_parquet(run_dir / "model_fold_summary.parquet")
    rows = [
        "# Kronos K0 result",
        "",
        f"Decision: **{decision['outcome']}** — {decision['reason']}.",
        "",
        "| Model | Decisions | Fold A IC | Fold B IC | Mean IC | Momentum | Parent corr. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, result in results.items():
        rows.append(
            "| {name} | {decisions} | {a:.6f} | {b:.6f} | {mean:.6f} | "
            "{momentum:.6f} | {correlation:.6f} |".format(
                name=model_name,
                decisions=",".join(map(str, result["decisions"])),
                a=result["primary_ic"]["fold_a"],
                b=result["primary_ic"]["fold_b"],
                mean=result["primary_ic"]["mean_folds"],
                momentum=result["momentum_control_ic"]["mean_folds"],
                correlation=result["score_parent_spearman"]["mean_folds"],
            )
        )
    rows.extend(
        [
            "",
            "Passing K0 would only authorize a separately preregistered K1. K1 was not run.",
            "Official validation accessed: false. Held-out test accessed: false.",
            "",
        ]
    )
    (run_dir / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")
    return run_dir / "results.json"


def _verify_determinism(
    predictor: object,
    model_name: str,
    sidecar: BarSidecar,
    sidecar_dir: Path,
    reference: Mapping[str, np.ndarray],
    *,
    use_bf16: bool,
) -> list[dict[str, object]]:
    scope = pl.read_parquet(sidecar_dir / "scope_index.parquet").sort("scope_idx")
    equities = pl.read_parquet(sidecar_dir / "equity_index.parquet").sort("equity_slot")
    eligible = list(_eligible_contexts(reference, DECISIONS))
    selected = [eligible[0], eligible[len(eligible) // 2], eligible[-1]]
    rows = []
    for scope_idx, slot in selected:
        trade_date, decision, security_id = _context_identity(
            scope, equities, scope_idx, slot
        )
        context = sidecar.context(int(reference["date_idx"][scope_idx]), decision, slot)
        if context is None:
            raise RuntimeError("Eligible deterministic context disappeared")
        seed = stable_context_seed(model_name, trade_date, decision, security_id)
        left = _predict_context(predictor, context, seed=seed, use_bf16=use_bf16)
        right = _predict_context(predictor, context, seed=seed, use_bf16=use_bf16)
        if not np.array_equal(left, right):
            raise ValueError("Kronos context rerun was not bitwise deterministic")
        rows.append(
            {
                "trade_date": str(trade_date),
                "decision_idx": decision,
                "security_id": security_id,
                "seed": seed,
                "score_sha256": hashlib.sha256(left.tobytes()).hexdigest(),
            }
        )
    return rows


def run_k0(run_dir: Path, sidecar_dir: Path, kronos_repo: Path) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists():
        reference_path = run_dir / "scope_reference.npz"
        if not manifest_path.is_file() or not reference_path.is_file():
            raise FileExistsError(run_dir)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "KRONOS_K0_RUN"
            or manifest.get("status") == "completed"
        ):
            raise ValueError("Run directory is not a resumable K0 run")
        current_commit = _repository_commit()
        previous_commit = str(manifest.get("repository_commit"))
        if previous_commit != current_commit:
            manifest.setdefault("pre_score_runtime_corrections", []).append(
                {
                    "from_commit": previous_commit,
                    "to_commit": current_commit,
                    "reason": "timestamp wrapper now passes pandas Series to shipped .dt accessor",
                    "score_existed_before_correction": False,
                }
            )
            manifest["repository_commit"] = current_commit
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
    else:
        run_dir.mkdir(parents=True)
        reference_path = _prepare_reference(
            sidecar_dir, run_dir / "scope_reference.npz"
        )
        manifest = {}
    reference = _load_reference(reference_path)
    sidecar = BarSidecar(sidecar_dir)
    if not manifest:
        coverage = {
            fold: {
                "eligible_contexts": int(
                    reference["coverage"][reference["fold"] == fold].sum()
                ),
                "eligible_labeled_groups": int(
                    np.any(
                        reference["label_mask"][reference["fold"] == fold], axis=1
                    ).sum()
                ),
                "mean_synthetic_fraction": float(
                    reference["synthetic_count"][
                        reference["coverage"] & (reference["fold"] == fold)[:, None]
                    ].mean()
                    / CONTEXT_BARS
                ),
            }
            for fold in ("fold_a", "fold_b")
        }
        manifest = {
            "schema": "KRONOS_K0_RUN",
            "status": "running",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository_commit": _repository_commit(),
            "instance": {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "bar_sidecar": {
                "path": str(sidecar_dir.resolve()),
                "manifest_sha256": _sha256(sidecar_dir / "manifest.json"),
            },
            "scope_reference": {
                "path": str(reference_path.resolve()),
                "sha256": _sha256(reference_path),
            },
            "coverage": coverage,
            "settings": {
                "context_bars": CONTEXT_BARS,
                "prediction_bars": PREDICTION_BARS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "sample_count": SAMPLE_COUNT,
                "global_seed": GLOBAL_SEED,
                "seed_formula": (
                    "unsigned low 63 bits of first eight SHA-256 bytes over "
                    "model_name\\0date\\0decision_idx\\0security_id"
                ),
                "per_context_predict_batch_size": 1,
                "samples_batched_inside_upstream": SAMPLE_COUNT,
            },
            "models": {},
            "official_validation_accessed": False,
            "test_accessed": False,
            "k1_started": False,
            "resume_count": 0,
        }
    _atomic_json(manifest_path, manifest)

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    for model_name in ("Kronos-small", "Kronos-base"):
        model_output = run_dir / model_name.casefold()
        if (
            model_name in manifest["models"]
            and (model_output / "scores.npy").is_file()
            and (model_output / "done.npy").is_file()
        ):
            continue
        predictor, pinned = _load_kronos(model_name, kronos_repo)
        precision = _audit_precision(
            predictor, model_name, sidecar, sidecar_dir, reference
        )
        use_bf16 = precision["adopted_precision"] == "bf16"
        throughput = _audit_throughput(
            predictor,
            model_name,
            sidecar,
            sidecar_dir,
            reference,
            use_bf16=use_bf16,
        )
        full_count = sum(1 for _ in _eligible_contexts(reference, DECISIONS))
        projected_full_hours = (
            float(throughput["seconds_per_context"]) * full_count / 3600.0
        )
        decisions = (
            DECISIONS
            if model_name == "Kronos-small" or projected_full_hours <= 24.0
            else REDUCED_BASE_DECISIONS
        )
        deterministic = _verify_determinism(
            predictor,
            model_name,
            sidecar,
            sidecar_dir,
            reference,
            use_bf16=use_bf16,
        )
        inference = _infer_model(
            predictor,
            model_name,
            sidecar,
            sidecar_dir,
            reference,
            model_output,
            decisions=decisions,
            use_bf16=use_bf16,
        )
        manifest["models"][model_name] = {
            **pinned,
            "precision_audit": precision,
            "throughput": {
                **throughput,
                "full_scope_context_count": full_count,
                "projected_full_scope_gpu_hours": projected_full_hours,
                "base_full_scope_threshold_hours": 24.0,
            },
            "decisions": list(decisions),
            "determinism_audit": deterministic,
            "inference": inference,
        }
        _atomic_json(manifest_path, manifest)
        del predictor
        torch.cuda.empty_cache()

    manifest["status"] = "inference_completed"
    manifest["inference_completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(manifest_path, manifest)
    analyze_run(run_dir)
    manifest["status"] = "completed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["result_artifacts"] = {
        name: {
            "sha256": _sha256(run_dir / name),
            "bytes": (run_dir / name).stat().st_size,
        }
        for name in (
            "results.json",
            "RESULTS.md",
            "model_fold_summary.parquet",
            "model_fold_horizon_metrics.parquet",
            "model_fold_time_of_day_metrics.parquet",
        )
    }
    manifest["official_validation_accessed"] = False
    manifest["test_accessed"] = False
    manifest["k1_started"] = False
    _atomic_json(manifest_path, manifest)
    return run_dir


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered Kronos K0 kill-test"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--sidecar-dir", type=Path, required=True)
    run.add_argument("--kronos-repo", type=Path, required=True)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(arguments)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        print(prepare_bar_sidecar(args.output_dir))
    elif args.command == "run":
        print(run_k0(args.run_dir, args.sidecar_dir, args.kronos_repo))
    else:
        print(analyze_run(args.run_dir))


if __name__ == "__main__":
    main()
