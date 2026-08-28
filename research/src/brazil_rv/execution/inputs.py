from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ..modeling.contract import TRAIN_END, TRAIN_START, workspace_path
from ..modeling.data import (
    feature_store_axis_identity,
    feature_store_identity,
    int64_identity_sha256,
)
from ..modeling.metrics import average_ranks
from ..preprocessing.contract import EQUITY_SESSION_MINUTES
from ..preprocessing.io import (
    cotahist_files,
    dense_grid,
    load_assignments,
    load_market_dates_and_security_dates,
    load_source_file,
    prepare_session_bars,
    read_research_interval,
    validate_physical_source_identity,
    validate_source_date_isolation,
)

EXECUTION_PREDICTION_ARCHIVE_SCHEMA = "B3_EXECUTION_PREDICTION_ARCHIVE_V1"
OOF_PREDICTION_ARCHIVE_SCHEMA = "B3_EXECUTION_OOF_PREDICTION_ARCHIVE_V1"
_DISCOVERY_SPLITS = frozenset({"fold_a", "fold_b", "fold_c"})


@dataclass(frozen=True)
class DiscoveryPredictionArchive:
    ranks: np.ndarray
    valid: np.ndarray
    sample_id: np.ndarray
    date_idx: np.ndarray
    decision_idx: np.ndarray
    refresh_minutes: np.ndarray


@dataclass(frozen=True)
class DiscoveryEquityGrid:
    """One permanent-security M1 grid, bounded to discovery dates."""

    equity_slot: int
    security_id: str
    trade_dates: tuple[date, ...]
    open_price: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    real_volume: np.ndarray
    observed: np.ndarray
    active: np.ndarray


def _recorded_directory(manifest: Mapping[str, object], name: str) -> Path:
    inputs = manifest.get("canonical_inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get(name), dict):
        raise ValueError(f"Feature store does not record canonical input {name}")
    value = inputs[name].get("resolved_path")
    if not isinstance(value, str):
        raise ValueError(f"Canonical input {name} has no resolved path")
    return workspace_path(value)


def iter_discovery_equity_grids(store: Path) -> Iterator[DiscoveryEquityGrid]:
    """Stream identity-safe raw M1 grids through the end of training.

    A physical XP file may contain several historical securities. The function
    loads it once, then emits only the accepted date segment for each permanent
    ``security_id``. It never opens official-validation or test dates.
    """
    store = store.resolve()
    feature_store_identity(store)
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assignments_dir = _recorded_directory(manifest, "accepted_xp_assignments")
    cotahist_dir = _recorded_directory(manifest, "parsed_cotahist")
    universe_dir = _recorded_directory(manifest, "point_in_time_universe")
    research_start, research_end = read_research_interval(universe_dir)
    through = min(research_end, TRAIN_END)

    assignments = load_assignments(assignments_dir)
    security_ids = tuple(assignments["security_id"])
    market_dates, assignment_dates = load_market_dates_and_security_dates(
        cotahist_files(cotahist_dir),
        security_ids,
        research_start,
        through,
        allow_empty_security_dates=True,
    )
    validate_source_date_isolation(assignments, assignment_dates)
    dates = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= through)
        .sort("date_idx")
    )
    equities = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    if (
        tuple(dates["trade_date"]) != market_dates
        or tuple(equities["security_id"]) != security_ids
    ):
        raise ValueError("Raw M1 and canonical feature-store axes differ")
    active = np.asarray(
        np.load(store / "equity_membership.npy", mmap_mode="r", allow_pickle=False)[
            : len(market_dates)
        ]
        & np.load(store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False)[
            : len(market_dates)
        ],
        dtype=bool,
    )
    slot_by_security = {value: slot for slot, value in enumerate(security_ids)}
    for group in assignments.partition_by("source_file", maintain_order=True):
        source_path = Path(group.item(0, "source_file"))
        source = load_source_file(source_path)
        validate_physical_source_identity(group, source, source_path)
        allowed_dates = frozenset().union(
            *(assignment_dates[value] for value in group["security_id"])
        )
        bars = prepare_session_bars(
            source,
            source_path,
            allowed_dates,
            market_dates,
            10 * 60,
            EQUITY_SESSION_MINUTES,
        )
        for security_id in group["security_id"]:
            security_bars = bars.filter(
                pl.col("trade_date").is_in(tuple(assignment_dates[security_id]))
            )
            raw, observed = dense_grid(
                security_bars, len(market_dates), EQUITY_SESSION_MINUTES
            )
            slot = slot_by_security[security_id]
            yield DiscoveryEquityGrid(
                equity_slot=slot,
                security_id=security_id,
                trade_dates=market_dates,
                open_price=raw[..., 0],
                high=raw[..., 1],
                low=raw[..., 2],
                close=raw[..., 3],
                real_volume=raw[..., 4],
                observed=observed,
                active=active[:, slot],
            )


def _rank_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if values.shape == shape[:-1]:
        values = np.broadcast_to(values[..., None], shape)
    if values.shape != shape:
        raise ValueError("Causal mask must match scores with an optional horizon axis")
    return values


def causal_rank_scores(raw_scores: np.ndarray, causal_mask: np.ndarray) -> np.ndarray:
    """Tie-aware cross-sectional ranks using only a caller-supplied causal mask."""
    scores = np.asarray(raw_scores)
    if scores.ndim != 4:
        raise ValueError("Raw scores must have shape [day, refresh, name, horizon]")
    mask = _rank_mask(causal_mask, scores.shape)
    if np.any(mask & ~np.isfinite(scores)):
        raise ValueError("Valid raw scores must be finite")

    ranked = np.zeros(scores.shape, dtype=np.float64)
    for day in range(scores.shape[0]):
        for refresh in range(scores.shape[1]):
            for horizon in range(scores.shape[3]):
                valid = mask[day, refresh, :, horizon]
                count = int(valid.sum())
                if count:
                    ranks = average_ranks(scores[day, refresh, valid, horizon])
                    ranked[day, refresh, valid, horizon] = (
                        2.0 * ((ranks + 0.5) / count) - 1.0
                    )
    return ranked


def expand_refreshes(
    ranks: np.ndarray,
    valid: np.ndarray,
    refresh_minutes: np.ndarray,
    minute_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward-fill refresh states onto minutes without backfilling the first state."""
    values = np.asarray(ranks)
    if values.ndim != 4 or minute_count <= 0:
        raise ValueError("Ranks must be [day, refresh, name, horizon]")
    validity = _rank_mask(valid, values.shape)
    minutes = np.asarray(refresh_minutes, dtype=np.int64)
    if minutes.ndim == 1:
        if minutes.shape != (values.shape[1],):
            raise ValueError("Shared refresh minutes do not match the refresh axis")
        minutes = np.broadcast_to(minutes, values.shape[:2])
    if minutes.shape != values.shape[:2]:
        raise ValueError("Refresh minutes must be [refresh] or [day, refresh]")
    if np.any((minutes < 0) | (minutes >= minute_count)) or np.any(
        np.diff(minutes, axis=1) <= 0
    ):
        raise ValueError("Refresh minutes must be unique, increasing session indices")

    expanded = np.zeros(
        (values.shape[0], minute_count, values.shape[2], values.shape[3]),
        dtype=values.dtype,
    )
    expanded_valid = np.zeros(expanded.shape, dtype=bool)
    age = np.full((values.shape[0], minute_count), -1, dtype=np.int64)
    refresh_mask = np.zeros(age.shape, dtype=bool)
    clock = np.arange(minute_count)
    for day in range(values.shape[0]):
        source = np.searchsorted(minutes[day], clock, side="right") - 1
        on_grid = source >= 0
        expanded[day, on_grid] = values[day, source[on_grid]]
        expanded_valid[day, on_grid] = validity[day, source[on_grid]]
        age[day, on_grid] = clock[on_grid] - minutes[day, source[on_grid]]
        refresh_mask[day, minutes[day]] = True
    return expanded, expanded_valid, age, refresh_mask


def causal_liquidity(
    close: np.ndarray,
    real_volume: np.ndarray,
    observed: np.ndarray,
    lookback: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Prior-session ADV and same-minute median notional, excluding the current day."""
    prices = np.asarray(close, dtype=np.float64)
    volumes = np.asarray(real_volume, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if prices.shape != volumes.shape or prices.shape != mask.shape or prices.ndim != 3:
        raise ValueError("Liquidity inputs must align as [day, minute, name]")
    if lookback <= 0:
        raise ValueError("Liquidity lookback must be positive")
    if np.any(mask & (~np.isfinite(prices) | (prices <= 0.0))):
        raise ValueError("Observed closes must be finite and positive")
    if np.any(mask & (~np.isfinite(volumes) | (volumes < 0.0))):
        raise ValueError("Observed real volume must be finite and non-negative")

    notional = prices * volumes
    daily_valid = mask.any(axis=1)
    daily_notional = np.where(mask, notional, 0.0).sum(axis=1)
    adv = np.full((prices.shape[0], prices.shape[2]), np.nan, dtype=np.float64)
    minute_median = np.full(prices.shape, np.nan, dtype=np.float64)
    for day in range(1, prices.shape[0]):
        start = max(0, day - lookback)
        prior_daily_valid = daily_valid[start:day]
        counts = prior_daily_valid.sum(axis=0)
        totals = np.where(prior_daily_valid, daily_notional[start:day], 0.0).sum(axis=0)
        np.divide(totals, counts, out=adv[day], where=counts > 0)
        history = np.where(mask[start:day], notional[start:day], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            minute_median[day] = np.nanmedian(history, axis=0)
    return adv, minute_median


def causal_roll_spreads(
    close: np.ndarray,
    observed: np.ndarray,
    lookback: int = 60,
    minimum_pairs: int = 2,
) -> np.ndarray:
    """Estimate full Roll spreads from prior sessions only."""
    prices = np.asarray(close, dtype=np.float64)
    mask = np.asarray(observed, dtype=bool)
    if prices.shape != mask.shape or prices.ndim != 3:
        raise ValueError("Roll inputs must align as [day, minute, name]")
    if lookback <= 0 or minimum_pairs < 2:
        raise ValueError("Roll history and minimum pair count are invalid")
    if np.any(mask & (~np.isfinite(prices) | (prices <= 0))):
        raise ValueError("Observed Roll closes must be finite and positive")

    log_close = np.log(prices, where=mask, out=np.zeros_like(prices))
    triple = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    current = log_close[:, 2:] - log_close[:, 1:-1]
    lagged = log_close[:, 1:-1] - log_close[:, :-2]
    current = np.where(triple, current, 0.0)
    lagged = np.where(triple, lagged, 0.0)
    result = np.full((prices.shape[0], prices.shape[2]), np.nan, dtype=np.float64)
    for day in range(1, prices.shape[0]):
        start = max(0, day - lookback)
        history = slice(start, day)
        count = triple[history].sum(axis=(0, 1))
        sum_current = current[history].sum(axis=(0, 1))
        sum_lagged = lagged[history].sum(axis=(0, 1))
        sum_product = (current[history] * lagged[history]).sum(axis=(0, 1))
        safe_count = np.maximum(count, 1)
        covariance = (sum_product - sum_current * sum_lagged / safe_count) / np.maximum(
            count - 1, 1
        )
        valid = (count >= minimum_pairs) & (covariance < 0)
        result[day, valid] = 2.0 * np.sqrt(-covariance[valid])
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _previous_quarter(value: date) -> str:
    quarter = (value.month - 1) // 3 + 1
    return f"{value.year - 1}Q4" if quarter == 1 else f"{value.year}Q{quarter - 1}"


def lagged_quarter_spreads(
    schedule_path: Path,
    dates: Sequence[date],
    security_ids: Sequence[str],
    expected_sha256: str,
) -> np.ndarray:
    """Load full-spread fractions from only the previous completed quarter."""
    actual = _sha256(schedule_path)
    if actual.casefold() != expected_sha256.casefold():
        raise ValueError(f"Spread schedule SHA256 mismatch: {actual}")
    schedule = pl.read_parquet(schedule_path).select(
        "security_id", "quarter", "schedule_full_spread_fraction"
    )
    if schedule.select("security_id", "quarter").is_duplicated().any():
        raise ValueError("Spread schedule has duplicate security-quarter rows")
    by_key = {
        (str(security), str(quarter)): float(spread)
        for security, quarter, spread in schedule.iter_rows()
    }
    output = np.full((len(dates), len(security_ids)), np.nan, dtype=np.float64)
    for day, value in enumerate(dates):
        quarter = _previous_quarter(value)
        for name, security_id in enumerate(security_ids):
            spread = by_key.get((security_id, quarter))
            if spread is not None:
                output[day, name] = spread
    return output


def load_daily_cdi_rates(
    path: Path,
    dates: Sequence[date],
    expected_sha256: str,
) -> np.ndarray:
    """Load an explicit, hash-pinned per-session CDI return series."""
    actual = _sha256(path)
    if actual.casefold() != expected_sha256.casefold():
        raise ValueError(f"CDI series SHA256 mismatch: {actual}")
    rows = pl.read_parquet(path).select("trade_date", "daily_cdi_rate")
    if rows["trade_date"].dtype != pl.Date:
        raise ValueError("CDI trade_date must be a Parquet date")
    if rows["trade_date"].n_unique() != rows.height:
        raise ValueError("CDI series contains duplicate dates")
    by_date = {
        trade_date: float(rate)
        for trade_date, rate in rows.iter_rows()
        if rate is not None
    }
    result = np.asarray([by_date.get(value, np.nan) for value in dates])
    if not np.isfinite(result).all() or np.any(result <= -1.0):
        raise ValueError("CDI series is incomplete or contains an invalid daily rate")
    return result


def _discovery_split(manifest: dict[str, object]) -> str:
    split: object = manifest.get("split")
    if not isinstance(split, str) or split.casefold() not in _DISCOVERY_SPLITS:
        raise ValueError("Only canonical discovery-fold predictions may be loaded")
    return split.casefold()


def _source_prediction_manifest_sha256(
    path: Path,
    store_identity: Mapping[str, object],
    split: str,
    sample_id: np.ndarray,
    date_idx: np.ndarray,
) -> str:
    source = json.loads(path.read_text(encoding="utf-8"))
    source_split = source.get("split")
    if source.get("status") != "completed":
        raise ValueError("Source prediction manifest is not completed")
    if source.get("feature_store_identity") != store_identity:
        raise ValueError("Source prediction manifest uses a different feature store")

    if isinstance(source_split, dict):
        valid_split = (
            source_split.get("training") == split
            and source_split.get("selection") == split
        )
        access_sealed = (
            source_split.get("test_accessed") is False
            and source.get("official_validation_accessed") is not True
            and source.get("test_accessed") is not True
        )
        selection_window = source_split.get("selection_window")
    else:
        valid_split = False
        access_sealed = False
        selection_window = None
    if not valid_split:
        raise ValueError("Source prediction split differs from the execution wrapper")
    if not access_sealed:
        raise ValueError(
            "Source prediction manifest accessed an official or test split"
        )

    date_values = np.unique(np.asarray(date_idx, dtype=np.int64))
    sample_values = np.sort(np.asarray(sample_id, dtype=np.int64))
    expected_identity = {
        "date_count": int(date_values.size),
        "sample_count": int(sample_values.size),
        "date_identity_sha256": int64_identity_sha256(date_values),
        "sample_identity_sha256": int64_identity_sha256(sample_values),
    }
    if not isinstance(selection_window, dict) or any(
        selection_window.get(name) != value for name, value in expected_identity.items()
    ):
        raise ValueError("Source selection-window identity differs from the reference")
    return _sha256(path)


def _verify_oof_source_manifest(
    path: Path,
    *,
    store: Path,
    store_identity: Mapping[str, object],
    prediction_sha256: str,
    reference_sha256: str,
    date_idx: np.ndarray,
    source_fold_index: np.ndarray,
) -> str:
    from .splits import purged_training_folds

    source = json.loads(path.read_text(encoding="utf-8"))
    dates = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date").is_between(TRAIN_START, TRAIN_END))
        .sort("date_idx")
    )
    canonical_dates = tuple(dates["trade_date"])
    folds = purged_training_folds(canonical_dates)
    if (
        source.get("schema") != "BRAZIL_RV_OOF_MANUFACTURE_V1"
        or source.get("status") != "completed"
        or source.get("feature_store_identity") != store_identity
        or source.get("purged_folds") != folds.payload()
        or source.get("prediction_sha256") != prediction_sha256
        or source.get("reference_sha256") != reference_sha256
        or source.get("official_validation_accessed") is not False
        or source.get("test_accessed") is not False
    ):
        raise ValueError("OOF source manifest differs from the frozen contract")
    if (
        source_fold_index.shape != date_idx.shape
        or not np.issubdtype(source_fold_index.dtype, np.integer)
        or np.any((source_fold_index < 0) | (source_fold_index >= len(folds.folds)))
    ):
        raise ValueError("OOF source-fold identity is malformed")
    trade_date_by_idx = dict(dates.select("date_idx", "trade_date").iter_rows())
    for fold_index, fold in enumerate(folds.folds):
        emitted = {
            trade_date_by_idx[int(value)]
            for value in np.unique(date_idx[source_fold_index == fold_index])
        }
        if emitted != set(fold.heldout_dates):
            raise ValueError("OOF source fold emits a non-held-out date")
        if emitted.intersection(fold.fit_dates) or emitted.intersection(
            fold.embargo_dates
        ):
            raise ValueError("OOF source fold leaks a fit or embargo date")
    bindings = source.get("run_bindings")
    seeds = (11, 29, 47, 61, 79, 97, 113, 131, 149, 167)
    expected_keys = {
        f"{fold.name}/seed_{seed}" for fold in folds.folds for seed in seeds
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_keys:
        raise ValueError("OOF source manifest lacks an exact run binding")
    for fold in folds.folds:
        for seed in seeds:
            binding = bindings[f"{fold.name}/seed_{seed}"]
            if not isinstance(binding, dict):
                raise ValueError("OOF run binding is malformed")
            manifest_path = Path(str(binding.get("manifest", "")))
            if not manifest_path.is_absolute():
                manifest_path = path.parent / manifest_path
            if (
                not manifest_path.is_file()
                or binding.get("manifest_sha256") != _sha256(manifest_path)
            ):
                raise ValueError("OOF run-manifest binding hash differs")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            prediction_path = Path(str(binding.get("prediction", "")))
            reference_path = Path(str(binding.get("reference", "")))
            if not prediction_path.is_absolute():
                prediction_path = path.parent / prediction_path
            if not reference_path.is_absolute():
                reference_path = path.parent / reference_path
            if (
                not prediction_path.is_file()
                or not reference_path.is_file()
                or binding.get("prediction_sha256") != _sha256(prediction_path)
                or binding.get("reference_sha256") != _sha256(reference_path)
                or manifest.get("prediction_sha256") != _sha256(prediction_path)
                or manifest.get("reference_sha256") != _sha256(reference_path)
            ):
                raise ValueError("OOF run archive binding hash differs")
            training = manifest.get("training")
            proof = manifest.get("fit_exclusion_proof")
            if (
                manifest.get("schema") != "BRAZIL_RV_MONITOR_FREE_OOF_RUN_V1"
                or manifest.get("status") != "completed"
                or manifest.get("seed") != seed
                or manifest.get("source_fold_sha256") != fold.payload()["sha256"]
                or manifest.get("feature_store_identity") != store_identity
                or manifest.get("epochs_completed") != 20
                or manifest.get("monitor") is not None
                or not isinstance(training, dict)
                or training.get("heldout_evaluations_during_training") != 0
                or training.get("final_states")
                != ["epoch20_raw", "epoch20_ema_0995"]
                or not isinstance(proof, dict)
                or proof.get("fit_date_identity_sha256")
                != fold.payload()["fit_date_identity_sha256"]
                or proof.get("heldout_date_identity_sha256")
                != fold.payload()["heldout_date_identity_sha256"]
                or manifest.get("official_validation_accessed") is not False
                or manifest.get("test_accessed") is not False
            ):
                raise ValueError("OOF source run differs from its fold binding")
    return _sha256(path)


def write_discovery_prediction_manifest(
    path: Path,
    *,
    store: Path,
    prediction_path: Path,
    reference_path: Path,
    source_manifest_path: Path,
    split: str,
    refresh_minutes: Sequence[int],
    prediction_key: str,
) -> dict[str, object]:
    """Write the small execution wrapper for canonical prediction artifacts."""
    store = store.resolve()
    store_identity = feature_store_identity(store)
    with np.load(reference_path, allow_pickle=False) as reference:
        if "sample_id" not in reference or "date_idx" not in reference:
            raise ValueError("Prediction reference lacks source-window identity")
        sample_id = reference["sample_id"].copy()
        date_idx = reference["date_idx"].copy()
    source_sha256 = _source_prediction_manifest_sha256(
        source_manifest_path,
        store_identity,
        split.casefold(),
        sample_id,
        date_idx,
    )
    if not isinstance(prediction_key, str) or not prediction_key:
        raise ValueError("Prediction array key must be nonempty")
    refresh = list(refresh_minutes)
    if not refresh or any(not isinstance(value, int) for value in refresh):
        raise ValueError("Refresh minutes must be nonempty integers")
    payload: dict[str, object] = {
        "schema": EXECUTION_PREDICTION_ARCHIVE_SCHEMA,
        "split": split.casefold(),
        "official_validation_accessed": False,
        "test_accessed": False,
        "prediction_sha256": _sha256(prediction_path),
        "reference_sha256": _sha256(reference_path),
        "prediction_key": prediction_key,
        "feature_store_identity": store_identity,
        "axes": feature_store_axis_identity(store),
        "refresh_minutes": refresh,
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": source_sha256,
    }
    _discovery_split(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload


def _canonical_reference_contract(
    store: Path,
    sample_id: np.ndarray,
    date_idx: np.ndarray,
    decision_idx: np.ndarray,
) -> dict[int, int]:
    required = {
        "sample_id",
        "date_idx",
        "decision_idx",
        "trade_date",
        "equity_cutoff_index",
    }
    sample_index = pl.read_parquet(store / "sample_index.parquet")
    if not required.issubset(sample_index.columns):
        raise ValueError("Canonical sample index lacks execution timing identity")
    canonical = {
        int(row[0]): (int(row[1]), int(row[2]), row[3], int(row[4]))
        for row in sample_index.select(
            "sample_id",
            "date_idx",
            "decision_idx",
            "trade_date",
            "equity_cutoff_index",
        ).iter_rows()
    }
    cutoff_by_decision: dict[int, int] = {}
    for sample, date_value, decision in zip(
        sample_id.tolist(), date_idx.tolist(), decision_idx.tolist(), strict=True
    ):
        values = canonical.get(int(sample))
        if values is None or values[:2] != (int(date_value), int(decision)):
            raise ValueError(
                "Prediction reference differs from canonical sample identity"
            )
        _, _, trade_date, cutoff = values
        if trade_date > TRAIN_END:
            raise ValueError("Prediction reference extends beyond discovery dates")
        prior = cutoff_by_decision.setdefault(int(decision), cutoff)
        if prior != cutoff:
            raise ValueError("A decision ordinal maps to inconsistent session minutes")
    return cutoff_by_decision


def load_discovery_prediction_archive(
    prediction_path: Path,
    reference_path: Path,
    manifest_path: Path,
    store: Path,
) -> DiscoveryPredictionArchive:
    """Load a store-bound discovery archive without reading label masks.

    The execution wrapper manifest binds the prediction/reference bytes, store
    metadata and axes, prediction array key, split, and explicit session-minute
    refresh indices. Reference rows are then checked against the canonical
    sample index and bounded to ``TRAIN_END``. The causal rank mask is derived
    from store membership/readiness, never supplied by the caller.
    """
    store = store.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in (
        EXECUTION_PREDICTION_ARCHIVE_SCHEMA,
        OOF_PREDICTION_ARCHIVE_SCHEMA,
    ):
        raise ValueError("Prediction manifest has an unknown execution schema")
    is_oof = schema == OOF_PREDICTION_ARCHIVE_SCHEMA
    if is_oof:
        if manifest.get("split") != "oof_train":
            raise ValueError("OOF prediction archive split differs")
    else:
        _discovery_split(manifest)
    if (
        manifest.get("official_validation_accessed") is not False
        or manifest.get("test_accessed") is not False
    ):
        raise ValueError("Prediction archive accessed an official or test split")
    expected = {
        prediction_path: manifest.get("prediction_sha256"),
        reference_path: manifest.get("reference_sha256"),
    }
    for path, digest in expected.items():
        if not isinstance(digest, str) or _sha256(path).casefold() != digest.casefold():
            raise ValueError(f"Prediction archive hash mismatch: {path}")
    if manifest.get("feature_store_identity") != feature_store_identity(store):
        raise ValueError("Prediction archive feature store identity is misaligned")
    store_identity = manifest["feature_store_identity"]
    axes = feature_store_axis_identity(store)
    if manifest.get("axes") != axes:
        raise ValueError("Prediction archive date/equity axes are misaligned")
    prediction_key = manifest.get("prediction_key")
    if not isinstance(prediction_key, str) or not prediction_key:
        raise ValueError("Prediction manifest must identify its array key")
    source_value = manifest.get("source_manifest")
    source_sha256 = manifest.get("source_manifest_sha256")
    if not isinstance(source_value, str) or not isinstance(source_sha256, str):
        raise ValueError("Prediction archive source-manifest identity is missing")
    source_path = Path(source_value)
    if not source_path.is_absolute():
        source_path = manifest_path.parent / source_path

    with np.load(prediction_path, allow_pickle=False) as values:
        if prediction_key not in values:
            raise ValueError(f"Prediction archive has no {prediction_key!r} array")
        flat_scores = values[prediction_key].copy()
    with np.load(reference_path, allow_pickle=False) as values:
        required = (
            "sample_id",
            "date_idx",
            "decision_idx",
            *(("source_fold_index",) if is_oof else ()),
        )
        if any(name not in values for name in required):
            raise ValueError("Prediction reference lacks sample/date/decision identity")
        sample_id, date_idx, decision_idx = (
            values[name].copy()
            for name in ("sample_id", "date_idx", "decision_idx")
        )
        source_fold_index = values["source_fold_index"].copy() if is_oof else None
    if flat_scores.ndim != 3 or any(
        values.shape != (flat_scores.shape[0],)
        for values in (sample_id, date_idx, decision_idx)
    ):
        raise ValueError("Prediction and reference sample axes do not align")
    if (
        not all(
            np.issubdtype(values.dtype, np.integer)
            for values in (sample_id, date_idx, decision_idx)
        )
        or np.unique(sample_id).size != sample_id.size
    ):
        raise ValueError("Prediction reference identities must be unique integers")
    if flat_scores.shape[1] != axes["equity_count"]:
        raise ValueError("Prediction name axis differs from the canonical store")
    if is_oof:
        assert source_fold_index is not None
        actual_source_sha256 = _verify_oof_source_manifest(
            source_path,
            store=store,
            store_identity=store_identity,
            prediction_sha256=str(manifest["prediction_sha256"]),
            reference_sha256=str(manifest["reference_sha256"]),
            date_idx=date_idx,
            source_fold_index=source_fold_index,
        )
    else:
        actual_source_sha256 = _source_prediction_manifest_sha256(
            source_path,
            store_identity,
            _discovery_split(manifest),
            sample_id,
            date_idx,
        )
    if actual_source_sha256.casefold() != source_sha256.casefold():
        raise ValueError("Prediction archive source-manifest hash mismatch")

    cutoff_by_decision = _canonical_reference_contract(
        store, sample_id, date_idx, decision_idx
    )

    dates = np.unique(date_idx)
    decisions = np.unique(decision_idx)
    if dates.size * decisions.size != flat_scores.shape[0]:
        raise ValueError("Prediction archive is not a complete date-refresh grid")
    date_position = np.searchsorted(dates, date_idx)
    decision_position = np.searchsorted(decisions, decision_idx)
    pair = date_position * decisions.size + decision_position
    if np.unique(pair).size != pair.size:
        raise ValueError("Prediction archive has duplicate date-refresh rows")
    refresh_value = manifest.get("refresh_minutes")
    if (
        not isinstance(refresh_value, list)
        or len(refresh_value) != decisions.size
        or any(not isinstance(value, int) for value in refresh_value)
    ):
        raise ValueError("Prediction manifest refresh minutes do not align")
    refresh_minutes = np.asarray(refresh_value, dtype=np.int64)
    expected_minutes = np.asarray(
        [cutoff_by_decision[int(value)] for value in decisions], dtype=np.int64
    )
    if (
        not np.array_equal(refresh_minutes, expected_minutes)
        or np.any((refresh_minutes < 0) | (refresh_minutes >= EQUITY_SESSION_MINUTES))
        or np.any(np.diff(refresh_minutes) <= 0)
    ):
        raise ValueError("Prediction refresh minutes differ from canonical cutoffs")

    shape = (dates.size, decisions.size, *flat_scores.shape[1:])
    scores = np.empty(shape, dtype=flat_scores.dtype)
    sample_grid = np.empty((dates.size, decisions.size), dtype=sample_id.dtype)
    scores[date_position, decision_position] = flat_scores
    sample_grid[date_position, decision_position] = sample_id
    membership = np.load(
        store / "equity_membership.npy", mmap_mode="r", allow_pickle=False
    )
    ready = np.load(store / "equity_data_ready.npy", mmap_mode="r", allow_pickle=False)
    expected_activity_shape = (int(axes["date_count"]), int(axes["equity_count"]))
    if (
        membership.shape != expected_activity_shape
        or ready.shape != expected_activity_shape
    ):
        raise ValueError("Prediction store activity arrays differ from its axes")
    causal_activity = np.asarray(membership[dates] & ready[dates], dtype=bool)
    valid = np.broadcast_to(causal_activity[:, None, :, None], scores.shape).copy()
    return DiscoveryPredictionArchive(
        ranks=causal_rank_scores(scores, valid),
        valid=valid,
        sample_id=sample_grid,
        date_idx=dates,
        decision_idx=decisions,
        refresh_minutes=refresh_minutes,
    )
