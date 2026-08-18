from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import polars as pl
from numpy.lib.format import open_memmap

from ..modeling.contract import HORIZONS, VALIDATION_END, workspace_path
from ..modeling.data import (
    TARGET_SCALE_FILE,
    TARGET_SCALE_SCHEMA,
    target_scale_identity,
)
from .contract import EQUITY_SESSION_MINUTES, EQUITY_SESSION_START_MINUTE
from .io import (
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
from .transforms import build_equity_features, centered_midranks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_input(store_manifest: dict[str, object], name: str) -> Path:
    inputs = store_manifest.get("canonical_inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get(name), dict):
        raise ValueError(f"Feature store does not record canonical input {name}")
    value = inputs[name].get("resolved_path")
    if not isinstance(value, str):
        raise ValueError(f"Feature store canonical input {name} has no resolved path")
    return workspace_path(value)


def _validate_target_parity(store: Path, scale: np.ndarray) -> None:
    raw = np.load(store / "raw_returns.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(store / "label_mask.npy", mmap_mode="r", allow_pickle=False)
    targets = np.load(store / "targets.npy", mmap_mode="r", allow_pickle=False)
    medians = np.load(
        store / "cross_section_median.npy", mmap_mode="r", allow_pickle=False
    )
    for date_idx in range(scale.shape[0]):
        for decision_idx in range(raw.shape[2]):
            for horizon_idx, horizon in enumerate(HORIZONS):
                valid = masks[date_idx, :, decision_idx, horizon_idx]
                if not valid.any():
                    continue
                values = (
                    raw[date_idx, valid, decision_idx, horizon_idx].astype(np.float64)
                    - float(medians[date_idx, decision_idx, horizon_idx])
                ) / (scale[date_idx, valid] * np.sqrt(horizon))
                expected = centered_midranks(values)
                stored = targets[date_idx, valid, decision_idx, horizon_idx]
                if not np.allclose(expected, stored, rtol=0.0, atol=1e-7):
                    raise ValueError(
                        "Target-scale reconstruction disagrees with stored ranks at "
                        f"date_idx={date_idx}, decision_idx={decision_idx}, "
                        f"horizon={horizon}"
                    )


def build_target_scale_sidecar(
    store: Path,
    feature_store_identity: dict[str, object],
    output_dir: Path,
) -> Path:
    if output_dir.exists():
        target_scale_identity(output_dir, feature_store_identity)
        return output_dir

    store_manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assignments_dir = _recorded_input(store_manifest, "accepted_xp_assignments")
    cotahist_dir = _recorded_input(store_manifest, "parsed_cotahist")
    universe_dir = _recorded_input(store_manifest, "point_in_time_universe")
    research_start, research_end = read_research_interval(universe_dir)
    through = min(research_end, VALIDATION_END)
    assignments = load_assignments(assignments_dir)
    security_ids = tuple(assignments.get_column("security_id").to_list())
    market_dates, assignment_dates = load_market_dates_and_security_dates(
        cotahist_files(cotahist_dir),
        security_ids,
        research_start,
        through,
        allow_empty_security_dates=True,
    )
    validate_source_date_isolation(assignments, assignment_dates)
    date_index = (
        pl.read_parquet(store / "date_index.parquet")
        .filter(pl.col("trade_date") <= through)
        .sort("date_idx")
    )
    if tuple(date_index.get_column("trade_date")) != market_dates or not np.array_equal(
        date_index.get_column("date_idx").to_numpy(), np.arange(len(market_dates))
    ):
        raise ValueError("Feature-store dates differ from recorded source dates")
    equity_index = pl.read_parquet(store / "equity_index.parquet").sort("equity_slot")
    if tuple(equity_index.get_column("security_id")) != security_ids:
        raise ValueError(
            "Feature-store equity identity differs from recorded assignments"
        )

    partial = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    partial.mkdir(parents=True)
    scale_path = partial / TARGET_SCALE_FILE
    scale = open_memmap(
        scale_path,
        mode="w+",
        dtype=np.float64,
        shape=(len(market_dates), len(security_ids)),
    )
    scale[...] = 0.0
    slot_by_security = {
        security_id: slot for slot, security_id in enumerate(security_ids)
    }
    try:
        groups = assignments.partition_by("source_file", maintain_order=True)
        for source_number, group in enumerate(groups, start=1):
            source_path = Path(group.item(0, "source_file"))
            source = load_source_file(source_path)
            validate_physical_source_identity(group, source, source_path)
            allowed_dates = frozenset().union(
                *(assignment_dates[value] for value in group["security_id"])
            )
            session_bars = prepare_session_bars(
                source,
                source_path,
                allowed_dates,
                market_dates,
                EQUITY_SESSION_START_MINUTE,
                EQUITY_SESSION_MINUTES,
            )
            for assignment in group.iter_rows(named=True):
                security_id = assignment["security_id"]
                bars = session_bars.filter(
                    pl.col("trade_date").is_in(tuple(assignment_dates[security_id]))
                )
                raw_grid, observed = dense_grid(
                    bars, len(market_dates), EQUITY_SESSION_MINUTES
                )
                identity_day = np.fromiter(
                    (
                        assignment["first_overlap_date"]
                        <= trade_date
                        <= assignment["last_overlap_date"]
                        for trade_date in market_dates
                    ),
                    dtype=bool,
                    count=len(market_dates),
                )
                result = build_equity_features(
                    raw_grid,
                    observed,
                    identity_day,
                    market_dates=market_dates,
                )
                scale[:, slot_by_security[security_id]] = result.sigma
            if source_number % 20 == 0 or source_number == len(groups):
                print(f"Built exact target scales {source_number}/{len(groups)}")
        scale.flush()
        if not np.isfinite(scale).all():
            raise ValueError("Target scale contains non-finite values")
        label_mask = np.load(
            store / "label_mask.npy", mmap_mode="r", allow_pickle=False
        )[: len(market_dates)]
        required = label_mask.any(axis=(2, 3))
        if np.any(scale[required] <= 0.0):
            raise ValueError("A valid development label has no causal target scale")
        _validate_target_parity(store, scale)
        scale.flush()
        mapping = getattr(scale, "_mmap", None)
        if mapping is not None:
            mapping.close()
        del scale
        manifest = {
            "schema": TARGET_SCALE_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_feature_store": feature_store_identity,
            "through": through.isoformat(),
            "test_accessed": False,
            "shape": [len(market_dates), len(security_ids)],
            "dtype": "float64",
            "target_scale_sha256": _sha256(scale_path),
            "construction": (
                "exact causal equity sigma from recorded immutable sources"
            ),
        }
        (partial / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output_dir)
    except BaseException:
        current = locals().get("scale")
        if isinstance(current, np.memmap):
            current.flush()
            mapping = getattr(current, "_mmap", None)
            if mapping is not None:
                mapping.close()
        shutil.rmtree(partial, ignore_errors=True)
        raise
    target_scale_identity(output_dir, feature_store_identity)
    return output_dir
