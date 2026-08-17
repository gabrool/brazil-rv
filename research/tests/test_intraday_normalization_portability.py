from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import brazil_rv.modeling.contract as modeling_contract
from brazil_rv.modeling.data import _build_patch_batch, resolve_feature_store
from brazil_rv.modeling.feature_variant import OverlayArray
from brazil_rv.preprocessing.contract import EXPECTED_EQUITIES
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    iter_reconstructed_equities,
    load_source_context,
)


def _windows_data_path(relative: Path, *, workspace_prefix: bool = False) -> str:
    suffix = str(relative).replace("/", "\\")
    prefix = "C:\\Brazil-RV\\quant-data" if workspace_prefix else "C:\\quant-data"
    return f"{prefix}\\{suffix}"


@pytest.mark.parametrize("workspace_prefix", (False, True))
def test_workspace_path_rebases_supported_windows_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workspace_prefix: bool,
) -> None:
    root = tmp_path / "workspace"
    relative = Path("b3") / "interim" / "portable" / "artifact"
    target = root / "quant-data" / relative
    target.mkdir(parents=True)
    recorded = _windows_data_path(
        relative,
        workspace_prefix=workspace_prefix,
    )

    assert (
        modeling_contract.workspace_path(recorded, project_root=root)
        == target.resolve()
    )
    pointer = tmp_path / "feature_store_pointer.txt"
    pointer.write_text(recorded, encoding="utf-8")
    monkeypatch.setattr(modeling_contract, "PROJECT_ROOT", root)
    assert resolve_feature_store(pointer) == target.resolve()


def test_workspace_path_preserves_native_paths_and_rejects_ambiguity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    native = tmp_path / "native"
    native.mkdir()
    missing = root / "quant-data" / "portable" / "obsolete"

    assert (
        modeling_contract.workspace_path(native, project_root=root) == native.resolve()
    )
    assert (
        modeling_contract.workspace_path(
            r"C:\quant-data\portable\obsolete",
            must_exist=False,
            project_root=root,
        )
        == missing.resolve()
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        modeling_contract.workspace_path(
            r"C:\unsupported\quant-data\artifact",
            project_root=root,
        )
    with pytest.raises(ValueError):
        modeling_contract.workspace_path(
            r"C:\quant-data\..\outside",
            must_exist=False,
            project_root=root,
        )
    with pytest.raises(ValueError):
        modeling_contract.workspace_path("", project_root=root)


def _write_portable_source_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "workspace"
    data_root = root / "quant-data"
    relative = Path("portable") / tmp_path.name
    base = data_root / relative
    parent = base / "parent"
    assignments_dir = base / "assignments"
    cotahist_dir = base / "cotahist"
    universe_dir = base / "universe"
    sources_dir = base / "sources"
    for directory in (
        parent,
        assignments_dir,
        cotahist_dir / "year=2025",
        universe_dir,
        sources_dir,
    ):
        directory.mkdir(parents=True)

    security_ids = tuple(f"SEC-{slot:03d}" for slot in range(EXPECTED_EQUITIES))
    trade_date = modeling_contract.VALIDATION_END
    pl.DataFrame(
        {
            "date_idx": pl.Series([0], dtype=pl.Int32),
            "trade_date": [trade_date],
        }
    ).write_parquet(parent / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": pl.Series(range(EXPECTED_EQUITIES), dtype=pl.Int16),
            "security_id": security_ids,
        }
    ).write_parquet(parent / "equity_index.parquet")

    source_paths = [
        _windows_data_path(relative / "sources" / f"source-{slot:03d}.parquet")
        for slot in range(EXPECTED_EQUITIES)
    ]
    pl.DataFrame(
        {
            "security_id": security_ids,
            "isin": [f"BR{slot:010d}" for slot in range(EXPECTED_EQUITIES)],
            "latest_ticker": [f"T{slot:03d}" for slot in range(EXPECTED_EQUITIES)],
            "xp_symbol": [f"XP{slot:03d}" for slot in range(EXPECTED_EQUITIES)],
            "source_file": source_paths,
            "source_assignment_type": ["EXACT"] * EXPECTED_EQUITIES,
            "first_overlap_date": [str(trade_date)] * EXPECTED_EQUITIES,
            "last_overlap_date": [str(trade_date)] * EXPECTED_EQUITIES,
            "manual_decision": ["ACCEPTED"] * EXPECTED_EQUITIES,
            "normalization_rule": ["FILTER_TO_COTAHIST_SECURITY_DATES"]
            * EXPECTED_EQUITIES,
        }
    ).write_parquet(assignments_dir / "xp_accepted_source_assignments_v1.parquet")
    pl.DataFrame(
        {
            "trade_date": [trade_date] * EXPECTED_EQUITIES,
            "security_id": security_ids,
        }
    ).write_parquet(cotahist_dir / "year=2025" / "equities_daily_fixture.parquet")
    (universe_dir / "manifest.json").write_text(
        json.dumps(
            {
                "resolved_start_date": str(trade_date),
                "resolved_end_date": str(trade_date),
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "ts_exchange": [datetime(2025, 6, 30, 10, 0)],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "real_volume": [100.0],
            "symbol": ["XP000"],
        }
    ).write_parquet(sources_dir / "source-000.parquet")
    manifest = {
        "contract_version": "portable-fixture",
        "canonical_inputs": {
            "accepted_xp_assignments": {
                "resolved_path": _windows_data_path(
                    relative / "assignments",
                    workspace_prefix=True,
                )
            },
            "parsed_cotahist": {
                "resolved_path": _windows_data_path(relative / "cotahist")
            },
            "point_in_time_universe": {
                "resolved_path": _windows_data_path(
                    relative / "universe",
                    workspace_prefix=True,
                )
            },
        },
    }
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(modeling_contract, "PROJECT_ROOT", root)
    return parent, sources_dir / "source-000.parquet"


class _DevelopmentOnlyArray:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim
        self.accessed_dates: list[int] = []

    def __getitem__(self, key: object) -> object:
        date_key = key[0] if isinstance(key, tuple) else key
        dates = np.asarray(np.arange(self.shape[0])[date_key]).reshape(-1)
        if dates.size:
            self.accessed_dates.extend(int(value) for value in dates)
        if dates.size and int(dates.max()) >= 1:
            pytest.fail("candidate batch read the held-out feature tail")
        return self.values[key]


def test_portable_source_context_and_candidate_batch_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, first_source = _write_portable_source_context(tmp_path, monkeypatch)

    context = load_source_context(parent)
    first = next(iter_reconstructed_equities(context))

    assert context.allowed_date_count == 1
    assert first.source_path == first_source.resolve()

    parent_features = np.zeros(
        (2, EXPECTED_EQUITIES, 15, 26),
        dtype=np.float32,
    )
    parent_features[1] = 999.0
    guarded = _DevelopmentOnlyArray(parent_features)
    overlay = np.zeros(
        (1, EXPECTED_EQUITIES, 15, len(AFFECTED_DYNAMIC_CHANNELS)),
        dtype=np.float32,
    )
    overlay[..., 0] = 0.25
    arrays = {
        "equity_features.npy": OverlayArray(
            guarded,
            overlay,
            AFFECTED_DYNAMIC_CHANNELS,
        ),
        "equity_slow.npy": np.zeros(
            (2, EXPECTED_EQUITIES, modeling_contract.SLOW_FEATURE_COUNT),
            dtype=np.float32,
        ),
    }
    batch = _build_patch_batch(
        arrays,
        np.asarray([0]),
        np.asarray([15]),
        np.asarray([0]),
        np.asarray([75]),
        np.ones((1, EXPECTED_EQUITIES), dtype=bool),
        None,
    )

    assert batch["patches"].shape[0] == 1
    assert set(guarded.accessed_dates) == {0}
