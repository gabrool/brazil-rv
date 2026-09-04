import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from torch.utils.data import DataLoader

import brazil_rv.v2.build_store as build_store_module
import brazil_rv.v2.store as store_module
from brazil_rv.v2.build_store import (
    EXPECTED_V1_DATES,
    MinutePanel,
    V1_STORE_START,
    _feature_validity_by_survival,
    _require_clean_implementation_commit,
    _validate_v1_calendar,
    build_daily_store,
)
from brazil_rv.v2.contract import ACCUMULATED_TEST_AFTER, FINETUNE_START
from brazil_rv.v2.data import (
    V1_STORE_V2_ZERO_DYNAMIC_CHANNELS,
    V1_STORE_V2_ZERO_SLOW_FIELDS,
    V2DailyDataset,
)
from brazil_rv.v2.store import (
    V2Store,
    open_store_for_dates,
    open_store_for_samples,
    sha256_file,
    write_store,
)


def test_store_cli_requires_clean_worktree_before_binding_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def dirty_run(command, **kwargs):
        del kwargs
        calls.append(tuple(command))
        return SimpleNamespace(stdout="?? untracked.txt\n")

    monkeypatch.setattr(build_store_module.subprocess, "run", dirty_run)
    with pytest.raises(ValueError, match="clean tracked and untracked worktree"):
        _require_clean_implementation_commit(tmp_path, "a" * 40)
    assert calls == [
        ("git", "status", "--porcelain=v1", "--untracked-files=all")
    ]

    responses = iter((SimpleNamespace(stdout=""), SimpleNamespace(stdout="b" * 40)))
    monkeypatch.setattr(
        build_store_module.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )
    with pytest.raises(ValueError, match="does not match"):
        _require_clean_implementation_commit(tmp_path, "a" * 40)


def test_feature_validity_survivorship_gate_uses_family_population() -> None:
    dates = np.asarray(
        ["2023-12-27", "2023-12-28", "2023-12-29", "2024-01-02"],
        dtype="datetime64[D]",
    )
    observed = np.asarray(
        [[True, True], [True, True], [True, True], [False, True]]
    )
    active = observed.copy()
    present = observed.copy()
    balanced = np.ones((4, 2, 2), dtype=np.bool_)
    table = _feature_validity_by_survival(
        dates,
        active,
        observed,
        {"slow": (balanced, present)},
    )
    assert table.height == 2

    skewed = balanced.copy()
    skewed[:3, 0, 0] = False
    with pytest.raises(ValueError, match="more than 5 percentage points"):
        _feature_validity_by_survival(
            dates,
            active,
            observed,
            {"slow": (skewed, present)},
        )
def _base_store(tmp_path, *, external_fast=None, stored_fast_present=None):
    days, names = 25, 3
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(days)]
    slow = np.broadcast_to(
        np.arange(days, dtype=np.float32)[:, None, None], (days, names, 2)
    ).copy()
    arrays = {
        "slow_values": slow,
        "slow_valid": np.ones_like(slow, dtype=bool),
        "active": np.ones((days, names), dtype=bool),
        "target_to_close": np.ones((days, names), dtype=np.float32),
        "target_to_close_valid": np.ones((days, names), dtype=bool),
    }
    metadata = {}
    tables = {}
    if external_fast is not None:
        metadata = {
            "v1_fast_store": str(external_fast),
            "v1_fast_files": [
                {
                    "path": str(external_fast / name),
                    "bytes": (external_fast / name).stat().st_size,
                    "sha256": sha256_file(external_fast / name),
                }
                for name in (
                    "equity_features.npy",
                    "equity_slow.npy",
                    "equity_data_ready.npy",
                )
            ],
            "v1_store_v2_zero_slow_fields": list(
                V1_STORE_V2_ZERO_SLOW_FIELDS
            ),
        }
        tables = {
            "v1_fast_date_mapping": pl.DataFrame(
                {
                    "trade_date": [dates[20]],
                    "v2_date_index": [20],
                    "v1_date_index": [0],
                }
            ),
            "v1_fast_isin_mapping": pl.DataFrame(
                {
                    "isin": ["BRTESTACNOR1", "BRTESTACNPR0"],
                    "security_id": ["one", "two"],
                    "v2_isin_index": [0, 2],
                    "v1_equity_slot": [0, 1],
                }
            ),
        }
    if stored_fast_present is not None:
        arrays["fast_present"] = np.asarray(stored_fast_present, dtype=bool)
    path = write_store(
        tmp_path / "store",
        dates=dates,
        isins=("BRTESTACNOR1", "BRANOTHRNOR1", "BRTESTACNPR0"),
        arrays=arrays,
        metadata=metadata,
        tables=tables,
    )
    return path


def test_store_is_immutable_and_hash_verified(tmp_path) -> None:
    path = _base_store(tmp_path)
    store, _ = open_store_for_dates(
        path, list(range(25)), purpose="training"
    )
    assert store.array_shape("slow_values") == (25, 3, 2)
    assert store.array_dtype("slow_values") == np.dtype(np.float32)
    assert store.read("slow_values", 0).shape == (3, 2)
    store.close()
    with pytest.raises(FileExistsError):
        write_store(
            path,
            dates=[date(2024, 1, 1)],
            isins=["BRTESTACNOR1"],
            arrays={"active": np.ones((1, 1), dtype=bool)},
        )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    array_path = path / manifest["arrays"]["slow_values"]["path"]
    with array_path.open("r+b") as stream:
        stream.seek(-1, 2)
        final = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([final[0] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        open_store_for_dates(path, [0], purpose="training")


def test_store_hashes_are_cached_per_process_for_unchanged_files(
    tmp_path, monkeypatch
) -> None:
    path = _base_store(tmp_path)
    store_module._VERIFIED_HASHES.clear()
    original = store_module.sha256_file
    calls = 0

    def counted(item: Path) -> str:
        nonlocal calls
        calls += 1
        return original(item)

    monkeypatch.setattr(store_module, "sha256_file", counted)
    first, _ = open_store_for_dates(path, list(range(25)), purpose="training")
    first.close()
    first_count = calls
    second, _ = open_store_for_dates(path, list(range(25)), purpose="training")
    second.close()
    assert first_count > 0
    assert calls == first_count


def test_store_writer_selects_source_rows_one_array_at_a_time(tmp_path) -> None:
    path = write_store(
        tmp_path / "selected_rows",
        dates=[date(2024, 1, 2), date(2024, 1, 4)],
        isins=["BRTESTACNOR1"],
        arrays={
            "slow_values": np.arange(4, dtype=np.float32).reshape(4, 1, 1),
            "slow_valid": np.ones((4, 1, 1), dtype=np.bool_),
            "active": np.ones((4, 1), dtype=np.bool_),
        },
        row_indices=np.asarray([1, 3], dtype=np.int64),
    )
    store, _ = open_store_for_dates(path, [0, 1], purpose="training")
    assert store.array_shape("slow_values") == (2, 1, 1)
    assert store.read("slow_values", [0, 1]).ravel().tolist() == [1.0, 3.0]
    with pytest.raises(ValueError, match="exactly one row"):
        write_store(
            tmp_path / "wrong_selected_rows",
            dates=[date(2024, 1, 2)],
            isins=["BRTESTACNOR1"],
            arrays={"active": np.ones((4, 1), dtype=np.bool_)},
            row_indices=[0, 1],
        )


def test_store_writer_does_not_copy_a_contiguous_row_selection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = np.arange(6, dtype=np.float32).reshape(6, 1, 1)
    original_save = np.save
    slow_shares_memory: list[bool] = []

    def tracking_save(path, value, *args, **kwargs):
        if Path(path).name == "slow_values.npy":
            slow_shares_memory.append(np.shares_memory(value, slow))
        return original_save(path, value, *args, **kwargs)

    monkeypatch.setattr(np, "save", tracking_save)
    write_store(
        tmp_path / "contiguous_rows",
        dates=[date(2024, 1, 2) + timedelta(days=index) for index in range(4)],
        isins=["BRTESTACNOR1"],
        arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "active": np.ones((6, 1), dtype=np.bool_),
        },
        row_indices=np.arange(2, 6, dtype=np.int64),
    )
    assert slow_shares_memory == [True]
    store, _ = open_store_for_dates(
        tmp_path / "contiguous_rows", [0, 1, 2, 3], purpose="training"
    )
    assert store.read("slow_values", [0, 1, 2, 3]).ravel().tolist() == [
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    store.close()


def test_dataset_applies_slow_shift_and_external_sparse_fast_mapping(tmp_path) -> None:
    fast = tmp_path / "v1"
    fast.mkdir()
    features = np.ones((1, 2, 405, 26), dtype=np.float32)
    np.save(fast / "equity_features.npy", features, allow_pickle=False)
    legacy_slow = np.arange(64, dtype=np.float32).reshape(1, 2, 32)
    np.save(fast / "equity_slow.npy", legacy_slow, allow_pickle=False)
    np.save(
        fast / "equity_data_ready.npy",
        np.array([[True, False]]),
        allow_pickle=False,
    )
    path = _base_store(tmp_path, external_fast=fast)
    dataset = V2DailyDataset(path, [20], stage="finetune", lookback=20)
    sample = dataset[0]
    assert sample["slow_features"][0, -1, 0] == 19.0
    assert sample["days_since_last_slow_row"][0] == 1.0
    assert sample["fast_present"].tolist() == [True, False, False]
    expected_slow = legacy_slow[0, 0].copy()
    expected_slow[list(V1_STORE_V2_ZERO_SLOW_FIELDS)] = 0.0
    assert sample["v1_equity_slow"].shape == (3, 32)
    np.testing.assert_array_equal(sample["v1_equity_slow"][0], expected_slow)
    assert not sample["v1_equity_slow"][1:].any()
    assert "fast_state_position" not in sample
    patches = sample["fast_patches"]
    assert patches.shape == (3, 69, 130)
    for channel in V1_STORE_V2_ZERO_DYNAMIC_CHANNELS:
        assert np.all(patches[0, :, channel::26] == 0.0)
    kept = set(range(26)) - set(V1_STORE_V2_ZERO_DYNAMIC_CHANNELS)
    assert all(np.all(patches[0, :, channel::26] == 1.0) for channel in kept)
    assert np.all(patches[1:] == 0.0)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    assert tuple(batch["v1_equity_slow"].shape) == (1, 3, 32)
    np.testing.assert_array_equal(batch["v1_equity_slow"][0].numpy(), sample["v1_equity_slow"])


def test_finetune_dataset_keeps_first_active_day_with_empty_slow_history(
    tmp_path: Path,
) -> None:
    days, names = 25, 2
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(days)]
    slow = np.zeros((days, names, 1), dtype=np.float32)
    slow_valid = np.ones_like(slow, dtype=np.bool_)
    slow_valid[:20, 1] = False
    active = np.ones((days, names), dtype=np.bool_)
    active[:20, 1] = False
    targets = np.zeros((days, names, 5), dtype=np.float32)
    target_valid = np.ones_like(targets, dtype=np.bool_)
    path = write_store(
        tmp_path / "first_active_day",
        dates=dates,
        isins=("BRTESTACNOR1", "BRNEWCOACNOR2"),
        arrays={
            "slow_values": slow,
            "slow_valid": slow_valid,
            "active": active,
            "target_primary": targets,
            "target_valid": target_valid,
        },
    )

    sample = V2DailyDataset(
        path, list(range(20, 25)), stage="finetune", lookback=20
    )[0]

    assert sample["active_mask"].tolist() == [True, True]
    assert not sample["slow_history_mask"][1].any()
    assert not sample["slow_features"][1].any()
    assert sample["target_mask"][1].tolist() == [True, True, True, False, False]


def test_external_fast_ready_is_intersected_with_store_presence(tmp_path) -> None:
    fast = tmp_path / "v1"
    fast.mkdir()
    np.save(
        fast / "equity_features.npy",
        np.ones((1, 2, 405, 26), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        fast / "equity_slow.npy",
        np.full((1, 2, 32), 7.0, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        fast / "equity_data_ready.npy",
        np.ones((1, 2), dtype=bool),
        allow_pickle=False,
    )
    stored = np.zeros((25, 3), dtype=bool)
    stored[20, 0] = True
    path = _base_store(
        tmp_path,
        external_fast=fast,
        stored_fast_present=stored,
    )
    sample = V2DailyDataset(path, [20], stage="finetune", lookback=20)[0]
    assert sample["fast_present"].tolist() == [True, False, False]
    assert sample["to_close_mask"].tolist() == [True, False, False]
    assert not sample["fast_patch_mask"][2].any()
    assert np.all(sample["fast_patches"][2] == 0.0)
    retained = sorted(set(range(32)) - set(V1_STORE_V2_ZERO_SLOW_FIELDS))
    assert np.all(sample["v1_equity_slow"][0, retained] == 7.0)
    assert not sample["v1_equity_slow"][0, V1_STORE_V2_ZERO_SLOW_FIELDS].any()
    assert not sample["v1_equity_slow"][1:].any()


def test_external_v1_slow_is_hash_bound(tmp_path) -> None:
    fast = tmp_path / "v1"
    fast.mkdir()
    np.save(
        fast / "equity_features.npy",
        np.ones((1, 2, 405, 26), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        fast / "equity_slow.npy",
        np.ones((1, 2, 32), dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        fast / "equity_data_ready.npy",
        np.ones((1, 2), dtype=np.bool_),
        allow_pickle=False,
    )
    path = _base_store(tmp_path, external_fast=fast)
    changed = np.load(fast / "equity_slow.npy", allow_pickle=False)
    changed[0, 0, 0] = 2.0
    np.save(fast / "equity_slow.npy", changed, allow_pickle=False)
    with pytest.raises(ValueError, match="external v1 fast hash mismatch"):
        V2DailyDataset(path, [20], stage="finetune", lookback=20)


def test_joint_dataset_uses_each_windows_entry_alignment(tmp_path) -> None:
    dates = [date(2021, 7, 1) + timedelta(days=index) for index in range(25)]
    dates.append(date(2021, 8, 16))
    slow = np.broadcast_to(
        np.arange(26, dtype=np.float32)[:, None, None], (26, 1, 1)
    ).copy()
    path = write_store(
        tmp_path / "joint",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=bool),
            "active": np.ones((26, 1), dtype=bool),
        },
    )
    dataset = V2DailyDataset(path, [24, 25], stage="joint", lookback=20)
    pretrain, finetune = dataset[0], dataset[1]
    assert pretrain["slow_features"][0, -1, 0] == 24.0
    assert pretrain["days_since_last_slow_row"][0] == 0.0
    assert finetune["slow_features"][0, -1, 0] == 24.0
    assert finetune["days_since_last_slow_row"][0] == 1.0
    assert not pretrain["v1_equity_slow"].any()
    assert not finetune["v1_equity_slow"].any()


def test_store_with_sealed_dates_rejects_direct_ungated_open(tmp_path) -> None:
    path = write_store(
        tmp_path / "sealed",
        dates=[date(2025, 1, 2)],
        isins=["BRTESTACNOR1"],
        arrays={"active": np.ones((1, 1), dtype=bool)},
    )
    with pytest.raises(PermissionError, match="open_store_for_dates"):
        V2Store.open(path)
    with pytest.raises(PermissionError, match="preregistration"):
        open_store_for_dates(path, [0], purpose="evaluation")


def test_direct_store_open_is_always_gated_and_capability_is_date_bounded(
    tmp_path,
) -> None:
    path = _base_store(tmp_path)
    with pytest.raises(PermissionError, match="open_store_for_dates"):
        V2Store.open(path)
    store, _ = open_store_for_dates(path, [20], purpose="training")
    with pytest.raises(PermissionError, match="not authorized"):
        V2DailyDataset(store, [21], stage="finetune", lookback=20)


def test_authorized_store_never_exposes_whole_or_ungranted_array_rows(
    tmp_path,
) -> None:
    dates = [date(2024, 12, 30), date(2025, 1, 2), date(2026, 1, 2)]
    path = write_store(
        tmp_path / "mixed_windows",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays={"active": np.asarray([[True], [False], [True]])},
    )
    store, _ = open_store_for_dates(path, [0], purpose="training")
    assert not hasattr(store, "arrays")
    assert not hasattr(store, "require")
    assert store.read("active", 0).tolist() == [True]
    for selector in (1, 2, slice(0, 2), range(0, 2), slice(None)):
        with pytest.raises(PermissionError, match="authorization grant"):
            store.read("active", selector)
    with pytest.raises(IndexError, match="negative"):
        store.read("active", -1)


def test_authorized_store_exposes_only_bounded_runtime_tables(tmp_path) -> None:
    dates = [date(2024, 12, 30), date(2025, 1, 2)]
    path = write_store(
        tmp_path / "table_capability",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays={"active": np.ones((2, 1), dtype=np.bool_)},
        tables={
            "v1_fast_date_mapping": pl.DataFrame(
                {
                    "trade_date": dates,
                    "v2_date_index": [0, 1],
                    "v1_date_index": [8, 9],
                }
            ),
            "v1_fast_isin_mapping": pl.DataFrame(
                {
                    "isin": ["BRTESTACNOR1"],
                    "security_id": ["one"],
                    "v2_isin_index": [0],
                    "v1_equity_slot": [0],
                }
            ),
            "universe_size": pl.DataFrame(
                {"trade_date": dates, "member_count": [1, 1]}
            ),
        },
    )
    store, _ = open_store_for_dates(path, [0], purpose="training")
    bounded = store.read_table("v1_fast_date_mapping", [0])
    assert bounded.get_column("trade_date").to_list() == [dates[0]]
    assert store.read_table("v1_fast_isin_mapping").height == 1
    with pytest.raises(PermissionError, match="authorized dates"):
        store.read_table("v1_fast_date_mapping")
    with pytest.raises(PermissionError, match="sealed store capability"):
        store.read_table("universe_size", [0])
    with pytest.raises(PermissionError, match="authorization grant"):
        store.read_table("v1_fast_date_mapping", [1])


def test_causal_history_capability_allows_only_bounded_pre_sample_rows(
    tmp_path,
) -> None:
    dates = [
        value.astype(object)
        for value in np.arange(
            np.datetime64("2021-07-01"), np.datetime64("2021-08-17")
        )
        if np.is_busday(value)
    ]
    dates.extend((date(2025, 1, 2), date(2026, 1, 2)))
    path = write_store(
        tmp_path / "causal_gap",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays={"active": np.ones((len(dates), 1), dtype=np.bool_)},
    )
    sample_index = dates.index(FINETUNE_START)
    gap_index = dates.index(date(2021, 8, 2))
    store, ledger = open_store_for_samples(
        path,
        [sample_index],
        purpose="training",
        history_lookbacks=20,
        history_end_offsets=-1,
    )
    assert ledger.purpose == "training"
    assert not ledger.official_validation_accessed
    assert store.read("active", gap_index).tolist() == [True]
    for forbidden in (0, len(dates) - 2, len(dates) - 1):
        with pytest.raises(PermissionError, match="authorization grant"):
            store.read("active", forbidden)
    exact, _ = open_store_for_dates(path, [sample_index], purpose="training")
    with pytest.raises(PermissionError, match="authorization grant"):
        exact.read("active", gap_index)
    with pytest.raises(ValueError, match="frozen slow or baseline span"):
        open_store_for_samples(
            path,
            [sample_index],
            purpose="training",
            history_lookbacks=254,
            history_end_offsets=-1,
        )


def test_dataset_clips_f3_tail_and_sealed_target_endpoints(
    tmp_path,
) -> None:
    dates = [
        value.astype(object)
        for value in np.arange(
            np.datetime64("2024-11-25"), np.datetime64("2024-12-31")
        )
        if np.is_busday(value)
    ]
    dates.append(date(2025, 1, 2))
    days = len(dates)
    target = np.full((days, 1, 5), 11.0, dtype=np.float32)
    raw_target = np.full((days, 1, 5), 22.0, dtype=np.float32)
    raw_return = np.full((days, 1, 5), 33.0, dtype=np.float32)
    target_valid = np.ones_like(target, dtype=np.bool_)
    to_close = np.full((days, 1), 44.0, dtype=np.float32)
    to_close_valid = np.ones_like(to_close, dtype=np.bool_)
    to_close_valid[-3, 0] = False
    slow = np.zeros((days, 1, 1), dtype=np.float32)
    path = write_store(
        tmp_path / "target_endpoint",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "active": np.ones((days, 1), dtype=np.bool_),
            "target_primary": target,
            "target_valid": target_valid,
            "target_raw_midrank": raw_target,
            "target_raw_valid": target_valid,
            "target_raw_log_return": raw_return,
            "target_to_close": to_close,
            "target_to_close_valid": to_close_valid,
        },
    )
    dataset = V2DailyDataset(
        path, [days - 3, days - 2], stage="evaluation", lookback=20
    )
    first = dataset[0]
    second = dataset[1]
    assert first["target_mask"].tolist() == [[True, False, False, False, False]]
    assert first["raw_target_mask"].tolist() == [
        [True, False, False, False, False]
    ]
    assert first["targets"].tolist() == [[11.0, 0.0, 0.0, 0.0, 0.0]]
    assert first["raw_targets"].tolist() == [[22.0, 0.0, 0.0, 0.0, 0.0]]
    assert first["raw_log_returns"].tolist() == [
        [33.0, 0.0, 0.0, 0.0, 0.0]
    ]
    assert first["to_close_mask"].tolist() == [False]
    assert first["to_close_target"].tolist() == [0.0]
    assert second["target_mask"].tolist() == [[False] * 5]
    assert second["targets"].tolist() == [[0.0] * 5]
    assert second["raw_targets"].tolist() == [[0.0] * 5]
    assert second["raw_log_returns"].tolist() == [[0.0] * 5]
    assert second["to_close_mask"].tolist() == [False]
    assert second["to_close_target"].tolist() == [0.0]
    direct_targets = dataset.store.read(
        "target_primary", [days - 3, days - 2]
    )
    direct_mask = dataset.store.read("target_valid", [days - 3, days - 2])
    assert direct_targets[:, 0].tolist() == [
        [11.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert direct_mask[:, 0].tolist() == [
        [True, False, False, False, False],
        [False, False, False, False, False],
    ]
    gappy = V2DailyDataset(
        path, [days - 4, days - 2], stage="evaluation", lookback=20
    )
    assert gappy[0]["target_mask"].tolist() == [[False] * 5]
    with pytest.raises(PermissionError, match="authorization grant"):
        dataset.store.read("target_primary", days - 1)


def test_dataset_rejects_dates_outside_its_stage_before_array_open(tmp_path) -> None:
    path = _base_store(tmp_path)
    with pytest.raises(ValueError, match="pretrain stage"):
        V2DailyDataset(path, [20], stage="pretrain", lookback=20)

    dates = [date(2021, 7, 30), date(2021, 8, 5), FINETUNE_START]
    path = write_store(
        tmp_path / "stage_windows",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays={
            "slow_values": np.zeros((3, 1, 1), dtype=np.float32),
            "slow_valid": np.ones((3, 1, 1), dtype=bool),
            "active": np.ones((3, 1), dtype=bool),
        },
    )
    with pytest.raises(ValueError, match="finetune stage"):
        V2DailyDataset(path, [0], stage="finetune", lookback=20)
    with pytest.raises(ValueError, match="joint stage"):
        V2DailyDataset(path, [1], stage="joint", lookback=20)


def test_v1_calendar_includes_physical_warmup_before_finetune() -> None:
    calendar = [
        V1_STORE_START + timedelta(days=index)
        for index in range(EXPECTED_V1_DATES - 1)
    ]
    calendar.append(ACCUMULATED_TEST_AFTER)
    _validate_v1_calendar(calendar)
    assert V1_STORE_START < FINETUNE_START
    with pytest.raises(ValueError, match="fixed axis"):
        _validate_v1_calendar([FINETUNE_START, *calendar[1:]])


def test_store_to_close_uses_cotahist_close_anchor(tmp_path) -> None:
    days = 70
    names = ("BRTESTACNOR1", "BRTESTACNPR0")
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]
    daily_rows = []
    cotahist_close = float((100.0 + 404 * 0.01) * 1.0001)
    for day_index, day in enumerate(dates):
        for name_index, isin in enumerate(names):
            daily_rows.append(
                {
                    "trade_date": day,
                    "isin": isin,
                    "ticker": f"TEST{name_index + 3}",
                    "security_spec_base": "ON",
                    "bdi_code": "02",
                    "market_type": 10,
                    "open_brl": cotahist_close,
                    "high_brl": cotahist_close * 1.01,
                    "low_brl": cotahist_close * 0.99,
                    "close_brl": cotahist_close,
                    "volume_brl": 3_000_000.0,
                    "trades": 100.0,
                    "quantity": 100_000.0,
                    "distribution_number": (
                        2 if name_index == 1 and day_index >= 63 else 1
                    ),
                }
            )
    minute = np.broadcast_to(
        100.0 + np.arange(405, dtype=np.float64) * 0.01,
        (days, 2, 405),
    ).copy()
    minute_close = minute * 1.0001
    observed = np.ones_like(minute, dtype=bool)
    minute_close[65, 0, -1] = 123.45
    observed[65, 1, -1] = False
    panel = MinutePanel(
        dates=np.asarray(dates, dtype="datetime64[D]"),
        isins=names,
        open_brl=minute,
        high_brl=minute * 1.001,
        low_brl=minute * 0.999,
        close_brl=minute_close,
        volume=np.ones_like(minute),
        observed=observed,
    )
    actions = pl.DataFrame(
        {
            "isin": [names[1], names[0]],
            "ex_date": [dates[10], dates[63]],
            "action_type": ["dividend", "subscription_rights"],
            "split_factor": [1.0, 1.0],
            "cash_distribution_brl": [0.5, 0.0],
            "unresolved": [False, True],
        },
        schema_overrides={"ex_date": pl.Date},
    )
    successful_audit = pl.DataFrame(
        {
            "isin": list(names),
            "first_date": [dates[0]] * len(names),
            "last_date": [dates[-1]] * len(names),
            "status": ["downloaded"] * len(names),
            "action_rows": [1] * len(names),
        }
    )
    failed_audit = successful_audit.with_columns(
        pl.lit("failed").alias("status"),
        pl.lit(0).alias("action_rows"),
    )
    root = build_daily_store(
        pl.DataFrame(daily_rows),
        actions,
        tmp_path / "daily_store",
        minute_panel=panel,
        action_acquisition_audit=successful_audit,
        minimum_calendar_names=1,
        store_start=None,
    )
    provider_empty_root = build_daily_store(
        pl.DataFrame(daily_rows),
        actions.head(0),
        tmp_path / "daily_store_provider_empty",
        minute_panel=panel,
        action_acquisition_audit=failed_audit,
        minimum_calendar_names=1,
        store_start=None,
    )
    assert {
        path.name: path.read_bytes() for path in root.glob("*.npy")
    } == {
        path.name: path.read_bytes()
        for path in provider_empty_root.glob("*.npy")
    }
    raw = np.load(root / "target_to_close_raw_log_return.npy")
    valid = np.load(root / "target_to_close_valid.npy")
    expected = np.log(cotahist_close / minute[64, 0, 345])
    assert raw[64, 0] == expected
    assert valid[64].all()
    assert np.isnan(raw[65, 0])
    assert not valid[65].any()
    consistent = np.load(root / "m1_cotahist_close_consistent_mask.npy")
    assert not consistent[65].any()
    cash_event = np.load(root / "detected_cash_event_mask.npy")
    target_exclusion = np.load(root / "target_exclusion_event_mask.npy")
    ambiguous = np.load(root / "ambiguous_action_mask.npy")
    assert cash_event[63, 1]
    assert target_exclusion[63, 1]
    assert not ambiguous.any()
    slow_valid = np.load(root / "slow_valid.npy")
    assert slow_valid[63, 1, 3]
    assert slow_valid[64, 0, 0]
    assert slow_valid[64, 0, 25]
    target_raw_valid = np.load(root / "target_raw_valid.npy")
    assert target_raw_valid[61, 1, 0]
    assert not target_raw_valid[62, 1, 0]
    assert target_raw_valid[63, 1, 0]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "cash_reinvestment_unavailable_count_foundation" not in manifest["metadata"]
    assert manifest["metadata"]["future_total_return_variant"].startswith(
        "registered but not implemented"
    )
    assert tuple(
        manifest["metadata"]["v1_store_v2_zero_slow_fields"]
    ) == V1_STORE_V2_ZERO_SLOW_FIELDS
