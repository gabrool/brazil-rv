import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from torch.utils.data import DataLoader

import brazil_rv.v2.build_store as build_store_module
from brazil_rv.v2.build_store import (
    EXPECTED_V1_DATES,
    MinutePanel,
    V1_STORE_START,
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
    target = np.zeros((days, 1, 5), dtype=np.float32)
    target_valid = np.ones_like(target, dtype=np.bool_)
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
        },
    )
    dataset = V2DailyDataset(
        path, [days - 3, days - 2], stage="evaluation", lookback=20
    )
    assert dataset[0]["target_mask"].tolist() == [[True, False, False, False, False]]
    assert dataset[1]["target_mask"].tolist() == [[False] * 5]
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


def test_store_to_close_uses_exact_m1_final_close_not_cotahist(tmp_path) -> None:
    days = 70
    names = ("BRTESTACNOR1", "BRTESTACNPR0")
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]
    daily_rows = []
    for day in dates:
        for name_index, isin in enumerate(names):
            if day == dates[10] and name_index == 1:
                continue
            daily_rows.append(
                {
                    "trade_date": day,
                    "isin": isin,
                    "ticker": f"TEST{name_index + 3}",
                    "security_spec_base": "ON",
                    "bdi_code": "02",
                    "market_type": 10,
                    "open_brl": 200.0,
                    "high_brl": 201.0,
                    "low_brl": 199.0,
                    "close_brl": 200.0,
                    "volume_brl": 3_000_000.0,
                    "trades": 100.0,
                    "quantity": 100_000.0,
                    "distribution_number": 1,
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
    root = build_daily_store(
        pl.DataFrame(daily_rows),
        actions,
        tmp_path / "daily_store",
        minute_panel=panel,
        minimum_calendar_names=1,
        store_start=None,
    )
    raw = np.load(root / "target_to_close_raw_log_return.npy")
    valid = np.load(root / "target_to_close_valid.npy")
    expected = np.log(123.45 / minute[65, 0, 345])
    assert raw[65, 0] == expected
    assert raw[65, 0] != pytest.approx(np.log(200.0 / minute[65, 0, 345]))
    assert valid[65, 0]
    assert not valid[65, 1]
    unavailable = np.load(root / "cash_reinvestment_unavailable_mask.npy")
    unresolved = np.load(root / "unresolved_action.npy")
    assert unavailable[10, 1]
    assert unresolved[10, 1]
    assert unresolved[63, 0]
    slow_valid = np.load(root / "slow_valid.npy")
    assert not slow_valid[63, 1, 3]
    assert not slow_valid[63, 0, 0]
    assert slow_valid[64, 0, 0]
    target_raw_valid = np.load(root / "target_raw_valid.npy")
    assert not target_raw_valid[5, 1, 4]
    assert not target_raw_valid[62, 0, 0]
    assert target_raw_valid[63, 0, 0]
    target_valid = np.load(root / "target_valid.npy")
    assert not target_valid[62, 0, 0]
    assert target_valid[63, 0, 0]
    review = pl.read_parquet(
        root / "corporate_action_cash_reinvestment_review.parquet"
    )
    assert review.select("observed", "unresolved").to_dicts() == [
        {"observed": False, "unresolved": True}
    ]
    assert review.select("trade_date", "isin", "cash_distribution_brl").to_dicts() == [
        {
            "trade_date": dates[10],
            "isin": names[1],
            "cash_distribution_brl": 0.5,
        }
    ]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["metadata"]["cash_reinvestment_unavailable_count_foundation"]
        == 1
    )
    assert (
        manifest["metadata"]["cash_reinvestment_unavailable_count_store_rows"]
        == 1
    )
    assert tuple(
        manifest["metadata"]["v1_store_v2_zero_slow_fields"]
    ) == V1_STORE_V2_ZERO_SLOW_FIELDS
