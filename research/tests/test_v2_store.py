import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

import brazil_rv.v2.build_store as build_store_module
import brazil_rv.v2.store as store_module
from brazil_rv.v2.build_store import (
    MinutePanel,
    _external_feature_validity_by_survival_liquidity,
    _feature_validity_by_survival,
    _prior_adv20,
    _require_clean_implementation_commit,
    build_daily_store,
    load_minute_npz,
)
from brazil_rv.v2.corporate_actions import normalize_yfinance_actions
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.contract import FINETUNE_START, INTRADAY_DAILY_FEATURES
from brazil_rv.v2.data import (
    V1_STORE_V2_ZERO_DYNAMIC_CHANNELS,
    V1_STORE_V2_ZERO_SLOW_FIELDS,
    V2DailyDataset,
    collate_v2_daily,
    lazy_slow_window,
    read_scalar_feature_view,
)
from brazil_rv.v2.decision_clock import SessionDefinition
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.store import (
    V2Store,
    open_store_for_dates,
    open_store_for_samples,
    sha256_file,
    write_store as write_store_without_feature_schema,
)
from v2_store_fixtures import fixture_feature_schema, write_fixture_store as write_store


def _session_schedule(dates: list[date]) -> tuple[SessionDefinition, ...]:
    return tuple(
        SessionDefinition(
            trade_date=value,
            continuous_open=time(10, 0),
            decision_time=time(15, 45),
            continuous_close=time(16, 45),
            auction_close=time(17, 0),
            source="synthetic_test_schedule",
        )
        for value in dates
    )


def test_close_memmap_reaches_mapping_through_ndarray_view(tmp_path: Path) -> None:
    path = tmp_path / "view_backed.npy"
    mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(2,))
    view = np.asarray(mapped)
    assert not isinstance(view, np.memmap)
    store_module.close_memmap(view)
    assert mapped._mmap.closed
    path.unlink()


def test_minute_archive_requires_independent_activity_and_source_masks(
    tmp_path: Path,
) -> None:
    shape = (1, 1, 5)
    market = np.ones(shape, dtype=np.float64)
    observed = np.ones(shape, dtype=np.bool_)
    common = {
        "dates": np.asarray(["2024-01-02"], dtype="datetime64[D]"),
        "isins": np.asarray(["BRTESTACNOR1"]),
        "open": market,
        "high": market,
        "low": market,
        "close": market,
        "volume": market,
        "observed": observed,
    }
    missing_masks = tmp_path / "missing_masks.npz"
    np.savez(missing_masks, **common)
    with pytest.raises(ValueError, match="session_valid.*volume_valid|volume_valid"):
        load_minute_npz(missing_masks)

    complete = tmp_path / "complete.npz"
    np.savez(
        complete,
        **common,
        volume_valid=observed,
        session_valid=np.ones((1, 1), dtype=np.bool_),
    )
    panel = load_minute_npz(complete)
    assert panel.volume_valid.all()
    assert panel.session_valid.all()


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
    assert calls == [("git", "status", "--porcelain=v1", "--untracked-files=all")]

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
    observed = np.asarray([[True, True], [True, True], [True, True], [False, True]])
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


def test_linked_survival_identity_treats_predecessor_as_continuing() -> None:
    dates = np.asarray(
        ["2023-12-28", "2023-12-29", "2024-01-02"], dtype="datetime64[D]"
    )
    observed = np.asarray([[True, False], [True, False], [False, True]])
    valid = np.ones((3, 2, 1), dtype=np.bool_)
    table = _feature_validity_by_survival(
        dates,
        observed,
        observed,
        {"slow_return": (valid, observed)},
        ("BRROOTACNOR1", "BRROOTACNOR1"),
    )
    delisted = table.filter(pl.col("group") == "delisted_within_panel")
    assert delisted[0, "name_count"] == 0
    assert (
        table.filter(pl.col("group") == "survives_to_final_year")[0, "name_count"] == 2
    )


def test_prior_adv20_uses_only_sessions_before_the_decision() -> None:
    volume = np.arange(25, dtype=np.float64)[:, None] + 1.0
    observed = np.ones_like(volume, dtype=np.bool_)
    baseline = _prior_adv20(volume, observed)
    changed = volume.copy()
    changed[21:] = 1_000_000.0
    mutated = _prior_adv20(changed, observed)
    assert baseline[20, 0] == np.mean(volume[:20, 0])
    np.testing.assert_array_equal(baseline[:22], mutated[:22])
    unknown = observed.copy()
    unknown[5, 0] = False
    invalidated = _prior_adv20(volume, unknown)
    assert np.isnan(invalidated[20, 0])


def test_external_gate_uses_supported_name_clustered_one_sided_intervals() -> None:
    dates = np.datetime64("2023-01-01") + np.arange(800).astype("timedelta64[D]")
    observed = np.ones((800, 40), dtype=np.bool_)
    observed[400:, :20] = False
    active = observed.copy()
    prior_adv = np.broadcast_to(
        (1.0 + np.arange(800) % 4)[:, None],
        observed.shape,
    ).copy()
    present = observed.copy()
    valid = present[..., None].copy()
    table = _external_feature_validity_by_survival_liquidity(
        dates,
        active,
        observed,
        prior_adv,
        "sidecar_options",
        valid,
        present,
    )
    assert set(table.get_column("prior_adv20_quartile")) == {0, 1, 2, 3, 4}
    binding = table.filter(pl.col("prior_adv20_quartile") > 0)
    assert binding.get_column("stratum_is_binding").all()
    assert binding.get_column("stratified_gate_passed").all()
    assert binding.get_column("bootstrap_replications").unique().to_list() == [1000]
    assert binding.get_column("supported_continuation_name_count").min() == 20
    assert binding.get_column("family_present_name_days").min() == 2000
    assert table.schema["coverage_note"] == pl.String
    assert table.get_column("coverage_note").eq("").all()

    delisted_above = valid.copy()
    q2_survivors = (prior_adv == 2.0) & observed
    q2_survivors[:, :20] = False
    delisted_above[q2_survivors] = False
    one_sided = _external_feature_validity_by_survival_liquidity(
        dates,
        active,
        observed,
        prior_adv,
        "sidecar_options",
        delisted_above,
        present,
    )
    q2 = one_sided.filter(pl.col("prior_adv20_quartile") == 2)
    assert q2.get_column("survivor_minus_delisted_gap").unique().to_list() == [-1.0]
    assert q2.get_column("bootstrap_upper_95").max() == -1.0
    assert q2.get_column("stratified_gate_passed").all()

    survivor_above = valid.copy()
    q2_delisted = (prior_adv == 2.0) & observed
    q2_delisted[:, 20:] = False
    survivor_above[q2_delisted] = False
    with pytest.raises(ValueError, match="name-bootstrap lower bound"):
        _external_feature_validity_by_survival_liquidity(
            dates,
            active,
            observed,
            prior_adv,
            "sidecar_options",
            survivor_above,
            present,
        )


def test_external_gate_reports_but_does_not_gate_thin_strata() -> None:
    dates = np.datetime64("2023-01-01") + np.arange(800).astype("timedelta64[D]")
    observed = np.ones((800, 40), dtype=np.bool_)
    observed[400:, :19] = False
    active = observed.copy()
    prior_adv = np.broadcast_to(
        (1.0 + np.arange(800) % 4)[:, None], observed.shape
    ).copy()
    present = observed.copy()
    valid = present[..., None].copy()
    q2_delisted = (prior_adv == 2.0) & observed
    q2_delisted[:, 19:] = False
    valid[q2_delisted] = False

    table = _external_feature_validity_by_survival_liquidity(
        dates,
        active,
        observed,
        prior_adv,
        "sidecar_lending",
        valid,
        present,
    )
    q2 = table.filter(pl.col("prior_adv20_quartile") == 2)
    assert q2.get_column("survivor_minus_delisted_gap").unique().to_list() == [1.0]
    assert not q2.get_column("stratum_is_binding").any()
    assert q2.get_column("stratified_gate_passed").null_count() == 2
    assert table.schema["stratified_gate_passed"] == pl.Boolean
    assert table.schema["bootstrap_lower_95"] == pl.Float64
    assert q2.get_column("gate_decision").unique().to_list() == [
        "reported_not_gated_insufficient_support"
    ]
    assert q2.get_column("coverage_note").str.contains("5,212").all()


def _base_store(
    tmp_path,
    *,
    external_fast=None,
    stored_fast_present=None,
    extra_arrays=None,
    extra_tables=None,
):
    days, names = 25, 3
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(days)]
    slow = np.broadcast_to(
        np.arange(days, dtype=np.float32)[:, None, None], (days, names, 2)
    ).copy()
    arrays = {
        "slow_values": slow,
        "slow_valid": np.ones_like(slow, dtype=bool),
        "slow_age_sessions": np.zeros_like(slow, dtype=np.float32),
        **_sample_support_arrays(days, names),
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
            "v1_store_v2_zero_slow_fields": list(V1_STORE_V2_ZERO_SLOW_FIELDS),
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
    arrays.update(extra_arrays or {})
    tables.update(extra_tables or {})
    path = write_store(
        tmp_path / "store",
        dates=dates,
        isins=("BRTESTACNOR1", "BRANOTHRNOR1", "BRTESTACNPR0"),
        arrays=arrays,
        metadata=metadata,
        tables=tables,
    )
    return path


def _sample_support_arrays(
    days: int,
    names: int,
    *,
    timestep_valid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    timesteps = (
        np.ones((days, names), dtype=np.bool_)
        if timestep_valid is None
        else np.asarray(timestep_valid, dtype=np.bool_)
    )
    current_shape = (days, names, len(INTRADAY_DAILY_FEATURES))
    return {
        "slow_timestep_valid": timesteps,
        "intraday_values": np.zeros(current_shape, dtype=np.float32),
        "intraday_valid": np.zeros(current_shape, dtype=np.bool_),
        "intraday_age_sessions": np.full(current_shape, -1.0, dtype=np.float32),
    }


def _feature_age(valid: np.ndarray) -> np.ndarray:
    return np.where(valid, 0.0, -1.0).astype(np.float32)


def test_lazy_slow_window_preserves_source_age_and_left_padding() -> None:
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1)
    valid = np.ones_like(values, dtype=np.bool_)
    timesteps = np.ones((3, 1), dtype=np.bool_)
    ages = np.asarray([0.0, 1.0, 2.0], dtype=np.float32).reshape(3, 1, 1)

    window, window_valid, history, window_age = lazy_slow_window(
        values,
        valid,
        timesteps,
        ages,
        end_index=2,
        lookback=20,
    )

    assert window.shape == (1, 20, 1)
    assert window_valid[0, -3:, 0].all()
    assert history[0, -3:].all()
    assert not history[0, :-3].any()
    assert window_age[0, -3:, 0].tolist() == [0.0, 1.0, 2.0]
    assert np.all(window_age[0, :-3, 0] == -1.0)


def test_store_is_immutable_and_hash_verified(tmp_path) -> None:
    path = _base_store(tmp_path)
    store, _ = open_store_for_dates(path, list(range(25)), purpose="training")
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


def test_current_store_requires_complete_featurespec_identity(tmp_path: Path) -> None:
    dates = [date(2024, 1, 2)]
    arrays = {"active": np.ones((1, 1), dtype=np.bool_)}
    with pytest.raises(ValueError, match="metadata.feature_schema"):
        write_store_without_feature_schema(
            tmp_path / "missing_schema",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=arrays,
        )

    names = {
        "sidecar_zeta": ("zeta_value",),
        "slow": ("slow_value",),
        "sidecar_alpha": ("alpha_value",),
        "intraday": ("intraday_value",),
    }
    schema = fixture_feature_schema(names)
    stale_schema = json.loads(json.dumps(schema))
    stale_schema["specifications"][0]["formula"] = "changed without rehashing"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        write_store_without_feature_schema(
            tmp_path / "stale_schema_hash",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=arrays,
            feature_names=names,
            metadata={"feature_schema": stale_schema},
        )

    path = write_store_without_feature_schema(
        tmp_path / "canonical_schema_order",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays=arrays,
        feature_names=names,
        metadata={"feature_schema": schema},
    )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_schema_source"] == "metadata_feature_specifications"
    assert [
        record["family"]
        for record in manifest["metadata"]["feature_schema"]["specifications"]
    ] == ["slow", "intraday", "sidecar_alpha", "sidecar_zeta"]


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
            "slow_age_sessions": np.zeros((4, 1, 1), dtype=np.float32),
            "slow_timestep_valid": np.ones((4, 1), dtype=np.bool_),
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


def test_store_writer_accepts_aligned_date_common_state_audit_arrays(tmp_path) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    path = write_store(
        tmp_path / "common_state",
        dates=dates,
        isins=["BRTESTACNOR1"],
        arrays={
            "common_state_diagnostic_values": np.ones((2, 3), dtype=np.float32),
            "common_state_diagnostic_valid": np.ones((2, 3), dtype=np.bool_),
        },
    )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["arrays"]["common_state_diagnostic_values"]["shape"] == [2, 3]

    with pytest.raises(ValueError, match="misaligned"):
        write_store(
            tmp_path / "bad_common_state",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays={
                "common_state_diagnostic_values": np.ones((2, 3), dtype=np.float32),
                "common_state_diagnostic_valid": np.ones((2, 2), dtype=np.bool_),
            },
        )


def test_current_store_rejects_superseded_arrays_and_unmapped_native_fast(
    tmp_path: Path,
) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    with pytest.raises(ValueError, match="superseded synthetic-adjustment"):
        write_store(
            tmp_path / "legacy_array",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays={"adjusted_close": np.ones((2, 1), dtype=np.float32)},
        )
    native = {
        "fast_patch_values": np.zeros((2, 1, 1, 7), dtype=np.float32),
        "fast_patch_valid": np.zeros((2, 1, 1, 7), dtype=np.bool_),
        "fast_patch_mask": np.zeros((2, 1, 1), dtype=np.bool_),
    }
    with pytest.raises(ValueError, match="native_fast_security_mapping"):
        write_store(
            tmp_path / "unmapped_native",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=native,
        )
    with pytest.raises(ValueError, match="disagree with the store axis"):
        write_store(
            tmp_path / "wrong_native_mapping",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=native,
            tables={
                "native_fast_security_mapping": pl.DataFrame(
                    {
                        "fast_index": [0],
                        "store_name_index": [0],
                        "isin": ["BRWRONGACNOR1"],
                    }
                )
            },
        )


def test_open_store_rejects_the_superseded_daily_schema(tmp_path: Path) -> None:
    path = _base_store(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "BRAZIL_RV_V2_DAILY_STORE_V1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="superseded v2 daily store"):
        open_store_for_dates(path, [0], purpose="training")


def test_store_requires_complete_bounded_action_contract(tmp_path: Path) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    with pytest.raises(ValueError, match="one complete contract"):
        write_store(
            tmp_path / "partial_actions",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays={"action_shares_per_prior_share": np.ones((2, 1), dtype=np.float32)},
        )
    complete = {
        "action_shares_per_prior_share": np.ones((2, 1), dtype=np.float32),
        "action_cash_per_prior_share": np.zeros((2, 1), dtype=np.float32),
        "action_session_resolved": np.ones((2, 1), dtype=np.bool_),
        "action_has_action": np.zeros((2, 1), dtype=np.bool_),
        "action_successor_index": np.zeros((2, 1), dtype=np.int32),
        "action_payment_session": np.asarray([[-1], [3]], dtype=np.int32),
    }
    with pytest.raises(ValueError, match="payment sessions"):
        write_store(
            tmp_path / "late_payment",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=complete,
        )


def test_store_rejects_action_arrays_the_ledger_cannot_interpret(
    tmp_path: Path,
) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    shape = (2, 1)

    def action_arrays() -> dict[str, np.ndarray]:
        return {
            "action_shares_per_prior_share": np.ones(shape, dtype=np.float32),
            "action_cash_per_prior_share": np.zeros(shape, dtype=np.float32),
            "action_session_resolved": np.ones(shape, dtype=np.bool_),
            "action_has_action": np.zeros(shape, dtype=np.bool_),
            "action_successor_index": np.zeros(shape, dtype=np.int32),
            "action_payment_session": np.full(shape, -1, dtype=np.int32),
        }

    silent = action_arrays()
    silent["action_shares_per_prior_share"][0, 0] = 2.0
    with pytest.raises(ValueError, match="require has_action"):
        write_store(
            tmp_path / "silent_action",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=silent,
        )

    early_payment = action_arrays()
    early_payment["action_has_action"][1, 0] = True
    early_payment["action_cash_per_prior_share"][1, 0] = 1.0
    early_payment["action_payment_session"][1, 0] = 0
    with pytest.raises(ValueError, match="cannot precede"):
        write_store(
            tmp_path / "early_payment",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=early_payment,
        )

    unbound_payment = action_arrays()
    unbound_payment["action_payment_session"][0, 0] = 1
    with pytest.raises(ValueError, match="resolved cash-bearing"):
        write_store(
            tmp_path / "unbound_payment",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays=unbound_payment,
        )


def test_store_validates_reference_prices_and_audit_only_survival_group(
    tmp_path: Path,
) -> None:
    dates = [date(2024, 1, 2), date(2024, 1, 3)]
    with pytest.raises(ValueError, match="positive prices or NaN"):
        write_store(
            tmp_path / "bad_reference",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays={
                "prior_reference_close": np.asarray([[10.0], [0.0]], dtype=np.float32)
            },
        )
    with pytest.raises(ValueError, match="must be boolean"):
        write_store(
            tmp_path / "bad_survival_audit",
            dates=dates,
            isins=["BRTESTACNOR1"],
            arrays={
                "audit_eventual_survives_to_final_year": np.ones((2, 1), dtype=np.int8)
            },
        )


def test_store_writer_streams_a_contiguous_row_selection(
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
            "slow_age_sessions": np.zeros_like(slow, dtype=np.float32),
            "slow_timestep_valid": np.ones((6, 1), dtype=np.bool_),
            "active": np.ones((6, 1), dtype=np.bool_),
        },
        row_indices=np.arange(2, 6, dtype=np.int64),
    )
    # Store arrays are now opened at their final shape and filled in bounded
    # chunks; np.save is used only for the two small index arrays.
    assert slow_shares_memory == []
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


def test_dataset_reads_canonical_decision_row_and_external_sparse_fast_mapping(
    tmp_path,
) -> None:
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
    slow_age = np.zeros((25, 3, 2), dtype=np.float32)
    slow_age[20, 0, 0] = 3.0
    current_publication = np.broadcast_to(
        np.arange(25, dtype=np.float32)[:, None, None] + 100.0,
        (25, 3, 1),
    ).copy()
    path = _base_store(
        tmp_path,
        external_fast=fast,
        extra_arrays={
            "slow_age_sessions": slow_age,
            "sidecar_fundamentals_values": current_publication,
            "sidecar_fundamentals_valid": np.ones_like(
                current_publication, dtype=np.bool_
            ),
            "sidecar_fundamentals_age_sessions": np.zeros_like(
                current_publication, dtype=np.float32
            ),
        },
    )
    dataset = V2DailyDataset(
        path,
        [20],
        stage="finetune",
        lookback=20,
        enabled_sidecars=("fundamentals",),
    )
    sample = dataset[0]
    slow_view = read_scalar_feature_view(
        dataset.store,
        [20],
        ("sidecar_fundamentals", "slow"),
    )
    current_view = read_scalar_feature_view(dataset.store, [20], ("intraday",))
    assert slow_view.names == (
        "slow_fixture_0",
        "slow_fixture_1",
        "sidecar_fundamentals_fixture_0",
    )
    assert slow_view.date_indices.tolist() == [20]
    np.testing.assert_array_equal(slow_view.dates, dataset.store.dates[[20]])
    assert slow_view.isins == dataset.store.isins
    np.testing.assert_array_equal(slow_view.active[0], sample["active_mask"])
    np.testing.assert_array_equal(
        sample["slow_features"][:, -1], slow_view.values[0]
    )
    np.testing.assert_array_equal(
        sample["slow_feature_mask"][:, -1], slow_view.valid[0]
    )
    np.testing.assert_array_equal(
        sample["slow_feature_age_sessions"][:, -1], slow_view.age_sessions[0]
    )
    np.testing.assert_array_equal(sample["current_features"], current_view.values[0])
    np.testing.assert_array_equal(
        sample["current_feature_mask"], current_view.valid[0]
    )
    np.testing.assert_array_equal(
        sample["current_feature_age_sessions"], current_view.age_sessions[0]
    )
    assert sample["slow_features"][0, -1, 0] == 20.0
    assert sample["slow_features"][0, -1, 2] == 120.0
    assert sample["slow_feature_age_sessions"][0, -1, 0] == 3.0
    assert np.count_nonzero(sample["slow_feature_age_sessions"][0]) == 1
    assert sample["fast_present"].tolist() == [True, False, False]
    expected_slow = legacy_slow[0, 0].copy()
    expected_slow[list(V1_STORE_V2_ZERO_SLOW_FIELDS)] = 0.0
    assert sample["v1_equity_slow"].shape == (1, 32)
    np.testing.assert_array_equal(sample["v1_equity_slow"][0], expected_slow)
    assert sample["fast_name_index"].tolist() == [0]
    patches = sample["fast_patch_values"]
    assert patches.shape == (1, 69, 130)
    for channel in V1_STORE_V2_ZERO_DYNAMIC_CHANNELS:
        assert np.all(patches[0, :, channel::26] == 0.0)
    kept = set(range(26)) - set(V1_STORE_V2_ZERO_DYNAMIC_CHANNELS)
    assert all(np.all(patches[0, :, channel::26] == 1.0) for channel in kept)
    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_v2_daily,
            )
        )
    )
    assert tuple(batch["v1_equity_slow"].shape) == (1, 1, 32)
    np.testing.assert_array_equal(
        batch["v1_equity_slow"][0].numpy(), sample["v1_equity_slow"]
    )


def test_finetune_dataset_keeps_first_active_day_with_empty_slow_history(
    tmp_path: Path,
) -> None:
    days, names = 25, 2
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(days)]
    slow = np.zeros((days, names, 1), dtype=np.float32)
    slow_valid = np.ones_like(slow, dtype=np.bool_)
    slow_valid[:21, 1] = False
    timestep_valid = np.ones((days, names), dtype=np.bool_)
    timestep_valid[:21, 1] = False
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
            "slow_age_sessions": _feature_age(slow_valid),
            **_sample_support_arrays(days, names, timestep_valid=timestep_valid),
            "active": active,
            "target_primary": targets,
            "target_valid": target_valid,
        },
    )

    sample = V2DailyDataset(path, list(range(20, 25)), stage="finetune", lookback=20)[0]

    assert sample["active_mask"].tolist() == [True, True]
    assert not sample["slow_history_mask"][1].any()
    assert not sample["slow_features"][1].any()
    assert np.all(sample["slow_feature_age_sessions"][1] == -1.0)
    assert sample["target_mask"][1].tolist() == [True, True, True, False, False]


def test_dataset_boundary_cleans_every_masked_array_before_model_use(
    tmp_path: Path,
) -> None:
    fast = tmp_path / "v1"
    fast.mkdir()
    external_fast = np.ones((1, 2, 405, 26), dtype=np.float32)
    external_fast[0, 1] = np.nan
    np.save(fast / "equity_features.npy", external_fast, allow_pickle=False)
    external_slow = np.ones((1, 2, 32), dtype=np.float32)
    external_slow[0, 1] = np.nan
    np.save(fast / "equity_slow.npy", external_slow, allow_pickle=False)
    np.save(
        fast / "equity_data_ready.npy",
        np.asarray([[True, False]], dtype=np.bool_),
        allow_pickle=False,
    )

    days, names = 25, 3
    slow = np.ones((days, names, 2), dtype=np.float32)
    slow_valid = np.ones_like(slow, dtype=np.bool_)
    slow[:, 1] = np.nan
    slow_valid[:, 1] = False
    slow[19, 0, 0] = np.nan
    slow_valid[19, 0, 0] = False
    arrays: dict[str, np.ndarray] = {
        "slow_values": slow,
        "slow_valid": slow_valid,
        "slow_age_sessions": _feature_age(slow_valid),
        "slow_timestep_valid": np.ones((days, names), dtype=np.bool_),
        "target_primary": np.ones((days, names, 5), dtype=np.float32),
        "target_valid": np.ones((days, names, 5), dtype=np.bool_),
        "target_shareholder_midrank": np.ones((days, names, 5), dtype=np.float32),
        "target_shareholder_valid": np.ones((days, names, 5), dtype=np.bool_),
        "target_shareholder_simple_return": np.ones((days, names, 5), dtype=np.float32),
        "target_terminal_wealth": np.ones((days, names, 5), dtype=np.float32),
        "target_terminal_loss": np.zeros((days, names, 5), dtype=np.bool_),
        "target_price_midrank": np.ones((days, names, 5), dtype=np.float32),
        "target_price_valid": np.ones((days, names, 5), dtype=np.bool_),
        "target_price_simple_return": np.ones((days, names, 5), dtype=np.float32),
        "target_to_close": np.ones((days, names), dtype=np.float32),
        "target_to_close_valid": np.ones((days, names), dtype=np.bool_),
        "intraday_values": np.ones((days, names, 3), dtype=np.float32),
        "intraday_valid": np.ones((days, names, 3), dtype=np.bool_),
    }
    masked_pairs = (
        ("target_primary", "target_valid"),
        ("target_shareholder_midrank", "target_shareholder_valid"),
        ("target_shareholder_simple_return", "target_shareholder_valid"),
        ("target_terminal_wealth", "target_shareholder_valid"),
        ("target_price_midrank", "target_price_valid"),
        ("target_price_simple_return", "target_price_valid"),
        ("target_to_close", "target_to_close_valid"),
        ("intraday_values", "intraday_valid"),
    )
    for value_name, valid_name in masked_pairs:
        arrays[value_name][20, 0, ...] = np.nan
        arrays[valid_name][20, 0, ...] = False
    arrays["intraday_age_sessions"] = _feature_age(arrays["intraday_valid"])
    sidecars = ("options", "lending", "oddlot", "rebalance", "events", "fundamentals")
    for group in sidecars:
        values = np.ones((days, names, 1), dtype=np.float32)
        valid = np.ones_like(values, dtype=np.bool_)
        values[:, 1] = np.nan
        valid[:, 1] = False
        values[19, 0] = np.nan
        valid[19, 0] = False
        arrays[f"sidecar_{group}_values"] = values
        arrays[f"sidecar_{group}_valid"] = valid
        arrays[f"sidecar_{group}_age_sessions"] = _feature_age(valid)

    path = _base_store(tmp_path, external_fast=fast, extra_arrays=arrays)
    dataset = V2DailyDataset(
        path,
        list(range(20, 25)),
        stage="finetune",
        lookback=20,
        enabled_sidecars=sidecars,
    )
    sample = dataset[0]
    for value in sample.values():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            assert np.isfinite(value).all()
    assert sample["slow_history_mask"][1].all()
    assert not sample["slow_features"][1].any()
    assert not sample["fast_present"][2]
    assert sample["fast_name_index"].tolist() == [0]

    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_v2_daily,
            )
        )
    )
    model = DailyMultiHorizonModel(
        ModelConfig(
            slow_feature_count=int(sample["slow_features"].shape[-1]),
            current_feature_count=int(sample["current_features"].shape[-1]),
            slow_lookback=20,
            fast_encoder_mode="legacy_v1_contaminated",
            allow_contaminated_v1_initialization=True,
        )
    ).eval()
    with torch.no_grad():
        predictions = model(
            batch["slow_features"],
            batch["slow_feature_mask"],
            batch["slow_history_mask"],
            batch["active_mask"],
            current_features=batch["current_features"],
            current_feature_mask=batch["current_feature_mask"],
            slow_feature_age_sessions=batch["slow_feature_age_sessions"],
            current_feature_age_sessions=batch["current_feature_age_sessions"],
            fast_patch_mask=batch["fast_patch_mask"],
            fast_present=batch["fast_present"],
            fast_state_position=batch["fast_state_position"],
            v1_equity_slow=batch["v1_equity_slow"],
            fast_patch_values=batch["fast_patch_values"],
            fast_patch_valid=batch["fast_patch_valid"],
            fast_name_index=batch["fast_name_index"],
        )
    assert torch.isfinite(predictions[batch["active_mask"]]).all()


def test_dataset_boundary_rejects_nonfinite_available_value(tmp_path: Path) -> None:
    slow = np.ones((25, 3, 2), dtype=np.float32)
    slow[19, 0, 0] = np.nan
    path = _base_store(
        tmp_path,
        extra_arrays={
            "slow_values": slow,
            "slow_valid": np.ones_like(slow, dtype=np.bool_),
            "slow_age_sessions": np.zeros_like(slow, dtype=np.float32),
        },
    )
    dataset = V2DailyDataset(path, [20], stage="finetune", lookback=20)
    with pytest.raises(ValueError, match="non-finite values marked available"):
        dataset[0]


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
    assert sample["fast_name_index"].tolist() == [0]
    assert sample["fast_patch_mask"].all()
    patches = sample["fast_patch_values"]
    for channel in V1_STORE_V2_ZERO_DYNAMIC_CHANNELS:
        assert not patches[0, :, channel::26].any()
    retained_dynamic = set(range(26)) - set(V1_STORE_V2_ZERO_DYNAMIC_CHANNELS)
    assert all(
        np.all(patches[0, :, channel::26] == 1.0) for channel in retained_dynamic
    )
    retained = sorted(set(range(32)) - set(V1_STORE_V2_ZERO_SLOW_FIELDS))
    assert np.all(sample["v1_equity_slow"][0, retained] == 7.0)
    assert not sample["v1_equity_slow"][0, V1_STORE_V2_ZERO_SLOW_FIELDS].any()
    assert sample["v1_equity_slow"].shape == (1, 32)


def test_native_fast_arrays_do_not_open_legacy_v1_context(tmp_path) -> None:
    fast = tmp_path / "legacy_fast"
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
    path = _base_store(
        tmp_path,
        external_fast=fast,
        extra_arrays={
            "fast_patch_values": np.ones((25, 2, 69, 7), dtype=np.float32),
            "fast_patch_valid": np.ones((25, 2, 69, 7), dtype=np.bool_),
            "fast_patch_mask": np.ones((25, 2, 69), dtype=np.bool_),
            "fast_present": np.column_stack(
                (
                    np.ones(25, dtype=np.bool_),
                    np.zeros(25, dtype=np.bool_),
                    np.ones(25, dtype=np.bool_),
                )
            ),
            "fast_last_price_age_minutes": np.zeros((25, 2, 69), dtype=np.float32),
            "fast_last_price_age_valid": np.ones((25, 2, 69), dtype=np.bool_),
        },
        extra_tables={
            "native_fast_security_mapping": pl.DataFrame(
                {
                    "fast_index": [0, 1],
                    "store_name_index": [0, 2],
                    "isin": ["BRTESTACNOR1", "BRTESTACNPR0"],
                    "security_id": ["one", "two"],
                }
            )
        },
    )

    dataset = V2DailyDataset(path, [20], stage="finetune", lookback=20)
    sample = dataset[0]

    assert dataset.external_artifact_resolutions == ()
    assert sample["fast_present"].tolist() == [True, False, True]
    assert sample["fast_patch_values"].shape == (2, 69, 7)
    assert sample["fast_name_index"].tolist() == [0, 2]
    assert not sample["v1_equity_slow"].any()


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


def test_joint_dataset_uses_same_canonical_decision_row_in_each_window(
    tmp_path,
) -> None:
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
            "slow_age_sessions": np.zeros_like(slow, dtype=np.float32),
            **_sample_support_arrays(26, 1),
            "active": np.ones((26, 1), dtype=bool),
        },
    )
    early = V2DailyDataset(path, [5], stage="joint", lookback=20)[0]
    dataset = V2DailyDataset(path, [24, 25], stage="joint", lookback=20)
    pretrain, finetune = dataset[0], dataset[1]
    assert np.all(early["slow_feature_age_sessions"][0, :14] == -1.0)
    assert np.all(early["slow_feature_age_sessions"][0, 14:] == 0.0)
    assert pretrain["slow_features"][0, -1, 0] == 24.0
    assert finetune["slow_features"][0, -1, 0] == 25.0
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
        for value in np.arange(np.datetime64("2021-07-01"), np.datetime64("2021-08-17"))
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
    monkeypatch,
) -> None:
    dates = [
        value.astype(object)
        for value in np.arange(np.datetime64("2024-11-25"), np.datetime64("2024-12-31"))
        if np.is_busday(value)
    ]
    dates.append(date(2025, 1, 2))
    days = len(dates)
    target = np.full((days, 1, 5), 11.0, dtype=np.float32)
    shareholder_target = np.full((days, 1, 5), 22.0, dtype=np.float32)
    shareholder_return = np.full((days, 1, 5), 33.0, dtype=np.float32)
    price_target = np.full((days, 1, 5), 55.0, dtype=np.float32)
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
            "slow_age_sessions": np.zeros_like(slow, dtype=np.float32),
            **_sample_support_arrays(days, 1),
            "active": np.ones((days, 1), dtype=np.bool_),
            "target_primary": target,
            "target_valid": target_valid,
            "target_shareholder_midrank": shareholder_target,
            "target_shareholder_valid": target_valid,
            "target_shareholder_simple_return": shareholder_return,
            "target_price_midrank": price_target,
            "target_price_valid": target_valid,
            "target_to_close": to_close,
            "target_to_close_valid": to_close_valid,
        },
    )
    dataset = V2DailyDataset(
        path, [days - 3, days - 2], stage="evaluation", lookback=20
    )
    reads: list[str] = []
    original_read = dataset.store.read
    original_read_target = dataset.store.read_target

    def tracked_read(name, selector):
        reads.append(name)
        return original_read(name, selector)

    def tracked_read_target(name, selector, *, valid_mask):
        reads.append(name)
        return original_read_target(name, selector, valid_mask=valid_mask)

    monkeypatch.setattr(dataset.store, "read", tracked_read)
    monkeypatch.setattr(dataset.store, "read_target", tracked_read_target)
    first = dataset[0]
    second = dataset[1]
    assert reads.index("target_valid") < reads.index("target_primary")
    assert reads.index("target_shareholder_valid") < reads.index(
        "target_shareholder_midrank"
    )
    assert reads.index("target_price_valid") < reads.index("target_price_midrank")
    assert reads.index("target_to_close_valid") < reads.index("target_to_close")
    assert first["target_mask"].tolist() == [[True, False, False, False, False]]
    assert first["shareholder_target_mask"].tolist() == [
        [True, False, False, False, False]
    ]
    assert first["price_target_mask"].tolist() == [[True, False, False, False, False]]
    assert first["targets"].tolist() == [[11.0, 0.0, 0.0, 0.0, 0.0]]
    assert first["shareholder_targets"].tolist() == [[22.0, 0.0, 0.0, 0.0, 0.0]]
    assert first["shareholder_simple_returns"].tolist() == [[33.0, 0.0, 0.0, 0.0, 0.0]]
    assert first["price_targets"].tolist() == [[55.0, 0.0, 0.0, 0.0, 0.0]]
    assert first["to_close_mask"].tolist() == [False]
    assert first["to_close_target"].tolist() == [0.0]
    assert second["target_mask"].tolist() == [[False] * 5]
    assert second["targets"].tolist() == [[0.0] * 5]
    assert second["shareholder_targets"].tolist() == [[0.0] * 5]
    assert second["shareholder_simple_returns"].tolist() == [[0.0] * 5]
    assert second["price_targets"].tolist() == [[0.0] * 5]
    assert second["to_close_mask"].tolist() == [False]
    assert second["to_close_target"].tolist() == [0.0]
    direct_targets = dataset.store.read("target_primary", [days - 3, days - 2])
    direct_mask = dataset.store.read("target_valid", [days - 3, days - 2])
    assert direct_targets[:, 0].tolist() == [
        [11.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert direct_mask[:, 0].tolist() == [
        [True, False, False, False, False],
        [False, False, False, False, False],
    ]

    backing = dataset.store._arrays["target_primary"]
    payload_reads: list[object] = []

    class TargetReadGuard:
        shape = backing.shape
        dtype = backing.dtype

        def __getitem__(self, key):
            payload_reads.append(key)
            if not isinstance(key, tuple):
                raise AssertionError("target payload was read before endpoint masking")
            return backing[key]

    dataset.store._arrays["target_primary"] = TargetReadGuard()
    payload_reads.clear()
    guarded_sample = dataset[0]
    assert guarded_sample["targets"].tolist() == [[11.0, 0.0, 0.0, 0.0, 0.0]]
    assert len(payload_reads) == 1
    assert np.asarray(payload_reads[0][-1]).tolist() == [0]
    payload_reads.clear()
    assert not dataset[1]["targets"].any()
    assert payload_reads == []
    guarded = dataset.store.read("target_primary", [days - 3, days - 2])
    np.testing.assert_array_equal(guarded, direct_targets)
    assert len(payload_reads) == 1
    assert np.asarray(payload_reads[0][-1]).tolist() == [0]
    payload_reads.clear()
    local_mask = direct_mask.copy()
    local_mask[0, 0, 0] = False
    narrowed = dataset.store.read_target(
        "target_primary",
        [days - 3, days - 2],
        valid_mask=local_mask,
    )
    assert not narrowed.any()
    assert payload_reads == []
    exceeding = direct_mask.copy()
    exceeding[0, 0, 1] = True
    with pytest.raises(PermissionError, match="exceeds the store capability"):
        dataset.store.read_target(
            "target_primary",
            [days - 3, days - 2],
            valid_mask=exceeding,
        )
    gappy = V2DailyDataset(path, [days - 4, days - 2], stage="evaluation", lookback=20)
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
            "slow_age_sessions": np.zeros((3, 1, 1), dtype=np.float32),
            **_sample_support_arrays(3, 1),
            "active": np.ones((3, 1), dtype=bool),
        },
    )
    with pytest.raises(ValueError, match="finetune stage"):
        V2DailyDataset(path, [0], stage="finetune", lookback=20)
    with pytest.raises(ValueError, match="joint stage"):
        V2DailyDataset(path, [1], stage="joint", lookback=20)


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
                    "currency": "BRL",
                    "quote_factor": 1.0,
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
        volume_valid=observed.copy(),
        session_valid=np.ones(observed.shape[:2], dtype=np.bool_),
    )
    actions = pl.DataFrame(
        {
            "isin": [names[1], names[0]],
            "ex_date": [dates[10], dates[63]],
            "action_type": ["dividend", "subscription_rights"],
            "split_factor": [1.0, 1.0],
            "cash_distribution_brl": [0.5, 0.0],
            "unresolved": [False, True],
            "fetched_at": [
                datetime(2023, 4, 1, tzinfo=timezone.utc),
                datetime(2023, 4, 1, tzinfo=timezone.utc),
            ],
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
            "economic_terms_complete": [True] * len(names),
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
        session_schedule=_session_schedule(dates),
        minimum_rank_names=1,
        store_start=None,
    )
    build_metadata = json.loads((root / "manifest.json").read_text())["metadata"]
    assert 0 < build_metadata["build_peak_rss_bytes"] < 8 * 1024**3
    np.testing.assert_array_equal(
        np.load(root / "trade_observed.npy"),
        np.load(root / "observed.npy"),
    )
    assert np.load(root / "activity_valid.npy").all()
    assert np.load(root / "source_session_complete.npy").all()
    provider_empty_root = build_daily_store(
        pl.DataFrame(daily_rows),
        actions.head(0),
        tmp_path / "daily_store_provider_empty",
        minute_panel=panel,
        action_acquisition_audit=failed_audit,
        session_schedule=_session_schedule(dates),
        minimum_rank_names=1,
        store_start=None,
    )
    assert not np.load(provider_empty_root / "target_shareholder_valid.npy").any()
    raw = np.load(root / "target_to_close_raw_log_return.npy")
    valid = np.load(root / "target_to_close_valid.npy")
    expected = np.log(cotahist_close / minute[64, 0, 345])
    assert raw.dtype == np.float32
    np.testing.assert_allclose(raw[64, 0], expected, rtol=1e-5, atol=1e-8)
    assert valid[64].all()
    assert np.isnan(raw[65, 0])
    assert not valid[65].any()
    consistent = np.load(root / "m1_cotahist_close_consistent_mask.npy")
    assert not consistent[65].any()
    cash_event = np.load(root / "detected_cash_event_mask.npy")
    anomaly = np.load(root / "price_jump_anomaly_mask.npy")
    ambiguous = np.load(root / "ambiguous_action_mask.npy")
    assert cash_event[63, 1]
    assert not anomaly.any()
    assert not ambiguous.any()
    slow_valid = np.load(root / "slow_valid.npy")
    assert not slow_valid[63, 1, 3]
    # The provider row was acquired after the historical event, so the
    # decision-time 63-session feature stays invalid while its window crosses
    # that then-unknown action.  Retrospective targets still use the term.
    assert not slow_valid[64, 1, 3]
    assert not slow_valid[64, 0, 0]
    assert slow_valid[64, 1, 0]
    assert slow_valid[64, 0, 25]
    intraday_valid = np.load(root / "intraday_valid.npy")
    intraday_age = np.load(root / "intraday_age_sessions.npy")
    assert intraday_valid[64, 0, 1]
    assert intraday_age[64, 0, 1] == 0.0
    # The lagged price-path intraday summaries cross the unresolved action at
    # row 63, so they remain invalid at decision row 64.  Independent activity
    # summaries above remain available.
    for feature_index in (8, 9, 10):
        assert not intraday_valid[64, 0, feature_index]
        assert intraday_age[64, 0, feature_index] == 2.0
    shareholder_valid = np.load(root / "target_shareholder_valid.npy")
    price_valid = np.load(root / "target_price_valid.npy")
    assert not shareholder_valid[62, 0, 0]
    assert shareholder_valid[63, 0, 0]
    assert not price_valid[62, 0, 0]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert "cash_reinvestment_unavailable_count_foundation" not in manifest["metadata"]
    assert manifest["metadata"]["return_definition"]["synthetic_cash_distribution"] == (
        "disabled"
    )
    assert manifest["metadata"]["native_fast"] == {
        "enabled": True,
        "security_count": 2,
        "channels": [
            "adjacent_endpoint_log_return_over_s5",
            "block_log_high_low_over_s5",
            "signed_close_location",
            "relative_same_clock_volume",
            "observed_fraction",
            "elapsed_fraction",
            "last_price_age_fraction",
        ],
        "time_axis": (
            "date-specific continuous-open to 15:45 decision prefix; "
            "five-minute completed blocks"
        ),
        "legacy_v1_artifact_required": False,
    }
    assert manifest["metadata"]["cotahist_action_detection_role"] == "diagnostic_only"
    assert "cotahist_action_counts_by_year" in manifest["tables"]
    assert "corporate_actions_verified_terms" in manifest["tables"]


def test_store_routes_a_verified_isin_conversion_without_restart_or_double_count(
    tmp_path: Path,
) -> None:
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(75)]
    predecessor = "BRTESTACNOR1"
    successor = "BRTESTACNPR0"
    boundary = 65
    split_day = 30
    price = np.empty(len(dates), dtype=np.float64)
    price[0] = 100.0
    for day_index in range(1, len(dates)):
        price[day_index] = price[day_index - 1] * 1.001
        if day_index == split_day:
            price[day_index] /= 2.0
    price[boundary] = price[boundary - 1] - 1.0
    for day_index in range(boundary + 1, len(dates)):
        price[day_index] = price[day_index - 1] * 1.001
    daily = pl.DataFrame(
        [
            {
                "trade_date": day,
                "isin": predecessor if day_index < boundary else successor,
                "ticker": "TEST3",
                "security_spec_base": "ON",
                "bdi_code": "02",
                "market_type": 10,
                "open_brl": float(price[day_index]),
                "high_brl": float(price[day_index] * 1.01),
                "low_brl": float(price[day_index] * 0.99),
                "close_brl": float(price[day_index]),
                "volume_brl": 3_000_000.0,
                "trades": 100.0,
                "quantity": 100_000.0,
                "distribution_number": 2 if day_index >= split_day else 1,
                "currency": "BRL",
                "quote_factor": 1.0,
            }
            for day_index, day in enumerate(dates)
        ]
    )
    actions = normalize_yfinance_actions(
        pl.DataFrame(
            {
                "Date": [dates[split_day]],
                "Dividends": [0.0],
                "Stock Splits": [2.0],
            },
            schema_overrides={"Date": pl.Date},
        ),
        isin=predecessor,
        ticker="TEST3",
        fetched_at=datetime.combine(
            dates[split_day - 1], time(12), tzinfo=timezone.utc
        ),
    )
    acquisition = pl.DataFrame(
        {
            "isin": [predecessor, successor],
            "first_date": [dates[0], dates[boundary]],
            "last_date": [dates[boundary - 1], dates[-1]],
            "status": ["downloaded", "downloaded"],
            "action_rows": [1, 0],
            "economic_terms_complete": [True, True],
        }
    )
    allowlist = tmp_path / "isin_links_allowlist.csv"
    allowlist.write_text(
        "ticker,predecessor_isin,successor_isin,effective_date,first_known_at,"
        "shares_received_per_prior_share,cash_entitlement_per_prior_share,"
        "currency,source,evidence_sha256\n"
        f"TEST3,{predecessor},{successor},{dates[boundary].isoformat()},"
        f"{dates[boundary - 1].isoformat()}T12:00:00Z,1.0,1.0,BRL,"
        f"issuer_notice,{'a' * 64}\n",
        encoding="utf-8",
    )
    root = build_daily_store(
        daily,
        actions,
        tmp_path / "linked_store",
        action_acquisition_audit=acquisition,
        session_schedule=_session_schedule(dates),
        isin_link_allowlist=allowlist,
        minimum_rank_names=1,
        store_start=None,
    )
    isins = (
        pl.read_parquet(root / "security_master.parquet")
        .get_column("isin")
        .unique()
        .sort()
        .to_list()
    )
    predecessor_index = isins.index(predecessor)
    successor_index = isins.index(successor)
    active = np.load(root / "active.npy", allow_pickle=False)
    assert active[boundary - 1, predecessor_index]
    assert not active[boundary - 1, successor_index]
    assert not active[boundary, predecessor_index]
    assert active[boundary, successor_index]
    assert active[boundary].sum() == 1
    prior_reference = np.load(root / "prior_reference_close.npy", allow_pickle=False)
    np.testing.assert_allclose(
        prior_reference[boundary, successor_index], price[boundary - 1]
    )
    actions_successor = np.load(root / "action_successor_index.npy", allow_pickle=False)
    assert actions_successor[boundary, predecessor_index] == successor_index
    shareholder_return = np.load(
        root / "target_shareholder_simple_return.npy", allow_pickle=False
    )
    price_return = np.load(root / "target_price_simple_return.npy", allow_pickle=False)
    shareholder_valid = np.load(
        root / "target_shareholder_valid.npy", allow_pickle=False
    )
    price_valid = np.load(root / "target_price_valid.npy", allow_pickle=False)
    assert shareholder_valid[boundary - 1, predecessor_index, 0]
    assert price_valid[boundary - 1, predecessor_index, 0]
    np.testing.assert_allclose(
        shareholder_return[boundary - 1, predecessor_index, 0], 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        price_return[boundary - 1, predecessor_index, 0],
        -1.0 / price[boundary - 1],
        rtol=1e-5,
    )
    slow_timestep_valid = np.load(root / "slow_timestep_valid.npy", allow_pickle=False)
    assert slow_timestep_valid[boundary, successor_index]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["isin_succession_link_count"] == 1
    assert "corporate_action_alignment_roles" in manifest["tables"]


def test_raw_to_feature_store_build_is_causal_through_cutoff(tmp_path) -> None:
    days = 75
    cutoff = 65
    dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(days)]
    names = ("BRTESTACNOR1", "BRTESTACNPR0")
    close = np.empty((days, 2), dtype=np.float64)
    quantity = np.full((days, 2), 100_000.0, dtype=np.float64)
    distribution = np.ones((days, 2), dtype=np.float64)
    for day_index in range(days):
        close[day_index] = (100.0 + 0.1 * day_index, 80.0 + 0.08 * day_index)
    close[30:, 0] *= 0.5
    quantity[30:, 0] *= 2.0
    distribution[30:, 0] = 2.0
    close[70:, 1] *= 10.0
    quantity[70:, 1] *= 0.1
    distribution[70:, 1] = 2.0

    def daily_frame(
        price: np.ndarray,
        qty: np.ndarray,
        dismes: np.ndarray,
        volume_scale: float = 1.0,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    "trade_date": day,
                    "isin": isin,
                    "ticker": f"TEST{name_index + 3}",
                    "security_spec_base": "ON",
                    "bdi_code": "02",
                    "market_type": 10,
                    "open_brl": float(price[day_index, name_index] * 0.999),
                    "high_brl": float(price[day_index, name_index] * 1.01),
                    "low_brl": float(price[day_index, name_index] * 0.99),
                    "close_brl": float(price[day_index, name_index]),
                    "volume_brl": float(
                        (volume_scale if day_index > cutoff else 1.0)
                        * (3_000_000.0 + 1_000.0 * day_index)
                    ),
                    "trades": float(
                        (volume_scale if day_index > cutoff else 1.0)
                        * (100 + day_index)
                    ),
                    "quantity": float(qty[day_index, name_index]),
                    "distribution_number": float(dismes[day_index, name_index]),
                    "currency": "BRL",
                    "quote_factor": 1.0,
                }
                for day_index, day in enumerate(dates)
                for name_index, isin in enumerate(names)
            ]
        )

    minute_fraction = np.linspace(-0.001, 0.0, 405, dtype=np.float64)

    def minute_panel(price: np.ndarray, *, future_scale: float = 1.0) -> MinutePanel:
        minute_close = price[:, :, None] * (1.0 + minute_fraction)
        minute_volume = np.full_like(minute_close, 1_000.0)
        if future_scale != 1.0:
            minute_close[cutoff + 1 :] *= future_scale
            minute_volume[cutoff + 1 :] *= 7.0
        return MinutePanel(
            dates=np.asarray(dates, dtype="datetime64[D]"),
            isins=names,
            open_brl=minute_close * 0.9999,
            high_brl=minute_close * 1.0002,
            low_brl=minute_close * 0.9998,
            close_brl=minute_close,
            volume=minute_volume,
            observed=np.ones_like(minute_close, dtype=np.bool_),
            volume_valid=np.ones_like(minute_close, dtype=np.bool_),
            session_valid=np.ones(minute_close.shape[:2], dtype=np.bool_),
        )

    actions = normalize_yfinance_actions(
        pl.DataFrame(
            schema={
                "Date": pl.Date,
                "Dividends": pl.Float64,
                "Stock Splits": pl.Float64,
            }
        ),
        isin=names[0],
        ticker="TEST3",
        fetched_at=datetime(2023, 4, 1, tzinfo=timezone.utc),
    )
    sidecar_rows = [
        {
            "available_date": day,
            "isin": isin,
            "total_liabilities_brl": float(10.0 + name_index + 0.01 * day_index),
            "total_assets_brl": 100.0,
        }
        for day_index, day in enumerate(dates)
        for name_index, isin in enumerate(names)
    ]
    sidecar_a = tmp_path / "fundamentals_a.parquet"
    sidecar_b = tmp_path / "fundamentals_b.parquet"
    pl.DataFrame(sidecar_rows).write_parquet(sidecar_a)
    pl.DataFrame(sidecar_rows).with_columns(
        pl.when(pl.col("available_date") > dates[cutoff])
        .then(pl.col("total_liabilities_brl") + 100.0)
        .otherwise(pl.col("total_liabilities_brl"))
        .alias("total_liabilities_brl")
    ).write_parquet(sidecar_b)

    mutated_close = close.copy()
    mutated_quantity = quantity.copy()
    mutated_distribution = distribution.copy()
    mutated_close[cutoff + 1 :] *= 3.0
    mutated_quantity[cutoff + 1 :] *= 7.0
    mutated_distribution[cutoff + 1 :] += 5.0
    acquisition_audit = pl.DataFrame(
        {
            "isin": list(names),
            "first_date": [dates[0]] * len(names),
            "last_date": [dates[-1]] * len(names),
            "status": ["downloaded"] * len(names),
            "action_rows": [0] * len(names),
            "economic_terms_complete": [True] * len(names),
        }
    )
    first = build_daily_store(
        daily_frame(close, quantity, distribution),
        actions,
        tmp_path / "causal_a",
        minute_panel=minute_panel(close),
        sidecar_arguments=(f"fundamentals={sidecar_a}",),
        action_acquisition_audit=acquisition_audit,
        session_schedule=_session_schedule(dates),
        minimum_rank_names=1,
        store_start=None,
    )
    second = build_daily_store(
        daily_frame(
            mutated_close,
            mutated_quantity,
            mutated_distribution,
            volume_scale=7.0,
        ),
        actions,
        tmp_path / "causal_b",
        minute_panel=minute_panel(close, future_scale=3.0),
        sidecar_arguments=(f"fundamentals={sidecar_b}",),
        action_acquisition_audit=acquisition_audit,
        session_schedule=_session_schedule(dates),
        minimum_rank_names=1,
        store_start=None,
    )

    exact_names = {
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "shareholder_wealth_open",
        "shareholder_wealth_high",
        "shareholder_wealth_low",
        "shareholder_wealth_close",
        "shareholder_wealth_valid",
        "detected_split_mask",
        "active",
        "fast_present",
        "intraday_boundary_lagged_mask",
        "intraday_boundary_sameday_mask",
    }
    exact_names.update(
        path.stem
        for path in first.glob("*.npy")
        if path.stem.startswith(("slow_", "intraday_", "sidecar_"))
    )
    for name in sorted(exact_names):
        left = np.load(first / f"{name}.npy", mmap_mode="r")
        right = np.load(second / f"{name}.npy", mmap_mode="r")
        np.testing.assert_array_equal(
            left[: cutoff + 1], right[: cutoff + 1], err_msg=name
        )

    detected = np.load(first / "detected_split_mask.npy")
    lagged_boundary = np.load(first / "intraday_boundary_lagged_mask.npy")
    np.testing.assert_array_equal(lagged_boundary, detected)
    assert detected[30, 0]
    assert detected[70, 1]
