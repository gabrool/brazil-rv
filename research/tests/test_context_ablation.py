from __future__ import annotations

import copy
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.modeling import process_lock, stage1_context_ablation, train
from brazil_rv.modeling.analyze_context_ablation import (
    analyze_sweep,
    paired_moving_block_bootstrap,
)
from brazil_rv.modeling.context_ablation import (
    CONTEXT_ABLATIONS,
    CONTEXT_ABLATION_KEYS,
    GROUP_CONTEXT_ABLATIONS,
    INDIVIDUAL_CONTEXT_ABLATIONS,
    STAGE1_CONTEXT_ABLATION_ORDER,
    get_context_ablation,
    resolve_context_ablation,
    resolve_context_ablation_for_store,
)
from brazil_rv.modeling.contract import (
    DYNAMIC_CHANNEL_COUNT,
    EQUITY_COUNT,
    FEATURE_CONTRACT_VERSION,
    GLOBAL_CONTEXT_COUNT,
    GLOBAL_CONTEXT_SYMBOLS,
    HORIZONS,
    INSTRUMENT_COUNT,
    LOCAL_CONTEXT_COUNT,
    LOCAL_CONTEXT_SYMBOLS,
    SLOW_FEATURE_COUNT,
    TABULAR_FEATURE_COUNT,
    TABULAR_OFFSETS,
    TCNSettings,
    VALIDATION_END,
    VALIDATION_START,
    architecture_for_model,
)
from brazil_rv.modeling.data import BatchRequest, VectorizedFeatureDataset
from brazil_rv.modeling.process_lock import (
    active_lock_owner,
    exclusive_process_lock,
)
from brazil_rv.modeling.stage1_context_ablation import (
    _configuration,
    _production_run_directories,
    build_stage1_command,
    stage1_jobs,
    validate_completed_run,
)
from brazil_rv.preprocessing.contract import SLOW_CHANNELS


EXPECTED_KEYS = (
    "none",
    "drop_win",
    "drop_wdo",
    "drop_di1f27",
    "drop_di1f28",
    "drop_di1f29",
    "drop_di1f31",
    "drop_di1n",
    "drop_es",
    "drop_nq",
    "drop_zt",
    "drop_zn",
    "drop_cl",
    "drop_hg",
    "drop_6e",
    "drop_6m",
    "drop_fixed_di",
    "drop_all_di",
    "drop_us_equities",
    "drop_us_rates",
    "drop_commodities",
    "drop_global_fx",
    "drop_all_local",
    "drop_all_global",
    "drop_all_context",
    "drop_global_non_rates",
    "drop_win_and_global_non_rates",
    "drop_win_and_global_non_rates_except_es",
    "drop_win_and_global_non_rates_except_nq",
    "drop_win_and_global_non_rates_except_cl",
    "drop_win_and_global_non_rates_except_hg",
    "drop_win_and_global_non_rates_except_6e",
    "drop_win_and_global_non_rates_except_6m",
)


def _resolved(key: str):
    return resolve_context_ablation(
        get_context_ablation(key),
        local_symbols=LOCAL_CONTEXT_SYMBOLS,
        global_symbols=GLOBAL_CONTEXT_SYMBOLS,
        equity_slow_features=SLOW_CHANNELS,
    )


def _synthetic_store(path: Path) -> tuple[Path, pl.DataFrame]:
    generator = np.random.default_rng(2905)
    date_count = 2
    arrays = {
        "equity_features.npy": generator.normal(
            size=(date_count, EQUITY_COUNT, 405, DYNAMIC_CHANNEL_COUNT)
        ).astype(np.float32),
        "equity_slow.npy": generator.normal(
            size=(date_count, EQUITY_COUNT, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "equity_membership.npy": np.zeros((date_count, EQUITY_COUNT), dtype=bool),
        "equity_data_ready.npy": np.zeros((date_count, EQUITY_COUNT), dtype=bool),
        "context_features.npy": generator.normal(
            size=(date_count, LOCAL_CONTEXT_COUNT, 465, DYNAMIC_CHANNEL_COUNT)
        ).astype(np.float32),
        "context_slow.npy": generator.normal(
            size=(date_count, LOCAL_CONTEXT_COUNT, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "context_data_ready.npy": np.ones(
            (date_count, LOCAL_CONTEXT_COUNT), dtype=bool
        ),
        "global_features.npy": generator.normal(
            size=(date_count, GLOBAL_CONTEXT_COUNT, 615, DYNAMIC_CHANNEL_COUNT)
        ).astype(np.float32),
        "global_slow.npy": generator.normal(
            size=(date_count, GLOBAL_CONTEXT_COUNT, 55, SLOW_FEATURE_COUNT)
        ).astype(np.float32),
        "global_data_ready.npy": np.ones(
            (date_count, GLOBAL_CONTEXT_COUNT, 55), dtype=bool
        ),
        "targets.npy": generator.normal(
            size=(date_count, EQUITY_COUNT, 55, len(HORIZONS))
        ).astype(np.float32),
        "label_mask.npy": np.zeros(
            (date_count, EQUITY_COUNT, 55, len(HORIZONS)), dtype=bool
        ),
        "raw_returns.npy": generator.normal(
            size=(date_count, EQUITY_COUNT, 55, len(HORIZONS))
        ).astype(np.float32),
    }
    arrays["equity_membership.npy"][:, :3] = True
    arrays["equity_data_ready.npy"][:, :3] = True
    arrays["label_mask.npy"][:, :3] = True
    arrays["equity_features.npy"][..., 5] = 1.0
    arrays["context_features.npy"][..., 5] = 1.0
    arrays["global_features.npy"][..., 5] = 1.0
    for filename, array in arrays.items():
        np.save(path / filename, array)
    rows = pl.DataFrame(
        {
            "sample_id": [10, 11],
            "date_idx": [0, 1],
            "decision_idx": [0, 54],
            "equity_cutoff_index": [15, 285],
            "context_cutoff_index": [75, 345],
        }
    )
    return path, rows


def _patch_batch(store: Path, rows: pl.DataFrame, key: str, global_: str = "enabled"):
    return VectorizedFeatureDataset(
        store,
        rows,
        "context_pooled",
        global_,
        context_ablation=_resolved(key),
    )[BatchRequest((0, 1), 2)]


def test_registry_exact_keys_groups_and_stable_metadata() -> None:
    assert CONTEXT_ABLATION_KEYS == EXPECTED_KEYS
    assert len(CONTEXT_ABLATIONS) == 33
    assert len(INDIVIDUAL_CONTEXT_ABLATIONS) == 15
    assert len(GROUP_CONTEXT_ABLATIONS) == 9
    assert STAGE1_CONTEXT_ABLATION_ORDER == (
        "none",
        *GROUP_CONTEXT_ABLATIONS,
        *INDIVIDUAL_CONTEXT_ABLATIONS,
    )
    assert len(set(STAGE1_CONTEXT_ABLATION_ORDER)) == 25
    assert "drop_global_non_rates" not in STAGE1_CONTEXT_ABLATION_ORDER
    for specification in CONTEXT_ABLATIONS.values():
        metadata = specification.metadata()
        assert metadata["key"] == specification.key
        assert len(metadata["specification_sha256"]) == 64
        assert (
            json.loads(metadata["serialized_specification"])["key"] == specification.key
        )


def test_group_definitions_are_exact_unions() -> None:
    fixed = ("DI1F27", "DI1F28", "DI1F29", "DI1F31")
    assert CONTEXT_ABLATIONS["drop_fixed_di"].removed_local_symbols == fixed
    assert CONTEXT_ABLATIONS["drop_all_di"].removed_local_symbols == (*fixed, "DI1$N")
    assert CONTEXT_ABLATIONS["drop_us_equities"].removed_global_symbols == (
        "ES.v.0",
        "NQ.v.0",
    )
    assert CONTEXT_ABLATIONS["drop_us_rates"].removed_global_symbols == (
        "ZT.v.0",
        "ZN.v.0",
    )
    assert CONTEXT_ABLATIONS["drop_commodities"].removed_global_symbols == (
        "CL.v.0",
        "HG.v.0",
    )
    assert CONTEXT_ABLATIONS["drop_global_fx"].removed_global_symbols == (
        "6E.v.0",
        "6M.v.0",
    )
    assert (
        CONTEXT_ABLATIONS["drop_all_local"].removed_local_symbols
        == LOCAL_CONTEXT_SYMBOLS
    )
    assert (
        CONTEXT_ABLATIONS["drop_all_global"].removed_global_symbols
        == GLOBAL_CONTEXT_SYMBOLS
    )
    assert (
        CONTEXT_ABLATIONS["drop_all_context"].removed_local_symbols
        == LOCAL_CONTEXT_SYMBOLS
    )
    assert (
        CONTEXT_ABLATIONS["drop_all_context"].removed_global_symbols
        == GLOBAL_CONTEXT_SYMBOLS
    )
    non_rates = CONTEXT_ABLATIONS["drop_global_non_rates"]
    assert non_rates.removed_local_symbols == ()
    assert non_rates.removed_global_symbols == (
        "ES.v.0",
        "NQ.v.0",
        "CL.v.0",
        "HG.v.0",
        "6E.v.0",
        "6M.v.0",
    )
    assert set(GLOBAL_CONTEXT_SYMBOLS) - set(non_rates.removed_global_symbols) == {
        "ZT.v.0",
        "ZN.v.0",
    }
    assert non_rates.neutralized_equity_slow_features == ()


def test_store_resolution_validates_axes_and_feature_names(tmp_path: Path) -> None:
    pl.DataFrame(
        {"context_slot": range(LOCAL_CONTEXT_COUNT), "symbol": LOCAL_CONTEXT_SYMBOLS}
    ).write_parquet(tmp_path / "context_index.parquet")
    pl.DataFrame(
        {
            "global_slot": range(GLOBAL_CONTEXT_COUNT),
            "continuous_symbol": GLOBAL_CONTEXT_SYMBOLS,
        }
    ).write_parquet(tmp_path / "global_context_index.parquet")
    schema = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "slow_channels": [
            {"index": index, "name": name} for index, name in enumerate(SLOW_CHANNELS)
        ],
        "local_context": {
            "symbols": list(LOCAL_CONTEXT_SYMBOLS),
            "exposure_beta_source_symbols": list(LOCAL_CONTEXT_SYMBOLS[:6]),
        },
    }
    (tmp_path / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    resolved = resolve_context_ablation_for_store(tmp_path, "drop_all_local")
    assert resolved.local_slots == tuple(range(LOCAL_CONTEXT_COUNT))
    assert resolved.equity_slow_indices == tuple(range(20, 26))
    bad = pl.read_parquet(tmp_path / "context_index.parquet").with_columns(
        pl.when(pl.col("context_slot") == 0)
        .then(pl.lit("UNKNOWN"))
        .otherwise(pl.col("symbol"))
        .alias("symbol")
    )
    bad.write_parquet(tmp_path / "context_index.parquet")
    with pytest.raises(ValueError, match="local context"):
        resolve_context_ablation_for_store(tmp_path, "drop_win")


def test_none_is_bitwise_identical_and_ablation_preserves_common_batch(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest((0, 1), 2)
    legacy = VectorizedFeatureDataset(store, rows, "context_pooled", "enabled")[request]
    explicit = _patch_batch(store, rows, "none")
    for key in legacy:
        np.testing.assert_array_equal(explicit[key], legacy[key])

    ablated = _patch_batch(store, rows, "drop_win")
    for key in (
        "targets",
        "label_mask",
        "raw_returns",
        "sample_valid_mask",
        "sample_id",
        "date_idx",
        "decision_idx",
    ):
        np.testing.assert_array_equal(ablated[key], legacy[key])
    np.testing.assert_array_equal(
        ablated["instrument_mask"][:, :EQUITY_COUNT],
        legacy["instrument_mask"][:, :EQUITY_COUNT],
    )


def test_local_ablation_zeroes_all_paths_dependency_only_and_not_memmaps(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    watched = {
        filename: np.load(store / filename).copy()
        for filename in (
            "equity_slow.npy",
            "context_features.npy",
            "context_slow.npy",
            "context_data_ready.npy",
        )
    }
    baseline = _patch_batch(store, rows, "none")
    changed = _patch_batch(store, rows, "drop_win")
    win = EQUITY_COUNT
    assert not changed["patches"][:, win].any()
    assert not changed["slow_features"][:, win].any()
    assert not changed["history_patch_mask"][:, win].any()
    assert not changed["instrument_mask"][:, win].any()
    assert not changed["slow_features"][:, :EQUITY_COUNT, 20].any()
    np.testing.assert_array_equal(
        changed["slow_features"][:, :EQUITY_COUNT, 21:],
        baseline["slow_features"][:, :EQUITY_COUNT, 21:],
    )
    np.testing.assert_array_equal(
        changed["patches"][:, win + 1 :], baseline["patches"][:, win + 1 :]
    )
    for filename, expected in watched.items():
        np.testing.assert_array_equal(np.load(store / filename), expected)


@pytest.mark.parametrize(
    "key",
    (
        "drop_us_rates",
        "drop_global_non_rates",
        "drop_fixed_di",
        "drop_all_local",
        "drop_all_context",
    ),
)
def test_family_and_all_context_ablation_masks_expected_slots(
    tmp_path: Path, key: str
) -> None:
    store, rows = _synthetic_store(tmp_path)
    batch = _patch_batch(store, rows, key)
    resolved = _resolved(key)
    slots = [EQUITY_COUNT + slot for slot in resolved.local_slots]
    slots.extend(
        EQUITY_COUNT + LOCAL_CONTEXT_COUNT + slot for slot in resolved.global_slots
    )
    for slot in slots:
        assert not batch["patches"][:, slot].any()
        assert not batch["slow_features"][:, slot].any()
        assert not batch["history_patch_mask"][:, slot].any()
        assert not batch["instrument_mask"][:, slot].any()
    for index in resolved.equity_slow_indices:
        assert not batch["slow_features"][:, :EQUITY_COUNT, index].any()
    assert batch["patches"].shape[1] == INSTRUMENT_COUNT
    assert batch["instrument_mask"][:, :3].all()


def test_stage3_composite_masks_only_removed_paths_and_preserves_eligibility(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    baseline = _patch_batch(store, rows, "none")
    key = "drop_win_and_global_non_rates_except_es"
    changed = _patch_batch(store, rows, key)
    resolved = _resolved(key)
    removed_slots = {
        *(EQUITY_COUNT + slot for slot in resolved.local_slots),
        *(EQUITY_COUNT + LOCAL_CONTEXT_COUNT + slot for slot in resolved.global_slots),
    }
    kept_slots = set(range(EQUITY_COUNT, INSTRUMENT_COUNT)) - removed_slots
    for slot in removed_slots:
        assert not changed["patches"][:, slot].any()
        assert not changed["slow_features"][:, slot].any()
        assert not changed["history_patch_mask"][:, slot].any()
        assert not changed["instrument_mask"][:, slot].any()
    for slot in kept_slots:
        np.testing.assert_array_equal(
            changed["patches"][:, slot], baseline["patches"][:, slot]
        )
        np.testing.assert_array_equal(
            changed["slow_features"][:, slot],
            baseline["slow_features"][:, slot],
        )
        np.testing.assert_array_equal(
            changed["history_patch_mask"][:, slot],
            baseline["history_patch_mask"][:, slot],
        )
        np.testing.assert_array_equal(
            changed["instrument_mask"][:, slot],
            baseline["instrument_mask"][:, slot],
        )
    assert not changed["slow_features"][:, :EQUITY_COUNT, 20].any()
    np.testing.assert_array_equal(
        changed["slow_features"][:, :EQUITY_COUNT, :20],
        baseline["slow_features"][:, :EQUITY_COUNT, :20],
    )
    np.testing.assert_array_equal(
        changed["slow_features"][:, :EQUITY_COUNT, 21:],
        baseline["slow_features"][:, :EQUITY_COUNT, 21:],
    )
    for field in (
        "targets",
        "label_mask",
        "raw_returns",
        "sample_valid_mask",
        "sample_id",
        "date_idx",
        "decision_idx",
    ):
        np.testing.assert_array_equal(changed[field], baseline[field])
    np.testing.assert_array_equal(
        changed["patches"][:, :EQUITY_COUNT],
        baseline["patches"][:, :EQUITY_COUNT],
    )
    np.testing.assert_array_equal(
        changed["instrument_mask"][:, :EQUITY_COUNT],
        baseline["instrument_mask"][:, :EQUITY_COUNT],
    )


def test_drop_all_global_matches_legacy_masked_semantics(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    dropped = _patch_batch(store, rows, "drop_all_global")
    masked = _patch_batch(store, rows, "none", "masked")
    for key in dropped:
        np.testing.assert_array_equal(dropped[key], masked[key])


def test_tabular_ablation_zeroes_values_validity_readiness_and_dependency(
    tmp_path: Path,
) -> None:
    store, rows = _synthetic_store(tmp_path)
    request = BatchRequest((0, 1), 2)
    baseline = VectorizedFeatureDataset(store, rows, "mlp", "enabled")[request][
        "tabular_features"
    ]
    changed = VectorizedFeatureDataset(
        store,
        rows,
        "mlp",
        "enabled",
        context_ablation=_resolved("drop_win"),
    )[request]["tabular_features"]
    context_dynamic_start = (
        SLOW_FEATURE_COUNT + len(TABULAR_OFFSETS) * DYNAMIC_CHANNEL_COUNT
    )
    local_block_width = len(TABULAR_OFFSETS) * 16
    assert not changed[
        :, :, context_dynamic_start : context_dynamic_start + local_block_width
    ].any()
    np.testing.assert_array_equal(
        changed[
            :,
            :,
            context_dynamic_start + local_block_width : context_dynamic_start
            + 2 * local_block_width,
        ],
        baseline[
            :,
            :,
            context_dynamic_start + local_block_width : context_dynamic_start
            + 2 * local_block_width,
        ],
    )
    assert not changed[:, :, 20].any()
    local_slow_start = context_dynamic_start + 15 * local_block_width
    assert not changed[
        :, :, local_slow_start : local_slow_start + SLOW_FEATURE_COUNT
    ].any()
    validity_start = local_slow_start + 15 * SLOW_FEATURE_COUNT + 2
    local_validity_start = validity_start + len(TABULAR_OFFSETS)
    assert not changed[
        :,
        :,
        local_validity_start : local_validity_start + len(TABULAR_OFFSETS),
    ].any()
    local_readiness_start = local_validity_start + 35 + 40
    assert not changed[:, :, local_readiness_start].any()
    assert changed.shape[-1] == TABULAR_FEATURE_COUNT


def test_cli_accepts_all_keys_and_rejects_ambiguous_or_context_free() -> None:
    for key in CONTEXT_ABLATION_KEYS:
        args = train.parse_args(build_stage1_command(key)[3:])
        assert args.context_ablation == key
        assert args.global_context == "enabled"
    with pytest.raises(SystemExit):
        train.parse_args(
            [
                *build_stage1_command("drop_win")[3:],
                "--global-context",
                "masked",
            ]
        )
    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--model",
                "temporal_only",
                "--optimizer",
                "adamw",
                "--soft-rank-temperature",
                "0.50",
                "--seed",
                "29",
                "--context-ablation",
                "drop_win",
            ]
        )


def test_context_free_dataset_rejects_non_none_ablation(tmp_path: Path) -> None:
    store, rows = _synthetic_store(tmp_path)
    with pytest.raises(ValueError, match="Context-free"):
        VectorizedFeatureDataset(
            store,
            rows,
            "pooled_market",
            None,
            context_ablation=_resolved("drop_win"),
        )


def test_run_names_and_matrix_jobs_bind_ablation_identity() -> None:
    created = train.datetime(2026, 1, 2, tzinfo=train.timezone.utc)
    baseline = train._run_directory_name(
        "tcn",
        TCNSettings("context_pooled", 64, "full", "swiglu"),
        "sam_adamw",
        "soft_spearman",
        0.5,
        0.125,
        "enabled",
        29,
        created,
    )
    ablated = train._run_directory_name(
        "tcn",
        TCNSettings("context_pooled", 64, "full", "swiglu"),
        "sam_adamw",
        "soft_spearman",
        0.5,
        0.125,
        "enabled",
        29,
        created,
        "drop_win",
    )
    assert "_ablation-" not in baseline
    assert "_ablation-drop_win_seed29_" in ablated
    jobs = stage1_jobs()
    assert len(jobs) == 25
    assert {job["context_ablation"] for job in jobs} == set(
        STAGE1_CONTEXT_ABLATION_ORDER
    )
    assert {job["seed"] for job in jobs} == {29}
    assert all("--context-ablation" in job["command"] for job in jobs)


def test_production_run_discovery_excludes_lock_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_directory = tmp_path / "_ops"
    run_directory = tmp_path / "model_run"
    lock_directory.mkdir()
    run_directory.mkdir()
    monkeypatch.setattr(
        "brazil_rv.modeling.stage1_context_ablation.RUN_OUTPUT_BASE", tmp_path
    )

    assert _production_run_directories() == {run_directory.resolve()}


def _lock_payload(
    *,
    hostname: str = "lambda-a",
    boot_id: str | None = "boot-a",
    pid: int = 1234,
    token: str = "owner-token",
    purpose: str = "test lock",
) -> dict[str, object]:
    return {
        "schema": process_lock.LOCK_SCHEMA,
        "version": process_lock.LOCK_VERSION,
        "pid": pid,
        "hostname": hostname,
        "boot_id": boot_id,
        "purpose": purpose,
        "token": token,
        "created_at_utc": "2026-08-05T12:00:00+00:00",
    }


def _set_test_host(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hostname: str = "lambda-a",
    boot_id: str | None = "boot-a",
) -> None:
    monkeypatch.setattr(
        process_lock,
        "_current_host_identity",
        lambda: process_lock.HostIdentity(hostname, boot_id),
    )


def _forbid_pid_check(pid: int) -> bool:
    raise AssertionError(f"foreign PID {pid} must not be queried locally")


def test_same_host_live_pid_remains_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", lambda pid: pid == 1234)
    lock_path = tmp_path / "live.lock"
    lock_path.write_text(json.dumps(_lock_payload()), encoding="utf-8")

    owner = active_lock_owner(lock_path)

    assert owner is not None
    assert owner["status"] == "active_local"
    assert owner["hostname"] == "lambda-a"
    assert owner["boot_id"] == "boot-a"
    assert lock_path.exists()


def test_same_host_dead_pid_is_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", lambda pid: False)
    lock_path = tmp_path / "dead.lock"
    lock_path.write_text(json.dumps(_lock_payload()), encoding="utf-8")

    assert active_lock_owner(lock_path) is None
    assert not lock_path.exists()


def test_foreign_hostname_is_never_reclaimed_from_local_pid_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", _forbid_pid_check)
    lock_path = tmp_path / "foreign.lock"
    lock_path.write_text(
        json.dumps(_lock_payload(hostname="lambda-b")), encoding="utf-8"
    )

    owner = active_lock_owner(lock_path)

    assert owner is not None
    assert owner["status"] == "foreign"
    assert lock_path.exists()


def test_same_hostname_different_boot_is_foreign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", _forbid_pid_check)
    lock_path = tmp_path / "replaced-instance.lock"
    lock_path.write_text(json.dumps(_lock_payload(boot_id="boot-b")), encoding="utf-8")

    owner = active_lock_owner(lock_path)

    assert owner is not None
    assert owner["status"] == "foreign"
    assert lock_path.exists()


def test_legacy_lock_is_ambiguous_and_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", _forbid_pid_check)
    lock_path = tmp_path / "legacy.lock"
    lock_path.write_text(json.dumps({"pid": 1234}), encoding="utf-8")

    owner = active_lock_owner(lock_path)

    assert owner is not None
    assert owner["status"] == "legacy_ambiguous"
    assert lock_path.exists()


def test_recent_partial_lock_is_initializing(tmp_path: Path) -> None:
    lock_path = tmp_path / "partial.lock"
    lock_path.write_text("{", encoding="utf-8")

    owner = active_lock_owner(lock_path)

    assert owner is not None
    assert owner["status"] == "initializing"
    assert lock_path.exists()


def test_old_malformed_lock_requires_operator_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "malformed.lock"
    lock_path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(process_lock, "_lock_age_seconds", lambda path: 120.0)

    owner = active_lock_owner(lock_path)

    assert owner is not None
    assert owner["status"] == "malformed_requires_operator_cleanup"
    assert lock_path.exists()


def test_ownership_token_mismatch_prevents_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    lock_path = tmp_path / "token.lock"
    with exclusive_process_lock(lock_path, "outer"):
        replacement = json.loads(lock_path.read_text(encoding="utf-8"))
        replacement["token"] = "replacement-owner"
        lock_path.write_text(json.dumps(replacement), encoding="utf-8")

    assert lock_path.exists()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == (
        "replacement-owner"
    )
    lock_path.unlink()


def test_same_host_lock_attempts_are_exclusive_and_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", lambda pid: True)
    lock_path = tmp_path / "production.lock"
    with exclusive_process_lock(lock_path, "outer"):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert {
            "schema",
            "version",
            "pid",
            "hostname",
            "boot_id",
            "purpose",
            "token",
            "created_at_utc",
        } <= payload.keys()
        with pytest.raises(RuntimeError) as captured:
            with exclusive_process_lock(lock_path, "inner"):
                pass
        message = str(captured.value)
        for expected in (
            str(lock_path),
            f"pid={payload['pid']}",
            "hostname='lambda-a'",
            "boot_id='boot-a'",
            "purpose='outer'",
            f"token='{payload['token']}'",
            f"created_at_utc='{payload['created_at_utc']}'",
        ):
            assert expected in message

    assert not lock_path.exists()


def test_production_lock_blocks_stage1_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", lambda pid: True)
    production_lock = tmp_path / "production.lock"
    monkeypatch.setattr(
        stage1_context_ablation, "PRODUCTION_TRAINING_LOCK", production_lock
    )
    monkeypatch.setattr(
        stage1_context_ablation,
        "_prepare",
        lambda require_clean: ("a" * 40, True, tmp_path / "store", {}),
    )

    with exclusive_process_lock(production_lock, "production training"):
        with pytest.raises(RuntimeError, match="Another production training run"):
            stage1_context_ablation.run_sweep(tmp_path / "state")


def test_sweep_lock_blocks_second_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_host(monkeypatch)
    monkeypatch.setattr(process_lock, "_pid_is_active", lambda pid: True)
    monkeypatch.setattr(
        stage1_context_ablation,
        "_prepare",
        lambda require_clean: ("a" * 40, True, tmp_path / "store", {}),
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    sweep_lock = state_dir / "sweep.lock"

    with exclusive_process_lock(sweep_lock, "first sweep"):
        with pytest.raises(RuntimeError, match="already active"):
            stage1_context_ablation.run_sweep(state_dir)


def _completed_run(
    run_dir: Path,
    configuration: dict[str, object],
    key: str,
    score: float,
    *,
    common_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    run_dir.mkdir(parents=True)
    manifest = {
        "status": "completed",
        "model_name": "tcn",
        "model_family": "tcn",
        "tcn_settings": configuration["tcn_settings"],
        "architecture_constants": {
            **asdict(
                architecture_for_model(
                    "tcn", TCNSettings(**configuration["tcn_settings"])
                )
            ),
        },
        "parameter_count": configuration["parameter_count"],
        "peer_features": configuration["peer_features"],
        "optimizer_variant": configuration["optimizer_variant"],
        "objective": configuration["objective"],
        "sam": configuration["sam"],
        "global_context": "enabled",
        "context_ablation": get_context_ablation(key).metadata(),
        "seed": 29,
        "git_commit_sha": configuration["git_commit_sha"],
        "resolved_feature_store_path": configuration["resolved_feature_store_path"],
        "feature_manifest_contract_version": FEATURE_CONTRACT_VERSION,
        "split_boundaries": configuration["split_boundaries"],
        "best_validation_primary_score": score,
        "best_epoch": 3,
        "training_duration_seconds": 120.0,
        "global_context_source_hashes": {"source": "hash"},
        "global_context_normalized_store_hashes": {"store": "hash"},
        "training_constants": {"maximum_epochs": 20},
        "optimizer_constants": {"adamw": {"lr": 0.0003}},
        "scheduler_constants": {"schedule": "cosine"},
        "scheduler_steps": {"steps_per_epoch": 77},
        "physical_microbatch_size": 64,
        "accumulation_steps": 8,
        "effective_batch_size": 512,
        "evaluation_batch_size": 256,
        "num_workers": 8,
        "prefetch_factor": 4,
        "precision": "bf16",
        "bf16": True,
        "grad_scaler_used": False,
        "pytorch_version": "test",
        "cuda_version": "test",
        "hardware": {"device_name": "GH200"},
        **(common_fields or {}),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "validation_metrics.json").write_text(
        json.dumps({"primary_score": score, "horizons": []}), encoding="utf-8"
    )
    pl.DataFrame({"placeholder": [1]}).write_parquet(
        run_dir / "validation_daily_metrics.parquet"
    )
    (run_dir / "best.pt").write_bytes(b"fixture")
    (run_dir / "history.csv").write_text("epoch\n3\n", encoding="utf-8")
    return manifest


@pytest.mark.parametrize(
    "field",
    (
        "git_commit_sha",
        "resolved_feature_store_path",
        "seed",
        "tcn_settings",
        "context_ablation",
    ),
)
def test_resume_validation_rejects_wrong_identity(tmp_path: Path, field: str) -> None:
    feature_store = tmp_path / "store"
    configuration = _configuration("a" * 40, feature_store)
    run_dir = tmp_path / "run"
    manifest = _completed_run(run_dir, configuration, "drop_win", 0.01)
    assert validate_completed_run(run_dir, configuration, "drop_win") == 0.01
    manifest[field] = (
        {"wrong": field} if field in {"tcn_settings", "context_ablation"} else "wrong"
    )
    if field == "seed":
        manifest[field] = 11
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        validate_completed_run(run_dir, configuration, "drop_win")


def _validation_dates() -> list[date]:
    dates: list[date] = []
    current = VALIDATION_START
    while current <= VALIDATION_END:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    indices = np.linspace(0, len(dates) - 1, 244, dtype=int)
    selected = [dates[index] for index in indices]
    assert len(set(selected)) == 244
    assert selected[0] == VALIDATION_START and selected[-1] == VALIDATION_END
    return selected


def _analyzer_fixture(tmp_path: Path) -> Path:
    configuration = _configuration("b" * 40, tmp_path / "feature_store")
    jobs: list[dict[str, object]] = []
    dates = _validation_dates()
    for key_index, key in enumerate(STAGE1_CONTEXT_ABLATION_ORDER):
        run_dir = tmp_path / f"run_{key}"
        horizon_values = {
            horizon: 0.01 + key_index * 0.0001 + horizon * 0.000001
            for horizon in HORIZONS
        }
        score = float(np.mean(tuple(horizon_values.values())))
        manifest = _completed_run(run_dir, configuration, key, score)
        rows = [
            {
                "trade_date": trade_date,
                "date_idx": date_index,
                "horizon_minutes": horizon,
                "spearman_ic": horizon_values[horizon],
            }
            for date_index, trade_date in enumerate(dates)
            for horizon in HORIZONS
        ]
        pl.DataFrame(rows).write_parquet(run_dir / "validation_daily_metrics.parquet")
        (run_dir / "validation_metrics.json").write_text(
            json.dumps(
                {
                    "primary_score": score,
                    "horizons": [
                        {
                            "horizon_minutes": horizon,
                            "mean_daily_spearman_ic": value,
                        }
                        for horizon, value in horizon_values.items()
                    ],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        jobs.append(
            {
                "context_ablation": key,
                "status": "completed",
                "run_dir": str(run_dir),
            }
        )
    state = {
        "state_version": 1,
        "sweep_name": "stage1_context_ablation_seed29",
        "status": "completed",
        "configuration": configuration,
        "jobs": jobs,
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    return state_path


def test_analyzer_rejects_partial_and_inconsistent_matrices(tmp_path: Path) -> None:
    state_path = _analyzer_fixture(tmp_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    partial = copy.deepcopy(state)
    partial["status"] = "running"
    partial_path = tmp_path / "partial.json"
    partial_path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ValueError, match="completed"):
        analyze_sweep(partial_path, tmp_path / "partial_output")

    wrong_run = Path(state["jobs"][1]["run_dir"])
    manifest_path = wrong_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["precision"] = "fp32"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-ablation"):
        analyze_sweep(state_path, tmp_path / "inconsistent_output")


def test_analyzer_fixture_and_bootstrap_are_deterministic(tmp_path: Path) -> None:
    first = paired_moving_block_bootstrap(
        np.arange(20, dtype=np.float64),
        np.arange(20, dtype=np.float64) + 0.25,
        replications=500,
    )
    second = paired_moving_block_bootstrap(
        np.arange(20, dtype=np.float64),
        np.arange(20, dtype=np.float64) + 0.25,
        replications=500,
    )
    assert first == second
    assert first["paired_mean_delta"] == pytest.approx(0.25)

    state_path = _analyzer_fixture(tmp_path / "fixture")
    json_path, csv_path = analyze_sweep(state_path, tmp_path / "analysis")
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    table = pl.read_csv(csv_path)
    assert summary["run_count"] == 25
    assert len(summary["results"]) == 25
    assert table.height == 25
    assert tuple(table["context_ablation"]) == STAGE1_CONTEXT_ABLATION_ORDER
    baseline = summary["results"][0]
    assert (
        baseline["context_ablation_specification"]
        == get_context_ablation("none").metadata()
    )
    assert Path(baseline["run_manifest_path"]).is_file()
