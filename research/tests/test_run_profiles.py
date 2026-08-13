from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

import brazil_rv.modeling.data as modeling_data
from brazil_rv.modeling.contract import CONTEXT_COUNT, EQUITY_COUNT
from brazil_rv.modeling.data import (
    DecisionGroupedPaddedBatchSampler,
    DateStratifiedMicrobatchSampler,
    create_evaluation_loader,
    create_training_loaders,
)
from brazil_rv.modeling.routing_identity_preflight import (
    _seeded_models,
    _synthetic_batch,
)
from brazil_rv.modeling.run_profiles import (
    EXPERIMENT_DECISION_INDICES,
    RUN_PROFILE_SCHEMA_VERSION,
    filter_profile_rows,
    resolve_run_profile,
    validate_run_profile_artifact,
    write_run_profile,
)
from brazil_rv.modeling.contract import GH200_RUNTIME


def _profile_store(tmp_path: Path) -> Path:
    store = tmp_path / "feature-store"
    universe = tmp_path / "pit-universe"
    store.mkdir()
    universe.mkdir()
    (universe / "manifest.json").write_text('{"version": 1}\n', encoding="utf-8")
    axis = pl.DataFrame(
        {
            "equity_slot": np.arange(EQUITY_COUNT, dtype=np.int16),
            "security_id": [f"SEC-{slot:03d}" for slot in range(EQUITY_COUNT)],
            "xp_symbol": [f"EQ{slot:03d}" for slot in range(EQUITY_COUNT)],
        }
    )
    axis.write_parquet(store / "equity_index.parquet")
    training = [
        {
            "effective_from": date(2022, 1, 3),
            "effective_to_exclusive": date(2022, 2, 1),
            "security_id": f"SEC-{slot:03d}",
            "accepted_identity": True,
            "is_member": True,
            "median_daily_turnover_brl": float(EQUITY_COUNT - slot),
        }
        for slot in range(EQUITY_COUNT)
    ]
    future = [
        {
            "effective_from": date(2025, 1, 2),
            "effective_to_exclusive": date(2025, 2, 3),
            "security_id": f"SEC-{slot:03d}",
            "accepted_identity": True,
            "is_member": True,
            "median_daily_turnover_brl": float(slot * 10_000),
        }
        for slot in range(EQUITY_COUNT)
    ]
    pl.DataFrame([*training, *future]).write_parquet(
        universe / "universe_metrics_monthly.parquet"
    )
    (store / "manifest.json").write_text(
        json.dumps(
            {
                "canonical_inputs": {
                    "point_in_time_universe": {"resolved_path": str(universe)}
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return store


def test_experiment_profile_is_training_only_deterministic_and_hashed(
    tmp_path: Path,
) -> None:
    store = _profile_store(tmp_path)
    profile = resolve_run_profile("experiment", store)

    assert profile.metadata()["schema_version"] == RUN_PROFILE_SCHEMA_VERSION
    assert profile.equity_count == 48
    assert profile.instrument_count == 48 + CONTEXT_COUNT
    assert profile.equity_slots == tuple(range(48))
    assert profile.decision_indices == EXPERIMENT_DECISION_INDICES
    assert len(profile.decision_indices) == 19
    assert profile.maximum_epochs == 3
    assert (
        profile.provenance["selection_interval"]["validation_or_test_rows_used"]
        is False
    )
    assert resolve_run_profile("experiment", store).metadata() == profile.metadata()
    production = resolve_run_profile("production", store)
    assert production.metadata()["schema_version"] == RUN_PROFILE_SCHEMA_VERSION
    assert production.equity_count == EQUITY_COUNT
    assert len(production.decision_indices) == 55
    assert production.decision_grouped_batches is True
    assert profile.decision_grouped_batches is True

    artifact = tmp_path / "run_profile.json"
    write_run_profile(artifact, profile)
    validate_run_profile_artifact(artifact, profile)
    artifact.write_text(artifact.read_text(encoding="utf-8").replace("EQ000", "DRIFT"))
    with pytest.raises(ValueError, match="exactly match"):
        validate_run_profile_artifact(artifact, profile)


def test_experiment_profile_filters_decisions_and_requires_512_training_dates(
    tmp_path: Path,
) -> None:
    store = _profile_store(tmp_path)
    profile = resolve_run_profile("experiment", store)
    date_count = 513
    membership = np.zeros((date_count, EQUITY_COUNT), dtype=bool)
    readiness = np.zeros_like(membership)
    membership[:, :35] = True
    readiness[:, :35] = True
    membership[-1, 29:35] = False
    np.save(store / "equity_membership.npy", membership)
    np.save(store / "equity_data_ready.npy", readiness)
    first = date(2022, 1, 3)
    rows = pl.DataFrame(
        [
            {
                "sample_id": date_idx * 55 + decision,
                "date_idx": date_idx,
                "trade_date": first + timedelta(days=date_idx),
                "decision_idx": decision,
            }
            for date_idx in range(date_count)
            for decision in range(55)
        ],
        schema_overrides={"trade_date": pl.Date},
    )

    filtered = filter_profile_rows(rows, store, profile)

    assert filtered.get_column("trade_date").n_unique() == 512
    assert filtered.height == 512 * 19
    assert set(filtered.get_column("decision_idx")) == set(EXPERIMENT_DECISION_INDICES)
    assert filtered.get_column("profile_active_equity_count").min() == 35


@pytest.mark.parametrize(
    "decision_indices", (tuple(range(55)), EXPERIMENT_DECISION_INDICES)
)
def test_decision_grouping_preserves_selected_multiset_and_date_uniqueness(
    decision_indices: tuple[int, ...],
) -> None:
    first = date(2022, 1, 3)
    rows = pl.DataFrame(
        [
            {
                "sample_id": day * len(decision_indices) + decision_position,
                "trade_date": first + timedelta(days=day),
                "decision_idx": decision,
            }
            for day in range(600)
            for decision_position, decision in enumerate(decision_indices)
        ],
        schema_overrides={"trade_date": pl.Date},
    )
    plain = DateStratifiedMicrobatchSampler(rows, GH200_RUNTIME, 29)
    grouped = DateStratifiedMicrobatchSampler(
        rows, GH200_RUNTIME, 29, decision_grouped=True
    )
    plain_indices = [
        index
        for request in list(plain)[: GH200_RUNTIME.accumulation_steps]
        for index in request.indices
    ]
    grouped_indices = [
        index
        for request in list(grouped)[: GH200_RUNTIME.accumulation_steps]
        for index in request.indices
    ]
    grouped_decisions = rows[grouped_indices].get_column("decision_idx").to_list()
    grouped_dates = rows[grouped_indices].get_column("trade_date").to_list()
    assert sorted(plain_indices) == sorted(grouped_indices)
    assert grouped_decisions == sorted(grouped_decisions)
    assert len(set(grouped_dates)) == 512

    validation_rows = rows.head(101)
    validation = DecisionGroupedPaddedBatchSampler(validation_rows, 16)
    real_positions = []
    for request in validation:
        real = list(request.indices[: request.valid_count])
        padded = list(request.indices[request.valid_count :])
        real_positions.extend(real)
        decisions = validation_rows[real].get_column("decision_idx")
        assert decisions.n_unique() == 1
        assert all(position == real[-1] for position in padded)
    assert sorted(real_positions) == list(range(validation_rows.height))


@pytest.mark.parametrize("profile_name", ("production", "experiment"))
def test_standard_loaders_apply_grouping_for_both_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_name: str
) -> None:
    store = _profile_store(tmp_path)
    profile = resolve_run_profile(profile_name, store)
    first = date(2022, 1, 3)
    rows = pl.DataFrame(
        {
            "sample_id": np.arange(512, dtype=np.int64),
            "trade_date": [first + timedelta(days=index) for index in range(512)],
            "decision_idx": np.arange(512, dtype=np.int64) % 55,
        },
        schema_overrides={"trade_date": pl.Date},
    )
    captured: list[object] = []
    monkeypatch.setattr(modeling_data, "VectorizedFeatureDataset", lambda *_: object())

    def capture_loader(_: object, sampler: object, *__: object) -> object:
        captured.append(sampler)
        return sampler

    monkeypatch.setattr(modeling_data, "_create_loader", capture_loader)
    _, validation_loader, sampler = create_training_loaders(
        store,
        rows,
        rows,
        "tcn",
        "enabled",
        GH200_RUNTIME,
        29,
        run_profile=profile,
    )
    evaluation_loader = create_evaluation_loader(
        store,
        rows,
        "tcn",
        "enabled",
        GH200_RUNTIME,
        29,
        run_profile=profile,
    )

    assert sampler.decision_grouped is True
    assert isinstance(validation_loader, DecisionGroupedPaddedBatchSampler)
    assert isinstance(evaluation_loader, DecisionGroupedPaddedBatchSampler)
    assert len(captured) == 3


def test_tcn_profile_packing_changes_only_runtime_tensor_axes() -> None:
    batch = _synthetic_batch(48)
    legacy, scaffold = _seeded_models("cpu", 48)
    assert batch["patches"].shape[1] == 48 + CONTEXT_COUNT
    assert batch["targets"].shape == (1, 48, 3)
    assert legacy.equity_count == scaffold.equity_count == 48
    assert legacy.instrument_count == scaffold.instrument_count == 48 + CONTEXT_COUNT
    assert list(legacy.state_dict()) == [
        name for name in scaffold.state_dict() if not name.startswith("routing.")
    ]
    legacy.eval()
    with torch.no_grad():
        predictions = legacy(
            batch["patches"],
            batch["history_patch_mask"],
            batch["instrument_mask"],
            batch["slow_features"],
            batch["state_position"],
            batch["peer_state"],
        )
    assert predictions.shape == (1, 48, 3)
