from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.execution.inputs import (
    causal_liquidity,
    causal_rank_scores,
    causal_roll_spreads,
    expand_refreshes,
    lagged_quarter_spreads,
    load_daily_cdi_rates,
    load_discovery_prediction_archive,
    write_discovery_prediction_manifest,
)
from brazil_rv.modeling.contract import EQUITY_COUNT
from brazil_rv.modeling.data import (
    FEATURE_STORE_CONTRACT,
    feature_store_identity,
    int64_identity_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_causal_ranks_are_tie_aware_masked_and_centered() -> None:
    scores = np.asarray([[[[3.0, 8.0], [1.0, 7.0], [3.0, 6.0], [9.0, 5.0]]]])
    mask = np.asarray([[[True, True, True, False]]])

    ranked = causal_rank_scores(scores, mask)

    assert ranked.dtype == np.float64
    assert np.allclose(ranked[0, 0, :, 0], [1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 0.0])
    assert np.allclose(ranked[0, 0, :, 1], [2.0 / 3.0, 0.0, -2.0 / 3.0, 0.0])


def test_refresh_expansion_supports_shared_and_per_day_grids() -> None:
    ranks = np.arange(2 * 3 * 1 * 1, dtype=np.float64).reshape(2, 3, 1, 1)
    valid = np.ones((2, 3, 1), dtype=bool)
    minutes = np.asarray([[2, 4, 6], [1, 2, 3]])

    expanded, expanded_valid, age, refresh = expand_refreshes(
        ranks, valid, minutes, minute_count=8
    )

    assert not expanded_valid[0, :2].any()
    assert np.array_equal(age[0], [-1, -1, 0, 1, 0, 1, 0, 1])
    assert np.array_equal(age[1], [-1, 0, 0, 0, 1, 2, 3, 4])
    assert np.array_equal(np.flatnonzero(refresh[0]), [2, 4, 6])
    assert np.array_equal(expanded[0, :, 0, 0], [0, 0, 0, 0, 1, 1, 2, 2])

    shared = expand_refreshes(ranks, valid, np.asarray([1, 3, 5]), 7)
    assert np.array_equal(shared[2][0], shared[2][1])


def test_liquidity_uses_only_prior_sessions() -> None:
    close = np.asarray(
        [
            [[10.0], [20.0]],
            [[30.0], [40.0]],
            [[500.0], [600.0]],
        ]
    )
    volume = np.asarray(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[100.0], [100.0]],
        ]
    )
    observed = np.ones_like(close, dtype=bool)

    adv, profile = causal_liquidity(close, volume, observed, lookback=2)

    assert np.isnan(adv[0, 0])
    assert adv[1, 0] == 50.0
    assert adv[2, 0] == (50.0 + 250.0) / 2.0
    assert np.array_equal(profile[1, :, 0], [10.0, 40.0])
    assert np.array_equal(profile[2, :, 0], [50.0, 100.0])

    changed = close.copy()
    changed[2] *= 10.0
    changed_adv, changed_profile = causal_liquidity(
        changed, volume, observed, lookback=2
    )
    assert np.array_equal(changed_adv[:3], adv[:3], equal_nan=True)
    assert np.array_equal(changed_profile[:3], profile[:3], equal_nan=True)


def test_roll_fallback_uses_strictly_prior_closes() -> None:
    returns = np.asarray([0.01, -0.01, 0.01, -0.01])
    one_day = np.exp(np.r_[0.0, np.cumsum(returns)])
    close = np.broadcast_to(one_day[None, :, None], (3, 5, 1)).copy()
    observed = np.ones_like(close, dtype=bool)

    spread = causal_roll_spreads(close, observed, lookback=2)

    assert np.isnan(spread[0, 0])
    assert spread[1, 0] > 0.0
    changed = close.copy()
    changed[2, :, 0] = np.linspace(10.0, 50.0, 5)
    changed_spread = causal_roll_spreads(changed, observed, lookback=2)
    assert np.array_equal(changed_spread[:3], spread[:3], equal_nan=True)


def test_spreads_are_hash_verified_and_lagged_one_quarter(tmp_path: Path) -> None:
    schedule = tmp_path / "roll_schedule.parquet"
    pl.DataFrame(
        {
            "security_id": ["a", "a", "b", "b"],
            "quarter": ["2024Q4", "2025Q1", "2024Q4", "2025Q1"],
            "schedule_full_spread_fraction": [0.01, 0.02, 0.03, 0.04],
        }
    ).write_parquet(schedule)

    result = lagged_quarter_spreads(
        schedule,
        [date(2025, 1, 15), date(2025, 4, 1)],
        ["a", "b"],
        _sha256(schedule),
    )

    assert np.array_equal(result, [[0.01, 0.03], [0.02, 0.04]])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        lagged_quarter_spreads(schedule, [date(2025, 1, 15)], ["a"], "0" * 64)


def test_cdi_loader_requires_exact_hash_and_complete_dates(tmp_path: Path) -> None:
    series = tmp_path / "cdi.parquet"
    pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "daily_cdi_rate": [0.0004, 0.0005],
        }
    ).write_parquet(series)

    result = load_daily_cdi_rates(
        series, [date(2024, 1, 3), date(2024, 1, 2)], _sha256(series)
    )

    assert np.array_equal(result, [0.0005, 0.0004])
    with pytest.raises(ValueError, match="incomplete"):
        load_daily_cdi_rates(series, [date(2024, 1, 4)], _sha256(series))


def _write_store(
    tmp_path: Path,
    *,
    trade_dates: tuple[date, ...] = (date(2024, 1, 2), date(2024, 1, 3)),
    decisions: tuple[int, ...] = (1, 3),
    cutoffs: tuple[int, ...] = (20, 30),
) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    (store / "manifest.json").write_text("{}\n", encoding="utf-8")
    (store / "feature_schema.json").write_text(
        json.dumps({"contract_version": FEATURE_STORE_CONTRACT}), encoding="utf-8"
    )
    pl.DataFrame(
        {"date_idx": range(len(trade_dates)), "trade_date": trade_dates}
    ).write_parquet(store / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": range(EQUITY_COUNT),
            "security_id": [f"security_{slot:03d}" for slot in range(EQUITY_COUNT)],
        }
    ).write_parquet(store / "equity_index.parquet")
    rows = []
    for date_idx, trade_date in enumerate(trade_dates):
        for decision, cutoff in zip(decisions, cutoffs, strict=True):
            rows.append(
                {
                    "sample_id": len(rows),
                    "date_idx": date_idx,
                    "decision_idx": decision,
                    "trade_date": trade_date,
                    "equity_cutoff_index": cutoff,
                }
            )
    pl.DataFrame(rows).write_parquet(store / "sample_index.parquet")
    activity = np.zeros((len(trade_dates), EQUITY_COUNT), dtype=bool)
    activity[:, :3] = True
    np.save(store / "equity_membership.npy", activity)
    np.save(store / "equity_data_ready.npy", activity)
    return store


def _write_archive(
    tmp_path: Path,
    *,
    trade_dates: tuple[date, ...] = (date(2024, 1, 2), date(2024, 1, 3)),
    decisions: tuple[int, ...] = (1, 3),
    cutoffs: tuple[int, ...] = (20, 30),
) -> tuple[Path, Path, Path, Path]:
    store = _write_store(
        tmp_path, trade_dates=trade_dates, decisions=decisions, cutoffs=cutoffs
    )
    prediction = tmp_path / "predictions.npz"
    reference = tmp_path / "reference.npz"
    manifest = tmp_path / "manifest.json"
    source_manifest = tmp_path / "source_run_manifest.json"
    sample_index = pl.read_parquet(store / "sample_index.parquet")
    scores = np.zeros((sample_index.height, EQUITY_COUNT, 1), dtype=np.float32)
    for sample in range(sample_index.height):
        scores[sample, :3, 0] = np.asarray([3.0, 1.0, 2.0]) + 3.0 * sample
    np.savez(prediction, predictions=scores)
    np.savez(
        reference,
        sample_id=sample_index["sample_id"].to_numpy(),
        date_idx=sample_index["date_idx"].to_numpy(),
        decision_idx=sample_index["decision_idx"].to_numpy(),
        label_mask=np.zeros_like(scores, dtype=bool),
    )
    sample_ids = sample_index["sample_id"].to_numpy().astype(np.int64)
    date_indices = sample_index["date_idx"].unique().sort().to_numpy().astype(np.int64)
    selection_window = {
        "date_count": int(date_indices.size),
        "sample_count": int(sample_ids.size),
        "date_identity_sha256": int64_identity_sha256(date_indices),
        "sample_identity_sha256": int64_identity_sha256(np.sort(sample_ids)),
    }
    source_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "feature_store_identity": feature_store_identity(store),
                "split": {
                    "training": "fold_a",
                    "selection": "fold_a",
                    "selection_window": selection_window,
                    "test_accessed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    write_discovery_prediction_manifest(
        manifest,
        store=store,
        prediction_path=prediction,
        reference_path=reference,
        source_manifest_path=source_manifest,
        split="fold_a",
        refresh_minutes=cutoffs,
        prediction_key="predictions",
    )
    return prediction, reference, manifest, store


def test_discovery_loader_hashes_identity_and_uses_store_refresh_indices(
    tmp_path: Path,
) -> None:
    prediction, reference, manifest, store = _write_archive(tmp_path)

    archive = load_discovery_prediction_archive(prediction, reference, manifest, store)

    assert archive.ranks.shape == (2, 2, EQUITY_COUNT, 1)
    assert np.array_equal(archive.date_idx, [0, 1])
    assert np.array_equal(archive.decision_idx, [1, 3])
    assert np.array_equal(archive.refresh_minutes, [20, 30])
    assert np.array_equal(archive.sample_id, [[0, 1], [2, 3]])
    assert archive.valid[..., :3, :].all()
    assert not archive.valid[..., 3:, :].any()
    assert np.allclose(archive.ranks[0, 0, :3, 0], [2.0 / 3.0, -2.0 / 3.0, 0.0])
    assert not archive.ranks[..., 3:, :].any()


def test_discovery_loader_accepts_explicit_dense_refresh_minutes(
    tmp_path: Path,
) -> None:
    prediction, reference, manifest, store = _write_archive(
        tmp_path, decisions=(0, 1, 2), cutoffs=(0, 1, 2)
    )

    archive = load_discovery_prediction_archive(prediction, reference, manifest, store)

    assert np.array_equal(archive.decision_idx, [0, 1, 2])
    assert np.array_equal(archive.refresh_minutes, [0, 1, 2])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"split": "validation"}, "canonical discovery-fold"),
        ({"split": "fold_test"}, "canonical discovery-fold"),
        ({"split": "oof"}, "canonical discovery-fold"),
        ({"official_validation_accessed": True}, "official or test"),
        ({"prediction_sha256": "0" * 64}, "hash mismatch"),
        ({"refresh_minutes": [21, 30]}, "canonical cutoffs"),
    ],
)
def test_discovery_loader_rejects_unsafe_or_unverified_archives(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    prediction, reference, manifest, store = _write_archive(tmp_path)
    values = json.loads(manifest.read_text(encoding="utf-8"))
    values.update(change)
    manifest.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_discovery_prediction_archive(prediction, reference, manifest, store)


def test_discovery_loader_binds_canonical_samples_and_training_dates(
    tmp_path: Path,
) -> None:
    prediction, reference, manifest, store = _write_archive(tmp_path)
    with np.load(reference, allow_pickle=False) as values:
        arrays = {name: values[name].copy() for name in values.files}
    arrays["date_idx"][0] = 1
    np.savez(reference, **arrays)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["reference_sha256"] = _sha256(reference)
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical sample identity"):
        load_discovery_prediction_archive(prediction, reference, manifest, store)

    other = tmp_path / "future"
    other.mkdir()
    prediction, reference, manifest, store = _write_archive(
        other,
        trade_dates=(date(2025, 1, 2), date(2025, 1, 3)),
    )
    with pytest.raises(ValueError, match="beyond discovery"):
        load_discovery_prediction_archive(prediction, reference, manifest, store)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("test_access", "official or test"),
        ("official_split", "split differs"),
        ("selection_identity", "selection-window identity"),
    ],
)
def test_discovery_loader_rechecks_source_provenance(
    tmp_path: Path, change: str, message: str
) -> None:
    prediction, reference, manifest, store = _write_archive(tmp_path)
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    source_path = Path(metadata["source_manifest"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if change == "test_access":
        source["split"]["test_accessed"] = True
    elif change == "official_split":
        source["split"]["training"] = "official"
        source["split"]["selection"] = "official"
    else:
        source["split"]["selection_window"]["sample_identity_sha256"] = "0" * 64
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_discovery_prediction_archive(prediction, reference, manifest, store)
