from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pytest

from brazil_rv.modeling.data import (
    DI_TILT_SIDECAR_SCHEMA,
    di_tilt_sidecar_identity,
)
from brazil_rv.preprocessing.contract import (
    BETA_MIN_PAIRED_SESSIONS,
    CONTEXT_SESSION_MINUTES,
    DECISION_CONTEXT_INDICES,
    HORIZONS,
    MIN_ACTIVE_EQUITIES,
)
from brazil_rv.preprocessing.analyze_preprocessing import AuditDates
from brazil_rv.preprocessing.next_stage_sidecars import (
    _residual_variant,
    causal_unclipped_beta,
    exact_context_returns,
)


def test_exact_context_returns_use_entry_open_and_exact_exit() -> None:
    grid = np.zeros((1, CONTEXT_SESSION_MINUTES, 5), dtype=np.float64)
    observed = np.zeros(grid.shape[:2], dtype=bool)
    entry = DECISION_CONTEXT_INDICES[0]
    grid[0, entry, 0] = 100.0
    observed[0, entry] = True
    for horizon, close in zip(HORIZONS, (101.0, 102.0, 103.0), strict=True):
        exit_idx = entry + horizon - 1
        grid[0, exit_idx, 3] = close
        observed[0, exit_idx] = True

    returns, mask = exact_context_returns(grid, observed, is_rate=False)
    np.testing.assert_allclose(
        returns[0, 0],
        np.log(np.asarray([101.0, 102.0, 103.0]) / 100.0),
        rtol=0.0,
        atol=1e-7,
    )
    assert mask[0, 0].all()

    rate, _ = exact_context_returns(grid, observed, is_rate=True)
    np.testing.assert_allclose(rate[0, 0], (100.0, 200.0, 300.0))

    after = grid.copy()
    after[0, entry + HORIZONS[0], 3] = 1e9
    changed, _ = exact_context_returns(after, observed, is_rate=False)
    assert changed[0, 0, 0] == returns[0, 0, 0]

    missing = observed.copy()
    missing[0, entry + HORIZONS[1] - 1] = False
    _, missing_mask = exact_context_returns(grid, missing, is_rate=False)
    assert not missing_mask[0, 0, 1]


def test_short_unclipped_beta_is_causal_and_not_clipped() -> None:
    dates = BETA_MIN_PAIRED_SESSIONS + 5
    factor = np.linspace(-0.01, 0.01, dates)
    equity = (12.0 * factor + 0.001 * np.sin(np.arange(dates)))[:, None]
    valid = np.ones_like(equity, dtype=bool)
    factor_valid = np.ones(dates, dtype=bool)
    beta, ready = causal_unclipped_beta(
        equity, valid, factor, factor_valid, half_life_days=5
    )
    assert not ready[BETA_MIN_PAIRED_SESSIONS - 1, 0]
    assert ready[BETA_MIN_PAIRED_SESSIONS, 0]
    assert abs(beta[-1, 0]) > 5.0

    mutated = factor.copy()
    mutated[-1] = 10_000.0
    changed, _ = causal_unclipped_beta(
        equity, valid, mutated, factor_valid, half_life_days=5
    )
    assert changed[-1, 0] == beta[-1, 0]


def test_di_tilt_sidecar_identity_checks_hashes(tmp_path) -> None:
    feature_identity = {"path": "store", "metadata_sha256": "abc"}
    hashes = {}
    for name in ("tilt_exposure.npy", "tilt_ready.npy"):
        values = np.asarray([1.0], dtype=np.float32)
        if name == "tilt_ready.npy":
            values = values.astype(bool)
        np.save(tmp_path / name, values, allow_pickle=False)
        hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / "audit.json").write_text("{}", encoding="utf-8")
    hashes["audit.json"] = hashlib.sha256(
        (tmp_path / "audit.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": DI_TILT_SIDECAR_SCHEMA,
                "source_feature_store": feature_identity,
                "test_accessed": False,
                "files_sha256": hashes,
            }
        ),
        encoding="utf-8",
    )
    assert (
        di_tilt_sidecar_identity(tmp_path, feature_identity)["files_sha256"] == hashes
    )
    with (tmp_path / "tilt_exposure.npy").open("ab") as output:
        output.write(b"mutation")
    with pytest.raises(ValueError, match="recorded contract"):
        di_tilt_sidecar_identity(tmp_path, feature_identity)


def test_residual_sidecar_requires_auxiliary_arrays(tmp_path) -> None:
    feature_identity = {"path": "store", "metadata_sha256": "abc"}
    hashes = {}
    values_by_name = {
        "tilt_exposure.npy": np.asarray([1.0], dtype=np.float32),
        "tilt_ready.npy": np.asarray([True]),
        "residual_targets.npy": np.zeros((1, 3), dtype=np.float32),
        "residual_mask.npy": np.ones((1, 3), dtype=bool),
    }
    for name, values in values_by_name.items():
        np.save(tmp_path / name, values, allow_pickle=False)
        hashes[name] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / "audit.json").write_text("{}", encoding="utf-8")
    hashes["audit.json"] = hashlib.sha256(
        (tmp_path / "audit.json").read_bytes()
    ).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": DI_TILT_SIDECAR_SCHEMA,
                "source_feature_store": feature_identity,
                "test_accessed": False,
                "files_sha256": hashes,
            }
        ),
        encoding="utf-8",
    )
    assert (
        di_tilt_sidecar_identity(tmp_path, feature_identity, require_residual=True)[
            "files_sha256"
        ]
        == hashes
    )


def test_residual_variant_stops_at_training_boundary(tmp_path) -> None:
    equities = MIN_ACTIVE_EQUITIES
    shape = (2, equities, 1, len(HORIZONS))
    ranks = np.linspace(-1.0, 1.0, equities, dtype=np.float32)
    raw = np.broadcast_to(ranks[None, :, None, None], shape).copy()
    for name, values in {
        "raw_returns.npy": raw,
        "targets.npy": raw,
        "label_mask.npy": np.ones(shape, dtype=bool),
        "cross_section_median.npy": np.zeros((2, 1, len(HORIZONS))),
    }.items():
        np.save(tmp_path / name, values, allow_pickle=False)
    dates = AuditDates(
        trade_dates=(date(2024, 1, 2), date(2025, 1, 2)),
        train=np.asarray([0], dtype=np.int64),
        validation=np.asarray([1], dtype=np.int64),
    )
    component = (
        np.zeros((2, equities), dtype=np.float32),
        np.ones((2, equities), dtype=bool),
        np.zeros((2, 1, len(HORIZONS)), dtype=np.float32),
        np.ones((2, 1, len(HORIZONS)), dtype=bool),
    )

    targets, mask, _ = _residual_variant(
        tmp_path,
        dates,
        np.ones((2, equities), dtype=np.float64),
        (component,),
    )

    assert mask[0].all()
    assert not mask[1].any()
    assert not targets[1].any()
