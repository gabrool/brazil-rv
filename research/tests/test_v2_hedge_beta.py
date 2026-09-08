from datetime import date
from pathlib import Path

import numpy as np
import pytest

from brazil_rv.v2.hedge_beta import (
    build_hedge_beta_sidecar,
    load_hedge_beta_sidecar,
    resolve_hedge_beta,
    rolling_hedge_beta,
)


def _prices() -> tuple[np.ndarray, np.ndarray]:
    x = np.sin(np.arange(100)) * 0.01
    hedge = 100 * np.cumprod(1 + x)
    equity = 100 * np.cumprod(1 + x[:, None] * [1.0, 1.2, 8.0, -4.0], axis=0)
    return equity, hedge


def test_economic_beta_blume_clip_and_same_day_causality() -> None:
    equity, hedge = _prices()
    valid = np.ones_like(equity, dtype=bool)
    beta, mask = rolling_hedge_beta(equity, valid, hedge)
    np.testing.assert_allclose(beta[60], [1.0, 1.134, 3.0, -1.0], atol=1e-12)
    assert not mask[:41].any()
    assert mask[41:].all()
    assert -np.dot([-1, 1], beta[60, :2]) == pytest.approx(-0.134)
    changed = equity.copy()
    changed[60:] *= 3
    changed_hedge = hedge.copy()
    changed_hedge[60:] *= 5
    other, other_mask = rolling_hedge_beta(changed, valid, changed_hedge)
    np.testing.assert_array_equal(beta[:61], other[:61])
    np.testing.assert_array_equal(mask[:61], other_mask[:61])
    scaled, _ = rolling_hedge_beta(equity * [5, 0.2, 3, 10], valid, hedge * 7)
    np.testing.assert_allclose(beta, scaled, atol=1e-12)


def test_calendar_window_never_bridges_missing_observations() -> None:
    equity, hedge = _prices()
    valid = np.ones_like(equity, dtype=bool)
    valid[42:65, 0] = False
    beta, mask = rolling_hedge_beta(equity, valid, hedge)
    # Endpoints 41..64: 24 of the 60 returns are invalid at decision 80.
    assert not mask[80, 0]
    assert np.isnan(beta[80, 0])
    assert mask[80, 1]
    # Invalid finite values must never participate.
    equity[~valid] = 1e12
    changed, changed_mask = rolling_hedge_beta(equity, valid, hedge)
    np.testing.assert_array_equal(beta, changed)
    np.testing.assert_array_equal(mask, changed_mask)


def test_beta_fallback_expiry_and_fold_boundary_history() -> None:
    values = np.zeros((25, 2))
    valid = np.zeros_like(values, dtype=bool)
    values[0] = [1.4, 0.8]
    valid[0] = True
    resolved, fallback = resolve_hedge_beta(values, valid)
    np.testing.assert_array_equal(resolved[:21], np.tile([1.4, 0.8], (21, 1)))
    np.testing.assert_array_equal(resolved[21:], 1.0)
    assert not fallback[0].any()
    assert fallback[1:].all()
    sliced, _ = resolve_hedge_beta(
        values[10:], valid[10:], history=(values[:10], valid[:10])
    )
    np.testing.assert_array_equal(sliced, resolved[10:])


@pytest.mark.parametrize("day", [date(2025, 1, 2), date(2026, 1, 2)])
def test_sidecar_refuses_heldout_dates_before_reading_files(
    tmp_path: Path, day: date
) -> None:
    with pytest.raises(PermissionError, match="2025/2026"):
        build_hedge_beta_sidecar(
            store_root=tmp_path / "absent",
            expected_store_manifest_sha256="",
            bova11_root=tmp_path / "absent",
            expected_bova11_manifest_sha256="",
            output_root=tmp_path / "out",
            end_date=day,
        )
    with pytest.raises(PermissionError, match="2025/2026"):
        load_hedge_beta_sidecar(
            tmp_path / "absent",
            expected_manifest_sha256="",
            expected_store_manifest_sha256="",
            expected_bova11_manifest_sha256="",
            canonical_dates=[day],
            isins=["A"],
        )
