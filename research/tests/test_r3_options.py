from __future__ import annotations

from pathlib import Path

import polars as pl

from brazil_rv.modeling.r3_options import FOLDS, _single_dead, options_gate


def _single_fold(overall: float, horizons: tuple[float, float, float]) -> dict:
    return {
        "parent_minus_ablated_ic": overall,
        "per_horizon_parent_minus_ablated_ic": {
            "30": horizons[0],
            "60": horizons[1],
            "120": horizons[2],
        },
    }


def test_r3_singleton_rule_preserves_overall_and_horizon_conditions() -> None:
    dead = {
        fold: _single_fold(-0.001, (-0.001, 0.001, -0.002)) for fold in FOLDS
    }
    positive_fold = {**dead, "fold_a": _single_fold(0.0001, (-1.0, -1.0, -1.0))}
    positive_horizons = {
        **dead,
        "fold_b": _single_fold(-0.001, (0.001, 0.002, -0.001)),
    }

    assert _single_dead(dead)
    assert not _single_dead(positive_fold)
    assert not _single_dead(positive_horizons)


def _gate_folds(tmp_path: Path, deltas: tuple[float, float, float]) -> dict:
    result = {}
    for fold, delta in zip(FOLDS, deltas, strict=True):
        path = tmp_path / fold
        path.mkdir(parents=True)
        daily = path / "daily_delta.parquet"
        pl.DataFrame(
            {
                "date_idx": list(
                    range(100 * len(result), 100 * len(result) + 80)
                ),
                "candidate_minus_parent_ic": [delta] * 80,
            }
        ).write_parquet(daily)
        result[fold] = {
            "primary_standalone": {
                "candidate_minus_parent_ic": delta,
                "daily_delta": str(daily),
            }
        }
    return result


def test_options_gate_requires_all_four_predeclared_checks(tmp_path: Path) -> None:
    passed = options_gate(
        _gate_folds(tmp_path / "pass", (0.0007, 0.0006, 0.0005)),
        "primary_standalone",
    )
    failed_floor = options_gate(
        _gate_folds(tmp_path / "floor", (0.003, 0.002, -0.0006)),
        "primary_standalone",
    )

    assert passed["passed"]
    assert passed["pooled_daily_delta"]["lower_90"] > 0
    assert not failed_floor["passed"]
    assert not failed_floor["checks"]["no_fold_below_minus_0_0005"]
