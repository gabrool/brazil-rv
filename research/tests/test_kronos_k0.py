from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.modeling.kronos_k0 import (
    BAR_FIELDS,
    CONTEXT_BARS,
    EQUITY_COUNT,
    SESSION_BARS,
    BarSidecar,
    _decision,
    aggregate_five_minute_bars,
    stable_context_seed,
)


def _minute_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date_idx": [0] * 11,
            "minute_idx": list(range(11)),
            "open": np.arange(100.0, 111.0),
            "high": np.arange(100.5, 111.5),
            "low": np.arange(99.5, 110.5),
            "close": np.arange(100.25, 111.25),
            "real_volume": np.arange(1, 12),
        }
    )


def test_future_minute_mutation_does_not_change_prior_aggregate() -> None:
    rows = _minute_rows()
    original = aggregate_five_minute_bars(rows).filter(pl.col("bar_idx") <= 1)
    mutated = rows.with_columns(
        pl.when(pl.col("minute_idx") == 10)
        .then(pl.lit(1_000_000.0))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    changed = aggregate_five_minute_bars(mutated).filter(pl.col("bar_idx") <= 1)
    assert original.equals(changed)


def test_five_minute_aggregation_uses_observed_minutes_only() -> None:
    rows = _minute_rows().filter(pl.col("minute_idx") != 2)
    result = aggregate_five_minute_bars(rows).row(0, named=True)
    assert result["open"] == 100.0
    assert result["high"] == 104.5
    assert result["low"] == 99.5
    assert result["close"] == 104.25
    assert result["volume"] == 1 + 2 + 4 + 5
    assert result["observed_minutes"] == 4


def test_context_ends_at_decision_close_and_ignores_future(tmp_path: Path) -> None:
    date_count = 8
    bars = np.zeros(
        (date_count, EQUITY_COUNT, SESSION_BARS, len(BAR_FIELDS)), np.float32
    )
    synthetic = np.ones((date_count, EQUITY_COUNT, SESSION_BARS), dtype=bool)
    available = np.zeros_like(synthetic)
    timestamps = np.zeros((date_count, SESSION_BARS), dtype=np.int64)
    sequence = np.arange(date_count * SESSION_BARS, dtype=np.float32).reshape(
        date_count, SESSION_BARS
    )
    bars[:, 0, :, :4] = sequence[:, :, None] + 100.0
    bars[:, 0, :, 4] = 1.0
    synthetic[:, 0] = False
    available[:, 0] = True
    timestamps[:] = np.arange(date_count * SESSION_BARS, dtype=np.int64).reshape(
        date_count, SESSION_BARS
    )
    np.save(tmp_path / "bars.npy", bars)
    np.save(tmp_path / "synthetic.npy", synthetic)
    np.save(tmp_path / "available.npy", available)
    np.save(tmp_path / "bar_close_timestamp_ns.npy", timestamps)

    sidecar = BarSidecar(tmp_path)
    context = sidecar.context(7, 0, 0)
    assert context is not None
    assert context.bars.shape == (CONTEXT_BARS, len(BAR_FIELDS))
    assert context.timestamp_ns[-1] == timestamps[7, 2]
    before = context.bars.copy()
    mutable = np.load(tmp_path / "bars.npy", mmap_mode="r+")
    mutable[7, 0, 3, 3] = 1_000_000.0
    mutable.flush()
    del mutable
    reloaded = BarSidecar(tmp_path).context(7, 0, 0)
    assert reloaded is not None
    assert np.array_equal(reloaded.bars, before)


def test_context_seed_is_stable_and_identity_specific() -> None:
    first = stable_context_seed("Kronos-small", date(2024, 1, 2), 10, "ISIN:A")
    assert first == stable_context_seed("Kronos-small", date(2024, 1, 2), 10, "ISIN:A")
    assert first != stable_context_seed("Kronos-small", date(2024, 1, 2), 10, "ISIN:B")
    assert 0 <= first < 2**63


def _result(ic: float, momentum: float, correlation: float) -> dict[str, object]:
    return {
        "primary_ic": {"mean_folds": ic},
        "momentum_control_ic": {"mean_folds": momentum},
        "score_parent_spearman": {"mean_folds": correlation},
    }


def test_preregistered_decision_rule() -> None:
    assert _decision({"small": _result(0.0149, 0.0, 0.0)})["outcome"] == "kill"
    assert _decision({"small": _result(0.02, 0.02, 0.0)})["outcome"] == "kill"
    assert _decision({"small": _result(0.02, 0.01, 0.5)})["outcome"] == "park"
    assert (
        _decision({"small": _result(0.02, 0.01, 0.49)})["outcome"]
        == "eligible_for_separately_preregistered_k1"
    )
