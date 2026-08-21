from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

import brazil_rv.preprocessing.index_rebalance as module
from brazil_rv.preprocessing.index_rebalance import (
    Disclosure,
    INDEXES,
    Portfolio,
    _first_available_decision,
    build_frame,
    parse_composition,
)


class _Sheet:
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def to_polars(self) -> pl.DataFrame:
        return self._frame


class _Workbook:
    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    def load_sheet(self, _name: str, *, header_row: None) -> _Sheet:
        return _Sheet(self._frame)


def test_http_timestamp_activates_strictly_after_release() -> None:
    sessions = [date(2023, 8, 1), date(2023, 8, 2)]
    assert _first_available_decision(
        datetime(2023, 8, 1, 15, 25, tzinfo=timezone.utc), sessions
    ) == (date(2023, 8, 1), 27)
    assert _first_available_decision(
        datetime(2023, 8, 1, 18, 0, tzinfo=timezone.utc), sessions
    ) == (date(2023, 8, 2), 0)


def test_composition_parser_does_not_multiply_numeric_quantities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pl.DataFrame(
        {
            "column_0": [f"AA{index:02d}" for index in range(20)],
            "column_1": ["issuer"] * 20,
            "column_2": ["ON NM"] * 20,
            "column_3": [1_000_000.0] * 20,
            "column_4": [5.0] * 20,
        }
    )
    monkeypatch.setattr(module, "_composition_workbook", lambda _path: b"workbook")
    monkeypatch.setattr(
        module.fastexcel,
        "read_excel",
        lambda _source: _Workbook(frame),
    )
    portfolios = parse_composition(Path("ignored.xlsx"))
    assert {item.index for item in portfolios} == set(INDEXES)
    assert all(item.quantities["AA00"] == 1_000_000.0 for item in portfolios)
    assert all(sum(item.weights.values()) == pytest.approx(1.0) for item in portfolios)


def _portfolio(index: str, first_weight: float) -> Portfolio:
    return Portfolio(
        index,
        {"AAA3": first_weight, "BBB3": 1.0 - first_weight},
        {"AAA3": 100.0 * first_weight, "BBB3": 100.0 * (1.0 - first_weight)},
    )


def test_frame_uses_prior_adv_and_preserves_preview_delta_at_effective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = [date(2023, 11, 1) + timedelta(days=offset) for offset in range(40)]
    initial_date = sessions[1]
    preview_date = sessions[25]
    effective_date = sessions[28]
    first_id = "ISIN:FIRST"
    second_id = "ISIN:SECOND"
    first = pl.DataFrame(
        {
            "trade_date": sessions,
            "security_id": [first_id] * len(sessions),
            "ticker": ["AAA3"] * len(sessions),
            "close_brl": [10.0] * len(sessions),
            "volume_brl": [100.0] * len(sessions),
        }
    )
    second = pl.DataFrame(
        {
            "trade_date": sessions,
            "security_id": [second_id] * len(sessions),
            "ticker": ["BBB3"] * len(sessions),
            "close_brl": [10.0] * len(sessions),
            "volume_brl": [300.0] * len(sessions),
        }
    )
    cotahist = first.vstack(second).sort("trade_date", "ticker")
    initial = tmp_path / "initial.xlsx"
    preview = tmp_path / "preview.xlsx"
    effective = tmp_path / "effective.xlsx"
    for path in (initial, preview, effective):
        path.write_bytes(path.stem.encode())
    disclosures = [
        Disclosure(
            initial_date,
            initial_date,
            "effective",
            datetime.combine(initial_date, time(12), timezone.utc),
            initial,
        ),
        Disclosure(
            preview_date,
            effective_date,
            "preview_3",
            datetime.combine(preview_date, time(12), timezone.utc),
            preview,
        ),
        Disclosure(
            effective_date,
            effective_date,
            "effective",
            datetime.combine(effective_date, time(12), timezone.utc),
            effective,
        ),
    ]

    def fake_parse(path: Path) -> list[Portfolio]:
        first_weight = 0.0 if path == preview else 0.9
        return [_portfolio(index, first_weight) for index in INDEXES]

    monkeypatch.setattr(module, "parse_composition", fake_parse)
    frame, _ = build_frame(
        disclosures,
        sessions,
        cotahist,
        [first_id, second_id],
        available_start=preview_date,
        available_end=sessions[-1],
    )
    preview_row = frame.filter(
        (pl.col("available_date") == preview_date) & (pl.col("security_id") == first_id)
    ).row(0, named=True)
    assert preview_row["ibov_preview_delete"] == 1.0
    assert preview_row["ibov_preview_pressure"] < 0.0
    post_row = frame.filter(
        (pl.col("available_date") == sessions[29]) & (pl.col("security_id") == first_id)
    ).row(0, named=True)
    assert post_row["ibov_post_effective_reversal"] > 0.0

    mutated = cotahist.with_columns(
        pl.when(pl.col("trade_date") > preview_date)
        .then(pl.col("volume_brl") * 1_000_000.0)
        .otherwise(pl.col("volume_brl"))
        .alias("volume_brl")
    )
    replay, _ = build_frame(
        disclosures,
        sessions,
        mutated,
        [first_id, second_id],
        available_start=preview_date,
        available_end=sessions[-1],
    )
    replay_preview = replay.filter(
        (pl.col("available_date") == preview_date) & (pl.col("security_id") == first_id)
    ).row(0, named=True)
    assert (
        replay_preview["ibov_preview_pressure"] == preview_row["ibov_preview_pressure"]
    )
