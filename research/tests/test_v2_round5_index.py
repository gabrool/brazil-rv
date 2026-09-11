from datetime import UTC, date, datetime

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.round5_index import (
    PORTFOLIO_SCHEMA,
    _identity_at,
    linked_assets,
    parse_bdi_tables,
    pressure_panel,
    publication_date,
)


def test_bdi_keeps_effective_and_preview_separate_and_parses_brazilian_units():
    tables, dates = parse_bdi_tables(
        [
            "Composição das Carteiras de Índices\n"
            "abertura dos negócios do dia 01/04/2024\nIBOVESPA\n"
            "ABCD3 COMPANY ON 1.200.000 4,2500\nParticipação total: 100\n",
            "Prévia das Carteiras Teóricas de Índices\n"
            "abertura dos negócios do dia  01/04/2024\nIBOVESPA\n"
            "ABCD3 COMPANY ON 1.300.000 5,7500\nParticipação total: 100\n"
            "IBRX\nABCD3 COMPANY ON 2.300.000 6,7500\n",
        ]
    )
    assert tables["effective", "IBOV"]["ABCD3"]["weight_fraction"] == 0.0425
    assert tables["preview", "IBOV"]["ABCD3"]["quantity"] == 1_300_000
    assert tables["preview", "IBXX"]["ABCD3"]["weight_fraction"] == 0.0675
    assert dates == {"effective": ["01/04/2024"], "preview": ["01/04/2024"]}


def test_publication_date_and_explicit_safe_link_resolution():
    page = '<small>01/04/2024</small><a href="https://mail/safe?url=https%3A%2F%2Fb3.test%2Fx.xlsx&amp;data=x">x</a>'
    assert publication_date(page) == date(2024, 4, 1)
    assert linked_assets(page, "https://b3.test/page") == {"https://b3.test/x.xlsx"}
    with pytest.raises(ValueError, match="publication date"):
        publication_date('<meta name="last-modified" content="2024-04-01">')


def _cash():
    return pl.DataFrame(
        {
            "source_trade_date": [date(2024, 1, 2), date(2024, 4, 1), date(2024, 4, 1)],
            "ticker": ["OLD3", "NEW3", "BBBB3"],
            "isin": ["ISIN_A", "ISIN_A", "ISIN_B"],
        }
    )


def _snapshots(second_preview=True):
    rows = []
    events = [
        (date(2024, 1, 2), date(2024, 1, 2), "effective", "OLD3", 0.1),
        (date(2024, 4, 1), date(2024, 4, 5), "preview_1", "NEW3", 0.2),
    ]
    if second_preview:
        events.append((date(2024, 4, 3), date(2024, 4, 5), "preview_2", "NEW3", 0.15))
    for day, effective, stage, ticker, weight in events:
        for index in ("IBOV", "IBXX", "SMLL"):
            rows.append(
                {
                    "disclosure_date": day,
                    "effective_date": effective,
                    "stage": stage,
                    "available_at": datetime.combine(day, datetime.min.time(), UTC),
                    "index": index,
                    "ticker": ticker,
                    "weight_fraction": weight,
                    "quantity": 1000.0,
                    "source_file": stage,
                }
            )
    return pl.DataFrame(rows, schema=PORTFOLIO_SCHEMA)


def _panel(snapshots, cash=None):
    sessions = [date(2024, 4, day) for day in range(1, 6)]
    return pressure_panel(
        snapshots,
        _cash() if cash is None else cash,
        sessions,
        ("ISIN_A", "ISIN_B"),
        np.ones((5, 2), dtype=bool),
        np.full((5, 2), 20e6),
    )[0]


def test_pressure_uses_dated_identity_prior_effective_and_native_units():
    frame = _panel(_snapshots())
    a = frame.filter(pl.col("isin") == "ISIN_A")
    assert a["index_pressure"].to_list() == pytest.approx([6, 4.5, 1.5, 0.75])
    assert a["index_event_age"].to_list() == [0, 1, 0, 1]
    assert (
        frame.filter(pl.col("isin") == "ISIN_B")["index_pressure"].to_list() == [0] * 4
    )
    assert frame["date"].max() == date(2024, 4, 4)  # Stops before effective opening.


def test_future_preview_and_ticker_mutations_do_not_change_earlier_pressure():
    source = _snapshots()
    before = _panel(source).filter(pl.col("date") < date(2024, 4, 3))
    changed = source.with_columns(
        pl.when(pl.col("disclosure_date") == date(2024, 4, 3))
        .then(0.999)
        .otherwise(pl.col("weight_fraction"))
        .alias("weight_fraction")
    )
    future_cash = pl.concat(
        [
            _cash(),
            pl.DataFrame(
                {
                    "source_trade_date": [date(2024, 4, 4)],
                    "ticker": ["NEW3"],
                    "isin": ["FUTURE_ID"],
                }
            ),
        ]
    )
    after = _panel(changed, future_cash).filter(pl.col("date") < date(2024, 4, 3))
    assert before.equals(after)
    assert _identity_at(future_cash, date(2024, 4, 1))["NEW3"] == "ISIN_A"


def test_date_only_preview_is_not_visible_on_its_publication_date():
    source = _snapshots(False).with_columns(
        pl.when(pl.col("stage") == "preview_1")
        .then(datetime(2024, 4, 2, 2, 59, 59, tzinfo=UTC))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    a = _panel(source).filter(pl.col("isin") == "ISIN_A")
    assert a["date"].min() == date(2024, 4, 2)
    assert a["index_event_age"].to_list() == [0, 1, 2]


def test_missing_prior_snapshot_or_identity_is_not_zero_information():
    previews = _snapshots().filter(pl.col("stage") != "effective")
    assert _panel(previews).is_empty()
    assert _panel(_snapshots(), _cash().filter(pl.col("ticker") != "OLD3")).is_empty()


def test_missing_effective_cycle_does_not_reuse_an_older_composition():
    source = _snapshots().with_columns(
        pl.when(pl.col("stage") == "effective")
        .then(pl.lit(date(2023, 9, 4)))
        .otherwise(pl.col("effective_date"))
        .alias("effective_date")
    )
    assert _panel(source).is_empty()


def test_official_dated_workbook_without_bdi_uses_next_decision(tmp_path, monkeypatch):
    from brazil_rv.preprocessing.index_rebalance import Portfolio
    from brazil_rv.v2 import round5_index as module
    from brazil_rv.v2.artifacts import sha256_file

    page = tmp_path / "page.html"
    page.write_text(
        '<small>01/04/2024</small><a href="https://b3.test/composition.xlsx">data</a>'
    )
    asset = tmp_path / "composition.xlsx"
    asset.write_bytes(b"fixture")
    monkeypatch.setattr(
        module,
        "parse_composition",
        lambda _: [
            Portfolio(index=i, weights={"AAAA3": 1.0}, quantities={"AAAA3": 1.0})
            for i in ("IBOV", "IBXX", "SMLL")
        ],
    )
    rows, audit = module.normalize_events(
        [
            {
                "disclosure_date": "2024-04-01",
                "effective_date": "2024-05-06",
                "stage": "preview_1",
                "page": {
                    "path": str(page),
                    "sha256": sha256_file(page),
                    "url": "https://b3.test/news",
                },
                "assets": [
                    {
                        "path": str(asset),
                        "sha256": sha256_file(asset),
                        "url": "https://b3.test/composition.xlsx",
                    }
                ],
            }
        ]
    )
    assert rows["available_at"][0] == datetime(2024, 4, 2, 2, 59, 59, tzinfo=UTC)
    assert audit[0]["bdi_text_sha256"] is None
    assert (
        audit[0]["availability_evidence"]
        == "official_announcement_date_only_end_of_day_bound"
    )


def test_unknown_new_preview_stops_previous_preview_instead_of_hiding_update():
    source = _snapshots().with_columns(
        pl.when(pl.col("stage") == "preview_2")
        .then(pl.lit("UNKNOWN3"))
        .otherwise(pl.col("ticker"))
        .alias("ticker")
    )
    frame = _panel(source)
    assert frame["date"].max() == date(2024, 4, 2)


def test_future_effective_calendar_requires_explicit_known_sessions():
    source = _snapshots(False).with_columns(
        pl.when(pl.col("stage") == "preview_1")
        .then(date(2024, 4, 8))
        .otherwise(pl.col("effective_date"))
        .alias("effective_date")
    )
    with pytest.raises(ValueError, match="verified session calendar"):
        _panel(source)
    frame, _ = pressure_panel(
        source,
        _cash(),
        [date(2024, 4, d) for d in range(1, 6)],
        ("ISIN_A",),
        np.ones((5, 1), bool),
        np.full((5, 1), 20e6),
        future_sessions=[date(2024, 4, 8)],
    )
    assert frame["index_pressure"].to_list() == pytest.approx([7.5, 6, 4.5, 3, 1.5])
