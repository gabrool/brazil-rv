from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from brazil_rv.preprocessing import global_features as global_features_module
from brazil_rv.preprocessing import global_source as global_source_module
from brazil_rv.preprocessing.contract import (
    DECISION_GLOBAL_INDICES,
    GLOBAL_CONTEXT_SYMBOLS,
    GLOBAL_SESSION_MINUTES,
)
from brazil_rv.preprocessing.global_features import (
    build_global_grid,
    build_global_instrument_features,
)
from brazil_rv.preprocessing.global_source import (
    API_KEY_ENV,
    HISTORICAL_SCHEMAS,
    INSTRUMENT_ID_STYPE,
    NORMALIZED_COLUMNS,
    RequestRange,
    _validate_raw_acquisition,
    _with_mapping_changes,
    download_history,
    load_global_symbol,
    normalize_bars,
    request_plan,
    require_api_key,
    write_shadow_daily_chunks,
)
from brazil_rv.preprocessing.transforms import build_dynamic_features


B3_TIMEZONE = ZoneInfo("America/Sao_Paulo")


def _raw_frame(
    timestamps: list[datetime],
    *,
    raw_symbols: list[str] | None = None,
) -> pl.DataFrame:
    count = len(timestamps)
    symbols = raw_symbols or ["ESU6"] * count
    close = np.arange(count, dtype=np.float64) + 100.0
    return pl.DataFrame(
        {
            "ts_event_utc": timestamps,
            "instrument_id": np.arange(count, dtype=np.uint32) + 101,
            "symbol": symbols,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": np.arange(count, dtype=np.float64) + 10.0,
        }
    )


def _b3_utc(value: date, clock: time) -> datetime:
    return datetime.combine(value, clock, B3_TIMEZONE).astimezone(UTC)


def _global_history_frame(decision_date: date) -> pl.DataFrame:
    timestamps: list[datetime] = []
    for day_offset in range(25, 0, -1):
        session_date = decision_date - timedelta(days=day_offset)
        start = datetime.combine(session_date, time(18), UTC)
        timestamps.extend(start + timedelta(minutes=minute) for minute in range(31))
    timestamps.append(_b3_utc(decision_date - timedelta(days=1), time(16, 44)))
    current_start = _b3_utc(decision_date, time(4, 30))
    timestamps.extend(
        current_start + timedelta(minutes=minute) for minute in range(346)
    )
    raw = _raw_frame(timestamps).with_columns(
        pl.lit(101).cast(pl.UInt32).alias("instrument_id")
    )
    return normalize_bars(raw, "ES.v.0")


def test_normalized_schema_availability_and_mapping_identity() -> None:
    decision_date = date(2026, 1, 15)
    timestamps = [
        _b3_utc(decision_date, time(10, 14)),
        _b3_utc(decision_date, time(10, 15)),
    ]
    normalized = normalize_bars(_raw_frame(timestamps), "ES.v.0")
    assert normalized.schema["bar_end_utc"] == pl.Datetime("ns", "UTC")
    assert tuple(normalized.columns) == NORMALIZED_COLUMNS
    assert normalized["bar_end_utc"].to_list() == [
        timestamp + timedelta(minutes=1) for timestamp in timestamps
    ]
    assert normalized["global_slot"].to_list() == [0, 0]
    assert normalized["raw_symbol"].to_list() == ["ESU6", "ESU6"]

    raw, observed, _, index = build_global_grid(normalized, (decision_date,))
    cutoff = DECISION_GLOBAL_INDICES[0]
    assert cutoff == 345
    assert observed[0, cutoff - 1]
    assert observed[0, cutoff]
    assert observed[0, :cutoff].sum() == 1
    assert not observed[0, cutoff - 2]
    assert not raw[0, cutoff - 2].any()
    assert raw[0, :cutoff].shape == (345, 5)
    assert index.filter(pl.col("minute_idx") == cutoff - 1)["bar_end_utc"].item() == (
        timestamps[0] + timedelta(minutes=1)
    )

    future_changed = raw.copy()
    future_changed[0, cutoff:] = 1_000_000.0
    np.testing.assert_array_equal(raw[0, :cutoff], future_changed[0, :cutoff])


def test_definition_records_resolve_exact_outrights_and_fail_closed() -> None:
    timestamps = [
        datetime(2026, 1, 2, 14, 0, tzinfo=UTC),
        datetime(2026, 1, 2, 14, 1, tzinfo=UTC),
    ]
    bars = _raw_frame(timestamps).drop("symbol")
    definitions = pl.DataFrame(
        {
            "instrument_id": [101, 102],
            "raw_symbol": ["ESH6", "ESM6"],
            "expiration": [
                datetime(2026, 3, 20, tzinfo=UTC),
                datetime(2026, 6, 19, tzinfo=UTC),
            ],
        }
    )
    normalized = normalize_bars(bars, "ES.v.0", definitions)
    assert normalized["raw_symbol"].to_list() == ["ESH6", "ESM6"]
    assert normalized["expiration_utc"].to_list() == definitions["expiration"].to_list()

    with pytest.raises(ValueError, match="missing an instrument mapping"):
        normalize_bars(bars, "ES.v.0", definitions.head(1))

    ambiguous = pl.concat(
        [
            definitions,
            definitions.head(1).with_columns(pl.lit("ESZ6").alias("raw_symbol")),
        ]
    )
    with pytest.raises(ValueError, match="ambiguous outright mapping"):
        normalize_bars(bars, "ES.v.0", ambiguous)


def test_global_grid_handles_b3_alignment_across_us_dst() -> None:
    market_dates = (date(2026, 1, 15), date(2026, 7, 15))
    timestamps = [_b3_utc(value, time(10, 14)) for value in market_dates]
    assert timestamps[0].hour == timestamps[1].hour
    normalized = normalize_bars(_raw_frame(timestamps), "ES.v.0")
    _, observed, _, index = build_global_grid(normalized, market_dates)
    assert observed[:, DECISION_GLOBAL_INDICES[0] - 1].all()
    assert index["minute_idx"].to_list() == [344, 344]


def test_first_and_last_global_windows_are_exact() -> None:
    first, last = DECISION_GLOBAL_INDICES[0], DECISION_GLOBAL_INDICES[-1]
    assert first - 345 == 0
    assert last - 345 == 270
    assert first // 5 == 69
    assert last <= GLOBAL_SESSION_MINUTES


def test_decision_features_use_only_the_eligible_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_date = date(2026, 2, 2)
    market_dates = (decision_date - timedelta(days=1), decision_date)
    baseline_frame = _global_history_frame(decision_date)

    def build(frame: pl.DataFrame):
        monkeypatch.setattr(
            global_features_module,
            "load_global_symbol",
            lambda _source, _symbol: frame,
        )
        return build_global_instrument_features(Path(), "ES.v.0", market_dates)

    def change_prices(
        frame: pl.DataFrame, timestamp: datetime, factor: float
    ) -> pl.DataFrame:
        return frame.with_columns(
            *(
                pl.when(pl.col("ts_event_utc") == timestamp)
                .then(pl.col(name) * factor)
                .otherwise(pl.col(name))
                .alias(name)
                for name in ("open", "high", "low", "close")
            )
        )

    baseline = build(baseline_frame)
    date_idx = 1
    cutoff = DECISION_GLOBAL_INDICES[0]
    assert baseline.data_ready[date_idx, 0]

    future = build(
        change_prices(baseline_frame, _b3_utc(decision_date, time(10, 15)), 1.1)
    )
    np.testing.assert_array_equal(
        baseline.dynamic[date_idx, :cutoff], future.dynamic[date_idx, :cutoff]
    )
    np.testing.assert_array_equal(baseline.slow[date_idx, 0], future.slow[date_idx, 0])

    eligible = build(
        change_prices(baseline_frame, _b3_utc(decision_date, time(10, 14)), 1.1)
    )
    assert not np.array_equal(
        baseline.dynamic[date_idx, :cutoff], eligible.dynamic[date_idx, :cutoff]
    )
    assert baseline.slow[date_idx, 0, 1] != eligible.slow[date_idx, 0, 1]


def test_roll_boundary_is_recorded_and_cross_roll_returns_are_suppressed() -> None:
    first = normalize_bars(
        _raw_frame([datetime(2026, 1, 2, 14, 0, tzinfo=UTC)]), "ES.v.0"
    )
    second = normalize_bars(
        _raw_frame(
            [datetime(2026, 1, 2, 14, 1, tzinfo=UTC)],
            raw_symbols=["ESH7"],
        ),
        "ES.v.0",
    )
    combined = _with_mapping_changes(
        second, previous_raw_symbol=str(first.item(-1, "raw_symbol"))
    )
    assert combined["mapping_changed"].to_list() == [True]

    raw = np.zeros((1, 20, 5), dtype=np.float64)
    close = np.linspace(100.0, 101.9, 20)
    raw[0, :, 0] = close
    raw[0, :, 1] = close + 0.1
    raw[0, :, 2] = close - 0.1
    raw[0, :, 3] = close
    raw[0, :, 4] = 10.0
    observed = np.ones((1, 20), dtype=bool)
    mapping_changed = np.zeros_like(observed)
    mapping_changed[0, 15] = True
    dynamic, validity = build_dynamic_features(
        raw,
        observed,
        np.ones(1, dtype=bool),
        np.full(1, 0.01),
        is_rate=False,
        mapping_changed=mapping_changed,
    )
    assert not dynamic[0, 15, :4].any()
    assert dynamic[0, 15, 7] == 0.0
    assert not validity[0, 15, 0]


def test_shadow_chunks_match_historical_schema_and_resume(tmp_path: Path) -> None:
    first_date = date(2026, 1, 2)
    second_date = date(2026, 1, 3)
    first_raw = _raw_frame([datetime(2026, 1, 2, 14, 0, tzinfo=UTC)])
    second_raw = _raw_frame(
        [datetime(2026, 1, 3, 14, 0, tzinfo=UTC)],
        raw_symbols=["ESH7"],
    )
    expected_columns = tuple(normalize_bars(first_raw, "ES.v.0").columns)
    write_shadow_daily_chunks(first_raw, "ES.v.0", tmp_path)
    write_shadow_daily_chunks(first_raw, "ES.v.0", tmp_path)
    write_shadow_daily_chunks(second_raw, "ES.v.0", tmp_path)

    stored = load_global_symbol(tmp_path, "ES.v.0")
    assert tuple(stored.columns) == expected_columns == NORMALIZED_COLUMNS
    assert stored.height == 2
    assert stored["mapping_changed"].to_list() == [False, True]
    assert (tmp_path / "bars/slot=00" / f"date={first_date}.parquet").is_file()
    assert (tmp_path / "bars/slot=00" / f"date={second_date}.parquet").is_file()
    assert not tuple(tmp_path.rglob("*.tmp"))

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "shadow"
    assert manifest["symbols"] == list(GLOBAL_CONTEXT_SYMBOLS)
    assert manifest["row_count"] == 2
    assert manifest["source_hashes"] == {}


class _FakeMetadata:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_cost(self, **kwargs: object) -> float:
        self.calls.append(kwargs)
        return 2.0 if kwargs["schema"] == "ohlcv-1m" else 0.5


class _FakeTimeseries:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_on_call = fail_on_call

    def get_range(self, *, path: Path, **kwargs: object) -> None:
        if kwargs.get("stype_out") != INSTRUMENT_ID_STYPE:
            raise AssertionError("unsupported historical symbology")
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("provider failure")
        Path(path).write_bytes(
            f"{kwargs['schema']}:{kwargs['symbols']}:{kwargs['start']}".encode()
        )


class _FakeHistorical:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.metadata = _FakeMetadata()
        self.timeseries = _FakeTimeseries(fail_on_call)


def _fake_dbn_metadata(
    plan: tuple[object, ...],
    *,
    wrong_path: Path | None = None,
):
    by_path = {request.data_path.resolve(): request for request in plan}

    def load(path: Path) -> dict[str, object]:
        request = by_path[path.resolve()]
        schema = "trades" if path == wrong_path else request.schema
        return {
            "dataset": global_source_module.GLOBAL_DATASET,
            "schema": schema,
            "stype_in": global_source_module.CONTINUOUS_STYPE,
            "stype_out": INSTRUMENT_ID_STYPE,
            "start": global_source_module._date_ns(request.start),
            "end": global_source_module._date_ns(request.end),
            "symbols": [request.continuous_symbol],
            "partial": [],
            "not_found": [],
            "mappings": {
                request.continuous_symbol: [
                    {
                        "start_date": str(request.start),
                        "end_date": str(request.end),
                        "symbol": "101",
                    }
                ]
            },
        }

    return load


def _fake_acquisition(
    raw_dir: Path,
    request: RequestRange,
) -> tuple[object, ...]:
    return download_history(
        _FakeHistorical(),
        request,
        raw_dir,
        confirmed_paid_download=True,
        symbols=("ES.v.0",),
    )


def test_supported_symbology_and_cost_plan_cover_every_remaining_request(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = RequestRange(date(2026, 1, 2), date(2026, 2, 4))
    client = _FakeHistorical()
    expected_plan = request_plan(request, tmp_path, ("ES.v.0",))

    with pytest.raises(RuntimeError, match="confirm-paid-download"):
        download_history(
            client,
            request,
            tmp_path,
            confirmed_paid_download=False,
            symbols=("ES.v.0",),
        )
    estimate = json.loads(capsys.readouterr().out.strip())
    assert estimate == {
        "remaining_request_count": 4,
        "by_schema": {
            "ohlcv-1m": {"request_count": 2, "estimated_cost_usd": 4.0},
            "definition": {"request_count": 2, "estimated_cost_usd": 1.0},
        },
        "total_usd": 5.0,
    }
    assert len(client.metadata.calls) == len(expected_plan)
    assert not client.timeseries.calls

    plan = download_history(
        client,
        request,
        tmp_path,
        confirmed_paid_download=True,
        symbols=("ES.v.0",),
    )
    assert [call["schema"] for call in client.timeseries.calls] == [
        request.schema for request in expected_plan
    ]
    assert all(
        call["stype_in"] == "continuous" and call["stype_out"] == "instrument_id"
        for call in client.timeseries.calls
    )
    monkeypatch.setattr(
        global_source_module,
        "_dbn_metadata",
        _fake_dbn_metadata(plan),
    )
    validated, _ = _validate_raw_acquisition(tmp_path, request, symbols=("ES.v.0",))
    assert validated == plan

    cost_calls = len(client.metadata.calls)
    download_calls = len(client.timeseries.calls)
    download_history(
        client,
        request,
        tmp_path,
        confirmed_paid_download=True,
        symbols=("ES.v.0",),
    )
    assert len(client.metadata.calls) == cost_calls
    assert len(client.timeseries.calls) == download_calls
    assert "super-secret-key" not in (tmp_path / "manifest.json").read_text(
        encoding="utf-8"
    )


def test_resume_estimates_and_downloads_only_unfinished_requests(
    tmp_path: Path,
) -> None:
    request = RequestRange(date(2026, 1, 2), date(2026, 2, 4))
    first = _FakeHistorical(fail_on_call=3)
    with pytest.raises(RuntimeError, match="historical download failed"):
        download_history(
            first,
            request,
            tmp_path,
            confirmed_paid_download=True,
            symbols=("ES.v.0",),
        )
    assert len(first.metadata.calls) == 4
    assert not tuple(tmp_path.rglob("*.partial"))
    assert not (tmp_path / "manifest.json").exists()

    resumed = _FakeHistorical()
    download_history(
        resumed,
        request,
        tmp_path,
        confirmed_paid_download=True,
        symbols=("ES.v.0",),
    )
    assert len(resumed.metadata.calls) == 2
    assert len(resumed.timeseries.calls) == 2
    assert {call["schema"] for call in resumed.metadata.calls} == set(
        HISTORICAL_SCHEMAS
    )


@pytest.mark.parametrize(
    "issue",
    ("missing", "extra", "duplicate", "gap", "overlap", "hash", "descriptor"),
)
def test_raw_manifest_rejects_incomplete_or_inconsistent_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    issue: str,
) -> None:
    request = RequestRange(date(2026, 1, 2), date(2026, 2, 4))
    plan = _fake_acquisition(tmp_path, request)
    monkeypatch.setattr(
        global_source_module,
        "_dbn_metadata",
        _fake_dbn_metadata(plan),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if issue == "missing":
        plan[0].data_path.unlink()
    elif issue == "extra":
        extra = tmp_path / "requests" / "unexpected.json"
        extra.write_text("{}", encoding="utf-8")
        extra_data = tmp_path / "bars" / "unexpected.dbn.zst"
        extra_data.write_bytes(b"unexpected")
    elif issue == "duplicate":
        manifest["requests"].append(dict(manifest["requests"][0]))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif issue == "gap":
        manifest["requests"][2]["start"] = str(
            date.fromisoformat(manifest["requests"][2]["start"]) + timedelta(days=1)
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif issue == "overlap":
        manifest["requests"][2]["start"] = str(
            date.fromisoformat(manifest["requests"][2]["start"]) - timedelta(days=1)
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif issue == "hash":
        plan[0].data_path.write_bytes(b"changed")
    else:
        descriptor = json.loads(plan[0].descriptor_path.read_text(encoding="utf-8"))
        descriptor["dataset"] = "wrong"
        plan[0].descriptor_path.write_text(json.dumps(descriptor), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        _validate_raw_acquisition(tmp_path, request, symbols=("ES.v.0",))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider", "wrong"),
        ("dataset", "wrong"),
        ("schemas", ["ohlcv-1m"]),
        ("stype_out", "raw_symbol"),
        ("symbols", ["NQ.v.0"]),
        ("requested_end", "2026-02-05"),
    ),
)
def test_raw_manifest_rejects_contract_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    request = RequestRange(date(2026, 1, 2), date(2026, 2, 4))
    plan = _fake_acquisition(tmp_path, request)
    monkeypatch.setattr(
        global_source_module,
        "_dbn_metadata",
        _fake_dbn_metadata(plan),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest contract mismatch"):
        _validate_raw_acquisition(tmp_path, request, symbols=("ES.v.0",))


def test_raw_manifest_rejects_dbn_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = RequestRange(date(2026, 1, 2), date(2026, 1, 3))
    plan = _fake_acquisition(tmp_path, request)
    monkeypatch.setattr(
        global_source_module,
        "_dbn_metadata",
        _fake_dbn_metadata(plan, wrong_path=plan[0].data_path),
    )
    with pytest.raises(ValueError, match="DBN request metadata mismatch"):
        _validate_raw_acquisition(tmp_path, request, symbols=("ES.v.0",))


def test_credentials_and_bad_source_rows_fail_cleanly() -> None:
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        require_api_key({})
    assert require_api_key({API_KEY_ENV: "super-secret-key"}) == "super-secret-key"

    malformed = _raw_frame([datetime(2026, 1, 2, 14, 0, tzinfo=UTC)]).with_columns(
        pl.lit(-1.0).alias("volume")
    )
    with pytest.raises(ValueError, match="Malformed OHLCV"):
        normalize_bars(malformed, "ES.v.0")

    unresolved = _raw_frame([datetime(2026, 1, 2, 14, 0, tzinfo=UTC)]).with_columns(
        pl.lit(None, dtype=pl.String).alias("symbol")
    )
    with pytest.raises(ValueError, match="Malformed OHLCV"):
        normalize_bars(unresolved, "ES.v.0")

    with pytest.raises(ValueError, match="Unsupported continuous symbol"):
        normalize_bars(_raw_frame([datetime(2026, 1, 2, 14, 0, tzinfo=UTC)]), "BAD")

    duplicate = _raw_frame([datetime(2026, 1, 2, 14, 0, tzinfo=UTC)])
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_bars(pl.concat([duplicate, duplicate]), "ES.v.0")

    malformed_ohlc = _raw_frame([datetime(2026, 1, 2, 14, 0, tzinfo=UTC)]).with_columns(
        pl.lit(0.5).alias("high")
    )
    with pytest.raises(ValueError, match="Malformed OHLCV"):
        normalize_bars(malformed_ohlc, "ES.v.0")
