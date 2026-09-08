from __future__ import annotations

import json
import shutil
import sys
import zipfile
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.build_store import (
    _validate_cotahist_parse_audit,
    build_daily_store,
)
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.contract import (
    INTRADAY_DAILY_FEATURES,
    RAW_PATIENCE_SCHEMA,
    SCORE_ARTIFACT_SCHEMA,
    SIDECAR_FEATURES,
    SLOW_FEATURES,
)
from brazil_rv.v2.corporate_actions import normalize_yfinance_actions
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.decision_clock import SessionDefinition
from brazil_rv.v2.evaluate import (
    EVALUATION_SCHEMA,
    evaluate_scores,
    write_evaluation_report,
)
from brazil_rv.v2.intraday_features import NATIVE_FAST_FEATURES
from brazil_rv.v2.lending_archive import LendingBorrowPanels
from brazil_rv.v2.research_rounds import (
    RESEARCH_SCORE_SCHEMA,
    _evaluation_from_artifacts,
    _persist_scores,
)
from brazil_rv.v2.run_many import load_plan
from brazil_rv.v2.score import score_checkpoint_artifact
from brazil_rv.v2.sidecars import SidecarResult
from brazil_rv.v2.splits import development_folds
from brazil_rv.v2.store import STORE_SCHEMA
from brazil_rv.v2.train import DatePairBatchSampler, train_stage
from brazil_rv.v2.validate_pipeline import _evaluation_inputs

_COLLECTOR_SCRIPTS = Path(__file__).resolve().parents[2] / "collector" / "scripts"
sys.path.insert(0, str(_COLLECTOR_SCRIPTS))
from parse_b3_cotahist import RECORD_LENGTH, parse_year  # noqa: E402


_DATE_AXIS = np.arange(
    np.datetime64("2022-11-01"),
    np.datetime64("2024-12-31"),
    dtype="datetime64[D]",
)
_DATE_AXIS = _DATE_AXIS[np.is_busday(_DATE_AXIS)]
_DAY_COUNT = int(_DATE_AXIS.size)
# The production ledger selects 30 names per side.  Sixty deterministic
# identities are the smallest fixture that exercises real order generation.
_NAME_COUNT = 60
_MINUTE_COUNT = 20
_DECISION_OFFSET = 10
_MUTATION_DATE_INDEX = 66


def _fixture_dates() -> list[date]:
    return _DATE_AXIS.astype(object).tolist()


def _fixture_market(
    *, mutate_post_decision: bool
) -> tuple[
    dict[str, np.ndarray],
    tuple[str, ...],
    tuple[str, ...],
    tuple[SessionDefinition, ...],
    pl.DataFrame,
]:
    dates = _fixture_dates()
    isins = tuple(f"BRTEST{index:02d}NOR1" for index in range(_NAME_COUNT))
    tickers = tuple(f"T{index:03d}3" for index in range(_NAME_COUNT))
    name_index = np.arange(_NAME_COUNT, dtype=np.float64)
    closes = np.empty((_DAY_COUNT, _NAME_COUNT), dtype=np.float64)
    closes[0] = 25.0 + 0.15 * name_index
    for day_index in range(1, _DAY_COUNT):
        log_return = 0.0015 * np.sin(0.31 * day_index + 0.17 * name_index) + 0.00002 * (
            name_index - (_NAME_COUNT - 1) / 2.0
        )
        closes[day_index] = closes[day_index - 1] * np.exp(log_return)
    daily_open = np.empty_like(closes)
    daily_open[0] = closes[0] * 0.999
    daily_open[1:] = closes[:-1] * np.exp(
        0.0003
        * np.cos(
            0.23 * np.arange(1, _DAY_COUNT, dtype=np.float64)[:, None]
            + 0.13 * name_index[None, :]
        )
    )

    fraction = np.linspace(0.0, 1.0, _MINUTE_COUNT, dtype=np.float64)
    minute_close = np.exp(
        np.log(daily_open)[..., None]
        + fraction[None, None, :] * (np.log(closes) - np.log(daily_open))[..., None]
    )
    minute_open = np.empty_like(minute_close)
    minute_open[..., 0] = daily_open
    minute_open[..., 1:] = minute_close[..., :-1]
    if mutate_post_decision:
        factors = np.linspace(
            1.01,
            1.06,
            _MINUTE_COUNT - _DECISION_OFFSET - 1,
            dtype=np.float64,
        )
        minute_close[_MUTATION_DATE_INDEX, :, _DECISION_OFFSET + 1 :] *= factors
        minute_open[_MUTATION_DATE_INDEX, :, _DECISION_OFFSET + 1] = minute_close[
            _MUTATION_DATE_INDEX, :, _DECISION_OFFSET
        ]
        minute_open[_MUTATION_DATE_INDEX, :, _DECISION_OFFSET + 2 :] = minute_close[
            _MUTATION_DATE_INDEX, :, _DECISION_OFFSET + 1 : -1
        ]
    minute_high = np.maximum(minute_open, minute_close) * 1.0003
    minute_low = np.minimum(minute_open, minute_close) * 0.9997
    closes = minute_close[..., -1].copy()
    daily_high = minute_high.max(axis=2)
    daily_low = minute_low.min(axis=2)
    daily_volume = (
        3_000_000.0
        + 10_000.0 * name_index[None, :]
        + 1_000.0 * np.arange(_DAY_COUNT, dtype=np.float64)[:, None]
    )
    minute_volume = np.broadcast_to(
        daily_volume[..., None] / _MINUTE_COUNT, minute_close.shape
    ).copy()
    schedule = tuple(
        SessionDefinition(
            trade_date=day,
            continuous_open=time(15, 35),
            decision_time=time(15, 45),
            continuous_close=time(15, 55),
            auction_close=time(16, 0),
            source="t24_deterministic_fixture",
        )
        for day in dates
    )
    audit = pl.DataFrame(
        {
            "isin": list(isins),
            "first_date": [dates[0]] * _NAME_COUNT,
            "last_date": [dates[-1]] * _NAME_COUNT,
            "status": ["downloaded"] * _NAME_COUNT,
            "action_rows": [0] * _NAME_COUNT,
            "economic_terms_complete": [True] * _NAME_COUNT,
        }
    )
    return (
        {
            "daily_open": daily_open,
            "daily_high": daily_high,
            "daily_low": daily_low,
            "daily_close": closes,
            "daily_volume": daily_volume,
            "minute_open": minute_open,
            "minute_high": minute_high,
            "minute_low": minute_low,
            "minute_close": minute_close,
            "minute_volume": minute_volume,
        },
        isins,
        tickers,
        schedule,
        audit,
    )


def _put_field(line: bytearray, start: int, stop: int, value: str) -> None:
    width = stop - start
    line[start:stop] = value.ljust(width)[:width].encode("latin-1")


def _cotahist_quote(
    *,
    trade_date: date,
    ticker: str,
    isin: str,
    open_brl: float,
    high_brl: float,
    low_brl: float,
    close_brl: float,
    volume_brl: float,
    trades: int,
    quantity: int,
) -> bytes:
    line = bytearray(b" " * RECORD_LENGTH)
    _put_field(line, 0, 2, "01")
    _put_field(line, 2, 10, trade_date.strftime("%Y%m%d"))
    _put_field(line, 10, 12, "02")
    _put_field(line, 12, 24, ticker)
    _put_field(line, 24, 27, "010")
    _put_field(line, 27, 39, "T24 ISSUER")
    _put_field(line, 39, 49, "ON")
    _put_field(line, 52, 56, "R$")
    average_brl = (high_brl + low_brl + close_brl) / 3.0
    prices = (
        open_brl,
        high_brl,
        low_brl,
        average_brl,
        close_brl,
        close_brl,
        close_brl,
    )
    for (start, stop), value in zip(
        (
            (56, 69),
            (69, 82),
            (82, 95),
            (95, 108),
            (108, 121),
            (121, 134),
            (134, 147),
        ),
        prices,
        strict=True,
    ):
        _put_field(line, start, stop, str(round(value * 100.0)).zfill(stop - start))
    _put_field(line, 147, 152, str(trades).zfill(5))
    _put_field(line, 152, 170, str(quantity).zfill(18))
    _put_field(line, 170, 188, str(round(volume_brl * 100.0)).zfill(18))
    _put_field(line, 210, 217, "0000001")
    _put_field(line, 230, 242, isin)
    _put_field(line, 242, 245, "001")
    return bytes(line)


def _write_and_parse_cotahist(
    root: Path,
    market: dict[str, np.ndarray],
    isins: tuple[str, ...],
    tickers: tuple[str, ...],
) -> tuple[pl.DataFrame, tuple[Path, ...], tuple[Path, ...], Path]:
    raw_root = root / "cotahist_raw"
    parsed_root = root / "cotahist_parsed"
    raw_root.mkdir(parents=True)
    parsed_root.mkdir()
    dates = _fixture_dates()
    audits = []
    raw_paths: list[Path] = []
    for year in sorted({value.year for value in dates}):
        quotes = [
            _cotahist_quote(
                trade_date=trade_date,
                ticker=tickers[name],
                isin=isins[name],
                open_brl=float(market["daily_open"][day, name]),
                high_brl=float(market["daily_high"][day, name]),
                low_brl=float(market["daily_low"][day, name]),
                close_brl=float(market["daily_close"][day, name]),
                volume_brl=float(market["daily_volume"][day, name]),
                trades=100 + day + name,
                quantity=100_000 + 100 * name,
            )
            for day, trade_date in enumerate(dates)
            if trade_date.year == year
            for name in range(_NAME_COUNT)
        ]
        header = bytearray(b" " * RECORD_LENGTH)
        _put_field(header, 0, 2, "00")
        _put_field(header, 23, 31, f"{year}1231")
        trailer = bytearray(b" " * RECORD_LENGTH)
        _put_field(trailer, 0, 2, "99")
        _put_field(trailer, 31, 42, str(len(quotes) + 2).zfill(11))
        raw_path = raw_root / f"COTAHIST_A{year}.ZIP"
        with zipfile.ZipFile(raw_path, mode="w") as archive:
            archive.writestr(
                f"COTAHIST_A{year}.TXT",
                b"\r\n".join((bytes(header), *quotes, bytes(trailer))) + b"\r\n",
            )
        audit = parse_year(raw_path, year, parsed_root)
        assert audit.error == ""
        assert audit.record_count_valid is True
        audits.append(audit)
        raw_paths.append(raw_path)
    parsed_paths = tuple(sorted(parsed_root.glob("year=*/equities_daily_*.parquet")))
    daily = pl.concat([pl.read_parquet(path) for path in parsed_paths])
    audit_path = parsed_root / "parse_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "script_version": "2",
                "audits": [asdict(item) for item in audits],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return daily, tuple(raw_paths), parsed_paths, audit_path


def _write_timestamped_m1(
    root: Path,
    market: dict[str, np.ndarray],
    isins: tuple[str, ...],
    tickers: tuple[str, ...],
) -> tuple[pl.DataFrame, Path, tuple[Path, ...]]:
    m1_root = root / "m1_raw"
    m1_root.mkdir(parents=True)
    dates = _fixture_dates()
    source_paths: list[Path] = []
    assignments: list[dict[str, object]] = []
    for name, (isin, ticker) in enumerate(zip(isins, tickers, strict=True)):
        timestamps = [
            datetime.combine(trade_date, time(15, 35)) + timedelta(minutes=minute)
            for trade_date in dates
            for minute in range(_MINUTE_COUNT)
        ]
        day_index = np.repeat(np.arange(_DAY_COUNT), _MINUTE_COUNT)
        minute_index = np.tile(np.arange(_MINUTE_COUNT), _DAY_COUNT)
        source_path = (m1_root / f"{ticker}.parquet").resolve()
        pl.DataFrame(
            {
                "ts_exchange": timestamps,
                "open": market["minute_open"][day_index, name, minute_index],
                "high": market["minute_high"][day_index, name, minute_index],
                "low": market["minute_low"][day_index, name, minute_index],
                "close": market["minute_close"][day_index, name, minute_index],
                "real_volume": market["minute_volume"][day_index, name, minute_index],
                "symbol": [ticker] * len(timestamps),
            }
        ).write_parquet(source_path)
        source_paths.append(source_path)
        assignments.append(
            {
                "security_id": f"ISIN:{isin}",
                "isin": isin,
                "xp_symbol": ticker,
                "source_file": str(source_path),
                "first_overlap_date": dates[0],
                "last_overlap_date": dates[-1],
                "manual_decision": "ACCEPTED",
                "normalization_rule": "FILTER_TO_COTAHIST_SECURITY_DATES",
            }
        )
    assignment_path = root / "m1_assignments.parquet"
    assignment_frame = pl.DataFrame(assignments)
    assignment_frame.write_parquet(assignment_path)
    return assignment_frame, assignment_path, tuple(source_paths)


def _build_fixture_store(root: Path, *, mutate_post_decision: bool = False) -> Path:
    source_root = root.with_name(f"{root.name}_sources")
    source_root.mkdir(parents=True)
    dates = _fixture_dates()
    market, isins, tickers, schedule, audit = _fixture_market(
        mutate_post_decision=mutate_post_decision
    )
    daily, raw_archives, parsed_paths, parse_audit_path = _write_and_parse_cotahist(
        source_root, market, isins, tickers
    )
    assignments, assignment_path, m1_paths = _write_timestamped_m1(
        source_root, market, isins, tickers
    )
    _validate_cotahist_parse_audit(parse_audit_path, raw_archives)
    immutable_hashes = {
        path: sha256_file(path)
        for path in (
            *raw_archives,
            *parsed_paths,
            parse_audit_path,
            assignment_path,
            *m1_paths,
        )
    }
    actions = normalize_yfinance_actions(
        pl.DataFrame(
            schema={
                "Date": pl.Date,
                "Dividends": pl.Float64,
                "Stock Splits": pl.Float64,
            }
        ),
        isin=isins[0],
        ticker="T0003",
        fetched_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    lending_names = SIDECAR_FEATURES["lending"]
    lending_values = np.zeros(
        (_DAY_COUNT, _NAME_COUNT, len(lending_names)), dtype=np.float32
    )
    lending_values[..., lending_names.index("loan_balance_to_volume_20")] = 0.1
    lending_values[..., lending_names.index("loan_rate")] = 0.02
    lending = SidecarResult(
        group="lending",
        feature_names=lending_names,
        values=lending_values,
        valid=np.ones_like(lending_values, dtype=np.bool_),
        age_sessions=np.zeros_like(lending_values, dtype=np.float32),
        coverage_by_year=(),
        archive_semantics_available=lending_names,
        publication_lag_reproduced=True,
        publication_lag_valid_cells=int(lending_values.size),
        publication_lag_source_rows=_DAY_COUNT * _NAME_COUNT,
        d_plus_one_rows_checked=_DAY_COUNT * _NAME_COUNT,
        d_plus_one_violations=0,
    )
    lending_rate_path = source_root / "bdi_lending_strong.parquet"
    lending_rate_encoded = np.tanh(np.log1p(2.0) / 2.0)
    pl.DataFrame(
        {
            "source_trade_date": pl.Series(
                np.repeat(dates[:-1], _NAME_COUNT).tolist(), dtype=pl.Date
            ),
            "available_date": pl.Series(
                np.repeat(dates[1:], _NAME_COUNT).tolist(), dtype=pl.Date
            ),
            "security_id": np.tile(
                [f"ISIN:{isin}" for isin in isins], _DAY_COUNT - 1
            ),
            "lending_taker_fee_level_log_tanh": np.full(
                (_DAY_COUNT - 1) * _NAME_COUNT,
                lending_rate_encoded,
                dtype=np.float64,
            ),
            "lending_taker_fee_level_log_tanh_mask": np.ones(
                (_DAY_COUNT - 1) * _NAME_COUNT, dtype=np.bool_
            ),
        }
    ).write_parquet(lending_rate_path)
    immutable_hashes[lending_rate_path] = sha256_file(lending_rate_path)
    store = build_daily_store(
        daily,
        actions,
        root,
        stream_intraday=True,
        sidecars={"lending": lending},
        action_acquisition_audit=audit,
        m1_assignments=assignments,
        source_paths=(*parsed_paths, assignment_path, lending_rate_path),
        cotahist_raw_sources=raw_archives,
        cotahist_parse_audit=parse_audit_path,
        session_schedule=schedule,
        minimum_rank_names=20,
        store_start=None,
    )
    assert all(sha256_file(path) == digest for path, digest in immutable_hashes.items())
    return store


def _loader(
    store: Path,
    indices: np.ndarray,
    *,
    stage: str,
    purpose: str,
    target_window: np.ndarray | None = None,
) -> DataLoader:
    dataset = V2DailyDataset(
        store,
        indices,
        stage=stage,
        lookback=20,
        purpose=purpose,
        target_window_indices=target_window,
    )
    return DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_v2_daily,
    )


def _f1_axes(
    store: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dates = np.load(store / "date_index.npy", allow_pickle=False).astype(
        "datetime64[D]", copy=False
    )
    fold = {value.name: value for value in development_folds(dates.astype(object))}[
        "F1"
    ]
    positions = {value: index for index, value in enumerate(dates.astype(object))}

    def indexes(values: tuple[date, ...]) -> np.ndarray:
        return np.asarray([positions[value] for value in values], dtype=np.int64)

    return (
        indexes(fold.fit_dates),
        indexes(fold.purge_before_dates),
        indexes(fold.selection_dates),
        indexes(fold.purge_after_dates),
        indexes(fold.evaluation_dates),
    )


def _tiny_native_fit(store: Path, output_dir: Path) -> tuple[ModelConfig, float, Path]:
    canonical_fit, purge_before, canonical_selection, purge_after, _ = _f1_axes(store)
    fit_indices = canonical_fit[-17:]
    fit_dataset = V2DailyDataset(
        store,
        fit_indices,
        stage="finetune",
        lookback=20,
        purpose="training",
        target_window_indices=np.concatenate((canonical_fit, purge_before)),
    )
    fit_loader = DataLoader(
        fit_dataset,
        batch_sampler=DatePairBatchSampler(
            fit_indices,
            pairs_per_batch=8,
            seed=29,
            drop_last=True,
        ),
        collate_fn=collate_v2_daily,
    )
    selection_indices = canonical_selection[:10]
    selection_loader = _loader(
        store,
        selection_indices,
        stage="finetune",
        purpose="selection",
        target_window=np.concatenate((canonical_selection, purge_after)),
    )
    config = ModelConfig(
        slow_feature_count=len(SLOW_FEATURES),
        current_feature_count=len(INTRADAY_DAILY_FEATURES),
        slow_lookback=20,
        hidden_width=4,
        fusion_width=8,
        trunk_blocks=1,
        trunk_swiglu_hidden=4,
        dropout=0.0,
        compile_forward=False,
    )
    result = train_stage(
        stage="F",
        seed=29,
        fold="F1_T24",
        train_loader=fit_loader,
        selection_loader=selection_loader,
        output_dir=output_dir,
        model_config=config,
        maximum_epochs=1,
        patience=1,
        microbatch_pairs=1,
        device=torch.device("cpu"),
    )
    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history
    loss = float(history[0]["training_loss"])
    assert np.isfinite(loss)
    checkpoint_payload = torch.load(
        result.raw_patience_checkpoint, map_location="cpu", weights_only=True
    )
    assert checkpoint_payload["schema"] == RAW_PATIENCE_SCHEMA
    return config, loss, result.raw_patience_checkpoint


def test_t24_raw_store_native_fit_score_ledger_report_relocation_and_stale_resume(
    tmp_path: Path,
) -> None:
    store = _build_fixture_store(tmp_path / "store")
    mutated_store = _build_fixture_store(
        tmp_path / "store_post_decision_mutated", mutate_post_decision=True
    )
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == STORE_SCHEMA
    assert manifest["axes"]["date_count"] == _DAY_COUNT
    assert manifest["axes"]["isin_count"] == _NAME_COUNT
    assert manifest["feature_names"]["slow"] == list(SLOW_FEATURES)
    assert manifest["feature_names"]["intraday"] == list(INTRADAY_DAILY_FEATURES)
    assert manifest["feature_names"]["native_fast"] == list(NATIVE_FAST_FEATURES)
    assert manifest["metadata"]["native_fast"]["legacy_v1_artifact_required"] is False
    assert manifest["official_validation_accessed"] is False
    assert manifest["test_accessed"] is False

    fast_values = np.load(store / "fast_patch_values.npy", allow_pickle=False)
    fast_valid = np.load(store / "fast_patch_valid.npy", allow_pickle=False)
    fast_mask = np.load(store / "fast_patch_mask.npy", allow_pickle=False)
    assert fast_values.shape == (_DAY_COUNT, _NAME_COUNT, 2, 7)
    assert fast_values.dtype == np.float32
    assert fast_valid.shape == fast_values.shape and fast_valid.dtype == np.bool_
    assert fast_mask.shape == fast_values.shape[:-1] and fast_mask.dtype == np.bool_
    assert fast_mask[_MUTATION_DATE_INDEX].all()
    assert not fast_valid[_MUTATION_DATE_INDEX, :, 0, 0].any()
    assert fast_valid[_MUTATION_DATE_INDEX, :, 1, 0].all()
    mapping_record = manifest["tables"]["native_fast_security_mapping"]
    mapping = pl.read_parquet(store / mapping_record["path"]).sort("fast_index")
    assert mapping.height == _NAME_COUNT
    assert mapping.get_column("fast_index").to_list() == list(range(_NAME_COUNT))

    for name in (
        "intraday_values",
        "intraday_valid",
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_present",
    ):
        baseline = np.load(store / f"{name}.npy", allow_pickle=False)
        mutated = np.load(mutated_store / f"{name}.npy", allow_pickle=False)
        np.testing.assert_array_equal(
            baseline[_MUTATION_DATE_INDEX], mutated[_MUTATION_DATE_INDEX]
        )
    assert not np.array_equal(
        np.load(store / "raw_close.npy", allow_pickle=False)[_MUTATION_DATE_INDEX],
        np.load(mutated_store / "raw_close.npy", allow_pickle=False)[
            _MUTATION_DATE_INDEX
        ],
    )

    config, fit_loss, checkpoint = _tiny_native_fit(
        store, tmp_path / "tiny_native_training"
    )
    assert np.isfinite(fit_loss)
    checkpoint_sha256 = sha256_file(checkpoint)
    evaluation_indices = _f1_axes(store)[-1][:10]
    scoring_loader = _loader(
        store,
        evaluation_indices,
        stage="evaluation",
        purpose="evaluation",
    )
    score_artifact = score_checkpoint_artifact(
        checkpoint=checkpoint,
        model_config=config,
        loader=scoring_loader,
        output_dir=tmp_path / "scores",
        expected_checkpoint_sha256=checkpoint_sha256,
        device=torch.device("cpu"),
    )
    score_manifest = json.loads(
        score_artifact.manifest_path.read_text(encoding="utf-8")
    )
    scores = np.load(score_artifact.scores_path, allow_pickle=False)
    score_mask = np.load(score_artifact.score_mask_path, allow_pickle=False)
    assert score_manifest["schema"] == SCORE_ARTIFACT_SCHEMA
    assert score_manifest["fast_initialization_provenance"]["mode"] == "fresh"
    assert score_manifest["fast_initialization_provenance"]["contaminated"] is False
    assert (
        score_manifest["scoring_input"]["store"]["external_artifact_resolutions"] == []
    )
    assert scores.shape == (evaluation_indices.size, _NAME_COUNT, 5)
    assert score_mask.shape == scores.shape
    expected_active = np.load(store / "active.npy", allow_pickle=False)[
        evaluation_indices
    ]
    np.testing.assert_array_equal(
        score_mask, np.repeat(expected_active[..., None], 5, axis=-1)
    )

    cdi = np.zeros(_DAY_COUNT, dtype=np.float64)
    source_hashes = {"fixture_store": sha256_file(store / "manifest.json")}
    bova11_close = 100.0 + 0.01 * np.arange(_DAY_COUNT, dtype=np.float64)
    bova11_binding = {
        "manifest_sha256": "a" * 64,
        "data_sha256": "b" * 64,
    }
    lending_borrow = LendingBorrowPanels(
        annual_taker_rate=np.full((_DAY_COUNT, _NAME_COUNT), 0.02),
        rate_imputed=np.zeros((_DAY_COUNT, _NAME_COUNT), dtype=np.bool_),
        shortable_strict=np.ones((_DAY_COUNT, _NAME_COUNT), dtype=np.bool_),
        shortable_balance=np.ones((_DAY_COUNT, _NAME_COUNT), dtype=np.bool_),
        shortable_open=np.ones((_DAY_COUNT, _NAME_COUNT), dtype=np.bool_),
        manifest_sha256="c" * 64,
        balance_sha256="d" * 64,
        rate_sha256="e" * 64,
        source_label="lending_archive_v2_2009_202412",
        source_unavailable_dates=(),
    )
    inputs = _evaluation_inputs(
        scoring_loader.dataset.store,
        evaluation_indices,
        scores,
        score_mask,
        cdi,
        bova11_close,
        bova11_binding,
        lending_borrow,
        source_hashes,
        transfer_chronology_clean=True,
    )
    result = evaluate_scores(inputs, window_name="T24_deterministic_fixture")
    report = result.report
    assert report["schema"] == EVALUATION_SCHEMA
    assert report["primary_support"]["used_date_count"] > 0
    assert report["economics"]["headline"]["intended_order_count"] > 0
    assert report["economics"]["headline"]["fill_count"] > 0
    assert report["economics"]["daily_table"]
    assert report["economics"]["d5_only_diagnostic"]["daily_table"]
    first_orders = [
        row
        for row in report["economics"]["headline_audit"]["intended_orders"]
        if row["decision_session"] == 0
    ]
    assert first_orders
    prior_reference = np.load(store / "prior_reference_close.npy", allow_pickle=False)[
        evaluation_indices[0]
    ]
    for order in first_orders:
        assert order["reference_price"] == pytest.approx(
            prior_reference[order["security_index"]]
        )
    assert report["mask_coverage"]["primary_common_score_and_outcome_name_days"] > 0
    assert report["transfer_chronology_clean"] is True

    retained_root = tmp_path / "retained_evaluation"
    retained_manifest, _ = _persist_scores(
        retained_root,
        {"scores": scores, "score_mask": score_mask},
        {
            "evaluation_date_indices": evaluation_indices.tolist(),
            "fold": "T24",
            "action_terms_source": "inferred_cotahist_dismes_v1",
            "schedule_source": "reconstructed_v1",
        },
    )
    assert (
        json.loads(retained_manifest.read_text(encoding="utf-8"))["schema"]
        == RESEARCH_SCORE_SCHEMA
    )
    evaluation_path = retained_root / "evaluation.json"
    report_sha256 = write_evaluation_report(evaluation_path, result)
    assert report_sha256 == sha256_file(evaluation_path)

    relocated_root = tmp_path / "relocated"
    relocated_store = relocated_root / "store"
    relocated_checkpoint = relocated_root / checkpoint.name
    relocated_retained = relocated_root / "retained_evaluation"
    relocated_root.mkdir()
    shutil.copytree(store, relocated_store)
    shutil.copy2(checkpoint, relocated_checkpoint)
    shutil.copytree(retained_root, relocated_retained)
    assert sha256_file(relocated_checkpoint) == checkpoint_sha256
    assert (
        sha256_file(relocated_store / "manifest.json") == source_hashes["fixture_store"]
    )

    relocated_loader = _loader(
        relocated_store,
        evaluation_indices,
        stage="evaluation",
        purpose="evaluation",
    )
    relocated_scores = score_checkpoint_artifact(
        checkpoint=relocated_checkpoint,
        model_config=config,
        loader=relocated_loader,
        output_dir=relocated_root / "scores",
        expected_checkpoint_sha256=checkpoint_sha256,
        device=torch.device("cpu"),
    )
    assert (
        score_artifact.scores_path.read_bytes()
        == relocated_scores.scores_path.read_bytes()
    )
    assert (
        score_artifact.score_mask_path.read_bytes()
        == relocated_scores.score_mask_path.read_bytes()
    )
    relocated_score_manifest = json.loads(
        relocated_scores.manifest_path.read_text(encoding="utf-8")
    )
    assert (
        relocated_score_manifest["scoring_input_sha256"]
        == score_manifest["scoring_input_sha256"]
    )
    rebuilt = _evaluation_from_artifacts(
        relocated_retained / "evaluation.json",
        store=relocated_loader.dataset.store,
        indices=evaluation_indices,
        cdi=cdi,
        bova11_close_by_index=bova11_close,
        bova11_binding=bova11_binding,
        lending_borrow=lending_borrow,
    )
    np.testing.assert_array_equal(
        rebuilt.result.daily_primary_ic, result.daily_primary_ic
    )
    np.testing.assert_array_equal(
        rebuilt.result.headline_net_excess_bps, result.headline_net_excess_bps
    )
    assert rebuilt.result.report["input_hashes"] == report["input_hashes"]

    stale_root = tmp_path / "stale_resume"
    shutil.copytree(relocated_retained, stale_root)
    stale_report = json.loads(
        (stale_root / "evaluation.json").read_text(encoding="utf-8")
    )
    stale_report["schema"] = "BRAZIL_RV_V2_EVALUATION_V3"
    write_json_atomic(stale_root / "evaluation.json", stale_report)
    with pytest.raises(ValueError, match="not a v2 evaluation"):
        _evaluation_from_artifacts(
            stale_root / "evaluation.json",
            store=relocated_loader.dataset.store,
            indices=evaluation_indices,
            cdi=cdi,
            bova11_close_by_index=bova11_close,
            bova11_binding=bova11_binding,
            lending_borrow=lending_borrow,
        )

    stale_plan = tmp_path / "stale_plan.json"
    stale_plan.write_text(
        json.dumps(
            {
                "schema": "BRAZIL_RV_V2_RUN_MANY_PLAN_V1",
                "phase": "registered_arms",
                "max_parallel": 1,
                "jobs": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale or lacks the current schema"):
        load_plan(stale_plan)
