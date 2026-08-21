from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from brazil_rv.preprocessing.regular_trade_activity import (
    FEATURES,
    build_artifact,
    build_normalized_frame,
)

SECURITY_ID = "ISIN:BRPETRACNPR6"
ISIN = "BRPETRACNPR6"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _daily_frame(
    sessions: list[date],
    *,
    security_id: str = SECURITY_ID,
    isin: str = ISIN,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, trade_date in enumerate(sessions[:-1]):
        trades = 100 + (index * 7) % 31
        shares_per_trade = 10 + (index * 3) % 11
        value_per_trade = 800 + (index * 13) % 101
        rows.append(
            {
                "trade_date": trade_date,
                "security_id": security_id,
                "security_id_is_fallback": False,
                "isin": isin,
                "bdi_code": "02",
                "market_type": 10,
                "currency": "R$",
                "close_brl": 40.0,
                "trades": trades,
                "quantity": trades * shares_per_trade,
                "volume_brl": float(trades * value_per_trade),
                "distribution_number": 100,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("trade_date").cast(pl.Date),
        pl.col("security_id_is_fallback").cast(pl.Boolean),
    )


def _sessions(count: int = 82) -> list[date]:
    start = date(2021, 1, 4)
    return [start + timedelta(days=index) for index in range(count)]


def test_features_use_prior_observations_and_exact_next_session() -> None:
    sessions = _sessions()
    daily = _daily_frame(sessions)
    frame, _ = build_normalized_frame(daily, sessions)
    assert frame[0, "source_trade_date"] == sessions[0]
    assert frame[0, "available_date"] == sessions[1]
    assert not frame[19, "regular_log1p_trades_robust_z20_mask"]
    assert frame[20, "regular_log1p_trades_robust_z20_mask"]
    assert not frame[59, "regular_log1p_trades_robust_z60_mask"]
    assert frame[60, "regular_log1p_trades_robust_z60_mask"]

    prior = [math.log1p(100 + (index * 7) % 31) for index in range(20)]
    center = statistics.median(prior)
    scale = 1.4826 * statistics.median(abs(value - center) for value in prior)
    current = math.log1p(100 + (20 * 7) % 31)
    expected = min(max((current - center) / scale, -5.0), 5.0)
    assert frame[20, "regular_log1p_trades_robust_z20"] == pytest.approx(
        expected, abs=1e-6
    )

    mutated = daily.with_columns(
        pl.when(pl.col("trade_date") == sessions[75])
        .then(pl.lit(9_999_999))
        .otherwise(pl.col("trades"))
        .alias("trades")
    )
    changed, _ = build_normalized_frame(mutated, sessions)
    earlier = frame.filter(pl.col("source_trade_date") < sessions[75])
    earlier_changed = changed.filter(pl.col("source_trade_date") < sessions[75])
    assert earlier.equals(earlier_changed)


def test_missing_ratios_are_zero_masked_without_poisoning_history() -> None:
    sessions = _sessions()
    daily = _daily_frame(sessions).with_columns(
        pl.when(pl.col("trade_date") == sessions[61])
        .then(0)
        .otherwise(pl.col("trades"))
        .alias("trades"),
        pl.when(pl.col("trade_date") == sessions[61])
        .then(0)
        .otherwise(pl.col("quantity"))
        .alias("quantity"),
        pl.when(pl.col("trade_date") == sessions[61])
        .then(0.0)
        .otherwise(pl.col("volume_brl"))
        .alias("volume_brl"),
    )
    frame, audit = build_normalized_frame(daily, sessions)
    row = frame.filter(pl.col("source_trade_date") == sessions[61]).row(0, named=True)
    assert row["regular_log1p_trades_robust_z60_mask"]
    for metric in ("log_avg_trade_value", "log_shares_per_trade"):
        for window in (20, 60):
            feature = f"regular_{metric}_robust_z{window}"
            assert row[feature] == 0
            assert not row[f"{feature}_mask"]
    following = frame.filter(pl.col("source_trade_date") == sessions[62]).row(
        0, named=True
    )
    assert following["regular_log_shares_per_trade_robust_z60_mask"]
    assert audit["invalid_avg_trade_value_source_rows"] == 1
    assert audit["invalid_shares_per_trade_source_rows"] == 1
    for feature in FEATURES:
        invalid = frame.filter(~pl.col(f"{feature}_mask"))
        assert invalid.get_column(feature).eq(0).all()


def test_share_history_resets_only_at_causal_unit_break() -> None:
    sessions = _sessions(90)
    daily = _daily_frame(sessions).with_columns(
        pl.when(pl.col("trade_date") >= sessions[61])
        .then(101)
        .otherwise(pl.col("distribution_number"))
        .alias("distribution_number"),
        pl.when(pl.col("trade_date") >= sessions[61])
        .then(16.0)
        .otherwise(pl.col("close_brl"))
        .alias("close_brl"),
        pl.when(pl.col("trade_date") >= sessions[61])
        .then(pl.col("quantity") * 3)
        .otherwise(pl.col("quantity"))
        .alias("quantity"),
    )
    frame, audit = build_normalized_frame(daily, sessions)
    event = frame.filter(pl.col("source_trade_date") == sessions[61]).row(0, named=True)
    assert event["regular_log1p_trades_robust_z60_mask"]
    assert event["regular_log_avg_trade_value_robust_z60_mask"]
    assert not event["regular_log_shares_per_trade_robust_z20_mask"]
    assert not event["regular_log_shares_per_trade_robust_z60_mask"]
    after_twenty = frame.filter(pl.col("source_trade_date") == sessions[81]).row(
        0, named=True
    )
    assert after_twenty["regular_log_shares_per_trade_robust_z20_mask"]
    assert not after_twenty["regular_log_shares_per_trade_robust_z60_mask"]
    assert audit["share_unit_break_count"] == 1


def test_exact_identity_and_duplicate_guards() -> None:
    sessions = _sessions(4)
    exact = _daily_frame(sessions).head(1)
    fallback = exact.with_columns(
        pl.lit("FALLBACK:PETR4").alias("security_id"),
        pl.lit(True).alias("security_id_is_fallback"),
    )
    frame, audit = build_normalized_frame(pl.concat([exact, fallback]), sessions)
    assert frame.get_column("security_id").to_list() == [SECURITY_ID]
    assert audit["fallback_identity_rows_removed"] == 1

    with pytest.raises(ValueError, match="exactly equal"):
        build_normalized_frame(
            exact.with_columns(pl.lit("ISIN:WRONG").alias("security_id")), sessions
        )
    with pytest.raises(ValueError, match="duplicate security/date"):
        build_normalized_frame(pl.concat([exact, exact]), sessions)


def test_artifact_records_source_identity_and_feature_coverage(tmp_path: Path) -> None:
    sessions = _sessions()
    source_dir = tmp_path / "cotahist"
    year_dir = source_dir / "year=2021"
    year_dir.mkdir(parents=True)
    source = year_dir / "equities_daily_2021.parquet"
    source_frame = _daily_frame(sessions)
    calendar_tail = source_frame.tail(1).with_columns(
        pl.lit(sessions[-1], dtype=pl.Date).alias("trade_date")
    )
    pl.concat([source_frame, calendar_tail]).write_parquet(source)
    (source_dir / "parse_audit.json").write_text("{}", encoding="utf-8")
    security_index = tmp_path / "security_index.parquet"
    pl.DataFrame({"security_id": [SECURITY_ID]}).write_parquet(security_index)
    output = tmp_path / "output"
    manifest = build_artifact(
        source_dir,
        output,
        security_index=security_index,
        available_start=sessions[61],
        available_end=sessions[-1],
    )
    frame = pl.read_parquet(output / "regular_trade_activity.parquet")
    stored = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stored["output_sha256"] == _sha256(output / "regular_trade_activity.parquet")
    assert stored["source_files"] == [
        {"path": str(source.resolve()), "sha256": _sha256(source)}
    ]
    assert stored["security_index"]["sha256"] == _sha256(security_index)
    assert set(stored["feature_valid_rows"]) == set(FEATURES)
    assert frame.get_column("available_date").min() == sessions[61]
    assert frame.get_column("available_date").max() == sessions[-1]
    assert manifest["output_row_count"] == frame.height
    with pytest.raises(FileExistsError):
        build_artifact(source_dir, output)
