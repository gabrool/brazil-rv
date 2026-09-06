from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from brazil_rv.v2.build_store import _parse_sidecars, _sidecar_coverage_table
from brazil_rv.v2.sidecars import (
    available_archive_mapping,
    derive_known_archive_features,
    materialize_known_archive,
    materialize_sidecar,
    rebuild_publication_lag_validity,
)


def _oddlot_row_reference(source: pl.DataFrame, days: list[date]) -> pl.DataFrame:
    """Small no-revision oracle for odd-lot share and exact lag-5 change."""

    positions = {day: index for index, day in enumerate(days)}
    shares: dict[tuple[str, date], tuple[float, bool]] = {}
    rows: list[dict[str, object]] = []
    for row in source.iter_rows(named=True):
        source_date = row["source_trade_date"]
        regular = row["regular_volume_brl"]
        odd = row["odd_lot_volume_brl"]
        valid = (
            source_date in positions
            and regular is not None
            and odd is not None
            and np.isfinite(float(regular))
            and np.isfinite(float(odd))
            and float(regular) >= 0.0
            and float(odd) >= 0.0
            and float(regular) + float(odd) > 0.0
        )
        share = float(odd) / (float(regular) + float(odd)) if valid else 0.0
        key = (str(row["isin"]), source_date)
        shares[key] = (share, valid)
        rows.append(
            {
                "available_date": row["available_date"],
                "isin": str(row["isin"]),
                "source_trade_date": source_date,
                "oddlot_volume_share": share,
                "oddlot_volume_share_mask": valid,
            }
        )
    for row in rows:
        day = positions.get(row["source_trade_date"])
        prior_date = days[day - 5] if day is not None and day >= 5 else None
        current = shares[(row["isin"], row["source_trade_date"])]
        prior = shares.get((row["isin"], prior_date), (0.0, False))
        valid = current[1] and prior[1]
        row["oddlot_volume_share_change_5"] = current[0] - prior[0] if valid else 0.0
        row["oddlot_volume_share_change_5_mask"] = valid
    return pl.DataFrame(rows).sort("isin", "available_date", "source_trade_date")


def test_raw_annual_lending_rate_and_exact_change_are_supported() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(7)]
    raw_rates = [0.10, 0.11, 0.12, 0.13, 0.14, 0.25]
    source = pl.DataFrame(
        {
            "available_date": days[1:],
            "source_trade_date": days[:-1],
            "isin": ["BRTESTACNOR1"] * 6,
            "loan_rate_annual_decimal": raw_rates,
            "loan_rate_annual_decimal_mask": [True] * 6,
        }
    )
    derived = derive_known_archive_features(
        source, days, ["BRTESTACNOR1"], group="lending",
        daily_volume_brl=np.ones((7, 1)),
    )
    result = materialize_known_archive(derived, days, ["BRTESTACNOR1"], group="lending")
    assert result.feature_names == ("loan_rate", "loan_rate_change_5")
    assert result.values[1, 0, 0] == pytest.approx(0.10)
    assert result.values[6, 0, 1] == pytest.approx(0.15)
    assert result.valid[6, 0].all()


def test_legacy_transformed_options_and_clipped_leverage_are_disabled() -> None:
    options = available_archive_mapping(
        "options",
        [
            "options_put_call_oi_log_ratio_tanh",
            "options_oi_change_to_stock_adv20_tanh",
            "options_atm_iv_prior20_robust_z_scaled",
            "options_put_skew_tanh",
        ],
    )
    assert options == {}
    assert available_archive_mapping("fundamentals", ["fund_leverage"]) == {}


def test_raw_options_counts_use_registered_smoothed_log_ratio() -> None:
    source = pl.DataFrame(
        {
            "available_date": [date(2024, 1, 2)],
            "isin": ["BRTESTACNOR1"],
            "put_oi": [2],
            "call_oi": [1],
            "oi_snapshot_complete": [True],
        }
    )
    derived = derive_known_archive_features(
        source, [date(2024, 1, 2)], ["BRTESTACNOR1"], group="options"
    )
    result = materialize_known_archive(
        derived, [date(2024, 1, 2)], ["BRTESTACNOR1"], group="options"
    )
    assert result.feature_names == ("put_call_log_oi_ratio",)
    assert result.values[0, 0, 0] == pytest.approx(np.log(1.5))
    assert result.valid[0, 0, 0]
    assert result.source_missing_candidates == (
        "delta_oi_to_volume_1", "atm_iv_to_median_20", "put_skew"
    )


def test_options_known_zero_counts_do_not_create_sentiment() -> None:
    day = date(2024, 1, 2)
    source = pl.DataFrame(
        {
            "available_date": [day],
            "isin": ["BRTESTACNOR1"],
            "put_open_interest": [0],
            "call_open_interest": [0],
            "open_interest_snapshot_complete": [True],
        }
    )
    derived = derive_known_archive_features(source, [day], ["BRTESTACNOR1"], group="options")
    result = materialize_known_archive(
        derived, [day], ["BRTESTACNOR1"], group="options"
    )
    assert result.feature_names == ("put_call_log_oi_ratio",)
    assert not result.valid.any()
    assert result.values[0, 0, 0] == 0.0


def test_raw_compatible_liabilities_to_assets_is_not_clipped() -> None:
    day = date(2024, 1, 2)
    source = pl.DataFrame(
        {
            "available_date": [day],
            "isin": ["BRTESTACNOR1"],
            "total_liabilities_brl": [350.0],
            "total_assets_brl": [100.0],
        }
    )
    derived = derive_known_archive_features(
        source, [day], ["BRTESTACNOR1"], group="fundamentals"
    )
    result = materialize_known_archive(
        derived, [day], ["BRTESTACNOR1"], group="fundamentals"
    )
    assert result.feature_names == ("liabilities_to_assets",)
    assert result.values[0, 0, 0] == pytest.approx(3.5)
    assert result.valid[0, 0, 0]


def test_overlapping_financial_filings_reset_exact_session_age() -> None:
    days = [date(2024, 1, 2) + timedelta(days=index) for index in range(4)]
    isin = "BRTESTACNOR1"
    b3 = ZoneInfo("America/Sao_Paulo")
    source = pl.DataFrame(
        {
            "isin": [isin, isin],
            "filing_receipt_timestamp": [
                datetime(2024, 1, 2, 15, 0, tzinfo=b3),
                datetime(2024, 1, 3, 15, 46, tzinfo=b3),
            ],
            "filing_receipt_timestamp_precision": ["second", "second"],
            "filing_receipt_timestamp_processing_latency_seconds": [0.0, 0.0],
        }
    )
    derived = derive_known_archive_features(source, days, [isin], group="events")
    result = materialize_known_archive(derived, days, [isin], group="events")
    assert result.feature_names == ("sessions_since_financial_filing",)
    assert result.values[:, 0, 0].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert result.valid[:, 0, 0].all()


def test_financial_filing_adapter_requires_typed_timestamps() -> None:
    source = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR1"],
            "filing_receipt_timestamp": ["2024-01-02T15:00:00-03:00"],
        }
    )
    with pytest.raises(ValueError, match="Datetime dtype"):
        derive_known_archive_features(
            source,
            [date(2024, 1, 2)],
            ["BRTESTACNOR1"],
            group="events",
        )


def test_event_age_requires_a_timestamp_not_a_recent_flag() -> None:
    source = pl.DataFrame({
        "available_date": [date(2024, 1, 2)],
        "isin": ["BRTESTACNOR1"],
        "event_itr_dfp_recent_5s": [1.0],
    })
    result = materialize_known_archive(
        derive_known_archive_features(source, [date(2024, 1, 2)], ["BRTESTACNOR1"], group="events"),
        [date(2024, 1, 2)], ["BRTESTACNOR1"], group="events"
    )
    assert result.feature_names == ()
    assert result.source_missing_candidates == (
        "sessions_since_financial_filing", "standardized_unexpected_earnings"
    )


def test_sidecar_timestamps_compare_absolute_instants_to_b3_decision() -> None:
    decision_date = date(2024, 1, 2)
    # UTC-5 13:40 is 15:40 in Sao Paulo and is usable; UTC-5 14:00 is
    # 16:00 in Sao Paulo and is not. A non-B3 offset is therefore accepted or
    # rejected by its instant, never by its timezone label.
    utc_minus_five = timezone(timedelta(hours=-5))
    early = pl.DataFrame(
        {
            "available_date": [decision_date],
            "available_timestamp": [
                datetime(2024, 1, 2, 13, 40, tzinfo=utc_minus_five)
            ],
            "available_timestamp_precision": ["second"],
            "available_timestamp_processing_latency_seconds": [0.0],
            "isin": ["BRTESTACNOR1"],
            "oddlot_volume_share": [0.25],
            "oddlot_volume_share_change_5": [0.0],
        }
    )
    accepted = materialize_sidecar(
        early,
        [decision_date],
        ["BRTESTACNOR1"],
        group="oddlot",
        decision_time=time(15, 45),
    )
    assert accepted.valid[0, 0].all()
    late = early.with_columns(
        pl.lit(datetime(2024, 1, 2, 14, 0, tzinfo=utc_minus_five)).alias(
            "available_timestamp"
        )
    )
    rejected = materialize_sidecar(
        late,
        [decision_date],
        ["BRTESTACNOR1"],
        group="oddlot",
        decision_time=time(15, 45),
    )
    assert not rejected.valid.any()


def test_minute_precision_and_latency_are_conservative_at_strict_decision() -> None:
    day = date(2024, 1, 2)
    b3 = ZoneInfo("America/Sao_Paulo")
    source = pl.DataFrame(
        {
            "available_date": [day, day, day],
            "available_timestamp": [
                datetime(2024, 1, 2, 15, 44, tzinfo=b3),
                datetime(2024, 1, 2, 15, 45, tzinfo=b3),
                datetime(2024, 1, 2, 15, 44, 30, tzinfo=b3),
            ],
            "available_timestamp_precision": ["minute", "minute", "exact"],
            "available_timestamp_processing_latency_seconds": [0.0, 0.0, 31.0],
            "isin": ["EARLY", "LATE", "LATENCY"],
            "oddlot_volume_share": [0.1, 0.2, 0.3],
        }
    )
    result = materialize_sidecar(
        source, [day], ["EARLY", "LATE", "LATENCY"], group="oddlot"
    )
    assert result.valid[0, :, 0].tolist() == [True, False, False]


def test_enabled_features_are_an_ordered_supported_subset() -> None:
    day = date(2024, 1, 2)
    source = pl.DataFrame(
        {
            "available_date": [day],
            "isin": ["BRTESTACNOR1"],
            "oddlot_volume_share_change_5": [0.2],
        }
    )
    result = materialize_known_archive(
        source, [day], ["BRTESTACNOR1"], group="oddlot"
    )
    assert result.feature_names == ("oddlot_volume_share_change_5",)
    assert result.values.shape == (1, 1, 1)
    assert result.source_missing_candidates == ("oddlot_volume_share",)


def test_raw_lending_formulas_use_source_date_volume_and_d_plus_one() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(27)]
    isin = "BRTESTACNOR1"
    source = pl.DataFrame(
        {
            "source_position_date": days[19:26],
            "available_date": days[20:27],
            "isin": [isin] * 7,
            "lending_balance_brl": [200.0, 220.0, 240.0, 260.0, 280.0, 300.0, 360.0],
        }
    )
    volume = np.full((len(days), 1), 100.0)
    derived = derive_known_archive_features(
        source, days, [isin], group="lending", daily_volume_brl=volume
    )
    result = materialize_known_archive(derived, days, [isin], group="lending")
    rebuilt = rebuild_publication_lag_validity(
        derived,
        days,
        [isin],
        group="lending",
        feature_columns=available_archive_mapping("lending", derived.columns),
        date_only_available_before_decision=True,
    )
    np.testing.assert_array_equal(result.valid, rebuilt)
    # The position at source session 19 becomes visible only on session 20.
    assert not result.valid[19, 0].any()
    assert result.values[20, 0, 0] == pytest.approx(2.0)
    assert result.values[21, 0, 1] == pytest.approx(0.2)
    assert result.values[26, 0, 2] == pytest.approx(1.4)
    assert not result.valid[..., 3:].any()
    assert set(result.archive_semantics_available) == {
        "loan_balance_to_volume_20",
        "loan_balance_change_1",
        "loan_balance_change_5",
    }

    changed = volume.copy()
    changed[20:] = 1_000_000.0
    causal = derive_known_archive_features(
        source, days, [isin], group="lending", daily_volume_brl=changed
    )
    causal_result = materialize_known_archive(causal, days, [isin], group="lending")
    assert causal_result.values[20, 0, 0] == result.values[20, 0, 0]


def test_lending_lags_use_the_vintage_known_at_each_publication() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(27)]
    isin = "BRTESTACNOR1"
    original = pl.DataFrame(
        {
            "source_position_date": [days[19], days[20]],
            "available_date": [days[20], days[21]],
            "isin": [isin, isin],
            "lending_balance_brl": [100.0, 120.0],
        }
    )
    revision = pl.DataFrame(
        {
            "source_position_date": [days[19]],
            "available_date": [days[25]],
            "isin": [isin],
            "lending_balance_brl": [900.0],
        }
    )
    volume = np.full((len(days), 1), 100.0)
    baseline = derive_known_archive_features(
        original, days, [isin], group="lending", daily_volume_brl=volume
    )
    revised_source = pl.concat([original, revision], rechunk=False).sample(
        fraction=1.0, shuffle=True, seed=29
    )
    revised = derive_known_archive_features(
        revised_source, days, [isin], group="lending", daily_volume_brl=volume
    )
    before_revision = revised.filter(pl.col("available_date") < days[25])
    assert_frame_equal(before_revision, baseline, check_exact=True)
    # The source-day-20 publication saw the original source-day-19 vintage:
    # +0.2 must not become -7.8 after the future correction is appended.
    row = revised.filter(pl.col("source_position_date") == days[20]).row(
        0, named=True
    )
    assert row["loan_balance_change_1"] == pytest.approx(0.2)
    assert row["loan_balance_change_1_mask"]
    chunked = pl.concat(
        [revised_source.slice(0, 1), revised_source.slice(1)], rechunk=False
    )
    chunked_result = derive_known_archive_features(
        chunked, days, [isin], group="lending", daily_volume_brl=volume
    )
    assert_frame_equal(chunked_result, revised, check_exact=True)


def test_lending_volume_window_does_not_invent_prelisting_zeroes() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(31)]
    isin = "BRTESTACNOR1"
    source = pl.DataFrame(
        {
            "source_position_date": [days[19], days[29]],
            "available_date": [days[20], days[30]],
            "isin": [isin, isin],
            "lending_balance_brl": [200.0, 200.0],
        }
    )
    volume = np.full((len(days), 1), np.nan)
    volume[10:] = 100.0
    derived = derive_known_archive_features(
        source, days, [isin], group="lending", daily_volume_brl=volume
    )
    result = materialize_known_archive(derived, days, [isin], group="lending")
    assert not result.valid[20, 0, 0]
    assert result.values[20, 0, 0] == 0.0
    assert result.valid[30, 0, 0]
    assert result.values[30, 0, 0] == pytest.approx(2.0)


def test_lending_volume_window_rejects_an_internal_missing_session() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(21)]
    isin = "BRTESTACNOR1"
    source = pl.DataFrame(
        {
            "source_position_date": [days[19]],
            "available_date": [days[20]],
            "isin": [isin],
            "lending_balance_brl": [200.0],
        }
    )
    volume = np.full((len(days), 1), 100.0)
    volume[15, 0] = np.nan
    derived = derive_known_archive_features(
        source, days, [isin], group="lending", daily_volume_brl=volume
    )
    result = materialize_known_archive(derived, days, [isin], group="lending")
    assert not result.valid[20, 0, 0]
    assert result.values[20, 0, 0] == 0.0


def test_lending_volume_window_counts_observed_zero_as_support() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(21)]
    isin = "BRTESTACNOR1"
    source = pl.DataFrame(
        {
            "source_position_date": [days[19]],
            "available_date": [days[20]],
            "isin": [isin],
            "lending_balance_brl": [200.0],
        }
    )
    volume = np.full((len(days), 1), 100.0)
    volume[15, 0] = 0.0
    derived = derive_known_archive_features(
        source, days, [isin], group="lending", daily_volume_brl=volume
    )
    result = materialize_known_archive(derived, days, [isin], group="lending")
    assert result.valid[20, 0, 0]
    assert result.values[20, 0, 0] == pytest.approx(200.0 / 95.0)


def test_balance_and_raw_rate_archives_combine_without_legacy_inversion() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(127)]
    isin = "BRTESTACNOR1"
    security_id = "security-one"
    balance = pl.DataFrame(
        {
            "source_position_date": days[19:126],
            "available_date": days[20:127],
            "security_id": [security_id] * 107,
            "lending_balance_brl": [200.0] * 107,
        }
    )
    rates = [0.10, 0.11, 0.12, 0.13, 0.14, 0.25, 0.30]
    raw_rate = pl.DataFrame(
        {
            # Leave more than Polars' default 100-row inference window before
            # the first non-null rate value, matching the combined archives.
            "source_trade_date": days[119:126],
            "available_date": days[120:127],
            "security_id": [security_id] * 7,
            "loan_rate_annual_decimal": rates,
            "loan_rate_annual_decimal_mask": [True] * 7,
        }
    )
    combined = pl.concat([balance, raw_rate], how="diagonal_relaxed").with_columns(
        pl.lit(isin).alias("isin")
    ).drop("security_id")
    value_columns = [
        column for column in combined.columns if column not in {"available_date", "isin"}
    ]
    combined = combined.group_by("available_date", "isin", maintain_order=True).agg(
        pl.col(column).drop_nulls().last().alias(column) for column in value_columns
    )
    derived = derive_known_archive_features(
        combined, days, [isin], group="lending",
        daily_volume_brl=np.full((len(days), 1), 100.0),
    )
    result = materialize_known_archive(derived, days, [isin], group="lending")
    assert result.values[120, 0, 0] == pytest.approx(2.0)
    assert result.values[120, 0, 3] == pytest.approx(0.10)
    assert result.values[125, 0, 4] == pytest.approx(0.15)
    assert result.valid[125, 0].tolist() == [True, True, True, True, True]


def test_raw_oddlot_share_and_exact_session_lag_change() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(7)]
    isin = "BRTESTACNOR1"
    shares = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    source = pl.DataFrame(
        {
            "source_trade_date": days[:6],
            "available_date": days[1:7],
            "isin": [isin] * 6,
            "regular_volume_brl": [(1.0 - value) * 100.0 for value in shares],
            "odd_lot_volume_brl": [value * 100.0 for value in shares],
        }
    )
    derived = derive_known_archive_features(source, days, [isin], group="oddlot")
    result = materialize_known_archive(derived, days, [isin], group="oddlot")
    assert result.values[1, 0, 0] == pytest.approx(0.1)
    assert not result.valid[5, 0, 1]
    assert result.values[6, 0, 1] == pytest.approx(0.6)


def test_vectorized_oddlot_matches_randomized_row_oracle() -> None:
    rng = np.random.default_rng(57)
    sessions = [date(2024, 1, 1) + timedelta(days=index) for index in range(25)]
    archive_dates = [*sessions, sessions[-1] + timedelta(days=2)]
    rows: list[dict[str, object]] = []
    for name_index in range(4):
        regular = rng.uniform(0.0, 1_000.0, size=len(archive_dates)).tolist()
        odd = rng.uniform(0.0, 500.0, size=len(archive_dates)).tolist()
        regular[2] = None
        odd[3] = None
        regular[4] = -1.0
        regular[5] = 0.0
        odd[5] = 0.0
        rows.extend(
            {
                "source_trade_date": day,
                "available_date": day + timedelta(days=1),
                "isin": f"BRTEST{name_index:02d}",
                "regular_volume_brl": regular_value,
                "odd_lot_volume_brl": odd_value,
            }
            for day, regular_value, odd_value in zip(
                archive_dates, regular, odd, strict=True
            )
        )
    source = pl.DataFrame(rows, infer_schema_length=None).sample(
        fraction=1.0, shuffle=True, seed=61
    )
    expected = _oddlot_row_reference(source, sessions)
    actual = derive_known_archive_features(
        source, sessions, sorted(source.get_column("isin").unique()), group="oddlot"
    ).sort("isin", "available_date", "source_trade_date")
    assert_frame_equal(actual, expected, check_exact=False, abs_tol=1e-12)


def test_oddlot_lags_use_the_vintage_known_at_each_publication() -> None:
    days = [date(2024, 1, 1) + timedelta(days=index) for index in range(10)]
    isin = "BRTESTACNOR1"
    original = pl.DataFrame(
        {
            "source_trade_date": [days[0], days[5]],
            "available_date": [days[1], days[6]],
            "isin": [isin, isin],
            "regular_volume_brl": [90.0, 70.0],
            "odd_lot_volume_brl": [10.0, 30.0],
        }
    )
    revision = pl.DataFrame(
        {
            "source_trade_date": [days[0]],
            "available_date": [days[8]],
            "isin": [isin],
            "regular_volume_brl": [10.0],
            "odd_lot_volume_brl": [90.0],
        }
    )
    baseline = derive_known_archive_features(
        original, days, [isin], group="oddlot"
    )
    shuffled = pl.concat([original, revision], rechunk=False).sample(
        fraction=1.0, shuffle=True, seed=97
    )
    revised = derive_known_archive_features(
        shuffled, days, [isin], group="oddlot"
    )
    assert_frame_equal(
        revised.filter(pl.col("available_date") < days[8]),
        baseline,
        check_exact=True,
    )
    row = revised.filter(pl.col("source_trade_date") == days[5]).row(0, named=True)
    assert row["oddlot_volume_share_change_5"] == pytest.approx(0.2)
    assert row["oddlot_volume_share_change_5_mask"]
    chunked = pl.concat([shuffled.head(1), shuffled.tail(shuffled.height - 1)], rechunk=False)
    assert_frame_equal(
        derive_known_archive_features(chunked, days, [isin], group="oddlot"),
        revised,
        check_exact=True,
    )


def test_rebalance_contract_preserves_experiment33_field_names() -> None:
    columns = [
        "ibov_current_weight_sqrt",
        "ibov_preview_delta_signed_sqrt",
        "ibov_preview_add",
        "ibov_preview_delete",
        "ibov_preview_pressure",
        "ibov_pre_effective_ramp",
        "ibov_post_effective_reversal",
    ]
    mapping = available_archive_mapping("rebalance", columns)
    assert all(mapping[name] == name for name in columns)
    assert not any(name.startswith("rebalance_") for name in mapping)


def test_sidecar_coverage_reports_total_and_active_denominators() -> None:
    dates = np.asarray(["2024-01-02", "2024-01-03"], dtype="datetime64[D]")
    mask = np.asarray([[[True]], [[True]]])
    active = np.asarray([[True], [False]])
    table = _sidecar_coverage_table(
        dates,
        {"oddlot": (("oddlot_volume_share",), mask, ("oddlot_volume_share",))},
        active,
    )
    assert table[0, "valid_count"] == 2
    assert table[0, "possible_count"] == 2
    assert table[0, "active_valid_count"] == 1
    assert table[0, "active_possible_count"] == 1
    assert table[0, "archive_semantics_available"]


def test_unknown_same_day_availability_is_rejected_and_latest_snapshot_wins() -> None:
    source = pl.DataFrame(
        {
            "available_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "decision_idx": [1, 2],
            "isin": ["BRTESTACNOR1", "BRTESTACNOR1"],
            "oddlot_volume_share": [0.1, 0.2],
            "oddlot_volume_share_change_5": [0.0, 0.1],
        }
    )
    result = materialize_sidecar(
        source,
        [date(2024, 1, 2)],
        ["BRTESTACNOR1"],
        group="oddlot",
    )
    assert result.values[0, 0].tolist() == pytest.approx([0.2, 0.1])
    unknown = source.drop("decision_idx").head(1)
    rejected = materialize_sidecar(
        unknown,
        [date(2024, 1, 2)],
        ["BRTESTACNOR1"],
        group="oddlot",
    )
    assert not rejected.valid.any()


def test_stateful_sidecar_preserves_pre_window_asof_record_and_age() -> None:
    days = [date(2024, 1, 3), date(2024, 1, 4)]
    source = pl.DataFrame(
        {
            "available_date": [date(2024, 1, 2)],
            "isin": ["BRTESTACNOR1"],
            "liabilities_to_assets": [1.25],
            "liabilities_to_assets_mask": [True],
        }
    )
    result = materialize_known_archive(
        source, days, ["BRTESTACNOR1"], group="fundamentals"
    )
    assert result.valid[:, 0, 0].all()
    assert result.values[:, 0, 0].tolist() == pytest.approx([1.25, 1.25])
    assert result.age_sessions[:, 0, 0].tolist() == [0.0, 1.0]


def test_equal_priority_conflicting_logical_records_are_rejected() -> None:
    day = date(2024, 1, 2)
    source = pl.DataFrame(
        {
            "available_date": [day, day],
            "isin": ["BRTESTACNOR1", "BRTESTACNOR1"],
            "liabilities_to_assets": [1.0, 2.0],
            "liabilities_to_assets_mask": [True, True],
            "__record_family": ["fundamental_filing", "fundamental_filing"],
        }
    )
    with pytest.raises(ValueError, match="conflicting logical records"):
        materialize_known_archive(
            source, [day], ["BRTESTACNOR1"], group="fundamentals"
        )


def test_rebalance_requires_timestamped_current_state_not_legacy_grid() -> None:
    day = date(2024, 1, 2)
    legacy = pl.DataFrame(
        {
            "available_date": [day],
            "decision_idx": [54],
            "isin": ["BRTESTACNOR1"],
            "ibov_current_weight_sqrt": [0.2],
        }
    )
    assert derive_known_archive_features(
        legacy, [day], ["BRTESTACNOR1"], group="rebalance"
    ).is_empty()
    current = legacy.drop("decision_idx").with_columns(
        pl.lit(datetime(2024, 1, 2, 15, 0, tzinfo=ZoneInfo("America/Sao_Paulo")))
        .alias("public_available_at"),
        pl.lit("minute").alias("public_available_at_precision"),
        pl.lit(0.0).alias("public_available_at_processing_latency_seconds"),
    )
    result = materialize_known_archive(
        derive_known_archive_features(
            current, [day], ["BRTESTACNOR1"], group="rebalance"
        ),
        [day],
        ["BRTESTACNOR1"],
        group="rebalance",
    )
    assert result.valid[0, 0, 0]


def test_legacy_clipped_fundamental_is_source_missing(tmp_path) -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    source = pl.DataFrame(
        {
            "available_date": [day for day in days for _ in range(55)],
            "decision_idx": list(range(55)) * len(days),
            "security_id": ["security-one"] * (55 * len(days)),
            "fund_leverage": [float(value) for value in range(55)] * len(days),
            "fund_leverage_mask": [True] * (55 * len(days)),
        }
    )
    path = tmp_path / "fundamentals.parquet"
    source.write_parquet(path)
    assignments = pl.DataFrame(
        {"security_id": ["security-one"], "isin": ["BRTESTACNOR1"]}
    )
    result = _parse_sidecars(
        [f"fundamentals={path}"], days, ["BRTESTACNOR1"], assignments
    )["fundamentals"]
    assert result.feature_names == ()
    assert result.values.shape == (2, 1, 0)
    assert result.source_missing_candidates == (
        "log_market_cap", "book_to_market", "gross_profitability",
        "liabilities_to_assets",
    )
