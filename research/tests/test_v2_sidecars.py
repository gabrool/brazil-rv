from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.build_store import _parse_sidecars, _sidecar_coverage_table
from brazil_rv.v2.sidecars import (
    available_archive_mapping,
    derive_known_archive_features,
    materialize_known_archive,
    materialize_sidecar,
)


def test_transformed_lending_proxy_is_not_relabelled_as_exact_rate() -> None:
    source = pl.DataFrame(
        {
            "available_date": [date(2024, 1, 2)],
            "isin": ["BRTESTACNOR1"],
            "lending_taker_fee_level_log_tanh": [0.25],
        }
    )
    mapping = available_archive_mapping("lending", source.columns)
    assert mapping["loan_balance_to_volume_20"] is None
    result = materialize_known_archive(
        source, [date(2024, 1, 2)], ["BRTESTACNOR1"], group="lending"
    )
    assert result.valid.sum() == 0
    assert not result.valid[0, 0, 0]


def test_unavailable_option_fields_stay_masked_and_leverage_is_exact() -> None:
    options = available_archive_mapping(
        "options",
        [
            "options_put_call_oi_log_ratio_tanh",
            "options_oi_change_to_stock_adv20_tanh",
            "options_atm_iv_prior20_robust_z_scaled",
            "options_put_skew_tanh",
        ],
    )
    assert all(source is None for source in options.values())
    fundamentals = available_archive_mapping(
        "fundamentals", ["fund_leverage"]
    )
    assert fundamentals["leverage"] == "fund_leverage"
    assert sum(source is not None for source in fundamentals.values()) == 1


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


def test_intraday_archive_is_collapsed_to_latest_snapshot(tmp_path) -> None:
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
    assert result.valid[:, 0, 3].all()
    assert result.values[:, 0, 3].tolist() == [54.0, 54.0]
