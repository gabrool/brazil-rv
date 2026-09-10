import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.round5_derived import (
    beta_source_age,
    cdi_axis,
    identity_axes,
    joined_clock_fixture,
    write_family,
)
from brazil_rv.v2.feature_spec import feature_specs, transform_feature_panel_into
from brazil_rv.v2.round5_exposures import (
    exposure_panel,
    market_shocks,
    sector_relative_panel,
    shock_axes,
)
from brazil_rv.v2.round5_magnitude import (
    FEATURE_NAMES as MAGNITUDE_NAMES,
    magnitude_panel,
)
from brazil_rv.v2.round5_market import OBSERVATION_SCHEMA
from brazil_rv.v2.round5_cvm import build_identity
from brazil_rv.v2.round5_store import align_family


def test_joined_market_clocks_enter_at_first_eligible_decision():
    assert len(joined_clock_fixture()["market_first_decision_mutations"]) == 2


def test_sector_identity_mutation_enters_on_receipt_without_prior_projection():
    dates = [date(2024, 1, 2) + timedelta(days=i) for i in range(4)]
    isins = ("A", "B", "C", "D")
    rows = [
        dict(
            date=day,
            isin=isin,
            cnpj=str(n) * 14,
            cvm_code=str(n),
            sector="original",
            identity_known_date=dates[0],
            sector_known_date=dates[0],
        )
        for day in dates
        for n, isin in enumerate(isins)
    ]
    values = np.broadcast_to(np.arange(4)[None, :, None], (4, 4, 3)).astype(float)
    valid, active = np.ones_like(values, bool), np.ones((4, 4), bool)
    sectors, issuers = identity_axes(pl.DataFrame(rows), dates, isins)
    before = sector_relative_panel(values, valid, sectors, issuers, active)
    for row in rows:
        if row["isin"] == "D" and row["date"] >= dates[2]:
            row.update(
                sector="changed",
                identity_known_date=dates[2],
                sector_known_date=dates[2],
            )
    sectors, issuers = identity_axes(pl.DataFrame(rows), dates, isins)
    after = sector_relative_panel(values, valid, sectors, issuers, active)
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left[:2], right[:2])
    assert before[0][2, 0, 0] != after[0][2, 0, 0]
    assert not after[1][2:, 3].any()
    rows[-1]["identity_known_date"] = dates[-1] + timedelta(days=1)
    with pytest.raises(ValueError, match="precedes"):
        identity_axes(pl.DataFrame(rows), dates, isins)
    rows[-1]["identity_known_date"] = dates[2]
    rows[-1]["sector_known_date"] = dates[-1] + timedelta(days=1)
    with pytest.raises(ValueError, match="translation"):
        identity_axes(pl.DataFrame(rows), dates, isins)


def test_beta_age_uses_last_pair_strictly_before_decision():
    wealth = np.array([[10], [11], [12], [np.nan], [14], [15.0]])
    seen = np.isfinite(wealth)
    valid = np.ones_like(seen)
    valid[:2] = False
    bova = np.arange(100, 106, dtype=float)
    age = beta_source_age(wealth, seen, bova, valid)
    assert age[:, 0].tolist() == [-1, -1, 1, 1, 2, 3]
    wealth[5] = 200
    np.testing.assert_array_equal(age, beta_source_age(wealth, seen, bova, valid))


def test_cdi_join_preserves_accepted_values_and_rejects_revision(tmp_path):
    days = [date(2016, 7, 18), date(2016, 7, 19)]
    early, accepted = tmp_path / "early.json", tmp_path / "accepted.parquet"
    early.write_text(json.dumps([{"data": "18/07/2016", "valor": "0.05"}]))
    pl.DataFrame(
        {"trade_date": days, "daily_cdi_rate": [0.0005, 0.0006]}
    ).write_parquet(accepted)
    values, audit = cdi_axis(early, accepted, days)
    assert values.tolist() == [0.0005, 0.0006]
    assert audit["overlap_rows"] == 1
    early.write_text(json.dumps([{"data": "18/07/2016", "valor": "0.06"}]))
    with pytest.raises(ValueError, match="conflicts"):
        cdi_axis(early, accepted, days)


def test_family_archive_keeps_missingness_and_physical_values(tmp_path):
    values = np.array([[[12.5], [99.0]], [[25.0], [0.0]]], np.float32)
    valid = np.array([[[True], [False]], [[True], [True]]])
    ages = np.array([[[0], [-1]], [[1], [0]]], np.float32)
    active = np.array([[True, True], [True, False]])
    result = write_family(
        tmp_path,
        "magnitudes",
        ("economic_beta_60",),
        values,
        valid,
        ages,
        [date(2024, 1, 2), date(2024, 1, 3)],
        ("A", "B"),
        active,
        {},
        {"commit": "fixture"},
        joined_clock_fixture(),
    )
    frame = pl.read_parquet(result["data"]["path"])
    assert frame["isin"].to_list() == ["A", "A"]
    assert frame["economic_beta_60"].to_list() == [12.5, 25.0]
    assert frame["economic_beta_60_age_sessions"].to_list() == [0, 1]


def _transformed_archive(root, family, names, panel, days, isins):
    values, valid, ages = panel
    active = np.ones(values.shape[:2], bool)
    record = write_family(
        root,
        family,
        names,
        values,
        valid,
        ages,
        days,
        isins,
        active,
        {},
        {"commit": "fixture"},
        {},
    )
    raw, observed, source_age = align_family(
        pl.read_parquet(record["data"]["path"]), days, isins, names
    )
    output, mask = np.empty_like(raw), np.empty_like(observed)
    transform_feature_panel_into(
        raw, observed, active, feature_specs(f"sidecar_{family}", names), output, mask
    )
    source_age[~mask] = -1
    return output, mask, source_age


def _assert_first_change(before, after, first, *, mask_changes=False):
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left[:first], right[:first])
    assert not np.array_equal(before[0][first], after[0][first])
    if mask_changes:
        assert not np.array_equal(before[1][first], after[1][first])


def test_transformed_market_archive_preserves_publication_and_native_common_state(
    tmp_path,
):
    days = np.busday_offset("2024-01-02", np.arange(130)).astype(object).tolist()
    isins = tuple(f"BRFIXTURE{n:03}" for n in range(24))
    rng = np.random.default_rng(13)
    levels = np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    stock_returns = rng.normal(0, 0.01, (len(days), len(isins)))
    rows = [
        dict(
            series="ptax_brl_per_usd",
            reference_date=day,
            available_at=datetime.combine(day, datetime.min.time(), UTC).replace(
                hour=21
            ),
            value=levels[i],
            source_file="fixture",
        )
        for i, day in enumerate(days)
    ]
    empty_returns = pl.DataFrame(schema={"series": pl.String})
    names = ("shock_fx_1", "exposure_fx", "exposure_fx_times_shock_1")

    def archive(label):
        current, age, history, published = shock_axes(
            market_shocks(pl.DataFrame(rows, schema=OBSERVATION_SCHEMA), empty_returns),
            days,
        )["fx"]
        beta, beta_mask, beta_age = exposure_panel(
            stock_returns,
            np.ones_like(stock_returns, bool),
            history,
            published,
            np.full(stock_returns.shape, "sector", object),
            np.broadcast_to(isins, stock_returns.shape),
        )
        shock = np.broadcast_to(current[:, :1], beta.shape)
        shock_age = np.broadcast_to(age[:, :1], beta.shape)
        values = np.stack((shock, beta, beta * shock), axis=-1).astype(np.float32)
        valid = np.stack(
            (np.isfinite(shock), beta_mask, beta_mask & np.isfinite(shock)), axis=-1
        )
        ages = np.stack((shock_age, beta_age, np.maximum(shock_age, beta_age)), axis=-1)
        return _transformed_archive(
            tmp_path / label, "cross_market", names, (values, valid, ages), days, isins
        )

    before = archive("before")
    rows[90]["value"] *= 1.1
    changed = archive("changed")
    _assert_first_change(before, changed, 91)
    # A common scalar keeps its nonzero physical value across all active names.
    assert np.unique(before[0][91, :, 0]).size == 1 and before[0][91, 0, 0] != 0
    assert before[1][91].all()
    rows[90]["value"] = np.nan
    missing = archive("unavailable_fixing")
    _assert_first_change(before, missing, 91, mask_changes=True)


def test_transformed_sector_archive_changes_only_at_known_classification(tmp_path):
    days = np.busday_offset("2024-01-02", np.arange(5)).astype(object).tolist()
    isins = tuple(f"BRFIXTURE{n:03}" for n in range(24))
    documents = [
        dict(
            id=str(n + 1),
            cnpj=f"{n:014}",
            cvm_code=str(n),
            reference=date(2023, 1, 1),
            version=1,
            receipt=days[0],
            available_index=1,
            sector_code=None if n == 0 else "17",
            sector_label="Annual label" if n == 0 else "Direct label",
            securities=[
                dict(
                    ticker=f"STOCK{n}",
                    **{"class": "ON"},
                    preferred_class="",
                    unit_composition="",
                    start=date(2000, 1, 1),
                    end=date.max,
                )
            ],
        )
        for n, isin in enumerate(isins)
    ]
    observed = pl.DataFrame(
        {
            "trade_date": [days[0]] * len(isins),
            "isin": isins,
            "ticker": [f"STOCK{n}" for n in range(len(isins))],
            "security_spec_base": ["ON"] * len(isins),
        }
    )
    values = np.broadcast_to(np.arange(24)[None, :, None], (5, 24, 3)).astype(float)
    names = (
        "name_minus_sector_return_5",
        "name_minus_sector_return_21",
        "sector_momentum_12_1",
    )

    def archive(label, evidence):
        identity = build_identity([*documents, *evidence], observed, days, list(isins))
        sectors, issuers = identity_axes(identity, days, isins)
        panel, mask = sector_relative_panel(
            values,
            np.ones_like(values, bool),
            sectors,
            issuers,
            np.ones(values.shape[:2], bool),
        )
        ages = np.where(mask, 1, -1)
        transformed = _transformed_archive(
            tmp_path / label,
            "sector",
            names,
            (panel, mask, ages),
            days,
            isins,
        )
        return (panel, mask, ages), transformed

    future_mapping = dict(
        id="100",
        receipt=days[2],
        available_index=3,
        version=1,
        sector_code="17",
        sector_label="Annual label",
    )
    before = archive("before", [])
    newly_known = archive("future_mapping", [future_mapping])
    for old, new in zip(before, newly_known, strict=True):
        _assert_first_change(old, new, 3, mask_changes=True)
    # Translation admitted today enables yesterday's return, without making it age0.
    assert newly_known[1][1][3].all()
    assert np.all(newly_known[1][2][3] == 1)
    prior_mapping = {**future_mapping, "receipt": days[0], "available_index": 1}
    future_conflict = {**future_mapping, "id": "101", "sector_code": "18"}
    known = archive("prior_mapping", [prior_mapping])
    conflicted = archive("future_conflict", [prior_mapping, future_conflict])
    for old, new in zip(known, conflicted, strict=True):
        _assert_first_change(old, new, 3, mask_changes=True)
    assert conflicted[1][1][3, 1:].all()  # The23 other names retain rank support.
    assert np.all(conflicted[1][2][3, 1:] == 1)
    future_classification = deepcopy(documents[1])
    future_classification.update(
        id="102", version=2, receipt=days[2], available_index=3, sector_code="18"
    )
    classified = archive(
        "future_classification", [prior_mapping, future_classification]
    )
    for old, new in zip(known, classified, strict=True):
        _assert_first_change(old, new, 3, mask_changes=True)


def test_transformed_magnitude_archive_uses_next_close_and_current_dated_beta(tmp_path):
    rng = np.random.default_rng(4)
    close = np.exp(np.cumsum(rng.normal(0, 0.02, (65, 24)), axis=0))
    days = np.busday_offset("2024-01-02", np.arange(65)).astype(object).tolist()
    isins = tuple(f"BRFIXTURE{n:03}" for n in range(24))
    inputs = dict(
        wealth_open=close * 0.998,
        wealth_high=close * 1.012,
        wealth_low=close * 0.988,
        wealth_close=close,
        wealth_valid=np.ones(close.shape, bool),
        volume_brl=np.full(close.shape, 1e7),
        activity_valid=np.ones(close.shape, bool),
        daily_feature_valid=np.ones((*close.shape, 3), bool),
        slow_age_sessions=np.ones((*close.shape, 3), np.float32),
        slow_feature_names=("log_return_1", "yang_zhang_vol_20", "log_volume_mean_20"),
        economic_beta=np.full(close.shape, 1.2),
        economic_beta_valid=np.ones(close.shape, bool),
        economic_beta_age_sessions=np.ones(close.shape, np.float32),
    )

    def archive(label):
        return _transformed_archive(
            tmp_path / label,
            "magnitudes",
            MAGNITUDE_NAMES,
            magnitude_panel(**inputs),
            days,
            isins,
        )

    before = archive("before")
    inputs["wealth_close"][45, 0] *= 1.01
    changed = archive("changed_close")
    _assert_first_change(before, changed, 46)
    inputs["wealth_valid"][45, 0] = False
    missing = archive("missing_close")
    _assert_first_change(before, missing, 46, mask_changes=True)
    inputs["economic_beta"][45, 0] = 2
    beta = archive("changed_beta")
    _assert_first_change(missing, beta, 45)
