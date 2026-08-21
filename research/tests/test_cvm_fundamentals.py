from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from brazil_rv.preprocessing.cvm_fundamentals import (
    AccountValue,
    FactorState,
    FcaDocument,
    FcaSecurity,
    FilingDocument,
    VALUE_FEATURES,
    _compute_factor_state,
    _receipt_availability,
    _quarterly_values,
    _scale_value,
    _ttm_values,
    build_intraday_frame,
    build_factor_events,
)

ISSUER = "001234"
CNPJ = "12345678000199"
SECURITY_ON = "ISIN:BRTESTACNOR1"
SECURITY_PN = "ISIN:BRTESTACNPR8"


def _document(
    reference_date: date,
    available_date: date,
    *,
    version: int = 1,
    basis: str = "con",
    assets: float = 1_000.0,
    equity: float = 400.0,
    revenue: float = 100.0,
    net_income: float = 10.0,
    cash: float = 8.0,
) -> FilingDocument:
    fiscal_start = date(reference_date.year, 1, 1)
    document = FilingDocument(
        cvm_code=ISSUER,
        cnpj=CNPJ,
        category="DFP" if reference_date.month == 12 else "ITR",
        reference_date=reference_date,
        version=version,
        sequence_id=f"{reference_date:%Y%m%d}{version}",
        receipt_date=available_date - timedelta(days=1),
        available_date=available_date,
        decision_idx=0,
        receipt_order=datetime.combine(available_date, datetime.min.time()),
    )
    document.values[basis] = {
        "assets": AccountValue(assets, None, reference_date),
        "equity": AccountValue(equity, None, reference_date),
        "revenue": AccountValue(revenue, fiscal_start, reference_date),
        "net_income": AccountValue(net_income, fiscal_start, reference_date),
        "operating_cash_flow": AccountValue(cash, fiscal_start, reference_date),
    }
    return document


def _quarter_documents(*, basis: str = "con") -> list[FilingDocument]:
    dates = [
        date(2019, 3, 31),
        date(2019, 6, 30),
        date(2019, 9, 30),
        date(2019, 12, 31),
        date(2020, 3, 31),
        date(2020, 6, 30),
    ]
    revenues = [10.0, 30.0, 60.0, 100.0, 15.0, 35.0]
    incomes = [1.0, 3.0, 6.0, 10.0, 1.5, 3.5]
    cash = [0.8, 2.5, 5.0, 8.0, 1.2, 3.0]
    return [
        _document(
            reference,
            date(2021, 1, index + 2),
            basis=basis,
            assets=900.0 + index * 20,
            equity=360.0 + index * 8,
            revenue=revenues[index],
            net_income=incomes[index],
            cash=cash[index],
        )
        for index, reference in enumerate(dates)
    ]


def test_scale_and_cumulative_quarters_before_ttm() -> None:
    assert _scale_value("1.5", "REAL", "MIL") == 1_500.0
    assert _scale_value("1.5", "REAL", "UNIDADE") == 1.5
    assert _scale_value("1.5", "USD", "MIL") is None

    documents = _quarter_documents()
    quarters = _quarterly_values(documents, "con", "revenue")
    assert quarters[date(2019, 3, 31)] == 10.0
    assert quarters[date(2019, 6, 30)] == 20.0
    assert quarters[date(2019, 9, 30)] == 30.0
    assert quarters[date(2019, 12, 31)] == 40.0
    assert quarters[date(2020, 3, 31)] == 15.0
    assert quarters[date(2020, 6, 30)] == 20.0
    ttm = _ttm_values(quarters)
    assert ttm[date(2019, 12, 31)] == 100.0
    assert ttm[date(2020, 3, 31)] == 105.0
    assert ttm[date(2020, 6, 30)] == 105.0


def test_version_revision_is_available_only_from_its_receipt() -> None:
    documents = _quarter_documents()
    original = documents[-1]
    original.available_date = date(2021, 2, 1)
    revised = _document(
        date(2020, 6, 30),
        date(2021, 2, 4),
        version=2,
        assets=1_050.0,
        equity=420.0,
        revenue=45.0,
        net_income=5.0,
        cash=4.0,
    )
    events = build_factor_events([*documents, revised])[ISSUER]
    before = [state for state in events if state.available_date <= date(2021, 2, 3)][-1]
    after = [state for state in events if state.available_date <= date(2021, 2, 4)][-1]
    assert before.available_date == date(2021, 2, 1)
    assert after.available_date == date(2021, 2, 4)
    assert (
        before.values["fund_net_margin_ttm"][0]
        != after.values["fund_net_margin_ttm"][0]
    )

    lower_late = _document(
        date(2020, 6, 30),
        date(2021, 2, 5),
        version=1,
        revenue=999.0,
        net_income=999.0,
    )
    with_lower = build_factor_events([*documents, revised, lower_late])[ISSUER]
    assert with_lower[-1].available_date == date(2021, 2, 4)


def test_consolidated_preferred_and_individual_fallback_is_explicit() -> None:
    individual = _quarter_documents(basis="ind")
    state = _compute_factor_state(individual, date(2021, 2, 1), 0, date(2021, 1, 31))
    assert state is not None and not state.consolidated
    consolidated = _quarter_documents()
    preferred = _compute_factor_state(
        [*individual, *consolidated], date(2021, 2, 1), 0, date(2021, 1, 31)
    )
    assert preferred is not None and preferred.consolidated


def _factor_state(
    available: date, margin: float = 0.1, *, decision_idx: int = 0
) -> FactorState:
    values = {feature: (0.2, True) for feature in VALUE_FEATURES}
    values["fund_net_margin_ttm"] = (margin, True)
    return FactorState(
        available_date=available,
        decision_idx=decision_idx,
        filing_date=available - timedelta(days=1),
        reference_date=date(2020, 12, 31),
        consolidated=True,
        values=values,
    )


def _fca(sector: str = "Comércio") -> FcaDocument:
    return FcaDocument(
        cvm_code=ISSUER,
        cnpj=CNPJ,
        reference_date=date(2020, 1, 1),
        version=1,
        sequence_id="1",
        available_date=date(2021, 1, 4),
        sector=sector,
        securities=(
            FcaSecurity("TEST3", date(2020, 1, 1), None),
            FcaSecurity("TEST4", date(2020, 1, 1), None),
        ),
    )


def _cotahist() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "ticker": ["TEST3", "TEST4"],
            "security_id": [SECURITY_ON, SECURITY_PN],
        },
        schema_overrides={"trade_date": pl.Date},
    )


def test_intraday_identity_broadcast_financial_masks_and_future_mutation() -> None:
    sessions = [date(2021, 1, 4) + timedelta(days=index) for index in range(6)]
    states = {ISSUER: [_factor_state(sessions[1])]}
    frame, _ = build_intraday_frame(
        states,
        [_fca("Bancos")],
        _cotahist(),
        sessions,
        available_start=sessions[0],
        available_end=sessions[-1],
    )
    assert frame.get_column("available_date").min() == sessions[1]
    first = frame.filter(pl.col("available_date") == sessions[1])
    assert set(first.get_column("security_id")) == {SECURITY_ON, SECURITY_PN}
    assert first.get_column("fund_financial_sector").eq(1).all()
    for feature in (
        "fund_net_margin_ttm",
        "fund_sales_growth_yoy",
        "fund_accruals_assets_ttm",
    ):
        assert not first.get_column(f"{feature}_mask").any()
        assert first.get_column(feature).eq(0).all()
    assert first.get_column("fund_roa_ttm_mask").all()

    future = _factor_state(sessions[4], margin=0.9)
    changed, _ = build_intraday_frame(
        {ISSUER: [states[ISSUER][0], future]},
        [_fca()],
        _cotahist(),
        sessions,
        available_start=sessions[0],
        available_end=sessions[-1],
    )
    parent, _ = build_intraday_frame(
        states,
        [_fca()],
        _cotahist(),
        sessions,
        available_start=sessions[0],
        available_end=sessions[-1],
    )
    assert changed.filter(pl.col("available_date") < sessions[4]).equals(
        parent.filter(pl.col("available_date") < sessions[4])
    )


def test_midday_revision_starts_at_its_exact_available_decision() -> None:
    sessions = [date(2021, 1, 4), date(2021, 1, 5)]
    initial = _factor_state(sessions[0], margin=0.1)
    revised = _factor_state(sessions[1], margin=0.9, decision_idx=2)
    frame, _ = build_intraday_frame(
        {ISSUER: [initial, revised]},
        [_fca()],
        _cotahist(),
        sessions,
        available_start=sessions[1],
        available_end=sessions[1],
    )
    before = frame.filter(
        (pl.col("decision_idx") == 1) & (pl.col("security_id") == SECURITY_ON)
    )
    after = frame.filter(
        (pl.col("decision_idx") == 2) & (pl.col("security_id") == SECURITY_ON)
    )
    assert before.get_column("fund_net_margin_ttm").item() == pytest.approx(0.1)
    assert after.get_column("fund_net_margin_ttm").item() == pytest.approx(0.9)


def test_exact_rad_conversion_uses_first_strictly_later_decision() -> None:
    sessions = [date(2024, 5, 6), date(2024, 5, 7), date(2024, 5, 8)]
    assert _receipt_availability(datetime(2024, 5, 6, 10, 14), sessions) == (
        sessions[0],
        0,
    )
    assert _receipt_availability(datetime(2024, 5, 6, 10, 15), sessions) == (
        sessions[0],
        1,
    )
    assert _receipt_availability(datetime(2024, 5, 6, 14, 44), sessions) == (
        sessions[0],
        54,
    )
    assert _receipt_availability(datetime(2024, 5, 6, 14, 45), sessions) == (
        sessions[1],
        0,
    )
    assert _receipt_availability(datetime(2024, 5, 5, 9, 0), sessions) == (
        sessions[0],
        0,
    )
