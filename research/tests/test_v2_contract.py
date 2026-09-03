from brazil_rv.v2.contract import (
    DECISION_MINUTE_INDEX,
    FAST_REAL_PATCHES,
    HORIZONS,
    INTRADAY_DAILY_FEATURES,
    SLOW_FEATURES,
    StoreSchema,
)


def test_v2_contract_has_frozen_feature_and_horizon_axes() -> None:
    schema = StoreSchema()
    assert len(SLOW_FEATURES) == 32
    assert len(INTRADAY_DAILY_FEATURES) == 20
    assert schema.horizons == HORIZONS == (1, 2, 3, 5, 10)


def test_v2_decision_is_single_1545_cutoff() -> None:
    assert DECISION_MINUTE_INDEX == 345
    assert FAST_REAL_PATCHES == 69
