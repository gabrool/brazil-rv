from copy import deepcopy
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.feature_spec import feature_specs, transform_feature_panel_into
from brazil_rv.v2.round5_cvm import fundamental_features, issuer_market_cap
from brazil_rv.v2.round5_store import align_family


def valuation_fixture():
    sessions = [date(2024, 1, day) for day in (2, 3, 4, 5)]
    document = dict(
        id="1",
        cnpj="00000001000100",
        cvm_code="1",
        kind="DFP",
        version=1,
        reference=date(2023, 12, 31),
        receipt=sessions[0],
        available_index=0,
        shares={"ON": 100.0, "PN": 50.0},
        accounts={},
    )
    identity = [{"isin": "ON", "class": "ON"}, {"isin": "PN", "class": "PN"}]
    market = dict(
        columns={"ON": 0, "PN": 1},
        close=np.array([[10.0, 30.0], [20.0, 40.0], [30.0, 50.0], [40.0, 60.0]]),
        observed=np.ones((4, 2), bool),
        barrier_prefix=np.zeros((5, 2), int),
    )
    return sessions, document, identity, market


def test_issuer_valuation_prices_each_class_and_ignores_duplicate_unit_claims():
    sessions, document, identity, market = valuation_fixture()
    ledger = {("DFP", document["reference"]): document}
    expected = 100 * 20 + 50 * 40
    assert issuer_market_cap(ledger, identity, 2, sessions, market, []) == (expected, 0)
    # A unit is a claim on existing shares, not additional issuer capital.
    identity.append({"isin": "UNIT", "class": "UNIT"})
    assert issuer_market_cap(ledger, identity, 2, sessions, market, []) == (expected, 0)


@pytest.mark.parametrize(
    "gap",
    ["missing_class", "multiple_pn", "subclass", "missing_close", "unit_boundary"],
)
def test_valuation_requires_complete_class_prices_and_unit_continuity(gap):
    sessions, document, identity, market = valuation_fixture()
    if gap == "missing_class":
        identity.pop()
    elif gap == "multiple_pn":
        identity.append({"isin": "OTHER_PN", "class": "PN"})
    elif gap == "subclass":
        identity[1]["preferred_class"] = "A"
    elif gap == "missing_close":
        market["observed"][1, 1] = False
    else:
        market["barrier_prefix"][2:, 1] = 1
    assert issuer_market_cap({1: document}, identity, 2, sessions, market, []) == (
        None,
        None,
    )


def test_multiclass_publication_and_close_timing_through_store_transform(tmp_path):
    sessions, template, _, _ = valuation_fixture()
    documents, rad, identities, isins = [], [], [], []
    for issuer in range(24):
        document = deepcopy(template)
        document.update(
            id=str(issuer + 1), cnpj=f"{issuer + 1:08}000100", cvm_code=str(issuer + 1)
        )
        document["shares"]["ON"] += issuer
        documents.append(document)
        rad.append(
            dict(
                id=document["id"],
                group="structured",
                cvm_code=document["cvm_code"],
                reference=document["reference"],
                version="1",
                receipt=datetime(2024, 1, 2, 15, 44),
            )
        )
        for share_class in ("ON", "PN"):
            isin = f"ISSUER{issuer:02}{share_class}"
            isins.append(isin)
            identities.extend(
                dict(
                    date=day,
                    isin=isin,
                    cnpj=document["cnpj"],
                    cvm_code=document["cvm_code"],
                    sector="Industrial",
                    **{"class": share_class},
                    identity_known_date=sessions[0],
                )
                for day in sessions
            )
    identity = pl.DataFrame(identities)
    market = dict(
        columns={isin: i for i, isin in enumerate(isins)},
        close=np.broadcast_to(np.arange(48, dtype=float) + 10, (4, 48)).copy(),
        observed=np.ones((4, 48), bool),
        barrier_prefix=np.zeros((5, 48), int),
    )

    def transformed(label):
        frame, _ = fundamental_features(
            deepcopy(documents), rad, identity, sessions, market
        )
        path = tmp_path / f"{label}.parquet"
        frame.write_parquet(path)
        raw, valid, age = align_family(
            pl.read_parquet(path), sessions, tuple(isins), ("log_market_cap",)
        )
        values, mask = np.empty_like(raw), np.empty_like(valid)
        transform_feature_panel_into(
            raw,
            valid,
            np.ones((4, 48), bool),
            feature_specs("sidecar_fundamentals", ("log_market_cap",)),
            values,
            mask,
        )
        age[~mask] = -1
        return values, mask, age

    baseline = transformed("baseline")
    market["close"][1, 1] = 1000
    changed = transformed("changed_prior_pn_close")
    for left, right in zip(baseline, changed, strict=True):
        np.testing.assert_array_equal(left[:2], right[:2])
    assert not np.array_equal(baseline[0][2], changed[0][2])
    assert changed[1][2].all() and np.all(changed[2][2] == 2)
    market["observed"][1, 1] = False
    missing = transformed("missing_pn_close")
    assert not missing[1][2, :2].any()
    assert missing[1][2, 2:].all()
    for left, right in zip(baseline, missing, strict=True):
        np.testing.assert_array_equal(left[:2], right[:2])
