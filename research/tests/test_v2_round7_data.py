from datetime import date, datetime, timedelta

import numpy as np
import polars as pl

from brazil_rv.v2.round7_data import NATIVE_FIELDS, corroborate_u2, native_fundamentals


def test_native_fields_retain_negative_sparse_values_and_source_ages():
    fields = [n for n in NATIVE_FIELDS if n != "earnings_negative_flag"]
    frame = pl.DataFrame(
        {
            "date": [date(2020, 1, 2), date(2020, 1, 3)],
            "isin": ["A", "A"],
            **{n: [-0.2, 0.5] for n in fields},
            **{n + "_age_sessions": [3.0, 4.0] for n in fields},
        }
    )
    result = native_fundamentals(frame)
    assert result["earnings_yield_ttm"].to_list() == [-0.2, 0.5]
    assert result["earnings_negative_flag"].to_list() == [1.0, 0.0]
    assert result["book_to_market"].to_list()[0] is None
    assert np.isclose(result["book_to_market"][1], np.log(0.5))
    assert result["earnings_negative_flag_age_sessions"].to_list() == [3.0, 4.0]
    mutated = frame.with_columns(
        pl.when(pl.col("date") > date(2020, 1, 2))
        .then(100.0)
        .otherwise(pl.col("earnings_yield_ttm"))
        .alias("earnings_yield_ttm")
    )
    assert native_fundamentals(mutated).head(1).equals(result.head(1))


def test_u2_requires_relevant_nearby_evidence_and_retains_cash_terms():
    dates = np.array(
        [date(2020, 1, 1) + timedelta(days=i) for i in range(70)], dtype="datetime64[D]"
    )
    terms = pl.DataFrame(
        {
            "isin": ["A", "B", "C", "D", "E", "A"],
            "effective_date": [date(2020, 2, 1)] * 6,
            "evidence": ["U2:{}"] * 5 + ["C1:{}"],
            "shares_per_prior_share": [2.0] * 5 + [1.0],
        }
    )
    provider = pl.DataFrame(
        {
            "isin": ["A", "B"],
            "split_factor": [1.0, 2.0],
            "ex_date": [date(2020, 2, 1)] * 2,
            "source": ["archive"] * 2,
        }
    )
    identity = pl.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 2, 2)],
            "isin": ["C", "E"],
            "cvm_code": ["001234"] * 2,
            "identity_known_date": [date(2020, 1, 1), date(2020, 2, 2)],
        }
    )
    filing = {
        "kind": "Aviso",
        "subject": "Grupamento",
        "cvm_code": "001234",
        "receipt": datetime(2020, 2, 2, 12),
        "id": "1",
        "source": "IPE",
    }
    changed = np.zeros((70, 5), bool)
    changed[33, 3] = True
    reports = corroborate_u2(
        terms, provider, identity, [filing], dates, ("A", "B", "C", "D", "E"), changed
    )
    assert [r["classification"] for r in reports] == [
        "large_move_no_action",
        "corroborated_U2",
        "corroborated_U2",
        "corroborated_U2",
        "large_move_no_action",
    ]
    assert len(reports) == 5  # Cash terms never enter the unit reclassification.
    assert reports[2]["corroboration"][0]["offset_sessions"] == 1
