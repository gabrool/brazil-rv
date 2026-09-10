"""Reusable 24-issuer receipt proof through actual CVM/store transforms."""

from copy import deepcopy
from datetime import date, datetime, time, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from brazil_rv.v2.feature_spec import feature_specs, transform_feature_panel_into
from brazil_rv.v2.round5_cvm import (
    FEATURES_EVENTS,
    FEATURES_FUNDAMENTALS,
    event_features,
    fundamental_features,
)
from brazil_rv.v2.round5_store import align_family


FUNDAMENTALS = FEATURES_FUNDAMENTALS + (
    "fundamental_financial_flag",
    "fundamental_consolidated_flag",
    "valuation_available_flag",
)
CLOCKS = (("15_44", 2), ("15_45", 3), ("date_only", 3))


def _fixture(clock):
    days = [date(2024, 4, day) for day in (1, 2, 3, 4, 5, 8, 9)]
    isins = tuple(f"BRFIXTURE{i:03}" for i in range(24))
    identities, documents, rad = [], [], []
    quarters = [
        date(year, month, day)
        for year in range(2019, 2024)
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31))
    ]

    def filing(issuer, q, reference, receipt):
        start = date(reference.year, 1, 1)
        year_start = q - (reference.month // 3 - 1)
        income = sum(
            2 + issuer * 0.1 + i * 0.2 + i**2 * 0.07 for i in range(year_start, q + 1)
        )
        revenue = sum(50 + issuer + i * 3 for i in range(year_start, q + 1))
        values = {
            "assets": 1000 + issuer * 40 + q * 10,
            "equity": 200 + issuer * 15 + q * 2,
            "minority_equity": issuer,
            "revenue": revenue,
            "gross_profit": revenue * 0.4,
            "net_income": income,
            "parent_income": income * 0.9,
            "cash_flow": income * 0.5,
        }
        return {
            "id": str(issuer * 100 + q + 1),
            "cnpj": f"{issuer + 1:08}000100",
            "cvm_code": str(issuer + 1),
            "reference": reference,
            "version": 1,
            "kind": "DFP" if reference.month == 12 else "ITR",
            "receipt": receipt,
            "shares": {"ON": 100 + issuer, "PN": 0},
            "accounts": {
                "con": {
                    metric: {
                        "value": value,
                        "start": None
                        if metric in {"assets", "equity", "minority_equity"}
                        else start,
                        "end": reference,
                        "description": metric,
                    }
                    for metric, value in values.items()
                }
            },
        }

    def receipt_row(document):
        return {
            key: document[key]
            for key in ("id", "cvm_code", "reference", "kind", "receipt")
        } | {"group": "structured", "version": "1"}

    for issuer, isin in enumerate(isins):
        for q, reference in enumerate(quarters):
            receipt = datetime.combine(reference + timedelta(days=45), time(15, 44))
            document = filing(issuer, q, reference, receipt)
            documents.append(document)
            rad.append(receipt_row(document))
        identities.extend(
            {
                "date": day,
                "isin": isin,
                "cnpj": f"{issuer + 1:08}000100",
                "cvm_code": str(issuer + 1),
                "sector": "Industrial",
                "class": "ON",
                "identity_known_date": days[0],
            }
            for day in days
        )

    receipt = (
        days[2]
        if clock == "date_only"
        else datetime.combine(days[2], time(15, 44 if clock == "15_44" else 45))
    )
    future = filing(0, 20, date(2024, 3, 31), receipt)
    documents.append(future)
    if clock != "date_only":
        rad.append(receipt_row(future))
    # A date-only financial header uses fundamental_features' real fallback;
    # it does not acquire a fabricated precise RAD event timestamp.
    rad.append(
        {
            "id": "future_fact",
            "cvm_code": "1",
            "group": "material_fact",
            "kind": "Fato Relevante",
            "subject": "Dividendos e recompra",
            "reference": None,
            "version": "",
            "receipt": receipt,
        }
    )
    market = {
        "columns": {isin: i for i, isin in enumerate(isins)},
        "close": np.broadcast_to(
            np.arange(24, dtype=float) + 10, (len(days), 24)
        ).copy(),
        "observed": np.ones((len(days), 24), bool),
        "barrier_prefix": np.zeros((len(days) + 1, 24), np.int32),
    }
    return days, isins, pl.DataFrame(identities), documents, rad, market


def _panel(frame, path, family, names, days, isins):
    frame.write_parquet(path)
    raw, valid, age = align_family(pl.read_parquet(path), days, isins, names)
    specs = feature_specs(f"sidecar_{family}", names)
    assert all(
        spec.minimum_support == 20 for spec in specs if spec.transform == "rank_gauss"
    )
    values, mask = np.empty_like(raw), np.empty_like(valid)
    transform_feature_panel_into(
        raw, valid, np.ones(raw.shape[:2], bool), specs, values, mask
    )
    return {
        "raw": raw,
        "raw_mask": valid,
        "raw_age": age,
        "values": values,
        "mask": mask,
        "age": np.where(mask, age, -1),
    }


def _run(root, variant, fixture):
    days, isins, identity, documents, rad, market = deepcopy(fixture)
    if variant == "financial_values":
        future = documents[-1]
        future["shares"]["ON"] *= 100
        for metric in ("equity", "parent_income", "net_income"):
            future["accounts"]["con"][metric]["value"] *= 1000
    elif variant == "financial_missing":
        # A newly reported PN class has no dated class price in this fixture.
        documents[-1]["shares"]["PN"] = 50
        del documents[-1]["accounts"]["con"]["equity"]
    elif variant == "material_fact_missing":
        rad = [row for row in rad if row["id"] != "future_fact"]
    fundamentals, audit = fundamental_features(documents, rad, identity, days, market)
    events = (
        identity.select("date", "isin", "cvm_code")
        .join(event_features(rad, days), on=["date", "cvm_code"], how="left")
        .drop("cvm_code")
    )
    root.mkdir(parents=True, exist_ok=True)
    return {
        "fundamentals": _panel(
            fundamentals,
            root / "fundamentals.parquet",
            "fundamentals",
            FUNDAMENTALS,
            days,
            isins,
        ),
        "events": _panel(
            events, root / "events.parquet", "events", FEATURES_EVENTS, days, isins
        ),
    }, audit


def _before_equal(left, right, first):
    for family in left:
        for array in left[family]:
            # Includes float representations, every mask and source-age field.
            assert (
                left[family][array][:first].tobytes()
                == right[family][array][:first].tobytes()
            ), (family, array)


def _clock_evidence(output, clock, first):
    fixture = _fixture(clock)
    baseline, audit = _run(output / "baseline", "baseline", fixture)
    changes = {}
    for variant in ("financial_values", "financial_missing", "material_fact_missing"):
        changed, _ = _run(output / variant, variant, fixture)
        _before_equal(baseline, changed, first)
        family = "events" if variant == "material_fact_missing" else "fundamentals"
        assert not np.array_equal(
            baseline[family]["raw"][first], changed[family]["raw"][first]
        )
        assert not np.array_equal(
            baseline[family]["values"][first], changed[family]["values"][first]
        )
        changes[variant] = changed

    f = baseline["fundamentals"]
    # All eight numeric fundamentals, including seasonal surprise, really have
    # >=20-name transformed support; these are not all-missing rank fixtures.
    assert f["mask"][1, :, :8].all()
    assert f["mask"][first, :, :8].all()
    for name in (
        "log_market_cap",
        "book_to_market",
        "liabilities_to_assets",
        "statement_age_sessions",
    ):
        column = FUNDAMENTALS.index(name)
        for panel in (f, changes["financial_values"]["fundamentals"]):
            assert panel["raw_age"][first, 0, column] == 0
            assert panel["age"][first, 0, column] == 0
            assert panel["age"][first + 1, 0, column] == 1
    # The TTM bridge also consumes already published annual/prior-YTD history.
    assert f["age"][first, 0, FUNDAMENTALS.index("earnings_yield_ttm")] == first
    missing = changes["financial_missing"]["fundamentals"]
    for name in (
        "log_market_cap",
        "book_to_market",
        "earnings_yield_ttm",
        "liabilities_to_assets",
    ):
        column = FUNDAMENTALS.index(name)
        assert not missing["raw_mask"][first, 0, column]
        assert not missing["mask"][first, 0, column]
        assert missing["raw_age"][first, 0, column] == -1
        assert missing["age"][first, 0, column] == -1
        assert missing["mask"][
            first, 1:, column
        ].all()  # Remaining23 pass actual20-name minimum.

    e, removed = baseline["events"], changes["material_fact_missing"]["events"]
    for name in ("sessions_since_material_fact", "dividend_announcement_age"):
        column = FEATURES_EVENTS.index(name)
        assert e["raw_mask"][first, 0, column] and e["mask"][first, 0, column]
        assert e["raw"][first, 0, column] == 0
        assert e["age"][first, 0, column] == 0
        assert e["age"][first + 1, 0, column] == 1
        assert not removed["mask"][first, 0, column]
        assert removed["age"][first, 0, column] == -1
    count = FEATURES_EVENTS.index("material_fact_count_20")
    np.testing.assert_array_equal(e["raw"][first, :, count], [1] + [0] * 23)
    assert e["mask"][first, :, count].all()
    assert not np.array_equal(
        e["values"][first, :, count], removed["values"][first, :, count]
    )
    if clock == "date_only":
        assert audit["date_only_receipt_documents"] == 1
    else:
        assert audit["date_only_receipt_documents"] == 0
        assert e["raw"][first, 0, FEATURES_EVENTS.index("filing_is_dfp")] == 0
    return {
        "clock": clock,
        "first_eligible_index": first,
        "first_eligible_date": str(fixture[0][first]),
        "issuers": 24,
        "fundamental_fields": list(FUNDAMENTALS),
        "event_fields": list(FEATURES_EVENTS),
        "minimum_rank_names": 20,
        "all_eight_numeric_fundamentals_observed_at_boundary": True,
        "prearrival_raw_transformed_masks_ages_bit_identical": True,
        "first_eligible_value_and_mask_changes_verified": True,
        "current_filing_age_zero_ttm_oldest_source_age_preserved": True,
        "mutations": list(changes),
    }


def receipt_transform_evidence(output_root: Path) -> dict:
    """Run the same committed proof from an offline acceptance artifact script."""
    cases = [
        _clock_evidence(output_root / clock, clock, first) for clock, first in CLOCKS
    ]
    files = {
        path.relative_to(output_root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output_root.rglob("*.parquet"))
    }
    result = {"cases": cases, "fixture_parquet_sha256": files}
    (output_root / "receipt_transform_evidence.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


@pytest.mark.parametrize(("clock", "first"), CLOCKS)
def test_cvm_receipts_survive_dated_join_and_actual_rank_transforms(
    tmp_path, clock, first
):
    _clock_evidence(tmp_path, clock, first)
