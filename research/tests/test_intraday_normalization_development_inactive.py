from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import brazil_rv.preprocessing.intraday_normalization as normalization
import brazil_rv.preprocessing.intraday_normalization_variants as variants
from brazil_rv.modeling.contract import TEST_END, TEST_START, VALIDATION_END
from brazil_rv.preprocessing.contract import EQUITY_SESSION_MINUTES, EXPECTED_EQUITIES
from brazil_rv.preprocessing.io import (
    SOURCE_COLUMNS,
    cotahist_files,
    load_market_dates_and_security_dates,
)
from brazil_rv.preprocessing.intraday_normalization import (
    AFFECTED_DYNAMIC_CHANNELS,
    PROFILE_BIN_COUNT,
    build_equity_tod_profile,
    equity_source_hashes,
    iter_reconstructed_equities,
    load_source_context,
)

POST_VALIDATION_IDS = (
    "ISIN:BRAXIAACNOR0",
    "ISIN:BRAXIAACNPC9",
    "ISIN:BREMBJACNOR1",
    "ISIN:BRMBRFACNOR1",
    "ISIN:BRNATUACNOR6",
)
ACTIVE_ID = "ISIN:ACTIVE000"


def _write_development_context(
    tmp_path: Path,
    *,
    contradiction: str | None = None,
    extra_development_date: bool = False,
) -> SimpleNamespace:
    parent = tmp_path / "parent"
    assignments_dir = tmp_path / "assignments"
    cotahist_dir = tmp_path / "cotahist"
    universe_dir = tmp_path / "universe"
    sources_dir = tmp_path / "sources"
    for directory in (
        parent,
        assignments_dir,
        cotahist_dir / "year=2025",
        cotahist_dir / "year=2026",
        universe_dir,
        sources_dir,
    ):
        directory.mkdir(parents=True)

    filler = tuple(
        f"ISIN:ZZ{slot:010d}"
        for slot in range(EXPECTED_EQUITIES - 1 - len(POST_VALIDATION_IDS))
    )
    security_ids = tuple(sorted((ACTIVE_ID, *POST_VALIDATION_IDS, *filler)))
    active_slot = security_ids.index(ACTIVE_ID)
    inactive_slots = frozenset(range(EXPECTED_EQUITIES)) - {active_slot}
    active_source = sources_dir / "active.parquet"
    inactive_source = sources_dir / "post_validation_only.parquet"

    pl.DataFrame(
        {
            "date_idx": pl.Series([0, 1], dtype=pl.Int32),
            "trade_date": [VALIDATION_END, TEST_START],
        }
    ).write_parquet(parent / "date_index.parquet")
    pl.DataFrame(
        {
            "equity_slot": pl.Series(range(EXPECTED_EQUITIES), dtype=pl.Int16),
            "security_id": security_ids,
        }
    ).write_parquet(parent / "equity_index.parquet")

    active = np.zeros(EXPECTED_EQUITIES, dtype=bool)
    active[active_slot] = True
    source_files = np.where(active, str(active_source), str(inactive_source)).tolist()
    overlap_dates = np.where(active, str(VALIDATION_END), str(TEST_END)).tolist()
    symbols = np.where(active, "ACTIVE", "POST_VALIDATION").tolist()
    pl.DataFrame(
        {
            "security_id": security_ids,
            "isin": [f"BR{slot:010d}" for slot in range(EXPECTED_EQUITIES)],
            "latest_ticker": [f"T{slot:03d}" for slot in range(EXPECTED_EQUITIES)],
            "xp_symbol": symbols,
            "source_file": source_files,
            "source_assignment_type": ["EXACT"] * EXPECTED_EQUITIES,
            "first_overlap_date": overlap_dates,
            "last_overlap_date": overlap_dates,
            "manual_decision": ["ACCEPTED"] * EXPECTED_EQUITIES,
            "normalization_rule": ["FILTER_TO_COTAHIST_SECURITY_DATES"]
            * EXPECTED_EQUITIES,
        }
    ).write_parquet(assignments_dir / "xp_accepted_source_assignments_v1.parquet")

    development_dates = [VALIDATION_END]
    if extra_development_date:
        development_dates.append(date(2025, 6, 27))
    pl.DataFrame(
        {
            "trade_date": development_dates,
            "security_id": [ACTIVE_ID] * len(development_dates),
        }
    ).write_parquet(cotahist_dir / "year=2025" / "equities_daily_2025.parquet")
    pl.DataFrame(
        {
            "trade_date": [TEST_END] * len(inactive_slots),
            "security_id": [security_ids[slot] for slot in sorted(inactive_slots)],
        }
    ).write_parquet(cotahist_dir / "year=2026" / "equities_daily_2026.parquet")
    research_start = date(2025, 6, 27) if extra_development_date else VALIDATION_END
    (universe_dir / "manifest.json").write_text(
        json.dumps(
            {
                "resolved_start_date": str(research_start),
                "resolved_end_date": str(TEST_END),
            }
        ),
        encoding="utf-8",
    )

    pl.DataFrame(
        {
            "ts_exchange": [date(2025, 6, 30).strftime("%Y-%m-%d") + " 10:00:00"],
            "open": [10.0],
            "high": [10.1],
            "low": [9.9],
            "close": [10.0],
            "real_volume": [100.0],
            "symbol": ["ACTIVE"],
        }
    ).with_columns(pl.col("ts_exchange").str.to_datetime()).select(
        SOURCE_COLUMNS
    ).write_parquet(active_source)

    membership = np.zeros((2, EXPECTED_EQUITIES), dtype=bool)
    readiness = np.zeros_like(membership)
    membership[0, active_slot] = True
    membership[1, list(inactive_slots)] = True
    readiness[1, list(inactive_slots)] = True
    features = np.zeros(
        (2, EXPECTED_EQUITIES, EQUITY_SESSION_MINUTES, 26), dtype=np.float32
    )
    features[1, list(inactive_slots), 0, AFFECTED_DYNAMIC_CHANNELS[0]] = 9.0
    if contradiction == "membership":
        membership[0, min(inactive_slots)] = True
    elif contradiction == "readiness":
        readiness[0, min(inactive_slots)] = True
    elif contradiction == "features":
        features[0, min(inactive_slots), 0, AFFECTED_DYNAMIC_CHANNELS[0]] = 1.0
    np.save(parent / "equity_membership.npy", membership, allow_pickle=False)
    np.save(parent / "equity_data_ready.npy", readiness, allow_pickle=False)
    np.save(parent / "equity_features.npy", features, allow_pickle=False)

    manifest = {
        "contract_version": "development-inactive-fixture",
        "canonical_inputs": {
            "accepted_xp_assignments": {"resolved_path": str(assignments_dir)},
            "parsed_cotahist": {"resolved_path": str(cotahist_dir)},
            "point_in_time_universe": {"resolved_path": str(universe_dir)},
        },
    }
    (parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(
        parent=parent,
        cotahist_dir=cotahist_dir,
        active_source=active_source,
        inactive_source=inactive_source,
        security_ids=security_ids,
        active_slot=active_slot,
        inactive_slots=inactive_slots,
    )


def test_development_context_retains_axis_and_skips_post_validation_sources(
    tmp_path: Path,
) -> None:
    case = _write_development_context(tmp_path)

    context = load_source_context(case.parent)
    equities = list(iter_reconstructed_equities(context))
    source_identity = equity_source_hashes(context)

    assert len(context.accepted_dates) == EXPECTED_EQUITIES
    assert context.market_dates == (VALIDATION_END,)
    assert context.development_inactive_slots == case.inactive_slots
    assert all(
        not context.accepted_dates[security_id] for security_id in POST_VALIDATION_IDS
    )
    assert len(equities) == 1
    assert equities[0].security_id == ACTIVE_ID
    assert set(source_identity["sources"]) == {str(case.active_source.resolve())}
    assert not case.inactive_source.exists()


def test_full_loader_remains_strict_for_missing_accepted_securities(
    tmp_path: Path,
) -> None:
    case = _write_development_context(tmp_path)

    with pytest.raises(ValueError, match="without exact COTAHIST dates"):
        load_market_dates_and_security_dates(
            cotahist_files(case.cotahist_dir),
            case.security_ids,
            VALIDATION_END,
            VALIDATION_END,
        )


@pytest.mark.parametrize("contradiction", ("membership", "readiness", "features"))
def test_development_inactive_parent_contradictions_fail_closed(
    tmp_path: Path,
    contradiction: str,
) -> None:
    case = _write_development_context(tmp_path, contradiction=contradiction)

    with pytest.raises(ValueError, match=f"parent {contradiction}"):
        load_source_context(case.parent)


def test_development_context_keeps_active_date_axis_strict(tmp_path: Path) -> None:
    case = _write_development_context(tmp_path, extra_development_date=True)

    with pytest.raises(ValueError, match="differs from canonical COTAHIST"):
        load_source_context(case.parent)


def test_inactive_slots_reach_profile_and_raw_variant_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _write_development_context(tmp_path)
    context = load_source_context(case.parent)
    equity = next(iter_reconstructed_equities(context))
    parent_dynamic = np.load(
        case.parent / "equity_features.npy", mmap_mode="r+", allow_pickle=False
    )
    parent_dynamic[0, equity.slot] = equity.dynamic
    parent_dynamic.flush()
    del parent_dynamic

    identity = {
        "path": str(case.parent.resolve()),
        "contract_version": "development-inactive-fixture",
        "metadata_sha256": "fixture",
        "hash_scope": {"kind": "development_only"},
    }
    monkeypatch.setattr(
        normalization, "parent_artifact_hashes", lambda _context: {"fixture": True}
    )
    monkeypatch.setattr(normalization, "parent_identity", lambda _context: identity)
    profile_dir = build_equity_tod_profile(case.parent, tmp_path / "profile")
    relative_variance = np.load(
        profile_dir / "equity_tod_profile.npy", allow_pickle=False
    )
    overlays = {
        arm: np.zeros(
            (
                context.allowed_date_count,
                EXPECTED_EQUITIES,
                variants.VISIBLE_EQUITY_MINUTES,
                len(AFFECTED_DYNAMIC_CHANNELS),
            ),
            dtype=np.float32,
        )
        for arm, gamma in normalization.ARMS.items()
        if gamma > 0.0
    }

    variants._populate_raw_channels(context, relative_variance, overlays)

    for overlay in overlays.values():
        assert not overlay[:, list(case.inactive_slots)].any()
    assert relative_variance.shape == (1, PROFILE_BIN_COUNT)
