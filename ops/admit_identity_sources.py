"""Bind three explicit rename links and measure their causal universe effect."""

from datetime import datetime, time, timezone
import json
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.build_store import _route_decision_known_continuations
from brazil_rv.v2.data_foundation import load_isin_link_allowlist
from brazil_rv.v2.universe import build_daily_universe

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    pointer = json.loads(pointer_path.read_text())
    root = Path(pointer["root"])
    evidence_path = root / "identity_source_admission.json"
    if evidence_path.exists():
        raise FileExistsError(evidence_path)
    binding = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    store = Path(binding["root"])
    assert sha256_file(store / "manifest.json") == binding["manifest_sha256"]
    sources = json.loads(
        (PROJECT / "docs/v2_foundation_primary_sources.json").read_text()
    )
    issuer = [
        sources["allos_ticker_20231017"],
        sources["isa_ticker_20241118"],
        json.loads((root / "primary_sources/isa_rename_20241107.json").read_text()),
    ]
    for record in issuer:
        assert sha256_file(Path(record["path"])) == record["sha256"]
    audit = json.loads((root / "daily_source_audit/result.json").read_text())
    quotes = [
        r["source"]
        for r in audit["sources"]
        if any(f"year={y}" in r["source"]["path"] for y in (2023, 2024))
    ]
    for record in quotes:
        assert sha256_file(Path(record["path"])) == record["sha256"]
    daily = pl.concat([pl.read_parquet(r["path"]) for r in quotes])
    definitions = [
        ("ALOS3", "BRALSOACNOR5", "BRALOSACNOR5", "2023-10-25", "2023-10-18T03:00:00Z"),
        ("ISAE4", "BRTRPLACNPR1", "BRISAEACNPR9", "2024-11-18", "2024-11-08T03:00:00Z"),
        ("ISAE3", "BRTRPLACNOR4", "BRISAEACNOR2", "2024-11-18", "2024-11-08T03:00:00Z"),
    ]
    rows = []
    for ticker, old, new, day, known in definitions:
        for isin in (old, new):
            assert isin in np.load(store / "isin_index.npy").tolist()
        boundary = daily.filter(
            pl.col("isin").is_in([old, new])
            & pl.col("trade_date").is_between(
                datetime.fromisoformat(day).date().replace(day=1),
                datetime.fromisoformat(day).date(),
            )
        )
        last = boundary.filter(pl.col("isin") == old).sort("trade_date").tail(1)
        first = boundary.filter(pl.col("isin") == new).sort("trade_date").head(1)
        for field in ("security_spec_base", "market_type", "quote_factor", "currency"):
            assert last[field][0] == first[field][0]
        rows.append(
            {
                "ticker": ticker,
                "predecessor_isin": old,
                "successor_isin": new,
                "effective_date": day,
                "first_known_at": known,
                "shares_received_per_prior_share": 1.0,
                "cash_entitlement_per_prior_share": 0.0,
                "currency": "BRL",
                "boundary_source_rows": json.loads(pl.concat([last, first]).write_json()),
            }
        )
    evidence = {
        "status": "source_verified_equity_rename_links_new_store_pending",
        "store": binding,
        "issuer_receipts": issuer,
        "cotahist_receipts": quotes,
        "links": rows,
        "interpretation": "Same issued share classes under new names/codes; unit quantity and no cash follow issuer rename terms, not price matching. ISINs remain distinct dated source attributes.",
        "source_correction": "Nov7 ISA notice prints TRLP4; archived Nov18 issuer notice and dated COTAHIST establish TRPL4. Preserve original typo.",
        "availability": "Next local day after the dated notices, before both effective sessions; no precise publication clock inferred.",
        "loan_alias_timing": "Not admitted: BDI may still publish predecessor loan codes after the spot rename; audit lending mapping separately.",
    }
    digest = write_json_atomic(evidence_path, evidence)
    config_path = PROJECT / "research/configs/v2/isin_links_allowlist.csv"
    config = pl.DataFrame(
        [
            {k: v for k, v in r.items() if k != "boundary_source_rows"}
            | {"source": str(evidence_path), "evidence_sha256": digest}
            for r in rows
        ]
    )
    config.write_csv(config_path)
    links = load_isin_link_allowlist(config_path, daily)
    isins = [name for row in definitions for name in row[1:3]]
    axes = np.load(store / "isin_index.npy").tolist()
    names = [axes.index(name) for name in isins]
    dates = np.load(store / "date_index.npy")
    assert dates[-1] <= np.datetime64("2024-12-30")
    arrays = {
        key: np.load(store / f"{key}.npy", mmap_mode="r")[:, names]
        for key in (
            "raw_close",
            "volume_brl",
            "trade_count",
            "observed",
            "trade_observed",
            "activity_valid",
            "active",
        )
    }
    complete = np.load(store / "source_session_complete.npy", mmap_mode="r")[
        :, names[0]
    ]
    original = build_daily_universe(
        arrays["raw_close"],
        arrays["volume_brl"],
        arrays["observed"],
        trade_observed=arrays["trade_observed"],
        activity_valid=arrays["activity_valid"],
        source_session_complete=complete,
    )
    routed = _route_decision_known_continuations(
        dates=dates,
        isins=isins,
        links=links,
        decision_timestamps=[
            datetime.combine(d.astype(object), time(12), tzinfo=timezone.utc)
            for d in dates
        ],
        raw_close=arrays["raw_close"],
        volume_brl=arrays["volume_brl"],
        trades=arrays["trade_count"],
        observed=arrays["observed"],
        trade_observed=arrays["trade_observed"],
        activity_valid=arrays["activity_valid"],
        ambiguous_action=np.zeros(arrays["active"].shape, bool),
        shareholder_wealth_arrays=(),
    )
    repaired = build_daily_universe(
        routed.close_brl,
        routed.volume_brl,
        routed.observed,
        trade_observed=routed.trade_observed,
        activity_valid=routed.activity_valid,
        source_session_complete=complete,
    )
    active = repaired.active & routed.claim_owner
    outcomes = []
    for ticker, old, new, day, _ in definitions:
        a, b = isins.index(old), isins.index(new)
        selected = dates >= np.datetime64(day)
        np.testing.assert_array_equal(
            original.active[selected][:, [a, b]], arrays["active"][selected][:, [a, b]]
        )
        gain = selected & active[:, b] & ~original.active[:, b]
        retirement = selected & original.active[:, a] & ~active[:, a]
        assert not (selected & active[:, a]).any()
        assert not (selected & original.active[:, b] & ~active[:, b]).any()
        assert not active[~selected, b].any()
        outcomes.append(
            {
                "ticker": ticker,
                "gained_eligible_days": int(gain.sum()),
                "retired_stale_predecessor_days": int(retirement.sum()),
                "first_repaired_eligible_date": str(
                    dates[np.flatnonzero(selected & active[:, b])[0]]
                )
                if (selected & active[:, b]).any()
                else None,
                "effective_prior_history_before": int(
                    original.history_sessions[np.flatnonzero(selected)[0], b]
                ),
                "effective_prior_history_after": int(
                    repaired.history_sessions[np.flatnonzero(selected)[0], b]
                ),
            }
        )
    receipt = {
        "status": "continuation_loader_and_universe_verified_derived_store_pending",
        "evidence": {"path": str(evidence_path), "sha256": digest},
        "allowlist": {
            "path": str(config_path.relative_to(PROJECT)),
            "sha256": sha256_file(config_path),
        },
        "cases": outcomes,
        "accepted_store_mutated": False,
        "labels_or_forecasts_read": False,
        "note": "Bounded six-name original-contract replay, not a new accepted data store. Actual feature/wealth/label/sidecar propagation still required in B.",
    }
    receipt_path = PROJECT / "docs/v2_identity_source_admission.json"
    receipt_sha = write_json_atomic(receipt_path, receipt)
    pointer["identity_source_admission"] = {
        "path": str(receipt_path.relative_to(PROJECT)),
        "sha256": receipt_sha,
    }
    write_json_atomic(pointer_path, pointer)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
