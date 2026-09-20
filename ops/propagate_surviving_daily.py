"""Bounded full-cross-section reducers for the two surviving-company renames."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    align_decision_known_action_terms,
    verified_action_terms_from_table,
    verified_conversion_terms_from_links,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.features import build_slow_features_into

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    admission = bound_json(run["surviving_rename_admission"])
    wealth_report = bound_json(run["surviving_rename_wealth"])
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = Path(admission["plan"]["path"]).parent / "daily"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_features.py").write_bytes(
        (PROJECT / "research/src/brazil_rv/v2/features.py").read_bytes()
    )
    dates = np.load(root / "date_index.npy")
    isins = np.load(root / "isin_index.npy").tolist()
    first = int(np.searchsorted(dates, np.datetime64("2019-08-06")))
    start = first - 253
    days = dates[start:]
    selected = np.arange(first - start, len(days))
    assert len(isins) == 933 and str(days[-1]) == "2024-12-30"
    write_json_atomic(
        out / "plan.json",
        {
            "parent": admission["parent"],
            "admission": run["surviving_rename_admission"],
            "wealth": run["surviving_rename_wealth"],
            "source_window": [str(days[0]), str(days[-1])],
            "rows": [start, first, len(dates)],
            "reason_for_tail": "Qualified Float32 wealth recurrence differs through2024, so retain its actual tail instead of assuming exact numerical scale invariance. Only normalized source rows/unchanged full-name reducers, no raw census or old fit.",
            "contrast": "Accepted Natura history versus two sourced surviving-company links. Existing3links and all33sparse succession target controls reused; full933 cross-sections,252/126/60 lookbacks and support thresholds unchanged.",
            "history_age": "Fields25/26 are unranked identity ages. Their accepted values are retained outside the new identity axes; exact predecessor first-observed age is propagated separately during qualification. Truncated-window scratch age is never used as a new original listing date.",
        },
    )
    np.save(out / "date_index.npy", days)
    np.save(out / "selected.npy", selected)

    def old(k):
        return np.load(root / m["arrays"][k]["path"], mmap_mode="r")[start:]

    history = bound_json(run["rename_history_propagation"])["history_basis"]
    old_links = pl.read_parquet(root / m["tables"]["isin_succession_links"]["path"])
    new_links = pl.read_parquet(admission["links"]["path"])
    # Start from accepted public coordinates; restore only the previously
    # qualified internal prebirth lookbacks of the old three links.
    base = [
        old("shareholder_wealth_" + k).copy()
        for k in ("open", "high", "low", "close", "valid")
    ]
    for row in old_links.iter_rows(named=True):
        b = isins.index(row["successor_isin"])
        before = days < np.datetime64(row["effective_date"])
        for k, array in zip(
            ("open", "high", "low", "close", "valid"), base, strict=True
        ):
            array[before, b] = np.load(
                history["shareholder_wealth_" + k]["path"], mmap_mode="r"
            )[start:][before, b]
    changed = [x.copy() for x in base]
    with np.load(wealth_report["deltas"]["path"]) as z:
        for k, array in zip(
            ("open", "high", "low", "close", "valid"), changed, strict=True
        ):
            key = "shareholder_wealth_" + k
            ix = z[key + "__indices"].copy()
            ix[:, 0] -= start
            array[tuple(ix.T)] = z[key + "__values"]
    for row in new_links.iter_rows(named=True):
        a, b = (isins.index(row[k]) for k in ("predecessor_isin", "successor_isin"))
        before = days < np.datetime64(row["effective_date"])
        for array in changed:
            array[before, b] = array[before, a]
    active = old("active").copy()
    with np.load(admission["liquidity_deltas"]["path"]) as z:
        ix = z["active__indices"].copy()
        ix[:, 0] -= start
        active[tuple(ix.T)] = z["active__values"]
    # Float64 source volume and printed raw prices are required for exact
    # unchanged reducer controls, especially Amihud and close location.
    records = [
        r
        for r in m["sources"]
        if Path(r.get("path", "")).name
        in [f"equities_daily_{y}.parquet" for y in range(int(str(days[0])[:4]), 2025)]
    ]
    quotes = pl.concat(
        [
            pl.scan_parquet(r["path"])
            .filter(
                pl.col("isin").is_in(isins)
                & (pl.col("market_type") == 10)
                & (pl.col("trade_date") >= days[0].astype(object))
            )
            .select(
                "trade_date",
                "isin",
                "volume_brl",
                "trades",
                "high_brl",
                "low_brl",
                "close_brl",
            )
            .collect()
            for r in records
        ]
    ).sort("trade_date", "isin")
    assert not quotes.select(
        pl.struct("trade_date", "isin").is_duplicated().any()
    ).item()
    quotes.write_parquet(out / "source_coordinates.parquet")
    ti = np.searchsorted(days, quotes["trade_date"].to_numpy())
    lookup = {s: j for j, s in enumerate(isins)}
    ni = np.array([lookup[s] for s in quotes["isin"]])
    volume, trades = (
        old("volume_brl").astype(np.float64),
        old("trade_count").astype(np.float64),
    )
    raw = [
        np.round(old("raw_" + k).astype(np.float64), 2)
        for k in ("high", "low", "close")
    ]
    for array, key in zip(
        [volume, trades, *raw],
        ["volume_brl", "trades", "high_brl", "low_brl", "close_brl"],
        strict=True,
    ):
        array[ti, ni] = quotes[key].to_numpy()
    observed, activity = old("observed").copy(), old("activity_valid").copy()
    # Daily feature reconstruction uses the original normalized daily rows.
    # Later accepted price-only marks do not create missing original OHLC rows.
    present = np.zeros_like(observed)
    present[ti, ni] = True
    observed &= present
    volume[activity & ~present] = 0.0
    trades[activity & ~present] = 0.0
    terms = verified_action_terms_from_table(
        pl.read_parquet(root / m["tables"]["corporate_actions_verified_terms"]["path"])
    )
    schedule = load_session_schedule(Path(admission["schedule"]["path"]))
    ss = tuple(
        s for s in schedule if days[0] <= np.datetime64(s.trade_date) <= days[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > days[-1]
    )
    cutoffs = next_session_decision_cutoffs(ss, following_decision_at=following)
    reports = []
    for label, wealth, membership, links in (
        ("control", base, old("active"), old_links),
        ("corrected", changed, active, new_links),
    ):
        folder = out / label
        folder.mkdir()
        actions = align_decision_known_action_terms(
            (*terms, *verified_conversion_terms_from_links(links)),
            days,
            isins,
            coverage_resolved=observed,
            decision_timestamps=cutoffs,
        )
        ambiguous = ~actions.session_resolved
        v, n = volume.copy(), trades.copy()
        shape = [x.copy() for x in raw]
        price_seen, history_seen, act = (
            observed.copy(),
            observed.copy(),
            activity.copy(),
        )
        mapping = []
        for row in links.iter_rows(named=True):
            a, b = (isins.index(row[k]) for k in ("predecessor_isin", "successor_isin"))
            boundary = int(np.searchsorted(days, np.datetime64(row["effective_date"])))
            assert row["first_known_at"] <= ss[boundary].decision_at
            for array in (v, n, *shape, price_seen, history_seen, act, ambiguous):
                array[:boundary, b] = array[:boundary, a]
            mapping.append(
                dict(
                    predecessor_index=a,
                    successor_index=b,
                    effective_index=boundary,
                    known_index=boundary,
                )
            )
        for k, array in zip(
            ("open", "high", "low", "close", "valid"), wealth, strict=True
        ):
            np.save(folder / ("wealth_" + k + ".npy"), array)
        for k, array in (
            ("active", membership),
            ("ambiguous", ambiguous),
            ("volume", v),
            ("activity", act),
        ):
            np.save(folder / (k + ".npy"), array)

        def consume(index, values, valid):
            if index in (25, 26):
                return
            np.savez_compressed(folder / f"{index}.npz", values=values, valid=valid)
            print(
                json.dumps(
                    dict(case=label, field=index, seconds=perf_counter() - tick)
                ),
                flush=True,
            )

        labels = build_slow_features_into(
            *wealth[:4],
            v,
            n,
            wealth[4],
            membership,
            days,
            raw_high=shape[0],
            raw_low=shape[1],
            raw_close=shape[2],
            price_observed=price_seen,
            history_observed=history_seen,
            activity_valid=act,
            ambiguous_action=ambiguous,
            consume=consume,
            history_links=mapping,
        )
        np.save(folder / "clusters.npy", labels)
        reports.append(dict(case=label, seconds=perf_counter() - tick))
    report = {
        "status": "reducers_saved_pending_qualification",
        "plan": binding(out / "plan.json"),
        "source_coordinates": binding(out / "source_coordinates.parquet"),
        "source_records": records,
        "source_rows": quotes.height,
        "cases": reports,
        "seconds": perf_counter() - tick,
    }
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_daily"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
