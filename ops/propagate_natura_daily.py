"""Recompute only wealth-dependent slow reducers on a bounded full-name window."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_actions import (
    align_decision_known_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import (
    load_session_schedule,
    next_session_decision_cutoffs,
)
from brazil_rv.v2.features import (
    _ambiguous_interval_clear,
    _peer_features,
    _rolling_high_low_range,
    _rolling_market_regression,
    _rolling_moments,
    _rolling_stat,
    exact_log_return,
    monthly_cluster_labels,
    yang_zhang_volatility,
)

PROJECT = Path(__file__).resolve().parents[1]


def fields(wealth, seen, active, ambiguous, volume, activity, dates, source_rows):
    open_, high, low, close = wealth
    returns = {}
    for index, horizon in enumerate((1, 5, 21, 63, 126, 252)):
        result = exact_log_return(
            close, horizon, ambiguous, shareholder_wealth_valid=seen
        )
        if horizon in (1, 5, 21, 252):
            returns[horizon] = result
        yield index, *result
    yield 6, returns[252][0] - returns[21][0], returns[252][1] & returns[21][1]
    for index, window in ((7, 5), (8, 20), (9, 60)):
        values, valid = yang_zhang_volatility(
            open_, high, low, close, window, ambiguous
        )
        yield index, values, valid
        if window == 5:
            yz5 = values
    values, valid = _rolling_stat(yz5, 60, "std")
    yield 10, values, valid & _ambiguous_interval_clear(ambiguous, 64)
    skew, kurtosis, valid = _rolling_moments(returns[1][0], 60)
    valid &= _ambiguous_interval_clear(ambiguous, 60)
    yield 11, skew, valid
    yield 12, kurtosis, valid
    values, valid = _rolling_stat(returns[1][0], 21, "max")
    yield 13, values, valid & _ambiguous_interval_clear(ambiguous, 21)
    maximum, valid = _rolling_stat(high, 252, "max")
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(close / maximum)
    yield (
        14,
        values,
        valid & seen & (close > 0) & _ambiguous_interval_clear(ambiguous, 251),
    )
    market = np.full(len(dates), np.nan, np.float64)
    for day in range(len(dates)):
        mask = returns[1][1][day] & active[day]
        if mask.any():
            market[day] = np.median(returns[1][0][day, mask])
    beta, idio, valid = _rolling_market_regression(
        returns[1][0], market, ambiguous, source_rows=source_rows
    )
    yield 15, beta, valid
    yield 16, idio, valid
    amihud = np.full(close.shape, np.nan, np.float64)
    usable = activity & returns[1][1] & np.isfinite(volume) & (volume > 0)
    amihud[usable] = np.abs(returns[1][0][usable]) / volume[usable]
    values, valid = _rolling_stat(amihud, 20, "mean", minimum=20)
    yield 19, values, valid & _ambiguous_interval_clear(ambiguous, 20)
    values, valid = _rolling_high_low_range(high, low, seen, 5, ambiguous)
    yield 23, values, valid
    residual, valid = returns[1][0].copy(), returns[1][1] & active
    for day in range(len(dates)):
        mask = valid[day]
        if mask.any():
            residual[day, mask] -= np.median(residual[day, mask])
    labels = monthly_cluster_labels(dates, residual, valid, active)
    yield "clusters", labels, np.ones_like(labels, bool)
    yield "market", market, np.isfinite(market)
    values, valid = _peer_features(*returns[5], *returns[21], labels, active)
    for offset in range(5):
        yield 27 + offset, values[..., offset], valid[..., offset]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    correction = bound_json(run["natura_wealth_amendment"])
    source = correction["parent_store"]
    root = Path(source["root"])
    manifest = bound_json(
        {"path": str(root / "manifest.json"), "sha256": source["manifest_sha256"]}
    )
    history = bound_json(run["rename_history_propagation"])
    source_clock = bound_json(run["held_event_source_qualification"])
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    first = int(np.searchsorted(dates, np.datetime64("2019-09-18")))
    start = first - 253
    stop = int(np.searchsorted(dates, np.datetime64("2020-07-06")))
    window = slice(start, stop)
    days = dates[window]
    selected = np.arange(first - start, len(days))
    assert len(isins) == 933 and days[-1] < np.datetime64("2025-01-01")
    out = Path(run["root"]) / "natura_daily"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_features.py").write_bytes(
        (PROJECT / "research/src/brazil_rv/v2/features.py").read_bytes()
    )
    plan = {
        "parent_store": source,
        "wealth": run["natura_wealth_amendment"],
        "history_basis": run["rename_history_propagation"],
        "first_decision": str(dates[first]),
        "source_window": [str(days[0]), str(days[-1])],
        "rows": [start, first, stop],
        "fields": [*range(17), 19, 23, *range(27, 32)],
        "unchanged_fields": [17, 18, 20, 21, 22, 24, 25, 26],
        "history": "253 prior sessions for original 252-session fields, then monthly peer history through the first clean July2020 refit plus controls. No history, support threshold, population, source quote or learning budget changes.",
    }
    write_json_atomic(out / "plan.json", plan)
    np.save(out / "date_index.npy", days)
    np.save(out / "selected.npy", selected)

    def old(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")[window]

    wealth = [
        np.load(
            history["history_basis"]["shareholder_wealth_" + k]["path"], mmap_mode="r"
        )[window]
        for k in ("open", "high", "low", "close")
    ]
    seen = np.load(
        history["history_basis"]["shareholder_wealth_valid"]["path"], mmap_mode="r"
    )[window]
    active, activity = old("active"), old("activity_valid")
    # Original daily market features used original Float64 BRL volumes. Recover
    # only necessary normalized date/name/volume coordinates, never raw archives.
    quote_records = [
        r
        for r in manifest["sources"]
        if Path(r.get("path", "")).name.startswith("equities_daily_")
        and any(str(y) in Path(r["path"]).name for y in (2018, 2019, 2020))
    ]
    quotes = pl.concat(
        [
            pl.scan_parquet(r["path"])
            .filter(
                pl.col("trade_date").is_between(
                    days[0].astype(object), days[-1].astype(object)
                )
                & pl.col("isin").is_in(isins)
            )
            .select("trade_date", "isin", "volume_brl")
            .collect()
            for r in quote_records
        ]
    ).sort("trade_date", "isin")
    assert not quotes.select(
        pl.struct("trade_date", "isin").is_duplicated().any()
    ).item()
    quotes.write_parquet(out / "source_volume_coordinates.parquet")
    volume = np.asarray(old("volume_brl"), np.float64).copy()
    lookup = {isin: i for i, isin in enumerate(isins)}
    ti = np.searchsorted(days, quotes["trade_date"].to_numpy())
    ni = np.array([lookup[s] for s in quotes["isin"]])
    volume[ti, ni] = quotes["volume_brl"].to_numpy()
    ancestor_root, ancestor = root, manifest
    while "round5_extension" in ancestor["metadata"]:
        rec = ancestor["metadata"]["round5_extension"]["base_store"]
        ancestor_root = Path(rec["root"])
        ancestor = bound_json(
            {
                "path": str(ancestor_root / "manifest.json"),
                "sha256": rec["manifest_sha256"],
            }
        )
    term_path = (
        ancestor_root / ancestor["tables"]["corporate_actions_verified_terms"]["path"]
    )
    terms = verified_action_terms_from_table(pl.read_parquet(term_path))
    schedule = load_session_schedule(Path(source_clock["schedule"]["path"]))
    selected_schedule = tuple(
        s for s in schedule if days[0] <= np.datetime64(s.trade_date) <= days[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > days[-1]
    )
    cutoffs = next_session_decision_cutoffs(
        selected_schedule, following_decision_at=following
    )
    actions = align_decision_known_action_terms(
        terms,
        days,
        isins,
        coverage_resolved=old("observed"),
        decision_timestamps=cutoffs,
    )
    ambiguous = ~actions.session_resolved
    np.save(out / "ambiguous.npy", ambiguous)
    changed = [a.copy() for a in wealth]
    with np.load(correction["deltas"]["path"]) as z:
        for field, array in zip(("open", "high", "low", "close"), changed, strict=True):
            indices = z["shareholder_wealth_" + field + "__indices"].copy()
            indices[:, 0] -= start
            assert (indices[:, 0] >= 0).all() and (indices[:, 0] < len(days)).all()
            array[tuple(indices.T)] = z["shareholder_wealth_" + field + "__values"]
    # No split/cash support barrier changes on the two admitted observed dates.
    for label, arrays in (("control", wealth), ("corrected", changed)):
        target = out / label
        target.mkdir()
        for key, values, mask in fields(
            arrays,
            seen,
            active,
            ambiguous,
            volume,
            activity,
            days,
            np.arange(selected[0] - 65, len(days)),
        ):
            mask = mask & np.isfinite(values)
            np.savez_compressed(target / f"{key}.npz", values=values, valid=mask)
            print(
                json.dumps(
                    {"case": label, "field": key, "seconds": perf_counter() - started}
                ),
                flush=True,
            )
    report = {
        "status": "produced_bounded_daily_reducers_pending_control_and_consumer_qualification",
        "plan": binding(out / "plan.json"),
        "source_terms": binding(term_path),
        "volume_coordinates": binding(out / "source_volume_coordinates.parquet"),
        "volume_source_receipts": quote_records,
        "volume_rows": quotes.height,
        "basis_shape": list(wealth[0].shape),
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["natura_daily_propagation"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
