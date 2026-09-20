"""Independent arithmetic on saved, dated matched-event source subsets."""

from datetime import date, datetime, time
import json
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json, source_families

PROJECT = Path(__file__).resolve().parents[1]


def main(reopening=False):
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    root = Path(admission["plan"]["path"]).parent / "auxiliaries"
    produced = bound_json(binding(root / "manifest.json"))
    original = root / "qualification"
    prior_qualification = (
        bound_json(binding(original / "manifest.json")) if reopening else None
    )
    out = root / ("reopening_qualification" if reopening else "qualification")
    resume = out.exists()
    assert not (out / "manifest.json").exists()
    out.mkdir(exist_ok=True)
    (out / ("executed_resume.py" if resume else "executed.py")).write_bytes(
        Path(__file__).read_bytes()
    )
    parent = Path(admission["parent"]["root"])
    dates, isins = (
        np.load(parent / "date_index.npy"),
        np.load(parent / "isin_index.npy").tolist(),
    )
    sessions = dates.astype(object).tolist()
    saved = (original if reopening else out) / "reducer_results.json"
    prior_checks = bound_json(binding(saved)) if saved.exists() else None
    activity, odd_cells = (
        (prior_checks["activity"], prior_checks["oddlot_cells"])
        if prior_checks
        else ([], 0)
    )
    for scope in [] if prior_checks else produced["scopes"]:
        folder = root / scope["successor"]
        maps = {
            k: {
                (r["source_trade_date"], r["isin"]): r
                for r in pl.read_parquet(folder / (k + ".parquet")).to_dicts()
            }
            for k in ("cash", "quantities", "snapshots", "nonregular", "odd_source")
        }
        # Keep the two families separate while looking up their original fields.
        outputs = {
            family: {
                r["date"]: r
                for r in pl.read_parquet(folder / (family + ".parquet")).to_dicts()
            }
            for family in ("microstructure", "options")
        }
        e = scope["link"]["effective_index"]

        def source(kind, t):
            return maps[kind].get(
                (sessions[t], scope["predecessor"] if t < e else scope["successor"]), {}
            )

        def total(values):
            return sum(values) if all(x is not None for x in values) else None

        count, supported = 0, 0

        def check(t, field, value, age):
            nonlocal count, supported
            family = (
                "microstructure"
                if field in ("avg_trade_size_20", "after_hours_volume_share_5")
                else "options"
            )
            actual = outputs[family].get(sessions[t], {})
            assert (actual.get(field) is None) == (value is None), (
                sessions[t],
                field,
                actual.get(field),
                value,
            )
            if value is not None:
                np.testing.assert_allclose(actual[field], value, rtol=3e-15, atol=1e-12)
                assert actual[field + "_age_sessions"] == age
                supported += 1
            count += 1

        for t in range(
            max(e, scope["link"]["known_index"]),
            sessions.index(date.fromisoformat(scope["last"])) + 1,
        ):
            cash = [source("cash", j) for j in range(t - 20, t)]
            volume, trades, stock = [
                total([r.get(k) for r in cash])
                for k in ("volume_brl", "trades", "quantity")
            ]
            check(
                t,
                "avg_trade_size_20",
                volume / trades
                if volume is not None and trades and trades > 0
                else None,
                1,
            )
            call, put = [], []
            for j in range(t - 20, t):
                oi, q = source("snapshots", j), source("quantities", j)
                for key, destination in (
                    ("call_quantity", call),
                    ("put_quantity", put),
                ):
                    destination.append(
                        q.get(key, 0 if oi.get("listed_series", 0) > 0 else None)
                    )
            a, b = total(call), total(put)
            check(
                t,
                "option_to_stock_volume_20",
                (a + b) / stock
                if a is not None and b is not None and stock and stock > 0
                else None,
                1,
            )
            a, b = total(call[-5:]), total(put[-5:])
            check(
                t,
                "put_call_volume_ratio_5",
                b / a if a and a > 0 and b is not None else None,
                1,
            )
            after, quantity = [], []
            for j in range(t - 5, t):
                r = source("nonregular", j)
                q, regular = r.get("quantity"), r.get("regular_quantity")
                x = r.get("nonregular_quantity")
                if x is None and q is not None and regular is not None:
                    x = q - regular
                after.append(
                    x if x is not None and q is not None and 0 <= x <= q else None
                )
                quantity.append(q)
            a, b = total(after), total(quantity)
            check(
                t,
                "after_hours_volume_share_5",
                a / b if a is not None and b and b > 0 else None,
                1,
            )
            oi = source("snapshots", t - 1)
            assert not oi.get("oi_all_listed_observed", False)
            for name in (
                "put_call_oi_log_ratio",
                "delta_oi_to_volume_1",
                "uncovered_call_share",
            ):
                check(t, name, None, None)
            a, b = oi.get("call_oi"), oi.get("put_oi")
            check(
                t,
                "observed_series_put_call_oi_log_ratio",
                np.log(b / a) if a and b and a > 0 and b > 0 else None,
                2,
            )
            check(
                t,
                "observed_series_oi_coverage",
                oi["oi_observed_series"] / oi["listed_series"]
                if oi.get("listed_series", 0) > 0
                else None,
                2,
            )

        def share(t, decision):
            r = source("odd_source", t)
            if not r or r["available_date"] > decision:
                return None
            a, b = r["regular_volume_brl"], r["odd_lot_volume_brl"]
            return (
                b / (a + b)
                if a is not None and b is not None and a >= 0 and b >= 0 and a + b > 0
                else None
            )

        for r in pl.read_parquet(folder / "oddlot.parquet").to_dicts():
            t = sessions.index(r["date"]) - 1
            a, b = share(t, r["date"]), share(t - 5, r["date"])
            for field, value in [
                ("oddlot_volume_share", a),
                (
                    "oddlot_volume_share_change_5",
                    a - b if a is not None and b is not None else None,
                ),
            ]:
                assert (r[field] is None) == (value is None)
                if value is not None:
                    np.testing.assert_allclose(r[field], value, rtol=0, atol=1e-15)
                    assert r[field + "_age_sessions"] == 1
                odd_cells += 1
        activity.append(
            dict(
                isin=scope["successor"],
                cells=count,
                supported=supported,
                full_oi_unsupported=True,
            )
        )

    if not saved.exists():
        write_json_atomic(saved, dict(activity=activity, oddlot_cells=odd_cells))
    # Independently select current/prior index portfolios, route dated identities,
    # and sum original weights. Audit actual changed/new rows, not a new source census.
    manifest = bound_json(
        dict(
            path=str(parent / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    im = bound_json(source_families(manifest)["rebalance"]["source_manifest"])
    base = Path(im["base_store"]["path"]).parent
    volume = np.load(base / "volume_brl.npy", mmap_mode="r")
    seen = np.load(base / "activity_valid.npy", mmap_mode="r")
    links = sorted(
        pl.read_parquet(admission["history_mapping"]["path"]).to_dicts(),
        key=lambda x: x["effective_index"],
    )
    for event in bound_json(admission["plan"])["events"]:
        if event["source_reopens_date"]:
            edge = next(
                x for x in links if isins[x["predecessor_index"]] == event["isin"]
            )
            edge["source_reopens_index"] = sessions.index(
                date.fromisoformat(event["source_reopens_date"])
            )
    portfolios = pl.read_parquet(root / "index_portfolios.parquet")
    tickers = pl.read_parquet(root / "index_tickers.parquet")
    before = pl.read_parquet(root / "index_control.parquet")
    after = pl.read_parquet(root / "index_corrected.parquet")
    changed = after.join(
        before, on=["date", "isin"], how="left", suffix="_before"
    ).filter(
        pl.col("index_pressure_before").is_null()
        | (pl.col("index_pressure") != pl.col("index_pressure_before"))
    )
    if reopening:
        previous = pl.read_json(original / "index_records.json").with_columns(
            pl.col("date").str.to_date()
        )
        changed = after.filter(
            pl.col("isin").is_in(["BRJSLGACNOR2", "BRSIMHACNOR0"])
            & pl.col("date").is_between(date(2020, 11, 11), date(2021, 5, 3))
        ).join(previous.select("date", "isin"), on=["date", "isin"], how="anti")
    events = []
    for (day, stage), frame in portfolios.partition_by(
        ["disclosure_date", "stage"], as_dict=True
    ).items():
        available = frame.item(0, "available_at")
        candidates = tickers.filter(pl.col("source_trade_date") <= day).sort(
            "source_trade_date"
        )
        identity, latest = {}, {}
        for r in candidates.iter_rows(named=True):
            key, trade_day = r["ticker"], r["source_trade_date"]
            if latest.get(key) != trade_day:
                identity[key] = set()
                latest[key] = trade_day
            identity[key].add(r["isin"])
        identity = {k: next(iter(v)) for k, v in identity.items() if len(v) == 1}
        weights, unmapped = {}, []
        for r in frame.iter_rows(named=True):
            if r["ticker"] not in identity:
                unmapped.append(r["ticker"])
                continue
            key = (r["index"], identity[r["ticker"]])
            weights[key] = weights.get(key, 0) + r["weight_fraction"]
        events.append(
            dict(
                day=day,
                stage=stage,
                available=available,
                effective=frame.item(0, "effective_date"),
                weights=weights,
                unmapped=unmapped,
            )
        )
    all_days = sorted(
        set(sessions)
        | {date.fromisoformat(d) for d in bound_json(im["calendar"])["sessions"]}
    )
    records = []
    for r in changed.iter_rows(named=True):
        t = sessions.index(r["date"])
        decision = datetime.combine(
            r["date"], time(15, 45), ZoneInfo("America/Sao_Paulo")
        )
        previews = [
            x
            for x in events
            if x["stage"] != "effective"
            and x["available"] <= decision
            and x["effective"] > r["date"]
        ]
        current = max(previews, key=lambda x: x["day"])
        prior = max(
            [
                x
                for x in events
                if x["stage"] == "effective"
                and x["effective"] < current["effective"]
                and x["effective"] <= current["day"]
                and x["available"] <= current["available"]
            ],
            key=lambda x: x["effective"],
        )
        assert (current["day"] - prior["effective"]).days <= 135
        assert not current["unmapped"] and not prior["unmapped"]

        def route(isin, disclosure):
            # Follow event time once. A new JSLG11->JSL edge must never travel
            # backward through the earlier holding-JSL->SIMH event.
            for edge in links:
                reopened = edge.get("source_reopens_index")
                if (
                    isins[edge["predecessor_index"]] == isin
                    and t >= max(edge["effective_index"], edge["known_index"])
                    and (reopened is None or disclosure < sessions[int(reopened)])
                ):
                    isin = isins[edge["successor_index"]]
            return isin

        weights = [
            sum(
                w
                for (_, name), w in x["weights"].items()
                if route(name, x["day"]) == r["isin"]
            )
            for x in (current, prior)
        ]

        n = isins.index(r["isin"])
        values = []
        for j in range(t - 20, t):
            ancestor = n
            while True:
                edge = next(
                    (
                        x
                        for x in links
                        if x["successor_index"] == ancestor
                        and j < x["effective_index"]
                        and t >= max(x["effective_index"], x["known_index"])
                    ),
                    None,
                )
                if edge is None:
                    break
                ancestor = edge["predecessor_index"]
            assert seen[j, ancestor]
            values.append(float(volume[j, ancestor]))
        adv = np.float32(sum(values) / 20)
        remaining = all_days.index(current["effective"]) - all_days.index(r["date"])
        numerator = 100 * (weights[0] - weights[1]) * remaining
        wide = numerator / (float(adv) / 1e6)
        # The original NumPy2 scalar contract rounds both operands at the
        # Float32 ADV division boundary; preserve it, rather than loosening a
        # Float64 comparison or changing the already accepted producer.
        expected = float(
            np.float32(np.float32(numerator) / np.float32(float(adv) / 1e6))
        )
        np.testing.assert_allclose(
            r["index_pressure"], expected, rtol=1e-12, atol=1e-12
        )
        start = next(
            i
            for i, d in enumerate(sessions)
            if datetime.combine(d, time(15, 45), ZoneInfo("America/Sao_Paulo"))
            >= current["available"]
        )
        assert r["index_event_age"] == t - start
        records.append(
            dict(
                date=str(r["date"]),
                isin=r["isin"],
                current=str(current["day"]),
                prior=str(prior["day"]),
                weights=weights,
                adv=float(adv),
                remaining=remaining,
                expected=expected,
                float64_reference=wide,
                actual=r["index_pressure"],
            )
        )
    write_json_atomic(out / "index_records.json", records)
    report = dict(
        status="qualified_independent_auxiliary_arithmetic",
        input=binding(root / "manifest.json"),
        activity=activity,
        oddlot_cells=odd_cells,
        index_rows=len(records),
        index_records=binding(out / "index_records.json"),
        reducer_checks=binding(saved),
        seconds=perf_counter() - tick,
        limits="Saved normalized source endpoints only; original census and prior source qualification reused. No newly sourced historical revision completeness, raw option series alias or loan identity.",
    )
    if reopening:
        report.update(
            status="qualified_reopened_identity_portfolio_controls",
            reused_qualification=binding(original / "manifest.json"),
            reused_index_rows=prior_qualification["index_rows"],
            scope="Additional unchanged JSL/SIMH rows spanning the first two post-reopening preview cycles; completed 80 changed rows/activity/oddlot are reused without re-execution.",
        )
    write_json_atomic(out / "manifest.json", report)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    run[
        "stage_c_event_index_reopening_qualification"
        if reopening
        else "stage_c_event_auxiliary_arithmetic"
    ] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main("--reopening" in sys.argv)
