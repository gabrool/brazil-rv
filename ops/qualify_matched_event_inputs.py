"""Independent arithmetic for the new minute and primary-target interactions."""

from decimal import Decimal
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.corporate_actions import (
    align_verified_action_terms,
    verified_action_terms_from_table,
)
from brazil_rv.v2.corporate_replay import apply_corporate_replay, load_corporate_replay
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.intraday_features import build_native_fast_features
from compose_surviving_store import read_layers, changed
from propagate_surviving_m1 import grid_for, minute
from verify_corporate_replay import inputs_on_axes
from verify_surviving_market_inputs import independent, rank

PROJECT = Path(__file__).resolve().parents[1]


def main(mode):
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    base = Path(admission["plan"]["path"]).parent
    out = base / (mode + "_qualification")
    resume = out.exists()
    out.mkdir(exist_ok=True)
    assert not (out / "manifest.json").exists()
    (out / ("executed_resume.py" if resume else "executed.py")).write_bytes(
        Path(__file__).read_bytes()
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    active = np.load(root / "active.npy").copy()
    with np.load(admission["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    schedule = {
        s.trade_date: s
        for s in load_session_schedule(
            root / m["tables"]["b3_session_schedule"]["path"]
        )
    }
    if mode == "minute":
        native = bound_json(run["stage_c_event_m1"])
        scalar = bound_json(run["stage_c_event_scalars"])
        market = bound_json(run["stage_c_event_market"])
        write_json_atomic(
            out / "plan.json",
            dict(
                native=run["stage_c_event_m1"],
                scalar=run["stage_c_event_scalars"],
                market=run["stage_c_event_market"],
                scope="Independent saved-source same-clock volume, scalar transforms/clock/rank/source-age and typed market formulas. One entry/post-decision mutation with21prior sessions and scalar future-prefix deletion. Reuse passed native price-block and family controls; no original source/reducer campaign.",
            ),
        )
        rec = next(r for r in native["cases"] if r["isin"] == "BRBRAVACNOR3")
        source = pl.read_parquet(rec["source"]["path"])
        rows = np.load(rec["date_indices"]["path"])
        sessions = tuple(schedule[d] for d in dates[rows].astype(object))
        grid, seen, _ = grid_for(
            source,
            set(dates[rows].astype(object)),
            sessions,
            Path(rec["source"]["path"]),
        )
        nr = Path(rec["source"]["path"]).parent
        with np.load(nr / "BRBRAVACNOR3_native.npz") as z:
            emitted = z["rows"].copy()
            values = z["values"].copy()
            valid = z["valid"].copy()
        volume_checks = 0
        for i, t in enumerate(rows):
            if t not in emitted:
                continue
            j = int(np.searchsorted(emitted, t))
            session = sessions[i]
            opening = minute(session.continuous_open)
            patches = (minute(session.decision_time) - opening) // 5
            for p in range(patches):
                clock = opening + p * 5
                history = []
                for h in range(i - 20, i):
                    start = clock - minute(sessions[h].continuous_open)
                    stop = start + 5
                    prefix = minute(sessions[h].decision_time) - minute(
                        sessions[h].continuous_open
                    )
                    if start < 0 or start % 5 or stop > prefix:
                        continue
                    v = grid[h, start:stop, 4].astype(float)
                    if (
                        seen[h, start:stop].all()
                        and np.isfinite(v).all()
                        and (v >= 0).all()
                    ):
                        history.append(float(v.sum()))
                current = grid[i, p * 5 : p * 5 + 5, 4].astype(float)
                ok = bool(
                    len(history) >= 16
                    and np.median(history) > 0
                    and seen[i, p * 5 : p * 5 + 5].all()
                    and np.isfinite(current).all()
                    and (current >= 0).all()
                )
                assert valid[j, p, 2] == ok, (t, p)
                expected = (
                    np.float32(
                        np.clip(
                            np.log1p(current.sum() / np.median(history)) - np.log(2),
                            -5,
                            5,
                        )
                    )
                    if ok
                    else np.float32(0)
                )
                assert values[j, p, 2] == expected, (t, p)
                volume_checks += 2
        with np.load(scalar["raw_corrected"]["path"]) as z:
            raw = {k: z[k].copy() for k in z.files}
        with np.load(scalar["outputs"]["path"]) as z:
            emitted = z["date_indices"].copy()
            outputs = {k: z[k].copy() for k in z.files if k != "date_indices"}
        take = emitted - raw["date_indices"][0]
        specs = [
            s
            for s in m["metadata"]["feature_schema"]["specifications"]
            if s["family"] == "intraday"
        ]
        v, mask = independent(
            raw["values"][take], raw["valid"][take], active[emitted], specs
        )
        np.testing.assert_array_equal(v, outputs["intraday_values"])
        np.testing.assert_array_equal(mask, outputs["intraday_valid"])
        formula_cells = v.size * 2
        target_count = 0
        for i, t in enumerate(take):
            entry, close, sigma = [
                raw[k][t].astype(float)
                for k in ("entry", "session_close", "realized_daily_vol")
            ]
            ok = (
                active[emitted[i]]
                & raw["fast_present"][t]
                & raw["entry_valid"][t]
                & raw["return_consistent"][t]
                & raw["session_close_valid"][t]
                & np.isfinite(entry + close + sigma)
                & (entry > 0)
                & (close > 0)
                & (sigma > 0)
            )
            np.testing.assert_array_equal(ok, outputs["target_to_close_valid"][i])
            if not ok.any():
                continue
            s = schedule[dates[emitted[i]].astype(object)]
            remaining = minute(s.continuous_close) - minute(s.decision_time)
            total = minute(s.continuous_close) - minute(s.continuous_open)
            ret = np.log(close[ok] / entry[ok])
            scaled = ret / (sigma[ok] * np.sqrt(remaining / total))
            centered = np.clip(scaled - np.median(scaled), -5, 5)
            ranked = (
                (rank(centered) / (len(centered) - 1)).astype(np.float32)
                if len(centered) > 1
                else np.array([0.5], np.float32)
            )
            for k, x in (
                ("target_to_close_raw_log_return", ret.astype(np.float32)),
                ("target_to_close_normalized_residual", centered.astype(np.float32)),
                ("target_to_close", ranked),
            ):
                np.testing.assert_array_equal(x, outputs[k][i, ok])
            target_count += int(ok.sum())
        # Independent last-observation walk seeded before the newly admitted link.
        p, n = isins.index("BRRRRPACNOR5"), isins.index("BRBRAVACNOR3")
        effect = int(np.searchsorted(dates, np.datetime64("2024-09-09")))
        old_age = np.load(root / "intraday_age_sessions.npy", mmap_mode="r")
        seed = old_age[effect - 1, p]
        last = np.where(seed >= 0, effect - 1 - seed, -1)
        age_cells = 0
        for t in range(effect, len(dates)):
            if t <= emitted[-1]:
                local = t - int(raw["date_indices"][0])
                ok = raw["valid"][local, n]
                last[ok] = t - raw["source_age_sessions"][local, n, ok]
            else:
                own = old_age[t, n]
                last = np.maximum(last, np.where(own >= 0, t - own, -1))
            wanted = np.where(last >= 0, t - last, -1).astype(np.float32)
            if not active[t, n]:
                wanted[:] = -1
            if t <= emitted[-1]:
                actual = outputs["intraday_age_sessions"][t - emitted[0], n]
            else:
                with np.load(
                    Path(scalar["outputs"]["path"]).parent / "age_tail.npz"
                ) as z:
                    actual = z["age"][t - emitted[-1] - 1]
            np.testing.assert_array_equal(wanted, actual)
            age_cells += 20
        end = int(take[9] + 1)
        v = np.empty((10, 933, 20), np.float32)
        mask = np.empty_like(v, bool)
        transform_feature_panel_into(
            raw["values"][:end],
            raw["valid"][:end],
            active[raw["date_indices"][:end]],
            [FeatureSpec(**s) for s in specs],
            v,
            mask,
            source_rows=take[:10],
        )
        np.testing.assert_array_equal(v, outputs["intraday_values"][:10])
        np.testing.assert_array_equal(mask, outputs["intraday_valid"][:10])
        # Include all21prior sessions so the changed volume channel is exercised.
        k = 22
        risk = np.load(root / "target_scale_sigma.npy", mmap_mode="r")[
            rows[:k], n
        ].copy()
        daily = bound_json(run["stage_c_event_daily_qualification"])
        with np.load(daily["deltas"]["path"]) as z:
            ix = z["target_scale_sigma__indices"]
            pick = (ix[:, 1] == n) & np.isin(ix[:, 0], rows[:k])
            risk[np.searchsorted(rows[:k], ix[pick, 0])] = z[
                "target_scale_sigma__values"
            ][pick]
        pair = []
        for mutation in (False, True):
            g, s = grid[:k].copy(), seen[:k].copy()
            if mutation:
                cutoff = minute(sessions[k - 1].decision_time) - minute(
                    sessions[k - 1].continuous_open
                )
                g[-1, cutoff:] = 123456.0
                s[-1, cutoff:] = False
            result = build_native_fast_features(
                *[g[:, None, :, j] for j in (1, 2, 3, 4)],
                s[:, None],
                volume_valid=s[:, None],
                session_valid=s.any(axis=1)[:, None],
                sigma_asof=risk[:, None],
                sessions=sessions[:k],
            )
            pair.append(result)
        for attr in (
            "values",
            "valid",
            "patch_mask",
            "last_price_age_minutes",
            "last_price_age_valid",
        ):
            np.testing.assert_array_equal(
                getattr(pair[0], attr), getattr(pair[1], attr)
            )
        # Typed market formulas use saved original regression outputs, once.
        patches = read_layers([market["deltas"]])
        mr = Path(run["stage_c_event_market"]["path"]).parent
        for group in ("magnitudes", "cross_market"):
            with np.load(mr / (group + "_corrected.npz")) as z:
                rr = z["rows"]
                raw_values = z["values"]
                raw_mask = z["valid"]
                ages = z["ages"]
            family = "sidecar_" + group
            fs = [
                s
                for s in m["metadata"]["feature_schema"]["specifications"]
                if s["family"] == family
            ]
            for a in range(0, len(rr), 64):
                b = min(a + 64, len(rr))
                selected = rr[a:b]
                v, known = independent(
                    raw_values[a:b], raw_mask[a:b], active[selected], fs
                )
                for suffix, x in (
                    ("values", v),
                    ("valid", known),
                    (
                        "age_sessions",
                        np.where(active[selected, :, None], ages[a:b], -1),
                    ),
                ):
                    key = family + "_" + suffix
                    expected = np.load(root / (key + ".npy"), mmap_mode="r")[
                        selected
                    ].copy()
                    for ix, value in patches.get(key, []):
                        take = np.isin(ix[:, 0], selected)
                        local = ix[take].copy()
                        local[:, 0] = np.searchsorted(selected, local[:, 0])
                        expected[tuple(local.T)] = value[take]
                    np.testing.assert_array_equal(x, expected)
                    formula_cells += x.size
        report = dict(
            status="qualified_minute_market_arithmetic",
            volume_cells=volume_checks,
            formula_cells=formula_cells,
            source_age_cells=age_cells,
            to_close_outcomes=target_count,
            native_entry_postdecision_mutation=True,
            scalar_future_prefix=True,
            seconds=perf_counter() - tick,
        )
    else:
        assert mode == "targets"
        target = bound_json(run["stage_c_event_targets"])
        folder = Path(target["plan"]["path"]).parent
        source = bound_json(run["stage_c_event_source_admission"])
        with np.load(folder / "corrected_targets.npz") as z:
            rows = z["date_indices"]
            after = {k: z[k].copy() for k in z.files if k != "date_indices"}
        terms = verified_action_terms_from_table(
            pl.read_parquet(folder / "corporate_actions_verified_terms.parquet")
        )
        actions = align_verified_action_terms(
            terms,
            dates,
            isins,
            coverage_resolved=np.load(root / "action_session_resolved.npy"),
        )
        rec = run["stage_c_event_candidate_terms"]
        spec, _ = load_corporate_replay(path=rec["path"], expected_sha256=rec["sha256"])
        data = inputs_on_axes(
            Path(spec["store"]["root"]), dates, np.arange(933), np.arange(len(dates))
        )
        events = apply_corporate_replay(
            data, spec, dates, rec["sha256"]
        ).share_distributions
        event_map = {(e.effective_session, e.source_index): e for e in events}
        close = np.round(np.load(root / "raw_close.npy").astype(float), 2)
        observed = np.load(root / "observed.npy")
        before_valid = np.load(root / "target_shareholder_valid.npy", mmap_mode="r")[
            rows
        ]
        before_returns = np.load(
            root / "target_shareholder_simple_return.npy", mmap_mode="r"
        )[rows]
        scope = set(
            map(
                tuple,
                np.argwhere(
                    (before_valid != after["target_shareholder_valid"])
                    | changed(before_returns, after["target_shareholder_simple_return"])
                ),
            )
        )
        for e in source["events"]:
            effect = int(np.searchsorted(dates, np.datetime64(e["effective_date"])))
            name = isins.index(e["isin"])
            for t in range(effect - max(HORIZONS), effect):
                if active[t, name] and observed[t, name]:
                    for h, horizon in enumerate(HORIZONS):
                        if t < effect <= t + horizon:
                            scope.add((int(np.searchsorted(rows, t)), name, h))
        records = []
        for local, name, h in sorted(scope):
            day = int(rows[local])
            end = day + HORIZONS[h]
            ok = bool(active[day, name] and observed[day, name] and end < len(dates))
            holdings, cash = {name: Decimal(1)}, Decimal(0)
            if ok:
                for t in range(day + 1, end + 1):
                    nxt = {}
                    for claim, q in holdings.items():
                        e = event_map.get((t, claim))
                        if e:
                            for leg in e.legs:
                                auction = leg.fractional_auction
                                if auction is not None and end >= (
                                    leg.delivery_session
                                    if auction.available_session is None
                                    else auction.available_session
                                ):
                                    ok = False
                                j = leg.successor_index
                                nxt[j] = nxt.get(j, Decimal(0)) + q * Decimal(
                                    str(leg.shares_per_prior_share)
                                )
                            cash += q * Decimal(str(e.cash_per_prior_share))
                        elif actions.session_resolved[t, claim]:
                            cash += q * Decimal(
                                str(actions.cash_per_prior_share[t, claim])
                            )
                            j = int(actions.successor_index[t, claim])
                            nxt[j] = nxt.get(j, Decimal(0)) + q * Decimal(
                                str(actions.shares_per_prior_share[t, claim])
                            )
                        else:
                            ok = False
                    holdings = {j: q for j, q in nxt.items() if q}
                ok &= all(
                    observed[end, j]
                    and np.isfinite(close[end, j])
                    and close[end, j] > 0
                    for j in holdings
                )
            assert bool(after["target_shareholder_valid"][local, name, h]) == ok, (
                dates[day],
                isins[name],
                h,
                ok,
            )
            record = dict(
                date=str(dates[day]),
                isin=isins[name],
                horizon=HORIZONS[h],
                valid=bool(ok),
            )
            if ok:
                equity = sum(
                    (q * Decimal(str(close[end, j])) for j, q in holdings.items()),
                    Decimal(0),
                )
                entry = Decimal(str(close[day, name]))
                for key, x in (
                    ("target_terminal_wealth", (equity + cash) / entry),
                    ("target_shareholder_simple_return", (equity + cash) / entry - 1),
                    ("target_price_simple_return", equity / entry - 1),
                ):
                    actual = after[key][local, name, h]
                    expected = np.float32(float(x))
                    if actual != expected:
                        error = abs(float(actual) - float(expected))
                        bound = 32 * np.finfo(np.float64).eps * max(1, abs(float(x)))
                        assert error <= bound, (
                            record,
                            key,
                            actual,
                            expected,
                            error,
                            bound,
                        )
                        record.setdefault("float64_roundoff", []).append(
                            dict(
                                field=key,
                                actual=float(actual),
                                decimal_float32=float(expected),
                                error=error,
                                bound=bound,
                            )
                        )
                record.update(equity=str(equity), cash=str(cash), entry=str(entry))
            records.append(record)
        write_json_atomic(out / "endpoints.json", records)
        report = dict(
            status="qualified_new_primary_endpoints",
            input=run["stage_c_event_targets"],
            checked=len(records),
            valid=sum(r["valid"] for r in records),
            gains=int((after["target_shareholder_valid"] & ~before_valid).sum()),
            losses=int((before_valid & ~after["target_shareholder_valid"]).sum()),
            records=binding(out / "endpoints.json"),
            seconds=perf_counter() - tick,
        )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_event_" + mode + "_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
