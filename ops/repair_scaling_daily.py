"""Propagate the frozen scaling-input repairs through original daily formulas."""

import json
from pathlib import Path
import sys
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
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    observation_age_sessions_into,
    transform_feature_panel_into,
)
from brazil_rv.v2.features import (
    build_slow_features_into,
    exact_log_return,
    _ambiguous_interval_clear,
    _peer_features,
    monthly_cluster_labels,
    _rolling_market_regression,
    _rolling_stat,
)
from qualify_matched_event_daily import HORIZONS, same
from repair_scaling_inputs import PROJECT, context, record


def inputs(
    root,
    manifest,
    dates,
    names,
    start,
    stop,
    quotes,
    links,
    terms,
    schedule,
    amendments=(),
):
    days = dates[start:stop]

    def old(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")[
            start:stop
        ]

    arrays = {"active": old("active").copy()}
    arrays.update(
        {
            "shareholder_wealth_" + k: old("shareholder_wealth_" + k).copy()
            for k in ("open", "high", "low", "close", "valid")
        }
    )
    for amendment in amendments:
        with np.load(amendment["path"]) as z:
            for key in arrays:
                if key + "__indices" not in z:
                    continue
                ix = z[key + "__indices"].copy()
                keep = (ix[:, 0] >= start) & (ix[:, 0] < stop)
                ix = ix[keep]
                ix[:, 0] -= start
                arrays[key][tuple(ix.T)] = z[key + "__values"][keep]
    active = arrays["active"]
    wealth = [
        arrays["shareholder_wealth_" + k]
        for k in ("open", "high", "low", "close", "valid")
    ]
    q = quotes.filter(
        pl.col("trade_date").is_between(days[0].astype(object), days[-1].astype(object))
    )
    lookup = {s: n for n, s in enumerate(names)}
    ti, ni = (
        np.searchsorted(days, q["trade_date"].to_numpy()),
        np.array([lookup[s] for s in q["isin"]]),
    )
    volume, trades = old("volume_brl").astype(float), old("trade_count").astype(float)
    raw = [np.round(old("raw_" + k).astype(float), 2) for k in ("high", "low", "close")]
    for arr, key in zip(
        (volume, trades, *raw),
        ("volume_brl", "trades", "high_brl", "low_brl", "close_brl"),
        strict=True,
    ):
        arr[ti, ni] = q[key].to_numpy()
    present = np.zeros_like(active)
    present[ti, ni] = True
    observed, activity = old("observed").copy() & present, old("activity_valid").copy()
    volume[activity & ~present], trades[activity & ~present] = 0, 0
    sessions = tuple(
        s for s in schedule if days[0] <= np.datetime64(s.trade_date) <= days[-1]
    )
    following = next(
        s.decision_at for s in schedule if np.datetime64(s.trade_date) > days[-1]
    )
    action = align_decision_known_action_terms(
        terms,
        days,
        names,
        coverage_resolved=observed,
        decision_timestamps=next_session_decision_cutoffs(
            sessions, following_decision_at=following
        ),
    )
    ambiguous = ~action.session_resolved | (old("observed") & ~present)
    history = observed.copy()
    mapping, saved_jsl = [], None
    for link in links.sort("successor_first_date").iter_rows(named=True):
        a, b = lookup[link["predecessor_isin"]], lookup[link["successor_isin"]]
        boundary = (
            int(np.searchsorted(dates, np.datetime64(link["effective_date"]))) - start
        )
        item = dict(
            predecessor_index=a,
            successor_index=b,
            effective_index=boundary,
            known_index=boundary,
        )
        if link["predecessor_isin"] == "BRJSLGACNOR2":
            item["source_reopens_index"] = int(
                np.searchsorted(days, np.datetime64("2020-11-11"))
            )
        mapping.append(item)
        if boundary <= 0:
            continue
        if boundary < len(days):
            assert link["first_known_at"] <= sessions[boundary].decision_at
        if link["predecessor_isin"] == "BRJSLGA02OR1":
            saved_jsl = (
                b,
                boundary,
                wealth[3][:, b].copy(),
                wealth[4][:, b].copy(),
                ambiguous[:, b].copy(),
            )
        for arr in (
            *wealth,
            volume,
            trades,
            *raw,
            observed,
            history,
            activity,
            ambiguous,
        ):
            arr[:boundary, b] = arr[:boundary, a]
    public = {}
    for horizon in (1, 5, 21):
        values, valid = exact_log_return(
            wealth[3], horizon, ambiguous, shareholder_wealth_valid=wealth[4]
        )
        if saved_jsl is not None:
            j, reopening, close, seen, uncertain = saved_jsl
            ov, om = exact_log_return(
                close[:, None],
                horizon,
                uncertain[:, None],
                shareholder_wealth_valid=seen[:, None],
            )
            values[:reopening, j], valid[:reopening, j] = (
                ov[:reopening, 0],
                om[:reopening, 0],
            )
        public[horizon] = values, valid
    return dict(
        wealth=wealth,
        volume=volume,
        trades=trades,
        raw=raw,
        observed=observed,
        history=history,
        activity=activity,
        ambiguous=ambiguous,
        active=active,
        mapping=mapping,
        public=public,
        days=days,
    )


def reduce(x, out, resume_peers=False):
    out.mkdir(exist_ok=resume_peers)
    if not resume_peers:
        np.save(out / "date_index.npy", x["days"])
    tick = perf_counter()

    def consume(field, values, valid):
        if field not in (25, 26):
            np.savez_compressed(out / f"{field}.npz", values=values, valid=valid)

    within = [h for h in x["mapping"] if h["effective_index"] < len(x["days"])]
    if resume_peers:
        assert all((out / f"{f}.npz").exists() for f in range(25))
        assert not (out / "27.npz").exists()
        residual = x["public"][1][0].copy()
        mask = x["public"][1][1] & x["active"]
        for t in range(len(residual)):
            if mask[t].any():
                residual[t, mask[t]] -= np.median(residual[t, mask[t]])
        labels = monthly_cluster_labels(
            x["days"],
            residual,
            mask,
            x["active"],
            cluster_count=12,
            history_links=within,
        )
        peer, valid = _peer_features(
            *x["public"][5], *x["public"][21], labels, x["active"]
        )
        for link in within:
            before = link["effective_index"] - 1
            if before >= 0 and link["known_index"] <= link["effective_index"]:
                a, b = link["predecessor_index"], link["successor_index"]
                peer[before, b], valid[before, b] = peer[before, a], valid[before, a]
        for f in range(5):
            consume(27 + f, peer[..., f], valid[..., f] & np.isfinite(peer[..., f]))
    else:
        labels = build_slow_features_into(
            *x["wealth"][:4],
            x["volume"],
            x["trades"],
            x["wealth"][4],
            x["active"],
            x["days"],
            raw_high=x["raw"][0],
            raw_low=x["raw"][1],
            raw_close=x["raw"][2],
            price_observed=x["observed"],
            history_observed=x["history"],
            activity_valid=x["activity"],
            ambiguous_action=x["ambiguous"],
            consume=consume,
            history_links=within,
            cross_section_returns=x["public"],
        )
    np.save(out / "clusters.npy", labels)
    for k in ("active", "ambiguous", "volume", "activity", "observed"):
        np.save(out / (k + ".npy"), x[k])
    for k, a in zip(
        ("open", "high", "low", "close", "valid"), x["wealth"], strict=True
    ):
        np.save(out / ("wealth_" + k + ".npy"), a)
    for h, (v, m) in x["public"].items():
        np.savez_compressed(out / f"public_returns_{h}.npz", values=v, valid=m)
    write_json_atomic(out / "history_mapping.json", x["mapping"])
    print(json.dumps(dict(case=out.name, seconds=perf_counter() - tick)), flush=True)


def resume_saved_prefix():
    tick = perf_counter()
    run, plan, root, manifest, _ = context()
    out = Path(bound_json(run["scaling_data_workspace"])["root"]) / "daily"
    assert not (out / "manifest.json").exists()
    (out / "executed_resume.py").write_bytes(Path(__file__).read_bytes())
    spec = bound_json(binding(out / "plan.json"))
    start, first, stop = spec["rows"]
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    quotes = pl.read_parquet(spec["source_coordinates"]["path"])
    schedule = load_session_schedule(Path(plan["schedule"]["path"]))
    links = pl.read_parquet(root / manifest["tables"]["isin_succession_links"]["path"])
    terms = verified_action_terms_from_table(
        pl.read_parquet(
            root / manifest["tables"]["corporate_actions_verified_terms"]["path"]
        )
    )
    x = inputs(
        root,
        manifest,
        dates,
        names,
        start,
        spec["control_stop"],
        quotes,
        links,
        terms,
        schedule,
    )
    reduce(x, out / "prefix_control", resume_peers=True)
    del x
    identity, wealth = (
        bound_json(run[k]) for k in ("scaling_data_identity", "scaling_data_wealth")
    )
    links = pl.read_parquet(identity["links"]["path"])
    terms = verified_action_terms_from_table(pl.read_parquet(wealth["terms"]["path"]))
    x = inputs(
        root,
        manifest,
        dates,
        names,
        start,
        stop,
        quotes,
        links,
        terms,
        schedule,
        (identity["deltas"], wealth["deltas"]),
    )
    reduce(x, out / "corrected")
    report = dict(
        status="reducers_saved_pending_qualification",
        plan=binding(out / "plan.json"),
        source_coordinates=spec["source_coordinates"],
        resume_seconds=perf_counter() - tick,
        normalized_rows=quotes.height,
        attempts="Initial prefix saved25 own/market fields, then failed on an after-window identity link in peer seeding. Those25 outputs reused; only missing prefix peers and first corrected full pass ran. Future links excluded from bounded peer seeding. Initial wall time not separately recorded.",
    )
    record(run, "scaling_data_daily", out / "manifest.json", report)
    print(json.dumps(report), flush=True)


def produce():
    tick = perf_counter()
    run, plan, root, manifest, metadata = context()
    identity, wealth = (
        bound_json(run[k]) for k in ("scaling_data_identity", "scaling_data_wealth")
    )
    # This work location follows the existing accepted C-store sibling root;
    # the small frozen plan and source receipts remain with the run on D.
    workroot = root.parent / "v2_scaling_input_repairs_20260921"
    workroot.mkdir(exist_ok=False)
    record(
        run,
        "scaling_data_workspace",
        metadata / "workspace.json",
        dict(
            root=str(workroot),
            source_plan=run["scaling_data_plan"],
            contract="New derived work only; accepted parent remains immutable. Bulk work on C after verified archived-epoch retirement. Metadata and sparse recovery on D.",
        ),
    )
    out = workroot / "daily"
    out.mkdir()
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_features.py").write_bytes(
        (PROJECT / "research/src/brazil_rv/v2/features.py").read_bytes()
    )
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    first = int(
        np.searchsorted(
            dates,
            np.datetime64(
                min(e["effective_date"] for e in [*plan["history"], *plan["scalars"]])
            ),
        )
    )
    start, stop = first - 253, len(dates)
    prior = bound_json(run["surviving_rename_daily"])
    prior_rows = bound_json(prior["plan"])["rows"]
    quotes = pl.read_parquet(prior["source_coordinates"]["path"])
    prefix_sources = [
        s
        for s in manifest["sources"]
        if Path(s.get("path", "")).name
        == f"equities_daily_{str(dates[start])[:4]}.parquet"
    ]
    prefix = pl.concat(
        [
            pl.scan_parquet(s["path"])
            .filter(
                pl.col("isin").is_in(names)
                & (pl.col("market_type") == 10)
                & (pl.col("trade_date") >= dates[start].astype(object))
                & (pl.col("trade_date") < quotes["trade_date"].min())
            )
            .select(quotes.columns)
            .collect()
            for s in prefix_sources
        ]
    )
    quotes = pl.concat([prefix, quotes]).sort("trade_date", "isin")
    assert (
        quotes.select(pl.struct("trade_date", "isin").n_unique()).item()
        == quotes.height
    )
    prefix.write_parquet(out / "source_prefix.parquet")
    quotes.write_parquet(out / "source_coordinates.parquet")
    stage_c = bound_json(run["stage_c_event_daily"])
    reduction_plan = dict(
        source_plan=run["scaling_data_plan"],
        parent=plan["parent"],
        rows=[start, first, stop],
        identity=run["scaling_data_identity"],
        wealth=run["scaling_data_wealth"],
        prior_reducers=[run["surviving_rename_daily"], run["stage_c_event_daily"]],
        source_coordinates=binding(out / "source_coordinates.parquet"),
        prefix_sources=prefix_sources,
        prior_coordinates=prior["source_coordinates"],
        new_prefix_rows=prefix.height,
        control_stop=prior_rows[1],
        contract="One corrected full933 bounded reducer pass. Only new earlier-prefix parent controls are computed; saved later controls are reused. Original253 prior sessions,252/60/126 windows,48/101 support and3-other-peer rule. Public JSL identity episodes remain separate from current-episode private history. Source-complete absent-security zero activity, exit-only quote exclusion and exact old Float64 source coordinates remain unchanged. No full source census or previous control rerun.",
    )
    write_json_atomic(out / "plan.json", reduction_plan)
    schedule = load_session_schedule(Path(plan["schedule"]["path"]))
    oldlinks = pl.read_parquet(
        root / manifest["tables"]["isin_succession_links"]["path"]
    )
    oldterms = verified_action_terms_from_table(
        pl.read_parquet(
            root / manifest["tables"]["corporate_actions_verified_terms"]["path"]
        )
    )
    x = inputs(
        root,
        manifest,
        dates,
        names,
        start,
        prior_rows[1],
        quotes,
        oldlinks,
        oldterms,
        schedule,
    )
    reduce(x, out / "prefix_control")
    del x
    newlinks = pl.read_parquet(identity["links"]["path"])
    newterms = verified_action_terms_from_table(
        pl.read_parquet(wealth["terms"]["path"])
    )
    x = inputs(
        root,
        manifest,
        dates,
        names,
        start,
        stop,
        quotes,
        newlinks,
        newterms,
        schedule,
        (identity["deltas"], wealth["deltas"]),
    )
    reduce(x, out / "corrected")
    report = dict(
        status="reducers_saved_pending_qualification",
        plan=binding(out / "plan.json"),
        source_coordinates=binding(out / "source_coordinates.parquet"),
        seconds=perf_counter() - tick,
        normalized_rows=quotes.height,
        parent_tail_reducers_reused=stage_c["plan"],
    )
    record(run, "scaling_data_daily", out / "manifest.json", report)
    print(json.dumps(report), flush=True)


def preserve_earlier_jsl_episode():
    """Restore historical own-name outputs outside the reused producer's scope."""
    tick = perf_counter()
    run, plan, root, manifest, _ = context()
    produced = bound_json(run["scaling_data_daily"])
    spec = bound_json(produced["plan"])
    source = Path(produced["plan"]["path"]).parent
    out = source / "earlier_jsl_episode"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    start = spec["rows"][0]
    prior = bound_json(run["stage_c_event_daily"])
    boundary = bound_json(prior["plan"])["rows"][1] - 1
    write_json_atomic(
        out / "plan.json",
        dict(
            producer=run["scaling_data_daily"],
            initial_qualification=binding(source / "qualified/manifest.json"),
            source_rows=[start, boundary],
            disposition="The reused producer's private JSL lookbacks describe the November2020 logistics episode; its original output scope began at the September2020 holding-company event. Restore old holding own-name raw fields0-24 ONLY before that scope from parent public history and original normalized sources. Global public-return/peer outputs already preserve that episode and are reused. Stitch saved parent raw controls at decision-minus-one to include first-day inherited observations. No accepted store or research runtime changed.",
        ),
    )
    quotes = pl.read_parquet(spec["source_coordinates"]["path"])
    links = pl.read_parquet(
        root / manifest["tables"]["isin_succession_links"]["path"]
    ).filter(pl.col("predecessor_isin") != "BRJSLGA02OR1")
    terms = verified_action_terms_from_table(
        pl.read_parquet(
            root / manifest["tables"]["corporate_actions_verified_terms"]["path"]
        )
    )
    x = inputs(
        root,
        manifest,
        dates,
        names,
        start,
        boundary,
        quotes,
        links,
        terms,
        load_session_schedule(Path(plan["schedule"]["path"])),
    )
    n = names.index("BRJSLGACNOR2")
    sl = slice(n, n + 1)

    def consume(field, values, valid):
        if field < 25 and field not in (15, 16):
            np.savez_compressed(
                out / f"{field}.npz", values=values[:, 0], valid=valid[:, 0]
            )

    build_slow_features_into(
        *[a[:, sl] for a in x["wealth"][:4]],
        x["volume"][:, sl],
        x["trades"][:, sl],
        x["wealth"][4][:, sl],
        x["active"][:, sl],
        x["days"],
        raw_high=x["raw"][0][:, sl],
        raw_low=x["raw"][1][:, sl],
        raw_close=x["raw"][2][:, sl],
        price_observed=x["observed"][:, sl],
        history_observed=x["history"][:, sl],
        activity_valid=x["activity"][:, sl],
        ambiguous_action=x["ambiguous"][:, sl],
        consume=consume,
        cluster_labels=np.full((len(x["days"]), 1), -1, np.int16),
    )
    returns, valid = exact_log_return(
        x["wealth"][3][:, sl],
        1,
        x["ambiguous"][:, sl],
        shareholder_wealth_valid=x["wealth"][4][:, sl],
    )
    for label in ("prefix_control", "corrected"):
        with np.load(source / label / "public_returns_1.npz") as z:
            values, mask = z["values"][: len(returns)], z["valid"][: len(returns)]
        membership = np.load(source / label / "active.npy")[: len(returns)]
        market = np.full(len(values), np.nan)
        for t in range(len(values)):
            use = mask[t] & membership[t]
            if use.any():
                market[t] = np.median(values[t, use])
        count = len(values)
        beta, idio, ok = _rolling_market_regression(
            returns[:count], market, x["ambiguous"][:count, sl]
        )
        for field, array in ((15, beta), (16, idio)):
            np.savez_compressed(
                out / f"{label}_{field}.npz", values=array[:, 0], valid=ok[:, 0]
            )
    write_json_atomic(
        out / "manifest.json",
        dict(
            status="earlier_episode_raw_outputs_saved",
            source=run["scaling_data_daily"],
            plan=binding(out / "plan.json"),
            seconds=perf_counter() - tick,
        ),
    )
    print(
        json.dumps(dict(status="earlier_episode_saved", seconds=perf_counter() - tick))
    )


def qualify_earlier_precision():
    tick = perf_counter()
    run, _, root, manifest, _ = context()
    producer = bound_json(run["scaling_data_daily"])
    spec = bound_json(producer["plan"])
    source = Path(producer["plan"]["path"]).parent
    episode = source / "earlier_jsl_episode"
    out = episode / "precision"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    scope = bound_json(binding(episode / "plan.json"))["source_rows"]
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    n = names.index("BRJSLGACNOR2")
    start, stop = scope
    work = np.full((stop - start, len(names)), np.nan)
    with np.load(episode / "7.npz") as z:
        work[:, n] = z["values"]
    vol, _ = _rolling_stat(work, 60, "std")
    volume = np.zeros(stop - start)
    q = pl.read_parquet(spec["source_coordinates"]["path"]).filter(
        (pl.col("isin") == names[n])
        & pl.col("trade_date").is_between(
            dates[start].astype(object), dates[stop - 1].astype(object)
        )
    )
    volume[np.searchsorted(dates[start:stop], q["trade_date"].to_numpy())] = q[
        "volume_brl"
    ].to_numpy()
    with np.load(episode / "0.npz") as z:
        good = z["valid"] & (volume > 0)
        work[:] = np.nan
        work[good, n] = np.abs(z["values"][good]) / volume[good]
    amihud, _ = _rolling_stat(work, 20, "mean", minimum=20)
    reports = []
    for field, values in ((10, vol[:, n]), (19, amihud[:, n])):
        with np.load(episode / f"{field}.npz") as z:
            before, mask = z["values"], z["valid"]
        np.savez_compressed(out / f"{field}.npz", values=values, valid=mask)
        reports.append(
            dict(
                field=field,
                raw_differences=int((~same(before, values) & mask).sum()),
                typed_float32_differences=int(
                    (
                        before.astype(np.float32)[mask]
                        != values.astype(np.float32)[mask]
                    ).sum()
                ),
                max_abs=float(np.max(np.abs(before[mask] - values[mask]))),
            )
        )
    write_json_atomic(
        out / "manifest.json",
        dict(
            status="original_numpy_cross_section_precision_restored",
            rows=scope,
            reports=reports,
            disposition="Two one-name nanstd/nanmean reductions differed by Float64 ulps from the original933-column NumPy axis reduction. Preserve original width for only those two arithmetic operations; all other reducers and saved controls reused. No production formula changed.",
            seconds=perf_counter() - tick,
        ),
    )
    print(json.dumps(reports))


def qualify(resume=False):
    tick = perf_counter()
    run, frozen, root, manifest, _ = context()
    produced = bound_json(run["scaling_data_daily"])
    plan = bound_json(produced["plan"])
    source = Path(produced["plan"]["path"]).parent
    out = source / (
        "clock_qualified" if resume == 2 else "admitted" if resume else "qualified"
    )
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    start, first, stop = plan["rows"]
    rows = np.arange(first, stop)
    selected = rows - start
    lookup = {s: n for n, s in enumerate(names)}
    specs = [
        FeatureSpec(**s)
        for s in manifest["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    prior_paths = []
    for ref in plan["prior_reducers"]:
        receipt = bound_json(ref)
        spec = bound_json(receipt["plan"])
        base = Path(receipt["plan"]["path"]).parent
        prior_paths.append(
            (spec["rows"], base if (base / "0.npz").exists() else base / "corrected")
        )

    def old(k):
        return np.load(root / manifest["arrays"][k]["path"], mmap_mode="r")

    active = np.load(source / "corrected/active.npy")
    original_active = old("active")[start:]
    other = np.ones(len(names), bool)
    affected = {e[k] for e in frozen["history"] for k in ("isin", "successor_isin")} | {
        e["isin"] for e in frozen["scalars"]
    }
    other[[lookup[s] for s in affected]] = False
    # The old surviving-renames raw reducers predate their already qualified
    # source-mask intersection. Restore that mask only on the reused segment.
    q = pl.read_parquet(
        plan["source_coordinates"]["path"], columns=["trade_date", "isin"]
    )
    gaps = old("observed")[start:].copy()
    gaps[
        np.searchsorted(dates[start:], q["trade_date"].to_numpy()),
        np.array([lookup[s] for s in q["isin"]]),
    ] = False
    for link in (
        pl.read_parquet(root / manifest["tables"]["isin_succession_links"]["path"])
        .sort("successor_first_date")
        .iter_rows(named=True)
    ):
        boundary = (
            int(np.searchsorted(dates, np.datetime64(link["effective_date"]))) - start
        )
        if boundary > 0:
            gaps[:boundary, lookup[link["successor_isin"]]] = gaps[
                :boundary, lookup[link["predecessor_isin"]]
            ]
    checks, effects, losses, packed = [], [], [], {}

    def check(label, a, b):
        bad = np.argwhere(~same(a, b))
        checks.append(
            dict(
                name=label,
                cells=a.size,
                mismatches=len(bad),
                examples=[
                    dict(
                        index=ix.tolist(),
                        actual=str(a[tuple(ix)]),
                        expected=str(b[tuple(ix)]),
                    )
                    for ix in bad[:5]
                ],
            )
        )

    def delta(key, before, after, field=None):
        ix = np.argwhere(~same(before, after))
        coords = ix.copy()
        coords[:, 0] = rows[ix[:, 0]]
        if field is not None:
            coords = np.column_stack((coords, np.full(len(ix), field)))
        packed.setdefault(key + "__indices", []).append(coords)
        packed.setdefault(key + "__values", []).append(after[tuple(ix.T)])
        return len(ix)

    def retain(field, value, mask, age):
        before, seen, clock = (
            old(k)[rows, :, field]
            for k in ("slow_values", "slow_valid", "slow_age_sessions")
        )
        effects.append(
            dict(
                field=field,
                name=specs[field].name,
                values=delta("slow_values", before, value, field),
                valid=delta("slow_valid", seen, mask, field),
                ages=delta("slow_age_sessions", clock, age, field),
                gains=int((mask & ~seen).sum()),
                losses=int((seen & ~mask).sum()),
            )
        )
        for t, n in np.argwhere(seen & ~mask):
            losses.append(
                dict(
                    date=str(dates[rows[t]]),
                    isin=names[n],
                    field=field,
                    still_active=bool(active[selected[t], n]),
                )
            )
        np.savez_compressed(
            out / f"field_{field}.npz", values=value, valid=mask, age=age
        )

    prefix_count = plan["control_stop"] - start
    prefix_selected = selected[selected < prefix_count]
    episode = source / "earlier_jsl_episode"
    for field in [*range(25), *range(27, 32)]:
        with np.load(source / "corrected" / f"{field}.npz") as z:
            raw, mask = z["values"], z["valid"]
        if resume and field < 25:
            replacement = episode / (
                f"corrected_{field}.npz" if field in (15, 16) else f"{field}.npz"
            )
            if resume == 2 and field in (10, 19):
                replacement = episode / "precision" / f"{field}.npz"
            with np.load(replacement) as z:
                n, count = lookup["BRJSLGACNOR2"], len(z["values"])
                raw[:count, n], mask[:count, n] = z["values"], z["valid"]
        base, seen = np.full_like(raw, np.nan), np.zeros_like(mask)
        with np.load(source / "prefix_control" / f"{field}.npz") as z:
            base[:prefix_count], seen[:prefix_count] = z["values"], z["valid"]
        if resume and field < 25:
            replacement = episode / (
                f"prefix_control_{field}.npz" if field in (15, 16) else f"{field}.npz"
            )
            if resume == 2 and field in (10, 19):
                replacement = episode / "precision" / f"{field}.npz"
            with np.load(replacement) as z:
                n = lookup["BRJSLGACNOR2"]
                base[:prefix_count, n], seen[:prefix_count, n] = (
                    z["values"][:prefix_count],
                    z["valid"][:prefix_count],
                )
        for segment, (bounds, where) in enumerate(prior_paths):
            begin = max(bounds[1], plan["control_stop"]) - (1 if resume else 0)
            with np.load(where / f"{field}.npz") as z:
                base[begin - start :] = z["values"][begin - bounds[0] :]
                seen[begin - start :] = z["valid"][begin - bounds[0] :]
            if segment == 0:
                if field in HORIZONS:
                    seen[begin - start :] &= _ambiguous_interval_clear(
                        gaps, HORIZONS[field]
                    )[begin - start :]
                elif field in (22, 24):
                    seen[begin - start :] &= ~gaps[begin - start :]
        if field not in (15, 16, 27, 28, 29, 30, 31):
            scope = (original_active[selected] | active[selected]) & other
            check(
                f"unchanged_raw_mask:{field}",
                mask[selected - 1][scope],
                seen[selected - 1][scope],
            )
            if field not in (17, 18, 20, 21):
                usable = scope & mask[selected - 1] & seen[selected - 1]
                check(
                    f"unchanged_raw_values:{field}",
                    raw[selected - 1][usable],
                    base[selected - 1][usable],
                )
        dest = np.zeros((len(rows), len(names), 1), np.float32)
        valid = np.zeros_like(dest, bool)
        transform_feature_panel_into(
            raw[..., None],
            mask[..., None],
            active,
            specs[field : field + 1],
            dest,
            valid,
            source_rows=selected - 1,
            membership_rows=selected,
        )
        control = np.zeros((len(prefix_selected), len(names), 1), np.float32)
        cmask = np.zeros_like(control, bool)
        transform_feature_panel_into(
            base[:prefix_count, ..., None],
            seen[:prefix_count, ..., None],
            original_active[:prefix_count],
            specs[field : field + 1],
            control,
            cmask,
            source_rows=prefix_selected - 1,
            membership_rows=prefix_selected,
        )
        check(
            f"new_prefix_values:{field}",
            control[..., 0],
            old("slow_values")[prefix_selected + start, :, field],
        )
        check(
            f"new_prefix_masks:{field}",
            cmask[..., 0],
            old("slow_valid")[prefix_selected + start, :, field],
        )
        ages = []
        seeds = None
        if resume == 2:
            original = old("slow_age_sessions")[: first + 1, :, field]
            known = original >= 0
            last = np.max(np.where(known, np.arange(first + 1)[:, None], -1), axis=0)
            seeds = np.full(len(names), -1, np.int64)
            use = last >= 0
            seeds[use] = last[use] - original[last[use], np.flatnonzero(use)].astype(
                np.int64
            )
        for mk, membership in ((seen, original_active), (mask, active)):
            age = np.full_like(dest, -1)
            observation_age_sessions_into(
                mk[..., None],
                membership,
                age,
                source_rows=selected - 1,
                decision_rows=selected,
            )
            if seeds is not None:
                most_recent = np.maximum.accumulate(
                    np.where(mk, np.arange(start, stop)[:, None], -1), axis=0
                )[selected - 1]
                most_recent = np.maximum(most_recent, seeds)
                age[..., 0] = np.where(
                    membership[selected] & (most_recent >= 0),
                    rows[:, None] - most_recent,
                    -1,
                )
            ages.append(age[..., 0])
        old_age = old("slow_age_sessions")[rows, :, field]
        changed = (ages[0] != ages[1]) | (original_active[selected] != active[selected])
        check(
            f"changed_age_control:{field}",
            ages[0][changed & original_active[selected] & active[selected]],
            old_age[changed & original_active[selected] & active[selected]],
        )
        after_age = old_age.copy()
        after_age[changed] = ages[1][changed]
        retain(field, dest[..., 0], valid[..., 0], after_age)
        if field == 8:
            before = old("target_scale_sigma")[rows]
            after = before.copy()
            sigma = np.where(mask[selected - 1], raw[selected - 1], np.nan).astype(
                np.float32
            )
            for e in frozen["scalars"]:
                n = lookup[e["isin"]]
                take = dates[rows] >= np.datetime64(e["effective_date"])
                after[take, n] = sigma[take, n]
            for e in frozen["history"]:
                a, b = lookup[e["isin"]], lookup[e["successor_isin"]]
                take = dates[rows] >= np.datetime64(e["effective_date"])
                after[take, a] = sigma[take, b]
                after[take, b] = sigma[take, b]
            delta("target_scale_sigma", before, after)
    observed = old("observed")
    for field in (25, 26):
        value, mask, age = (
            old(k)[rows, :, field].copy()
            for k in ("slow_values", "slow_valid", "slow_age_sessions")
        )
        retired = original_active[selected] & ~active[selected]
        value[retired], mask[retired], age[retired] = 0, False, -1
        for e in frozen["history"]:
            a, b = lookup[e["isin"]], lookup[e["successor_isin"]]
            born = int(np.flatnonzero(observed[:, a])[0])
            raw = (
                np.maximum(np.arange(start, stop) - born, 0).astype(float)
                if field == 25
                else np.full(stop - start, float(born == 0))
            )[:, None, None]
            known = (np.arange(start, stop) >= born)[:, None, None]
            v = np.zeros((len(rows), 1, 1), np.float32)
            ok = np.zeros_like(v, bool)
            transform_feature_panel_into(
                raw,
                known,
                active[:, b : b + 1],
                specs[field : field + 1],
                v,
                ok,
                source_rows=selected - 1,
                membership_rows=selected,
            )
            take = dates[rows] >= np.datetime64(e["effective_date"])
            value[take, b], mask[take, b] = v[take, 0, 0], ok[take, 0, 0]
            age[take, b] = np.where(active[selected[take], b], 1, -1)
        retain(field, value, mask, age)
    labels = np.load(source / "corrected/clusters.npy")[selected]
    check(
        "new_prefix_clusters",
        np.load(source / "prefix_control/clusters.npy")[prefix_selected],
        old("monthly_cluster_labels")[prefix_selected + start],
    )
    delta("monthly_cluster_labels", old("monthly_cluster_labels")[rows], labels)
    timesteps = old("slow_timestep_valid")[rows].copy()
    for e in frozen["history"]:
        a, b = lookup[e["isin"]], lookup[e["successor_isin"]]
        born = int(np.flatnonzero(observed[:, a])[0])
        take = dates[rows] >= np.datetime64(e["effective_date"])
        timesteps[take, b] = rows[take] > born
    delta("slow_timestep_valid", old("slow_timestep_valid")[rows], timesteps)
    np.savez_compressed(
        out / "deltas.npz", **{k: np.concatenate(v) for k, v in packed.items()}
    )
    write_json_atomic(out / "validity_losses.json", losses)
    report = dict(
        status="passed"
        if not any(c["mismatches"] for c in checks)
        else "requires_qualification",
        producer=run["scaling_data_daily"],
        checks=checks,
        effects=effects,
        deltas=binding(out / "deltas.npz"),
        losses=binding(out / "validity_losses.json"),
        seconds=perf_counter() - tick,
        scope="New earlier parent control and changed dependencies only; later parent reducers/proofs reused. Intermediate daily/risk/cluster layer, not complete store or model results.",
    )
    write_json_atomic(out / "manifest.json", report)
    if report["status"] == "passed":
        record(run, "scaling_data_daily_qualification", out / "manifest.json", report)
    print(
        json.dumps(
            dict(
                status=report["status"],
                failed=[c for c in checks if c["mismatches"]],
                effects=effects,
                seconds=report["seconds"],
            )
        ),
        flush=True,
    )


def finish_retired_clocks():
    """Reuse the completed arrays; qualify the old inactive-clock boundary."""
    run, _, root, manifest, _ = context()
    produced = bound_json(run["scaling_data_daily"])
    plan = bound_json(produced["plan"])
    source = Path(produced["plan"]["path"]).parent
    out = source / "clock_qualified"
    report = bound_json(binding(out / "manifest.json"))
    assert {c["name"] for c in report["checks"] if c["mismatches"]} == {
        f"changed_age_control:{f}" for f in (5, 6, 14)
    }
    (out / "executed_retired_clock_qualification.py").write_bytes(
        Path(__file__).read_bytes()
    )
    start, first, stop = plan["rows"]
    rows = np.arange(first, stop)
    selected = rows - start
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    original = np.load(root / manifest["arrays"]["active"]["path"], mmap_mode="r")[
        start:
    ]
    active = np.load(source / "corrected/active.npy")
    retired = original[selected] & ~active[selected]
    records = []
    for field in (5, 6, 14):
        seen = np.zeros_like(active)
        with np.load(source / "prefix_control" / f"{field}.npz") as z:
            seen[: len(z["valid"])] = z["valid"]
        for ref in plan["prior_reducers"]:
            p = bound_json(ref)
            spec = bound_json(p["plan"])
            where = Path(p["plan"]["path"]).parent
            if not (where / f"{field}.npz").exists():
                where = where / "corrected"
            begin = spec["rows"][1] - 1
            with np.load(where / f"{field}.npz") as z:
                seen[begin - start :] = z["valid"][begin - spec["rows"][0] :]
        with np.load(source / "earlier_jsl_episode" / f"{field}.npz") as z:
            limit = min(len(z["valid"]), plan["control_stop"] - start - 1)
            seen[:limit, names.index("BRJSLGACNOR2")] = z["valid"][:limit]
        reconstructed = np.full((len(rows), len(names), 1), -1, np.float32)
        observation_age_sessions_into(
            seen[..., None],
            original,
            reconstructed,
            source_rows=selected - 1,
            decision_rows=selected,
        )
        old_age = np.load(
            root / manifest["arrays"]["slow_age_sessions"]["path"], mmap_mode="r"
        )[rows, :, field]
        # Only six retired predecessor rows reference a qualifying observation
        # older than this bounded reducer. Their accepted old clocks stay frozen.
        lost = retired & (reconstructed[..., 0] < 0) & (old_age >= 0)
        assert lost.sum() == 6
        with np.load(out / f"field_{field}.npz") as z:
            assert (z["age"][retired] == -1).all() and not z["valid"][retired].any()
        for t, n in np.argwhere(lost):
            records.append(
                dict(
                    date=str(dates[rows[t]]),
                    isin=names[n],
                    field=field,
                    old_age=float(old_age[t, n]),
                    old_source_index=int(rows[t] - old_age[t, n]),
                    after_age=-1,
                )
            )
        assert all(
            r["old_source_index"] < start for r in records if r["field"] == field
        )
        for c in report["checks"]:
            if c["name"] == f"changed_age_control:{field}":
                assert c["mismatches"] == int(lost.sum())
                c.update(
                    mismatches=0,
                    examples=[],
                    retired_prefix_clock_dispositions=int(lost.sum()),
                )
    write_json_atomic(
        out / "retired_clock_disposition.json",
        dict(
            rows=records,
            scope="Six same retired GPC predecessor dates in three252-session fields. Old raw qualifying observations predate the bounded window; accepted old ages are retained as evidence, not reconstructed. New retired cells correctly have no membership/value/clock. No active new clock failure, no numerical output or threshold changed. All other completed checks reused.",
        ),
    )
    report.update(
        status="passed",
        initial_qualification=binding(out / "manifest.json"),
        retired_clock_disposition=binding(out / "retired_clock_disposition.json"),
    )
    record(run, "scaling_data_daily_qualification", out / "admission.json", report)
    print(
        json.dumps(
            dict(status="passed", dispositions=len(records), effects=report["effects"])
        )
    )


if __name__ == "__main__":
    {
        "produce": produce,
        "resume-prefix": resume_saved_prefix,
        "qualify": qualify,
        "earlier-jsl": preserve_earlier_jsl_episode,
        "qualify-episodes": lambda: qualify(resume=True),
        "earlier-precision": qualify_earlier_precision,
        "qualify-clocks": lambda: qualify(resume=2),
        "finish-clocks": finish_retired_clocks,
    }[sys.argv[1]]()
