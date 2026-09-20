"""Carry the admitted market episodes into magnitudes and cross-market inputs."""

from datetime import date
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_exposures as exp
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.bova11 import load_bova11_series
from brazil_rv.v2.cross_market_returns import (
    admit_brent,
    comparison_return_mask,
    relative_return_panel,
)
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.foreign_flow import decision_panel
from brazil_rv.v2.hedge_beta import rolling_hedge_beta
from brazil_rv.v2.round5_derived import beta_source_age, identity_axes
from brazil_rv.v2.round5_magnitude import FEATURE_NAMES, magnitude_panel
from propagate_matched_event_context import source_basis

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    events = bound_json(admission["plan"])["events"]
    daily = bound_json(run["stage_c_event_daily"])
    qualified = bound_json(run["stage_c_event_daily_qualification"])
    start, first, stop = bound_json(daily["plan"])["rows"]
    dr = Path(daily["plan"]["path"]).parent
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = Path(admission["plan"]["path"]).parent / "market"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_basis.py").write_bytes(
        (PROJECT / "ops/propagate_matched_event_context.py").read_bytes()
    )
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            daily=run["stage_c_event_daily_qualification"],
            issuer=run["stage_c_event_issuers"],
            prior=run["surviving_rename_market"],
            context=run["stage_c_event_context"],
            rows=[start, first, stop],
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
            contrast="Five market episodes only; original magnitude formulas, Float32 volume/wealth, six120/60 regressions and known-sector issuer shrinkage. Each regression uses two chronological segments with120prior observations at the JSL reopening; before that decision old holding returns remain public, afterwards own logistics history applies. No old regression rerun. Existing TIMS ADR assignment is retained, not a new alias. Original source shock/flow/FX clocks and separate oil return contract stay fixed.",
            verification="Saved parent raw coordinates transform exactly to accepted arrays. New output segments, sparse changes, losses and selected source inputs retained. Existing source/regression proofs reused; final combined consumer and causal guards follow remaining families.",
        ),
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    days, sessions = (
        dates[start:stop].astype(object).tolist(),
        dates.astype(object).tolist(),
    )
    rows, selected = np.arange(first, stop), np.arange(first - start, stop - start)
    columns = [isins.index(e["successor_isin"]) for e in events]
    jsl, reopening = (
        isins.index("BRJSLGACNOR2"),
        int(np.searchsorted(days, date(2020, 11, 11))),
    )
    prior = bound_json(run["surviving_rename_market"])
    prior_root = Path(prior["plan"]["path"]).parent

    def old(key):
        return np.load(root / m["arrays"][key]["path"], mmap_mode="r")[start:stop]

    def after(key):
        a = old(key).copy()
        with np.load(qualified["deltas"]["path"]) as z:
            if key + "__indices" in z:
                ix = z[key + "__indices"].copy()
                ix[:, 0] -= start
                a[tuple(ix.T)] = z[key + "__values"]
        return a

    membership = [old("active"), np.load(dr / "active.npy")]
    slow, ages = after("slow_valid"), after("slow_age_sessions")
    slow_names = m["feature_names"]["slow"]
    reports, deltas = {}, {}

    def parent(family):
        with np.load(prior_root / f"{family}_corrected.npz") as z:
            use = np.searchsorted(z["rows"], rows)
            assert np.array_equal(z["rows"][use], rows)
            arrays = [z[k][use] for k in ("values", "valid", "ages")]
        return m["feature_names"]["sidecar_" + family], arrays

    def finish(family, names, before, changed):
        specs = [
            FeatureSpec(**s)
            for s in m["metadata"]["feature_schema"]["specifications"]
            if s["family"] == "sidecar_" + family
        ]
        typed, checks = [], []
        for i, (label, arrays) in enumerate(
            (("control", before), ("corrected", changed))
        ):
            values, valid = (
                np.zeros_like(arrays[0], np.float32),
                np.zeros_like(arrays[1]),
            )
            active = membership[i][selected]
            transform_feature_panel_into(
                arrays[0], arrays[1], active, specs, values, valid
            )
            age = np.where(active[..., None], arrays[2], -1).astype(np.float32)
            typed.append((values, valid, age))
            np.savez_compressed(
                out / f"{family}_{label}.npz",
                rows=rows,
                values=arrays[0],
                valid=arrays[1],
                ages=arrays[2],
            )
        effects = {}
        for j, suffix in enumerate(("values", "valid", "age_sessions")):
            key = "sidecar_" + family + "_" + suffix
            accepted = old(key)[selected]
            eq = (accepted == typed[0][j]) | (
                np.isnan(accepted) & np.isnan(typed[0][j])
            )
            checks.append(
                dict(
                    key=key,
                    cells=accepted.size,
                    mismatches=int((~eq).sum()),
                    examples=np.argwhere(~eq)[:4].tolist(),
                )
            )
            eq = (accepted == typed[1][j]) | (
                np.isnan(accepted) & np.isnan(typed[1][j])
            )
            ix = np.argwhere(~eq)
            deltas[key + "__indices"] = np.column_stack((rows[ix[:, 0]], ix[:, 1:]))
            deltas[key + "__values"] = typed[1][j][tuple(ix.T)]
            effects[suffix] = len(ix)
        losses = np.argwhere(typed[0][1] & ~typed[1][1])
        write_json_atomic(
            out / f"{family}_losses.json",
            [
                dict(
                    date=str(dates[rows[t]]),
                    isin=isins[n],
                    field=names[f],
                    still_active=bool(membership[1][selected[t], n]),
                )
                for t, n, f in losses
            ],
        )
        effects.update(
            gains=int((typed[1][1] & ~typed[0][1]).sum()),
            losses=len(losses),
            live_losses=int(sum(membership[1][selected[t], n] for t, n, _ in losses)),
            shared_changed=int(
                (typed[1][1] & typed[0][1] & (typed[1][0] != typed[0][0])).sum()
            ),
        )
        reports[family] = dict(checks=checks, effects=effects, names=names)
        write_json_atomic(out / f"{family}_report.json", reports[family])
        print(
            json.dumps(
                dict(family=family, **reports[family], seconds=perf_counter() - tick)
            ),
            flush=True,
        )
        assert not any(c["mismatches"] for c in checks)

    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    fm = bound_json(
        dict(
            path=str(Path(foundation["root"]) / "manifest.json"),
            sha256=foundation["manifest_sha256"],
        )
    )
    families = source_families(fm)
    mm = bound_json(families["magnitudes"]["source_manifest"])["sources"]["inputs"]
    bova = load_bova11_series(
        Path(mm["bova_manifest"]["path"]).parent,
        expected_manifest_sha256=mm["bova_manifest"]["sha256"],
        canonical_dates=sessions,
    ).close_by_session
    names, mag_parent = parent("magnitudes")
    mag_after = [a.copy() for a in mag_parent]
    wealth = {
        k: np.load(dr / f"wealth_{k}.npy")[:, columns]
        for k in ("open", "high", "low", "close", "valid")
    }
    volume, activity = old("volume_brl").copy(), old("activity_valid").copy()
    links = pl.read_parquet(admission["history_mapping"]["path"]).sort(
        "effective_index"
    )
    for link in links.iter_rows(named=True):
        boundary = link["effective_index"] - start
        if boundary > 0:
            for array in (volume, activity):
                array[:boundary, link["successor_index"]] = array[
                    :boundary, link["predecessor_index"]
                ]
    beta, known = rolling_hedge_beta(wealth["close"], wealth["valid"], bova[start:stop])
    age = beta_source_age(wealth["close"], wealth["valid"], bova[start:stop], known)
    mag = magnitude_panel(
        **{"wealth_" + k: v for k, v in wealth.items()},
        volume_brl=np.ascontiguousarray(volume[:, columns]),
        activity_valid=activity[:, columns],
        daily_feature_valid=slow[:, columns],
        slow_age_sessions=ages[:, columns],
        slow_feature_names=tuple(slow_names),
        economic_beta=beta,
        economic_beta_valid=known,
        economic_beta_age_sessions=age,
    )
    np.savez_compressed(
        out / "magnitude_reducers.npz",
        values=mag[0],
        valid=mag[1],
        ages=mag[2],
        columns=columns,
    )
    fields = [names.index(n) for n in FEATURE_NAMES]
    for j, event in enumerate(events):
        n = columns[j]
        use = np.flatnonzero(dates[rows] >= np.datetime64(event["effective_date"]))
        mask = mag[1][selected[use], j] & membership[1][selected[use], n, None]
        for k, a in enumerate(mag):
            v = (
                mask
                if k == 1
                else np.where(mask, a[selected[use], j], -1 if k == 2 else 0).astype(
                    np.float32
                )
            )
            mag_after[k][np.ix_(use, [n], fields)] = v[:, None, :]
    finish("magnitudes", names, mag_parent, mag_after)

    cross_names, cross_parent = parent("cross_market")
    cross_after = [a.copy() for a in cross_parent]
    cm = bound_json(families["cross_market"]["source_manifest"])["sources"]
    ordinary = bound_json(families["sector"]["source_manifest"])["sources"]["inputs"][
        "base_manifest"
    ]
    original = bound_json(
        binding(Path(cm["old_cross_market"]["path"]).parent / "manifest.json")
    )
    shocks = pl.read_parquet(original["shocks"]["path"])
    cdi = pl.read_parquet(original["cdi"]["path"])["daily_cdi_rate"].to_numpy()[
        start:stop
    ]
    levels, us = (
        pl.read_parquet(cm["market_levels"]["path"]),
        pl.read_parquet(cm["us_returns"]["path"]),
    )
    oil = exp.market_shocks(
        admit_brent(levels.filter(pl.col("series") == "brent_spot"), sessions),
        us.head(0),
    )
    shock_axes = exp.shock_axes(shocks, sessions)
    shock_axes["oil"] = exp.shock_axes(oil, sessions)["oil"]
    identity = bound_json(run["stage_c_event_issuers"])["artifacts"]["identity"]
    sectors, issuers = identity_axes(
        pl.read_parquet(identity["path"]).filter(
            pl.col("date").is_between(days[0], days[-1])
        ),
        days,
        isins,
    )
    returns = []
    for kind, record in (("ordinary", ordinary), ("oil", cm["base_manifest"])):
        arrays = source_basis(run, record, root, m, start, stop)
        oldret, oldmask = exact_log_return(
            arrays["shareholder_wealth_close"],
            1,
            shareholder_wealth_valid=arrays["shareholder_wealth_valid"],
        )
        oldmask[:-1] &= arrays["slow_valid"][1:, :, slow_names.index("log_return_1")]
        oldmask[-1] = False
        if kind == "oil":
            oldmask = comparison_return_mask(oldmask, arrays["action_has_action"])
        for event in events:
            a, b = isins.index(event["isin"]), isins.index(event["successor_isin"])
            boundary = int(
                np.searchsorted(days, date.fromisoformat(event["effective_date"]))
            )
            for k in ("close", "valid"):
                arrays["shareholder_wealth_" + k][:, b] = np.load(
                    dr / f"wealth_{k}.npy", mmap_mode="r"
                )[:, b]
            arrays["slow_valid"][:boundary, b] = arrays["slow_valid"][:boundary, a]
            arrays["slow_valid"][boundary:, b] = slow[boundary:, b]
            arrays["action_has_action"][:boundary, b] = arrays["action_has_action"][
                :boundary, a
            ]
        # The rejected inferred JSL factor is absent from the admitted gross terms.
        arrays["action_has_action"][reopening, jsl] = False
        ret, mask = exact_log_return(
            arrays["shareholder_wealth_close"],
            1,
            shareholder_wealth_valid=arrays["shareholder_wealth_valid"],
        )
        mask[:-1] &= arrays["slow_valid"][1:, :, slow_names.index("log_return_1")]
        mask[-1] = False
        if kind == "oil":
            mask = comparison_return_mask(mask, arrays["action_has_action"])
        public_ret, public_mask = ret.copy(), mask.copy()
        public_ret[:reopening, jsl], public_mask[:reopening, jsl] = (
            oldret[:reopening, jsl],
            oldmask[:reopening, jsl],
        )
        np.savez_compressed(
            out / f"{kind}_return_inputs.npz",
            values=ret,
            valid=mask,
            public_jsl_values=oldret[:, jsl],
            public_jsl_valid=oldmask[:, jsl],
        )
        returns.append(
            (
                ret - np.log1p(cdi)[:, None],
                mask,
                public_ret - np.log1p(cdi)[:, None],
                public_mask,
            )
        )
        if kind == "oil":
            oil_raw_returns = ret
    for name, factor in exp.EXPOSURES.items():
        current, shock_age, historical, public_index = shock_axes[factor]
        ret, mask, public_ret, public_mask = returns[name == "oil"]
        # Disjoint output episodes preserve full120history using unchanged OLS.
        # Prefix warmup outputs are discarded, not separate experiments.
        beta = np.zeros_like(ret, np.float32)
        known = np.zeros_like(mask)
        age = np.full_like(ret, -1, np.float32)
        for left, right, offset, y, ok in (
            (0, reopening, 0, public_ret, public_mask),
            (reopening, len(days), reopening - 120, ret, mask),
        ):
            result = exp.exposure_panel(
                y[offset:right],
                ok[offset:right],
                historical[start + offset : start + right],
                public_index[start + offset : start + right] - start - offset,
                sectors[offset:right],
                issuers[offset:right],
            )
            for dest, values in zip((beta, known, age), result, strict=True):
                dest[left:right] = values[left - offset :]
        np.savez_compressed(
            out / f"{name}_regression.npz", beta=beta, known=known, age=age
        )
        fields = [
            cross_names.index(k)
            for k in (
                f"exposure_{name}",
                f"exposure_{name}_times_shock_1",
                f"exposure_{name}_times_shock_5",
            )
        ]
        values = np.stack(
            (
                beta,
                beta * current[start:stop, 0, None],
                beta * current[start:stop, 1, None],
            ),
            -1,
        ).astype(np.float32)
        valid = np.stack(
            (
                known,
                known & np.isfinite(current[start:stop, 0, None]),
                known & np.isfinite(current[start:stop, 1, None]),
            ),
            -1,
        )
        ages3 = np.stack(
            (
                age,
                np.maximum(age, shock_age[start:stop, 0, None]),
                np.maximum(age, shock_age[start:stop, 1, None]),
            ),
            -1,
        ).astype(np.float32)
        valid = valid[selected] & membership[1][selected, :, None]
        for k, a in enumerate((values, valid, ages3)):
            cross_after[k][:, :, fields] = (
                valid if k == 1 else np.where(valid, a[selected], -1 if k == 2 else 0)
            )
        print(json.dumps(dict(factor=name, seconds=perf_counter() - tick)), flush=True)

    common = [
        n
        for n in cross_names
        if n.startswith("shock_")
        or n.startswith("ewz_minus_bova11_")
        or n
        in [
            "foreign_flow_" + s for s in ("1", "5", "month_reset", "methodology_change")
        ]
    ]
    new_rows = membership[1][selected] & ~membership[0][selected]
    for t, n in np.argwhere(new_rows):
        candidates = np.flatnonzero(membership[0][selected[t]])
        for field in common:
            j = cross_names.index(field)
            for k in range(3):
                a = cross_parent[k][t, candidates, j]
                assert np.all(a == a[0])
                cross_after[k][t, n, j] = a[0]
    # Existing dated TIMS ADR evidence is already in the accepted source table.
    adr = bound_json(cm["adr_identity"])
    cash = pl.read_parquet(
        cm["cash_identity"]["path"], columns=["source_trade_date", "ticker", "isin"]
    ).filter(
        pl.col("source_trade_date").is_between(days[0], days[-1])
        & pl.col("isin").is_in([isins[n] for n in columns])
    )
    cash.write_parquet(out / "selected_adr_cash_identity.parquet")
    bova_ret = np.full(len(days), np.nan)
    bova_ret[1:] = np.log(bova[start + 1 : stop] / bova[start : stop - 1])
    _, oil_mask, _, _ = returns[1]
    gap, gap_mask, _, _, flag = relative_return_panel(
        us,
        levels,
        cash,
        adr["pairs"],
        days,
        [isins[n] for n in columns],
        oil_raw_returns[:, columns],
        oil_mask[:, columns],
        bova_ret,
    )
    for j, event in enumerate(events):
        n = columns[j]
        for t in np.flatnonzero(dates[rows] >= np.datetime64(event["effective_date"])):
            if not membership[1][selected[t], n]:
                continue
            for field, value, valid in (
                ("adr_listed_flag", flag[selected[t], j], True),
                ("adr_return_gap_1", gap[selected[t], j], gap_mask[selected[t], j]),
            ):
                f = cross_names.index(field)
                cross_after[0][t, n, f] = value if valid else 0
                cross_after[1][t, n, f] = valid
                cross_after[2][t, n, f] = 1 if valid else -1
    flow_rows = []
    for rec in bound_json(cm["foreign_flow"])["results"]:
        if "observation" in rec:
            a = dict(rec["observation"])
            for key in ("reference_date", "publication_date"):
                a[key] = date.fromisoformat(a[key])
            flow_rows.append(a)
    fn, fv, fmask, fage, _ = decision_panel(flow_rows, sessions)
    for j, event in enumerate(events):
        n = columns[j]
        for t in np.flatnonzero(dates[rows] >= np.datetime64(event["effective_date"])):
            if not membership[1][selected[t], n]:
                continue
            for h in (1, 5):
                f, day = fn.index("foreign_flow_" + str(h)), rows[t]
                for suffix in ("log_volume_mean_20", "adr_listed_flag"):
                    field = cross_names.index(f"foreign_flow_{h}_times_{suffix}")
                    if suffix == "adr_listed_flag":
                        value, valid, age = flag[selected[t], j], True, 1
                    else:
                        g = names.index("log_traded_value_20")
                        value, valid, age = (a[t, n, g] for a in mag_after)
                    known = fmask[day, f] and valid
                    cross_after[0][t, n, field] = (
                        np.float32(fv[day, f] * value) if known else 0
                    )
                    cross_after[1][t, n, field] = known
                    cross_after[2][t, n, field] = (
                        max(fage[day, f], age) if known else -1
                    )
    finish("cross_market", cross_names, cross_parent, cross_after)
    np.savez_compressed(out / "deltas.npz", **deltas)
    report = dict(
        status="qualified_intermediates_pending_combined_consumer",
        plan=binding(out / "plan.json"),
        families=reports,
        deltas=binding(out / "deltas.npz"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_event_market"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
