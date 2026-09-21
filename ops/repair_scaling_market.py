"""Bounded magnitude and cross-market propagation for the added-period repairs."""

from datetime import date
import json
from pathlib import Path
import sys
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
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.foreign_flow import decision_panel
from brazil_rv.v2.hedge_beta import rolling_hedge_beta
from brazil_rv.v2.round5_derived import beta_source_age, identity_axes
from brazil_rv.v2.round5_magnitude import FEATURE_NAMES, magnitude_panel
from repair_scaling_context import (
    accepted_family,
    amended,
    corrected_basis,
    finish_family,
    setup,
)
from repair_scaling_inputs import PROJECT, record


def main(resume=False):
    tick = perf_counter()
    run, plan, root, m, start, first, stop, dr, records, dates, isins = setup()
    out = Path(bound_json(run["scaling_data_workspace"])["root"]) / "market"
    out.mkdir(exist_ok=resume)
    (out / ("executed_resume.py" if resume else "executed.py")).write_bytes(
        Path(__file__).read_bytes()
    )
    (out / ("executed_basis_resume.py" if resume else "executed_basis.py")).write_bytes(
        (PROJECT / "ops/repair_scaling_context.py").read_bytes()
    )
    starts = {}
    for event in plan["history"]:
        for key in ("isin", "successor_isin"):
            starts[event[key]] = event["effective_date"]
    for event in plan["scalars"]:
        starts[event["isin"]] = min(
            starts.get(event["isin"], "9999"), event["effective_date"]
        )
    columns = [isins.index(n) for n in starts]
    write_json_atomic(
        out / ("resume_plan.json" if resume else "plan.json"),
        dict(
            parent=plan["parent"],
            rows=[start, first, stop],
            daily=run["scaling_data_daily_qualification"],
            issuer=run["scaling_data_issuers"],
            starts=starts,
            contrast="Only evidenced scalar/history repairs and final daily support. Reuse all old raw magnitude/cross-market tiers. One set of six corrected120/60 regressions, original issuer shrinkage, oil versus ordinary source contracts, Float32 prices/volume, original shock/flow clocks. Historical old JSL remains public; later own logistics lookbacks are separate. No new ADR/loan/source alias or source census. Combined actual consumer qualification follows all families.",
            resume=resume,
            control_boundary="Older saved return window has no preceding row on its first date; compare shared return inputs from its second date. The longer new window legitimately supports that first-date return. Reuse completed magnitude outputs and saved ordinary inputs on resume.",
        ),
    )
    rows, selected = np.arange(first, stop), np.arange(first - start, stop - start)
    days, sessions = (
        dates[start:stop].astype(object).tolist(),
        dates.astype(object).tolist(),
    )
    membership = [
        np.load(root / m["arrays"]["active"]["path"], mmap_mode="r")[start:stop],
        amended(root, m, "active", start, stop, records),
    ]
    slow, ages = [
        amended(root, m, k, start, stop, records)
        for k in ("slow_valid", "slow_age_sessions")
    ]
    slow_names = m["feature_names"]["slow"]
    reports, deltas = {}, {}

    def finish(family, names, before, after):
        report, delta = finish_family(
            out,
            family,
            names,
            before,
            after,
            m,
            root,
            dates,
            rows,
            [a[selected] for a in membership],
        )
        reports[family] = report
        deltas.update(delta)

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
    if resume:
        names = m["feature_names"]["sidecar_magnitudes"]
        with np.load(out / "magnitudes_corrected.npz") as z:
            mag_after = [z[k].copy() for k in ("values", "valid", "ages")]
        reports["magnitudes"] = bound_json(binding(out / "magnitudes_report.json"))
        assert not any(c["mismatches"] for c in reports["magnitudes"]["checks"])
        with np.load(out / "magnitudes_deltas.npz") as z:
            deltas.update({k: z[k].copy() for k in z.files})
    else:
        names, mag_parent = accepted_family(run, m, dates, isins, rows, "magnitudes")
        mag_after = [a.copy() for a in mag_parent]
        wealth = {
            k: np.load(dr / f"wealth_{k}.npy")[:, columns]
            for k in ("open", "high", "low", "close", "valid")
        }
        volume, activity = [
            np.load(root / m["arrays"][k]["path"], mmap_mode="r")[start:stop].copy()
            for k in ("volume_brl", "activity_valid")
        ]
        admission = bound_json(run["scaling_data_identity"])
        for link in (
            pl.read_parquet(admission["history_mapping"]["path"])
            .sort("effective_index")
            .iter_rows(named=True)
        ):
            e = link["effective_index"] - start
            if e > 0:
                for a in (volume, activity):
                    a[:e, link["successor_index"]] = a[:e, link["predecessor_index"]]
        beta, known = rolling_hedge_beta(
            wealth["close"], wealth["valid"], bova[start:stop]
        )
        age = beta_source_age(wealth["close"], wealth["valid"], bova[start:stop], known)
        mag = magnitude_panel(
            **{"wealth_" + k: a for k, a in wealth.items()},
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
        for j, (isin, boundary) in enumerate(starts.items()):
            n = isins.index(isin)
            use = np.flatnonzero(dates[rows] >= np.datetime64(boundary))
            mask = mag[1][selected[use], j] & membership[1][selected[use], n, None]
            for k, a in enumerate(mag):
                v = (
                    mask
                    if k == 1
                    else np.where(
                        mask, a[selected[use], j], -1 if k == 2 else 0
                    ).astype(np.float32)
                )
                mag_after[k][np.ix_(use, [n], fields)] = v[:, None, :]
        finish("magnitudes", names, mag_parent, mag_after)

    cross_names, cross_parent = accepted_family(
        run, m, dates, isins, rows, "cross_market"
    )
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
    identity = bound_json(run["scaling_data_issuers"])["artifacts"]["identity"]
    sectors, issuers = identity_axes(
        pl.read_parquet(identity["path"]).filter(
            pl.col("date").is_between(days[0], days[-1])
        ),
        days,
        isins,
    )
    jsl, reopening = (
        isins.index("BRJSLGACNOR2"),
        int(np.searchsorted(days, date(2020, 11, 11))),
    )
    returns, basis_checks = [], []
    previous = bound_json(run["stage_c_event_market"])
    pr = Path(previous["plan"]["path"]).parent
    ps = bound_json(previous["plan"])["rows"][0]
    unaffected = np.array([n for n in range(len(isins)) if n not in columns])
    for kind, receipt in (("ordinary", ordinary), ("oil", cm["base_manifest"])):
        if resume and (out / f"{kind}_return_inputs.npz").exists():
            with np.load(out / f"{kind}_return_inputs.npz") as z:
                ret, mask = z["values"].copy(), z["valid"].copy()
                oldret, oldmask = (
                    z["public_jsl_values"].copy(),
                    z["public_jsl_valid"].copy(),
                )
            public_ret, public_mask = ret.copy(), mask.copy()
            public_ret[:reopening, jsl], public_mask[:reopening, jsl] = (
                oldret[:reopening],
                oldmask[:reopening],
            )
        else:
            arrays, old_jsl = corrected_basis(
                run, plan, receipt, root, m, start, stop, dr, records
            )
            ret, mask = exact_log_return(
                arrays["shareholder_wealth_close"],
                1,
                shareholder_wealth_valid=arrays["shareholder_wealth_valid"],
            )
            mask[:-1] &= arrays["slow_valid"][1:, :, slow_names.index("log_return_1")]
            mask[-1] = False
            oldret, oldmask = exact_log_return(
                old_jsl["shareholder_wealth_close"][:, None],
                1,
                shareholder_wealth_valid=old_jsl["shareholder_wealth_valid"][:, None],
            )
            oldmask[:-1, 0] &= old_jsl["slow_valid"][
                1:, slow_names.index("log_return_1")
            ]
            oldmask[-1] = False
            if kind == "oil":
                mask = comparison_return_mask(mask, arrays["action_has_action"])
                oldmask = comparison_return_mask(
                    oldmask, old_jsl["action_has_action"][:, None]
                )
            public_ret, public_mask = ret.copy(), mask.copy()
            public_ret[:reopening, jsl], public_mask[:reopening, jsl] = (
                oldret[:reopening, 0],
                oldmask[:reopening, 0],
            )
            np.savez_compressed(
                out / f"{kind}_return_inputs.npz",
                values=ret,
                valid=mask,
                public_jsl_values=oldret[:, 0],
                public_jsl_valid=oldmask[:, 0],
            )
        with np.load(pr / f"{kind}_return_inputs.npz") as z:
            for key, a in (("values", ret), ("valid", mask)):
                before, after = z[key][1:, unaffected], a[ps - start + 1 :, unaffected]
                eq = (before == after) | (np.isnan(before) & np.isnan(after))
                basis_checks.append(
                    dict(
                        kind=kind,
                        key=key,
                        cells=eq.size,
                        mismatches=int((~eq).sum()),
                        examples=np.argwhere(~eq)[:5].tolist(),
                    )
                )
        write_json_atomic(
            out / ("basis_checks_resumed.json" if resume else "basis_checks.json"),
            basis_checks,
        )
        assert not any(c["mismatches"] for c in basis_checks)
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
        beta, known, age = (
            np.zeros_like(ret, np.float32),
            np.zeros_like(mask),
            np.full_like(ret, -1, np.float32),
        )
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
            for dest, a in zip((beta, known, age), result, strict=True):
                dest[left:right] = a[left - offset :]
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
        ages3 = np.where(
            valid,
            np.stack(
                (
                    age,
                    np.maximum(age, shock_age[start:stop, 0, None]),
                    np.maximum(age, shock_age[start:stop, 1, None]),
                ),
                -1,
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
    for t, n in np.argwhere(membership[1][selected] & ~membership[0][selected]):
        candidates = np.flatnonzero(membership[0][selected[t]])
        for field in common:
            j = cross_names.index(field)
            for k in range(3):
                a = cross_parent[k][t, candidates, j]
                assert np.all(a == a[0])
                cross_after[k][t, n, j] = a[0]
    adr = bound_json(cm["adr_identity"])
    cash = pl.read_parquet(
        cm["cash_identity"]["path"], columns=["source_trade_date", "ticker", "isin"]
    ).filter(
        pl.col("source_trade_date").is_between(days[0], days[-1])
        & pl.col("isin").is_in(list(starts))
    )
    cash.write_parquet(out / "selected_adr_cash_identity.parquet")
    bova_ret = np.full(len(days), np.nan)
    bova_ret[1:] = np.log(bova[start + 1 : stop] / bova[start : stop - 1])
    gap, gap_mask, _, _, flag = relative_return_panel(
        us,
        levels,
        cash,
        adr["pairs"],
        days,
        list(starts),
        oil_raw_returns[:, columns],
        returns[1][1][:, columns],
        bova_ret,
    )
    flow_rows = []
    for rec in bound_json(cm["foreign_flow"])["results"]:
        if "observation" in rec:
            a = dict(rec["observation"])
            for k in ("reference_date", "publication_date"):
                a[k] = date.fromisoformat(a[k])
            flow_rows.append(a)
    fn, fv, fmask, fage, _ = decision_panel(flow_rows, sessions)
    for j, (isin, boundary) in enumerate(starts.items()):
        n = isins.index(isin)
        for t in np.flatnonzero(dates[rows] >= np.datetime64(boundary)):
            if not membership[1][selected[t], n]:
                continue
            for field, value, valid in (
                ("adr_listed_flag", flag[selected[t], j], True),
                ("adr_return_gap_1", gap[selected[t], j], gap_mask[selected[t], j]),
            ):
                f = cross_names.index(field)
                (
                    cross_after[0][t, n, f],
                    cross_after[1][t, n, f],
                    cross_after[2][t, n, f],
                ) = value if valid else 0, valid, 1 if valid else -1
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
    result = dict(
        status="qualified_intermediates_pending_combined_consumer",
        plan=binding(out / "plan.json"),
        families=reports,
        basis_checks=binding(
            out / ("basis_checks_resumed.json" if resume else "basis_checks.json")
        ),
        deltas=binding(out / "deltas.npz"),
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_market", out / "manifest.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                effects={k: v["effects"] for k, v in reports.items()},
                seconds=result["seconds"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main("resume" in sys.argv)
