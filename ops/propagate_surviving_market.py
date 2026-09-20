"""Finish magnitude and cross-market dependencies using saved identity/daily data."""

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_exposures as exp
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.bova11 import load_bova11_series
from brazil_rv.v2.cross_market_returns import admit_brent, comparison_return_mask
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.foreign_flow import decision_panel
from brazil_rv.v2.hedge_beta import rolling_hedge_beta
from brazil_rv.v2.round5_derived import beta_source_age, identity_axes
from brazil_rv.v2.round5_magnitude import FEATURE_NAMES, magnitude_panel
from brazil_rv.v2.round5_store import align_family

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    resume = parser.parse_args().resume
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["surviving_rename_admission"])
    daily = bound_json(run["surviving_rename_daily"])
    qualified = bound_json(run["surviving_rename_daily_qualification"])
    source = Path(admission["plan"]["path"]).parent
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = source / "market"
    out.mkdir(exist_ok=resume)
    assert not (out / "manifest.json").exists(), "Reuse completed results"
    (out / ("executed_resume.py" if resume else "executed.py")).write_bytes(
        Path(__file__).read_bytes()
    )
    plan = bound_json(daily["plan"])
    start, first, stop = plan["rows"]
    dates = np.load(root / "date_index.npy")
    isins = np.load(root / "isin_index.npy").tolist()
    days = dates[start:stop].astype(object).tolist()
    sessions = dates.astype(object).tolist()
    selected = np.arange(first - start, stop - start)
    rows = np.arange(first, stop)
    if not resume:
        write_json_atomic(
            out / "plan.json",
            dict(
                parent=admission["parent"],
                daily=run["surviving_rename_daily_qualification"],
                issuer=run["surviving_rename_issuers"],
                registration=binding(
                    PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
                ),
                closeout=run["stage_a_closeout_plan"],
                rows=[start, first, stop],
                contrast="Only magnitude and cross-market dependencies of new surviving identities, eligibility and saved wealth/risk. Reuse old accepted family rows and Natura corrections; preserve separate oil/ordinary wealth contracts, original120/60regressions, issuer shrinkage, raw source flow/shock clocks, originalFloat32 panel casting and all933 axes. No repeat common/sector/financial/lending/auxiliary reducers.",
                verification="Old-coordinate raw and actual typed controls before admission, every changed/new family observation retained, future-source prefix and actual full60 consumer checks. No neural forward or old fit.",
            ),
        )
    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    fm = bound_json(
        dict(
            path=str(Path(foundation["root"]) / "manifest.json"),
            sha256=foundation["manifest_sha256"],
        )
    )
    families = source_families(fm)
    old_history = bound_json(run["rename_history_propagation"])["history_basis"]
    old_links = pl.read_parquet(root / "slow_history_links.parquet").to_dicts()
    links = (
        pl.read_parquet(admission["history_mapping"]["path"])
        .sort("effective_index")
        .to_dicts()
    )
    old_identity = bound_json(run["rename_issuer_propagation"])["artifacts"]["identity"]
    identity = bound_json(run["surviving_rename_issuers"])["artifacts"]["identity"]
    identities = [
        identity_axes(
            pl.read_parquet(rec["path"]).filter(
                pl.col("date").is_between(days[0], days[-1])
            ),
            days,
            isins,
        )
        for rec in [old_identity, identity]
    ]
    membership = [
        np.load(source / "daily" / label / "active.npy", mmap_mode="r")
        for label in ["control", "corrected"]
    ]
    slow_names = m["feature_names"]["slow"]

    def old(key):
        return np.load(root / (key + ".npy"), mmap_mode="r")[start:stop]

    def after(key):
        a = old(key).copy()
        with np.load(qualified["deltas"]["path"]) as z:
            ix = z[key + "__indices"].copy()
            ix[:, 0] -= start
            a[tuple(ix.T)] = z[key + "__values"]
        return a

    slow = [old("slow_valid"), after("slow_valid")]
    ages = [old("slow_age_sessions"), after("slow_age_sessions")]
    prior = bound_json(run["activity_magnitude_propagation"])["artifacts"]
    natura = bound_json(run["natura_auxiliary_qualification"])
    natura_producer = bound_json(natura["producer"])
    nr = np.load(natura_producer["rows"]["path"])
    reports, deltas = {}, {}

    def parent(family):
        names = m["feature_names"]["sidecar_" + family]
        frame = pl.read_parquet(prior[family]["path"]).filter(
            pl.col("date").is_between(days[selected[0]], days[-1])
        )
        result = list(
            align_family(
                frame, dates[rows].astype(object).tolist(), tuple(isins), tuple(names)
            )
        )
        rec = natura_producer["families"][family]
        with np.load(
            Path(run["natura_auxiliary_qualification"]["path"]).parent
            / (family + "_rows.npz")
        ) as z:
            use = (nr >= first) & (nr < stop)
            for j, k in enumerate(["values", "valid", "ages"]):
                result[j][nr[use] - first] = z[k][use]
        return names, result, rec

    def finish(family, names, before, changed):
        # Family panels are stored Float32 before cross-sectional transforms.
        specs = [
            FeatureSpec(**s)
            for s in m["metadata"]["feature_schema"]["specifications"]
            if s["family"] == "sidecar_" + family
        ]
        checks = []
        typed = []
        for label, arrays, active in [
            ("control", before, membership[0][selected]),
            ("corrected", changed, membership[1][selected]),
        ]:
            raw, valid, age = arrays
            values = np.zeros(raw.shape, np.float32)
            mask = np.zeros(raw.shape, bool)
            transform_feature_panel_into(raw, valid, active, specs, values, mask)
            age = np.where(active[..., None], age, -1).astype(np.float32)
            typed.append((values, mask, age))
            np.savez_compressed(
                out / (family + "_" + label + ".npz"),
                rows=rows,
                values=raw,
                valid=valid,
                ages=arrays[2],
            )
        effects = {}
        for j, suffix in enumerate(["values", "valid", "age_sessions"]):
            key = "sidecar_" + family + "_" + suffix
            accepted = old(key)[selected]
            equal = (accepted == typed[0][j]) | (
                np.isnan(accepted) & np.isnan(typed[0][j])
            )
            ix = np.argwhere(~equal)
            checks.append(
                dict(
                    key=key,
                    cells=accepted.size,
                    mismatches=len(ix),
                    examples=ix[:8].tolist(),
                )
            )
            a = typed[1][j]
            ix = np.argwhere(~((a == accepted) | (np.isnan(a) & np.isnan(accepted))))
            deltas[key + "__indices"] = np.column_stack((rows[ix[:, 0]], ix[:, 1:]))
            deltas[key + "__values"] = a[tuple(ix.T)]
            effects[suffix] = len(ix)
        effects.update(
            gains=int((typed[1][1] & ~typed[0][1]).sum()),
            losses=int((typed[0][1] & ~typed[1][1]).sum()),
            shared_changed=int(
                (typed[1][1] & typed[0][1] & (typed[1][0] != typed[0][0])).sum()
            ),
        )
        reports[family] = dict(checks=checks, effects=effects, names=names)
        write_json_atomic(out / (family + "_report.json"), reports[family])
        print(
            json.dumps(
                dict(family=family, **reports[family], seconds=perf_counter() - tick)
            ),
            flush=True,
        )
        assert not any(c["mismatches"] for c in checks)

    names, mag_parent, _ = parent("magnitudes")
    mag_after = [a.copy() for a in mag_parent]
    mag_path = out / "magnitude_reducers.npz"
    columns = [isins.index(n) for n in ["BRALSOACNOR5", "BRALOSACNOR5", "BRAZZAACNOR9"]]
    if not mag_path.exists():
        mm = bound_json(families["magnitudes"]["source_manifest"])["sources"]["inputs"]
        bova = load_bova11_series(
            Path(mm["bova_manifest"]["path"]).parent,
            expected_manifest_sha256=mm["bova_manifest"]["sha256"],
            canonical_dates=sessions,
        ).close_by_session[start:stop]
        raw = {
            k: np.load(old_history["shareholder_wealth_" + k]["path"], mmap_mode="r")[
                start:stop
            ][:, columns].copy()
            for k in ["open", "high", "low", "close", "valid"]
        }
        corrected = {
            k: np.load(
                source / "daily/corrected" / ("wealth_" + k + ".npy"), mmap_mode="r"
            )[:, columns]
            for k in raw
        }
        volume = old("volume_brl").copy()
        activity = old("activity_valid").copy()

        def route(array, rules):
            for link in rules:
                boundary = link["effective_index"] - start
                if boundary > 0:
                    array[:boundary, link["successor_index"]] = array[
                        :boundary, link["predecessor_index"]
                    ]
            return array

        controls = []
        for i, wealth in enumerate([raw, corrected]):
            vol = route(volume.copy(), old_links if i == 0 else links)
            act = route(activity.copy(), old_links if i == 0 else links)
            beta, known = rolling_hedge_beta(wealth["close"], wealth["valid"], bova)
            age = beta_source_age(wealth["close"], wealth["valid"], bova, known)
            controls.append(
                magnitude_panel(
                    **{"wealth_" + k: v for k, v in wealth.items()},
                    volume_brl=np.ascontiguousarray(vol[:, columns]),
                    activity_valid=act[:, columns],
                    daily_feature_valid=slow[i][:, columns],
                    slow_age_sessions=ages[i][:, columns],
                    slow_feature_names=tuple(slow_names),
                    economic_beta=beta,
                    economic_beta_valid=known,
                    economic_beta_age_sessions=age,
                )
            )
        np.savez_compressed(
            mag_path,
            **{
                f"{label}_{k}": a
                for label, result in zip(
                    ["control", "corrected"], controls, strict=True
                )
                for k, a in zip(["values", "valid", "age"], result, strict=True)
            },
        )
    with np.load(mag_path) as z:
        raw_checks = []
        for j, k in enumerate(["values", "valid", "age"]):
            fields = [names.index(n) for n in FEATURE_NAMES]
            ix = np.ix_(np.arange(len(rows)), columns, fields)
            for scenario, label in enumerate(["control", "corrected"]):
                valid = (
                    z[label + "_valid"][selected]
                    & membership[scenario][selected][:, columns, None]
                )
                a = z[label + "_" + k][selected]
                a = (
                    valid
                    if j == 1
                    else np.where(valid, a, 0 if j == 0 else -1).astype(np.float32)
                )
                if scenario == 0:
                    expected = np.where(
                        membership[0][selected][:, columns, None],
                        mag_parent[j][ix],
                        -1 if j == 2 else 0,
                    )
                    raw_checks.append(
                        dict(
                            kind=k,
                            cells=a.size,
                            mismatches=int((a != expected).sum()),
                        )
                    )
                else:
                    mag_after[j][ix] = a
        write_json_atomic(out / "magnitude_controls.json", raw_checks)
        assert not any(c["mismatches"] for c in raw_checks), raw_checks
    finish("magnitudes", names, mag_parent, mag_after)

    cross_names, cross_parent, _ = parent("cross_market")
    cross_after = [a.copy() for a in cross_parent]
    cm = bound_json(families["cross_market"]["source_manifest"])["sources"]
    sb = bound_json(families["sector"]["source_manifest"])["sources"]["inputs"][
        "base_manifest"
    ]
    original_cross = bound_json(
        binding(Path(cm["old_cross_market"]["path"]).parent / "manifest.json")
    )
    shocks = pl.read_parquet(original_cross["shocks"]["path"])
    cdi = pl.read_parquet(original_cross["cdi"]["path"])["daily_cdi_rate"].to_numpy()[
        start:stop
    ]
    levels = pl.read_parquet(cm["market_levels"]["path"])
    oil = exp.market_shocks(
        admit_brent(levels.filter(pl.col("series") == "brent_spot"), sessions),
        pl.read_parquet(cm["us_returns"]["path"]).head(0),
    )
    shock_axes = exp.shock_axes(shocks, sessions)
    shock_axes["oil"] = exp.shock_axes(oil, sessions)["oil"]
    returns = []
    for kind, record in [("ordinary", sb), ("oil", cm["base_manifest"])]:
        sm = bound_json(record)
        sr = Path(record["path"]).parent

        def source_array(k):
            return np.load(sr / sm["arrays"][k]["path"], mmap_mode="r")[
                start:stop
            ].copy()

        wealth, seen, support, actions = [
            source_array(k)
            for k in [
                "shareholder_wealth_close",
                "shareholder_wealth_valid",
                "slow_valid",
                "action_has_action",
            ]
        ]
        for link in old_links:
            a, b, e = (
                link["predecessor_index"],
                link["successor_index"],
                link["effective_index"] - start,
            )
            wealth[:, b] = np.load(
                old_history["shareholder_wealth_close"]["path"], mmap_mode="r"
            )[start:stop, b]
            seen[:, b] = np.load(
                old_history["shareholder_wealth_valid"]["path"], mmap_mode="r"
            )[start:stop, b]
            support[:e, b] = support[:e, a]
            support[e:, b] = slow[0][e:, b]
            actions[:e, b] = actions[:e, a]
        n = isins.index("BRNATUACNOR6")
        wealth[:, n] = old("shareholder_wealth_close")[:, n]
        seen[:, n] = old("shareholder_wealth_valid")[:, n]
        pair = []
        for i in range(2):
            if i:
                affected = {isins.index("BRSSBRACNOR1"), isins.index("BRARZZACNOR3")}
                for link in links:
                    a, b, e = (
                        link["predecessor_index"],
                        link["successor_index"],
                        link["effective_index"] - start,
                    )
                    if a not in affected:
                        continue
                    affected.add(b)
                    wealth[:, b] = np.load(
                        source / "daily/corrected/wealth_close.npy", mmap_mode="r"
                    )[:, b]
                    seen[:, b] = np.load(
                        source / "daily/corrected/wealth_valid.npy", mmap_mode="r"
                    )[:, b]
                    support[:e, b] = support[:e, a]
                    support[e:, b] = slow[1][e:, b]
                    actions[:e, b] = actions[:e, a]
            ret, known = exact_log_return(wealth, 1, shareholder_wealth_valid=seen)
            known[:-1] &= support[1:, :, slow_names.index("log_return_1")]
            known[-1] = False
            if kind == "oil":
                known = comparison_return_mask(known, actions)
            pair.append((ret - np.log1p(cdi)[:, None], known))
        returns.append(pair)
    raw_checks = []
    for name, factor in exp.EXPOSURES.items():
        current, shock_age, historical, public = shock_axes[factor]
        fields = [
            cross_names.index(k)
            for k in [
                f"exposure_{name}",
                f"exposure_{name}_times_shock_1",
                f"exposure_{name}_times_shock_5",
            ]
        ]
        for i, label in enumerate(["control", "corrected"]):
            path = out / (name + "_" + label + "_regression.npz")
            if path.exists():
                with np.load(path) as z:
                    beta, known, age = [z[k] for k in ["beta", "known", "age"]]
            else:
                ret, mask = returns[name == "oil"][i]
                beta, known, age = exp.exposure_panel(
                    ret,
                    mask,
                    historical[start:stop],
                    public[start:stop] - start,
                    *identities[i],
                )
                np.savez_compressed(path, beta=beta, known=known, age=age)
            values = np.stack(
                [beta, *[beta * current[start:stop, j, None] for j in range(2)]], -1
            ).astype(np.float32)
            valid = np.stack(
                [
                    known,
                    *[
                        known & np.isfinite(current[start:stop, j, None])
                        for j in range(2)
                    ],
                ],
                -1,
            )
            age = np.stack(
                [
                    age,
                    *[
                        np.maximum(age, shock_age[start:stop, j, None])
                        for j in range(2)
                    ],
                ],
                -1,
            ).astype(np.float32)
            valid = valid[selected] & membership[i][selected, :, None]
            result = [
                np.where(valid, values[selected], 0),
                valid,
                np.where(valid, age[selected], -1),
            ]
            for j, a in enumerate(result):
                if i == 0:
                    expected = np.where(
                        membership[0][selected, :, None],
                        cross_parent[j][:, :, fields],
                        -1 if j == 2 else 0,
                    )
                    raw_checks.append(
                        dict(
                            factor=name,
                            kind=j,
                            cells=a.size,
                            mismatches=int((a != expected).sum()),
                        )
                    )
                else:
                    cross_after[j][:, :, fields] = a
        write_json_atomic(out / "cross_controls.json", raw_checks)
        print(
            json.dumps(
                dict(
                    factor=name, controls=raw_checks[-3:], seconds=perf_counter() - tick
                )
            ),
            flush=True,
        )
    assert not any(c["mismatches"] for c in raw_checks), raw_checks
    # New eligible rows receive the already published common market coordinates.
    common = [
        n
        for n in cross_names
        if n.startswith("shock_")
        or n.startswith("ewz_minus_bova11_")
        or n
        in [
            "foreign_flow_" + s for s in ["1", "5", "month_reset", "methodology_change"]
        ]
    ]
    new_rows = membership[1][selected] & ~membership[0][selected]
    adr = bound_json(cm["adr_identity"])
    assert not any(isins[n] in json.dumps(adr["pairs"]) for n in columns)
    for t, n in np.argwhere(new_rows):
        candidates = np.flatnonzero(membership[0][selected[t]])
        for field in common:
            j = cross_names.index(field)
            for k in range(3):
                source_values = cross_parent[k][t, candidates, j]
                assert np.all(source_values == source_values[0])
                cross_after[k][t, n, j] = source_values[0]
        j = cross_names.index("adr_listed_flag")
        cross_after[0][t, n, j] = 0
        cross_after[1][t, n, j] = True
        cross_after[2][t, n, j] = 1
    flow_rows = []
    for rec in bound_json(cm["foreign_flow"])["results"]:
        if "observation" in rec:
            a = dict(rec["observation"])
            for key in ["reference_date", "publication_date"]:
                a[key] = date.fromisoformat(a[key])
            flow_rows.append(a)
    fn, fv, fmask, fage, _ = decision_panel(flow_rows, sessions)
    for t, n in np.argwhere(new_rows):
        for h in [1, 5]:
            f = fn.index("foreign_flow_" + str(h))
            day = rows[t]
            for suffix in ["log_volume_mean_20", "adr_listed_flag"]:
                j = cross_names.index(f"foreign_flow_{h}_times_{suffix}")
                g = names.index("log_traded_value_20")
                known = bool(
                    fmask[day, f]
                    and (suffix == "adr_listed_flag" or mag_after[1][t, n, g])
                )
                value = 0 if suffix == "adr_listed_flag" else mag_after[0][t, n, g]
                age = 1 if suffix == "adr_listed_flag" else mag_after[2][t, n, g]
                cross_after[0][t, n, j] = np.float32(fv[day, f] * value) if known else 0
                cross_after[1][t, n, j] = known
                cross_after[2][t, n, j] = max(fage[day, f], age) if known else -1
    finish("cross_market", cross_names, cross_parent, cross_after)
    np.savez_compressed(out / "deltas.npz", **deltas)
    report = dict(
        status="qualified_old_coordinates_pending_consumer",
        plan=binding(out / "plan.json"),
        parent=admission["parent"],
        families=reports,
        deltas=binding(out / "deltas.npz"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_market"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
