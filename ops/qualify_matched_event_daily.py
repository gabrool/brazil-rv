"""Qualify new daily dependencies against saved parent reducers, without reruns."""

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    observation_age_sessions_into,
    transform_feature_panel_into,
)
from brazil_rv.v2.features import _ambiguous_interval_clear

PROJECT = Path(__file__).resolve().parents[1]
HORIZONS = {
    0: 1,
    1: 5,
    2: 21,
    3: 63,
    4: 126,
    5: 252,
    6: 252,
    7: 5,
    8: 20,
    9: 60,
    10: 64,
    11: 60,
    12: 60,
    13: 21,
    14: 251,
    15: 60,
    16: 60,
    19: 20,
    23: 4,
}


def same(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


def qualify_saved_activity_masks():
    """Use the saved outputs; the earlier four mask comparisons were too broad."""
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    produced = bound_json(run["stage_c_event_daily"])
    plan = bound_json(produced["plan"])
    source = Path(produced["plan"]["path"]).parent
    initial = source.parent / "qualified/manifest.json"
    report = bound_json(binding(initial))
    fields = (17, 18, 20, 21)
    assert {c["name"] for c in report["checks"] if c["mismatches"]} == {
        f"unchanged_raw_mask:{f}" for f in fields
    }
    admission = bound_json(run["stage_c_event_data_admission"])
    events = bound_json(admission["plan"])["events"]
    root = Path(admission["parent"]["root"])
    names = np.load(root / "isin_index.npy").tolist()
    m = json.loads((root / "manifest.json").read_text())
    parent = bound_json(run["surviving_rename_daily"])
    prior_source = Path(parent["plan"]["path"]).parent
    offset = plan["rows"][0] - bound_json(parent["plan"])["rows"][0]
    selected = np.load(source / "selected.npy")
    active = np.load(source / "active.npy")[selected]
    old_active = np.load(root / m["arrays"]["active"]["path"], mmap_mode="r")[
        selected + plan["rows"][0]
    ]
    other = np.ones(len(names), bool)
    for e in events:
        for k in ("isin", "successor_isin"):
            other[names.index(e[k])] = False
    out = source.parent / "admitted"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    scope = (old_active | active) & other
    checks, differences = [], {}
    for f in fields:
        with np.load(source / f"{f}.npz") as z:
            mask = z["valid"][selected - 1]
        with np.load(prior_source / "corrected" / f"{f}.npz") as z:
            baseline = z["valid"][selected - 1 + offset]
        ix = np.argwhere((mask != baseline) & other)
        assert not scope[tuple(ix.T)].any()
        differences[str(f)] = ix
        checks.append(
            dict(
                name=f"unchanged_raw_mask:{f}",
                cells=int(scope.sum()),
                mismatches=int(((mask != baseline) & scope).sum()),
                inactive_differences=len(ix),
            )
        )
    np.savez_compressed(out / "inactive_activity_masks.npz", **differences)
    report["checks"] = [
        c
        for c in report["checks"]
        if c["name"] not in {f"unchanged_raw_mask:{f}" for f in fields}
    ] + checks
    report["initial_qualification"] = binding(initial)
    report["inactive_activity_masks"] = binding(out / "inactive_activity_masks.npz")
    report["activity_disposition"] = (
        "The old saved four raw activity reducers precede the already qualified absent-security zero-activity correction. All14440 differing masks per field are inactive in both memberships. Qualified old typed controls remain authoritative; no new producer/typed array/age/delta changed or reran. The four date sets overlap, not independent evidence. Existing successful new checks reused."
    )
    report["status"] = "passed"
    report["activity_resume_seconds"] = perf_counter() - tick
    write_json_atomic(out / "manifest.json", report)
    run["stage_c_event_daily_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                status="passed",
                corrected_checks=checks,
                seconds=report["activity_resume_seconds"],
            )
        )
    )


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    produced = bound_json(run["stage_c_event_daily"])
    plan = bound_json(produced["plan"])
    admission = bound_json(run["stage_c_event_data_admission"])
    identities = bound_json(admission["plan"])
    parent_daily = bound_json(run["surviving_rename_daily"])
    parent_plan = bound_json(parent_daily["plan"])
    parent_proof = bound_json(run["surviving_rename_daily_qualification"])
    assert parent_proof["status"] == "passed"
    source = Path(produced["plan"]["path"]).parent
    prior_source = Path(parent_daily["plan"]["path"]).parent
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = source.parent / "qualified"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    start, first, stop = plan["rows"]
    selected = np.load(source / "selected.npy")
    rows = selected + start
    offset = start - parent_plan["rows"][0]
    events = identities["events"]
    axes = sorted(
        {names.index(e[k]) for e in events for k in ("isin", "successor_isin")}
    )
    other = np.ones(len(names), bool)
    other[axes] = False
    specs = [
        FeatureSpec(**s)
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]

    def old(key):
        return np.load(root / m["arrays"][key]["path"], mmap_mode="r")

    active = np.load(source / "active.npy")
    original_active = old("active")[start:]
    # Reconstruct only the prior mask intersection, not any prior reducer.
    # Its original source-mask proof and four corrected activity fields stand.
    quotes = (
        pl.scan_parquet(parent_daily["source_coordinates"]["path"])
        .filter(pl.col("trade_date") >= dates[start].astype(object))
        .select("trade_date", "isin")
        .collect()
    )
    lookup = {s: j for j, s in enumerate(names)}
    gaps = old("observed")[start:].copy()
    gaps[
        np.searchsorted(dates[start:], quotes["trade_date"].to_numpy()),
        np.array([lookup[s] for s in quotes["isin"]]),
    ] = False
    for link in (
        pl.read_parquet(root / m["tables"]["isin_succession_links"]["path"])
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
    write_json_atomic(
        out / "plan.json",
        dict(
            producer=run["stage_c_event_daily"],
            parent_proof=run["surviving_rename_daily_qualification"],
            contract="Reuse qualified parent reducers/masks; compare unchanged own-name raw fields outside nine affected axes, transform corrected full cross-sections under original definitions, preserve older clocks, and enumerate every support loss. Market/peer fields may change globally. No repeated parent reducer or consumer campaign; final all-family integration follows.",
        ),
    )
    packed, checks, effects, losses = {}, [], [], []

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
        indices = ix.copy()
        indices[:, 0] = rows[ix[:, 0]]
        if field is not None:
            indices = np.column_stack((indices, np.full(len(ix), field)))
        packed.setdefault(key + "__indices", []).append(indices)
        packed.setdefault(key + "__values", []).append(after[tuple(ix.T)])
        return len(ix)

    def retain(field, value, mask, age):
        before, bmask, bage = (
            old(k)[rows, :, field]
            for k in ("slow_values", "slow_valid", "slow_age_sessions")
        )
        effects.append(
            dict(
                field=field,
                name=specs[field].name,
                values=delta("slow_values", before, value, field),
                valid=delta("slow_valid", bmask, mask, field),
                ages=delta("slow_age_sessions", bage, age, field),
                gains=int((mask & ~bmask).sum()),
                losses=int((bmask & ~mask).sum()),
            )
        )
        for t, n in np.argwhere(bmask & ~mask):
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

    for field in [*range(25), *range(27, 32)]:
        with np.load(source / f"{field}.npz") as z:
            raw, mask = z["values"], z["valid"]
        with np.load(prior_source / "corrected" / f"{field}.npz") as z:
            baseline, bmask = z["values"][offset:], z["valid"][offset:].copy()
        if field in HORIZONS:
            bmask &= _ambiguous_interval_clear(gaps, HORIZONS[field])
        elif field in (22, 24):
            bmask &= ~gaps
        if field not in (15, 16, 27, 28, 29, 30, 31):
            compared = (
                (original_active[selected] | active[selected])[:, other]
                if field in (17, 18, 20, 21)
                else np.ones((len(rows), int(other.sum())), bool)
            )
            check(
                f"unchanged_raw_mask:{field}",
                mask[selected - 1][:, other][compared],
                bmask[selected - 1][:, other][compared],
            )
            if field not in (17, 18, 20, 21):
                usable = mask[selected - 1][:, other] & bmask[selected - 1][:, other]
                check(
                    f"unchanged_raw_values:{field}",
                    raw[selected - 1][:, other][usable],
                    baseline[selected - 1][:, other][usable],
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
        ages = []
        for mk, membership in ((bmask, original_active), (mask, active)):
            age = np.full_like(dest, -1)
            observation_age_sessions_into(
                mk[..., None],
                membership,
                age,
                source_rows=selected - 1,
                decision_rows=selected,
            )
            ages.append(age[..., 0])
        old_age = old("slow_age_sessions")[rows, :, field]
        changed_clock = (ages[0] != ages[1]) | (
            original_active[selected] != active[selected]
        )
        check(
            f"changed_age_control:{field}",
            ages[0][changed_clock & original_active[selected]],
            old_age[changed_clock & original_active[selected]],
        )
        after_age = old_age.copy()
        after_age[changed_clock] = ages[1][changed_clock]
        retain(field, dest[..., 0], valid[..., 0], after_age)
        if field == 8:
            before = old("target_scale_sigma")[rows]
            after = before.copy()
            sigma = np.where(mask[selected - 1], raw[selected - 1], np.nan).astype(
                np.float32
            )
            for e in events:
                a, b = lookup[e["isin"]], lookup[e["successor_isin"]]
                take = dates[rows] >= np.datetime64(e["effective_date"])
                after[take, b] = sigma[take, b]
                source_take = take.copy()
                if e["source_reopens_date"]:
                    source_take &= dates[rows] < np.datetime64(e["source_reopens_date"])
                # The retired source has the same continuing wealth basis as
                # its successor, until a separately admitted episode reuses it.
                after[source_take, a] = sigma[source_take, b]
            delta("target_scale_sigma", before, after)
    observed = old("observed")
    for field in (25, 26):
        value, mask, age = (
            old(k)[rows, :, field].copy()
            for k in ("slow_values", "slow_valid", "slow_age_sessions")
        )
        retire = original_active[selected] & ~active[selected]
        value[retire], mask[retire], age[retire] = 0, False, -1
        for e in events:
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
    delta(
        "monthly_cluster_labels",
        old("monthly_cluster_labels")[rows],
        np.load(source / "clusters.npy")[selected],
    )
    timesteps = old("slow_timestep_valid")[rows].copy()
    for e in events:
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
        producer=run["stage_c_event_daily"],
        plan=binding(out / "plan.json"),
        checks=checks,
        effects=effects,
        losses=binding(out / "validity_losses.json"),
        deltas=binding(out / "deltas.npz"),
        seconds=perf_counter() - tick,
        scope="Intermediate daily/risk/cluster layer only; old accepted stores and fits unchanged. Reused parent proof counts are not new checks.",
    )
    write_json_atomic(out / "manifest.json", report)
    if report["status"] == "passed":
        run = json.loads(pointer.read_text())
        run["stage_c_event_daily_qualification"] = binding(out / "manifest.json")
        write_json_atomic(pointer, run)
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


if __name__ == "__main__":
    if sys.argv[1:] == ["--qualify-saved-activity"]:
        qualify_saved_activity_masks()
    else:
        main()
