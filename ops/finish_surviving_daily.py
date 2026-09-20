"""Resolve four original-activity controls, reusing all other saved proofs."""

import json
from pathlib import Path
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
from brazil_rv.v2.features import _rolling_stat

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    produced = bound_json(run["surviving_rename_daily"])
    previous = bound_json(run["surviving_rename_daily_qualification"])
    affected = (17, 18, 20, 21)
    assert {c["name"] for c in previous["checks"] if c["mismatches"]} == {
        f"values:{f}" for f in affected
    }
    source = Path(produced["plan"]["path"]).parent
    prior = Path(run["surviving_rename_daily_qualification"]["path"]).parent
    out = source / "admitted"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    plan = bound_json(produced["plan"])
    root = Path(plan["parent"]["root"])
    m = bound_json(
        dict(path=str(root / "manifest.json"), sha256=plan["parent"]["manifest_sha256"])
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    start, first, stop = plan["rows"]
    selected = np.load(source / "selected.npy")
    rows = start + selected

    def old(key):
        return np.load(root / m["arrays"][key]["path"], mmap_mode="r")

    coordinates = pl.read_parquet(produced["source_coordinates"]["path"])
    present = np.zeros((stop - start, len(isins)), bool)
    ti = np.searchsorted(dates[start:], coordinates["trade_date"].to_numpy())
    lookup = {s: j for j, s in enumerate(isins)}
    ni = np.array([lookup[s] for s in coordinates["isin"]])
    present[ti, ni] = True
    # Missing original normalized security rows imply zero activity only under
    # the already certified complete-session/started-security activity mask.
    volume = old("volume_brl")[start:].astype(np.float64)
    trades = old("trade_count")[start:].astype(np.float64)
    volume[ti, ni] = coordinates["volume_brl"].to_numpy()
    trades[ti, ni] = coordinates["trades"].to_numpy()
    activity = old("activity_valid")[start:]
    gaps = activity & ~present
    changes = gaps & ((volume != 0) | (trades != 0))
    assert old("source_session_complete")[start:][gaps].all()
    ix = np.argwhere(changes)
    write_json_atomic(
        out / "activity_resolution.json",
        {
            "previous": run["surviving_rename_daily_qualification"],
            "coordinates": produced["source_coordinates"],
            "rows": [
                dict(
                    date=str(dates[start + t]),
                    isin=isins[n],
                    accepted_volume=float(volume[t, n]),
                    accepted_trades=float(trades[t, n]),
                )
                for t, n in ix
            ],
            "disposition": "Reconstruct the existing original-daily feature contract with certified complete-session zero activity for absent normalized security rows. This does not delete or rewrite accepted later raw-price/activity observations, or assert a new source defect. Changed aliases never admit those rows as newly observed. Only four saved activity reducers require replacement; all other qualified controls reused.",
            "executed_producer": binding(source / "executed.py"),
            "qualified_producer": binding(PROJECT / "ops/propagate_surviving_daily.py"),
        },
    )
    volume[gaps] = 0
    trades[gaps] = 0
    admission = bound_json(run["surviving_rename_admission"])
    tables = [
        pl.read_parquet(root / m["tables"]["isin_succession_links"]["path"]),
        pl.read_parquet(admission["links"]["path"]),
    ]
    specs = [
        FeatureSpec(**s)
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    checks = [
        c
        for c in previous["checks"]
        if c["name"]
        not in {
            f"{kind}:{f}"
            for f in affected
            for kind in ("values", "valid", "age_control")
        }
    ]
    effects = [e for e in previous["effects"] if e["field"] not in affected]
    with np.load(previous["deltas"]["path"]) as z:
        packed = {k: z[k].copy() for k in z.files}
    for key in ("slow_values", "slow_valid", "slow_age_sessions"):
        keep = ~np.isin(packed[key + "__indices"][:, -1], affected)
        for suffix in ("__indices", "__values"):
            packed[key + suffix] = packed[key + suffix][keep]
    outputs = {f: [] for f in affected}
    for label, links in zip(("control", "corrected"), tables, strict=True):
        v, n = volume.copy(), trades.copy()
        act = activity.copy()
        for link in links.iter_rows(named=True):
            a, b = (lookup[link[k]] for k in ("predecessor_isin", "successor_isin"))
            t = (
                int(np.searchsorted(dates, np.datetime64(link["effective_date"])))
                - start
            )
            for x in (v, n, act):
                x[:t, b] = x[:t, a]
        mean, known = _rolling_stat(np.where(act, v, np.nan), 20, "mean", minimum=20)
        std, _ = _rolling_stat(np.where(act, v, np.nan), 20, "std", minimum=20)
        tm, tk = _rolling_stat(np.where(act, n, np.nan), 20, "mean", minimum=20)
        ts, _ = _rolling_stat(np.where(act, n, np.nan), 20, "std", minimum=20)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = {
                17: (np.log(mean), known & (mean > 0)),
                18: ((v - mean) / std, act & known & (std > 0)),
                20: ((n - tm) / ts, act & tk & (ts > 0)),
                21: (v / mean, act & known & (mean > 0)),
            }
        active = np.load(source / label / "active.npy")
        for f, (values, mask) in raw.items():
            mask &= np.isfinite(values)
            dest = np.zeros((len(rows), len(isins), 1), np.float32)
            valid = np.zeros_like(dest, bool)
            age = np.full_like(dest, -1)
            transform_feature_panel_into(
                values[..., None],
                mask[..., None],
                active,
                specs[f : f + 1],
                dest,
                valid,
                source_rows=selected - 1,
                membership_rows=selected,
            )
            observation_age_sessions_into(
                mask[..., None],
                active,
                age,
                source_rows=selected - 1,
                decision_rows=selected,
            )
            outputs[f].append((dest[..., 0], valid[..., 0], age[..., 0]))
    for f in affected:
        before = [
            old(k)[rows, :, f]
            for k in ("slow_values", "slow_valid", "slow_age_sessions")
        ]
        for kind, a, b in zip(
            ("values", "valid", "age_control"), outputs[f][0], before, strict=True
        ):
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)))
            checks.append(
                dict(name=f"{kind}:{f}", cells=a.size, mismatches=int(bad.sum()))
            )
        counts = []
        for key, b, a in zip(
            ("slow_values", "slow_valid", "slow_age_sessions"),
            before,
            outputs[f][1],
            strict=True,
        ):
            ix = np.argwhere(~((a == b) | (np.isnan(a) & np.isnan(b))))
            indices = np.column_stack((rows[ix[:, 0]], ix[:, 1], np.full(len(ix), f)))
            packed[key + "__indices"] = np.concatenate(
                (packed[key + "__indices"], indices)
            )
            packed[key + "__values"] = np.concatenate(
                (packed[key + "__values"], a[tuple(ix.T)])
            )
            counts.append(len(ix))
        mask = outputs[f][1][1]
        effects.append(
            dict(
                field=f,
                name=specs[f].name,
                values=counts[0],
                valid=counts[1],
                ages=counts[2],
                gains=int((mask & ~before[1]).sum()),
                losses=int((~mask & before[1]).sum()),
            )
        )
    np.savez_compressed(out / "deltas.npz", **packed)
    report = {
        "status": "passed"
        if not any(x["mismatches"] for x in checks)
        else "control_mismatch_requires_qualification",
        "parent": plan["parent"],
        "producer": run["surviving_rename_daily"],
        "previous": run["surviving_rename_daily_qualification"],
        "source_mask_resolution": binding(prior / "source_mask_resolution.json"),
        "activity_resolution": binding(out / "activity_resolution.json"),
        "checks": checks,
        "effects": sorted(effects, key=lambda r: r["field"]),
        "deltas": binding(out / "deltas.npz"),
        "seconds": perf_counter() - tick,
        "remaining": "Actual full60 daily consumer, common state, issuer/auxiliary/native/scalar dependencies and final target/store composition; not StageA or model acceptance.",
    }
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_daily_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed": [c for c in checks if c["mismatches"]],
                "source_activity_rows": len(np.argwhere(changes)),
                "seconds": report["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
