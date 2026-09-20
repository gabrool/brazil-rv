"""Qualify saved rename reducers and write sparse intermediate amendments."""

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
from brazil_rv.v2.features import _ambiguous_interval_clear

PROJECT = Path(__file__).resolve().parents[1]


def same(a, b):
    return (a == b) | (np.isnan(a) & np.isnan(b))


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    produced = bound_json(run["surviving_rename_daily"])
    plan = bound_json(produced["plan"])
    source = Path(produced["plan"]["path"]).parent
    out = source / "qualified"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
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
    rows = selected + start
    active = [
        np.load(source / label / "active.npy") for label in ("control", "corrected")
    ]
    specs = [
        FeatureSpec(**s)
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]

    def old(k):
        return np.load(root / m["arrays"][k]["path"], mmap_mode="r")

    # Reuse every saved reducer. The first recipe used accepted raw-price
    # availability where original daily feature rows were absent. That widens
    # masks outside the rename scope. Restore the source-row mask explicitly;
    # original wealth already contains its missing-row/restart boundaries.
    quotes = pl.read_parquet(produced["source_coordinates"]["path"])
    present = np.zeros((stop - start, len(isins)), bool)
    qi = np.searchsorted(dates[start:], quotes["trade_date"].to_numpy())
    lookup = {s: j for j, s in enumerate(isins)}
    qj = np.array([lookup[s] for s in quotes["isin"]])
    present[qi, qj] = True
    gaps = old("observed")[start:] & ~present
    admission = bound_json(run["surviving_rename_admission"])
    tables = [
        pl.read_parquet(root / m["tables"]["isin_succession_links"]["path"]),
        pl.read_parquet(admission["links"]["path"]),
    ]
    gap_masks = []
    for table in tables:
        mask = gaps.copy()
        for link in table.sort("successor_first_date").iter_rows(named=True):
            a, b = (
                isins.index(link[k]) for k in ("predecessor_isin", "successor_isin")
            )
            boundary = (
                int(np.searchsorted(dates, np.datetime64(link["effective_date"])))
                - start
            )
            mask[:boundary, b] = mask[:boundary, a]
        gap_masks.append(mask)
    gap_rows = np.argwhere(gaps)
    write_json_atomic(
        out / "source_mask_resolution.json",
        {
            "failed_qualification": binding(source / "qualification/manifest.json"),
            "source_coordinates": produced["source_coordinates"],
            "rows": [
                dict(date=str(dates[start + t]), isin=isins[n]) for t, n in gap_rows
            ],
            "disposition": "Accepted raw-price marks and original normalized feature OHLC rows are distinct source contracts. Restore the original feature-source mask; no accepted observation is removed, no new source defect or fabricated raw bar is claimed. Existing saved wealth already has gaps and restarts. Intersect only the unchanged feature's explicit source/ambiguity window; reuse raw values and both saved cluster/market calculations.",
            "executed_producer": binding(source / "executed.py"),
            "qualified_producer": binding(PROJECT / "ops/propagate_surviving_daily.py"),
        },
    )
    packed, checks, effects = {}, [], []

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

    for field in [*range(25), *range(27, 32)]:
        outputs, masks, ages, raws = [], [], [], []
        for i, label in enumerate(("control", "corrected")):
            with np.load(source / label / f"{field}.npz") as z:
                raw, mask = z["values"], z["valid"]
            horizons = {
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
            if field in horizons:
                mask &= _ambiguous_interval_clear(gap_masks[i], horizons[field])
            elif field in (22, 24):
                mask &= ~gap_masks[i]
            if field not in (17, 18, 20, 21, 22, 24, 27, 28, 29, 30, 31):
                raw = np.where(mask, raw, np.nan)
            values = np.zeros((len(selected), len(isins), 1), np.float32)
            valid = np.zeros_like(values, bool)
            transform_feature_panel_into(
                raw[..., None],
                mask[..., None],
                active[i],
                specs[field : field + 1],
                values,
                valid,
                source_rows=selected - 1,
                membership_rows=selected,
            )
            age = np.full_like(values, -1)
            observation_age_sessions_into(
                mask[..., None],
                active[i],
                age,
                source_rows=selected - 1,
                decision_rows=selected,
            )
            outputs.append(values[..., 0])
            masks.append(valid[..., 0])
            ages.append(age[..., 0])
            raws.append((raw, mask))
        before, bmask, bage = (
            old(k)[rows, :, field]
            for k in ("slow_values", "slow_valid", "slow_age_sessions")
        )
        check(f"values:{field}", outputs[0], before)
        check(f"valid:{field}", masks[0], bmask)
        changed_clock = (ages[0] != ages[1]) | (
            active[0][selected] != active[1][selected]
        )
        # Preserve old clocks that predate this reducer window. Changed active
        # source clocks are checked; new eligibility uses the same source mask.
        check(
            f"age_control:{field}",
            ages[0][changed_clock & active[0][selected]],
            bage[changed_clock & active[0][selected]],
        )
        after_age = bage.copy()
        after_age[changed_clock] = ages[1][changed_clock]
        effects.append(
            dict(
                field=field,
                name=specs[field].name,
                values=delta("slow_values", before, outputs[1], field),
                valid=delta("slow_valid", bmask, masks[1], field),
                ages=delta("slow_age_sessions", bage, after_age, field),
                gains=int((masks[1] & ~bmask).sum()),
                losses=int((~masks[1] & bmask).sum()),
            )
        )
        np.savez_compressed(
            out / f"field_{field}.npz", values=outputs[1], valid=masks[1], age=after_age
        )
        if field == 8:
            # Sigma is own-name. Only the new identity chains can authorize
            # changes; unrelated inactive scratch discrepancies stay explicit.
            sigma = [
                np.where(v[selected - 1], a[selected - 1], np.nan).astype(np.float32)
                for a, v in raws
            ]
            links = pl.read_parquet(
                bound_json(run["surviving_rename_admission"])["links"]["path"]
            )
            for row in links.iter_rows(named=True):
                j = isins.index(row["successor_isin"])
                before_birth = dates[rows] < np.datetime64(row["effective_date"])
                for a in sigma:
                    a[before_birth, j] = old("target_scale_sigma")[
                        rows[before_birth], j
                    ]
            scope = np.array(
                [
                    isins.index(s)
                    for s in (
                        "BRSSBRACNOR1",
                        "BRALSOACNOR5",
                        "BRALOSACNOR5",
                        "BRARZZACNOR3",
                        "BRAZZAACNOR9",
                    )
                ]
            )
            other = np.ones(len(isins), bool)
            other[scope] = False
            check("sigma_unchanged_other_names", sigma[1][:, other], sigma[0][:, other])
            baseline = old("target_scale_sigma")[rows]
            check("sigma_scope_control", sigma[0][:, scope], baseline[:, scope])
            check(
                "sigma_eligible_control",
                sigma[0][active[0][selected]],
                baseline[active[0][selected]],
            )
            off = np.argwhere(~same(sigma[0], baseline))
            write_json_atomic(
                out / "offscope_sigma.json",
                {
                    "rows": [
                        dict(
                            date=str(dates[rows[t]]),
                            isin=isins[n],
                            active=bool(active[0][selected[t], n]),
                            scope=bool(n in scope),
                            scratch=str(sigma[0][t, n]),
                            accepted=str(baseline[t, n]),
                        )
                        for t, n in off
                    ],
                    "disposition": "Offscope cells retain their accepted values; this is not evidence of a new source defect.",
                },
            )
            after = baseline.copy()
            after[:, scope] = sigma[1][:, scope]
            delta("target_scale_sigma", baseline, after)
    # Unranked original listing age: infer no arbitrary age from the bounded
    # window. Both new predecessors first print inside the accepted full axis.
    observed = old("observed")
    for field in (25, 26):
        before = old("slow_values")[rows, :, field]
        bmask = old("slow_valid")[rows, :, field]
        bage = old("slow_age_sessions")[rows, :, field]
        after, amask, aage = before.copy(), bmask.copy(), bage.copy()
        retire = active[0][selected] & ~active[1][selected]
        after[retire] = 0
        amask[retire] = False
        aage[retire] = -1
        for ancestor, successor, effect in (
            ("BRSSBRACNOR1", "BRALSOACNOR5", "2019-08-06"),
            ("BRSSBRACNOR1", "BRALOSACNOR5", "2023-10-25"),
            ("BRARZZACNOR3", "BRAZZAACNOR9", "2024-08-01"),
        ):
            a, b = isins.index(ancestor), isins.index(successor)
            birth = np.flatnonzero(observed[:, a])[0]
            assert birth > 0
            raw = (
                np.maximum(np.arange(start, stop) - birth, 0).astype(np.float64)[
                    :, None, None
                ]
                if field == 25
                else np.zeros((stop - start, 1, 1))
            )
            mask = (np.arange(start, stop) >= birth)[:, None, None]
            value = np.zeros((len(rows), 1, 1), np.float32)
            valid = np.zeros_like(value, bool)
            transform_feature_panel_into(
                raw,
                mask,
                active[1][:, b : b + 1],
                specs[field : field + 1],
                value,
                valid,
                source_rows=selected - 1,
                membership_rows=selected,
            )
            take = dates[rows] >= np.datetime64(effect)
            after[take, b] = value[take, 0, 0]
            amask[take, b] = valid[take, 0, 0]
            aage[take, b] = np.where(active[1][selected[take], b], 1, -1)
        effects.append(
            dict(
                field=field,
                name=specs[field].name,
                values=delta("slow_values", before, after, field),
                valid=delta("slow_valid", bmask, amask, field),
                ages=delta("slow_age_sessions", bage, aage, field),
                gains=int((amask & ~bmask).sum()),
                losses=int((~amask & bmask).sum()),
            )
        )
        np.savez_compressed(
            out / f"field_{field}.npz", values=after, valid=amask, age=aage
        )
    clusters = [
        np.load(source / label / "clusters.npy")[selected]
        for label in ("control", "corrected")
    ]
    check("cluster_control", clusters[0], old("monthly_cluster_labels")[rows])
    delta("monthly_cluster_labels", clusters[0], clusters[1])
    np.savez_compressed(
        out / "deltas.npz", **{k: np.concatenate(v) for k, v in packed.items()}
    )
    report = {
        "status": "passed"
        if not any(x["mismatches"] for x in checks)
        else "control_mismatch_requires_qualification",
        "producer": run["surviving_rename_daily"],
        "checks": checks,
        "effects": effects,
        "deltas": binding(out / "deltas.npz"),
        "seconds": perf_counter() - tick,
        "scope": "Daily fields/risk only; common state, auxiliary/native/scalar/issuer/targets and complete-store consumers still pending.",
    }
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_daily_qualification"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed": [x for x in checks if x["mismatches"]],
                "effects": effects,
                "seconds": report["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
