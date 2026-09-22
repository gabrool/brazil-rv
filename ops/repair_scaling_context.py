"""Compose added-period context using saved parent coordinates and new histories."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_exposures as exp
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import _common_state_diagnostic_panel
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.round5_derived import identity_axes
from brazil_rv.v2.round5_store import align_family
from propagate_fca_peers import panel_frame
from propagate_matched_event_context import source_basis
from repair_scaling_inputs import PROJECT, context, record


def amended(root, manifest, key, start, stop, records):
    a = np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")[
        start:stop
    ].copy()
    for receipt in records:
        with np.load(receipt["path"]) as z:
            if key + "__indices" not in z:
                continue
            ix = z[key + "__indices"]
            keep = (ix[:, 0] >= start) & (ix[:, 0] < stop)
            selected = ix[keep].copy()
            selected[:, 0] -= start
            a[tuple(selected.T)] = z[key + "__values"][keep]
    return a


def setup():
    run, plan, root, m, _ = context()
    daily = bound_json(run["scaling_data_daily"])
    start, first, stop = bound_json(daily["plan"])["rows"]
    dr = Path(daily["plan"]["path"]).parent / "corrected"
    records = [
        bound_json(run[k])["deltas"]
        for k in (
            "scaling_data_identity",
            "scaling_data_wealth",
            "scaling_data_daily_qualification",
        )
    ]
    dates = np.load(root / "date_index.npy")
    names = np.load(root / "isin_index.npy").tolist()
    return run, plan, root, m, start, first, stop, dr, records, dates, names


def accepted_family(run, m, dates, names, rows, family):
    """Reuse full raw parent families and their already-qualified sparse layers."""
    key = (
        "rename_dependents" if family == "sector" else "activity_magnitude_propagation"
    )
    base = bound_json(run[key])["artifacts"][family]
    fields = m["feature_names"]["sidecar_" + family]
    frame = pl.read_parquet(base["path"]).filter(
        pl.col("date").is_between(
            dates[rows[0]].astype(object), dates[rows[-1]].astype(object)
        )
    )
    arrays = list(
        align_family(frame, dates[rows].astype(object).tolist(), tuple(names), fields)
    )
    if family == "sector":
        for key in ("surviving_rename_context", "stage_c_event_context"):
            receipt = bound_json(run[key])["artifacts"]["corrected_sector"]
            f = pl.read_parquet(receipt["path"])
            mask = (dates[rows] >= f["date"].min()) & (dates[rows] <= f["date"].max())
            panel = align_family(
                f, dates[rows[mask]].astype(object).tolist(), tuple(names), fields
            )
            for a, v in zip(arrays, panel, strict=True):
                a[mask] = v
    else:
        producer = bound_json(run["natura_auxiliary_propagation"])
        nr = np.load(producer["rows"]["path"])
        location = Path(run["natura_auxiliary_qualification"]["path"]).parent
        sources = [(location / f"{family}_rows.npz", nr)]
        for key in ("surviving_rename_market", "stage_c_event_market"):
            location = Path(bound_json(run[key])["plan"]["path"]).parent
            sources.append((location / f"{family}_corrected.npz", None))
        for path, indices in sources:
            with np.load(path) as z:
                indices = z["rows"] if indices is None else indices
                use = np.isin(indices, rows)
                dest = np.searchsorted(rows, indices[use])
                for a, k in zip(arrays, ("values", "valid", "ages"), strict=True):
                    a[dest] = z[k][use]
    return fields, arrays


def corrected_basis(run, plan, record_, root, m, start, stop, dr, records):
    """Keep original oil/sector bases, prior admitted episodes and public old JSL."""
    arrays = source_basis(run, record_, root, m, start, stop)
    dates = np.load(root / "date_index.npy")[start:stop]
    names = np.load(root / "isin_index.npy").tolist()
    jsl = names.index("BRJSLGACNOR2")
    old_jsl = {k: a[:, jsl].copy() for k, a in arrays.items()}
    slow = amended(root, m, "slow_valid", start, stop, records)

    def extend(event, location, source_start):
        a, b = names.index(event["isin"]), names.index(event["successor_isin"])
        boundary = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
        prefix = max(0, source_start - start)
        offset = max(0, start - source_start)
        for key in ("close", "valid"):
            dest = arrays["shareholder_wealth_" + key]
            dest[:prefix, b] = dest[:prefix, a]
            dest[prefix:, b] = np.load(location / f"wealth_{key}.npy", mmap_mode="r")[
                offset : offset + stop - start - prefix, b
            ]
        arrays["slow_valid"][:boundary, b] = arrays["slow_valid"][:boundary, a]
        arrays["slow_valid"][boundary:, b] = slow[boundary:, b]
        arrays["action_has_action"][:boundary, b] = arrays["action_has_action"][
            :boundary, a
        ]

    previous = bound_json(run["stage_c_event_daily"])
    prior_start = bound_json(previous["plan"])["rows"][0]
    prior_dr = Path(previous["plan"]["path"]).parent
    old_admission = bound_json(run["stage_c_event_data_admission"])
    for event in bound_json(old_admission["plan"])["events"]:
        extend(event, prior_dr, prior_start)
    reopening = int(np.searchsorted(dates, np.datetime64("2020-11-11")))
    arrays["action_has_action"][reopening, jsl] = False

    # Scalar amendments also affect the predecessor's observed public history.
    for event in plan["scalars"]:
        n = names.index(event["isin"])
        e = int(np.searchsorted(dates, np.datetime64(event["effective_date"])))
        for k in ("shareholder_wealth_close", "shareholder_wealth_valid"):
            arrays[k][e:, n] = amended(root, m, k, start, stop, records)[e:, n]
        arrays["slow_valid"][e:, n] = slow[e:, n]
    for event in plan["history"]:
        extend(event, dr, start)
    return arrays, old_jsl


def finish_family(
    out, family, fields, before, changed, m, root, dates, rows, membership
):
    """Check saved parent transforms, retain new raw coordinates and sparse deltas."""
    specs = [
        FeatureSpec(**s)
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "sidecar_" + family
    ]
    typed, checks, packed = [], [], {}
    for index, (label, arrays) in enumerate(
        (("control", before), ("corrected", changed))
    ):
        values, valid = np.zeros_like(arrays[0], np.float32), np.zeros_like(arrays[1])
        transform_feature_panel_into(
            arrays[0], arrays[1], membership[index], specs, values, valid
        )
        ages = np.where(membership[index][..., None], arrays[2], -1).astype(np.float32)
        typed.append((values, valid, ages))
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
        old = np.load(root / m["arrays"][key]["path"], mmap_mode="r")[rows]
        equal = (old == typed[0][j]) | (np.isnan(old) & np.isnan(typed[0][j]))
        checks.append(
            dict(
                key=key,
                cells=old.size,
                mismatches=int((~equal).sum()),
                examples=np.argwhere(~equal)[:5].tolist(),
            )
        )
        equal = (old == typed[1][j]) | (np.isnan(old) & np.isnan(typed[1][j]))
        ix = np.argwhere(~equal)
        packed[key + "__indices"] = np.column_stack((rows[ix[:, 0]], ix[:, 1:]))
        packed[key + "__values"] = typed[1][j][tuple(ix.T)]
        effects[suffix] = len(ix)
    names = np.load(root / "isin_index.npy").tolist()
    losses = np.argwhere(typed[0][1] & ~typed[1][1])
    write_json_atomic(
        out / f"{family}_losses.json",
        [
            dict(
                date=str(dates[rows[t]]),
                isin=names[n],
                field=fields[f],
                still_active=bool(membership[1][t, n]),
            )
            for t, n, f in losses
        ],
    )
    effects.update(
        gains=int((typed[1][1] & ~typed[0][1]).sum()),
        losses=len(losses),
        live_losses=int(sum(membership[1][t, n] for t, n, _ in losses)),
        shared_changed=int(
            (typed[1][1] & typed[0][1] & (typed[1][0] != typed[0][0])).sum()
        ),
    )
    report = dict(checks=checks, effects=effects, names=fields)
    write_json_atomic(out / f"{family}_report.json", report)
    np.savez_compressed(out / f"{family}_deltas.npz", **packed)
    print(json.dumps(dict(family=family, **report)), flush=True)
    assert not any(c["mismatches"] for c in checks)
    return report, packed


def main():
    tick = perf_counter()
    run, plan, root, m, start, first, stop, dr, records, dates, names = setup()
    out = Path(bound_json(run["scaling_data_workspace"])["root"]) / "context"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    (out / "executed_basis.py").write_bytes(
        (PROJECT / "ops/propagate_matched_event_context.py").read_bytes()
    )
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=plan["parent"],
            rows=[start, first, stop],
            daily=run["scaling_data_daily_qualification"],
            issuer=run["scaling_data_issuers"],
            contrast="Four own-company histories and two evidenced scalar corrections only. Reuse parent raw sector tiers; original horizons/two-other-issuer support, public historical JSL and private later own history. Common state diagnostic only. No source census or repeated old regression. Combined consumer follows all new families.",
        ),
    )
    days, rows = dates[start:stop], np.arange(first, stop)
    selected = rows - start
    active = amended(root, m, "active", start, stop, records)
    sigma = amended(root, m, "target_scale_sigma", start, stop, records)
    oldactive = np.load(root / m["arrays"]["active"]["path"], mmap_mode="r")[rows]
    public = {k: np.load(dr / f"wealth_{k}.npy").copy() for k in ("close", "valid")}
    ambiguous = np.load(dr / "ambiguous.npy").copy()
    jsl = names.index("BRJSLGACNOR2")
    holding_end = int(np.searchsorted(days, np.datetime64("2020-09-18")))
    for k in public:
        public[k][:holding_end, jsl] = np.load(
            root / m["arrays"]["shareholder_wealth_" + k]["path"], mmap_mode="r"
        )[start : start + holding_end, jsl]
    # Reuse already qualified public return masks for the old holding episode.
    with np.load(dr / "public_returns_1.npz") as z:
        ambiguous[:holding_end, jsl] = ~z["valid"][:holding_end, jsl]
    common = _common_state_diagnostic_panel(
        public["close"],
        public["valid"],
        ambiguous,
        active,
        sigma,
        selected,
        days,
        minimum_names=20,
    )
    common[2].write_parquet(out / "common_rows.parquet")
    oracle, valid = np.zeros_like(common[0]), np.zeros_like(common[1])
    for i, t in enumerate(selected):
        mask = (
            active[t]
            & public["valid"][t - 1]
            & public["valid"][t - 2]
            & ~ambiguous[t - 1]
        )
        for s in (t - 1, t - 2):
            mask &= np.isfinite(public["close"][s]) & (public["close"][s] > 0)
        if mask.sum() >= 20:
            ret = np.log(
                public["close"][t - 1, mask].astype(float)
                / public["close"][t - 2, mask].astype(float)
            )
            oracle[i, 0], oracle[i, 2] = np.median(ret), np.std(ret)
            valid[i, (0, 2)] = True
        use = active[t] & np.isfinite(sigma[t]) & (sigma[t] > 1e-8)
        if use.sum() >= 20:
            oracle[i, 1], valid[i, 1] = np.median(sigma[t, use].astype(float)), True
    assert np.array_equal(common[0], oracle) and np.array_equal(common[1], valid)
    packed, effects = {}, {}
    for j, suffix in enumerate(("values", "valid")):
        key = "common_state_diagnostic_" + suffix
        old = np.load(root / m["arrays"][key]["path"], mmap_mode="r")[rows]
        ix = np.argwhere(common[j] != old)
        packed[key + "__indices"] = np.column_stack((rows[ix[:, 0]], ix[:, 1:]))
        packed[key + "__values"] = common[j][tuple(ix.T)]
        effects[key] = len(ix)
    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    fm = bound_json(
        dict(
            path=str(Path(foundation["root"]) / "manifest.json"),
            sha256=foundation["manifest_sha256"],
        )
    )
    original = bound_json(source_families(fm)["sector"]["source_manifest"])["sources"][
        "inputs"
    ]["base_manifest"]
    arrays, old_jsl = corrected_basis(
        run, plan, original, root, m, start, stop, dr, records
    )
    identity = bound_json(run["scaling_data_issuers"])["artifacts"]["identity"]
    sectors, issuers = identity_axes(
        pl.read_parquet(identity["path"]).filter(
            pl.col("date").is_between(days[0].astype(object), days[-1].astype(object))
        ),
        days.astype(object).tolist(),
        names,
    )
    values, valid = np.zeros((*active.shape, 3)), np.zeros((*active.shape, 3), bool)
    reopening = int(np.searchsorted(days, np.datetime64("2020-11-11")))
    for j, h in enumerate((5, 21, 252)):
        outputs = []
        for a in (arrays, {k: v[:, None] for k, v in old_jsl.items()}):
            raw, mask = exact_log_return(
                a["shareholder_wealth_close"],
                h,
                shareholder_wealth_valid=a["shareholder_wealth_valid"],
            )
            field = f"log_return_{h}"
            if h == 252:
                short, ok = exact_log_return(
                    a["shareholder_wealth_close"],
                    21,
                    shareholder_wealth_valid=a["shareholder_wealth_valid"],
                )
                raw -= short
                mask &= ok
                field = "momentum_12_1"
            outputs.append((raw, mask))
        raw, mask = outputs[0]
        raw[:reopening, jsl], mask[:reopening, jsl] = (
            outputs[1][0][:reopening, 0],
            outputs[1][1][:reopening, 0],
        )
        support = arrays["slow_valid"][
            :, :, m["feature_names"]["slow"].index(field)
        ].copy()
        support[:reopening, jsl] = old_jsl["slow_valid"][
            :reopening, m["feature_names"]["slow"].index(field)
        ]
        values[1:, :, j], valid[1:, :, j] = raw[:-1], mask[:-1] & support[1:]
    values, valid = exp.sector_relative_panel(
        values[selected],
        valid[selected],
        sectors[selected],
        issuers[selected],
        active[selected],
    )
    frame = panel_frame(
        values,
        valid,
        np.where(valid, 1, -1),
        exp.SECTOR_FEATURE_NAMES,
        dates[rows].astype(object).tolist(),
        names,
        active[selected],
    )
    frame.write_parquet(out / "corrected_sector.parquet")
    changed = list(
        align_family(
            frame,
            dates[rows].astype(object).tolist(),
            tuple(names),
            exp.SECTOR_FEATURE_NAMES,
        )
    )
    fields, before = accepted_family(run, m, dates, names, rows, "sector")
    report, delta = finish_family(
        out,
        "sector",
        fields,
        before,
        changed,
        m,
        root,
        dates,
        rows,
        (oldactive, active[selected]),
    )
    packed.update(delta)
    effects["sector"] = report["effects"]
    np.savez_compressed(out / "deltas.npz", **packed)
    result = dict(
        status="qualified_intermediates_pending_combined_consumer",
        plan=binding(out / "plan.json"),
        effects=effects,
        sector=report,
        common_oracle_cells=oracle.size + valid.size,
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        deltas=binding(out / "deltas.npz"),
        seconds=perf_counter() - tick,
    )
    record(run, "scaling_data_context", out / "manifest.json", result)
    print(
        json.dumps(
            dict(status=result["status"], effects=effects, seconds=result["seconds"])
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
