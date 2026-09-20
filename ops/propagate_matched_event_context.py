"""Compose common and sector state, retaining the historical public JSL episode."""

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

PROJECT = Path(__file__).resolve().parents[1]


def source_basis(run, record, root, manifest, start, stop):
    """Recompose the accepted original-sector/oil basis without source reducers."""
    sm = bound_json(record)
    sr = Path(record["path"]).parent
    arrays = {
        k: np.load(sr / sm["arrays"][k]["path"], mmap_mode="r")[start:stop].copy()
        for k in (
            "shareholder_wealth_close",
            "shareholder_wealth_valid",
            "slow_valid",
            "action_has_action",
        )
    }
    prior = bound_json(run["surviving_rename_admission"])
    prior_root = Path(prior["parent"]["root"])
    names = np.load(root / "isin_index.npy").tolist()
    old_links = pl.read_parquet(prior_root / "slow_history_links.parquet")
    history = bound_json(run["rename_history_propagation"])["history_basis"]
    slow = np.load(root / manifest["arrays"]["slow_valid"]["path"], mmap_mode="r")[
        start:stop
    ]

    def extend(link, wealth_root, offset):
        a, b = link["predecessor_index"], link["successor_index"]
        e = max(0, link["effective_index"] - start)
        for key in ("close", "valid"):
            arrays["shareholder_wealth_" + key][:, b] = np.load(
                wealth_root[key], mmap_mode="r"
            )[offset : offset + stop - start, b]
        arrays["slow_valid"][:e, b] = arrays["slow_valid"][:e, a]
        arrays["slow_valid"][e:, b] = slow[e:, b]
        arrays["action_has_action"][:e, b] = arrays["action_has_action"][:e, a]

    for link in old_links.sort("effective_index").iter_rows(named=True):
        extend(
            link,
            {k: history["shareholder_wealth_" + k]["path"] for k in ("close", "valid")},
            start,
        )
    n = names.index("BRNATUACNOR6")
    for k in ("shareholder_wealth_close", "shareholder_wealth_valid"):
        arrays[k][:, n] = np.load(root / manifest["arrays"][k]["path"], mmap_mode="r")[
            start:stop, n
        ]
    daily = bound_json(run["surviving_rename_daily"])
    dp = bound_json(daily["plan"])
    dr = Path(daily["plan"]["path"]).parent / "corrected"
    affected = {names.index("BRSSBRACNOR1"), names.index("BRARZZACNOR3")}
    for link in (
        pl.read_parquet(root / "slow_history_links.parquet")
        .sort("effective_index")
        .iter_rows(named=True)
    ):
        if link["predecessor_index"] in affected:
            affected.add(link["successor_index"])
            extend(
                link,
                {k: dr / f"wealth_{k}.npy" for k in ("close", "valid")},
                start - dp["rows"][0],
            )
    return arrays


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    events = bound_json(admission["plan"])["events"]
    daily = bound_json(run["stage_c_event_daily"])
    qualified = bound_json(run["stage_c_event_daily_qualification"])
    dp = bound_json(daily["plan"])
    start, first, stop = dp["rows"]
    dr = Path(daily["plan"]["path"]).parent
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = Path(admission["plan"]["path"]).parent / "context"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            daily=run["stage_c_event_daily_qualification"],
            issuer=run["stage_c_event_issuers"],
            prior=run["surviving_rename_context"],
            registration=binding(
                PROJECT / "research/preregistrations/v2_economic_data_scaling.md"
            ),
            rows=[start, first, stop],
            contrast="Only five admitted market continuations, their current legal identities and final daily support/risk. Original sector basis/horizons/two-other-issuer rule; common state diagnostic only. Preserve old public JSL before its logistics reopening while new-episode own lookbacks use JSLG11. No source census, old reducers, model inference or full consumer campaign.",
            verification="Recompose accepted raw sector controls and typed controls once for this new source-basis alignment; new common state has an independent direct formula. Preserve all sparse outputs and enumerated losses. One combined full-store consumer qualification follows remaining families.",
        ),
    )
    dates = np.load(root / "date_index.npy")
    names = np.load(root / "isin_index.npy").tolist()
    days, rows = dates[start:stop], np.arange(first, stop)
    selected = rows - start

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

    active = np.load(dr / "active.npy")
    sigma = after("target_scale_sigma")
    slow = after("slow_valid")
    private = {k: np.load(dr / f"wealth_{k}.npy") for k in ("close", "valid")}
    ambiguous = np.load(dr / "ambiguous.npy")
    jsl = names.index("BRJSLGACNOR2")
    holding_end = int(np.searchsorted(days, np.datetime64("2020-09-18")))
    reopening = int(np.searchsorted(days, np.datetime64("2020-11-11")))
    public = {k: v.copy() for k, v in private.items()}
    for k in public:
        public[k][:holding_end, jsl] = old("shareholder_wealth_" + k)[:holding_end, jsl]
    # JSL is inactive between these episodes, so a new private basis is safe
    # there and supplies the two prior observations at its later decision.
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
    checks, packed, effects = [], {}, {}

    def check(name, a, b):
        equal = (a == b) | (np.isnan(a) & np.isnan(b))
        checks.append(
            dict(
                name=name,
                cells=a.size,
                mismatches=int((~equal).sum()),
                examples=np.argwhere(~equal)[:5].tolist(),
            )
        )

    oracle = np.zeros_like(common[0])
    oracle_valid = np.zeros_like(common[1])
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
            r = np.log(
                public["close"][t - 1, mask].astype(np.float64)
                / public["close"][t - 2, mask].astype(np.float64)
            )
            oracle[i, 0], oracle[i, 2] = np.median(r), np.std(r)
            oracle_valid[i, (0, 2)] = True
        use = active[t] & np.isfinite(sigma[t]) & (sigma[t] > 1e-8)
        if use.sum() >= 20:
            oracle[i, 1] = np.median(sigma[t, use].astype(np.float64))
            oracle_valid[i, 1] = True
    for j, suffix in enumerate(("values", "valid")):
        check("common_" + suffix, common[j], (oracle, oracle_valid)[j])
        key = "common_state_diagnostic_" + suffix
        ix = np.argwhere(common[j] != old(key)[selected])
        packed[key + "__indices"] = np.column_stack([rows[ix[:, 0]], ix[:, 1]])
        packed[key + "__values"] = common[j][tuple(ix.T)]
        effects[key] = len(ix)

    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    fm = bound_json(
        dict(
            path=str(Path(foundation["root"]) / "manifest.json"),
            sha256=foundation["manifest_sha256"],
        )
    )
    families = source_families(fm)
    record = bound_json(families["sector"]["source_manifest"])["sources"]["inputs"][
        "base_manifest"
    ]
    before = source_basis(run, record, root, m, start, stop)
    changed = {k: a.copy() for k, a in before.items()}
    for e in events:
        a, b = names.index(e["isin"]), names.index(e["successor_isin"])
        boundary = int(np.searchsorted(days, np.datetime64(e["effective_date"])))
        for k in ("close", "valid"):
            changed["shareholder_wealth_" + k][:, b] = private[k][:, b]
        changed["slow_valid"][:boundary, b] = changed["slow_valid"][:boundary, a]
        changed["slow_valid"][boundary:, b] = slow[boundary:, b]
    prior = bound_json(run["surviving_rename_context"])["artifacts"]["corrected_sector"]
    parent_frame = pl.read_parquet(prior["path"]).filter(
        pl.col("date").is_between(dates[first].astype(object), dates[-1].astype(object))
    )
    expected = align_family(
        parent_frame,
        dates[rows].astype(object).tolist(),
        tuple(names),
        exp.SECTOR_FEATURE_NAMES,
    )
    identities = [
        bound_json(run[k])["artifacts"]["identity"]
        for k in ("surviving_rename_issuers", "stage_c_event_issuers")
    ]
    old_public_values, old_public_valid = None, None
    typed = []
    for index, (label, arrays) in enumerate(
        (("control", before), ("corrected", changed))
    ):
        sectors, issuers = identity_axes(
            pl.read_parquet(identities[index]["path"]).filter(
                pl.col("date").is_between(
                    days[0].astype(object), days[-1].astype(object)
                )
            ),
            days.astype(object).tolist(),
            names,
        )
        values = np.zeros((*active.shape, 3))
        valid = np.zeros_like(values, bool)
        for j, horizon in enumerate((5, 21, 252)):
            raw, mask = exact_log_return(
                arrays["shareholder_wealth_close"],
                horizon,
                shareholder_wealth_valid=arrays["shareholder_wealth_valid"],
            )
            field = f"log_return_{horizon}"
            if horizon == 252:
                short, short_valid = exact_log_return(
                    arrays["shareholder_wealth_close"],
                    21,
                    shareholder_wealth_valid=arrays["shareholder_wealth_valid"],
                )
                raw -= short
                mask &= short_valid
                field = "momentum_12_1"
            values[1:, :, j] = raw[:-1]
            valid[1:, :, j] = (
                mask[:-1]
                & arrays["slow_valid"][1:, :, m["feature_names"]["slow"].index(field)]
            )
        if not index:
            old_public_values, old_public_valid = (
                values[:, jsl].copy(),
                valid[:, jsl].copy(),
            )
        else:
            values[:reopening, jsl] = old_public_values[:reopening]
            valid[:reopening, jsl] = old_public_valid[:reopening]
        membership = active if index else old("active")
        values, valid = exp.sector_relative_panel(
            values[selected],
            valid[selected],
            sectors[selected],
            issuers[selected],
            membership[selected],
        )
        frame = panel_frame(
            values,
            valid,
            np.where(valid, 1, -1),
            exp.SECTOR_FEATURE_NAMES,
            dates[rows].astype(object).tolist(),
            names,
            membership[selected],
        )
        frame.write_parquet(out / f"{label}_sector.parquet")
        result = align_family(
            frame,
            dates[rows].astype(object).tolist(),
            tuple(names),
            exp.SECTOR_FEATURE_NAMES,
        )
        if not index:
            for k, a, b in zip(
                ("values", "valid", "age"), result, expected, strict=True
            ):
                check("sector_raw_" + k, a, b)
        specs = [
            FeatureSpec(**s)
            for s in m["metadata"]["feature_schema"]["specifications"]
            if s["family"] == "sidecar_sector"
        ]
        tv, tm = np.zeros_like(result[0], np.float32), np.zeros_like(result[1])
        transform_feature_panel_into(
            result[0], result[1], membership[selected], specs, tv, tm
        )
        ta = np.where(membership[selected, :, None], result[2], -1).astype(np.float32)
        typed.append((tv, tm, ta))
        np.savez_compressed(
            out / f"{label}_sector.npz", rows=rows, values=tv, valid=tm, age_sessions=ta
        )
    for j, suffix in enumerate(("values", "valid", "age_sessions")):
        key = "sidecar_sector_" + suffix
        check(key, typed[0][j], old(key)[selected])
        equal = (typed[1][j] == old(key)[selected]) | (
            np.isnan(typed[1][j]) & np.isnan(old(key)[selected])
        )
        ix = np.argwhere(~equal)
        packed[key + "__indices"] = np.column_stack([rows[ix[:, 0]], ix[:, 1:]])
        packed[key + "__values"] = typed[1][j][tuple(ix.T)]
    losses = np.argwhere(typed[0][1] & ~typed[1][1])
    write_json_atomic(
        out / "validity_losses.json",
        [
            dict(
                date=str(dates[rows[t]]),
                isin=names[n],
                field=exp.SECTOR_FEATURE_NAMES[f],
                still_active=bool(active[selected[t], n]),
            )
            for t, n, f in losses
        ],
    )
    effects["sector"] = dict(
        gains=int((typed[1][1] & ~typed[0][1]).sum()),
        losses=len(losses),
        live_losses=int(sum(active[selected[t], n] for t, n, _ in losses)),
        shared_numeric=int(
            (typed[1][1] & typed[0][1] & (typed[1][0] != typed[0][0])).sum()
        ),
    )
    np.savez_compressed(out / "deltas.npz", **packed)
    report = dict(
        status="passed"
        if not any(c["mismatches"] for c in checks)
        else "failed_outputs_preserved",
        plan=binding(out / "plan.json"),
        checks=checks,
        effects=effects,
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        deltas=binding(out / "deltas.npz"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    print(json.dumps(report), flush=True)
    assert report["status"] == "passed"
    run = json.loads(pointer.read_text())
    run["stage_c_event_context"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
