"""Compose common-state and sector dependencies from saved rename reducers."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_exposures as exp
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.build_store import _common_state_diagnostic_panel
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.round5_derived import identity_axes
from brazil_rv.v2.round5_store import align_family
from propagate_fca_peers import panel_frame

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    admission = bound_json(run["surviving_rename_admission"])
    daily = bound_json(run["surviving_rename_daily"])
    qualified = bound_json(run["surviving_rename_daily_qualification"])
    source = Path(daily["plan"]["path"]).parent
    plan = bound_json(daily["plan"])
    root = Path(admission["parent"]["root"])
    manifest = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    start, first, stop = plan["rows"]
    out = source.parent / "context"
    resume = out.exists()
    assert not (out / "manifest.json").exists(), "Completed outputs must be reused"
    out.mkdir(exist_ok=True)
    (out / ("executed_resume.py" if resume else "executed.py")).write_bytes(
        Path(__file__).read_bytes()
    )
    issuer_path = source.parent / "issuers/manifest.json"
    issuer = bound_json(binding(issuer_path))
    if not resume:
        write_json_atomic(
            out / "plan.json",
            {
                "parent": admission["parent"],
                "daily": run["surviving_rename_daily_qualification"],
                "issuer": binding(issuer_path),
                "rows": [start, first, stop],
                "contrast": "Two admitted identity/wealth/liquidity dependencies only. Reuse saved daily reducers and qualified support/risk deltas. Common state remains diagnostic. Sector retains original source wealth, same horizons and leave-one-issuer-out minimum support. Full933 names, prior252history; no model inference.",
                "verification": "Old-coordinate exact controls against accepted diagnostics and prior sector rows; retain all outputs before qualification. Current decision consumes only prior closes; all old stores immutable.",
            },
        )
    dates = np.load(root / "date_index.npy")
    days = dates[start:stop]
    isins = np.load(root / "isin_index.npy").tolist()
    rows = np.arange(first, stop)
    selected = rows - start

    def old(key):
        return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")[
            start:stop
        ]

    def after(key):
        array = old(key).copy()
        with np.load(qualified["deltas"]["path"]) as z:
            if key + "__indices" in z:
                ix = z[key + "__indices"].copy()
                ix[:, 0] -= start
                array[tuple(ix.T)] = z[key + "__values"]
        return array

    membership = [
        np.load(source / label / "active.npy", mmap_mode="r")
        for label in ["control", "corrected"]
    ]
    packed, checks, effects = {}, [], {}

    def check(name, a, b):
        different = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        checks.append(
            dict(
                name=name,
                cells=a.size,
                mismatches=int(different.sum()),
                examples=np.argwhere(different)[:5].tolist(),
            )
        )

    for label, sigma, active in [
        ("control", old("target_scale_sigma"), membership[0]),
        ("corrected", after("target_scale_sigma"), membership[1]),
    ]:
        if (out / f"{label}_common_rows.parquet").exists():
            result = (
                np.load(out / f"{label}_common_state_diagnostic_values.npy"),
                np.load(out / f"{label}_common_state_diagnostic_valid.npy"),
                pl.read_parquet(out / f"{label}_common_rows.parquet"),
            )
        else:
            result = _common_state_diagnostic_panel(
                np.load(source / label / "wealth_close.npy"),
                np.load(source / label / "wealth_valid.npy"),
                np.load(source / label / "ambiguous.npy"),
                active,
                sigma,
                selected,
                days,
                minimum_names=20,
            )
            result[2].write_parquet(out / f"{label}_common_rows.parquet")
        for j, suffix in enumerate(["values", "valid"]):
            key = "common_state_diagnostic_" + suffix
            if not (out / f"{label}_{key}.npy").exists():
                np.save(out / f"{label}_{key}.npy", result[j])
            if label == "control":
                check(key, result[j], old(key)[selected])
            else:
                ix = np.argwhere(result[j] != old(key)[selected])
                packed[key + "__indices"] = np.column_stack([rows[ix[:, 0]], ix[:, 1]])
                packed[key + "__values"] = result[j][tuple(ix.T)]
                effects[key] = len(ix)

    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    base_manifest = bound_json(
        dict(
            path=str(Path(foundation["root"]) / "manifest.json"),
            sha256=foundation["manifest_sha256"],
        )
    )
    families = source_families(base_manifest)
    sector = bound_json(families["sector"]["source_manifest"])
    sb = sector["sources"]["inputs"]["base_manifest"]
    sm = bound_json(sb)
    sr = Path(sb["path"]).parent

    def source_array(key):
        return np.load(sr / sm["arrays"][key]["path"], mmap_mode="r")[start:stop].copy()

    history = bound_json(run["rename_history_propagation"])["history_basis"]
    old_links = pl.read_parquet(root / "slow_history_links.parquet")
    new_links = pl.read_parquet(admission["history_mapping"]["path"])
    wealth, seen, support = (
        source_array("shareholder_wealth_close"),
        source_array("shareholder_wealth_valid"),
        source_array("slow_valid"),
    )
    for link in old_links.iter_rows(named=True):
        a, b, boundary = (
            link["predecessor_index"],
            link["successor_index"],
            link["effective_index"] - start,
        )
        for key, target in [("close", wealth), ("valid", seen)]:
            target[:, b] = np.load(
                history["shareholder_wealth_" + key]["path"], mmap_mode="r"
            )[start:stop, b]
        support[:boundary, b] = support[:boundary, a]
        support[boundary:, b] = old("slow_valid")[boundary:, b]
    natura = isins.index("BRNATUACNOR6")
    wealth[:, natura] = old("shareholder_wealth_close")[:, natura]
    seen[:, natura] = old("shareholder_wealth_valid")[:, natura]
    changed_w, changed_seen, changed_support = (
        wealth.copy(),
        seen.copy(),
        support.copy(),
    )
    slow = after("slow_valid")
    affected = {isins.index("BRSSBRACNOR1"), isins.index("BRARZZACNOR3")}
    for link in new_links.sort("effective_index").iter_rows(named=True):
        a, b, boundary = (
            link["predecessor_index"],
            link["successor_index"],
            link["effective_index"] - start,
        )
        if a not in affected:
            continue
        affected.add(b)
        changed_w[:, b] = np.load(source / "corrected/wealth_close.npy", mmap_mode="r")[
            :, b
        ]
        changed_seen[:, b] = np.load(
            source / "corrected/wealth_valid.npy", mmap_mode="r"
        )[:, b]
        changed_support[:boundary, b] = changed_support[:boundary, a]
        changed_support[boundary:, b] = slow[boundary:, b]
    old_issuer = bound_json(run["rename_issuer_propagation"])
    prior = bound_json(run["rename_dependents"])["artifacts"]["sector"]
    parent_sector = pl.read_parquet(prior["path"]).filter(
        pl.col("date").is_between(
            dates[first].astype(object), dates[stop - 1].astype(object)
        )
    )
    expected = align_family(
        parent_sector,
        dates[rows].astype(object).tolist(),
        tuple(isins),
        exp.SECTOR_FEATURE_NAMES,
    )
    for label, w, seen_mask, masks, active, identity in [
        (
            "control",
            wealth,
            seen,
            support,
            membership[0],
            old_issuer["artifacts"]["identity"],
        ),
        (
            "corrected",
            changed_w,
            changed_seen,
            changed_support,
            membership[1],
            issuer["artifacts"]["identity"],
        ),
    ]:
        sectors, issuers = identity_axes(
            pl.read_parquet(identity["path"]).filter(
                pl.col("date") >= days[0].astype(object)
            ),
            days.astype(object).tolist(),
            isins,
        )
        values = np.zeros((*active.shape, 3))
        valid = np.zeros_like(values, bool)
        for j, h in enumerate([5, 21, 252]):
            raw, known = exact_log_return(w, h, shareholder_wealth_valid=seen_mask)
            field = f"log_return_{h}"
            if h == 252:
                short, short_valid = exact_log_return(
                    w, 21, shareholder_wealth_valid=seen_mask
                )
                raw -= short
                known &= short_valid
                field = "momentum_12_1"
            values[1:, :, j] = raw[:-1]
            valid[1:, :, j] = (
                known[:-1]
                & masks[1:, :, manifest["feature_names"]["slow"].index(field)]
            )
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
            isins,
            active[selected],
        )
        frame.write_parquet(out / f"{label}_sector.parquet")
        actual = align_family(
            frame,
            dates[rows].astype(object).tolist(),
            tuple(isins),
            exp.SECTOR_FEATURE_NAMES,
        )
        if label == "control":
            for j, name in enumerate(["values", "valid", "age"]):
                check("sector_" + name, actual[j], expected[j])
        else:
            effects["sector"] = {
                "gained": int((actual[1] & ~expected[1]).sum()),
                "lost": int((~actual[1] & expected[1]).sum()),
                "changed_shared": int(
                    (actual[1] & expected[1] & (actual[0] != expected[0])).sum()
                ),
            }
    np.savez_compressed(out / "common_deltas.npz", **packed)
    report = dict(
        status="qualified_context_intermediate"
        if not any(c["mismatches"] for c in checks)
        else "control_failed_outputs_preserved",
        plan=binding(out / "plan.json"),
        checks=checks,
        effects=effects,
        sector_parent=prior,
        artifacts={p.stem: binding(p) for p in out.glob("*.parquet")},
        common_deltas=binding(out / "common_deltas.npz"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    print(json.dumps(report), flush=True)
    assert report["status"] == "qualified_context_intermediate"


if __name__ == "__main__":
    main()
