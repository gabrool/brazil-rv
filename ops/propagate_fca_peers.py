"""Identity-only sector, cross-market peer and lending-denominator propagation.

Hold each family's original market inputs fixed to isolate this source amendment.
The resulting parquets are intermediate; a changed-wealth store needs its own build.
"""

import argparse
from datetime import date
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_b3, round5_cvm, round5_exposures as exp
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.cross_market_returns import admit_brent, comparison_return_mask
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.round5_derived import identity_axes, verified
from propagate_fca_financials import compare

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def panel_frame(values, masks, ages, names, sessions, isins, active):
    valid = masks & active[..., None] & np.isfinite(values)
    t, n = np.where(valid.any(axis=-1))
    columns = {
        "date": pl.Series(np.asarray(sessions, dtype="datetime64[D]")[t]),
        "isin": np.asarray(isins)[n],
    }
    for j, name in enumerate(names):
        known = valid[t, n, j]
        columns[name] = pl.Series(values[t, n, j].astype(np.float32)).set(
            pl.Series(~known), None
        )
        columns[name + "_age_sessions"] = np.where(known, ages[t, n, j], -1).astype(
            np.float32
        )
    return pl.DataFrame(columns)


def original_arrays(manifest_binding, sessions, isins):
    manifest = bound_json(manifest_binding)
    root = Path(manifest_binding["path"]).parent

    def array(name):
        return np.load(root / f"{name}.npy", mmap_mode="r")

    np.testing.assert_array_equal(
        array("date_index"), np.array(sessions, dtype="datetime64[D]")
    )
    np.testing.assert_array_equal(array("isin_index"), isins)
    wealth, valid = array("shareholder_wealth_close"), array("shareholder_wealth_valid")
    return manifest, array, wealth, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)

    def progress(stage, **detail):
        write_json_atomic(
            args.output / "progress.json",
            {"stage": stage, "seconds": time.perf_counter() - started, **detail},
        )
        print(json.dumps({"stage": stage, **detail}), flush=True)

    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    store = Path(pointer["store"]["root"])
    families = source_families(
        bound_json(
            {
                "path": str(store / "manifest.json"),
                "sha256": pointer["store"]["manifest_sha256"],
            }
        )
    )
    sessions = np.load(store / "date_index.npy").astype("datetime64[D]").tolist()
    isins = np.load(store / "isin_index.npy").tolist()
    assert sessions[-1] <= date(2024, 12, 31)
    active = np.load(store / "active.npy", mmap_mode="r")
    ti, ni = np.where(active)
    active_keys = pl.DataFrame(
        {"date": [sessions[t] for t in ti], "isin": [isins[n] for n in ni]}
    )
    admission = bound_json(binding(args.identity))
    identity = pl.read_parquet(verified(admission["artifacts"]["identity"]))
    old_identity = pl.read_parquet(verified(admission["baseline"]))
    sectors, issuers = identity_axes(identity, sessions, isins)
    old_sectors, old_issuers = identity_axes(old_identity, sessions, isins)
    changed = (sectors != old_sectors) | (issuers != old_issuers)
    affected = changed.copy()
    # Any changed issuer can alter all leave-one-issuer-out peers in its old/new sector.
    for t in np.flatnonzero(changed.any(axis=1)):
        groups = set(sectors[t, changed[t]]) | set(old_sectors[t, changed[t]])
        groups.discard("")
        affected[t] |= np.isin(sectors[t], list(groups)) | np.isin(
            old_sectors[t], list(groups)
        )
    t, n = np.where(affected)
    affected_keys = pl.DataFrame(
        {"date": [sessions[i] for i in t], "isin": [isins[i] for i in n]}
    )
    stats = {}

    def save(name, frame, unchanged_columns=None):
        old = pl.read_parquet(verified(families[name]["data"]))
        # Absent rows and explicit missing values are identical model coordinates.
        old_known = old.join(active_keys, on=KEYS, how="inner")
        new_known = frame.join(active_keys, on=KEYS, how="inner")
        joined = old_known.join(
            new_known, on=KEYS, how="full", coalesce=True, suffix="_new"
        )
        if unchanged_columns is None:
            joined = joined.join(affected_keys, on=KEYS, how="anti")
            fields = [c for c in old.columns if c not in KEYS]
        else:
            fields = unchanged_columns
        for c in fields:
            left, right = joined[c], joined[c + "_new"]
            if c.endswith("_age_sessions"):
                left, right = left.fill_null(-1), right.fill_null(-1)
            assert left.eq_missing(right).all(), (
                name,
                c,
                "unaffected coordinates changed",
            )
        frame.write_parquet(args.output / f"{name}.parquet")
        stats[name] = {
            "active_changes": compare(old, frame, active_keys),
            "unaffected_active_coordinates_exact": True,
        }
        progress(name, rows=len(frame))

    sector_manifest = bound_json(families["sector"]["source_manifest"])
    inputs = sector_manifest["sources"]["inputs"]
    manifest, array, wealth, wealth_valid = original_arrays(
        inputs["base_manifest"], sessions, isins
    )
    feature_names = manifest["feature_names"]["slow"]
    parent_valid = array("slow_valid")
    sector_values = np.zeros((*active.shape, 3))
    sector_mask = np.zeros_like(sector_values, bool)
    for j, h in enumerate((5, 21, 252)):
        values, mask = exact_log_return(
            wealth, h, shareholder_wealth_valid=wealth_valid
        )
        field = f"log_return_{h}"
        if h == 252:
            short, short_mask = exact_log_return(
                wealth, 21, shareholder_wealth_valid=wealth_valid
            )
            values -= short
            mask &= short_mask
            field = "momentum_12_1"
        sector_values[1:, :, j] = values[:-1]
        sector_mask[1:, :, j] = (
            mask[:-1] & parent_valid[1:, :, feature_names.index(field)]
        )
    values, mask = exp.sector_relative_panel(
        sector_values, sector_mask, sectors, issuers, array("active")
    )
    save(
        "sector",
        panel_frame(
            values,
            mask,
            np.where(mask, 1, -1),
            exp.SECTOR_FEATURE_NAMES,
            sessions,
            isins,
            active,
        ),
    )
    del sector_values, sector_mask, values, mask
    # Original non-oil factors; reuse sealed shocks and CDI without source reparsing.
    cross_manifest = bound_json(families["cross_market"]["source_manifest"])
    cross_inputs = cross_manifest["sources"]
    original_cross = (
        Path(cross_inputs["old_cross_market"]["path"]).parent / "manifest.json"
    )
    original_cross_manifest = json.loads(original_cross.read_text(encoding="utf8"))
    shocks = pl.read_parquet(verified(original_cross_manifest["shocks"]))
    cdi = pl.read_parquet(verified(original_cross_manifest["cdi"]))
    assert cdi["date"].to_list() == sessions
    excess, return_mask = exact_log_return(
        wealth, 1, shareholder_wealth_valid=wealth_valid
    )
    return_mask[:-1] &= parent_valid[1:, :, feature_names.index("log_return_1")]
    return_mask[-1] = False
    excess -= np.log1p(cdi["daily_cdi_rate"].to_numpy())[:, None]
    parts = []
    for name, factor in exp.EXPOSURES.items():
        if name == "oil":
            continue
        current, shock_age, historical, public = exp.shock_axes(
            shocks.filter(pl.col("factor") == factor), sessions
        )[factor]
        beta, mask, age = exp.exposure_panel(
            excess, return_mask, historical, public, sectors, issuers
        )
        vals, masks, ages = [beta], [mask], [age]
        names = [f"exposure_{name}"]
        for j, h in enumerate((1, 5)):
            valid = mask & np.isfinite(current[:, j, None])
            vals.append(np.where(valid, beta * current[:, j, None], 0))
            masks.append(valid)
            ages.append(np.where(valid, np.maximum(age, shock_age[:, j, None]), -1))
            names.append(f"exposure_{name}_times_shock_{h}")
        parts.append(
            panel_frame(
                np.stack(vals, -1),
                np.stack(masks, -1),
                np.stack(ages, -1),
                names,
                sessions,
                isins,
                active,
            )
        )
        progress("cross_market_factor", factor=factor)
    # Oil was introduced with a different, action-excluding original return contract.
    manifest, array, wealth, wealth_valid = original_arrays(
        cross_inputs["base_manifest"], sessions, isins
    )
    returns, valid = exact_log_return(wealth, 1, shareholder_wealth_valid=wealth_valid)
    valid[:-1] &= array("slow_valid")[
        1:, :, manifest["feature_names"]["slow"].index("log_return_1")
    ]
    valid[-1] = False
    valid = comparison_return_mask(valid, array("action_has_action"))
    levels = pl.read_parquet(verified(cross_inputs["market_levels"]))
    oil_shocks = exp.market_shocks(
        admit_brent(levels.filter(pl.col("series") == "brent_spot"), sessions),
        pl.read_parquet(verified(cross_inputs["us_returns"])).head(0),
    )
    current, shock_age, historical, public = exp.shock_axes(oil_shocks, sessions)["oil"]
    beta, mask, age = exp.exposure_panel(
        returns - np.log1p(cdi["daily_cdi_rate"].to_numpy())[:, None],
        valid,
        historical,
        public,
        sectors,
        issuers,
    )
    names = ["exposure_oil", "exposure_oil_times_shock_1", "exposure_oil_times_shock_5"]
    vals, masks, ages = [beta], [mask], [age]
    for j in range(2):
        known = mask & np.isfinite(current[:, j, None])
        vals.append(beta * current[:, j, None])
        masks.append(known)
        ages.append(np.maximum(age, shock_age[:, j, None]))
    parts.append(
        panel_frame(
            np.stack(vals, -1),
            np.stack(masks, -1),
            np.stack(ages, -1),
            names,
            sessions,
            isins,
            active,
        )
    )
    old = pl.read_parquet(verified(families["cross_market"]["data"]))
    replaced = [c for part in parts for c in part.columns if c not in KEYS]
    frame = old.drop(replaced)
    for part in parts:
        frame = frame.join(part, on=KEYS, how="full", coalesce=True)
    save("cross_market", frame.select(old.columns))
    # Utilization only: keep source balances/rates/flow fields and their clocks frozen.
    lending_manifest = bound_json(families["lending"]["source_manifest"])
    old_lending = Path(lending_manifest["sources"]["inputs"]["old_lending"]["path"])
    lending_inputs = bound_json(
        bound_json(binding(old_lending.parent / "manifest.json"))["source_bindings"]
    )
    source_files = lending_inputs["new_cvm_files"]
    changes = bound_json(source_files["capital_change_observations.json"])
    for change in changes:
        change["effective"] = date.fromisoformat(change["effective"])
    balances = pl.read_parquet(verified(lending_inputs["old_lending"]["balances"]))
    floating = pl.read_parquet(
        verified(source_files["free_float_observations.parquet"])
    )
    utilization, audit = round5_b3.lending_utilization_features(
        balances,
        floating,
        identity,
        sessions,
        round5_cvm.valuation_market(Path(inputs["base_manifest"]["path"]).parent),
        changes,
    )
    old = pl.read_parquet(verified(families["lending"]["data"]))
    fields = ["utilization_proxy", "utilization_proxy_age_sessions"]
    utilization = utilization.with_columns(
        [pl.col(c).cast(old.schema[c]) for c in fields]
    )
    frame = (
        old.drop(fields)
        .join(utilization, on=KEYS, how="full", coalesce=True)
        .select(old.columns)
    )
    save("lending", frame, [c for c in old.columns if c not in KEYS + fields])
    stats["lending"]["denominator_audit"] = audit
    write_json_atomic(
        args.output / "manifest.json",
        {
            "schema": "FCA_PEER_PROPAGATION_V1",
            "identity": binding(args.identity),
            "prior_families": {k: families[k]["source_manifest"] for k in stats},
            "old_cross_manifest": binding(original_cross),
            "lending_source_bindings": binding(
                Path(lending_inputs["driver"]["path"]).parent
                / "fca_dependencies_0aea7a8/inputs.json"
            ),
            "identity_changed_cells": int(changed.sum()),
            "potential_peer_affected_cells": int(affected.sum()),
            "families": stats,
            "artifacts": {p.stem: binding(p) for p in args.output.glob("*.parquet")},
            "source_code": {
                p.name: binding(p)
                for p in (
                    Path(__file__),
                    Path(exp.__file__),
                    Path(round5_cvm.__file__),
                    Path(round5_b3.__file__),
                )
            },
            "seconds": time.perf_counter() - started,
            "limits": [
                "Identity-only propagation on each original market-source contract; no accepted store or old fit changed.",
                "Independent financial/float denominator and revised wealth/history audits remain.",
                "Peer changes can affect names whose own identity never changed; direct unaffected sectors are checked exactly.",
                "Current single-vintage source uncertainties persist.",
            ],
        },
    )
    progress("complete")


if __name__ == "__main__":
    main()
