"""Bound Natura wealth effects in existing sector, magnitude and factor families."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2 import round5_exposures as exp
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.bova11 import load_bova11_series
from brazil_rv.v2.cross_market_returns import admit_brent, comparison_return_mask
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.hedge_beta import rolling_hedge_beta
from brazil_rv.v2.round5_derived import beta_source_age, identity_axes
from brazil_rv.v2.round5_magnitude import magnitude_panel, FEATURE_NAMES
from brazil_rv.v2.round5_store import align_family

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    daily = bound_json(run["natura_daily_propagation"])
    plan = bound_json(daily["plan"])
    parent = plan["parent_store"]
    store = Path(parent["root"])
    manifest = bound_json(
        {"path": str(store / "manifest.json"), "sha256": parent["manifest_sha256"]}
    )
    foundation = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    original = bound_json(
        {
            "path": str(Path(foundation["root"]) / "manifest.json"),
            "sha256": foundation["manifest_sha256"],
        }
    )
    families = source_families(original)
    identity = bound_json(run["rename_issuer_propagation"])["artifacts"]["identity"]
    prior = bound_json(run["activity_magnitude_propagation"])["artifacts"]
    prior["sector"] = bound_json(run["rename_dependents"])["artifacts"]["sector"]
    out = Path(run["root"]) / "natura_auxiliary"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    dates = np.load(store / "date_index.npy")
    isins = np.load(store / "isin_index.npy").tolist()
    axis = isins.index("BRNATUACNOR6")
    start, first, stop = plan["rows"]
    window = slice(start, stop)
    selected = np.arange(first - start, stop - start)
    days = dates[window].astype(object).tolist()
    sessions = dates.astype(object).tolist()
    active = np.load(store / "active.npy", mmap_mode="r")[window]
    slow_valid = np.load(store / "slow_valid.npy", mmap_mode="r")[window]
    slow_age = np.load(store / "slow_age_sessions.npy", mmap_mode="r")[window]
    slow_names = manifest["feature_names"]["slow"]
    sectors, issuers = identity_axes(
        pl.read_parquet(identity["path"]).filter(
            pl.col("date").is_between(days[0], days[-1])
        ),
        days,
        isins,
    )
    amendment = bound_json(run["natura_wealth_amendment"])
    with np.load(amendment["deltas"]["path"]) as z:
        wealth_patch = {k: z[k].copy() for k in z.files}

    def base(record):
        m = bound_json(record)
        root = Path(record["path"]).parent

        def a(key):
            return np.load(root / m["arrays"][key]["path"], mmap_mode="r")[window]

        return a

    def amend(value, field):
        a = np.array(value, copy=True)
        ix = wealth_patch["shareholder_wealth_" + field + "__indices"].copy()
        # Prove this family's original wealth coordinate agrees at every input
        # changed by the source correction before applying that sparse layer.
        baseline = np.load(
            store / ("shareholder_wealth_" + field + ".npy"), mmap_mode="r"
        )
        np.testing.assert_array_equal(
            a[ix[:, 0] - start, ix[:, 1]], baseline[tuple(ix.T)]
        )
        ix[:, 0] -= start
        a[tuple(ix.T)] = wealth_patch["shareholder_wealth_" + field + "__values"]
        return a

    reports = {}

    def save(family, names, controls, corrected, columns=None):
        frame = pl.read_parquet(prior[family]["path"]).filter(
            pl.col("date").is_between(days[selected[0]], days[-1])
        )
        original_values, original_valid, original_age = align_family(
            frame, [days[t] for t in selected], tuple(isins), tuple(names)
        )
        if columns is None:
            columns = np.arange(len(isins))
        baseline = (
            original_values[:, columns],
            original_valid[:, columns],
            original_age[:, columns],
        )
        checks = []
        for j, kind in enumerate(("values", "valid", "age")):
            membership = active[selected][:, columns, None]
            control = controls[j][selected].copy()
            after = corrected[j][selected].copy()
            if kind == "valid":
                control &= membership
                after &= membership
            else:
                cmask = controls[1][selected] & membership
                nmask = corrected[1][selected] & membership
                control = np.where(cmask, control, 0 if kind == "values" else -1)
                after = np.where(nmask, after, 0 if kind == "values" else -1)
                control = control.astype(np.float32)
                after = after.astype(np.float32)
            expected = baseline[j]
            if kind == "age":
                expected = np.where(baseline[1], expected, -1)
            same = (control == expected) | (np.isnan(control) & np.isnan(expected))
            bad = np.argwhere(~same)
            checks.append(
                {
                    "kind": kind,
                    "cells": control.size,
                    "mismatches": len(bad),
                    "examples": [
                        {
                            "date": str(dates[first + ix[0]]),
                            "isin": isins[columns[ix[1]]],
                            "field": names[ix[2]],
                            "actual": float(control[tuple(ix)]),
                            "expected": float(expected[tuple(ix)]),
                        }
                        for ix in bad[:5]
                    ],
                }
            )
            np.save(out / f"{family}_{kind}.npy", after)
            np.save(out / f"{family}_control_{kind}.npy", control)
        reports[family] = {
            "source": prior[family],
            "names": list(names),
            "columns": np.asarray(columns).tolist(),
            "checks": checks,
        }
        write_json_atomic(out / (family + "_control.json"), reports[family])
        print(
            json.dumps(
                {
                    "family": family,
                    "checks": checks,
                    "seconds": perf_counter() - started,
                }
            ),
            flush=True,
        )

    sm = bound_json(families["sector"]["source_manifest"])
    source = base(sm["sources"]["inputs"]["base_manifest"])
    w, seen, support = (
        source("shareholder_wealth_close"),
        source("shareholder_wealth_valid"),
        source("slow_valid"),
    )
    corrected_w = amend(w, "close")
    sector_pairs = []
    for wealth in (w, corrected_w):
        values = np.zeros((*active.shape, 3))
        valid = np.zeros_like(values, bool)
        for j, h in enumerate((5, 21, 252)):
            raw, known = exact_log_return(wealth, h, shareholder_wealth_valid=seen)
            field = f"log_return_{h}"
            if h == 252:
                short, sknown = exact_log_return(
                    wealth, 21, shareholder_wealth_valid=seen
                )
                raw -= short
                known &= sknown
                field = "momentum_12_1"
            values[1:, :, j] = raw[:-1]
            valid[1:, :, j] = known[:-1] & support[1:, :, slow_names.index(field)]
        vals, valid = exp.sector_relative_panel(values, valid, sectors, issuers, active)
        sector_pairs.append((vals, valid, np.where(valid, 1, -1)))
    save("sector", exp.SECTOR_FEATURE_NAMES, *sector_pairs)

    mm = bound_json(families["magnitudes"]["source_manifest"])["sources"]["inputs"]
    bova = load_bova11_series(
        Path(mm["bova_manifest"]["path"]).parent,
        expected_manifest_sha256=mm["bova_manifest"]["sha256"],
        canonical_dates=sessions,
    ).close_by_session[window]
    history = bound_json(run["rename_history_propagation"])["history_basis"]
    cols = [axis, (axis + 1) % len(isins), (axis + 2) % len(isins)]
    raw = {
        k: np.array(
            np.load(history["shareholder_wealth_" + k]["path"], mmap_mode="r")[window]
        )
        for k in ("open", "high", "low", "close", "valid")
    }
    magnitude_pairs = []
    for changed in (False, True):
        wealth = {
            k: (amend(a, k) if changed and k != "valid" else a)[:, cols]
            for k, a in raw.items()
        }
        beta, known = rolling_hedge_beta(wealth["close"], wealth["valid"], bova)
        age = beta_source_age(wealth["close"], wealth["valid"], bova, known)
        result = magnitude_panel(
            **{"wealth_" + k: a for k, a in wealth.items()},
            volume_brl=np.ascontiguousarray(
                np.load(store / "volume_brl.npy", mmap_mode="r")[window][:, cols]
            ),
            activity_valid=np.load(store / "activity_valid.npy", mmap_mode="r")[window][
                :, cols
            ],
            daily_feature_valid=slow_valid[:, cols],
            slow_age_sessions=slow_age[:, cols],
            slow_feature_names=tuple(slow_names),
            economic_beta=beta,
            economic_beta_valid=known,
            economic_beta_age_sessions=age,
        )
        magnitude_pairs.append(tuple(a[:, :1] for a in result))
    save("magnitudes", FEATURE_NAMES, *magnitude_pairs, columns=[axis])

    cm = bound_json(families["cross_market"]["source_manifest"])["sources"]
    original_cross = bound_json(
        binding(Path(cm["old_cross_market"]["path"]).parent / "manifest.json")
    )
    shocks = pl.read_parquet(original_cross["shocks"]["path"])
    cdi = pl.read_parquet(original_cross["cdi"]["path"])["daily_cdi_rate"].to_numpy()[
        window
    ]
    levels = pl.read_parquet(cm["market_levels"]["path"])
    oil = exp.market_shocks(
        admit_brent(levels.filter(pl.col("series") == "brent_spot"), sessions),
        pl.read_parquet(cm["us_returns"]["path"]).head(0),
    )
    shock_axes = exp.shock_axes(shocks, sessions)
    shock_axes["oil"] = exp.shock_axes(oil, sessions)["oil"]
    returns = []
    for kind in ("ordinary", "oil"):
        a = source if kind == "ordinary" else base(cm["base_manifest"])
        wealth, seen, supported = (
            a("shareholder_wealth_close"),
            a("shareholder_wealth_valid"),
            a("slow_valid"),
        )
        pair = []
        for changed in (False, True):
            ret, known = exact_log_return(
                amend(wealth, "close") if changed else wealth,
                1,
                shareholder_wealth_valid=seen,
            )
            known[:-1] &= supported[1:, :, slow_names.index("log_return_1")]
            known[-1] = False
            if kind == "oil":
                known = comparison_return_mask(known, a("action_has_action"))
            pair.append((ret - np.log1p(cdi)[:, None], known))
        returns.append(pair)
    all_values = [[], []]
    names = []
    for name, factor in exp.EXPOSURES.items():
        current, shock_age, historical, public = shock_axes[factor]
        names.extend(
            [f"exposure_{name}", *[f"exposure_{name}_times_shock_{h}" for h in (1, 5)]]
        )
        for scenario, (ret, known) in enumerate(returns[name == "oil"]):
            beta, mask, age = exp.exposure_panel(
                ret, known, historical[window], public[window] - start, sectors, issuers
            )
            all_values[scenario].append(
                (
                    np.stack(
                        [beta, *[beta * current[window, j, None] for j in range(2)]], -1
                    ),
                    np.stack(
                        [
                            mask,
                            *[
                                mask & np.isfinite(current[window, j, None])
                                for j in range(2)
                            ],
                        ],
                        -1,
                    ),
                    np.stack(
                        [
                            age,
                            *[
                                np.maximum(age, shock_age[window, j, None])
                                for j in range(2)
                            ],
                        ],
                        -1,
                    ),
                )
            )
    combined = [
        tuple(np.concatenate([p[j] for p in scenario], axis=-1) for j in range(3))
        for scenario in all_values
    ]
    save("cross_market", names, *combined)
    np.save(out / "rows.npy", np.arange(first, stop))
    report = {
        "status": "produced_pending_control_qualification",
        "parent_store": parent,
        "daily": run["natura_daily_input_audit"],
        "source_families": {
            k: families[k] for k in ("sector", "magnitudes", "cross_market")
        },
        "identity": identity,
        "families": reports,
        "rows": binding(out / "rows.npy"),
        "seconds": perf_counter() - started,
    }
    write_json_atomic(out / "manifest.json", report)
    run["natura_auxiliary_propagation"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps({"status": report["status"], "seconds": report["seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
