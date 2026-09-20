"""Propagate admitted rename history into sector, magnitude and market families.

Only the changed interval is calculated; all earlier family observations remain
byte-for-byte values from their verified parent. Original oil and non-oil market
contracts stay distinct. No learned checkpoint is read or scored.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from brazil_rv.v2 import round5_exposures as exp
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.bova11 import load_bova11_series
from brazil_rv.v2.cross_market_returns import admit_brent, comparison_return_mask
from brazil_rv.v2.data_repair import binding, bound_json, source_families
from brazil_rv.v2.features import exact_log_return
from brazil_rv.v2.hedge_beta import rolling_hedge_beta
from brazil_rv.v2.round5_derived import identity_axes, beta_source_age, verified
from brazil_rv.v2.round5_magnitude import magnitude_panel, FEATURE_NAMES
from propagate_fca_financials import compare
from propagate_fca_peers import panel_frame

PROJECT = Path(__file__).resolve().parents[1]
KEYS = ["date", "isin"]


def main():
    begun = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    source = Path(pointer["store"]["root"])
    manifest = bound_json(
        {
            "path": str(source / "manifest.json"),
            "sha256": pointer["store"]["manifest_sha256"],
        }
    )
    families = source_families(manifest)
    rename = bound_json(run["rename_history_propagation"])
    peers = bound_json(run["rename_peer_propagation"])
    previous = bound_json(run["fca_peer_propagation"])
    issuer = bound_json(run["rename_issuer_propagation"])
    output = Path(run["root"]) / "rename_dependents"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    axis = np.load(source / "date_index.npy")
    assert axis[-1] <= np.datetime64("2024-12-31")
    sessions, isins = (
        axis.astype(object).tolist(),
        np.load(source / "isin_index.npy").tolist(),
    )
    links = pl.read_parquet(verified(rename["history_mapping"])).to_dicts()
    first = min(r["effective_index"] for r in links)
    start = first - 253
    days = sessions[start:]
    offset = first - start
    active_full = np.load(verified(rename["arrays"]["active"]), mmap_mode="r")
    active = active_full[start:]
    slow_valid = np.load(verified(peers["arrays"]["slow_valid"]), mmap_mode="r")[start:]
    slow_age = np.load(verified(peers["arrays"]["slow_age_sessions"]), mmap_mode="r")[
        start:
    ]
    slow_names = manifest["feature_names"]["slow"]
    ti, ni = np.where(active_full)
    keys = pl.DataFrame({"date": pl.Series(axis[ti]), "isin": np.asarray(isins)[ni]})
    sectors, issuers = identity_axes(
        pl.read_parquet(verified(issuer["artifacts"]["identity"])).filter(
            pl.col("date") >= days[0]
        ),
        days,
        isins,
    )
    basis = {
        k: np.load(verified(v), mmap_mode="r")[start:]
        for k, v in rename["history_basis"].items()
    }
    statistics = {}

    def parent_frame(name):
        record = previous["artifacts"].get(name, families[name]["data"])
        return pl.read_parquet(verified(record))

    def save(name, frame, old):
        frame = frame.sort(KEYS)
        assert not frame.select(pl.struct(KEYS).is_duplicated().any()).item()
        assert_frame_equal(
            frame.filter(pl.col("date") < sessions[first]),
            old.filter(pl.col("date") < sessions[first]).sort(KEYS),
            check_exact=True,
        )
        frame.write_parquet(output / f"{name}.parquet")
        statistics[name] = compare(old, frame, keys)
        print(
            json.dumps(
                {"stage": name, "rows": len(frame), "seconds": perf_counter() - begun}
            ),
            flush=True,
        )

    def market_arrays(record):
        m = bound_json(record)
        root = Path(record["path"]).parent

        def array(name):
            return np.load(root / f"{name}.npy", mmap_mode="r")[start:]

        wealth = np.array(array("shareholder_wealth_close"))
        valid = np.array(array("shareholder_wealth_valid"))
        support = np.array(array("slow_valid"))
        for link in links:
            b = link["successor_index"]
            a = link["predecessor_index"]
            boundary = link["effective_index"] - start
            wealth[:, b] = basis["shareholder_wealth_close"][:, b]
            valid[:, b] = basis["shareholder_wealth_valid"][:, b]
            support[:boundary, b] = support[:boundary, a]
            support[boundary:, b] = slow_valid[boundary:, b]
        return m, array, wealth, valid, support

    sector_manifest = bound_json(families["sector"]["source_manifest"])
    base = sector_manifest["sources"]["inputs"]["base_manifest"]
    _, _, wealth, seen, support = market_arrays(base)
    sector_values = np.zeros((*active.shape, 3))
    sector_valid = np.zeros_like(sector_values, bool)
    for j, h in enumerate([5, 21, 252]):
        raw, mask = exact_log_return(wealth, h, shareholder_wealth_valid=seen)
        field = f"log_return_{h}"
        if h == 252:
            shorter, smask = exact_log_return(wealth, 21, shareholder_wealth_valid=seen)
            raw -= shorter
            mask &= smask
            field = "momentum_12_1"
        sector_values[1:, :, j] = raw[:-1]
        sector_valid[1:, :, j] = mask[:-1] & support[1:, :, slow_names.index(field)]
    vals, mask = exp.sector_relative_panel(
        sector_values[offset:],
        sector_valid[offset:],
        sectors[offset:],
        issuers[offset:],
        active[offset:],
    )
    old = parent_frame("sector")
    tail = panel_frame(
        vals,
        mask,
        np.where(mask, 1, -1),
        exp.SECTOR_FEATURE_NAMES,
        days[offset:],
        isins,
        active[offset:],
    )
    save("sector", pl.concat([old.filter(pl.col("date") < sessions[first]), tail]), old)
    # Four physical-scale fields need only the three successor columns.
    mag_manifest = bound_json(families["magnitudes"]["source_manifest"])
    inputs = mag_manifest["sources"]["inputs"]
    bova = load_bova11_series(
        Path(inputs["bova_manifest"]["path"]).parent,
        expected_manifest_sha256=inputs["bova_manifest"]["sha256"],
        canonical_dates=sessions,
    ).close_by_session[start:]
    selected = [r["successor_index"] for r in links]
    volume = np.array(np.load(source / "volume_brl.npy", mmap_mode="r")[start:, :])
    activity = np.array(np.load(source / "activity_valid.npy", mmap_mode="r")[start:])
    for link in links:
        boundary = link["effective_index"] - start
        a, b = link["predecessor_index"], link["successor_index"]
        volume[:boundary, b] = volume[:boundary, a]
        activity[:boundary, b] = activity[:boundary, a]
    beta, bmask = rolling_hedge_beta(
        basis["shareholder_wealth_close"][:, selected],
        basis["shareholder_wealth_valid"][:, selected],
        bova,
    )
    bage = beta_source_age(
        basis["shareholder_wealth_close"][:, selected],
        basis["shareholder_wealth_valid"][:, selected],
        bova,
        bmask,
    )
    vals, mask, age = magnitude_panel(
        **{
            f"wealth_{k}": basis[f"shareholder_wealth_{k}"][:, selected]
            for k in ["open", "high", "low", "close", "valid"]
        },
        volume_brl=np.ascontiguousarray(volume[:, selected]),
        activity_valid=activity[:, selected],
        daily_feature_valid=slow_valid[:, selected],
        slow_age_sessions=slow_age[:, selected],
        slow_feature_names=tuple(slow_names),
        economic_beta=beta,
        economic_beta_valid=bmask,
        economic_beta_age_sessions=bage,
    )
    dates_by_name = {
        isins[r["successor_index"]]: sessions[
            max(r["known_index"], r["effective_index"])
        ]
        for r in links
    }
    mag = panel_frame(
        vals,
        mask,
        age,
        FEATURE_NAMES,
        days,
        [isins[i] for i in selected],
        active[:, selected],
    ).filter(pl.col("date") >= sessions[first])
    old_mag = parent_frame("magnitudes")
    replaced = old_mag.filter(
        pl.any_horizontal(
            [
                (pl.col("isin") == n) & (pl.col("date") >= d)
                for n, d in dates_by_name.items()
            ]
        )
    )
    mag = pl.concat([old_mag.join(replaced.select(KEYS), on=KEYS, how="anti"), mag])
    save("magnitudes", mag, old_mag)
    # Preserve original non-oil and action-excluding oil return contracts.
    cross_manifest = bound_json(families["cross_market"]["source_manifest"])
    cross_inputs = cross_manifest["sources"]
    original = bound_json(
        binding(Path(cross_inputs["old_cross_market"]["path"]).parent / "manifest.json")
    )
    shocks = pl.read_parquet(verified(original["shocks"]))
    cdi = pl.read_parquet(verified(original["cdi"]))["daily_cdi_rate"].to_numpy()[
        start:
    ]
    levels = pl.read_parquet(verified(cross_inputs["market_levels"]))
    oil = exp.market_shocks(
        admit_brent(levels.filter(pl.col("series") == "brent_spot"), sessions),
        pl.read_parquet(verified(cross_inputs["us_returns"])).head(0),
    )
    axes = exp.shock_axes(shocks, sessions)
    axes["oil"] = exp.shock_axes(oil, sessions)["oil"]
    old = parent_frame("cross_market")
    columns = [c for c in old.columns if c.startswith("exposure_")]
    parts = []
    for name, factor in exp.EXPOSURES.items():
        if name == "oil":
            _, arr, w, v, s = market_arrays(cross_inputs["base_manifest"])
        else:
            w, v, s = wealth, seen, support
        returns, rvalid = exact_log_return(w, 1, shareholder_wealth_valid=v)
        rvalid[:-1] &= s[1:, :, slow_names.index("log_return_1")]
        rvalid[-1] = False
        if name == "oil":
            actions = np.array(arr("action_has_action"))
            for link in links:
                boundary = link["effective_index"] - start
                actions[:boundary, link["successor_index"]] = actions[
                    :boundary, link["predecessor_index"]
                ]
            rvalid = comparison_return_mask(rvalid, actions)
        current, shock_age, historical, public = axes[factor]
        beta, known, age = exp.exposure_panel(
            returns - np.log1p(cdi)[:, None],
            rvalid,
            historical[start:],
            public[start:] - start,
            sectors,
            issuers,
        )
        values, masks, ages = [beta], [known], [age]
        names = [f"exposure_{name}"]
        for j, h in enumerate([1, 5]):
            masks.append(known & np.isfinite(current[start:, j, None]))
            values.append(beta * current[start:, j, None])
            ages.append(np.maximum(age, shock_age[start:, j, None]))
            names.append(f"exposure_{name}_times_shock_{h}")
        parts.append(
            panel_frame(
                np.stack(values, -1)[offset:],
                np.stack(masks, -1)[offset:],
                np.stack(ages, -1)[offset:],
                names,
                days[offset:],
                isins,
                active[offset:],
            )
        )
    tail = old.filter(pl.col("date") >= sessions[first]).drop(columns)
    for part in parts:
        tail = tail.join(part, on=KEYS, how="full", coalesce=True)
    # Common market measurements do not require a security's 60-day history.
    # Verify their exact equality across old names before filling new rows.
    common = [
        c
        for c in old.columns
        if c.startswith("shock_")
        or c.startswith("ewz_minus_bova11_")
        or c
        in [
            f"foreign_flow_{s}{a}"
            for s in ["1", "5", "month_reset", "methodology_change"]
            for a in ["", "_age_sessions"]
        ]
    ]
    group = (
        old.filter(pl.col("date") >= sessions[first]).select("date", *common).unique()
    )
    assert group["date"].n_unique() == len(group)
    tail = tail.drop(common).join(group, on="date", how="left")
    admitted = bound_json(cross_inputs["adr_identity"])
    assert not any(isin in json.dumps(admitted["pairs"]) for isin in dates_by_name)
    successor = pl.col("isin").is_in(list(dates_by_name))
    tail = tail.with_columns(
        pl.when(successor)
        .then(0.0)
        .otherwise(pl.col("adr_listed_flag"))
        .cast(pl.Float32)
        .alias("adr_listed_flag"),
        pl.when(successor)
        .then(1.0)
        .otherwise(pl.col("adr_listed_flag_age_sessions"))
        .cast(pl.Float32)
        .alias("adr_listed_flag_age_sessions"),
    )
    # The original flow interaction consumes stored float32 magnitude and a
    # source float64 flow. Reuse the independently dated public flow ledger.
    from brazil_rv.v2.foreign_flow import decision_panel

    observations = []
    flow_manifest = bound_json(cross_inputs["foreign_flow"])
    from datetime import date

    for record in flow_manifest["results"]:
        if "observation" in record:
            row = dict(record["observation"])
            for key in ["reference_date", "publication_date"]:
                row[key] = date.fromisoformat(row[key])
            observations.append(row)
    fn, fv, fm, fa, _ = decision_panel(observations, sessions)
    flow = pl.DataFrame(
        {
            "date": sessions,
            **{
                n: pl.Series(fv[:, j]).set(pl.Series(~fm[:, j]), None)
                for j, n in enumerate(fn)
            },
        }
    )
    # Replace only successors' interactions; all other names retain their
    # earlier data contract (the GOLL update is separately attributed below).
    extra = (
        tail.filter(successor)
        .join(
            mag.select(
                KEYS + ["log_traded_value_20", "log_traded_value_20_age_sessions"]
            ),
            on=KEYS,
            how="left",
        )
        .join(
            flow.select("date", "foreign_flow_1", "foreign_flow_5"),
            on="date",
            suffix="_source",
        )
    )
    for h in [1, 5]:
        base = f"foreign_flow_{h}"
        extra = extra.with_columns(
            (pl.col(base + "_source") * pl.col("log_traded_value_20"))
            .cast(pl.Float32)
            .alias(base + "_times_log_volume_mean_20"),
            pl.max_horizontal(
                base + "_age_sessions", "log_traded_value_20_age_sessions"
            )
            .cast(pl.Float32)
            .alias(base + "_times_log_volume_mean_20_age_sessions"),
            (pl.col(base + "_source") * 0)
            .cast(pl.Float32)
            .alias(base + "_times_adr_listed_flag"),
            pl.max_horizontal(base + "_age_sessions", pl.lit(1))
            .cast(pl.Float32)
            .alias(base + "_times_adr_listed_flag_age_sessions"),
        )
    tail = pl.concat(
        [tail.filter(~successor).select(old.columns), extra.select(old.columns)]
    )
    save(
        "cross_market",
        pl.concat(
            [old.filter(pl.col("date") < sessions[first]), tail.select(old.columns)]
        ),
        old,
    )
    write_json_atomic(
        output / "manifest.json",
        {
            "status": "rename_sector_magnitude_market_intermediate",
            "inputs": {
                k: run[k]
                for k in [
                    "rename_history_propagation",
                    "rename_peer_propagation",
                    "rename_issuer_propagation",
                    "fca_peer_propagation",
                    "goll_activity_admission",
                ]
            },
            "families": statistics,
            "artifacts": {p.stem: binding(p) for p in output.glob("*.parquet")},
            "code": binding(Path(__file__)),
            "seconds": perf_counter() - begun,
            "limits": [
                "No accepted model store or saved checkpoint coordinates changed.",
                "Recovered lending, M1/other auxiliary joins and remaining contractual wealth/labels still require separate propagation.",
                "GOLL magnitude and flow interaction remain separate from this rename layer.",
            ],
        },
    )


if __name__ == "__main__":
    main()
