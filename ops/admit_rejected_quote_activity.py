"""Admit printed GOLL activity without admitting its inconsistent OHLC prices."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_foundation import panel_from_daily, prepare_cash_equities
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.features import _rolling_stat
from brazil_rv.v2.universe import build_daily_universe

PROJECT = Path(__file__).resolve().parents[1]


def raw_activity(volume, trades, valid):
    v = np.where(valid, volume, np.nan)
    t = np.where(valid, trades, np.nan)
    mean, ok = _rolling_stat(v, 20, "mean", minimum=20)
    std, _ = _rolling_stat(v, 20, "std", minimum=20)
    tmean, tok = _rolling_stat(t, 20, "mean", minimum=20)
    tstd, _ = _rolling_stat(t, 20, "std", minimum=20)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.stack(
            [np.log(mean), (v - mean) / std, (t - tmean) / tstd, v / mean], -1
        )
    masks = np.stack(
        [
            ok & (mean > 0),
            valid & ok & (std > 0),
            valid & tok & (tstd > 0),
            valid & ok & (mean > 0),
        ],
        -1,
    )
    return values, masks & np.isfinite(values)


def main():
    started = perf_counter()
    run = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text(encoding="utf8")
    )
    pointer = json.loads(
        (PROJECT / "docs/v2_data_inputs.json").read_text(encoding="utf8")
    )
    store = Path(pointer["store"]["root"])
    manifest = bound_json(
        {
            "path": str(store / "manifest.json"),
            "sha256": pointer["store"]["manifest_sha256"],
        }
    )
    output = Path(run["root"]) / "goll_activity_admission"
    output.mkdir(exist_ok=False)
    (output / "executed_reproducer.py").write_bytes(Path(__file__).read_bytes())
    dates, isins = (
        np.load(store / "date_index.npy"),
        np.load(store / "isin_index.npy").tolist(),
    )
    day, isin = np.datetime64("2011-02-16"), "BRGOLLACNPR4"
    ti, ni = int(np.searchsorted(dates, day)), isins.index(isin)
    source = next(
        r
        for r in manifest["sources"]
        if Path(r["path"]).name == "equities_daily_2011.parquet"
    )
    rows = pl.read_parquet(source["path"])
    row = rows.filter(
        (pl.col("trade_date") == day.astype(object)) & (pl.col("isin") == isin)
    )
    validation = prepare_cash_equities(row, v1_isins=[isin])
    panel = panel_from_daily(
        validation.accepted,
        dates=[day.astype(object)],
        isins=[isin],
        source_session_complete=[True],
        invalid_observations=validation.rejected,
    )
    assert (
        not panel.observed[0, 0]
        and panel.trade_observed[0, 0]
        and panel.activity_valid[0, 0]
    )
    assert np.isnan(panel.close_brl[0, 0])
    assert len(row) == 1
    row.write_parquet(output / "source_row.parquet")

    def old(name):
        return np.load(store / f"{name}.npy", mmap_mode="r")

    # Recover source precision only in the affected rolling interval. All other
    # activity masks, identities and membership remain the sealed contract.
    start, end = ti - 20, ti + 21
    volume = np.asarray(old("volume_brl")[start:end], np.float64).copy()
    trades = np.asarray(old("trade_count")[start:end], np.float64).copy()
    activity = np.array(old("activity_valid")[start:end])
    days = {d: i for i, d in enumerate(dates[start:end].astype(object))}
    names = {s: i for i, s in enumerate(isins)}
    good = prepare_cash_equities(
        rows.filter(
            pl.col("trade_date").is_between(
                dates[start].astype(object), dates[end - 1].astype(object)
            )
        ),
        v1_isins=isins,
    ).accepted
    for r in good.iter_rows(named=True):
        if r["isin"] in names:
            a, b = days[r["trade_date"]], names[r["isin"]]
            volume[a, b], trades[a, b] = r["volume_brl"], r["trades"]
    fields = [17, 18, 20, 21]
    specs = [
        FeatureSpec(**s)
        for s in manifest["metadata"]["feature_schema"]["specifications"]
        if s["family"] == "slow"
    ]
    selected = np.arange(ti + 1, end)

    def transformed(v, t):
        raw, mask = raw_activity(v, t, activity)
        values = np.empty((len(selected), len(isins), len(fields)), np.float32)
        valid = np.empty_like(values, bool)
        transform_feature_panel_into(
            raw,
            mask,
            old("active")[start:end],
            [specs[j] for j in fields],
            values,
            valid,
            source_rows=selected - start - 1,
            membership_rows=selected - start,
        )
        return values, valid

    control, control_mask = transformed(volume, trades)
    expected = old("slow_values")[selected][:, :, fields]
    np.testing.assert_array_equal(control, expected)
    np.testing.assert_array_equal(
        control_mask, old("slow_valid")[selected][:, :, fields]
    )
    volume[ti - start, ni] = panel.volume_brl[0, 0]
    trades[ti - start, ni] = panel.trades[0, 0]
    values, valid = transformed(volume, trades)
    np.testing.assert_array_equal(valid, control_mask)
    a, b, c = np.where(values != control)
    np.savez_compressed(
        output / "slow_patch.npz",
        date_index=selected[a],
        name_index=b,
        field_index=np.asarray(fields)[c],
        values=values[a, b, c],
        before=control[a, b, c],
    )
    v = np.array(old("volume_brl")[:, ni : ni + 1], np.float64)
    traded = np.array(old("trade_observed")[:, ni : ni + 1])
    v[ti, 0] = panel.volume_brl[0, 0]
    traded[ti, 0] = True
    universe = build_daily_universe(
        old("raw_close")[:, ni : ni + 1],
        v,
        old("observed")[:, ni : ni + 1],
        trade_observed=traded,
        activity_valid=old("activity_valid")[:, ni : ni + 1],
        source_session_complete=old("source_session_complete")[:, ni],
    )
    np.testing.assert_array_equal(
        universe.active[ti : ti + 21, 0], old("active")[ti : ti + 21, ni]
    )
    write_json_atomic(
        output / "manifest.json",
        {
            "status": "verified_activity_only_source_admission",
            "parent": pointer["store"],
            "source": source,
            "date_index": ti,
            "name_index": ni,
            "isin": isin,
            "date": str(day),
            "printed_activity": {
                "volume_brl": float(panel.volume_brl[0, 0]),
                "trade_count": int(panel.trades[0, 0]),
                "quantity": int(panel.quantity[0, 0]),
                "trade_observed": True,
            },
            "price_observed": False,
            "raw_prices_unchanged": True,
            "activity_valid_unchanged": True,
            "eligibility_unchanged": True,
            "slow_masks_ages_unchanged": True,
            "matched_control_cells": int(control.size),
            "changed_slow_values": len(a),
            "changed_names": len(np.unique(b)),
            "first_affected_decision": str(dates[selected[0]]),
            "last_affected_decision": str(dates[selected[-1]]),
            "fields": {
                specs[j].name: int((np.asarray(fields)[c] == j).sum()) for j in fields
            },
            "artifacts": {
                p.stem: binding(p)
                for p in output.iterdir()
                if p.suffix in (".npz", ".parquet")
            },
            "code": binding(Path(__file__)),
            "seconds": perf_counter() - started,
            "limits": [
                "Sparse activity/slow patch for the next derived store, not old-checkpoint inputs.",
                "The inconsistent printed prices remain rejected; no OHLC correction inferred.",
                "Quantity-dependent families and full-store assembly must explicitly consume this activity amendment.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "changed_slow_values": len(a),
                "seconds": perf_counter() - started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
