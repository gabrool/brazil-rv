"""Independent typed arithmetic and actual consumers for the last four families."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl
from scipy.special import ndtri

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.decision_clock import load_session_schedule
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    transform_feature_panel_into,
)
from brazil_rv.v2.intraday_features import build_native_fast_features
from brazil_rv.v2.store import StoreStaging
from propagate_surviving_m1 import grid_for, minute

PROJECT = Path(__file__).resolve().parents[1]


def rank(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - counts + (counts - 1) / 2)[inverse]


def independent(raw, valid, active, specs):
    values = np.zeros(raw.shape, np.float32)
    known = np.zeros(raw.shape, bool)
    for t in range(len(raw)):
        for j, spec in enumerate(specs):
            x = raw[t, :, j].astype(float)
            ok = valid[t, :, j] & active[t] & np.isfinite(x)
            kind = spec["transform"]
            if kind == "rank_gauss":
                if ok.sum() < spec["minimum_support"]:
                    ok[:] = False
                else:
                    values[t, ok, j] = np.clip(
                        ndtri((rank(x[ok]) + 0.5) / ok.sum()), -3, 3
                    )
            else:
                if kind == "binary":
                    ok &= (x == 0) | (x == 1)
                elif kind == "bounded_fraction":
                    ok &= (x >= 0) & (x <= 1)
                    x = 2 * x - 1
                elif kind == "signed_identity":
                    ok &= (x >= -1) & (x <= 1)
                elif kind == "age_sessions":
                    ok &= x >= 0
                    a = np.log1p(np.maximum(x, 0))
                    x = a / (a + np.log1p(252.0))
                elif kind == "signed_clip":
                    x = np.clip(x, -spec["clip"], spec["clip"])
                elif kind == "signed_asinh":
                    x = np.arcsinh(x)
                elif kind == "annual_rate":
                    x = np.arcsinh(x / 0.01)
                else:
                    assert kind == "precomputed_native", kind
                values[t, ok, j] = x[ok]
            known[t, :, j] = ok
    return values, known


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    m1 = bound_json(run["surviving_rename_m1"])
    market = bound_json(run["surviving_rename_market"])
    source = Path(run["surviving_rename_m1"]["path"]).parent
    out = source.parent / "market_consumer"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    root = Path(m1["parent"]["root"])
    m = bound_json(
        dict(path=str(root / "manifest.json"), sha256=m1["parent"]["manifest_sha256"])
    )
    prior_view = bound_json(run["surviving_rename_daily_input_audit"])["view"]
    vm = bound_json(prior_view)
    vr = Path(prior_view["path"]).parent
    dates = np.load(root / "date_index.npy")
    isins = np.load(root / "isin_index.npy").tolist()
    view_dates = np.load(vr / "date_index.npy")
    begin = int(np.searchsorted(dates, view_dates[0]))
    active = np.load(vr / "active.npy", mmap_mode="r")
    selected_families = [
        "slow",
        "native_fast",
        "intraday",
        "sidecar_magnitudes",
        "sidecar_cross_market",
    ]
    specs = [
        s
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] in selected_families
    ]
    deltas = {}
    for rec in [m1["native_deltas"], m1["scalar_deltas"], market["deltas"]]:
        with np.load(rec["path"]) as z:
            for key in z.files:
                if not key.endswith("__indices"):
                    continue
                field = key.split("__")[-2]
                deltas.setdefault(field, []).append(
                    (z[key], z[key.replace("__indices", "__values")])
                )
    write_json_atomic(
        out / "plan.json",
        dict(
            m1=run["surviving_rename_m1"],
            market=run["surviving_rename_market"],
            prior_slow_view=prior_view,
            parent=m1["parent"],
            contract="Only native/scalar/magnitude/cross-market and to-close checks. Reuse qualified slow history and other sidecars without recounting them. All933/full60 actual dataset and collator; independent typed formulas, own-source ages, target return/clock/rank and post-decision native mutation. No complete-store/model admission.",
        ),
    )
    expected = {}
    keys = [
        "fast_patch_values",
        "fast_patch_valid",
        "fast_patch_mask",
        "fast_last_price_age_minutes",
        "fast_last_price_age_valid",
        "fast_present",
        "intraday_values",
        "intraday_valid",
        "intraday_age_sessions",
        "intraday_support_fraction",
        *[
            "target_to_close" + s
            for s in ["", "_valid", "_normalized_residual", "_raw_log_return"]
        ],
        *[
            "sidecar_" + g + "_" + s
            for g in ["magnitudes", "cross_market"]
            for s in ["values", "valid", "age_sessions"]
        ],
    ]
    view = out / "view"
    with StoreStaging(view, dates=view_dates, isins=isins) as stage:
        for key in [
            "active",
            "slow_values",
            "slow_valid",
            "slow_age_sessions",
            "slow_timestep_valid",
        ]:
            stage.copy_array(key, vr / vm["arrays"][key]["path"])
        for key in keys:
            a = np.load(root / (key + ".npy"), mmap_mode="r")[begin:].copy()
            for ix, values in deltas.get(key, []):
                local = ix.copy()
                local[:, 0] -= begin
                assert not np.any(local[:, 0] < 0)
                a[tuple(local.T)] = values
            stage.write_array(key, a)
        stage.seal(
            feature_names={k: m["feature_names"][k] for k in selected_families},
            sources=[binding(out / "plan.json")],
            tables={
                "slow_history_links": vr / vm["tables"]["slow_history_links"]["path"],
                "native_fast_security_mapping": root
                / m["tables"]["native_fast_security_mapping"]["path"],
            },
            metadata={
                "purpose": "Last-four-family audit view only, not complete accepted store",
                "feature_schema": {
                    "minimum_rank_names": 20,
                    "specifications": specs,
                    "sha256": feature_schema_sha256([FeatureSpec(**s) for s in specs]),
                },
            },
        )
    expected = {k: np.load(view / (k + ".npy"), mmap_mode="r") for k in keys}
    formula_cells = 0
    with np.load(m1["scalar_raw"]["path"]) as z:
        raw = {k: z[k].copy() for k in z.files}
    with np.load(m1["scalar_output"]["path"]) as z:
        rows = z["date_indices"].copy()
        outputs = {k: z[k].copy() for k in z.files if k != "date_indices"}
    take = rows - raw["date_indices"][0]
    scalar_specs = [s for s in specs if s["family"] == "intraday"]
    values, valid = independent(
        raw["values"][take], raw["valid"][take], active[rows - begin], scalar_specs
    )
    np.testing.assert_array_equal(values, outputs["intraday_values"])
    np.testing.assert_array_equal(valid, outputs["intraday_valid"])
    formula_cells += 2 * values.size
    # Independent own-version source clock: never refresh from retired ARZZ.
    azza = isins.index("BRAZZAACNOR9")
    last = np.full(20, -1)
    for t in range(len(raw["date_indices"])):
        known = raw["valid"][t, azza]
        last[known] = t - raw["source_age_sessions"][t, azza, known].astype(int)
        if t in take:
            i = int(np.searchsorted(take, t))
            age = np.where(last >= 0, t - last, -1).astype(np.float32)
            if not active[rows[i] - begin, azza]:
                age[:] = -1
            np.testing.assert_array_equal(
                age, outputs["intraday_age_sessions"][i, azza]
            )
            formula_cells += 20
    schedule = {
        s.trade_date: s
        for s in load_session_schedule(
            root / m["tables"]["b3_session_schedule"]["path"]
        )
    }
    target_count = 0
    for i, t in enumerate(take):
        entry = raw["entry"][t].astype(float)
        close = raw["session_close"][t].astype(float)
        sigma = raw["realized_daily_vol"][t].astype(float)
        ok = (
            active[rows[i] - begin]
            & raw["fast_present"][t]
            & raw["entry_valid"][t]
            & raw["return_consistent"][t]
            & raw["session_close_valid"][t]
            & np.isfinite(entry + close + sigma)
            & (entry > 0)
            & (close > 0)
            & (sigma > 0)
        )
        np.testing.assert_array_equal(ok, outputs["target_to_close_valid"][i])
        if not ok.any():
            continue
        session = schedule[dates[rows[i]].astype(object)]
        total = minute(session.continuous_close) - minute(session.continuous_open)
        remaining = minute(session.continuous_close) - minute(session.decision_time)
        ret = np.log(close[ok] / entry[ok])
        residual = ret / (sigma[ok] * np.sqrt(remaining / total))
        residual = np.clip(residual - np.median(residual), -5, 5)
        ranked = (
            (rank(residual) / (len(residual) - 1)).astype(np.float32)
            if len(residual) > 1
            else np.array([0.5], np.float32)
        )
        for key, value in [
            ("target_to_close_raw_log_return", ret.astype(np.float32)),
            ("target_to_close_normalized_residual", residual.astype(np.float32)),
            ("target_to_close", ranked),
        ]:
            np.testing.assert_array_equal(value, outputs[key][i, ok])
        target_count += int(ok.sum())
    # Delete future scalar snapshots; the existing dated transform must retain
    # the prefix with every name and its original minimum support.
    end = int(take[9] + 1)
    v = np.empty((10, len(isins), 20), np.float32)
    mask = np.empty_like(v, bool)
    transform_feature_panel_into(
        raw["values"][:end],
        raw["valid"][:end],
        active[raw["date_indices"][:end] - begin],
        [FeatureSpec(**s) for s in scalar_specs],
        v,
        mask,
        source_rows=take[:10],
    )
    np.testing.assert_array_equal(v, outputs["intraday_values"][:10])
    np.testing.assert_array_equal(mask, outputs["intraday_valid"][:10])
    # Mutate the entry bar and entire post-decision path on a source-backed date.
    native = bound_json(m1["native"])
    rec = next(c for c in native["cases"] if c["isin"] == "BRAZZAACNOR9")
    bars = pl.read_parquet(rec["source"]["path"])
    day = min(bars["ts_exchange"].dt.date())
    session = schedule[day]
    grid, seen, _ = grid_for(bars, {day}, (session,), Path(rec["source"]["path"]))
    t = int(np.searchsorted(dates, np.datetime64(day)))
    n = azza
    sigma = np.load(root / "target_scale_sigma.npy", mmap_mode="r")[t, n]
    daily = bound_json(run["surviving_rename_daily_qualification"])
    with np.load(daily["deltas"]["path"]) as z:
        ix = z["target_scale_sigma__indices"]
        pick = (ix[:, 0] == t) & (ix[:, 1] == n)
        sigma = z["target_scale_sigma__values"][pick][0]
    previous = None
    for mutate in [False, True]:
        g = grid.copy()
        visible = seen.copy()
        if mutate:
            cut = minute(session.decision_time) - minute(session.continuous_open)
            g[:, cut:] = 123456.0
            visible[:, cut:] = False
        result = build_native_fast_features(
            *[g[:, None, :, j] for j in [1, 2, 3, 4]],
            visible[:, None],
            volume_valid=visible[:, None],
            session_valid=np.ones((1, 1), bool),
            sigma_asof=np.array([[sigma]]),
            sessions=(session,),
        )
        if previous is not None:
            np.testing.assert_array_equal(result.values, previous.values)
            np.testing.assert_array_equal(result.valid, previous.valid)
        previous = result
    market_root = Path(run["surviving_rename_market"]["path"]).parent
    for family in ["magnitudes", "cross_market"]:
        with np.load(market_root / (family + "_corrected.npz")) as z:
            family_rows = z["rows"]
            rr = z["values"]
            mm = z["valid"]
            aa = z["ages"]
        fs = [s for s in specs if s["family"] == "sidecar_" + family]
        # Chunk only to bound memory; cross-sectional formulas are decision-local.
        for a in range(0, len(rr), 64):
            b = min(a + 64, len(rr))
            local = family_rows[a:b] - begin
            values, valid = independent(rr[a:b], mm[a:b], active[local], fs)
            np.testing.assert_array_equal(
                values, expected["sidecar_" + family + "_values"][local]
            )
            np.testing.assert_array_equal(
                valid, expected["sidecar_" + family + "_valid"][local]
            )
            np.testing.assert_array_equal(
                np.where(active[local, :, None], aa[a:b], -1),
                expected["sidecar_" + family + "_age_sessions"][local],
            )
            formula_cells += 3 * values.size
    # Every new eligible date and changed native date, event controls, monthly
    # remaining market tail, and terminal development date. Slow proofs reused.
    samples = set(rows.tolist())
    old_active = np.load(root / "active.npy", mmap_mode="r")
    samples.update(
        (np.flatnonzero((active & ~old_active[begin:]).any(axis=1)) + begin).tolist()
    )
    for key in ["fast_patch_values", "fast_patch_valid"]:
        for ix, _ in deltas.get(key, []):
            samples.update(ix[:, 0].tolist())
    samples.update(
        [int(np.searchsorted(dates, np.datetime64("2019-08-05"))), len(dates) - 1]
    )
    samples.update(
        (
            np.flatnonzero(
                dates[1:].astype("datetime64[M]") != dates[:-1].astype("datetime64[M]")
            )
            + 1
        ).tolist()
    )
    samples = np.array(sorted(t for t in samples if t >= begin + 59))
    local = samples - begin
    dataset = V2DailyDataset(
        view,
        local.tolist(),
        stage="finetune",
        lookback=60,
        include_fast=True,
        include_intraday=True,
        enabled_sidecars=("magnitudes", "cross_market"),
        target_window_indices=local.tolist(),
    )
    original_read = dataset.store.read
    current = [0]

    def guarded(name, selector):
        assert np.atleast_1d(np.arange(len(view_dates))[selector]).max() <= current[0]
        return original_read(name, selector)

    dataset.store.read = guarded
    mapping = pl.read_parquet(
        root / m["tables"]["native_fast_security_mapping"]["path"]
    )
    permanent = mapping["store_name_index"].to_numpy()
    cells = 0
    for i, t in enumerate(local):
        current[0] = int(t)
        sample = dataset[i]
        batch = collate_v2_daily([sample])
        mask = expected["intraday_valid"][t]
        checks = dict(
            active_mask=active[t],
            current_features=np.where(mask, expected["intraday_values"][t], 0),
            current_feature_mask=mask,
            current_feature_age_sessions=expected["intraday_age_sessions"][t],
        )
        tm = expected["target_to_close_valid"][t] & expected["fast_present"][t]
        checks.update(
            to_close_mask=tm,
            to_close_target=np.where(tm, expected["target_to_close"][t], 0),
        )
        present = expected["fast_present"][t, permanent] & expected["fast_patch_mask"][
            t
        ].any(axis=-1)
        fm = expected["fast_patch_valid"][t, present]
        checks.update(
            fast_patch_values=np.where(
                fm, expected["fast_patch_values"][t, present], 0
            ),
            fast_patch_valid=fm,
            fast_patch_mask=expected["fast_patch_mask"][t, present],
            fast_name_index=permanent[present],
        )
        for family in ["magnitudes", "cross_market"]:
            prefix = "sidecar_" + family
            mask = expected[prefix + "_valid"][t]
            checks.update(
                {
                    prefix + "_values": np.where(
                        mask, expected[prefix + "_values"][t], 0
                    ),
                    prefix + "_valid": mask,
                    prefix + "_age_sessions": expected[prefix + "_age_sessions"][t],
                }
            )
        for key, value in checks.items():
            np.testing.assert_array_equal(sample[key], value, err_msg=key)
            np.testing.assert_array_equal(batch[key][0].numpy(), value, err_msg=key)
            cells += value.size
        assert sample["slow_features"].shape[:2] == (933, 60)
    dataset.store.close()
    report = dict(
        status="qualified_last_four_family_consumers",
        m1=run["surviving_rename_m1"],
        market=run["surviving_rename_market"],
        plan=binding(out / "plan.json"),
        view=binding(view / "manifest.json"),
        samples=len(samples),
        sample_dates=[str(dates[t]) for t in samples],
        sample_cells=cells,
        sample_and_collated_comparisons=2 * cells,
        independent_typed_formula_cells=formula_cells,
        independent_to_close_outcomes=target_count,
        native_post_decision_mutation=True,
        scalar_future_prefix=True,
        future_feature_reads=0,
        mismatches=0,
        names=933,
        history=60,
        seconds=perf_counter() - tick,
        limits="Purpose-limited last-four-family/to-close view. No primary-target/full-store/model result. Reused slow history not counted as another proof; no repeat prior sidecar consumers.",
    )
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_market_input_audit"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "samples",
                    "sample_cells",
                    "independent_typed_formula_cells",
                    "independent_to_close_outcomes",
                    "seconds",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
