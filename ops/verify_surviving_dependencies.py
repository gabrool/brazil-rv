"""Qualify four changed sidecar families and common state in actual consumers."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data import V2DailyDataset, collate_v2_daily
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    transform_feature_panel_into,
)
from brazil_rv.v2.round5_store import align_family
from brazil_rv.v2.store import StoreStaging
from audit_auxiliary_tensors import transformed

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auxiliary", action="store_true")
    args = parser.parse_args()
    tick = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    admission = bound_json(run["surviving_rename_admission"])
    source = Path(admission["plan"]["path"]).parent
    issuer = bound_json(binding(source / "issuers/manifest.json"))
    context = bound_json(binding(source / "context/manifest.json"))
    lending = bound_json(binding(source / "lending/manifest.json"))
    old_issuer = bound_json(run["rename_issuer_propagation"])
    old_lending = bound_json(run["lending_feature_propagation"])
    prior_view = bound_json(run["surviving_rename_daily_input_audit"])["view"]
    vm = bound_json(prior_view)
    vr = Path(prior_view["path"]).parent
    root = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    view_dates = np.load(vr / "date_index.npy")
    begin = int(np.searchsorted(dates, view_dates[0]))
    first = int(np.searchsorted(dates, np.datetime64("2019-08-06")))
    assert np.array_equal(dates[begin:], view_dates)
    out = source / ("auxiliary_consumer" if args.auxiliary else "dependency_consumer")
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    families = {
        "fundamentals": (
            old_issuer["artifacts"]["fundamentals"],
            issuer["artifacts"]["fundamentals"],
        ),
        "events": (old_issuer["artifacts"]["events"], issuer["artifacts"]["events"]),
        "lending": (
            old_lending["artifacts"]["features"],
            lending["artifacts"]["features"],
        ),
        "sector": (context["sector_parent"], context["artifacts"]["corrected_sector"]),
    }
    if args.auxiliary:
        prior = bound_json(run["remaining_auxiliaries"])
        auxiliary = bound_json(binding(source / "auxiliaries/manifest.json"))
        families = {
            g: (prior["artifacts"][g], auxiliary["artifacts"][g])
            for g in ("microstructure", "options", "rebalance", "oddlot")
        }
        families["oddlot"] = (
            auxiliary["artifacts"]["oddlot_control"],
            auxiliary["artifacts"]["oddlot"],
        )
    write_json_atomic(
        out / "plan.json",
        dict(
            parent=admission["parent"],
            prior_slow_view=prior_view,
            families=families,
            context=binding(source / "context/manifest.json"),
            contrast="Reuse qualified daily/slow history. Assemble and verify only the named four new sidecar dependencies; include common diagnostic state only in the issuer/context invocation. All933, actual60-session dataset/collator, no targets/native completeness or model inference. No new normalization/threshold/schema hypothesis.",
        ),
    )
    old_active = np.load(root / "active.npy", mmap_mode="r")
    active = np.load(vr / "active.npy", mmap_mode="r")
    feature_names = {
        k: v
        for k, v in m["feature_names"].items()
        if k == "slow"
        or (k == "common_state_diagnostic" and not args.auxiliary)
        or k in ["sidecar_" + f for f in families]
    }
    specs = [
        s
        for s in m["metadata"]["feature_schema"]["specifications"]
        if s["family"] in feature_names
    ]
    all_deltas, reports, transform_cells = {}, {}, 0
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
        for group, (before, after) in families.items():
            family = "sidecar_" + group
            fields = tuple(feature_names[family])
            fs = [FeatureSpec(**s) for s in specs if s["family"] == family]
            frames = [pl.read_parquet(rec["path"]) for rec in [before, after]]
            arrays = [
                np.load(root / m["arrays"][family + "_" + kind]["path"], mmap_mode="r")[
                    begin:
                ].copy()
                for kind in ["values", "valid", "age_sessions"]
            ]
            cells, failures = 0, []
            for t in range(first, len(dates), 64):
                end = min(t + 64, len(dates))
                days = dates[t:end].astype(object).tolist()
                for label, frame, membership in [
                    ("control", frames[0], old_active[t:end]),
                    ("corrected", frames[1], active[t - begin : end - begin]),
                ]:
                    raw, known, age = align_family(
                        frame.filter(pl.col("date").is_between(days[0], days[-1])),
                        days,
                        tuple(isins),
                        fields,
                    )
                    values, valid = np.zeros_like(raw), np.zeros_like(known)
                    transform_feature_panel_into(
                        raw, known, membership, fs, values, valid
                    )
                    output = [values, valid, np.where(membership[..., None], age, -1)]
                    if args.auxiliary and label == "corrected":
                        for f, spec in enumerate(fs):
                            v, oracle_known = transformed(
                                raw[..., f], known[..., f] & membership, spec.__dict__
                            )
                            np.testing.assert_array_equal(values[..., f], v)
                            np.testing.assert_array_equal(valid[..., f], oracle_known)
                            transform_cells += v.size * 2
                        # The oddlot producer is deliberately successor-scoped.
                    scope = None
                    if group == "oddlot":
                        scope = np.zeros(membership.shape, bool)
                        for event in auxiliary["scopes"]:
                            link = event["link"]
                            gate = max(link["effective_index"], link["known_index"])
                            scope[:, link["successor_index"]] = (
                                np.arange(t, end) >= gate
                            )
                    for j, array in enumerate(output):
                        sl = slice(t - begin, end - begin)
                        if label == "control":
                            compared = array != arrays[j][sl]
                            if scope is not None:
                                compared &= scope[..., None]
                            off = np.argwhere(compared)
                            cells += (
                                int(scope.sum()) * array.shape[-1]
                                if scope is not None
                                else array.size
                            )
                            if len(off):
                                failures.append(
                                    dict(
                                        kind=j,
                                        first=str(dates[t]),
                                        count=len(off),
                                        examples=off[:5].tolist(),
                                    )
                                )
                        else:
                            arrays[j][sl] = (
                                array
                                if scope is None
                                else np.where(scope[..., None], array, arrays[j][sl])
                            )
                            if scope is not None:
                                arrays[j][sl] = np.where(
                                    membership[..., None],
                                    arrays[j][sl],
                                    -1 if j == 2 else 0,
                                )
            for j, kind in enumerate(["values", "valid", "age_sessions"]):
                key = family + "_" + kind
                stage.write_array(key, arrays[j])
                original = np.load(root / m["arrays"][key]["path"], mmap_mode="r")[
                    begin:
                ]
                ix = np.argwhere(arrays[j] != original)
                all_deltas[key + "__indices"] = ix + np.array([begin, 0, 0])
                all_deltas[key + "__values"] = arrays[j][tuple(ix.T)]
            old_valid = np.load(root / (family + "_valid.npy"), mmap_mode="r")[begin:]
            reports[group] = dict(
                control_cells=cells,
                control_failures=failures,
                gained=int((arrays[1] & ~old_valid).sum()),
                lost=int((~arrays[1] & old_valid).sum()),
                changed_values=len(all_deltas[family + "_values__indices"]),
                changed_ages=len(all_deltas[family + "_age_sessions__indices"]),
            )
            write_json_atomic(out / (group + "_report.json"), reports[group])
            print(json.dumps({"family": group, **reports[group]}), flush=True)
        with np.load(context["common_deltas"]["path"]) as z:
            for kind in ["values", "valid"]:
                if args.auxiliary:
                    continue
                key = "common_state_diagnostic_" + kind
                array = np.load(root / (key + ".npy"), mmap_mode="r")[begin:].copy()
                ix = z[key + "__indices"].copy()
                ix[:, 0] -= begin
                array[tuple(ix.T)] = z[key + "__values"]
                stage.write_array(key, array)
        stage.seal(
            feature_names=feature_names,
            sources=[binding(out / "plan.json")],
            tables={
                "slow_history_links": vr / vm["tables"]["slow_history_links"]["path"]
            },
            metadata={
                "purpose": "Named four-family audit view only; slow bytes reused. No complete-store/target/native admission.",
                "feature_schema": {
                    "minimum_rank_names": 20,
                    "specifications": specs,
                    "sha256": feature_schema_sha256([FeatureSpec(**s) for s in specs]),
                },
            },
        )
    np.savez_compressed(out / "deltas.npz", **all_deltas)
    write_json_atomic(
        out / "assembly.json",
        dict(
            reports=reports,
            view=binding(view / "manifest.json"),
            deltas=binding(out / "deltas.npz"),
            seconds=perf_counter() - tick,
        ),
    )
    assert not any(report["control_failures"] for report in reports.values())
    samples = {first - 1, len(dates) - 1}
    for e in [first, int(np.searchsorted(dates, np.datetime64("2024-08-01")))]:
        samples.update(range(e, min(e + 61, len(dates))))
    for path in (
        [] if args.auxiliary else [source / "lending/denominator_losses.parquet"]
    ):
        samples.update(
            np.searchsorted(dates, pl.read_parquet(path)["date"].to_numpy()).tolist()
        )
    for entry in (
        []
        if args.auxiliary
        else bound_json(binding(source / "unit_history/records.json"))
    ):
        samples.add(int(np.searchsorted(dates, np.datetime64(entry["date"]))))
    samples = np.array(sorted(samples))
    local = samples - begin
    dataset = V2DailyDataset(
        view,
        local.tolist(),
        stage="finetune",
        lookback=60,
        enabled_sidecars=tuple(families),
        include_fast=False,
        include_intraday=False,
        include_common_state=not args.auxiliary,
        target_window_indices=local.tolist(),
    )
    current = [0]
    read = dataset.store.read

    def causal_read(name, selector):
        assert np.atleast_1d(np.arange(len(view_dates))[selector]).max() <= current[0]
        return read(name, selector)

    dataset.store.read = causal_read
    expected = {
        k: np.load(view / (k + ".npy"), mmap_mode="r")
        for k in [
            *[
                "sidecar_" + g + "_" + s
                for g in families
                for s in ["values", "valid", "age_sessions"]
            ],
            *(
                []
                if args.auxiliary
                else ["common_state_diagnostic_values", "common_state_diagnostic_valid"]
            ),
        ]
    }
    cells = 0
    for i, day in enumerate(local):
        current[0] = int(day)
        sample = dataset[i]
        batch = collate_v2_daily([sample])
        checks = {"active_mask": active[day]}
        for group in families:
            key = "sidecar_" + group
            mask = expected[key + "_valid"][day]
            checks.update(
                {
                    key + "_values": np.where(mask, expected[key + "_values"][day], 0),
                    key + "_valid": mask,
                    key + "_age_sessions": expected[key + "_age_sessions"][day],
                }
            )
        if not args.auxiliary:
            mask = expected["common_state_diagnostic_valid"][day]
            checks.update(
                common_state_features=np.where(
                    mask, expected["common_state_diagnostic_values"][day], 0
                ),
                common_state_feature_mask=mask,
            )
        for key, value in checks.items():
            np.testing.assert_array_equal(sample[key], value, err_msg=key)
            np.testing.assert_array_equal(batch[key][0].numpy(), value, err_msg=key)
            cells += value.size
    result = dict(
        status="qualified_actual_dependency_consumers",
        assembly=binding(out / "assembly.json"),
        deltas=binding(out / "deltas.npz"),
        view=binding(view / "manifest.json"),
        families=reports,
        samples=len(samples),
        sample_dates=[str(dates[t]) for t in samples],
        names=933,
        history=60,
        sample_cells=cells,
        sample_and_collated_comparisons=2 * cells,
        future_feature_reads=0,
        independent_transformed_cells=transform_cells,
        mismatches=0,
        seconds=perf_counter() - tick,
        limits="New checks cover only named four sidecars/common state where included; previously qualified slow history is reused without counting another slow audit. Native/scalar/magnitude/crossmarket/targets and full-store composition remain. No neural forward/fit/account result.",
    )
    write_json_atomic(out / "manifest.json", result)
    print(
        json.dumps(
            {k: result[k] for k in ["status", "samples", "sample_cells", "seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
