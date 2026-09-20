"""Type the seven remaining family tables; preserve prior reducer proofs."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.feature_spec import FeatureSpec, transform_feature_panel_into
from brazil_rv.v2.round5_store import align_family
from verify_surviving_market_inputs import independent

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["stage_c_event_data_admission"])
    root = Path(admission["parent"]["root"])
    manifest = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    out = Path(admission["plan"]["path"]).parent / "sidecars"
    out.mkdir(exist_ok=True)
    assert not (out / "manifest.json").exists()
    (out / ("executed_" + binding(Path(__file__))["sha256"][:12] + ".py")).write_bytes(
        Path(__file__).read_bytes()
    )
    dates, isins = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    old_active = np.load(root / "active.npy", mmap_mode="r")
    active = old_active.copy()
    with np.load(admission["deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    first = int(np.searchsorted(dates, np.datetime64("2020-09-18")))
    issuer, lending, auxiliary = [
        bound_json(run["stage_c_event_" + k])
        for k in ("issuers", "lending", "auxiliaries")
    ]
    pi, plend, pa = [
        bound_json(run["surviving_rename_" + k])
        for k in ("issuers", "lending", "auxiliaries")
    ]
    families = {
        g: (pi["artifacts"][g], issuer["artifacts"][g])
        for g in ("fundamentals", "events")
    }
    families["lending"] = (
        plend["artifacts"]["features"],
        lending["artifacts"]["features"],
    )
    families.update(
        {
            g: (pa["artifacts"][g], auxiliary["artifacts"][g])
            for g in ("microstructure", "options", "rebalance")
        }
    )
    families["oddlot"] = (
        auxiliary["artifacts"]["oddlot_control"],
        auxiliary["artifacts"]["oddlot"],
    )
    if not (out / "plan.json").exists():
        write_json_atomic(
            out / "plan.json",
            dict(
                parent=admission["parent"],
                admission=run["stage_c_event_data_admission"],
                families=families,
                contrast="Final typed alignment of saved seven-family tables, exact existing-coordinate controls and independent original feature transforms. No source/reducer or prior sector/market/consumer repetition. Keep every live loss and unchanged support. One complete-store consumer qualification follows.",
            ),
        )
    reports, all_deltas, losses = {}, {}, []
    for group, (before, after) in families.items():
        saved = out / (group + "_report.json")
        patch = out / (group + "_deltas.npz")
        if saved.exists():
            report = bound_json(binding(saved))
            assert report["mismatches"] == 0
            reports[group] = report
            with np.load(patch) as z:
                all_deltas.update({k: z[k].copy() for k in z.files})
            losses.extend(json.loads((out / (group + "_losses.json")).read_text()))
            continue
        family = "sidecar_" + group
        fields = tuple(manifest["feature_names"][family])
        spec_dicts = [
            s
            for s in manifest["metadata"]["feature_schema"]["specifications"]
            if s["family"] == family
        ]
        specs = [FeatureSpec(**s) for s in spec_dicts]
        frames = [pl.read_parquet(r["path"]) for r in (before, after)]
        original = [
            np.load(root / (family + "_" + k + ".npy"), mmap_mode="r")[first:]
            for k in ("values", "valid", "age_sessions")
        ]
        result = [a.copy() for a in original]
        cells = formula_cells = mismatches = 0
        failures = []
        for start in range(first, len(dates), 64):
            stop = min(start + 64, len(dates))
            days = dates[start:stop].astype(object).tolist()
            sl = slice(start - first, stop - first)
            scope = None
            if group == "oddlot":
                scope = np.zeros((stop - start, len(isins)), bool)
                for event in auxiliary["scopes"]:
                    edge = event["link"]
                    scope[:, edge["successor_index"]] = np.arange(start, stop) >= max(
                        edge["effective_index"], edge["known_index"]
                    )
            for label, frame, membership in (
                ("control", frames[0], old_active[start:stop]),
                ("corrected", frames[1], active[start:stop]),
            ):
                raw, known, age = align_family(
                    frame.filter(pl.col("date").is_between(days[0], days[-1])),
                    days,
                    tuple(isins),
                    fields,
                )
                values, valid = np.zeros_like(raw), np.zeros_like(known)
                transform_feature_panel_into(
                    raw, known, membership, specs, values, valid
                )
                arrays = [values, valid, np.where(membership[..., None], age, -1)]
                if label == "corrected":
                    reference, mask = independent(raw, known, membership, spec_dicts)
                    np.testing.assert_array_equal(values, reference, err_msg=group)
                    np.testing.assert_array_equal(valid, mask, err_msg=group)
                    formula_cells += values.size * 2
                for j, a in enumerate(arrays):
                    if label == "control":
                        bad = a != original[j][sl]
                        if scope is not None:
                            bad &= scope[..., None]
                        count = int(bad.sum())
                        mismatches += count
                        cells += (
                            a.size if scope is None else int(scope.sum()) * a.shape[-1]
                        )
                        if count:
                            failures.append(
                                dict(
                                    first=str(days[0]),
                                    kind=j,
                                    count=count,
                                    examples=np.argwhere(bad)[:3].tolist(),
                                )
                            )
                    else:
                        result[j][sl] = (
                            a
                            if scope is None
                            else np.where(scope[..., None], a, result[j][sl])
                        )
                        if scope is not None:
                            result[j][sl] = np.where(
                                membership[..., None],
                                result[j][sl],
                                -1 if j == 2 else 0,
                            )
        deltas = {}
        for kind, a, b in zip(
            ("values", "valid", "age_sessions"), result, original, strict=True
        ):
            key = family + "_" + kind
            ix = np.argwhere(a != b)
            deltas[key + "__indices"] = ix + np.array([first, 0, 0])
            deltas[key + "__values"] = a[tuple(ix.T)]
        lost = original[1] & ~result[1]
        records = [
            dict(
                family=group,
                date=str(dates[t + first]),
                isin=isins[n],
                field=fields[f],
                still_active=bool(active[t + first, n]),
            )
            for t, n, f in np.argwhere(lost)
        ]
        report = dict(
            control_cells=cells,
            formula_cells=formula_cells,
            mismatches=mismatches,
            failures=failures,
            gains=int((result[1] & ~original[1]).sum()),
            losses=int(lost.sum()),
            live_losses=int((lost & active[first:, ..., None]).sum()),
            shared_numeric=int(
                ((result[0] != original[0]) & result[1] & original[1]).sum()
            ),
            changes={k: len(v) for k, v in deltas.items() if k.endswith("__indices")},
        )
        np.savez_compressed(patch, **deltas)
        write_json_atomic(out / (group + "_losses.json"), records)
        write_json_atomic(saved, report)
        print(json.dumps(dict(family=group, **report)), flush=True)
        assert mismatches == 0, (group, failures)
        reports[group] = report
        all_deltas.update(deltas)
        losses.extend(records)
    np.savez_compressed(out / "deltas.npz", **all_deltas)
    write_json_atomic(out / "validity_losses.json", losses)
    report = dict(
        status="qualified_typed_layers_pending_complete_consumer",
        plan=binding(out / "plan.json"),
        reports=reports,
        deltas=binding(out / "deltas.npz"),
        losses=binding(out / "validity_losses.json"),
        seconds=perf_counter() - tick,
    )
    write_json_atomic(out / "manifest.json", report)
    run = json.loads(pointer.read_text())
    run["stage_c_event_sidecars"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
