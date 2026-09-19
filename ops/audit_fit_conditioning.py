"""Independently reconstruct saved fit scalers and exercise actual CPU tensors."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.train import _cli_stage_indices


PROJECT = Path(__file__).resolve().parents[1]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reconstruct(samples, rows, saved, parent):
    centers, scales, support, varying, inherited = [], [], [], [], []
    for i, values in enumerate(samples):
        values = np.asarray(values, np.float64)
        values = values[np.isfinite(values)]
        changes = bool(len(values) and np.max(values) > np.min(values))
        inherit = parent is not None and parent["varying"][i]
        if inherit:
            center, scale = parent["center"][i], parent["scale"][i]
        elif len(values):
            lower, center, upper = np.percentile(values, [25, 50, 75])
            scale = upper - lower
            if scale == 0 and changes:
                nonzero = np.abs(values - center)
                scale = np.median(nonzero[nonzero > 0])
            scale = scale if scale else 1.0
        else:
            center, scale = 0.0, 1.0
        centers.append(center)
        scales.append(scale)
        support.append(len(values))
        varying.append(changes or inherit)
        inherited.append(bool(inherit))
    dates = set(rows.tolist())
    if parent is not None and any(inherited):
        dates.update(parent["fit_date_indices"])
    expected = dict(
        center=centers,
        scale=scales,
        support=support,
        varying=varying,
        inherited=inherited,
        fit_date_indices=sorted(dates),
    )
    for key, value in expected.items():
        np.testing.assert_array_equal(value, saved[key], err_msg=key)
    return sum(support)


def main():
    started = time.perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    destination = Path(run["root"]) / "fit_conditioning_audit"
    destination.mkdir(exist_ok=True)
    if (destination / "report.json").exists():
        raise FileExistsError("completed fit audit must not be repeated")
    source = json.loads((PROJECT / "docs/v2_data_inputs.json").read_text())["store"]
    store = Path(source["root"])
    assert sha256_file(store / "manifest.json") == source["manifest_sha256"]
    metadata = json.loads((store / "manifest.json").read_text())
    arrays = {}

    def array(name):
        if name not in arrays:
            arrays[name] = np.load(
                store / metadata["arrays"][name]["path"], mmap_mode="r"
            )
        return arrays[name]

    foundation = Path(
        json.loads((PROJECT / "docs/v2_foundation_run.json").read_text())["root"]
    )
    manifests = []
    parents = {}
    for path in sorted((foundation / "fits").glob("*/*/run_manifest.json")):
        report = json.loads(path.read_text())
        contract = report["contract"]
        assert contract["store_manifest_sha256"] == source["manifest_sha256"]
        manifests.append((path, report))
        parents[report["artifacts"]["selected.pt"]] = contract["preprocessing"]
    # Every distinct saved coordinate system is checked; seed/architecture copies
    # are matched by complete payload, not presumed identical from their names.
    done = {}
    scalar_cache = {}
    records = []
    unique_observations = 0
    for path, report in manifests:
        c = report["contract"]
        prep = c["preprocessing"]
        fit, selection, evaluation, target_window = _cli_stage_indices(
            store, c["stage"], c["fold"]
        )
        assert list(target_window) == c["fit_target_window"]
        dates = np.load(store / "date_index.npy", mmap_mode="r")
        assert dates[np.r_[fit, selection, evaluation]].max() < np.datetime64(
            "2025-01-01"
        )
        parent = None if c["stage"] == "P" else parents[c["parent_sha256"]]
        key = fingerprint([prep, None if parent is None else parent, fit.tolist()])
        if key not in done:
            support = 0
            for family, saved in prep["families"].items():
                parent_family = (
                    None if parent is None else parent["families"].get(family)
                )
                names = prep["feature_names"][family]
                cache_key = fingerprint(
                    [family, names, saved, parent_family, fit.tolist()]
                )
                if cache_key in scalar_cache:
                    support += scalar_cache[cache_key]
                    continue
                prefix = "sidecar_" + family
                source_names = metadata["feature_names"][prefix]
                columns = [source_names.index(name) for name in names]
                values = array(prefix + "_values")[fit][..., columns]
                valid = (
                    array(prefix + "_valid")[fit][..., columns]
                    & array("active")[fit][..., None]
                )
                shared = {
                    i
                    for i, name in enumerate(names)
                    if family == "cross_market"
                    and (
                        name.startswith("shock_")
                        or name
                        in {
                            "foreign_flow_1",
                            "foreign_flow_5",
                            "foreign_flow_month_reset",
                            "foreign_flow_methodology_change",
                            "ewz_minus_bova11_1",
                        }
                    )
                }
                samples = []
                for i in range(len(names)):
                    if i in shared:
                        # One value per date, not repeated once per eligible stock.
                        v, m = values[..., i], valid[..., i]
                        known = m.any(axis=1)
                        first = m.argmax(axis=1)
                        samples.append(v[np.arange(len(fit)), first][known])
                    else:
                        samples.append(values[..., i][valid[..., i]])
                count = reconstruct(samples, fit, saved, parent_family)
                scalar_cache[cache_key] = count
                support += count
                unique_observations += count
                del values, valid, samples
            if prep["diagnostic"] is not None:
                v = array("common_state_diagnostic_values")[fit]
                m = array("common_state_diagnostic_valid")[fit]
                support += reconstruct(
                    [v[:, i][m[:, i]] for i in range(v.shape[1])],
                    fit,
                    prep["diagnostic"],
                    None if parent is None else parent["diagnostic"],
                )
            probes = fit[np.unique([0, len(fit) // 2, len(fit) - 1])]
            dataset = V2DailyDataset(
                store,
                probes,
                stage="pretrain" if c["stage"] == "P" else "finetune",
                lookback=60,
                enabled_sidecars=tuple(prep["families"]),
                include_intraday=False,
                include_fast=False,
                include_common_state=prep["diagnostic"] is not None,
                compact_names=True,
                target_window_indices=target_window,
                purpose="training",
            )
            transformer = Round7Preprocessing.from_payload(prep)
            tensor_cells = 0
            for i in range(len(probes)):
                raw = dataset[i]
                actual = transformer.transform_sample(raw)
                assert raw["slow_features"].shape[-2] == 60
                for family, saved in prep["families"].items():
                    prefix = "sidecar_" + family
                    columns = prep.get("source_columns", {}).get(family, slice(None))
                    values = raw[prefix + "_values"][..., columns]
                    valid = raw[prefix + "_valid"][..., columns]
                    expected = np.arcsinh(
                        (values.astype(np.float64) - saved["center"]) / saved["scale"]
                    )
                    expected = np.where(saved["passthrough"], values, expected)
                    expected = np.where(valid, expected, 0).astype(np.float32)
                    if family == "cross_market" and prep["common_columns"]:
                        expected = expected[..., prep["per_name_columns"]]
                    np.testing.assert_array_equal(actual[prefix + "_values"], expected)
                    tensor_cells += expected.size
                batch = transformer.collate(
                    [raw], fixed_name_count=c["padded_name_count"]
                )
                assert batch["slow_features"].shape[-2] == 60
                assert batch["active_mask"].sum().item() == raw["active_mask"].sum()
            dataset.store.close()
            done[key] = {
                "fit_rows": len(fit),
                "fields": sum(len(s["center"]) for s in prep["families"].values()),
                "fit_support": support,
                "probe_dates": probes.tolist(),
                "tensor_cells": tensor_cells,
            }
        records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "coordinates_sha256": key,
                "stage": c["stage"],
                "fold": c["fold"],
                "cell": c["cell"]["cell"],
                "seed": c["seed"],
            }
        )
        write_json_atomic(
            destination / "progress.json",
            {"checked_manifests": len(records), "unique_coordinates": len(done)},
        )
    write_json_atomic(
        destination / "report.json",
        {
            "store": source,
            "foundation_root": str(foundation),
            "fits": records,
            "coordinates": done,
            "unique_scalar_fit_samples": unique_observations,
            "mismatches": 0,
            "method": "independent exact median/IQR/MAD, support, varying/inheritance and fit-date union; saved parent hash; actual 60-session CPU dataset, scalar transform and padded collator",
            "limits": "does not establish original source publication/revision quality; no forward model, predictions, GPU fit or held-out consumer read; split schedule reused, scalar arithmetic independent",
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    print(
        json.dumps(
            {
                "fits": len(records),
                "coordinate_systems": len(done),
                "scalar_fit_samples": unique_observations,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()
