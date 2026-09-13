"""CPU acceptance of rebuilt data and actual scalar consumers; no model fitting."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.data import V2DailyDataset, stage_name_count
from brazil_rv.v2.data_repair import parent_store
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.round7 import configuration
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import forward
from brazil_rv.v2.train import _cli_stage_indices

parser = argparse.ArgumentParser()
parser.add_argument("--repair-root", type=Path, required=True)
args = parser.parse_args()
report = json.loads(
    (args.repair_root / "store_repair.json").read_text(encoding="utf-8")
)
root = Path(report["store"]["root"])
assert sha256_file(root / "manifest.json") == report["store"]["manifest_sha256"]
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
old_root, old_manifest, _ = parent_store()
assert manifest["axes"]["date_end"] <= "2024-12-30"
for name in report["protected_arrays"]:
    assert manifest["arrays"][name]["sha256"] == old_manifest["arrays"][name]["sha256"]


def read(name, where=root):
    return np.load(where / (name + ".npy"), mmap_mode="r")


active = read("active")
fields = []
for family, names in manifest["feature_names"].items():
    if family != "slow" and not family.startswith("sidecar_"):
        continue
    values, valid, ages = (
        read(family + suffix) for suffix in ("_values", "_valid", "_age_sessions")
    )
    for f, name in enumerate(names):
        mask = valid[..., f] & active
        sample = values[..., f][mask]
        assert np.isfinite(sample).all(), (family, name)
        assert np.all(ages[..., f][mask] >= 0), (family, name)
        fields.append(
            {
                "family": family,
                "field": name,
                "valid_active": int(mask.sum()),
                "known_age_missing_value": int(
                    (~valid[..., f] & (ages[..., f] >= 0) & active).sum()
                ),
                "value_quantiles": np.quantile(
                    sample, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
                ).tolist()
                if sample.size
                else [],
            }
        )
    if family == "slow":
        before, before_valid = (
            read("slow_values", old_root),
            read("slow_valid", old_root),
        )
        for f, name in enumerate(names):
            assert np.array_equal(valid[..., f], before_valid[..., f])
            if name not in {
                "realized_skew_60",
                "realized_kurtosis_60",
                "observed_history_age_sessions",
            }:
                assert np.array_equal(values[..., f], before[..., f])

families = tuple(
    n.removeprefix("sidecar_")
    for n in manifest["feature_names"]
    if n.startswith("sidecar_")
)
assert "fundamentals_native" not in families
assert len(manifest["feature_names"]["sidecar_magnitudes"]) == 4
torch.set_num_threads(4)
torch.manual_seed(713)
parent = None
stages = []
for stage, fold in (("P", "pretrain_internal"), ("F", "F2"), ("F", "F14")):
    fit, _, _, target_window = _cli_stage_indices(root, stage, fold)
    dataset = V2DailyDataset(
        root,
        fit,
        stage="pretrain" if stage == "P" else "finetune",
        enabled_sidecars=families,
        include_intraday=False,
        include_fast=False,
        include_common_state=True,
        compact_names=True,
        target_window_indices=target_window,
    )
    try:
        prep = Round7Preprocessing.fit(dataset, split_common=True, parent=parent)
        if stage == "P":
            parent = prep
        else:
            for family, scaler in prep.families.items():
                known = np.asarray(parent.families[family].varying)
                assert np.array_equal(
                    scaler.center[known], parent.families[family].center[known]
                )
                assert np.array_equal(
                    scaler.scale[known], parent.families[family].scale[known]
                )
        summaries = []
        for family, scaler in prep.families.items():
            names = manifest["feature_names"]["sidecar_" + family]
            chunks = [[] for _ in names]
            for start in range(0, len(fit), 128):
                rows = fit[start : start + 128]
                raw = dataset.store.read("sidecar_" + family + "_values", rows)
                valid = (
                    dataset.store.read("sidecar_" + family + "_valid", rows)
                    & dataset.store.read("active", rows)[..., None]
                )
                transformed = scaler.transform(raw, valid)
                assert np.isfinite(transformed).all()
                for f in range(len(names)):
                    chunks[f].append(transformed[..., f][valid[..., f]])
            for f, column in enumerate(chunks):
                sample = np.concatenate(column)
                summaries.append(
                    {
                        "family": family,
                        "field": names[f],
                        "fit_support": scaler.support[f],
                        "inherited": scaler.inherited[f],
                        "center": scaler.center[f],
                        "scale": scaler.scale[f],
                        "transformed_quantiles": np.quantile(
                            sample, [0.01, 0.5, 0.99]
                        ).tolist()
                        if sample.size
                        else [],
                        "maximum_absolute": float(np.abs(sample).max())
                        if sample.size
                        else None,
                    }
                )
        samples = [
            dataset[i]
            for i in np.linspace(
                max(0, len(dataset) - 40), len(dataset) - 1, 3, dtype=int
            )
        ]
        width = stage_name_count(dataset)
        checks = []
        for graph in ("s0", "c1"):
            current = (
                prep
                if graph == "c1"
                else replace(
                    prep, common_columns=(), per_name_columns=(), diagnostic=None
                )
            )
            model_samples = (
                samples
                if graph == "c1"
                else [
                    {k: v for k, v in s.items() if not k.startswith("common_state_")}
                    for s in samples
                ]
            )
            batch = current.collate(model_samples, fixed_name_count=width)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and (
                    key.startswith("sidecar_") or key.startswith("common_state_")
                ):
                    assert torch.isfinite(value).all(), key
            config = configuration(
                {"graph": graph, "inputs": "all"}, manifest["feature_names"]
            )
            model = (
                CharacteristicModel(config)
                if graph == "c1"
                else DailyMultiHorizonModel(config)
            ).eval()
            with torch.no_grad():
                scores = forward(model, batch, characteristic=graph == "c1")
                assert torch.isfinite(scores).all()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    mixed = forward(model, batch, characteristic=graph == "c1")
                assert torch.isfinite(mixed).all()
            checks.append(
                {
                    "graph": graph,
                    "fp32_finite": True,
                    "bf16_finite": True,
                    "score_shape": list(scores.shape),
                    "active_names_retained": int(batch["active_mask"].sum()),
                }
            )
        stages.append(
            {
                "stage": stage,
                "fold": fold,
                "fit_dates": len(fit),
                "fixed_name_count": width,
                "fields": summaries,
                "forward_checks": checks,
                "preprocessing": prep.payload(),
            }
        )
        print(json.dumps({"accepted_stage": stage, "fold": fold}), flush=True)
    finally:
        dataset.store.close()

acceptance = {
    "store": report["store"],
    "parent": report["parent"],
    "dates": manifest["axes"],
    "active_stock_days": int(active.sum()),
    "maximum_active_names": int(active.sum(axis=1).max()),
    "scalar_fields": len(fields),
    "protected_arrays_exact": True,
    "slow_masks_exact": True,
    "other_slow_coordinates_exact": True,
    "fields": fields,
    "stages": stages,
    "forward_capture": False,
    "heldout_access": False,
    "model_fitting": False,
    "scope": "Data correctness and conditioning acceptance; no alpha or architecture claim.",
}
write_json_atomic(args.repair_root / "data_acceptance.json", acceptance)
print(json.dumps({"accepted": True, "scalar_fields": len(fields)}))
