"""Read-only development-store and saved-trajectory diagnostics; no fitting."""

import json
import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from brazil_rv.v2.round5_store import align_family
from brazil_rv.v2.contract import (
    FINETUNE_START,
    PRETRAIN_END,
    TARGET_NEUTRALIZATION_FEATURES,
)
from brazil_rv.v2.store import open_store_for_samples, characteristic_neutral_targets

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path("D:/quant-data/b3/interim/round7_postmortem_20260913")
binding = json.loads((PROJECT / "docs/v2_round7_inputs.json").read_text())
ROOT = Path(binding["store"]["root"])
manifest = json.loads((ROOT / "manifest.json").read_text())


def read(name):
    return np.load(ROOT / manifest["arrays"][name]["path"], mmap_mode="r")


dates = np.load(ROOT / "date_index.npy")
assert dates.max() <= np.datetime64("2024-12-31")
active = read("active")
pretrain = dates <= np.datetime64("2016-06-30")
fields = []
for family, names in manifest["feature_names"].items():
    if family not in ("slow", "common_state_diagnostic") and not family.startswith(
        "sidecar_"
    ):
        continue
    values, valid = read(family + "_values"), read(family + "_valid")
    if values.ndim != 3:
        continue
    ages = read(family + "_age_sessions")
    for f, name in enumerate(names):
        known = valid[..., f] & active
        sample = values[..., f][known]
        counts = known.sum(axis=1)
        age = ages[..., f][known]
        fields.append(
            dict(
                family=family,
                field=name,
                active_valid=int(known.sum()),
                pretrain_valid=int(known[pretrain].sum()),
                active_total=int(active.sum()),
                nonfinite_valid=int((~np.isfinite(sample)).sum()),
                negative_age_valid=int((age < 0).sum()),
                percentiles=np.quantile(
                    sample, [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
                ).tolist()
                if sample.size
                else [],
                age_gt252_share=float(np.mean(age > 252)) if age.size else None,
                days_valid=int((counts > 0).sum()),
                days_below20=int(((counts > 0) & (counts < 20)).sum()),
            )
        )
    print(family, flush=True)

raw_losses = []
# Reuse the store's explicit ISIN axis, not ticker/name matching.
rows = np.flatnonzero(
    (dates <= np.datetime64(PRETRAIN_END)) | (dates >= np.datetime64(FINETUNE_START))
)
store, _ = open_store_for_samples(
    ROOT, rows, purpose="training", history_lookbacks=60, history_end_offsets=0
)
try:
    isins = store.isins
    sample_rows = rows[dates[rows] >= np.datetime64(FINETUNE_START)][::31]
    z_columns = [
        manifest["feature_names"]["slow"].index(n)
        for n in TARGET_NEUTRALIZATION_FEATURES
    ]
    y = store.read("target_shareholder_simple_return", sample_rows)
    m = store.read("target_valid", sample_rows)
    sigma = store.read("target_scale_sigma", sample_rows)
    z = store.read("slow_values", sample_rows)[..., z_columns]
    zm = store.read("slow_valid", sample_rows)[..., z_columns].all(axis=-1)
    original, known = characteristic_neutral_targets(y, m, sigma, z, zm)
    rng = np.random.default_rng(719)
    permutation = rng.permutation(y.shape[1])
    reverse = np.argsort(permutation)
    permuted, _ = characteristic_neutral_targets(
        y[:, permutation],
        m[:, permutation],
        sigma[:, permutation],
        z[:, permutation],
        zm[:, permutation],
    )
    delta = np.abs(original - permuted[:, reverse])
    objective = dict(
        sample_dates=dates[sample_rows].astype(str).tolist(),
        valid_labels=int(known.sum()),
        permutation_changed_labels=int(((delta > 1e-6) & known).sum()),
        permutation_max_error=float(delta.max()),
    )
    synthetic = rng.normal(size=(1, 100, 5))
    synthetic_mask = np.ones_like(synthetic, dtype=bool)
    tied = np.zeros((1, 100, 3))
    tied_mask = np.ones((1, 100), bool)
    unit_sigma = np.ones((1, 100))
    original, _ = characteristic_neutral_targets(
        synthetic, synthetic_mask, unit_sigma, tied, tied_mask
    )
    permutation = rng.permutation(100)
    reverse = np.argsort(permutation)
    permuted, _ = characteristic_neutral_targets(
        synthetic[:, permutation], synthetic_mask, unit_sigma, tied, tied_mask
    )
    delta = np.abs(original - permuted[:, reverse])
    objective["synthetic_tied_characteristics"] = dict(
        changed_labels=int((delta > 1e-6).sum()), max_error=float(delta.max())
    )
    sources = {}
    ancestor = manifest
    while extension := ancestor["metadata"].get("round5_extension"):
        for item in extension["families"]:
            sources.setdefault(item["family"], item)
        ancestor = json.loads(
            (Path(extension["base_store"]["root"]) / "manifest.json").read_text()
        )
    for item in sources.values():
        family = "sidecar_" + item["family"]
        if item["status"] != "admitted":
            continue
        frame = pl.read_parquet(item["data"]["path"])
        names = tuple(item["feature_names"])
        raw, mask, age = align_family(
            frame, dates.astype(object).tolist(), isins, names
        )
        stored = read(family + "_valid")
        for f, name in enumerate(names):
            known = mask[..., f] & active
            lost = known & ~stored[..., f]
            if lost.any():
                raw_losses.append(
                    dict(
                        family=family,
                        field=name,
                        raw_valid=int(known.sum()),
                        lost=int(lost.sum()),
                        lost_fraction=float(lost.sum() / known.sum()),
                        below20_lost=int(
                            (lost & (known.sum(axis=1) < 20)[:, None]).sum()
                        ),
                    )
                )
finally:
    store.close()

runs = Path("D:/quant-data/b3/processed/model_runs")
curves = []
for run in ("v2_round7_20260913T012300Z", "v2_round7_pathway_20260913T031500Z"):
    for path in (runs / run / "trajectories").glob("*/*/history.json"):
        hist = json.loads(path.read_text())
        if not isinstance(hist, list) or not hist or "selection" not in hist[-1]:
            continue
        curves.append(
            dict(
                run=run,
                cell=path.parent.parent.name,
                fit=path.parent.name,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                epochs=len(hist),
                selection=[h["selection"]["mean_ic"] for h in hist],
                clean_fit=[
                    h["clean_fit"]["mean_ic"] if h.get("clean_fit") else None
                    for h in hist
                ],
                sam_gap=[h["sam_gap"] for h in hist],
                loss=[h["training_loss"] for h in hist],
            )
        )

summary = []
for cell in sorted({c["cell"] for c in curves}):
    selected = [
        c
        for c in curves
        if c["cell"] == cell
        and c["fit"]
        in {
            f"{fold}_seed_{seed}"
            for fold in ("F2", "F6", "F10", "F14")
            for seed in (11, 29, 47)
        }
    ]
    summary.append(
        dict(
            cell=cell,
            n=len(selected),
            clean_fit=float(np.mean([c["clean_fit"][-1] for c in selected])),
            select_first=float(np.mean([c["selection"][0] for c in selected])),
            select_final=float(np.mean([c["selection"][-1] for c in selected])),
            loss=float(np.mean([c["loss"][-1] for c in selected])),
        )
    )
pretraining = []
for cell in ("s0_all", "c1_all", "c1_slow", "c1_attention_all"):
    paths = sorted(
        (runs / "v2_round7_20260913T012300Z/pretraining" / cell).glob("*/history.json")
    )
    histories = [json.loads(p.read_text()) for p in paths]
    pretraining.append(
        dict(
            cell=cell,
            n=len(histories),
            clean_fit=float(
                np.mean([h[-1]["clean_fit"]["mean_ic"] for h in histories])
            ),
            select_first=float(
                np.mean([h[0]["selection"]["mean_ic"] for h in histories])
            ),
            select_final=float(
                np.mean([h[-1]["selection"]["mean_ic"] for h in histories])
            ),
            sources=[
                dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for p in paths
            ],
        )
    )
calibration_paths = sorted(
    (runs / "v2_round7_20260913T012300Z/calibration").glob("*/history.json")
)
calibration_histories = [json.loads(p.read_text()) for p in calibration_paths]
calibration = dict(
    n=len(calibration_histories),
    mean_selection=np.mean(
        [[h["selection"]["mean_ic"] for h in hist] for hist in calibration_histories],
        axis=0,
    ).tolist(),
    mean_clean_fit=np.mean(
        [[h["clean_fit"]["mean_ic"] for h in hist] for hist in calibration_histories],
        axis=0,
    ).tolist(),
    selected_budget=json.loads(
        (runs / "v2_round7_20260913T012300Z/budget.json").read_text()
    ),
    sources=[
        dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in calibration_paths
    ],
)
calibration["best_mean_selection_epoch"] = int(
    np.argmax(calibration["mean_selection"]) + 1
)
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / "curve_summary.json").write_text(json.dumps(summary, indent=2))
(OUTPUT / "calibration_audit.json").write_text(json.dumps(calibration, indent=2))
(OUTPUT / "inputs_and_curves.json").write_text(
    json.dumps(
        dict(
            store=str(ROOT),
            fields=fields,
            objective=objective,
            rank_gate_losses=raw_losses,
            pretraining_summary=pretraining,
            curves=curves,
        ),
        indent=2,
    )
)
print(
    json.dumps(
        dict(fields=len(fields), rank_gate_losses=raw_losses, curves=len(curves))
    )
)
