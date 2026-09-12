"""Chronological archived-score residual trees; diagnostic only, no promotion vote."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .contract import REGISTERED_PRIMARY_TARGET, REGISTERED_PRIMARY_TARGET_MASK
from .gbdt import require_lightgbm
from .normalization import midrank_unit_interval, average_ranks
from .round7_acceptance import source_run_root
from .round7_data import registered_sources
from .store import open_store_for_samples
from .train import _cli_stage_indices, rank_average_ensemble
from .validate_pipeline import _window_target_mask

PARAMETERS = {
    "objective": "regression",
    "num_leaves": 31,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 1.0,
    "seed": 29,
    "num_threads": 4,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
ROUNDS = 500


def mature_before(sample_indices, decision_index, maximum_horizon=10):
    """Last label endpoint must strictly precede the new fold's first decision."""
    return np.asarray(sample_indices) + maximum_horizon < decision_index


def run(output, *, source_root=None, store_root=None, arm="S0"):
    lgb = require_lightgbm()
    if store_root is None:
        root, manifest, _, accepted = registered_sources()
    else:
        root = store_root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        accepted = {
            "store": {
                "root": str(root),
                "manifest_sha256": sha256_file(root / "manifest.json"),
            }
        }
    archived = source_root or source_run_root()
    folds = [(f"F{i}", _cli_stage_indices(root, "F", f"F{i}")[2]) for i in range(1, 15)]
    rows = np.concatenate([r for _, r in folds])
    families = sorted(k for k in manifest["feature_names"] if k.startswith("sidecar_"))
    names = [
        f"{family}:{name}"
        for family in families
        for name in manifest["feature_names"][family]
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output / "registration.json",
        {
            "parameters": PARAMETERS,
            "rounds": ROUNDS,
            "base_store": accepted["store"],
            "features": [
                "S0_three_seed_rank_composite",
                *names,
                *(n + ":age" for n in names),
            ],
            "target": "mean D3/D5/D10 neutral rank minus S0 rank composite",
            "population": "same complete-label active names for both trees; equal total weight per historical date",
            "purge": "past fold labels with t+10 strictly before next fold; no extra lag",
            "controls": [
                "sealed S0",
                "S0-only residual tree",
                "S0-plus-families residual tree",
            ],
            "promotion_weight": 0,
            "tuning": "none; fixed rounds and one deterministic tree seed",
        },
    )
    store, access = open_store_for_samples(
        root, rows, purpose="evaluation", history_lookbacks=60, history_end_offsets=0
    )
    panels = []
    try:
        for fold, ix in folds:
            members, mask = [], None
            for seed in (11, 29, 47):
                directory = (
                    archived / "trajectories" / arm / f"{fold}_seed_{seed}" / "scores"
                )
                score, valid = rr._score_artifact(
                    directory,
                    require_clean_transfer=True,
                    expected_dates=store.dates[ix],
                    expected_isins=store.isins,
                    expected_feature_schema_sha256=manifest["feature_schema_sha256"],
                )
                if mask is not None and not np.array_equal(mask, valid):
                    raise ValueError("S0 score populations differ")
                mask = valid
                members.append(score)
            scores = rank_average_ensemble(members, mask)[..., 2:5]
            active = store.read("active", ix) & mask[..., 2:5].all(axis=-1)
            target_mask = _window_target_mask(
                store.read(REGISTERED_PRIMARY_TARGET_MASK, ix), ix
            )
            targets = store.read_target(
                REGISTERED_PRIMARY_TARGET, ix, valid_mask=target_mask
            )[..., 2:5]
            valid = target_mask[..., 2:5].all(axis=-1) & active
            composite = np.zeros(active.shape, np.float32)
            for d in range(len(ix)):
                if active[d].any():
                    composite[d, active[d]] = np.stack(
                        [
                            midrank_unit_interval(scores[d, active[d], h])
                            for h in range(3)
                        ]
                    ).mean(axis=0)
            d, n = np.nonzero(valid)
            fields, ages = [], []
            for family in families:
                known = store.read(family + "_valid", ix)[d, n]
                value = store.read(family + "_values", ix)[d, n]
                age = store.read(family + "_age_sessions", ix)[d, n]
                fields.append(np.where(known, value, np.nan))
                ages.append(
                    np.where(
                        age >= 0.0,
                        np.log1p(np.clip(age, 0.0, 252.0)) / np.log1p(252.0),
                        np.nan,
                    )
                )
            panels.append(
                {
                    "fold": fold,
                    "first_index": int(ix[0]),
                    "date_indices": ix[d],
                    "name_indices": n,
                    "scores": composite[d, n],
                    "targets": targets[d, n].mean(axis=-1),
                    "X": np.column_stack((composite[d, n], *fields, *ages)),
                }
            )
        completed = []
        for k in range(1, len(panels)):
            current = panels[k]
            preceding = panels[:k]
            train_dates = np.concatenate([p["date_indices"] for p in preceding])
            mature = mature_before(train_dates, current["first_index"])
            x = np.concatenate([p["X"] for p in preceding])[mature]
            baseline = np.concatenate([p["scores"] for p in preceding])[mature]
            target = (
                np.concatenate([p["targets"] for p in preceding])[mature] - baseline
            )
            dates, counts = np.unique(train_dates[mature], return_counts=True)
            weight = 1.0 / counts[np.searchsorted(dates, train_dates[mature])]
            predictions = {"sealed_s0": current["scores"]}
            for label, columns in (
                ("score_only_corrector", slice(0, 1)),
                ("all_families_corrector", slice(None)),
            ):
                dataset = lgb.Dataset(
                    x[:, columns],
                    label=target,
                    weight=weight * len(weight) / weight.sum(),
                    free_raw_data=True,
                )
                model = lgb.train(PARAMETERS, dataset, num_boost_round=ROUNDS)
                predictions[label] = current["scores"] + model.predict(
                    current["X"][:, columns], num_threads=4
                )
                model.save_model(str(output / f"{current['fold']}_{label}.txt"))
            daily = []
            for day in np.unique(current["date_indices"]):
                selected = current["date_indices"] == day
                if selected.sum() < 20:
                    continue
                y = average_ranks(current["targets"][selected])
                y -= y.mean()
                row = {"date_index": int(day), "names": int(selected.sum())}
                for label, values in predictions.items():
                    z = average_ranks(values[selected])
                    z -= z.mean()
                    denominator = np.linalg.norm(z) * np.linalg.norm(y)
                    row[label] = float(z @ y / denominator) if denominator > 0 else None
                daily.append(row)
            result = {
                "fold": current["fold"],
                "fit_stock_days": len(target),
                "fit_dates": len(dates),
                "maximum_label_endpoint_index": int(train_dates[mature].max()) + 10,
                "evaluation_first_index": current["first_index"],
                "daily": daily,
                "parameters": PARAMETERS,
                "rounds": ROUNDS,
                "model_hashes": {
                    p.name: sha256_file(p)
                    for p in output.glob(current["fold"] + "_*.txt")
                },
            }
            write_json_atomic(output / (current["fold"] + ".json"), result)
            completed.append(result)
            print(
                {
                    "fold": current["fold"],
                    "fit_stock_days": len(target),
                    "scored_dates": len(daily),
                },
                flush=True,
            )
        write_json_atomic(
            output / "result.json",
            {
                "status": "complete",
                "access": access.payload(),
                "folds": completed,
                "interpretation": "Increment beyond the matched score-only correction is consistent with usable information beyond S0; it does not establish architecture as the unique cause.",
            },
        )
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--arm", default="S0", choices=("S0", "A0"))
    args = parser.parse_args()
    run(args.output, source_root=args.source_root, store_root=args.store, arm=args.arm)


if __name__ == "__main__":
    main()
