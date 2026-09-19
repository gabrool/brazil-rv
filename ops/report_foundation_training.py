"""Recover promised fit diagnostics from sealed histories/models; never refit."""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import ALLOWED_SEEDS
from brazil_rv.v2.foundation_residual import (
    STRENGTHS,
    daily_ic,
    encode_features,
    rank_targets,
)
from brazil_rv.v2.portfolio_program import read
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.store import open_store_for_samples
from brazil_rv.v2.train import _cli_stage_indices


def main():
    pointer = read(Path("docs/v2_foundation_run.json"))
    root = Path(pointer["root"])
    design = read(root / "frozen_design.json")
    source = Path(design["store"]["root"])
    if sha256_file(source / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("accepted store binding changed")
    neural = []
    for path in sorted((root / "fits").glob("*/*/run_manifest.json")):
        manifest = read(path)
        history_path = path.parent / "history.json"
        if sha256_file(history_path) != manifest["artifacts"]["history.json"]:
            raise ValueError("sealed history changed")
        history = read(history_path)
        selected = next(
            (r for r in history if r["epoch"] == manifest["selected_epoch"]), None
        )
        neural.append(
            {
                "cell": path.parent.parent.name,
                "stage": manifest["stage"],
                "fold": manifest["fold"],
                "seed": manifest["seed"],
                "selected_epoch": manifest["selected_epoch"],
                "epochs": manifest["epochs_completed"],
                "selected_clean_fit_probe_ic": None
                if selected is None
                else selected["clean_fit_probe"]["mean_ic"],
                "selected_selection_ic": manifest["selection_ic"],
                "last_clean_fit_probe_ic": history[-1]["clean_fit_probe"]["mean_ic"],
                "last_selection_ic": history[-1]["selection"]["mean_ic"],
                "seconds": manifest["seconds_this_process"],
                "peak_cuda_bytes": manifest["peak_cuda_bytes"],
                "manifest_sha256": sha256_file(path),
                "history_sha256": sha256_file(history_path),
            }
        )
    anchors_path = root / "residual/anchors.npz"
    if sha256_file(anchors_path) != read(root / "residual/anchors.json")["sha256"]:
        raise ValueError("sealed anchor changed")
    with np.load(anchors_path) as cache:
        ranks, indices = cache["ranks"], cache["indices"]
    residual = []
    for fold in ("F2", "F6", "F10", "F14"):
        fit, selection, evaluation, _ = _cli_stage_indices(source, "F", fold)
        rows = np.unique(np.r_[fit, selection])
        scaler_path = root / "residual" / f"{fold}_scalers.json"
        scaler = Round7Preprocessing.from_payload(read(scaler_path))
        store, _ = open_store_for_samples(
            source,
            rows,
            purpose="evaluation",
            history_lookbacks=60,
            history_end_offsets=0,
            target_window_indices=np.arange(fit[0], evaluation[0]),
        )
        try:
            sets = {}
            for label, ix, before in (
                ("fit", fit, selection[0]),
                ("selection", selection, evaluation[0]),
            ):
                x, common, dates, names = encode_features(store, ix, scaler)
                a = np.searchsorted(indices, dates)
                if not np.array_equal(indices[a], dates):
                    raise ValueError("anchor dates differ")
                y, valid = rank_targets(store, ix, before)
                d = np.searchsorted(ix, dates)
                sets[label] = (
                    x,
                    common,
                    dates,
                    ranks[:, a, names],
                    y[d, names],
                    valid[d, names],
                )
        finally:
            store.close()
        for seed_index, seed in enumerate(ALLOWED_SEEDS):
            for cell in ("ridge_score", "ridge_rich", "tree_score", "tree_rich"):
                directory = root / "residual" / cell / f"{fold}_seed_{seed}"
                result = read(directory / "result.json")
                if sha256_file(scaler_path) != result["binding"]["scaler_sha256"]:
                    raise ValueError("sealed scaler changed")
                record = {
                    "cell": cell,
                    "fold": fold,
                    "seed": seed,
                    "strength": result["strength"],
                    "iterations": result["iterations"],
                    "result_sha256": sha256_file(directory / "result.json"),
                    "models_sha256": {},
                }
                for label, (x, common, dates, anchors, y, valid) in sets.items():
                    anchor = anchors[seed_index]
                    matrix = (
                        np.column_stack(
                            (
                                anchor,
                                x,
                                (common[:, :, None] * anchor[:, None, :]).reshape(
                                    len(anchor), -1
                                ),
                            )
                        )
                        if cell.endswith("rich")
                        else anchor
                    )
                    correction = np.zeros_like(anchor)
                    for h in range(3):
                        model_path = (
                            directory
                            / f"head_{h}.{'npz' if cell.startswith('ridge') else 'txt'}"
                        )
                        record["models_sha256"][model_path.name] = sha256_file(
                            model_path
                        )
                        if cell.startswith("ridge"):
                            with np.load(model_path) as model:
                                correction[:, h] = (
                                    matrix @ model["beta"] + model["intercept"]
                                )
                        else:
                            model = lgb.Booster(model_file=str(model_path))
                            correction[:, h] = model.predict(
                                matrix,
                                num_iteration=result["iterations"][h],
                                num_threads=4,
                            )
                    prediction = (
                        anchor
                        if result["strength"] == 0
                        else anchor + result["strength"] * correction
                    )
                    for name, scores in (("anchor", anchor), ("corrected", prediction)):
                        record[f"{label}_{name}_ic"] = float(
                            np.mean(
                                [
                                    daily_ic(
                                        scores[valid[:, h], h],
                                        y[valid[:, h], h],
                                        dates[valid[:, h]],
                                    )
                                    for h in range(3)
                                ]
                            )
                        )
                expected = result["selection_scores"][
                    STRENGTHS.index(result["strength"])
                ]
                if abs(record["selection_corrected_ic"] - expected) > 1e-9:
                    raise ValueError("saved model does not reproduce selected IC")
                record["fit_minus_selection_ic"] = (
                    record["fit_corrected_ic"] - record["selection_corrected_ic"]
                )
                residual.append(record)
        print({"completed_diagnostics": fold}, flush=True)
    write_json_atomic(
        Path("docs/v2_foundation_training_diagnostics.json"),
        {
            "root": str(root),
            "script_sha256": sha256_file(Path(__file__)),
            "contract": "Post-hoc descriptive diagnostics only. Saved weights, scalers, selected strength/iteration and gates unchanged. No refitting or evaluation-label access. Neural fit IC is the recorded clean-fit probe; residual fit IC uses all valid fit observations. Fit/selection periods differ, so gaps include regime differences.",
            "neural": neural,
            "residual": residual,
        },
    )


if __name__ == "__main__":
    main()
