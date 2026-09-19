"""Fixed learners for the registered foundation residual-information probe."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS
from .portfolio_inputs import HEADS, normalized_ranks
from .portfolio_program import read
from .research_rounds import _score_artifact
from .store import open_store_for_samples
from .train import rank_average_ensemble


STRENGTHS = (0.0, 0.1, 0.25, 0.5, 1.0)


def build_anchors(root):
    """Assemble per-seed genuine OOS history; no fitted-score fallback."""
    from pathlib import Path

    design = read(root / "frozen_design.json")
    store_root = Path(design["store"]["root"])
    prior = Path(design["prior_decision_root"])
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("residual store binding differs")
    dates = np.load(store_root / "date_index.npy", allow_pickle=False)
    isins = np.load(store_root / "isin_index.npy", allow_pickle=False)
    schema = read(store_root / "manifest.json")["metadata"]["feature_schema"]["sha256"]
    original = read(prior / "cache/C6/manifest.json")["sources"]["parent_prelude"]
    panels, bindings, common_indices, common_mask = [], [], None, None
    for seed in ALLOWED_SEEDS:
        sources = [root / "residual_sources/prelude/C6" / f"seed_{seed}"]
        sources += [
            prior / "phase3/fits/C6/neutral" / f"{fold}_seed_{seed}"
            for fold in DEVELOPMENT_FOLDS
        ]
        pieces, indices, masks = [], [], []
        for block, source in enumerate(sources):
            score_dir = source / "scores"
            manifest = read(score_dir / "score_manifest.json")
            training = read(source / "run_manifest.json")
            if (
                manifest["store"]["manifest_sha256"]
                != design["store"]["manifest_sha256"]
            ):
                raise ValueError("anchor uses another store")
            if block == 0:
                expected = next(v for v in original if v["seed"] == seed)
                if sha256_file(score_dir / "score_manifest.json") != expected["sha256"]:
                    raise ValueError("unsealed prelude")
                parent = read(
                    root
                    / "residual_sources/parents/C6_bootstrap"
                    / f"seed_{seed}/run_manifest.json"
                )
                last = max(
                    v["last_index"]
                    for v in parent["checkpoint_input_contract"]["selection"][
                        "segments"
                    ]
                )
                if last + 10 >= manifest["dataset"]["first_date_index"]:
                    raise ValueError("prelude labels overlap forecasts")
                checkpoint_hash = parent["artifacts"]["raw_patience.pt"]
            else:
                checkpoint_hash = training["artifacts"]["selected.pt"]
                if training["status"] != "completed" or training["seed"] != seed:
                    raise ValueError("incomplete anchor fit")
            if manifest["checkpoint"]["sha256"] != checkpoint_hash:
                raise ValueError("anchor checkpoint binding differs")
            block_dates = np.load(score_dir / "date_index.npy", allow_pickle=False)
            ix = np.searchsorted(dates, block_dates)
            if np.any(ix >= len(dates)) or not np.array_equal(dates[ix], block_dates):
                raise ValueError("anchor date identity differs")
            values, mask = _score_artifact(
                score_dir,
                require_clean_transfer=True,
                expected_dates=dates[ix],
                expected_isins=tuple(isins),
                expected_feature_schema_sha256=schema,
            )
            values, mask = values[..., HEADS], mask[..., HEADS]
            pieces.append(normalized_ranks(rank_average_ensemble([values], mask), mask))
            masks.append(mask)
            indices.append(ix)
            bindings.append(
                {
                    "seed": seed,
                    "path": str(score_dir),
                    "manifest_sha256": sha256_file(score_dir / "score_manifest.json"),
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
        ix, mask = np.concatenate(indices), np.concatenate(masks)
        if not np.all(np.diff(ix) == 1):
            raise ValueError("OOS anchors overlap or omit dates")
        if common_indices is not None and (
            not np.array_equal(ix, common_indices)
            or not np.array_equal(mask, common_mask)
        ):
            raise ValueError("OOS seed populations differ")
        common_indices, common_mask = ix, mask
        panels.append(np.concatenate(pieces).astype(np.float32))
    store, _ = open_store_for_samples(
        store_root,
        common_indices,
        purpose="evaluation",
        history_lookbacks=60,
        history_end_offsets=0,
    )
    try:
        active = store.read("active", common_indices)
        if not np.array_equal(common_mask, np.repeat(active[..., None], 3, axis=-1)):
            raise ValueError("anchor panel loses eligible names")
    finally:
        store.close()
    destination = root / "residual"
    destination.mkdir(exist_ok=True)
    path = destination / "anchors.npz"
    np.savez(path, ranks=np.stack(panels), indices=common_indices, active=active)
    write_json_atomic(
        destination / "anchors.json",
        {
            "sources": bindings,
            "sha256": sha256_file(path),
            "seeds": list(ALLOWED_SEEDS),
            "first_date": str(dates[common_indices[0]]),
            "last_date": str(dates[common_indices[-1]]),
            "store": design["store"],
        },
    )


def date_weights(dates):
    """Equal total weight per observed date, with mean sample weight one."""
    _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / weights.mean()


def fit_ridge(x, residual, dates):
    """Average weighted MSE plus .1 ||beta||²; intercept is unpenalized."""
    x = np.asarray(x, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    weights = date_weights(dates)
    weights /= weights.sum()
    mean_x = weights @ x
    mean_y = weights @ residual
    centered = x - mean_x
    gram = centered.T @ (weights[:, None] * centered)
    gram.flat[:: gram.shape[0] + 1] += 0.1
    beta = np.linalg.solve(gram, centered.T @ (weights * (residual - mean_y)))
    return beta, float(mean_y - mean_x @ beta)


def daily_ic(prediction, target, dates):
    """Equal-date Spearman IC; constant predictions contribute zero."""
    result = []
    for date in np.unique(dates):
        mask = dates == date
        if mask.sum() < 2:
            continue
        p, y = rankdata(prediction[mask]), rankdata(target[mask])
        p, y = p - p.mean(), y - y.mean()
        denominator = np.linalg.norm(p) * np.linalg.norm(y)
        result.append(float(p @ y / denominator) if denominator else 0.0)
    return float(np.mean(result)) if result else 0.0


def choose_strength(anchor, correction, target, valid, dates):
    """Use preceding selection labels only; one strength across all three heads."""
    scores = []
    for strength in STRENGTHS:
        prediction = anchor if strength == 0 else anchor + strength * correction
        scores.append(
            np.mean(
                [
                    daily_ic(
                        prediction[valid[:, h], h],
                        target[valid[:, h], h],
                        dates[valid[:, h]],
                    )
                    for h in range(3)
                ]
            )
        )
    return STRENGTHS[int(np.argmax(scores))], [float(v) for v in scores]


def fit_tree(
    x, residual, dates, selection_x, selection_target, selection_anchor, selection_dates
):
    """Fixed seven-leaf tree; stop on prior-selection IC of the full correction."""
    from .gbdt import require_lightgbm

    lgb = require_lightgbm()
    training = lgb.Dataset(x, label=residual, weight=date_weights(dates))
    selection = lgb.Dataset(
        selection_x, label=selection_target - selection_anchor, reference=training
    )

    def metric(prediction, _dataset):
        return (
            "anchor_ic",
            daily_ic(selection_anchor + prediction, selection_target, selection_dates),
            True,
        )

    return lgb.train(
        {
            "objective": "regression",
            "metric": "None",
            "max_depth": 3,
            "num_leaves": 7,
            "learning_rate": 0.03,
            "min_data_in_leaf": 200,
            "lambda_l2": 10.0,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "max_bin": 255,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 4,
            "seed": 29,
            "verbosity": -1,
        },
        training,
        num_boost_round=500,
        valid_sets=[selection],
        feval=metric,
        callbacks=[lgb.early_stopping(30, first_metric_only=True, verbose=False)],
    )
