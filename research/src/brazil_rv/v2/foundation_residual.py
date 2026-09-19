"""Fixed learners for the registered foundation residual-information probe."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


STRENGTHS = (0.0, 0.1, 0.25, 0.5, 1.0)


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
