"""Checkpoint-selection statistics: trailing-window smoothing and selection noise.

The round-7 trainer selects the raw checkpoint with the best mean common-population
D3/D5/D10 IC on a 55-session selection window whose D10 labels end ten sessions
early, so at most 45 defined days decide a fit. These helpers quantify how well that
statistic can distinguish neighbouring epochs, and implement the trailing-window
smoothed selection score used by ``TrainingRecipe.selection_smoothing``. Nothing
here reads evaluation labels; every input is a saved fit/selection history.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def smoothed_selection(scores, window):
    """Trailing-window mean of per-epoch selection scores and its centre epoch.

    ``scores`` are the raw per-epoch selection means, epoch one first. For the
    epoch at 0-based index ``i`` the window covers ``max(0, i - window + 1)..i``;
    the returned centre is the 0-based index of that window's middle epoch, the
    epoch whose expected quality the trailing mean estimates. ``window == 1``
    reproduces raw selection exactly (the centre is the current epoch).
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("selection scores must be a one-dimensional epoch sequence")
    if window < 1:
        raise ValueError("selection smoothing window must be at least one epoch")
    smoothed = np.empty(values.shape)
    centre = np.empty(values.shape, dtype=np.int64)
    for i in range(values.size):
        start = max(0, i - window + 1)
        smoothed[i] = values[start : i + 1].mean()
        centre[i] = start + (i - start) // 2
    return smoothed, centre


def newey_west_mean_se(values, lags=10):
    """Standard error of the mean of a daily series with ``lags`` Bartlett lags.

    Undefined days (NaN/None) are excluded from the mean but keep their place in
    the lag structure, matching the archived-diagnostics convention.
    """
    series = np.asarray(
        [np.nan if v is None else float(v) for v in values], dtype=np.float64
    )
    known = np.isfinite(series)
    count = int(known.sum())
    if count < 2:
        return None
    centered = np.where(known, series - series[known].mean(), 0.0)
    variance = float(centered @ centered)
    for lag in range(1, min(lags, series.size - 1) + 1):
        variance += (
            2.0 * (1.0 - lag / (lags + 1)) * float(centered[lag:] @ centered[:-lag])
        )
    return float(np.sqrt(max(0.0, variance)) / count)


def _daily(record):
    return np.asarray(
        [np.nan if v is None else float(v) for v in record["selection"]["daily_ic"]],
        dtype=np.float64,
    )


def selection_noise_summary(history, *, smoothing=3, lags=10):
    """Quantify checkpoint-selection resolution for one saved round-7 history.

    ``history`` is the list of per-epoch records the trainer writes: each has
    ``epoch``, ``selection`` (``daily_ic`` and ``mean_ic``), optionally
    ``clean_fit_probe`` and the ``selected`` flag. The summary reports the
    Newey-West standard error of the selection mean at the selected epoch, the
    paired standard error of the difference between the selected epoch and the
    best-scoring epoch, how many epochs lie within one paired standard error of
    the maximum, and what a trailing-``smoothing`` rule would have selected.
    """
    if not history:
        raise ValueError("history is empty")
    epochs = [int(r["epoch"]) for r in history]
    means = np.asarray([float(r["selection"]["mean_ic"]) for r in history])
    daily = np.stack([_daily(r) for r in history])
    selected = [r["epoch"] for r in history if r.get("selected")]
    selected_epoch = int(selected[-1]) if selected else int(epochs[int(means.argmax())])
    selected_index = epochs.index(selected_epoch)
    best_index = int(means.argmax())
    smoothed, centre = smoothed_selection(means, smoothing)
    smoothed_index = int(centre[int(smoothed.argmax())])
    paired_se = [
        newey_west_mean_se(daily[best_index] - daily[i], lags)
        for i in range(len(epochs))
    ]
    within_one_se = [
        i
        for i in range(len(epochs))
        if i == best_index
        or (paired_se[i] is not None and means[best_index] - means[i] <= paired_se[i])
    ]
    probe = [
        r.get("clean_fit_probe", {}).get("mean_ic")
        if isinstance(r.get("clean_fit_probe"), dict)
        else None
        for r in history
    ]
    probe_values = np.asarray(
        [np.nan if v is None else float(v) for v in probe], dtype=np.float64
    )
    fit_selection_correlation = None
    if np.isfinite(probe_values).sum() >= 3:
        known = np.isfinite(probe_values)
        if means[known].std() > 0 and probe_values[known].std() > 0:
            fit_selection_correlation = float(
                np.corrcoef(means[known], probe_values[known])[0, 1]
            )
    defined = np.isfinite(daily).all(axis=0)
    return {
        "epochs": len(epochs),
        "selected_epoch": selected_epoch,
        "best_epoch": int(epochs[best_index]),
        "smoothed_selected_epoch": int(epochs[smoothed_index]),
        "smoothing_window": int(smoothing),
        "selected_mean_ic": float(means[selected_index]),
        "best_mean_ic": float(means[best_index]),
        "epoch_mean_ic": means.tolist(),
        "epoch_mean_spread": float(means.max() - means.min()),
        "defined_selection_days": int(defined.sum()),
        "selection_days": int(daily.shape[1]),
        "selected_epoch_mean_se": newey_west_mean_se(daily[selected_index], lags),
        "paired_se_best_minus_selected": paired_se[selected_index],
        "epochs_within_one_paired_se_of_best": len(within_one_se),
        "selected_within_one_paired_se_of_best": selected_index in within_one_se,
        "fit_probe_mean_ic": [
            None if not np.isfinite(v) else float(v) for v in probe_values
        ],
        "fit_selection_epoch_correlation": fit_selection_correlation,
        "newey_west_lags": int(lags),
    }


def _stage_of(history_path):
    manifest = history_path.with_name("run_manifest.json")
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload.get("stage")
    return None


def audit_histories(paths, *, smoothing=3, lags=10, stage=None):
    """Summarize many saved histories and pool the resolution statistics."""
    rows = []
    for path in sorted(Path(p) for p in paths):
        fit_stage = _stage_of(path)
        if stage is not None and fit_stage not in (None, stage):
            continue
        history = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(history, list) or not history:
            continue
        if any("selection" not in record for record in history):
            continue
        summary = selection_noise_summary(history, smoothing=smoothing, lags=lags)
        summary["history"] = str(path)
        summary["stage"] = fit_stage
        rows.append(summary)
    if not rows:
        return {"fits": 0, "rows": []}

    def collect(key):
        return np.asarray(
            [r[key] for r in rows if r[key] is not None], dtype=np.float64
        )

    se = collect("selected_epoch_mean_se")
    paired = collect("paired_se_best_minus_selected")
    spread = collect("epoch_mean_spread")
    within = collect("epochs_within_one_paired_se_of_best")
    ratios = np.asarray(
        [
            r["epoch_mean_spread"] / r["selected_epoch_mean_se"]
            for r in rows
            if r["selected_epoch_mean_se"]
        ],
        dtype=np.float64,
    )
    pooled = {
        "fits": len(rows),
        "median_selected_epoch": float(np.median(collect("selected_epoch"))),
        "median_epochs_completed": float(np.median(collect("epochs"))),
        "median_defined_selection_days": float(
            np.median(collect("defined_selection_days"))
        ),
        "median_selection_mean_se": float(np.median(se)) if se.size else None,
        "median_epoch_mean_spread": float(np.median(spread)) if spread.size else None,
        "median_spread_over_se": float(np.median(ratios)) if ratios.size else None,
        "median_paired_se_best_minus_selected": float(np.median(paired))
        if paired.size
        else None,
        "median_epochs_within_one_paired_se_of_best": float(np.median(within))
        if within.size
        else None,
        "fraction_selected_within_one_paired_se_of_best": float(
            np.mean([r["selected_within_one_paired_se_of_best"] for r in rows])
        ),
        "fraction_smoothed_rule_changes_selection": float(
            np.mean([r["smoothed_selected_epoch"] != r["selected_epoch"] for r in rows])
        ),
        "median_fit_selection_epoch_correlation": float(
            np.median(collect("fit_selection_epoch_correlation"))
        )
        if collect("fit_selection_epoch_correlation").size
        else None,
        "smoothing_window": int(smoothing),
        "newey_west_lags": int(lags),
        "interpretation": (
            "The selection statistic cannot resolve epochs whose true quality differs "
            "by less than its standard error. When the between-epoch spread is of the "
            "order of that error, and several epochs lie within one paired standard "
            "error of the maximum, the chosen checkpoint is largely a draw from the "
            "window's noise rather than a measured quality ranking. Evaluation labels "
            "are never read here."
        ),
    }
    return {"pooled": pooled, "rows": rows}
