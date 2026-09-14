"""Causal forecast cache, economic arrays and fit-only return calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, HORIZONS
from .hedge_beta import resolve_hedge_beta
from .portfolio_program import fit_path, read
from .post_data_readouts import prepare_economics
from .score import parent_prelude_indices

HEADS = tuple(HORIZONS.index(h) for h in (3, 5, 10))


def forecast_block(root, arm, fold, dates, isins):
    design = read(root / "frozen_design.json")
    members, records, mask = [], [], None
    for seed in ALLOWED_SEEDS:
        reused = design["reused_fits"].get(arm, {}).get(fold, {}).get(str(seed))
        directory = (
            Path(reused["manifest"]["path"]).parent
            if reused
            else root / "prelude" / arm / f"seed_{seed}"
            if fold == "parent_prelude"
            else fit_path(root, arm, fold, seed)
        )
        manifest = read(directory / "run_manifest.json")
        score_path = directory / "scores/score_manifest.json"
        score_manifest = read(score_path)
        if (
            manifest["status"] != "completed"
            or manifest["seed"] != seed
            or manifest["fold"] != fold
            or score_manifest["store"]["manifest_sha256"]
            != design["store"]["manifest_sha256"]
        ):
            raise ValueError("forecast block is incomplete or bound to another source")
        if reused and (
            sha256_file(directory / "run_manifest.json") != reused["manifest"]["sha256"]
            or sha256_file(score_path) != reused["scores"]["sha256"]
        ):
            raise ValueError("sealed forecast reuse changed")
        values, valid = rr._score_artifact(
            directory / "scores",
            require_clean_transfer=True,
            expected_dates=dates,
            expected_isins=isins,
            expected_feature_schema_sha256=design["feature_schema_sha256"],
        )
        if mask is not None and not np.array_equal(mask, valid):
            raise ValueError("forecast seed populations differ")
        members.append(values)
        mask = valid
        records.append(
            {
                "seed": seed,
                "manifest": str(score_path),
                "sha256": sha256_file(score_path),
                "training_manifest_sha256": sha256_file(
                    directory / "run_manifest.json"
                ),
            }
        )
    return rr.rank_average_ensemble(members, mask), mask, records


def build_forecast_cache(root: Path, arm: str):
    """Require every registered chronological block; never backfill fitted scores."""
    design = read(root / "frozen_design.json")
    store = Path(design["store"]["root"])
    dates = np.load(store / "date_index.npy", allow_pickle=False)
    isins = tuple(np.load(store / "isin_index.npy", allow_pickle=False).tolist())
    _, _, evaluation, _, _ = rr._fold_indices(dates)
    blocks = {
        "parent_prelude": parent_prelude_indices(store),
        **{fold: evaluation[fold] for fold in DEVELOPMENT_FOLDS},
    }
    values, masks, sources = [], [], {}
    for fold, ix in blocks.items():
        score, mask, binding = forecast_block(root, arm, fold, dates[ix], isins)
        values.append(score)
        masks.append(mask)
        sources[fold] = binding
    indices = np.concatenate(list(blocks.values()))
    if not np.all(np.diff(indices) == 1):
        raise ValueError("out-of-fit forecasts have missing or overlapping dates")
    output = root / "cache" / arm
    output.mkdir(parents=True, exist_ok=False)
    for name, array in {
        "scores": np.concatenate(values),
        "mask": np.concatenate(masks),
        "indices": indices,
    }.items():
        np.save(output / f"{name}.npy", array, allow_pickle=False)
    write_json_atomic(
        output / "manifest.json",
        {
            "arm": arm,
            "sources": sources,
            "forecast_design_sha256": sha256_file(root / "frozen_design.json"),
            "first_date": str(dates[indices[0]]),
            "last_date": str(dates[indices[-1]]),
            "dates": len(indices),
            "files": {p.name: sha256_file(p) for p in output.glob("*.npy")},
            "information_boundary": "P through 2016-06-30 plus purge; F scores only original evaluation windows",
            "component_scores_retained_in_source_artifacts": True,
        },
    )


def normalized_ranks(scores, mask):
    """Normalize ensemble midranks by that date/head's observed population."""
    counts = mask.sum(axis=1, keepdims=True)
    return np.where(mask, 2 * (scores + 0.5) / np.maximum(counts, 1) - 1, 0.0)


def causal_risk(wealth, seen, hedge, beta, active):
    """Daily risk at t uses adjacent observed return endpoints t-60..t-1.

    Missing pairs are excluded, never treated as zero. Shrink residual variance
    toward the same-day active-name median with 20 pseudo-observations; unsupported
    names receive that median without becoming ineligible. No future fitting.
    """
    wealth = np.asarray(wealth, dtype=float)
    returns = np.full_like(wealth, np.nan)
    valid = seen[1:] & seen[:-1] & (wealth[:-1] > 0) & (wealth[1:] >= 0)
    np.divide(wealth[1:], wealth[:-1], out=returns[1:], where=valid)
    returns[1:] -= 1
    market = np.full(len(hedge), np.nan)
    ok = (
        np.isfinite(hedge[1:])
        & np.isfinite(hedge[:-1])
        & (hedge[:-1] > 0)
        & (hedge[1:] > 0)
    )
    np.divide(hedge[1:], hedge[:-1], out=market[1:], where=ok)
    market[1:] -= 1
    diagonal = np.empty_like(wealth)
    market_variance = np.empty(len(wealth))
    support = np.zeros_like(wealth, dtype=np.int16)
    for day in range(len(wealth)):
        x = market[max(1, day - 60) : day]
        y = returns[max(1, day - 60) : day]
        residual = y - x[:, None] * beta[day]
        finite = np.isfinite(residual)
        count = finite.sum(0)
        clean = np.where(finite, residual, 0)
        variance = np.maximum(
            (clean**2).sum(0) - clean.sum(0) ** 2 / np.maximum(count, 1), 0
        ) / np.maximum(count - 1, 1)
        supported = (count >= 20) & active[day]
        # Only early history can reach this fallback; the actual policy prelude
        # starts after years of market data. It is fixed, never fitted on the future.
        prior = float(np.median(variance[supported])) if supported.any() else 0.02**2
        diagonal[day] = np.maximum((count * variance + 20 * prior) / (count + 20), 1e-8)
        support[day] = count
        observed = x[np.isfinite(x)]
        market_variance[day] = (
            max(float(np.var(observed, ddof=1)), 1e-8)
            if len(observed) >= 20
            else 0.015**2
        )
    return diagonal, market_variance, support


def open_policy_inputs(root, indices, scores, mask):
    """Accepted repaired economics, original source archives and causal history."""
    economic = prepare_economics(root)
    context = rr._open_ledger_replay(economic)
    try:
        policy, _ = rr.load_selected_policy(
            Path(economic["execution_policy"]["root"]),
            expected_result_sha256=economic["execution_policy"]["result_sha256"],
        )
        inputs = rr._evaluation_inputs(
            context.store,
            indices,
            scores,
            mask,
            context.cdi,
            context.bova11.close_by_session,
            context.bova11_binding,
            context.lending_borrow,
            {"forecast_design": sha256_file(root / "frozen_design.json")},
            transfer_chronology_clean=True,
            execution_policy=policy,
        )
        beta, _ = resolve_hedge_beta(
            inputs.hedge_beta,
            inputs.hedge_beta_valid,
            history=inputs.hedge_beta_history,
        )
        history = np.arange(max(0, indices[0] - 61), indices[-1] + 1)
        start = indices[0] - history[0]
        # Earlier beta coordinates only serve the discarded risk warm-up rows.
        extended_beta = np.concatenate(
            (np.ones((start, len(inputs.security_ids))), beta)
        )
        diagonal, factor, support = causal_risk(
            context.store.read("shareholder_wealth_close", history),
            context.store.read("shareholder_wealth_valid", history),
            context.bova11.close_by_session[history],
            extended_beta,
            context.store.read("active", history),
        )
        prior_cdi = context.cdi[indices - 1].copy()
        references = context.store.read("prior_reference_close", indices)
        return (
            inputs,
            beta,
            diagonal[start:],
            factor[start:],
            support[start:],
            prior_cdi,
            references,
        )
    finally:
        context.store.close()


@dataclass(frozen=True)
class Calibration:
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    covariance: np.ndarray | None = None
    diagnostics: dict | None = None

    def predict(self, ranks):
        if len(self.coefficient) == 1:
            ranks = ranks.mean(-1, keepdims=True)
        return (ranks - self.mean) / self.scale @ self.coefficient + self.intercept


def fit_calibration(
    ranks,
    valid,
    returns5,
    target_valid,
    cdi,
    fit_rows,
    *,
    benchmark_excess5=None,
    beta=None,
    equal_rank=False,
):
    """Date-weighted ridge. Both entry and five-session endpoint stay in fit."""
    rows = np.asarray(fit_rows)
    if not len(rows) or not np.all(np.diff(rows) == 1):
        raise ValueError("calibration fit dates must form a chronological window")
    eligible = rows[rows + 5 <= rows[-1]]
    masks = valid[eligible] & target_valid[eligible]
    cash5 = np.array([np.prod(1 + cdi[t + 1 : t + 6]) - 1 for t in eligible])
    outcome = returns5[eligible] - cash5[:, None]
    benchmark_missing = 0
    if benchmark_excess5 is not None:
        observed = np.isfinite(benchmark_excess5[eligible])
        benchmark_missing = int((masks & ~observed[:, None]).sum())
        masks &= observed[:, None]
        # Beta is fixed at the original decision, never estimated on its label.
        outcome = outcome - beta[eligible] * benchmark_excess5[eligible, None]
    count = masks.sum(1)
    if not np.any(count):
        raise ValueError("no within-fit calibration outcomes")
    weights = masks / np.maximum(count[:, None], 1)
    weights /= max(np.count_nonzero(count), 1)
    x = ranks[eligible]
    if equal_rank:
        x = x.mean(-1, keepdims=True)
    mean = (weights[..., None] * x).sum((0, 1))
    scale = np.maximum(
        np.sqrt((weights[..., None] * (x - mean) ** 2).sum((0, 1))), 1e-6
    )
    z = (x - mean) / scale
    y = np.where(masks, outcome / 5, 0)
    intercept = float((weights * y).sum())
    gram = np.einsum("dn,dni,dnj->ij", weights, z, z)
    lhs = gram + 1e-3 * np.eye(x.shape[-1])
    rhs = np.einsum("dn,dni,dn->i", weights, z, y - intercept)
    coefficient = np.linalg.solve(lhs, rhs)
    residual = np.where(masks, y - (z @ coefficient + intercept), 0)
    design = np.concatenate((np.ones((*z.shape[:-1], 1)), z), axis=-1)
    daily_score = np.einsum("dn,dnk,dn->dk", weights, design, residual)
    clusters = np.add.reduceat(daily_score, np.arange(0, len(eligible), 20))
    bread = np.zeros((len(coefficient) + 1, len(coefficient) + 1))
    bread[0, 0], bread[1:, 1:] = 1.0, lhs
    inverse = np.linalg.inv(bread)
    covariance = inverse @ (clusters.T @ clusters) @ inverse.T
    covariance *= len(clusters) / max(len(clusters) - 1, 1)
    diagnostics = {
        "fit_dates": int(np.count_nonzero(count)),
        "fit_observations": int(masks.sum()),
        "benchmark_missing_label_observations": benchmark_missing,
        "ridge_condition_number": float(np.linalg.cond(lhs)),
        "predictor_correlation": gram.tolist(),
        "weighted_daily_rmse_bps": float(np.sqrt((weights * residual**2).sum()) * 1e4),
        "cluster_sessions": 20,
        "clusters": len(clusters),
    }
    return Calibration(mean, scale, coefficient, intercept, covariance, diagnostics)
