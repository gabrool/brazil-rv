"""Prior-only peer ridge diagnostics; zero neural advancement weight."""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .contract import (
    DEVELOPMENT_FOLDS,
    REGISTERED_PRIMARY_TARGET,
    REGISTERED_PRIMARY_TARGET_MASK,
)
from .data_roots import resolve_external_root
from .features import monthly_cluster_labels
from .normalization import average_ranks, midrank_unit_interval
from .pathway_statistics import infer
from .round5_derived import identity_axes
from .round7_corrector import mature_before
from .round7_data import PROJECT
from .round7_program import read
from .round7_readouts import ensemble
from .store import open_store_for_samples
from .train import _cli_stage_indices
from .validate_pipeline import _window_target_mask

ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0)


def leave_one_out(values, known, members):
    observed = known & members
    clean = np.where(observed, values, 0.0)
    counts = observed.sum() - observed
    return np.divide(
        clean.sum() - clean, counts, out=np.full(values.shape, np.nan), where=counts > 0
    )


def peer_features(
    dates,
    active,
    magnitude,
    magnitude_valid,
    sectors,
    market,
    market_valid,
    market_names,
):
    # Product of the two admitted, unclipped physical channels reconstructs
    # t-1 return. Its volatility support is narrower than all original returns.
    known = magnitude_valid[..., :2].all(-1) & active
    returns = magnitude[..., 0].astype(float) * magnitude[..., 1]
    residual = np.zeros_like(returns)
    for day in range(len(dates)):
        if known[day].any():
            residual[day] = returns[day] - np.median(returns[day, known[day]])
    # monthly_cluster_labels consumes ending-session rows strictly before t.
    # Input here is already decision-dated: move t's t-1 observation back to
    # its ending-session slot. The last placeholder is never consumed.
    end_returns = np.zeros_like(residual)
    end_valid = np.zeros_like(known)
    end_active = np.zeros_like(active)
    end_returns[:-1], end_valid[:-1], end_active[:-1] = (
        residual[1:],
        known[1:],
        active[1:],
    )
    clusters = monthly_cluster_labels(dates, end_returns, end_valid, end_active)
    names = [
        f"{group}_residual_lag_{lag}"
        for group in ("cluster", "sector")
        for lag in range(1, 6)
    ]
    names += [f"liquid_leader_lag_{lag}_times_beta" for lag in range(1, 6)]
    names += [
        "adr_gap_mean_times_beta",
        "adr_gap_mean_times_fx_exposure",
        "adr_gap_mean_times_adr_flag",
    ]
    result = np.full((*active.shape, len(names)), np.nan, np.float32)
    adr = market_names.index("adr_return_gap_1")
    fx = market_names.index("exposure_fx")
    listed = market_names.index("adr_listed_flag")
    for day in range(len(dates)):
        for offset, groups, missing in ((0, clusters[day], -1), (5, sectors[day], "")):
            for group in np.unique(groups[active[day]]):
                if group == missing:
                    continue
                members = active[day] & (groups == group)
                for lag in range(1, min(5, day + 1) + 1):
                    row = day - lag + 1
                    value = leave_one_out(residual[row], known[row], members)
                    result[day, members, offset + lag - 1] = value[members]
        liquid = np.flatnonzero(active[day] & magnitude_valid[day, :, 2])
        leaders = liquid[np.argsort(-magnitude[day, liquid, 2], kind="stable")[:10]]
        members = np.zeros(active.shape[1], bool)
        members[leaders] = True
        beta = np.where(magnitude_valid[day, :, 3], magnitude[day, :, 3], np.nan)
        for lag in range(1, min(5, day + 1) + 1):
            row = day - lag + 1
            result[day, :, 10 + lag - 1] = (
                leave_one_out(residual[row], known[row], members) * beta
            )
        adr_mean = leave_one_out(
            market[day, :, adr], market_valid[day, :, adr], active[day]
        )
        result[day, :, 15] = adr_mean * beta
        for column, exposure in ((16, fx), (17, listed)):
            result[day, :, column] = adr_mean * np.where(
                market_valid[day, :, exposure], market[day, :, exposure], np.nan
            )
    result[~active] = np.nan
    return result, names


def fit_ridge(x, y, dates, alpha):
    """Training-only winsorization/scaling; mean imputation plus observed mask."""
    _, inverse, count = np.unique(dates, return_inverse=True, return_counts=True)
    weights = 1 / count[inverse]
    weights /= weights.sum()
    lower, upper, mean, scale = [], [], [], []
    for column in x.T:
        valid = np.isfinite(column)
        lo, hi = (
            np.quantile(column[valid], (0.005, 0.995)) if valid.any() else (0.0, 0.0)
        )
        clipped = np.clip(column[valid], lo, hi)
        w = weights[valid]
        mu = float(np.average(clipped, weights=w)) if valid.any() else 0.0
        sd = (
            float(np.sqrt(np.average((clipped - mu) ** 2, weights=w)))
            if valid.any()
            else 1.0
        )
        lower.append(lo)
        upper.append(hi)
        mean.append(mu)
        scale.append(max(sd, 1e-6))
    state = {
        "lower": np.array(lower),
        "upper": np.array(upper),
        "mean": np.array(mean),
        "scale": np.array(scale),
    }
    z = transform(x, state)
    matrix = z.T @ (weights[:, None] * z)
    penalty = np.eye(z.shape[1]) * alpha
    penalty[-1, -1] = 0
    state["coef"] = np.linalg.solve(matrix + penalty, z.T @ (weights * y))
    return state


def transform(x, state):
    valid = np.isfinite(x)
    z = (np.clip(x, state["lower"], state["upper"]) - state["mean"]) / state["scale"]
    return np.column_stack((np.where(valid, z, 0.0), valid, np.ones(len(x))))


def correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else np.nan


def daily_metrics(target, predictions, dates, baseline):
    rows = []
    for day in np.unique(dates):
        use = dates == day
        if use.sum() < 20:
            continue
        y = average_ranks(target[use])
        base = average_ranks(baseline[use])
        nuisance = np.column_stack((np.ones(use.sum()), base))
        residual_y = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
        row = {"date_index": int(day), "names": int(use.sum())}
        for name, values in predictions.items():
            x = average_ranks(values[use])
            row[name] = correlation(x, y)
            if name == "peer_only":
                residual_x = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
                row["partial_peer_ic_given_s0"] = correlation(residual_x, residual_y)
        rows.append(
            {
                k: None if isinstance(v, float) and not np.isfinite(v) else v
                for k, v in row.items()
            }
        )
    return rows


def run(root, output):
    design = read(root / "frozen_design.json")
    store_root = resolve_external_root(design["store"]["root"])[0]
    original = resolve_external_root(design["original_run"]["root"])[0]
    original_design = read(original / "frozen_design.json")
    manifest = read(store_root / "manifest.json")
    indices = np.arange(manifest["axes"]["date_count"])
    store, access = open_store_for_samples(
        store_root,
        indices,
        purpose="evaluation",
        history_lookbacks=60,
        history_end_offsets=0,
    )
    output.mkdir(parents=True, exist_ok=False)
    try:
        active = store.read("active", indices)
        magnitude = store.read("sidecar_magnitudes_values", indices)
        magnitude_valid = store.read("sidecar_magnitudes_valid", indices)
        identity_record = read(PROJECT / "docs/v2_round5_cvm_final_acceptance.json")[
            "identity_proof"
        ]["final_identity"]
        identity = resolve_external_root(identity_record["path"])[0]
        if sha256_file(identity) != identity_record["sha256"]:
            raise ValueError("accepted historical sector identity changed")
        sectors, _ = identity_axes(
            pl.read_parquet(identity), store.dates.astype(object).tolist(), store.isins
        )
        x, names = peer_features(
            store.dates,
            active,
            magnitude,
            magnitude_valid,
            sectors,
            store.read("sidecar_cross_market_values", indices),
            store.read("sidecar_cross_market_valid", indices),
            manifest["feature_names"]["sidecar_cross_market"],
        )
        write_json_atomic(
            output / "feature_audit.json",
            {
                "features": names,
                "valid_active_stock_days": {
                    n: int((np.isfinite(x[..., i]) & active).sum())
                    for i, n in enumerate(names)
                },
                "store_manifest_sha256": sha256_file(store_root / "manifest.json"),
                "sector_identity": identity_record,
                "support": "admitted magnitude product; narrower volatility support, not a claim to reproduce original cluster memberships",
                "timing": "lag1 ends t-1 with no extra delay; monthly clusters include only prior completed sessions; PIT sector and current decision-eligible peers",
                "promotion_weight": 0,
            },
        )

        def sample(ix):
            mask = _window_target_mask(
                store.read(REGISTERED_PRIMARY_TARGET_MASK, ix), ix
            )
            target = store.read_target(REGISTERED_PRIMARY_TARGET, ix, valid_mask=mask)[
                ..., 2:5
            ]
            valid = active[ix] & mask[..., 2:5].all(-1)
            d, n = np.nonzero(valid)
            return x[ix[d], n], target[d, n].mean(-1), ix[d], d, n

        fit, selection, _, _ = _cli_stage_indices(store_root, "P", "pretrain_internal")
        train_x, train_y, train_days, _, _ = sample(fit)
        val_x, val_y, val_days, _, _ = sample(selection)
        validation = {}
        for alpha in ALPHAS:
            model = fit_ridge(train_x, train_y, train_days, alpha)
            prediction = transform(val_x, model) @ model["coef"]
            rows = daily_metrics(
                val_y, {"peer_only": prediction}, val_days, np.zeros(len(val_y))
            )
            validation[alpha] = float(np.nanmean([r["peer_only"] for r in rows]))
        alpha = max(ALPHAS, key=lambda a: validation[a])
        write_json_atomic(
            output / "calibration.json",
            {
                "alpha": alpha,
                "stage_p_validation_ic": validation,
                "objective": "equal-date average squared loss plus alpha times squared non-intercept coefficients",
            },
        )
        panels, bindings = [], {}
        for fold in DEVELOPMENT_FOLDS:
            ix = _cli_stage_indices(store_root, "F", fold)[2]
            raw, target, days, d, n = sample(ix)
            scores, valid, records, _ = ensemble(
                original,
                original_design,
                "A0",
                fold,
                (11, 29, 47),
                store.dates[ix],
                store.isins,
            )
            matched = valid[d, n, 2:5].all(-1)
            score = np.zeros(active[ix].shape)
            for day in range(len(ix)):
                eligible = active[ix[day]] & valid[day, :, 2:5].all(-1)
                if eligible.any():
                    score[day, eligible] = np.stack(
                        [
                            midrank_unit_interval(scores[day, eligible, h])
                            for h in (2, 3, 4)
                        ]
                    ).mean(0)
            panels.append(
                {
                    "fold": fold,
                    "first": int(ix[0]),
                    "X": raw[matched],
                    "y": target[matched],
                    "days": days[matched],
                    "score": score[d, n][matched],
                }
            )
            bindings[fold] = records
        completed = []
        for index, current in enumerate(panels[1:], 1):
            preceding = panels[:index]
            days = np.concatenate([p["days"] for p in preceding])
            mature = mature_before(days, current["first"])
            y = np.concatenate([p["y"] for p in preceding])[mature]
            features = np.concatenate([p["X"] for p in preceding])[mature]
            score = np.concatenate([p["score"] for p in preceding])[mature, None]
            current_score = current["score"][:, None]
            predictions = {"s0": current["score"]}
            for name, train, evaluation in (
                ("score_only", score, current_score),
                ("peer_only", features, current["X"]),
                (
                    "score_plus_peer",
                    np.column_stack((score, features)),
                    np.column_stack((current_score, current["X"])),
                ),
            ):
                model = fit_ridge(train, y, days[mature], alpha)
                predictions[name] = transform(evaluation, model) @ model["coef"]
            for seed in range(10):
                # Flip entire 63-session feature blocks, independently of labels.
                # This is a diagnostic temporal-alignment null, not a claim of
                # exact exchangeability or a formal randomization p-value.
                signs = np.random.default_rng(4200 + seed).choice(
                    [-1.0, 1.0], size=len(indices) // 63 + 1
                )
                train = features * signs[days[mature] // 63, None]
                evaluation = current["X"] * signs[current["days"] // 63, None]
                model = fit_ridge(
                    np.column_stack((score, train)), y, days[mature], alpha
                )
                predictions[f"null_{seed}"] = (
                    transform(np.column_stack((current_score, evaluation)), model)
                    @ model["coef"]
                )
            daily = daily_metrics(
                current["y"], predictions, current["days"], current["score"]
            )
            result = {
                "fold": current["fold"],
                "fit_stock_days": len(y),
                "maximum_label_endpoint": int(days[mature].max()) + 10,
                "evaluation_first_index": current["first"],
                "daily": daily,
            }
            write_json_atomic(output / f"{current['fold']}.json", result)
            completed.append(result)
            print({"fold": current["fold"], "fit_stock_days": len(y)}, flush=True)
        paired = {
            name: infer(
                tuple(
                    np.array([r[name] for r in f["daily"]], float)
                    - np.array([r["score_only"] for r in f["daily"]], float)
                    for f in completed
                )
            )
            for name in ("score_plus_peer", *(f"null_{s}" for s in range(10)))
        }
        result = {
            "status": "complete",
            "folds": completed,
            "paired_vs_score_only": paired,
            "sources": bindings,
            "access": access.payload(),
            "alpha": alpha,
            "unavailable": "F1 and pre-F1 OOF score controls; no in-sample replacement used",
            "interpretation": "linear supporting evidence only; partial correlations residualize both sides within each evaluation cross-section and are descriptive, never fitted forecasts; negative results cannot veto neural comparisons",
        }
        write_json_atomic(output / "result.json", result)
        return result
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)


if __name__ == "__main__":
    main()
