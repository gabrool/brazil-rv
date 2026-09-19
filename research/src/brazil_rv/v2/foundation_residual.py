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


def rank_targets(store, rows, before):
    """Read only labels ending strictly before the next decision window."""
    from .contract import REGISTERED_PRIMARY_TARGET, REGISTERED_PRIMARY_TARGET_MASK

    mask = store.read(REGISTERED_PRIMARY_TARGET_MASK, rows).copy()
    for h, horizon in enumerate((1, 2, 3, 5, 10)):
        mask[..., h] &= (np.asarray(rows) + horizon < before)[:, None]
    values = store.read_target(REGISTERED_PRIMARY_TARGET, rows, valid_mask=mask)
    values, mask = values[..., HEADS], mask[..., HEADS]
    mask &= store.read("active", rows)[..., None]
    ranked = np.zeros(values.shape, np.float32)
    for d in range(len(rows)):
        for h in range(3):
            valid = mask[d, :, h]
            count = valid.sum()
            if count:
                ranked[d, valid, h] = (
                    2 * (rankdata(values[d, valid, h]) - 0.5) / count - 1
                )
    return ranked, mask


def run_residual(root, folds):
    """Four registered cells; prior-only selection, no evaluation-label tuning."""
    from pathlib import Path
    from time import perf_counter

    from .research_rounds import _git_identity
    from .train import _cli_stage_indices

    implementation = _git_identity()
    design = read(root / "frozen_design.json")
    source = Path(design["store"]["root"])
    anchor_manifest = read(root / "residual/anchors.json")
    if sha256_file(root / "residual/anchors.npz") != anchor_manifest["sha256"]:
        raise ValueError("residual anchor cache changed")
    with np.load(root / "residual/anchors.npz") as cache:
        anchor_ranks, anchor_indices = cache["ranks"], cache["indices"]
    roster = read(root / "waves/input.json")["cells"]["TE_full"]
    for fold in folds:
        fit, selection, evaluation, _ = _cli_stage_indices(source, "F", fold)
        rows = np.unique(np.r_[fit, selection, evaluation])
        target_window = np.arange(fit[0], evaluation[0])
        store, _ = open_store_for_samples(
            source,
            rows,
            purpose="evaluation",
            history_lookbacks=60,
            history_end_offsets=0,
            target_window_indices=target_window,
        )
        try:
            scaler = fit_feature_scalers(store, fit, roster)
            sets = {}
            for label, ix, before in (
                ("fit", fit, selection[0]),
                ("selection", selection, evaluation[0]),
                ("evaluation", evaluation, None),
            ):
                x, common, dates, names = encode_features(store, ix, scaler)
                a = np.searchsorted(anchor_indices, dates)
                if np.any(a >= len(anchor_indices)) or not np.array_equal(
                    anchor_indices[a], dates
                ):
                    raise ValueError("missing genuine OOS training anchor")
                sets[label] = {
                    "x": x,
                    "common": common,
                    "dates": dates,
                    "names": names,
                    "anchors": anchor_ranks[:, a, names],
                }
                if before is not None:
                    target, valid = rank_targets(store, ix, before)
                    d = np.searchsorted(ix, dates)
                    sets[label].update(target=target[d, names], valid=valid[d, names])
        finally:
            store.close()
        write_json_atomic(root / "residual" / f"{fold}_scalers.json", scaler.payload())
        for seed_index, seed in enumerate(ALLOWED_SEEDS):
            for learner in ("ridge", "tree"):
                for rich in (False, True):
                    cell = f"{learner}_{'rich' if rich else 'score'}"
                    output = root / "residual" / cell / f"{fold}_seed_{seed}"
                    binding = {
                        "implementation": implementation,
                        "anchor_sha256": anchor_manifest["sha256"],
                        "scaler_sha256": sha256_file(
                            root / "residual" / f"{fold}_scalers.json"
                        ),
                        "cell": cell,
                        "fold": fold,
                        "seed": seed,
                    }
                    if (output / "result.json").exists():
                        previous = read(output / "result.json")
                        if previous["binding"] != binding:
                            raise ValueError("residual resume source differs")
                        if (
                            sha256_file(output / "scores.npz")
                            != previous["score_sha256"]
                        ):
                            raise ValueError("residual score artifact changed")
                        continue
                    started = perf_counter()
                    output.mkdir(parents=True, exist_ok=True)
                    matrices, corrections = {}, {}
                    for label, data in sets.items():
                        anchor = data["anchors"][seed_index]
                        matrices[label] = (
                            np.column_stack(
                                (
                                    anchor,
                                    data["x"],
                                    (
                                        data["common"][:, :, None] * anchor[:, None, :]
                                    ).reshape(len(anchor), -1),
                                )
                            )
                            if rich
                            else anchor
                        )
                        corrections[label] = np.zeros_like(anchor)
                    train, select = sets["fit"], sets["selection"]
                    iterations = []
                    for h in range(3):
                        valid, selection_valid = (
                            train["valid"][:, h],
                            select["valid"][:, h],
                        )
                        residual = (
                            train["target"][valid, h]
                            - train["anchors"][seed_index, valid, h]
                        )
                        if learner == "ridge":
                            beta, intercept = fit_ridge(
                                matrices["fit"][valid], residual, train["dates"][valid]
                            )
                            np.savez(
                                output / f"head_{h}.npz", beta=beta, intercept=intercept
                            )
                            for label in sets:
                                corrections[label][:, h] = (
                                    matrices[label] @ beta + intercept
                                )
                        else:
                            model = fit_tree(
                                matrices["fit"][valid],
                                residual,
                                train["dates"][valid],
                                matrices["selection"][selection_valid],
                                select["target"][selection_valid, h],
                                select["anchors"][seed_index, selection_valid, h],
                                select["dates"][selection_valid],
                            )
                            model.save_model(str(output / f"head_{h}.txt"))
                            iterations.append(model.best_iteration)
                            for label in sets:
                                corrections[label][:, h] = model.predict(
                                    matrices[label], num_iteration=model.best_iteration
                                )
                    strength, selection_scores = choose_strength(
                        select["anchors"][seed_index],
                        corrections["selection"],
                        select["target"],
                        select["valid"],
                        select["dates"],
                    )
                    evaluation_set = sets["evaluation"]
                    anchor = evaluation_set["anchors"][seed_index]
                    prediction = (
                        anchor
                        if strength == 0
                        else anchor + strength * corrections["evaluation"]
                    )
                    np.savez(
                        output / "scores.npz",
                        scores=prediction,
                        dates=evaluation_set["dates"],
                        names=evaluation_set["names"],
                    )
                    write_json_atomic(
                        output / "result.json",
                        {
                            "binding": binding,
                            "strength": strength,
                            "selection_scores": selection_scores,
                            "iterations": iterations,
                            "seconds": perf_counter() - started,
                            "score_sha256": sha256_file(output / "scores.npz"),
                        },
                    )
                    print(
                        {
                            "completed_residual": [cell, fold, seed],
                            "strength": strength,
                        },
                        flush=True,
                    )


def fit_feature_scalers(store, fit_rows, cell):
    """Reuse accepted scalar semantics, including one common sample per date."""
    from types import SimpleNamespace

    from .round7_preprocessing import Round7Preprocessing

    return Round7Preprocessing.fit(
        SimpleNamespace(
            store=store, date_indices=fit_rows, enabled_sidecars=cell["families"]
        ),
        split_common=True,
        excluded_fields=cell["excluded_fields"],
    )


def age_channels(ages):
    known = ages >= 0
    value = np.log1p(np.maximum(ages, 0))
    value = value / (value + np.log1p(252.0))
    return np.where(known, value, 0).astype(np.float32), known.astype(np.float32)


def encode_features(store, rows, scalers):
    """Flatten active names only; feature missingness never changes eligibility.

    Returns rich scalar columns and common values for anchor interactions. Caller
    supplies fit-only scalers and reuses these same columns for all seed anchors.
    """
    pieces, common_pieces, date_pieces, name_pieces = [], [], [], []
    for start in range(0, len(rows), 64):
        chunk = np.asarray(rows[start : start + 64])
        active = store.read("active", chunk)
        day, name = np.nonzero(active)
        columns, common = [], []
        for family, scaler in scalers.families.items():
            prefix = "sidecar_" + family
            ix = scalers.source_columns.get(family, tuple(range(len(scaler.center))))
            values = store.read(prefix + "_values", chunk)[..., ix]
            valid = store.read(prefix + "_valid", chunk)[..., ix]
            ages = store.read(prefix + "_age_sessions", chunk)[..., ix]
            encoded = scaler.transform(values, valid)
            age, age_known = age_channels(ages[day, name])
            columns.extend(
                (
                    encoded[day, name],
                    valid[day, name].astype(np.float32),
                    age,
                    age_known,
                )
            )
            if family == "cross_market":
                common.append(encoded[day, name][:, scalers.common_columns])
        diagnostic_valid = store.read("common_state_diagnostic_valid", chunk)
        diagnostic = scalers.diagnostic.transform(
            store.read("common_state_diagnostic_values", chunk), diagnostic_valid
        )
        columns.extend((diagnostic[day], diagnostic_valid[day].astype(np.float32)))
        common.append(diagnostic[day])
        pieces.append(np.concatenate(columns, axis=1))
        common_pieces.append(np.concatenate(common, axis=1))
        date_pieces.append(chunk[day])
        name_pieces.append(name)
    return (
        np.concatenate(pieces),
        np.concatenate(common_pieces),
        np.concatenate(date_pieces),
        np.concatenate(name_pieces),
    )


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
