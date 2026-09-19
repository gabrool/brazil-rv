"""Chronological policy fits on frozen out-of-fit forecasts; CPU sparse finance."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import (
    PolicyData,
    PreferenceModel,
    account_decision,
    exact_replay,
    policy_ledger_config,
    portfolio_variance,
)
from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, HORIZONS
from .data_roots import resolve_external_root
from .portfolio_inputs import build_forecast_cache, fit_calibration, open_policy_inputs
from .portfolio_program import ARMS, PROJECT, read
from .research_rounds import _git_identity, _fold_indices


def prepare(root, arm):
    directory = root / "cache" / arm
    if not (directory / "manifest.json").exists():
        build_forecast_cache(root, arm)
    manifest = read(directory / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError("frozen forecast cache changed")
    completed = directory / "policy_data.json"
    if completed.exists():
        record = read(completed)
        if (
            record["forecast_manifest_sha256"]
            != sha256_file(directory / "manifest.json")
            or sha256_file(directory / "policy_data.pkl") != record["sha256"]
        ):
            raise ValueError("completed policy inputs differ from the frozen forecasts")
        return
    indices, scores, mask = (
        np.load(directory / f"{name}.npy", allow_pickle=False)
        for name in ("indices", "scores", "mask")
    )
    inputs, beta, diagonal, factor, support, cdi, refs = open_policy_inputs(
        root, indices, scores, mask
    )
    data = PolicyData(inputs, beta, diagonal, factor, cdi, refs)
    output = directory / "policy_data.pkl"
    # This is an internal, hash-bound local artifact produced here. Never load
    # an external/user-supplied pickle as research inputs.
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as target:
        pickle.dump(data, target, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    write_json_atomic(
        directory / "policy_data.json",
        {
            "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
            "forecast_manifest_sha256": sha256_file(directory / "manifest.json"),
            "economic_design_sha256": sha256_file(
                root / "economics/evaluation_design.json"
            ),
            "implementation": _git_identity(),
            "risk_support_min_active": int(support[inputs.active].min()),
            "risk_support_median_active": float(np.median(support[inputs.active])),
            "static_fields": [
                "rank3",
                "rank5",
                "rank10",
                "change3",
                "change5",
                "change10",
                "current_valid",
                "previous_valid",
                "log_daily_volatility",
                "beta",
                "asinh_daily_borrow",
                "prior_published_cdi",
                "prior_cdi_valid",
            ],
            "heldout_accessed": False,
        },
    )


def load_data(root, arm):
    directory = root / "cache" / arm
    record = read(directory / "policy_data.json")
    path = directory / "policy_data.pkl"
    if sha256_file(path) != record["sha256"]:
        raise ValueError("internal policy data artifact changed")
    with path.open("rb") as source:
        return pickle.load(source), sha256_file(directory / "policy_data.json")


def windows(root, data, fold):
    design = read(root / "frozen_design.json")
    store_root, _ = resolve_external_root(design["store"]["root"])
    dates = np.load(store_root / "date_index.npy", allow_pickle=False)
    fit, selection, evaluation, _, _ = _fold_indices(dates)
    start = int(data.inputs.session_indices[0])
    result = {
        key: value[fold] - start
        for key, value in (
            ("fit", fit),
            ("selection", selection),
            ("evaluation", evaluation),
        )
    }
    for key, rows in result.items():
        rows = rows[(rows >= 0) & (rows < len(data.inputs.dates))]
        if not len(rows) or not np.all(np.diff(rows) == 1):
            raise ValueError(f"{key} policy window is absent from the out-of-fit cache")
        result[key] = rows
    return result


def calibration_for(data, fit_rows):
    h5 = HORIZONS.index(5)
    return fit_calibration(
        data.ranks,
        data.valid,
        data.inputs.shareholder_simple_returns[..., h5],
        data.inputs.shareholder_target_mask[..., h5],
        data.inputs.cdi_returns,
        fit_rows,
    )


def utility_series(data, result, previous, start):
    stop = start + len(previous)
    beta = np.column_stack((data.beta[start:stop], np.ones(len(previous))))
    diagonal = np.column_stack(
        (data.diagonal[start:stop], np.full(len(previous), 1e-8))
    )
    risk = (previous**2 * diagonal).sum(1) + (previous * beta).sum(
        1
    ) ** 2 * data.factor[start:stop]
    return result.net_excess_all_cash_bps - 2.5 * risk * 1e4


def selection_value(data, model, bounds):
    # Burn in ONLY after all fitting labels have matured. The original ten
    # purge sessions supply this state, and are excluded from reported utility.
    start = int(bounds["fit"][-1] + 1)
    first, last = bounds["selection"][[0, -1]]
    result, targets, previous = exact_replay(data, model, start, int(last + 1))
    ix = slice(int(first - start), None)
    utility = utility_series(data, result, previous, start)[ix]
    if np.max(np.abs(result.reconciliation_error)) > 1e-7:
        raise ValueError("selection ledger does not self-finance")
    return {
        "utility_bps": float(utility.mean()),
        "net_excess_bps": float(result.net_excess_all_cash_bps[ix].mean()),
        "planned_gross": float(np.abs(targets[ix]).sum(1).mean()),
        "economics_unresolved": bool(result.economics_unresolved),
    }


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit_policy(data, root, arm, fold, seed, binding, *, max_epochs=30):
    torch.set_num_threads(1)
    bounds = windows(root, data, fold)
    output = root / "policies" / arm / fold / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "run_manifest.json"
    implementation = _git_identity()
    provenance = {
        "implementation": implementation,
        "policy_data_sha256": binding,
        "forecast_design_sha256": sha256_file(root / "frozen_design.json"),
        "registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_portfolio_policy.md"
        ),
        "implementation_resolutions_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_portfolio_policy_implementation.md"
        ),
        "arm": arm,
        "fold": fold,
        "seed": seed,
    }
    if completed.exists():
        saved = read(completed)
        if (
            saved["provenance"] != provenance
            or sha256_file(output / "selected.pt") != saved["selected_sha256"]
        ):
            raise ValueError("completed policy differs from requested trial")
        return saved
    torch.manual_seed(seed)
    calibration = calibration_for(data, bounds["fit"])
    model = PreferenceModel(data, calibration, bounds["fit"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    last = output / "resume.pt"
    selected = output / "selected.pt"
    started = time.monotonic()
    history, best_epoch, bad_epochs, start_epoch = [], 0, 0, 1
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        if checkpoint["provenance"] != provenance:
            raise ValueError("policy resume source changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_epoch, bad_epochs = (
            checkpoint["history"],
            checkpoint["best_epoch"],
            checkpoint["bad_epochs"],
        )
        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best"]
    else:
        value = selection_value(data, model, bounds)
        best = value["utility_bps"]
        history.append({"epoch": 0, **value})
        save_checkpoint(
            selected,
            {"provenance": provenance, "model": model.state_dict(), "epoch": 0},
        )
    for epoch in range(start_epoch, max_epochs + 1):
        if bad_epochs >= 5:
            break
        account = data.initial_account(int(bounds["fit"][0]), policy_ledger_config())
        values, grad_norms, zero_gradients = [], [], 0
        epoch_start = time.monotonic()
        fit_rows = bounds["fit"]
        for offset in range(0, len(fit_rows), 32):
            rows = fit_rows[offset : offset + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = tensor(0.0)
            for day in rows:
                day = int(day)
                risk = portfolio_variance(data, day, account.market_weights)
                target = account_decision(data, model, account, day)
                outcome = data.step(account, target, day)
                utility = 1e4 * (outcome["net_excess"] - 2.5 * risk)
                values.append(float(utility.detach()))
                loss = loss - utility / len(rows)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            grad_norms.append(float(gradient_norm))
            zero_gradients += int(gradient_norm < 1e-10)
            account.detach()
            optimizer.step()
        value = selection_value(data, model, bounds)
        improvement = value["utility_bps"] > best + 0.01
        if improvement:
            best, best_epoch, bad_epochs = value["utility_bps"], epoch, 0
            save_checkpoint(
                selected,
                {"provenance": provenance, "model": model.state_dict(), "epoch": epoch},
            )
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "fit_utility_bps": float(np.mean(values)),
            **value,
            "gradient_norm_median": float(np.median(grad_norms)),
            "gradient_norm_max": float(np.max(grad_norms)),
            "zero_gradient_chunks": zero_gradients,
            "chunks": len(grad_norms),
            "seconds": time.monotonic() - epoch_start,
            "selected": improvement,
        }
        history.append(row)
        save_checkpoint(
            last,
            {
                "provenance": provenance,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "best": best,
            },
        )
        write_json_atomic(
            output / "progress.json",
            {
                "status": "running",
                **provenance,
                "epoch": epoch,
                "selected_epoch": best_epoch,
                "last": row,
            },
        )
        print(json.dumps({"arm": arm, "fold": fold, "seed": seed, **row}), flush=True)
    manifest = {
        "status": "completed",
        "provenance": provenance,
        "history": history,
        "selected_epoch": best_epoch,
        "selected_sha256": sha256_file(selected),
        "epochs_completed": history[-1]["epoch"],
        "duration_seconds": time.monotonic() - started,
        "calibration": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in asdict(calibration).items()
        },
        "windows": {
            k: {
                "first": str(data.inputs.dates[v[0]]),
                "last": str(data.inputs.dates[v[-1]]),
                "sessions": len(v),
            }
            for k, v in bounds.items()
        },
        "accounting": "FP64 native QP adjoint and share/cash ledger; exact-ledger selection",
        "forward_capture": False,
        "heldout_accessed": False,
    }
    write_json_atomic(completed, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fold", choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.arm)
    else:
        data, binding = load_data(args.root, args.arm)
        for seed in ALLOWED_SEEDS if args.seed is None else (args.seed,):
            fit_policy(data, args.root, args.arm, args.fold, seed, binding)


if __name__ == "__main__":
    main()
