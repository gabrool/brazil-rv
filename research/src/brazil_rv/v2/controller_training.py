"""Behavioral acceptance and chronological controller comparisons."""

from __future__ import annotations

from dataclasses import asdict
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    account_decision,
    exact_replay,
    policy_ledger_config,
    portfolio_variance,
)
from brazil_rv.execution.opportunity_policy import OpportunityPolicy
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.decision_program import benchmark_excess_returns, calibrations
from brazil_rv.v2.portfolio_program import PROJECT, read
from brazil_rv.v2.portfolio_readouts import save_book, verify_book
from brazil_rv.v2.portfolio_training import (
    load_data,
    save_checkpoint,
    selection_value,
    windows,
)
from brazil_rv.v2.research_rounds import _git_identity

RECIPE = {
    "learning_rate": 0.003,
    "epochs": 30,
    "patience": 5,
    "minimum_epochs": 20,
    "chunk_sessions": 32,
    "gradient_clip": 1.0,
    "minimum_improvement_bps": 0.01,
}


def controller_epoch(
    data, model, optimizer, rows, *, chunk_sessions=32, gradient_clip=1.0
):
    account = data.initial_account(int(rows[0]), policy_ledger_config())
    values, norms = [], []
    for offset in range(0, len(rows), chunk_sessions):
        chunk = rows[offset : offset + chunk_sessions]
        optimizer.zero_grad(set_to_none=True)
        loss = tensor(0.0)
        for day in chunk:
            day = int(day)
            risk = portfolio_variance(data, day, account.weights)
            target = account_decision(data, model, account, day)
            outcome = data.step(account, target, day)
            utility = 1e4 * (outcome["net_excess"] - 2.5 * risk)
            loss = loss - utility / len(chunk)
            values.append(float(utility.detach()))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip, error_if_nonfinite=True
        )
        norms.append(float(norm))
        account.detach()
        optimizer.step()
    return {
        "fit_utility_bps": float(np.mean(values)),
        "gradient_norm_median": float(np.median(norms)),
        "gradient_norm_max": float(max(norms)),
        "zero_gradient_chunks": int((np.asarray(norms) < 1e-10).sum()),
        "chunks": len(norms),
    }


def learn(data, model, bounds, output, provenance, *, recipe=RECIPE):
    """Selection uses an independent exact ledger, including terminal costs."""
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe["learning_rate"])
    selected = output / "selected.pt"
    resume = output / "resume.pt"
    if resume.exists():
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint["provenance"] != provenance or checkpoint["recipe"] != recipe:
            raise ValueError("controller resume sources or recipe changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        best, best_epoch, bad = (
            checkpoint["best"],
            checkpoint["best_epoch"],
            checkpoint["bad"],
        )
    else:
        initial = selection_value(data, model, bounds)
        best, best_epoch, bad = initial["utility_bps"], 0, 0
        history = [{"epoch": 0, **initial}]
        save_checkpoint(
            selected,
            {"model": model.state_dict(), "epoch": 0, "provenance": provenance},
        )
    for epoch in range(history[-1]["epoch"] + 1, recipe["epochs"] + 1):
        if (
            bad >= recipe["patience"]
            and history[-1]["epoch"] >= recipe["minimum_epochs"]
        ):
            break
        epoch_start = time.monotonic()
        diagnostics = controller_epoch(
            data,
            model,
            optimizer,
            bounds["fit"],
            chunk_sessions=recipe["chunk_sessions"],
            gradient_clip=recipe["gradient_clip"],
        )
        selected_value = selection_value(data, model, bounds)
        improved = (
            selected_value["utility_bps"] > best + recipe["minimum_improvement_bps"]
        )
        if improved:
            best, best_epoch, bad = selected_value["utility_bps"], epoch, 0
            save_checkpoint(
                selected,
                {"model": model.state_dict(), "epoch": epoch, "provenance": provenance},
            )
        else:
            bad += 1
        row = {
            "epoch": epoch,
            **diagnostics,
            **selected_value,
            "selected": improved,
            "seconds": time.monotonic() - epoch_start,
        }
        history.append(row)
        save_checkpoint(
            resume,
            {
                "provenance": provenance,
                "recipe": recipe,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "bad": bad,
            },
        )
        write_json_atomic(
            output / "progress.json",
            {
                "provenance": provenance,
                "history": history,
                "selected_epoch": best_epoch,
            },
        )
        print(
            {
                **{
                    k: provenance[k]
                    for k in ("arm", "fold", "seed", "kind")
                    if k in provenance
                },
                **row,
            },
            flush=True,
        )
        if bad >= recipe["patience"] and epoch >= recipe["minimum_epochs"]:
            break
    model.load_state_dict(
        torch.load(selected, map_location="cpu", weights_only=True)["model"]
    )
    return {
        "history": history,
        "selected_epoch": best_epoch,
        "seconds": sum(row["seconds"] for row in history[1:]),
        "selected_sha256": sha256_file(selected),
        "preprocessing": model.preprocessing_payload(),
        "recipe": recipe,
    }


def run_controller(data, root, arm, fold, seed, kind, binding):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    bounds = windows(root, data, fold)
    market = benchmark_excess_returns(data)
    calibration = calibrations(data, bounds["fit"], market)["benchmark"]
    model = OpportunityPolicy(data, calibration, bounds["fit"], kind=kind)
    output = root / "phase2/policies" / arm / fold / kind / f"seed_{seed}"
    provenance = {
        "implementation": _git_identity(),
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "kind": kind,
        "policy_data_sha256": binding,
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
        "phase2_registration_sha256": sha256_file(
            PROJECT / "research/preregistrations/v2_decision_phase2.md"
        ),
        "context_sha256": sha256_file(root / "phase2/context" / f"{arm}_features.json"),
    }
    completed = output / "run_manifest.json"
    if completed.exists():
        saved = read(completed)
        if (
            saved["provenance"] != provenance
            or sha256_file(output / "selected.pt") != saved["selected_sha256"]
        ):
            raise ValueError("completed controller differs from requested experiment")
        return saved
    result = learn(data, model, bounds, output, provenance)
    control = CalibratedPolicy(calibration).eval()
    control_selection = selection_value(data, control, bounds)
    learned_selection = result["history"][result["selected_epoch"]]["utility_bps"]
    # Tie priority is cash, deterministic control, then learned; prior selection only.
    fallback = max(
        (
            (0.0, "cash"),
            (control_selection["utility_bps"], "benchmark"),
            (learned_selection, "learned"),
        ),
        key=lambda pair: pair[0],
    )[1]
    start = int(bounds["selection"][-1] + 1)
    first = int(bounds["evaluation"][0])
    stop = int(bounds["evaluation"][-1] + 1)
    books = {}
    for label, candidate in (
        ("candidate", model),
        ("fallback", {"cash": None, "benchmark": control, "learned": model}[fallback]),
    ):
        book_provenance = {
            **provenance,
            "scenario": "base",
            "policy": label,
            "fallback_selection": fallback,
        }
        if (output / label / "book.json").exists():
            books[label] = verify_book(output / label, book_provenance)["summary"]
            continue
        replay, target, previous = exact_replay(data, candidate, start, stop)
        book = save_book(
            output / label,
            data,
            replay,
            target,
            previous,
            start,
            first,
            book_provenance,
        )
        books[label] = book["summary"]
    manifest = {
        "status": "completed",
        "provenance": provenance,
        **result,
        "fallback": fallback,
        "control_selection": control_selection,
        "books": books,
        "calibration": {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in asdict(calibration).items()
        },
        "full_population_retained": True,
        "heldout_accessed": False,
    }
    write_json_atomic(completed, manifest)
    return manifest


def prepare_references(data, root, arm, binding):
    """Matched financial controls with the same refined numerical precision."""
    from brazil_rv.v2.round7 import SCREEN_FOLDS

    implementation = _git_identity()
    market = benchmark_excess_returns(data)
    for fold in SCREEN_FOLDS:
        bounds = windows(root, data, fold)
        fitted = calibrations(data, bounds["fit"], market)
        start, first, stop = (
            int(bounds["selection"][-1] + 1),
            int(bounds["evaluation"][0]),
            int(bounds["evaluation"][-1] + 1),
        )
        for label in ("benchmark", "equal_rank", "cash"):
            output = root / "phase2/references" / arm / fold / label
            provenance = {
                "implementation": implementation,
                "policy_data_sha256": binding,
                "arm": arm,
                "fold": fold,
                "policy": label,
                "scenario": "base",
                "phase": 2,
            }
            if (output / "book.json").exists():
                verify_book(output, provenance)
                continue
            model = None if label == "cash" else CalibratedPolicy(fitted[label])
            result, target, previous = exact_replay(data, model, start, stop)
            save_book(output, data, result, target, previous, start, first, provenance)


def main():
    from brazil_rv.v2.controller_context import build_context, load_context
    from brazil_rv.v2.contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("TE_all", "C6"), required=True)
    parser.add_argument("--fold", choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--kind", choices=("reliability", "stateful"))
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS)
    args = parser.parse_args()
    data, binding = load_data(args.root, args.arm)
    if args.command == "prepare":
        build_context(args.root, data, args.arm, benchmark_excess_returns(data))
        prepare_references(data, args.root, args.arm, binding)
    else:
        data.context = load_context(args.root, args.arm, binding)
        result = run_controller(
            data, args.root, args.arm, args.fold, args.seed, args.kind, binding
        )
        print(
            {
                "status": result["status"],
                "selected_epoch": result["selected_epoch"],
                "books": result["books"],
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
