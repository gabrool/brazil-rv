"""Registered economic calibration experiments on sealed causal forecasts."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import multiprocessing
import os
from pathlib import Path
import tarfile
import time

import numpy as np
import torch

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_FOLDS, HORIZONS
from .portfolio_inputs import fit_calibration
from .portfolio_program import ARMS, PROJECT, read
from .portfolio_readouts import book_summary, interval, save_book, verify_book
from .portfolio_training import load_data, windows
from .research_rounds import _git_identity

CELLS = (
    "raw",
    "no_intercept",
    "benchmark",
    "uncertainty",
    "equal_rank",
    "stock_only",
    "cash",
)


def freeze(root):
    pointer = read(PROJECT / "docs/v2_data_inputs.json")
    recovery = read(PROJECT / "docs/v2_portfolio_policy_recovery.json")
    inventory_path = Path(recovery["inventory"])
    if sha256_file(inventory_path) != recovery["inventory_sha256"]:
        raise ValueError("sealed forecast inventory changed")
    inventory = {r["path"]: r for r in read(inventory_path)}
    if (
        sha256_file(Path(pointer["store"]["root"]) / "manifest.json")
        != pointer["store"]["manifest_sha256"]
    ):
        raise ValueError("accepted store changed")
    destination = root / "frozen_design.json"
    if destination.exists():
        raise FileExistsError("decision program is already frozen")
    bindings = {}
    for arm in ARMS:
        for name in ("policy_data.pkl", "policy_data.json", "manifest.json"):
            relative = f"cache/{arm}/{name}"
            digest = sha256_file(root / relative)
            if digest != inventory[relative]["sha256"]:
                raise ValueError(f"copied forecast cache differs: {relative}")
            bindings[relative] = digest
    # Preserve complete daily controls for numerical reproduction, not only means.
    archive = next(
        Path(r["local_archive"])
        for r in recovery["archives"]
        if r["local_archive"].endswith("evidence.tar.gz")
    )
    controls = {}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            parts = member.name.removeprefix("./").split("/")
            if not (
                len(parts) == 6
                and parts[0] == "books"
                and parts[3] == "base"
                and parts[4] in ("optimizer", "cash")
                and parts[5] == "book.json"
            ):
                continue
            payload = source.extractfile(member).read()
            relative = "/".join(parts)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != inventory[relative]["sha256"]:
                raise ValueError("sealed control archive member differs")
            target = root / "source_books" / parts[1] / parts[2] / f"{parts[4]}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            controls[str(target.relative_to(root))] = digest
    registration = PROJECT / "research/preregistrations/v2_decision_research.md"
    design = {
        "implementation": _git_identity(),
        "registration_sha256": sha256_file(registration),
        "store": pointer["store"],
        "source_recovery_sha256": sha256_file(
            PROJECT / "docs/v2_portfolio_policy_recovery.json"
        ),
        "source_run": recovery["source"],
        "source_inventory_sha256": recovery["inventory_sha256"],
        "cache_files": bindings,
        "source_books": controls,
        "phase1_cells": CELLS,
        "forward_capture": False,
        "heldout_accessed": False,
    }
    write_json_atomic(destination, design)
    write_json_atomic(
        PROJECT / "docs/v2_decision_run.json",
        {
            "root": str(root),
            "frozen_design_sha256": sha256_file(destination),
            "source_cache_files_verified": len(bindings),
            "sealed_control_books": len(controls),
            "status": "frozen",
        },
    )
    return {
        "root": str(root),
        "verified_caches": len(bindings),
        "controls": len(controls),
    }


def benchmark_excess_returns(data, horizon=5):
    """Label-only BOVA excess; realized endpoints never enter policy features."""
    close = data.inputs.bova11_close
    result = np.full(len(close), np.nan)
    for day in range(len(close) - horizon):
        if (
            np.isfinite(close[day])
            and close[day] > 0
            and np.isfinite(close[day + horizon])
        ):
            cash = np.prod(1 + data.inputs.cdi_returns[day + 1 : day + horizon + 1]) - 1
            result[day] = close[day + horizon] / close[day] - 1 - cash
    return result


def calibrations(data, fit_rows, market):
    head = HORIZONS.index(5)
    args = (
        data.ranks,
        data.valid,
        data.inputs.shareholder_simple_returns[..., head],
        data.inputs.shareholder_target_mask[..., head],
        data.inputs.cdi_returns,
        fit_rows,
    )
    raw = fit_calibration(*args)
    benchmark = fit_calibration(*args, benchmark_excess5=market, beta=data.beta)
    equal = fit_calibration(
        *args, benchmark_excess5=market, beta=data.beta, equal_rank=True
    )
    return {
        "raw": raw,
        "no_intercept": replace(raw, intercept=0.0),
        "benchmark": benchmark,
        "uncertainty": benchmark,
        "equal_rank": equal,
        "stock_only": benchmark,
    }


def calibration_payload(calibration):
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in asdict(calibration).items()
    }


def calibration_readout(data, calibration, rows, market):
    """Same five-session benchmark-residual payoff for every predicted mean."""
    rows = rows[rows + 5 <= rows[-1]]
    head = HORIZONS.index(5)
    mask = data.valid[rows] & data.inputs.shareholder_target_mask[rows, :, head]
    mask &= np.isfinite(market[rows, None])
    cash = np.array(
        [np.prod(1 + data.inputs.cdi_returns[t + 1 : t + 6]) - 1 for t in rows]
    )
    outcome = (
        data.inputs.shareholder_simple_returns[rows, :, head]
        - cash[:, None]
        - data.beta[rows] * market[rows, None]
    ) / 5
    predicted = calibration.predict(data.ranks[rows])
    error = np.where(mask, outcome - predicted, 0)
    counts = mask.sum(1)
    observed = counts > 0
    weights = mask / np.maximum(counts[:, None], 1)
    weights /= max(int(observed.sum()), 1)
    pmean = float((weights * predicted).sum())
    omean = float((weights * np.where(mask, outcome, 0)).sum())
    variance = float((weights * (predicted - pmean) ** 2).sum())
    covariance = float(
        (weights * (predicted - pmean) * np.where(mask, outcome - omean, 0)).sum()
    )
    tails = {}
    for label, tail in (
        ("upper", data.ranks[rows].mean(-1) > 0.6),
        ("lower", data.ranks[rows].mean(-1) < -0.6),
    ):
        use = mask & tail
        n = use.sum(1)
        means = np.where(use, outcome, 0).sum(1) / np.maximum(n, 1)
        tails[label] = float(means[n > 0].mean() * 1e4) if np.any(n) else None
    return {
        "label": "daily_units_of_five_session_benchmark_residual",
        "dates": int(observed.sum()),
        "observations": int(mask.sum()),
        "predicted_mean_bps": pmean * 1e4,
        "realized_mean_bps": omean * 1e4,
        "rmse_bps": float(np.sqrt((weights * error**2).sum()) * 1e4),
        "calibration_slope": covariance / variance if variance > 1e-20 else None,
        "fixed_rank_tail_outcomes_bps": tails,
    }


def run_phase1_arm(root, arm, only_fold=None):
    torch.set_num_threads(1)
    implementation = _git_identity()
    data, binding = load_data(root, arm)
    design_hash = sha256_file(root / "frozen_design.json")
    market = benchmark_excess_returns(data)
    bounds = {f: windows(root, data, f) for f in DEVELOPMENT_FOLDS}
    calibrated = {f: calibrations(data, b["fit"], market) for f, b in bounds.items()}
    models = {
        f: {
            cell: CalibratedPolicy(
                c, uncertainty_scale=float(cell == "uncertainty")
            ).eval()
            for cell, c in values.items()
        }
        for f, values in calibrated.items()
    }
    cal_path = root / "phase1" / arm / "calibrations.json"
    cal_payload = {
        f: {cell: calibration_payload(c) for cell, c in values.items()}
        for f, values in calibrated.items()
    }
    if cal_path.exists() and read(cal_path) != cal_payload:
        raise ValueError("phase1 calibration changed on resume")
    write_json_atomic(cal_path, cal_payload)
    labels = (only_fold,) if only_fold else (*DEVELOPMENT_FOLDS, "continuous")
    records = []
    for label in labels:
        if label == "continuous":
            first = int(bounds["F1"]["evaluation"][0])
            start = int(bounds["F1"]["selection"][-1] + 1)
            stop = int(bounds["F14"]["evaluation"][-1] + 1)
        else:
            first = int(bounds[label]["evaluation"][0])
            start = int(bounds[label]["selection"][-1] + 1)
            stop = int(bounds[label]["evaluation"][-1] + 1)
        for cell in CELLS:
            output = root / "phase1/books" / arm / label / "base" / cell
            allocation = (
                replace(AllocationConfig(), hedge_cap=0.0)
                if cell == "stock_only"
                else AllocationConfig()
            )
            provenance = {
                "scenario": "base",
                "policy": cell,
                "arm": arm,
                "window": label,
                "phase": 1,
                "implementation": implementation,
                "policy_data_sha256": binding,
                "frozen_design_sha256": design_hash,
                "calibration_sha256": sha256_file(cal_path),
                "allocation": asdict(allocation),
            }
            if (output / "book.json").exists():
                record = verify_book(output, provenance)
                read(output / "diagnostics.json")
            else:
                started = time.monotonic()
                if cell == "cash":
                    model = None
                elif label != "continuous":
                    model = models[label][cell]
                else:
                    model = {}
                    for fold, b in bounds.items():
                        rows = b["evaluation"]
                        if fold == "F1":
                            rows = np.arange(start, rows[-1] + 1)
                        model.update({int(day): models[fold][cell] for day in rows})
                result, target, previous = exact_replay(
                    data,
                    model,
                    start,
                    stop,
                    allocation=allocation,
                    targets=np.zeros((stop - start, len(data.inputs.security_ids) + 1))
                    if cell == "cash"
                    else None,
                )
                _, daily = book_summary(data, result, previous, start, first)
                extra = {"seconds": time.monotonic() - started}
                if cell != "cash":
                    extra["mean_planned_gross"] = float(
                        np.abs(target[first - start :]).sum(1).mean()
                    )
                    extra["planned_gross_cap_fraction"] = float(
                        (np.abs(target[first - start :]).sum(1) > 2.2499).mean()
                    )
                    extra["planned_short_hedge_cap_fraction"] = float(
                        (target[first - start :, -1] < -0.5999).mean()
                    )
                    if label != "continuous":
                        extra["calibration_readout"] = calibration_readout(
                            data,
                            calibrated[label][cell],
                            bounds[label]["evaluation"],
                            market,
                        )
                if cell in ("raw", "cash"):
                    source = (
                        root
                        / "source_books"
                        / arm
                        / label
                        / ("optimizer.json" if cell == "raw" else "cash.json")
                    )
                    original = read(source)
                    errors = {
                        key: float(
                            np.max(np.abs(np.asarray(values) - original["daily"][key]))
                        )
                        for key, values in daily.items()
                    }
                    extra["original_daily_max_errors"] = errors
                    # Native CPU solvers can differ by a few millionths of a
                    # basis point across hosts. Compare in the field's units.
                    tolerances = {
                        key: 0.01 if key.endswith("_bps") else 1e-4 for key in errors
                    }
                    extra["original_daily_tolerances"] = tolerances
                    mean_errors = {
                        key: float(
                            abs(np.mean(values) - np.mean(original["daily"][key]))
                        )
                        for key, values in daily.items()
                        if key.endswith("_bps")
                    }
                    extra["original_mean_bps_errors"] = mean_errors
                    if (
                        any(errors[key] > tolerances[key] for key in errors)
                        or max(mean_errors.values()) > 1e-4
                    ):
                        raise ValueError(
                            f"original policy reproduction differs: {arm}/{label}/{cell}: {errors}"
                        )
                record = save_book(
                    output, data, result, target, previous, start, first, provenance
                )
                write_json_atomic(output / "diagnostics.json", extra)
                print(
                    {
                        "arm": arm,
                        "fold": label,
                        "cell": cell,
                        "seconds": extra["seconds"],
                        "net_bps": record["summary"]["mean"]["net_excess_bps"],
                    },
                    flush=True,
                )
            records.append({"fold": label, "cell": cell, "summary": record["summary"]})
            write_json_atomic(
                root / "phase1" / arm / "progress.json",
                {"status": "running", "completed": records, "only_fold": only_fold},
            )
    write_json_atomic(
        root
        / "phase1"
        / arm
        / ("screen_probe.json" if only_fold else "run_manifest.json"),
        {
            "status": "completed",
            "records": records,
            "phase": 1,
            "only_fold": only_fold,
            "frozen_design_sha256": design_hash,
            "policy_data_sha256": binding,
        },
    )
    return {"arm": arm, "books": len(records)}


def summarize(root):
    result = {"phase": 1, "continuous": {}, "folds": {}, "heldout_accessed": False}
    for arm in ARMS:
        result["continuous"][arm], result["folds"][arm] = {}, {}
        for label in (*DEVELOPMENT_FOLDS, "continuous"):
            directory = root / "phase1/books" / arm / label / "base"
            reference = read(directory / "benchmark/book.json")
            rows = {}
            for cell in CELLS:
                book = read(directory / cell / "book.json")
                row = {
                    "summary": book["summary"],
                    "diagnostics": read(directory / cell / "diagnostics.json"),
                }
                if label == "continuous":
                    row["versus_benchmark"] = {
                        metric: {
                            str(block): interval(
                                [
                                    np.asarray(book["daily"][metric])
                                    - reference["daily"][metric]
                                ],
                                block_length=block,
                            )
                            for block in (20, 40, 60)
                        }
                        for metric in ("net_excess_bps", "utility_bps")
                    }
                rows[cell] = row
            if label == "continuous":
                result["continuous"][arm] = rows
            else:
                result["folds"][arm][label] = rows
    write_json_atomic(root / "phase1_summary.json", result)
    write_json_atomic(PROJECT / "docs/v2_decision_phase1_results.json", result)
    return {"status": "completed", "source": str(root / "phase1_summary.json")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "phase1", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", choices=(*DEVELOPMENT_FOLDS, "continuous"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        print(freeze(args.root))
    elif args.command == "summarize":
        print(summarize(args.root))
    elif args.arm:
        print(run_phase1_arm(args.root, args.arm, args.fold))
    else:
        os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            jobs = [
                pool.submit(run_phase1_arm, args.root, arm, args.fold) for arm in ARMS
            ]
            for job in as_completed(jobs):
                print(job.result(), flush=True)


if __name__ == "__main__":
    main()
