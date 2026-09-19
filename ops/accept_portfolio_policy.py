"""Fit-date engineering acceptance on real market paths, not an alpha screen."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from brazil_rv.execution.portfolio_policy import (
    PolicyData,
    PreferenceModel,
    account_decision,
    exact_replay,
    policy_ledger_config,
    portfolio_variance,
)
from brazil_rv.execution.portfolio_account import tensor
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_roots import resolve_external_root
from brazil_rv.v2.portfolio_inputs import Calibration, open_policy_inputs
from brazil_rv.v2.portfolio_program import PROJECT, read, source_root
from brazil_rv.v2.score import parent_prelude_indices


def run(root):
    torch.set_num_threads(1)
    design = read(source_root() / "frozen_design.json")
    design["store"] = read(PROJECT / "docs/v2_data_inputs.json")["store"]
    store = resolve_external_root(design["store"]["root"])[0]
    design["store"]["root"] = str(store)
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "frozen_design.json"
    if not frozen.exists():
        write_json_atomic(frozen, design)
    indices = parent_prelude_indices(store)[:128]
    active = np.load(store / "active.npy", mmap_mode="r", allow_pickle=False)[indices]
    # Synthetic deterministic scores exercise real paths. No model or policy
    # parameter is selected from the resulting economic outcomes.
    from brazil_rv.v2.train import rank_average_ensemble

    scores = np.sin(
        np.arange(active.shape[1])[None, :, None] * 0.31
        + np.arange(len(indices))[:, None, None] * 0.07
        + np.arange(5)[None, None, :] * 0.2
    ).astype(np.float32)
    mask = np.repeat(active[..., None], 5, -1)
    scores = rank_average_ensemble([scores], mask)
    started = time.monotonic()
    inputs, beta, diagonal, factor, support, prior_cdi, references = open_policy_inputs(
        root, indices, scores, mask
    )
    data = PolicyData(inputs, beta, diagonal, factor, prior_cdi, references)
    loaded_seconds = time.monotonic() - started
    torch.manual_seed(11)
    calibration = Calibration(
        np.zeros(3), np.ones(3), np.array([0.0003, 0.0002, 0.0001]), 0.0
    )
    model = PreferenceModel(data, calibration, np.arange(100))
    config = policy_ledger_config()
    started = time.monotonic()
    exact, targets, _ = exact_replay(data, model, 0, len(indices))
    exact_seconds = time.monotonic() - started
    account = data.initial_account(0, config)
    differences = {k: 0.0 for k in ("nav", "cash", "restricted", "claims", "shares")}
    with torch.no_grad():
        for day, target in enumerate(targets):
            data.step(account, tensor(target), day, terminal=day == len(indices) - 1)
            values = {
                "nav": abs(account.nav.item() - exact.nav[day]),
                "cash": abs(account.cash.item() - exact.free_cash[day]),
                "restricted": abs(
                    account.restricted.sum().item()
                    - exact.restricted_cash[day]
                    - exact.hedge_restricted_cash[day]
                ),
                "claims": abs(
                    account.claims.sum().item()
                    - exact.receivables[day]
                    + exact.payables[day]
                ),
                "shares": float(
                    np.max(
                        np.abs(
                            account.shares.numpy()
                            - np.r_[
                                exact.signed_shares[day], exact.hedge_signed_shares[day]
                            ]
                        )
                    )
                ),
            }
            for k, value in values.items():
                differences[k] = max(differences[k], value)
            if max(values.values()) > 1e-7:
                write_json_atomic(
                    root / "accounting_failure.json",
                    {"day": day, "differences": values},
                )
                raise ValueError(f"real-path accounting mismatch at {day}: {values}")

    def gradient(chunk):
        model.zero_grad(set_to_none=True)
        account = data.initial_account(0, config)
        started = time.monotonic()
        loss = tensor(0.0)
        gross = []
        for day in range(64):
            variance = portfolio_variance(data, day, account.market_weights)
            target = account_decision(data, model, account, day)
            row = data.step(account, target, day)
            loss = loss - (row["net_excess"] - 2.5 * variance) * 1e4 / 64
            gross.append(float(target.detach().abs().sum()))
            if (day + 1) % chunk == 0:
                loss.backward()
                account.detach()
                loss = tensor(0.0)
        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        return grad, time.monotonic() - started, float(np.mean(gross))

    a, seconds32, gross32 = gradient(32)
    b, seconds64, _ = gradient(64)
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    ratio = float(a.norm() / b.norm())
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    record = {
        "passed": finite and cosine > 0.8 and 0.5 < ratio < 2.0,
        "scope": "synthetic preferences, 128 actual prelude market sessions; no alpha inference",
        "accounting_max_absolute_errors": differences,
        "gradient": {
            "cosine_32_64": cosine,
            "norm_ratio_32_64": ratio,
            "finite": finite,
            "norm32": float(a.norm()),
            "mean_planned_gross": gross32,
        },
        "timings_seconds": {
            "data_load": loaded_seconds,
            "exact_128": exact_seconds,
            "forward_backward_64_chunk32": seconds32,
            "forward_backward_64_chunk64": seconds64,
        },
        "config": asdict(config),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "risk_support_min_active": int(support[active].min()),
        "forecast_design_sha256": sha256_file(frozen),
    }
    write_json_atomic(root / "engineering_acceptance.json", record)
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
