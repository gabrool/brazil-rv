"""Matched cash/legacy/optimizer/learned books, continuous rolls and uncertainty."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.portfolio_policy import (
    PreferenceModel,
    exact_replay,
    ledger_arguments,
    policy_ledger_config,
)
from brazil_rv.execution.stateful_ledger import simulate_stateful_ledger
from .artifacts import sha256_file, write_json_atomic
from .contract import (
    ALLOWED_SEEDS,
    DEVELOPMENT_FOLDS,
    HORIZONS,
    TRADED_PRIMARY_HORIZONS,
)
from .evaluate import (
    _serialise_records,
    _primary_population_components,
    _primary_daily_metrics,
)
from .execution_policy import traded_signal
from .portfolio_program import ARMS, SCREEN_FOLDS, read
from .performance import performance
from .portfolio_training import calibration_for, load_data, utility_series, windows
from .research_rounds import _git_identity

SCENARIOS = {
    "base": {},
    "cost2": {"cost_bps_per_side": 2.0, "hedge_cost_bps_per_side": 2.0},
    "cost8": {"cost_bps_per_side": 8.0, "hedge_cost_bps_per_side": 8.0},
    "proceeds0": {"short_proceeds_remuneration": 0.0},
}
POLICIES = (
    "cash",
    "legacy",
    "optimizer",
    *(f"learned_{seed}" for seed in ALLOWED_SEEDS),
)


def models_for_fold(root, data, arm, fold, binding):
    bounds = windows(root, data, fold)
    calibration = calibration_for(data, bounds["fit"])
    optimizer = PreferenceModel(data, calibration, bounds["fit"]).eval()
    models = {"cash": None, "legacy": None, "optimizer": optimizer}
    sources = {
        "optimizer": {
            "calibration": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in asdict(calibration).items()
            }
        }
    }
    for seed in ALLOWED_SEEDS:
        directory = root / "policies" / arm / fold / f"seed_{seed}"
        record = read(directory / "run_manifest.json")
        path = directory / "selected.pt"
        if (
            record["status"] != "completed"
            or record["provenance"]["policy_data_sha256"] != binding
            or sha256_file(path) != record["selected_sha256"]
        ):
            raise ValueError("readout policy is incomplete or has changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = PreferenceModel(data, calibration, bounds["fit"]).eval()
        model.load_state_dict(checkpoint["model"])
        key = f"learned_{seed}"
        models[key] = model
        sources[key] = {
            "manifest_sha256": sha256_file(directory / "run_manifest.json"),
            "checkpoint_sha256": record["selected_sha256"],
            "selected_epoch": record["selected_epoch"],
        }
    return models, sources, bounds


def legacy_replay(data, start, stop, config):
    arguments = ledger_arguments(data, start, stop)
    score, mask = traded_signal(data.inputs, data.inputs.execution_policy)
    arguments["scores"], arguments["score_mask"] = score[start:stop], mask[start:stop]
    arguments["capacity_buffer_per_side"] = 30
    if data.inputs.execution_policy.inverse_volatility:
        arguments["entry_sizing_volatility"] = data.inputs.target_scale_sigma[
            start:stop
        ]
    result = simulate_stateful_ledger(
        **arguments, config=config, shortable=data.shortable[start:stop]
    )
    end = np.column_stack(
        (
            result.equity_market_weights,
            result.hedge_signed_notional / result.nav,
        )
    )
    previous = np.concatenate((np.zeros_like(end[:1]), end[:-1]))
    return result, None, previous


def book_summary(data, result, previous, start, first):
    trim = first - start
    selection = slice(trim, None)
    utility = utility_series(data, result, previous, start)[selection]
    net = result.net_excess_all_cash_bps[selection]
    absolute = result.daily_net_return[selection]
    nav = result.nav[selection]
    base = result.start_nav[trim]
    wealth = np.r_[base, nav]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    positions = (
        result.signed_shares * np.nan_to_num(result.mark_price) / result.nav[:, None]
    )
    hedge = result.hedge_signed_notional / result.nav
    beta = (
        result.equity_market_weights * data.beta[start : start + len(positions)]
    ).sum(1) + hedge
    gross = np.abs(positions).sum(1) + np.abs(hedge)
    signed_net = positions.sum(1) + hedge
    daily = {
        "net_excess_bps": net,
        "utility_bps": utility,
        "absolute_bps": absolute * 1e4,
        "equity_gross_bps": result.equity_gross_pnl_bps[selection],
        "hedge_gross_bps": result.hedge_gross_pnl_bps[selection],
        "trading_cost_bps": result.cost_bps[selection],
        "borrow_bps": result.borrow_bps[selection],
        "loan_rent_bps": (result.equity_borrow_raw_bps + result.hedge_borrow_raw_bps)[
            selection
        ],
        "b3_loan_fee_bps": (result.equity_borrow_fee_bps + result.hedge_borrow_fee_bps)[
            selection
        ],
        "interest_bps": result.interest_bps[selection],
        "cdi_bps": result.cdi_benchmark_bps[selection],
        "free_cash_fraction": (result.free_cash / result.nav)[selection],
        "restricted_cash_fraction": (
            (result.restricted_cash + result.hedge_restricted_cash) / result.nav
        )[selection],
        "gross": gross[selection],
        "signed_net": signed_net[selection],
        "beta": beta[selection],
        "turnover": result.turnover_fraction_nav[selection],
        "stale_fraction": result.stale_mark_inventory_fraction_nav[selection],
        "unresolved_claim_fraction": result.unresolved_claim_inventory_fraction_nav[
            selection
        ],
        "unpriced_inventory_fraction": result.unpriced_inventory_fraction_nav[
            selection
        ],
        "undelivered_share_fraction": (result.undelivered_share_notional / result.nav)[
            selection
        ],
    }
    if not np.isfinite(net).all() or np.max(np.abs(result.reconciliation_error)) > 1e-7:
        raise ValueError("readout ledger has invalid or unreconciled economic results")
    summary = {
        "performance": performance(absolute, result.cdi_benchmark_bps[selection] / 1e4),
        "mean": {k: float(v.mean()) for k, v in daily.items()},
        "sessions": len(net),
        "absolute_compounded_return": float(np.prod(1 + absolute) - 1),
        "relative_to_cdi_compounded_return": float(
            np.prod(1 + absolute)
            / np.prod(1 + data.inputs.cdi_returns[first : start + len(result.dates)])
            - 1
        ),
        "maximum_drawdown": float(drawdown.min()),
        "excess_sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(252))
        if net.std() > 1e-10
        else 0.0,
        "all_cash_sessions": int((gross[selection] < 1e-6).sum()),
        "max_absolute_reconciliation_error": float(
            np.max(np.abs(result.reconciliation_error))
        ),
        "economics_unresolved": bool(result.economics_unresolved),
        "max_joint_gross": float(gross[selection].max()),
        "max_absolute_net": float(np.abs(signed_net[selection]).max()),
        "max_absolute_beta": float(np.abs(beta[selection]).max()),
        "realized_drift_days_above_new_planned_limits": int(
            (
                (gross[selection] > 2.25 + 2e-6)
                | (np.abs(signed_net[selection]) > 0.05 + 2e-6)
                | (np.abs(beta[selection]) > 0.05 + 2e-6)
            ).sum()
        ),
        "terminal_haircut_scenario_relative_difference": float(
            (result.unpriced_haircut_scenario_nav[-1] - result.nav[-1]) / base
        ),
    }
    return summary, {k: v.tolist() for k, v in daily.items()}


def save_book(output, data, result, targets, previous, start, first, provenance):
    output.mkdir(parents=True, exist_ok=False)
    summary, daily = book_summary(data, result, previous, start, first)
    fields = (
        "start_nav",
        "nav",
        "free_cash",
        "restricted_cash",
        "hedge_restricted_cash",
        "receivables",
        "payables",
        "signed_shares",
        "mark_price",
        "hedge_signed_shares",
        "hedge_mark_price",
        "reconciliation_error",
        "pending_exit_count",
        "pending_entry_count",
        "unpriced_inventory_notional",
        "undelivered_share_notional",
        "unpriced_haircut_scenario_nav",
    )
    state = {k: getattr(result, k) for k in fields}
    state["prior_weights"] = previous
    if targets is not None:
        state["targets"] = targets
    np.savez_compressed(output / "account.npz", **state)
    if result.share_claim_positions:
        write_json_atomic(
            output / "share_claim_positions.json",
            [asdict(claim) for claim in result.share_claim_positions],
        )
    # Detailed primary audit; stresses keep exact positions/targets and daily
    # accounting, and are reproducible from the same frozen market inputs.
    if provenance["scenario"] == "base":
        for label, records in (
            ("fills", result.fills),
            ("orders", result.intended_orders),
        ):
            if records:
                pl.DataFrame(
                    _serialise_records(records), infer_schema_length=None
                ).write_parquet(output / f"{label}.parquet", compression="zstd")
    record = {
        "status": "completed",
        "provenance": provenance,
        "summary": summary,
        "daily": daily,
        "dates": [d.isoformat() for d in result.dates[first - start :]],
        "burn_in_sessions": first - start,
        "state_dates": [d.isoformat() for d in result.dates],
        "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()},
        "legacy_slot_diagnostics_applicable": provenance["policy"] == "legacy",
    }
    write_json_atomic(output / "book.json", record)
    return record


def verify_book(path, provenance):
    record = read(path / "book.json")
    if record["provenance"] != provenance:
        raise ValueError("completed book has another source or decision contract")
    if "equivalent_book" in record:
        equivalent = record["equivalent_book"]
        if sha256_file(Path(equivalent["path"])) != equivalent["sha256"]:
            raise ValueError("equivalent optimizer book changed")
    for name, digest in record["files"].items():
        if sha256_file(path / name) != digest:
            raise ValueError("completed portfolio accounting artifact changed")
    return record


@lru_cache(maxsize=32)
def block_draws(lengths, block_length):
    generator = np.random.default_rng(20260914)
    return tuple(
        generator.integers(
            0, length, size=(10_000, (length + block_length - 1) // block_length)
        )
        for length in lengths
    )


def interval(arrays, *, block_length=40):
    """Paired circular blocks give boundary observations equal expected weight."""
    arrays = tuple(np.asarray(a, dtype=float) for a in arrays)
    lengths = tuple(len(a) for a in arrays)
    if min(lengths) < block_length or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("economic intervals need finite daily values and full blocks")
    draws = np.zeros(10_000)
    for a, starts in zip(arrays, block_draws(lengths, block_length)):
        prefix = np.r_[0.0, np.cumsum(np.r_[a, a[:block_length]])]
        sizes = np.full(starts.shape[1], block_length)
        sizes[-1] = len(a) - block_length * (len(sizes) - 1)
        draws += (prefix[starts + sizes] - prefix[starts]).sum(1)
    draws /= sum(lengths)
    return {
        "estimate": float(np.concatenate(arrays).mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "replications": 10_000,
        "block_length_sessions": block_length,
        "circular": True,
        "resample_mean": float(draws.mean()),
        "fold_boundary_preserved": True,
        "observations": sum(lengths),
    }


def evaluate_books(root, arm, *, fold=None, continuous=False, loaded=None):
    torch.set_num_threads(1)
    data, binding = load_data(root, arm) if loaded is None else loaded
    implementation = _git_identity()
    if continuous:
        all_models, all_sources, all_bounds = {}, {}, {}
        for f in DEVELOPMENT_FOLDS:
            all_models[f], all_sources[f], all_bounds[f] = models_for_fold(
                root, data, arm, f, binding
            )
        first = int(all_bounds["F1"]["evaluation"][0])
        start = int(all_bounds["F1"]["selection"][-1] + 1)
        stop = int(all_bounds["F14"]["evaluation"][-1] + 1)
        models, sources = {}, {}
        for policy in POLICIES:
            models[policy] = {}
            for f in DEVELOPMENT_FOLDS:
                rows = all_bounds[f]["evaluation"]
                if f == "F1":
                    rows = np.arange(start, rows[-1] + 1)
                models[policy].update({int(day): all_models[f][policy] for day in rows})
            sources[policy] = {f: all_sources[f].get(policy) for f in DEVELOPMENT_FOLDS}
        label = "continuous"
    else:
        models, sources, bounds = models_for_fold(root, data, arm, fold, binding)
        first = int(bounds["evaluation"][0])
        start, stop = (
            int(bounds["selection"][-1] + 1),
            int(bounds["evaluation"][-1] + 1),
        )
        label = fold
    for scenario, changes in SCENARIOS.items():
        for policy in POLICIES:
            if policy == "cash" and scenario != "base":
                continue  # Same CDI-only path; neither trading costs nor short proceeds apply.
            output = root / "books" / arm / label / scenario / policy
            source = sources.get(policy)
            config = (
                replace(
                    data.inputs.execution_policy.ledger_config(),
                    **changes,
                )
                if policy == "legacy"
                else policy_ledger_config(**changes)
            )
            provenance = {
                "implementation": implementation,
                "policy_data_sha256": binding,
                "forecast_design_sha256": sha256_file(root / "frozen_design.json"),
                "controller": source,
                "policy": policy,
                "scenario": scenario,
                "arm": arm,
                "window": label,
                "planned_transaction_cost_bps": 4.0,
                "realized_changes": changes,
                "ledger_config": asdict(config),
            }
            if (output / "book.json").exists():
                verify_book(output, provenance)
                continue
            # A selected zero-residual epoch-zero controller is exactly the
            # deterministic optimizer, irrespective of random hidden weights.
            equivalent = (
                (
                    all(v["selected_epoch"] == 0 for v in source.values())
                    if continuous
                    else source["selected_epoch"] == 0
                )
                if policy.startswith("learned_")
                else False
            )
            if equivalent:
                other = output.parent / "optimizer" / "book.json"
                record = read(other)
                output.mkdir(parents=True, exist_ok=False)
                write_json_atomic(
                    output / "book.json",
                    {
                        **record,
                        "provenance": provenance,
                        "files": {},
                        "equivalent_book": {
                            "path": str(other),
                            "sha256": sha256_file(other),
                            "reason": "selected epoch-zero residual is identically zero",
                        },
                    },
                )
                continue
            if policy == "legacy":
                result, targets, previous = legacy_replay(data, start, stop, config)
            else:
                cash_targets = (
                    np.zeros((stop - start, len(data.inputs.security_ids) + 1))
                    if policy == "cash"
                    else None
                )
                result, targets, previous = exact_replay(
                    data,
                    models[policy],
                    start,
                    stop,
                    config=config,
                    targets=cash_targets,
                )
            save_book(output, data, result, targets, previous, start, first, provenance)
    if not continuous:
        bridge(root, data, arm, fold, bounds, binding, implementation)


def bridge(root, data, arm, fold, bounds, binding, implementation):
    """Original empty-start old-policy economics and fold-contained 3-head IC."""
    output = root / "bridge" / arm / fold
    first, last = (int(x) for x in bounds["evaluation"][[0, -1]])
    provenance = {
        "implementation": implementation,
        "policy_data_sha256": binding,
        "policy": "legacy",
        "scenario": "base",
        "arm": arm,
        "window": fold,
        "purpose": "empty-start historical execution bridge",
    }
    if (output / "book.json").exists():
        verify_book(output, provenance)
    else:
        config = data.inputs.execution_policy.ledger_config()
        result, targets, previous = legacy_replay(data, first, last + 1, config)
        save_book(output, data, result, targets, previous, first, first, provenance)
    if (output / "forecast.json").exists():
        return
    inputs = data.inputs
    mask = inputs.neutral_target_mask.copy()
    for h, horizon in enumerate(HORIZONS):
        mask[max(first, last + 1 - horizon) : last + 1, :, h] = False
    view = replace(inputs, neutral_target_mask=mask)
    s, y, outcomes, scored = _primary_population_components(
        view, TRADED_PRIMARY_HORIZONS
    )
    _, daily_ic, _ = _primary_daily_metrics(
        s, y, outcomes, scored, inputs.dates, TRADED_PRIMARY_HORIZONS
    )
    ic = daily_ic[first : last + 1]
    write_json_atomic(
        output / "forecast.json",
        {
            "dates": [d.isoformat() for d in inputs.dates[first : last + 1]],
            "neutral_ic": [float(v) if np.isfinite(v) else None for v in ic],
            "mean_neutral_ic": float(np.nanmean(ic)),
            "horizons": list(TRADED_PRIMARY_HORIZONS),
            "target_endpoints_inside_fold": True,
        },
    )


def daily_series(books, policy, metric):
    if policy == "learned_seed_mean":
        return np.mean(
            [books[f"learned_{seed}"]["daily"][metric] for seed in ALLOWED_SEEDS], 0
        )
    return np.asarray(books[policy]["daily"][metric])


def summarize(root):
    groups = {
        "original_screen": SCREEN_FOLDS,
        "remaining_ten": [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS],
        "all_fourteen": DEVELOPMENT_FOLDS,
    }
    result = {
        "groups": {},
        "continuous": {},
        "advancement_checks": {},
        "status": "completed",
        "heldout_accessed": False,
        "forward_capture": False,
        "bootstrap": "paired circular 40-session blocks within folds; nominal development inference",
        "source_books": {},
    }
    base_books = {}
    for arm in ARMS:
        books = {
            f: {
                s: {
                    p: read(root / "books" / arm / f / s / p / "book.json")
                    for p in POLICIES
                    if p != "cash" or s == "base"
                }
                for s in SCENARIOS
            }
            for f in DEVELOPMENT_FOLDS
        }
        result["groups"][arm] = {}
        base_books[arm] = {f: books[f]["base"] for f in DEVELOPMENT_FOLDS}
        result["source_books"][arm] = {
            f"{f}/{s}/{p}": sha256_file(root / "books" / arm / f / s / p / "book.json")
            for f in DEVELOPMENT_FOLDS
            for s in SCENARIOS
            for p in books[f][s]
        }

        def values(f, s, p, metric):
            return daily_series(books[f][s], p, metric)

        for group, folds in groups.items():
            scenarios = {}
            for scenario in SCENARIOS:
                policies = {}
                for policy in (*POLICIES, "learned_seed_mean"):
                    if policy == "cash" and scenario != "base":
                        continue
                    metrics = {}
                    for metric in ("net_excess_bps", "utility_bps"):
                        arrays = [values(f, scenario, policy, metric) for f in folds]
                        paired = [
                            a - values(f, scenario, "legacy", metric)
                            for a, f in zip(arrays, folds)
                        ]
                        metrics[metric] = {
                            "absolute": interval(arrays),
                            "minus_same_forecast_legacy": interval(paired),
                        }
                        if policy.startswith("learned_"):
                            metrics[metric]["minus_same_forecast_optimizer"] = interval(
                                [
                                    a - values(f, scenario, "optimizer", metric)
                                    for a, f in zip(arrays, folds)
                                ]
                            )
                    policies[policy] = metrics
                scenarios[scenario] = policies
            result["groups"][arm][group] = scenarios
        checks = {}
        for policy in ("optimizer", "learned_seed_mean"):
            delta = {
                f: {
                    m: float(
                        (
                            values(f, "base", policy, m)
                            - values(f, "base", "legacy", m)
                        ).mean()
                    )
                    for m in ("net_excess_bps", "utility_bps")
                }
                for f in DEVELOPMENT_FOLDS
            }
            checks[policy] = {
                "paired_fold_means": delta,
                "original_screen_positive_utility_folds": sum(
                    delta[f]["utility_bps"] > 0 for f in SCREEN_FOLDS
                ),
                "minimum_required_positive_screen_folds": 3,
                "economic_resolution_requires_review": any(
                    books[f]["base"][p]["summary"]["economics_unresolved"]
                    for f in DEVELOPMENT_FOLDS
                    for p in POLICIES
                ),
            }
        checks["learned_seed_paired_means"] = {
            str(seed): {
                m: float(
                    np.concatenate(
                        [
                            values(f, "base", f"learned_{seed}", m)
                            - values(f, "base", "legacy", m)
                            for f in DEVELOPMENT_FOLDS
                        ]
                    ).mean()
                )
                for m in ("net_excess_bps", "utility_bps")
            }
            for seed in ALLOWED_SEEDS
        }
        result["advancement_checks"][arm] = checks
        result["continuous"][arm] = {
            s: {
                p: read(root / "books" / arm / "continuous" / s / p / "book.json")[
                    "summary"
                ]
                for p in POLICIES
                if p != "cash" or s == "base"
            }
            for s in SCENARIOS
        }
    result["cross_forecast_base_scenario"] = {}
    for candidate, reference in (("TE_all", "S0"), ("C6", "S0"), ("TE_all", "C6")):
        for f in DEVELOPMENT_FOLDS:
            if (
                base_books[candidate][f]["legacy"]["dates"]
                != base_books[reference][f]["legacy"]["dates"]
            ):
                raise ValueError(
                    "cross-forecast comparison does not pair identical dates"
                )
        result["cross_forecast_base_scenario"][f"{candidate}_minus_{reference}"] = {
            group: {
                policy: {
                    metric: interval(
                        [
                            daily_series(base_books[candidate][f], policy, metric)
                            - daily_series(base_books[reference][f], policy, metric)
                            for f in folds
                        ]
                    )
                    for metric in ("net_excess_bps", "utility_bps")
                }
                for policy in ("legacy", "optimizer", "learned_seed_mean")
            }
            for group, folds in groups.items()
        }
    write_json_atomic(root / "portfolio_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fold", "continuous", "summarize"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--fold", choices=DEVELOPMENT_FOLDS)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.root)
    else:
        evaluate_books(
            args.root, args.arm, fold=args.fold, continuous=args.command == "continuous"
        )


if __name__ == "__main__":
    main()
