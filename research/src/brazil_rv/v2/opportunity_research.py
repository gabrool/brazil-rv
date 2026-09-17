"""CPU-only conditional opportunity identification and matched portfolio replays."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from urllib.request import urlopen
import json
import time

import numpy as np
import polars as pl
import torch

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.portfolio_policy import (
    CalibratedPolicy,
    exact_replay,
    policy_ledger_config,
)
from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_FOLDS, HORIZONS
from .controller_context import load_context, matured_shadow
from .decision_program import benchmark_excess_returns
from .objective_readouts import calibration, rank_view, read_panel
from .performance import align_benchmarks, performance
from .portfolio_program import PROJECT, read
from .portfolio_readouts import interval, save_book, verify_book
from .portfolio_training import load_data, windows
from .research_rounds import _git_identity
from .round5_derived import identity_axes

ARMS = ("TE_all", "C6")
ARRANGEMENTS = {
    "neutral": AllocationConfig(),
    "flexible_net": replace(AllocationConfig(), net_cap=0.45),
    "direction_permitted": replace(AllocationConfig(), net_cap=0.45, beta_cap=0.45),
    "sector_band": replace(AllocationConfig(), sector_net_cap=0.05),
    "sector_neutral": replace(AllocationConfig(), sector_net_cap=0.0),
}
EFFR_URL = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json?startDate=2016-01-01&endDate=2024-12-31"


def bound(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def checked(record):
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"opportunity source changed: {path}")
    return path


def targets(data):
    """Label-only shadow payoff, BOVA excess and average stock residual, per day."""
    market = benchmark_excess_returns(data)
    cash = np.full(len(market), np.nan)
    for t in range(len(market) - 5):
        cash[t] = np.prod(1 + data.inputs.cdi_returns[t + 1 : t + 6]) - 1
    h = HORIZONS.index(5)
    residual = (
        data.inputs.shareholder_simple_returns[..., h]
        - cash[:, None]
        - data.beta * market[:, None]
    )
    mask = (
        data.valid & data.inputs.shareholder_target_mask[..., h] & np.isfinite(residual)
    )
    count = mask.sum(1)
    x = data.ranks.mean(-1)
    x = x - (np.where(mask, x, 0).sum(1) / count.clip(1))[:, None]
    x = np.where(mask, x, 0)
    payoff = (x * np.where(mask, residual / 5, 0)).sum(1) / np.abs(x).sum(1).clip(1e-12)
    mean = np.where(mask, residual / 5, 0).sum(1) / count.clip(1)
    y = np.column_stack((payoff, market / 5, mean))
    y[count < 2] = np.nan
    return y, residual, mask


def fit_ridge(x, valid, y, rows, strength):
    """Fit-only median/IQR + smooth asinh; no evaluation statistics or clipping."""
    use = rows[np.isfinite(y[rows]).all(1)]
    observed = np.where(valid[use], x[use], np.nan)
    # All-missing fields carry no information; keep their coordinate at zero.
    usable = valid[use].any(0)
    median, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    median[usable] = np.nanmedian(observed[:, usable], axis=0)
    q = np.nanpercentile(observed[:, usable], (25, 75), axis=0)
    scale[usable] = np.maximum(q[1] - q[0], 1e-6)
    z = np.arcsinh(np.where(valid, (x - median) / scale, 0))
    z[:, ~usable] = 0
    center, average = z[use].mean(0), y[use].mean(0)
    a = z[use] - center
    coef = np.linalg.solve(
        a.T @ a + strength * np.eye(a.shape[1]), a.T @ (y[use] - average)
    )
    prediction = (z - center) @ coef + average
    return (
        prediction,
        average,
        {
            "strength": strength,
            "fit_count": len(use),
            "last_origin": int(use[-1]),
            "median": median.tolist(),
            "scale": scale.tolist(),
            "center": center.tolist(),
            "coefficient": coef.tolist(),
            "intercept": average.tolist(),
        },
    )


def rolling_predictions(x, valid, y, blocks, first):
    prediction, unconditional = np.full_like(y, np.nan), np.full_like(y, np.nan)
    fits = {}
    for fold, rows in blocks.items():
        if int(fold[1:]) < 5:
            continue
        cutoff = int(rows[0])
        available = np.arange(first, cutoff - 5)  # origin+5 <= cutoff-1
        selection = available[-126:]
        training = available[available + 5 < selection[0]]
        trials = []
        use = selection[np.isfinite(y[selection]).all(1)]
        for strength in (1.0, 10.0, 100.0):
            p, _, _ = fit_ridge(x, valid, y, training, strength)
            # Standardize target units by training-only variance so market scale
            # does not dominate selection of a common small regression.
            variance = np.nanvar(y[training, :2], axis=0).clip(1e-12)
            loss = float(((p[use, :2] - y[use, :2]) ** 2 / variance).mean())
            trials.append({"strength": strength, "selection_loss": loss})
        chosen = min(trials, key=lambda t: t["selection_loss"])["strength"]
        p, average, record = fit_ridge(x, valid, y, available, chosen)
        prediction[rows], unconditional[rows] = p[rows], average
        fits[fold] = record | {
            "trials": trials,
            "first_decision": cutoff,
            "last_usable_endpoint": cutoff - 1,
        }
    return prediction, unconditional, fits


class OpportunityPolicy(CalibratedPolicy):
    def __init__(self, base, predicted, *, directional=False):
        super().__init__(base)
        self.predicted, self.directional = predicted, directional

    def preference_for(self, data, day, names, state):
        all_rank = data.ranks[day].mean(-1)
        eligible = data.valid[day]
        center = all_rank[eligible].mean()
        x = all_rank[eligible] - center
        # Shadow payoff = slope * E[x²]/E[|x|]. This converts its forecast
        # back into per-stock daily residual-return units, including its sign.
        slope = (
            self.predicted[day, 0]
            * np.abs(x).mean()
            / max(float((x * x).mean()), 1e-12)
        )
        return torch.as_tensor(
            slope * (all_rank[names] - center) + self.predicted[day, 2],
            dtype=torch.float64,
        )

    def market_return_for(self, day):
        return float(self.predicted[day, 1]) if self.directional else 0.0


def prepare(root):
    torch.set_num_threads(1)
    code = _git_identity()
    old = Path(read(PROJECT / "docs/v2_decision_run.json")["root"])
    accepted = read(PROJECT / "docs/v2_data_inputs.json")["store"]
    if (
        sha256_file(Path(accepted["root"]) / "manifest.json")
        != accepted["manifest_sha256"]
    ):
        raise ValueError("accepted store changed")
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    # Resolve historical classification and FX from the accepted family lineage.
    family = read(PROJECT / "docs/v2_round5_store_acceptance.json")["families"]
    sector_source = next(
        f["admission"]["source_manifest"]
        for f in family
        if f.get("admission", {})
        .get("source_manifest", {})
        .get("path", "")
        .endswith("sector\\manifest.json")
    )
    source_inputs = read(checked(sector_source))["sources"]["inputs"]
    identity = pl.read_parquet(checked(source_inputs["identity"]))
    market = pl.read_parquet(checked(source_inputs["market_levels"]))
    fx = market.filter(pl.col("series") == "ptax_brl_per_usd").sort("reference_date")
    payload = urlopen(EFFR_URL, timeout=60).read()
    (inputs / "effr.json").write_bytes(payload)
    rates = sorted(json.loads(payload)["refRates"], key=lambda r: r["effectiveDate"])
    np.savez_compressed(
        inputs / "benchmarks.npz",
        fx_dates=fx["reference_date"].to_numpy(),
        fx=fx["value"].to_numpy(),
        rate_dates=np.asarray(
            [r["effectiveDate"] for r in rates], dtype="datetime64[D]"
        ),
        annual_us_percent=np.asarray([r["percentRate"] for r in rates]),
    )
    arms = {}
    for arm in ARMS:
        print(f"preparing {arm}", flush=True)
        data, binding = load_data(old, arm)
        blocks = {f: windows(old, data, f)["evaluation"] for f in DEVELOPMENT_FOLDS}
        rows = np.concatenate(list(blocks.values()))
        panels, masks, sources, disagreements = [], [], {}, []
        for fold, block in blocks.items():
            p, _, valid, source = read_panel(old, data, arm, "neutral", fold, block)
            panels.append(p["ensemble"])
            masks.append(valid)
            disagreements.append(
                np.std([p[str(s)] for s in (11, 29, 47)], axis=0).mean(-1)
            )
            sources[fold] = {
                "forecasts": source,
                "mapping": bound(old / "phase3/mappings" / f"{fold}.json"),
            }
        data = rank_view(data, rows, np.concatenate(panels), np.concatenate(masks))
        sectors, issuers = identity_axes(
            identity, data.inputs.dates, data.inputs.security_ids
        )
        context = load_context(old, arm, binding)
        y, residual, mask = targets(data)
        shadow, shadow_valid = matured_shadow(data.ranks, data.valid, residual, mask)
        for j, name in enumerate(
            ("shadow_payoff_bps", "shadow_volatility_bps", "shadow_mass")
        ):
            i = context.names.index(name)
            context.common[:, i] = shadow[:, j]
            context.valid[:, i] = shadow_valid[:, j]
            i = context.names.index(name.replace("_bps", "") + "_valid")
            context.common[:, i] = shadow_valid[:, j]
        count = data.valid.sum(1).clip(1)
        disagreement = np.zeros_like(data.valid, dtype=float)
        disagreement[rows] = np.concatenate(disagreements)
        context.common[:, context.names.index("mean_seed_disagreement")] = (
            disagreement * data.valid
        ).sum(1) / count
        context.common[:, context.names.index("mean_horizon_disagreement")] = (
            data.ranks.std(-1) * data.valid
        ).sum(1) / count
        path = inputs / f"{arm}.npz"
        np.savez_compressed(
            path,
            ranks=data.ranks,
            valid=data.valid,
            sectors=sectors.astype(str),
            issuers=issuers.astype(str),
            common=context.common,
            common_valid=context.valid,
            target=y,
            dates=np.asarray(data.inputs.dates, dtype="datetime64[D]"),
        )
        arms[arm] = {
            "cache_binding": binding,
            "prepared": bound(path),
            "sources": sources,
            "blocks": {f: b.tolist() for f, b in blocks.items()},
            "context_names": context.names,
            "sector_known_active_fraction": float((sectors[data.valid] != "").mean()),
            "issuer_known_active_fraction": float((issuers[data.valid] != "").mean()),
        }
    design = {
        "implementation": code,
        "source_root": str(old),
        "source_design": bound(old / "frozen_design.json"),
        "store": accepted,
        "registration": bound(
            PROJECT / "research/preregistrations/v2_opportunity_portfolios.md"
        ),
        "sector_source": sector_source,
        "identity": source_inputs["identity"],
        "market": source_inputs["market_levels"],
        "effr": bound(inputs / "effr.json") | {"url": EFFR_URL},
        "benchmarks": bound(inputs / "benchmarks.npz"),
        "arms": arms,
        "arrangements": {k: asdict(v) for k, v in ARRANGEMENTS.items()},
        "forward_capture": False,
        "heldout_accessed": False,
    }
    write_json_atomic(root / "frozen_design.json", design)
    return {"root": str(root), "status": "prepared"}


def benchmark_for(root, dates, previous):
    with np.load(root / "inputs/benchmarks.npz", allow_pickle=False) as source:
        return align_benchmarks(dates, previous, **dict(source))


def enriched_book(root, path, data, sectors, issuers, config):
    book = read(path / "book.json")
    dates = np.asarray(book["dates"], dtype="datetime64[D]")
    axis = np.asarray(data.inputs.dates, dtype="datetime64[D]")
    rows = np.searchsorted(axis, dates)
    daily = {k: np.asarray(v) for k, v in book["daily"].items()}
    bench = benchmark_for(root, dates, axis[rows[0] - 1])
    metrics = performance(daily["absolute_bps"] / 1e4, daily["cdi_bps"] / 1e4, bench)
    with np.load(path / "account.npz", allow_pickle=False) as account:
        prior = account["prior_weights"][-len(rows) :]
        stock = (
            account["signed_shares"][-len(rows) :]
            * np.nan_to_num(account["mark_price"][-len(rows) :])
            / account["nav"][-len(rows) :, None]
        )
        planned = account["targets"][-len(rows) :] if "targets" in account else None
    market_daily = (
        data.inputs.bova11_close[rows] / data.inputs.bova11_close[rows - 1] - 1
    )
    market_component = (prior[:, :-1] * data.beta[rows]).sum(1) * market_daily * 1e4
    daily["equity_beta_market_bps"] = market_component
    daily["equity_residual_bps"] = daily["equity_gross_bps"] - market_component
    daily["market_plus_hedge_bps"] = market_component + daily["hedge_gross_bps"]
    daily["unknown_sector_gross"] = (np.abs(stock) * (sectors[rows] == "")).sum(1)
    # A label-based contribution proxy cannot claim exact cash-action attribution.
    # Keep the difference against reconciled ledger stock P&L explicitly visible.
    h1 = HORIZONS.index(1)
    one = data.inputs.shareholder_simple_returns[rows - 1, :, h1]
    observed = data.inputs.shareholder_target_mask[rows - 1, :, h1] & np.isfinite(one)
    contribution = prior[:, :-1] * np.where(observed, one, 0) * 1e4
    daily["unattributed_stock_bps"] = daily["equity_gross_bps"] - contribution.sum(1)
    groups = {}
    for category, labels in (("sector", sectors[rows]), ("issuer", issuers[rows])):
        totals = {}
        max_net = np.zeros(len(rows))
        for label in np.unique(labels):
            membership = labels == label
            net = (stock * membership).sum(1)
            if label:
                max_net = np.maximum(max_net, np.abs(net))
            totals[str(label or "UNKNOWN")] = {
                "mean_contribution_proxy_bps": float(
                    (contribution * membership).sum(1).mean()
                ),
                "mean_gross": float((np.abs(stock) * membership).sum(1).mean()),
                "max_absolute_net": float(np.abs(net).max()),
            }
        groups[category] = totals
        daily[f"max_{category}_absolute_net"] = max_net
    labels = np.asarray(
        [str(d)[:4] + ("H1" if int(str(d)[5:7]) <= 6 else "H2") for d in dates]
    )
    halves = {
        label: {k: float(v[labels == label].mean()) for k, v in daily.items()}
        for label in np.unique(labels)
    }
    planned_sector = 0.0
    if planned is not None and config.sector_net_cap is not None:
        for i, day in enumerate(rows):
            for label in np.unique(sectors[day]):
                if label:
                    planned_sector = max(
                        planned_sector,
                        abs(float(planned[i, :-1][sectors[day] == label].sum())),
                    )
        if planned_sector > config.sector_net_cap + 2e-6:
            raise ValueError("saved plan violates dated sector constraints")
    result = {
        "book": bound(path / "book.json"),
        "metrics": metrics,
        "half_year": halves,
        "attribution": groups,
        "max_planned_sector_net": planned_sector,
        "planned_limits": asdict(config),
        "realized_drift_days_above_planned_limits": int(
            (
                (daily["gross"] > config.gross_cap + 2e-6)
                | (np.abs(daily["signed_net"]) > config.net_cap + 2e-6)
                | (np.abs(daily["beta"]) > config.beta_cap + 2e-6)
            ).sum()
        ),
        "economic_limitations": {
            "economics_unresolved": book["summary"]["economics_unresolved"],
            "terminal_haircut_scenario_relative_difference": book["summary"][
                "terminal_haircut_scenario_relative_difference"
            ],
            "sector_contribution_is_label_based_proxy": True,
        },
        "daily": {k: v.tolist() for k, v in daily.items()},
    }
    write_json_atomic(path / "analysis.json", result)
    return result


def run_arm(root, arm):
    torch.set_num_threads(1)
    design = read(root / "frozen_design.json")
    code = _git_identity()
    if code != design["implementation"]:
        raise ValueError("financial run must use its frozen source commit")
    old = Path(design["source_root"])
    data, binding = load_data(old, arm)
    entry = design["arms"][arm]
    if binding != entry["cache_binding"]:
        raise ValueError("economic cache differs")
    with np.load(checked(entry["prepared"]), allow_pickle=False) as payload:
        data = rank_view(
            data, np.arange(len(data.valid)), payload["ranks"], payload["valid"]
        )
        sectors, issuers = payload["sectors"].copy(), payload["issuers"].copy()
        x, valid, y = (payload[n].copy() for n in ("common", "common_valid", "target"))
    data.sectors = sectors
    blocks = {f: np.asarray(b) for f, b in entry["blocks"].items()}
    rows = np.sort(np.concatenate(list(blocks.values())))
    mappings = {
        f: calibration(read(checked(entry["sources"][f]["mapping"]))["arms"][arm])
        for f in blocks
    }
    models = {
        int(d): CalibratedPolicy(mappings[f]) for f, b in blocks.items() for d in b
    }
    start, stop = int(rows[0]), int(rows[-1] + 1)
    predicted, unconditional, fits = rolling_predictions(x, valid, y, blocks, start)
    # Only payoff and market are conditioned; common stock residual stays at the
    # identical historical mean so conditional timing is a controlled contrast.
    predicted[:, 2] = unconditional[:, 2]
    forecasts = root / arm / "supervised.npz"
    forecasts.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        forecasts, predicted=predicted, unconditional=unconditional, realized=y
    )
    write_json_atomic(root / arm / "fits.json", fits)
    trials = [(name, cfg, start, models) for name, cfg in ARRANGEMENTS.items()]
    conditional_start = int(blocks["F5"][0])
    for name, p, directional in (
        ("supervised_unconditional", unconditional, False),
        ("supervised_conditional", predicted, False),
        ("supervised_flexible", predicted, False),
        ("supervised_directional", predicted, True),
    ):
        cfg = (
            ARRANGEMENTS["direction_permitted"]
            if directional or name == "supervised_flexible"
            else ARRANGEMENTS["neutral"]
        )
        policy = OpportunityPolicy(mappings["F5"], p, directional=directional)
        trials.append((name, cfg, conditional_start, policy))
    # A matched empty-start control avoids attributing inherited 2018 inventory
    # to the supervised experiment that can only begin in 2020.
    trials.append(("neutral_2020", ARRANGEMENTS["neutral"], conditional_start, models))
    for name, cfg, first, policy in trials:
        destination = root / arm / "books" / name
        provenance = {
            "scenario": "base",
            "policy": name,
            "arm": arm,
            "implementation": code,
            "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
            "allocation": asdict(cfg),
            "first": first,
            "stop": stop,
        }
        t = time.monotonic()
        if (destination / "book.json").exists():
            verify_book(destination, provenance)
        else:
            ledger = policy_ledger_config(
                planned_absolute_net_cap=cfg.net_cap,
                planned_absolute_beta_cap=cfg.beta_cap,
            )
            result, target, previous = exact_replay(
                data, policy, first, stop, config=ledger, allocation=cfg
            )
            save_book(
                destination, data, result, target, previous, first, first, provenance
            )
            del result, target, previous
        enriched_book(root, destination, data, sectors, issuers, cfg)
        if name == "neutral":
            original = read(
                old / "phase3/continuous" / arm / "neutral/ensemble/book.json"
            )
            new = read(destination / "book.json")
            error = max(
                float(
                    np.max(np.abs(np.asarray(original["daily"][k]) - new["daily"][k]))
                )
                for k in original["daily"]
            )
            if error > 1e-7:
                raise ValueError(f"neutral control fails sealed replay: {error}")
        print(
            json.dumps(
                {"arm": arm, "book": name, "seconds": round(time.monotonic() - t, 1)}
            ),
            flush=True,
        )
    write_json_atomic(
        root / arm / "complete.json",
        {
            "books": len(trials),
            "neutral_max_daily_error": error,
            "supervised": bound(forecasts),
            "fits": bound(root / arm / "fits.json"),
        },
    )


def summarize(root):
    design = read(root / "frozen_design.json")
    output = {"frozen_design": bound(root / "frozen_design.json"), "arms": {}}
    for arm in ARMS:
        complete = read(root / arm / "complete.json")
        books = {
            p.parent.name: read(p)
            for p in (root / arm / "books").glob("*/analysis.json")
        }
        dates = np.asarray(
            read(root / arm / "books/neutral/book.json")["dates"], dtype="datetime64[D]"
        )
        contrasts = {}
        for name, book in books.items():
            control = (
                "supervised_unconditional"
                if name.startswith("supervised_")
                else ("neutral_2020" if name == "neutral_2020" else "neutral")
            )
            if name == "supervised_directional":
                control = "supervised_flexible"
            elif name == "supervised_flexible":
                control = "supervised_conditional"
            a = np.asarray(book["daily"]["net_excess_bps"])
            b = np.asarray(books[control]["daily"]["net_excess_bps"])
            if len(a) != len(b):
                raise ValueError("paired economic dates differ")
            d = dates[-len(a) :]
            periods = {
                "all": np.ones(len(a), bool),
                "screen_2020_2021": (d >= np.datetime64("2020-01-01"))
                & (d < np.datetime64("2022-01-01")),
                "later_2022_2024": d >= np.datetime64("2022-01-01"),
            }
            contrasts[name] = {
                "control": control,
                "net_bps": {
                    p: {
                        str(block): interval([(a - b)[m]], block_length=block)
                        for block in (20, 40, 60)
                    }
                    for p, m in periods.items()
                    if m.sum() >= 60
                },
            }
        with np.load(checked(complete["supervised"]), allow_pickle=False) as p:
            y, prediction, base = p["realized"], p["predicted"], p["unconditional"]
        entry = design["arms"][arm]
        with np.load(checked(entry["prepared"]), allow_pickle=False) as p:
            axis = p["dates"]
        learnability = {}
        for i, label in enumerate(("shadow_payoff", "market_excess")):
            result = {}
            for period, condition in (
                (
                    "screen_2020_2021",
                    (axis >= np.datetime64("2020-01-01"))
                    & (axis < np.datetime64("2022-01-01")),
                ),
                ("later_2022_2024", axis >= np.datetime64("2022-01-01")),
            ):
                m = condition & np.isfinite(prediction[:, i]) & np.isfinite(y[:, i])
                gain = (
                    (base[m, i] - y[m, i]) ** 2 - (prediction[m, i] - y[m, i]) ** 2
                ) * 1e8
                result[period] = {
                    "squared_error_improvement_bps_squared": {
                        str(b): interval([gain], block_length=b) for b in (20, 40, 60)
                    },
                    "conditional_rmse_bps": float(
                        np.sqrt(np.mean((prediction[m, i] - y[m, i]) ** 2)) * 1e4
                    ),
                    "unconditional_rmse_bps": float(
                        np.sqrt(np.mean((base[m, i] - y[m, i]) ** 2)) * 1e4
                    ),
                    "predicted_actual_correlation": float(
                        np.corrcoef(prediction[m, i], y[m, i])[0, 1]
                    ),
                }
            learnability[label] = result
        output["arms"][arm] = {
            "completion": complete,
            "learnability": learnability,
            "contrasts": contrasts,
            "books": {
                k: {n: v for n, v in b.items() if n != "daily"}
                for k, b in books.items()
            },
        }
    write_json_atomic(root / "summary.json", output)
    write_json_atomic(PROJECT / "docs/v2_opportunity_results.json", output)
    return {"status": "completed", "summary": str(root / "summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.task == "prepare":
        print(prepare(args.root))
    elif args.task == "run":
        run_arm(args.root, args.arm)
    else:
        print(summarize(args.root))


if __name__ == "__main__":
    main()
