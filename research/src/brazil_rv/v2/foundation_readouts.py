"""Matched neutral books and seed-size diagnostics; no new model training."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay
from .artifacts import sha256_file, write_json_atomic
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS
from .objective_readouts import calibration, forecast_readout, rank_view, read_panel
from .opportunity_research import benchmark_for, bound
from .performance import performance
from .portfolio_inputs import HEADS, normalized_ranks
from .portfolio_program import PROJECT, read
from .portfolio_readouts import block_draws, save_book, verify_book
from .portfolio_training import load_data, windows
from .research_rounds import _git_identity, _score_artifact
from .round7 import SCREEN_FOLDS
from .train import rank_average_ensemble


def paired_interval(arrays, block_length=40):
    """Same circular date-block protocol, with both registered confidence levels."""
    arrays = tuple(np.asarray(a, dtype=float) for a in arrays)
    lengths = tuple(len(a) for a in arrays)
    if min(lengths) < block_length or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("paired intervals require finite common dates and full blocks")
    draws = np.zeros(10_000)
    for a, starts in zip(arrays, block_draws(lengths, block_length), strict=True):
        prefix = np.r_[0.0, np.cumsum(np.r_[a, a[:block_length]])]
        sizes = np.full(starts.shape[1], block_length)
        sizes[-1] = len(a) - block_length * (len(sizes) - 1)
        draws += (prefix[starts + sizes] - prefix[starts]).sum(1)
    draws /= sum(lengths)
    return {
        "estimate": float(np.concatenate(arrays).mean()),
        **{
            f"{edge}_{level}": float(np.quantile(draws, q))
            for level, low, high in ((90, 0.05, 0.95), (95, 0.025, 0.975))
            for edge, q in (("lower", low), ("upper", high))
        },
        "replications": 10000,
        "block_length": block_length,
        "fold_boundary_preserved": True,
        "nominal_development_interval": True,
    }


def new_panel(root, data, cell, fold, rows, rule):
    design = read(root / "frozen_design.json")
    store = Path(design["store"]["root"])
    schema = read(store / "manifest.json")["metadata"]["feature_schema"]["sha256"]
    panels, sources, common = {}, {}, None
    for seed in ALLOWED_SEEDS:
        fit = root / "fits" / cell / f"{fold}_seed_{seed}"
        record = read(fit / "run_manifest.json")
        scores = fit / ("scores" if rule == "raw" else "ema_scores")
        checkpoint = "selected.pt" if rule == "raw" else "selected_ema.pt"
        score_record = read(scores / "score_manifest.json")
        if (
            record["status"] != "completed"
            or record["seed"] != seed
            or record["fold"] != fold
            or record["contract"]["store_manifest_sha256"]
            != design["store"]["manifest_sha256"]
            or score_record["round7_contract"] != record["contract"]
            or sha256_file(fit / checkpoint) != record["artifacts"][checkpoint]
            or score_record["checkpoint"]["sha256"] != record["artifacts"][checkpoint]
        ):
            raise ValueError("foundation score provenance differs from its fit")
        values, mask = _score_artifact(
            scores,
            require_clean_transfer=True,
            expected_dates=np.asarray(data.inputs.dates, dtype="datetime64[D]")[rows],
            expected_isins=data.inputs.security_ids,
            expected_feature_schema_sha256=schema,
        )
        mask = mask[..., HEADS]
        valid = mask.all(-1) & data.inputs.active[rows]
        if not np.array_equal(valid, data.inputs.active[rows]) or (
            common is not None and not np.array_equal(valid, common)
        ):
            raise ValueError("foundation score loses eligible names")
        common = valid
        panels[str(seed)] = normalized_ranks(
            rank_average_ensemble([values[..., HEADS]], mask), mask
        )
        sources[str(seed)] = {
            "fit": bound(fit / "run_manifest.json"),
            "scores": bound(scores / "score_manifest.json"),
            "checkpoint": bound(fit / checkpoint),
        }
    panels["ensemble"] = np.mean(list(panels.values()), axis=0)
    return panels, common, sources


def book(output, data, rows, ranks, valid, mapping, provenance, benchmark_root):
    if (output / "book.json").exists():
        record = verify_book(output, provenance)
    else:
        view = rank_view(data, rows, ranks, valid)
        result, targets, previous = exact_replay(
            view, CalibratedPolicy(mapping), int(rows[0]), int(rows[-1] + 1)
        )
        record = save_book(
            output,
            view,
            result,
            targets,
            previous,
            int(rows[0]),
            int(rows[0]),
            provenance,
        )
    forecast = forecast_readout(data, rows, ranks, valid, mapping)
    write_json_atomic(output / "forecast_readout.json", forecast)
    bench = benchmark_for(
        benchmark_root,
        data.inputs.dates[rows[0] : rows[-1] + 1],
        data.inputs.dates[rows[0] - 1],
    )
    write_json_atomic(
        output / "performance.json",
        performance(
            np.asarray(record["daily"]["absolute_bps"]) / 1e4,
            np.asarray(record["daily"]["cdi_bps"]) / 1e4,
            bench,
        ),
    )
    return record, forecast


def evaluate_wave(root, wave_name, folds):
    torch.set_num_threads(1)
    design = read(root / "frozen_design.json")
    wave = read(root / "waves" / f"{wave_name}.json")
    old = Path(design["prior_decision_root"])
    data, binding = load_data(old, "C6")
    benchmark_root = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    implementation = _git_identity()
    for fold in folds:
        rows = windows(old, data, fold)["evaluation"]
        mappings = read(old / "phase3/mappings" / f"{fold}.json")
        for cell in wave["cells"]:
            mapping = calibration(mappings["arms"]["TE_all"])
            for rule in ("raw", "ema"):
                panels, valid, sources = new_panel(root, data, cell, fold, rows, rule)
                for member, ranks in panels.items():
                    output = root / "books" / cell / rule / fold / member
                    provenance = {
                        "implementation": implementation,
                        "scenario": "base",
                        "policy": "equal_rank",
                        "economic_cache_binding": binding,
                        "mapping": bound(old / "phase3/mappings" / f"{fold}.json"),
                        "forecasts": sources,
                        "member": member,
                        "rule": rule,
                        "initial_state": "cash at first evaluation date",
                        "benchmark": bound(benchmark_root / "inputs/benchmarks.npz"),
                    }
                    book(
                        output,
                        data,
                        rows,
                        ranks,
                        valid,
                        mapping,
                        provenance,
                        benchmark_root,
                    )
                print({"completed_readout": [cell, rule, fold]}, flush=True)


def ensemble_diagnostics(root):
    torch.set_num_threads(1)
    old = Path(read(root / "frozen_design.json")["prior_decision_root"])
    data, binding = load_data(old, "C6")
    benchmark_root = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    implementation = _git_identity()
    all_results = {}
    for arm in ("C6", "TE_all"):
        by_member, pair_correlations = {}, []
        for fold in DEVELOPMENT_FOLDS:
            rows = windows(old, data, fold)["evaluation"]
            mapping = calibration(
                read(old / "phase3/mappings" / f"{fold}.json")["arms"][arm]
            )
            panels, _, valid, sources = read_panel(
                old, data, arm, "neutral", fold, rows
            )
            if fold == "F2":
                reference = read(
                    old / "phase3/books" / fold / arm / "neutral/ensemble/book.json"
                )
                view = rank_view(data, rows, panels["ensemble"], valid)
                replay, _, _ = exact_replay(
                    view, CalibratedPolicy(mapping), int(rows[0]), int(rows[-1] + 1)
                )
                error = float(
                    np.max(
                        np.abs(
                            replay.net_excess_all_cash_bps
                            - np.asarray(reference["daily"]["net_excess_bps"])
                        )
                    )
                )
                if error > 1e-8:
                    raise ValueError(
                        "current neutral account does not reproduce the sealed ensemble"
                    )
                write_json_atomic(
                    root / "seed_size" / arm / "account_bridge.json",
                    {
                        "fold": fold,
                        "max_abs_daily_net_bps_error": error,
                        "source": bound(
                            old
                            / "phase3/books"
                            / fold
                            / arm
                            / "neutral/ensemble/book.json"
                        ),
                        "implementation": implementation,
                        "passed": True,
                    },
                )
            for pair in combinations(map(str, ALLOWED_SEEDS), 2):
                panels["+".join(pair)] = np.mean([panels[s] for s in pair], axis=0)
                for day in range(len(rows)):
                    mask = valid[day]
                    a, b = (panels[s][day, mask].mean(-1) for s in pair)
                    pair_correlations.append(float(np.corrcoef(a, b)[0, 1]))
            for member, ranks in panels.items():
                # Reuse sealed single-seed/ensemble accounts exactly; only pairs need new books.
                if "+" not in member:
                    path = old / "phase3/books" / fold / arm / "neutral" / member
                    record = read(path / "book.json")
                    verify_book(path, record["provenance"])
                    if record["dates"] != [
                        data.inputs.dates[i].isoformat() for i in rows
                    ]:
                        raise ValueError("sealed member book dates differ")
                    forecast = read(path / "forecast_readout.json")
                else:
                    path = root / "seed_size" / arm / fold / member
                    record, forecast = book(
                        path,
                        data,
                        rows,
                        ranks,
                        valid,
                        mapping,
                        {
                            "implementation": implementation,
                            "scenario": "base",
                            "policy": "equal_rank",
                            "economic_cache_binding": binding,
                            "forecasts": sources,
                            "member": member,
                            "mapping": bound(old / "phase3/mappings" / f"{fold}.json"),
                            "initial_state": "cash at first evaluation date",
                            "benchmark": bound(
                                benchmark_root / "inputs/benchmarks.npz"
                            ),
                        },
                        benchmark_root,
                    )
                by_member.setdefault(member, []).append(
                    {
                        "fold": fold,
                        "book": bound(path / "book.json"),
                        "mean_ic": forecast["neutral_ic_mean"],
                        "summary": record["summary"],
                        "sessions": len(rows),
                        "ic_sessions": int(
                            np.isfinite(np.asarray(forecast["neutral_ic"], float)).sum()
                        ),
                    }
                )
            print({"ensemble_diagnostic": [arm, fold]}, flush=True)
        member_means = {
            member: {
                "ic": float(
                    np.average(
                        [v["mean_ic"] for v in records],
                        weights=[v["ic_sessions"] for v in records],
                    )
                ),
                **{
                    key: float(
                        np.average(
                            [v["summary"]["mean"][key] for v in records],
                            weights=[v["sessions"] for v in records],
                        )
                    )
                    for key in ("net_excess_bps", "utility_bps", "turnover")
                },
            }
            for member, records in by_member.items()
        }
        all_results[arm] = {
            "members": by_member,
            "member_means": member_means,
            "mean_by_ensemble_size": {
                str(size): {
                    key: float(
                        np.mean(
                            [
                                v[key]
                                for member, v in member_means.items()
                                if (
                                    3
                                    if member == "ensemble"
                                    else len(member.split("+"))
                                )
                                == size
                            ]
                        )
                    )
                    for key in ("ic", "net_excess_bps", "utility_bps", "turnover")
                }
                for size in (1, 2, 3)
            },
            "mean_pair_forecast_correlation": float(np.nanmean(pair_correlations)),
            "interpretation": "All seeds and pairs reported; common prior ensemble calibration; shared pairs are dependent. Fold accounts restart in cash.",
        }
        write_json_atomic(root / "ensemble_diagnostics.json", all_results)
    return all_results


def summarize_ensembles(root):
    diagnostics = read(root / "ensemble_diagnostics.json")
    results = {}
    for arm, values in diagnostics.items():
        arrays = {"net_excess_bps": [], "ic": []}
        for index in range(len(DEVELOPMENT_FOLDS)):
            paths = [
                Path(values["members"][member][index]["book"]["path"])
                for member in (*map(str, ALLOWED_SEEDS), "ensemble")
            ]
            books = [read(p) for p in paths]
            forecasts = [
                np.asarray(
                    read(p.with_name("forecast_readout.json"))["neutral_ic"], float
                )
                for p in paths
            ]
            if any(b["dates"] != books[-1]["dates"] for b in books[:-1]):
                raise ValueError("seed-size diagnostic dates differ")
            valid = np.isfinite(forecasts[-1])
            if any(not np.array_equal(np.isfinite(f), valid) for f in forecasts[:-1]):
                raise ValueError("seed-size IC availability differs")
            arrays["ic"].append(
                forecasts[-1][valid]
                - np.mean([f[valid] for f in forecasts[:-1]], axis=0)
            )
            arrays["net_excess_bps"].append(
                np.asarray(books[-1]["daily"]["net_excess_bps"])
                - np.mean([b["daily"]["net_excess_bps"] for b in books[:-1]], axis=0)
            )
        results[arm] = {
            "mean_by_ensemble_size": values["mean_by_ensemble_size"],
            "mean_pair_forecast_correlation": values["mean_pair_forecast_correlation"],
            "three_minus_mean_individual": {
                key: {str(b): paired_interval(parts, b) for b in (20, 40, 60)}
                for key, parts in arrays.items()
            },
        }
    write_json_atomic(
        root / "ensemble_summary.json",
        {
            "source": bound(root / "ensemble_diagnostics.json"),
            "results": results,
            "scope": "Matched earlier neutral F trajectories; descriptive reused development results; fold accounts start in cash.",
        },
    )
    return results


def evaluate_averages(root):
    """Compare fixed within-trajectory means with their exact original raw controls."""
    torch.set_num_threads(1)
    design = read(root / "frozen_design.json")
    plan = read(root / "averaging_plan.json")
    old = Path(design["prior_decision_root"])
    store = Path(design["store"]["root"])
    schema = read(store / "manifest.json")["metadata"]["feature_schema"]["sha256"]
    data, binding = load_data(old, "C6")
    benchmark_root = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    implementation = _git_identity()
    results = {}
    for arm in ("C6", "TE_all"):
        references = {}
        for fold in DEVELOPMENT_FOLDS:
            rows = windows(old, data, fold)["evaluation"]
            mapping = calibration(
                read(old / "phase3/mappings" / f"{fold}.json")["arms"][arm]
            )
            panels, sources = {}, {}
            valid = data.inputs.active[rows]
            for seed in ALLOWED_SEEDS:
                item = plan["checkpoints"][f"{arm}/{fold}/{seed}"]
                checkpoint = Path(item["path"])
                scores = checkpoint.parent / "scores"
                manifest = read(scores / "score_manifest.json")
                if (
                    sha256_file(checkpoint) != item["sha256"]
                    or manifest["checkpoint"]["sha256"] != item["sha256"]
                    or manifest["store"]["manifest_sha256"]
                    != design["store"]["manifest_sha256"]
                ):
                    raise ValueError("averaged scores differ from their bound source")
                values, mask = _score_artifact(
                    scores,
                    require_clean_transfer=True,
                    expected_dates=np.asarray(data.inputs.dates, dtype="datetime64[D]")[
                        rows
                    ],
                    expected_isins=data.inputs.security_ids,
                    expected_feature_schema_sha256=schema,
                )
                mask = mask[..., HEADS]
                if not np.array_equal(mask.all(-1) & valid, valid):
                    raise ValueError("averaged score loses eligible names")
                panels[str(seed)] = normalized_ranks(
                    rank_average_ensemble([values[..., HEADS]], mask), mask
                )
                sources[str(seed)] = {
                    "checkpoint": bound(checkpoint),
                    "scores": bound(scores / "score_manifest.json"),
                    "averaging": bound(checkpoint.parent / "average.json"),
                }
            panels["ensemble"] = np.mean(list(panels.values()), axis=0)
            for member, ranks in panels.items():
                reference = old / "phase3/books" / fold / arm / "neutral" / member
                verify_book(reference, read(reference / "book.json")["provenance"])
                references[fold, member] = reference
                book(
                    root / "books" / arm / "average" / fold / member,
                    data,
                    rows,
                    ranks,
                    valid,
                    mapping,
                    {
                        "implementation": implementation,
                        "scenario": "base",
                        "policy": "equal_rank",
                        "economic_cache_binding": binding,
                        "forecasts": sources,
                        "member": member,
                        "rule": "trailing up to three epochs ending at raw selection",
                        "mapping": bound(old / "phase3/mappings" / f"{fold}.json"),
                        "reference": bound(reference / "book.json"),
                        "benchmark": bound(benchmark_root / "inputs/benchmarks.npz"),
                        "initial_state": "cash at first evaluation date",
                    },
                    benchmark_root,
                )
            print({"averaging_readout": [arm, fold]}, flush=True)
        results[arm] = {
            "screen": compare(
                root,
                arm,
                arm,
                SCREEN_FOLDS,
                "average",
                allow_noninferiority=False,
                control_paths=references,
            ),
            "all_development": compare(
                root,
                arm,
                arm,
                DEVELOPMENT_FOLDS,
                "average",
                allow_noninferiority=False,
                control_paths=references,
            ),
        }
        write_json_atomic(root / "averaging_summary.json", results)
    return results


def fixed_blend(root):
    """Fixed half-C6/half-attention forecast diagnostic; no evaluation-chosen weight."""
    torch.set_num_threads(1)
    old = Path(read(root / "frozen_design.json")["prior_decision_root"])
    data, binding = load_data(old, "C6")
    benchmark_root = Path(read(PROJECT / "docs/v2_opportunity_run.json")["root"])
    implementation = _git_identity()
    records = []
    differences = {arm: {"ic": [], "net_bps": []} for arm in ("C6", "TE_all")}
    for fold in DEVELOPMENT_FOLDS:
        rows = windows(old, data, fold)["evaluation"]
        mappings = read(old / "phase3/mappings" / f"{fold}.json")
        # This calibration was fitted before evaluation on the same fixed .5 blend.
        trial = next(t for t in mappings["blend"]["trials"] if t["te_weight"] == 0.5)
        mapping = calibration(trial["calibration"])
        panels, sources = {}, {}
        valid = data.inputs.active[rows]
        for arm in differences:
            panel, _, mask, sources[arm] = read_panel(
                old, data, arm, "neutral", fold, rows
            )
            if not np.array_equal(mask, valid):
                raise ValueError("cross-architecture blend populations differ")
            panels[arm] = panel["ensemble"]
        output = root / "fixed_blend" / fold
        record, forecast = book(
            output,
            data,
            rows,
            np.mean(list(panels.values()), axis=0),
            valid,
            mapping,
            {
                "implementation": implementation,
                "scenario": "base",
                "policy": "equal_rank",
                "economic_cache_binding": binding,
                "forecasts": sources,
                "attention_weight": 0.5,
                "mapping": bound(old / "phase3/mappings" / f"{fold}.json"),
                "benchmark": bound(benchmark_root / "inputs/benchmarks.npz"),
                "initial_state": "cash at first evaluation date",
            },
            benchmark_root,
        )
        for arm, delta in differences.items():
            path = old / "phase3/books" / fold / arm / "neutral/ensemble"
            reference = verify_book(path, read(path / "book.json")["provenance"])
            original = read(path / "forecast_readout.json")
            if (
                reference["dates"] != record["dates"]
                or original["dates"] != forecast["dates"]
            ):
                raise ValueError("fixed blend comparison dates differ")
            a, b = (np.asarray(v["neutral_ic"], float) for v in (forecast, original))
            if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                raise ValueError("fixed blend IC populations differ")
            delta["ic"].append((a - b)[np.isfinite(a)])
            delta["net_bps"].append(
                np.asarray(record["daily"]["net_excess_bps"])
                - np.asarray(reference["daily"]["net_excess_bps"])
            )
        records.append(
            {
                "fold": fold,
                "book": bound(output / "book.json"),
                "mean_ic": forecast["neutral_ic_mean"],
                "summary": record["summary"],
            }
        )
        print({"fixed_blend_readout": fold}, flush=True)
    result = {
        "books": records,
        "differences": {
            arm: {
                metric: {str(b): paired_interval(parts, b) for b in (20, 40, 60)}
                for metric, parts in delta.items()
            }
            for arm, delta in differences.items()
        },
        "scope": "Fixed .5 forecast blend; reused development ensemble diagnostic, not promotion.",
    }
    write_json_atomic(root / "fixed_blend_summary.json", result)
    return result


def compare(
    root,
    candidate,
    control,
    folds,
    candidate_rule="raw",
    control_rule="raw",
    *,
    allow_noninferiority=True,
    control_paths=None,
):
    metrics = {}
    for member in (*map(str, ALLOWED_SEEDS), "ensemble"):
        series = {"ic": [], "net_bps": []}
        for fold in folds:
            paths = [
                root / "books" / cell / rule / fold / member
                for cell, rule in ((candidate, candidate_rule), (control, control_rule))
            ]
            if control_paths is not None:
                paths[1] = control_paths[fold, member]
            books = [read(p / "book.json") for p in paths]
            forecasts = [read(p / "forecast_readout.json") for p in paths]
            if (
                books[0]["dates"] != books[1]["dates"]
                or forecasts[0]["dates"] != forecasts[1]["dates"]
            ):
                raise ValueError("comparison dates differ")
            ic = [np.asarray(f["neutral_ic"], dtype=float) for f in forecasts]
            common = np.isfinite(ic[0]) & np.isfinite(ic[1])
            if not np.array_equal(np.isfinite(ic[0]), np.isfinite(ic[1])):
                raise ValueError("comparison defined-IC populations differ")
            series["ic"].append((ic[0] - ic[1])[common])
            series["net_bps"].append(
                np.asarray(books[0]["daily"]["net_excess_bps"])
                - np.asarray(books[1]["daily"]["net_excess_bps"])
            )
        metrics[member] = {
            key: {
                "intervals": {
                    str(length): paired_interval(parts, length)
                    for length in (20, 40, 60)
                },
                "fold_means": dict(
                    zip(folds, [float(v.mean()) for v in parts], strict=True)
                ),
            }
            for key, parts in series.items()
        }
    primary = {k: v["intervals"]["40"] for k, v in metrics["ensemble"].items()}
    claims = []
    for endpoint, threshold in (("ic", 0.001), ("net_bps", 0.25)):
        if (
            primary[endpoint]["estimate"] >= threshold
            and sum(v > 0 for v in metrics["ensemble"][endpoint]["fold_means"].values())
            >= int(np.ceil(0.75 * len(folds)))
            and sum(
                metrics[str(s)][endpoint]["intervals"]["40"]["estimate"] > 0
                for s in ALLOWED_SEEDS
            )
            >= 2
        ):
            claims.append(endpoint)
    margins = {"ic": -0.001, "net_bps": -0.25}
    improvement = bool(claims) and all(
        primary[k]["estimate"] > margin for k, margin in margins.items()
    )
    noninferior = all(primary[k]["lower_90"] > margin for k, margin in margins.items())
    return {
        "candidate": candidate,
        "control": control,
        "candidate_rule": candidate_rule,
        "control_rule": control_rule,
        "metrics": metrics,
        "screen_improvement": improvement,
        "claimed_endpoints": claims,
        "screen_noninferiority": noninferior,
        "admitted": improvement or (noninferior and allow_noninferiority),
    }


def summarize(root, wave_name, folds):
    wave = read(root / "waves" / f"{wave_name}.json")
    comparisons = {}
    for cell in wave["cells"]:
        if cell != wave["control"]:
            comparisons[cell] = compare(root, cell, wave["control"], folds)
    # EMA is a fixed shared-trajectory comparison, not a new fit or a selector picked from returns.
    comparisons["ema_control"] = compare(
        root, wave["control"], wave["control"], folds, "ema", allow_noninferiority=False
    )
    result = {
        "wave": wave_name,
        "folds": list(folds),
        "comparisons": comparisons,
        "survivors": [
            name for name, result in comparisons.items() if result["admitted"]
        ],
        "status": "wave_review_required",
        "promotion": False,
    }
    write_json_atomic(root / f"{wave_name}_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "ensemble",
            "ensemble-summary",
            "averages",
            "blend",
            "evaluate",
            "summarize",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--wave", default="input")
    args = parser.parse_args()
    if args.command == "ensemble":
        ensemble_diagnostics(args.root)
        summarize_ensembles(args.root)
    elif args.command == "ensemble-summary":
        summarize_ensembles(args.root)
    elif args.command == "averages":
        evaluate_averages(args.root)
    elif args.command == "blend":
        fixed_blend(args.root)
    elif args.command == "evaluate":
        evaluate_wave(args.root, args.wave, SCREEN_FOLDS)
    else:
        summarize(args.root, args.wave, SCREEN_FOLDS)


if __name__ == "__main__":
    main()
