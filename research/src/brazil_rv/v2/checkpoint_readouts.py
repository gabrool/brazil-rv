"""CPU readouts of checkpoint and Round-4 score artifacts, bounded one fold at a time."""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import inventory, sha256_file, write_json_atomic
from .contract import DEVELOPMENT_FOLDS, HORIZONS
from .research_checkpoint import CELLS
from .research_diagnostics import momentum_diagnostics, pooled_momentum_diagnostics


def candidate_readout(paths: dict[str, Path]) -> dict:
    series = {}
    economics_coverage = {}
    for fold, path in paths.items():
        saved = rr._read_json(path / "daily_readouts.json")
        report = rr._read_json(path / "evaluation.json")
        summary = report["economics"]["headline"]
        economics_coverage[fold] = {
            "economics_unresolved": summary["economics_unresolved"],
            "sessions": len(saved["dates"]),
            "terminal_unresolved_inventory_fraction_nav": summary[
                "terminal_unresolved_inventory_fraction_nav"
            ],
            "terminal_hedge_signed_notional": summary["terminal_hedge_signed_notional"],
            "unresolved_receivable": summary["unresolved_receivable"],
            "unresolved_payable": summary["unresolved_payable"],
            "terminal_no_print_dominates": summary["terminal_no_print_dominates"],
            "terminal_settlement_economics_unresolved": summary[
                "terminal_settlement_economics_unresolved"
            ],
            "terminal_settlement_notional_fraction_nav": summary[
                "terminal_settlement_notional_fraction_nav"
            ],
            "terminal_settlement_haircut_delta_nav": summary[
                "terminal_nav_settlement_haircut_scenario"
            ]
            - summary["terminal_nav"],
            "terminal_hedge_last_mark_settlement_notional": summary.get(
                "terminal_hedge_last_mark_settlement_notional", 0.0
            ),
        }
        fields = saved["series"]
        fields["resolved_fold_only_net_excess_bps"] = (
            [None] * len(saved["dates"])
            if summary["economics_unresolved"]
            else list(fields["headline_net_excess_bps"])
        )
        # These are descriptive projections of the retained report, never a replay
        # or a mutation of an already accepted CPU cell.
        headline = {
            row["date"]: row
            for row in report["economics"]["daily_table"]
            if row["scenario"] == "borrow_balance"
        }
        fields["turnover_fraction_nav"] = [
            headline[d]["turnover_fraction_nav"] for d in saved["dates"]
        ]
        for horizon in HORIZONS:
            rows = {
                row["date"]: row
                for row in report["daily_metric_table"]
                if row["horizon_sessions"] == horizon
            }
            fields[f"D{horizon}_neutral_ic"] = [
                rows[d]["neutral_target_spearman_ic"] for d in saved["dates"]
            ]
        series[fold] = fields
    values = {
        fold: {
            key: np.asarray([np.nan if x is None else x for x in row], dtype=np.float64)
            for key, row in fields.items()
        }
        for fold, fields in series.items()
    }
    labels = tuple(next(iter(values.values())))
    return {
        "economics_coverage": {
            "by_fold": economics_coverage,
            "unresolved_folds": [
                f
                for f, row in economics_coverage.items()
                if row["economics_unresolved"]
            ],
            "pooling_rule": report["economics"]["contract"].get(
                "economics_pooling", "resolved_folds_only"
            ),
            "interpretation": "inspect_terminal_settlement_and_unresolved_labels; resolved_fold_only_net_excess_bps_retains_secondary_pool",
        },
        "turnover_note": "includes_initial_and_terminal_book_trades; unchanged_non_circular_block_intervals_underweight_boundary_spikes",
        "pooled": {
            key: rr._folded_bootstrap(tuple(row[key] for row in values.values()))
            for key in labels
        },
        "folds": {
            fold: {key: rr._folded_bootstrap((row[key],)) for key in labels}
            for fold, row in values.items()
        },
        "round3_windows": {
            key: rr._folded_bootstrap(
                tuple(values[f][key] for f in DEVELOPMENT_FOLDS[-3:] if f in values)
            )
            for key in labels
        }
        if any(f in values for f in DEVELOPMENT_FOLDS[-3:])
        else None,
    }


def retained(context, path: Path, fold: str):
    return rr._evaluation_from_artifacts(
        path / "evaluation.json",
        store=context.store,
        indices=context.evaluation[fold],
        cdi=context.cdi,
        bova11_close_by_index=context.bova11.close_by_session,
        bova11_binding=context.bova11_binding,
        lending_borrow=context.lending_borrow,
        expected_fold=fold,
    )


def paired_readouts(
    context,
    paths: dict[str, dict[str, Path]],
    output: Path,
    *,
    informative_folds: dict[str, list[str]] | None = None,
) -> dict:
    """Retain per-fold population audits and pool their paired daily deltas."""
    names = tuple(paths)
    daily, fold_results = {}, {}
    for left_name, right_name in combinations(names, 2):
        key = f"{left_name}_minus_{right_name}"
        daily[key], fold_results[key] = {}, {}
        common_folds = tuple(f for f in paths[left_name] if f in paths[right_name])
        for fold in common_folds:
            left = retained(context, paths[left_name][fold], fold)
            right = retained(context, paths[right_name][fold], fold)
            pair = rr._paired_readouts({fold: left}, {fold: right})
            write_json_atomic(output / key / f"{fold}.json", pair)
            daily[key][fold] = {
                metric: np.asarray(
                    [np.nan if row["delta"] is None else row["delta"] for row in rows],
                    dtype=np.float64,
                )
                for metric, rows in pair["population_audit"][fold].items()
            }
            fold_results[key][fold] = pair["folds"][fold]
            del left, right
    result = {}
    for key, by_fold in daily.items():
        metrics = tuple(next(iter(by_fold.values())))
        pooled = {
            metric: rr._folded_bootstrap(tuple(row[metric] for row in by_fold.values()))
            for metric in metrics
        }
        result[key] = {"pooled": pooled, "folds": fold_results[key]}
        left, right = key.split("_minus_")
        if informative_folds is not None:
            subsets = {}
            for arm in (left, right):
                selected = [f for f in informative_folds[arm] if f in by_fold]
                subsets[arm] = {
                    "folds": selected,
                    "pooled": {
                        metric: rr._folded_bootstrap(
                            tuple(by_fold[f][metric] for f in selected)
                        )
                        for metric in metrics
                    },
                }
            result[key]["informative_subsets"] = subsets
        reverse = {}
        for metric, row in pooled.items():
            reverse[metric] = {
                **row,
                "estimate": None if row["estimate"] is None else -row["estimate"],
                "lower_95": None if row["upper_95"] is None else -row["upper_95"],
                "upper_95": None if row["lower_95"] is None else -row["lower_95"],
            }
        result[f"{right}_minus_{left}"] = {"pooled": reverse, "opposite_of": key}
        if informative_folds is not None:
            result[f"{right}_minus_{left}"]["informative_subsets"] = {
                arm: {
                    "folds": subset["folds"],
                    "pooled": {
                        metric: {
                            **row,
                            "estimate": None
                            if row["estimate"] is None
                            else -row["estimate"],
                            "lower_95": None
                            if row["upper_95"] is None
                            else -row["upper_95"],
                            "upper_95": None
                            if row["lower_95"] is None
                            else -row["lower_95"],
                        }
                        for metric, row in subset["pooled"].items()
                    },
                }
                for arm, subset in subsets.items()
            }
    return result


def gbdt_diagnostics(root: Path) -> str:
    implementation = rr._git_identity()
    if (root / "checkpoint_diagnostics.json").exists():
        raise FileExistsError(root / "checkpoint_diagnostics.json")
    if (
        rr._read_json(root / "checkpoint_cpu_result.json")["status"]
        != "cpu_rebaseline_complete"
    ):
        raise ValueError("GBDT diagnostics require the completed CPU panel")
    design = rr._read_json(root / "frozen_design.json")
    context = rr._open_ledger_replay(design)
    try:
        paths = {
            name: {fold: root / "gbdt" / name / fold for fold in DEVELOPMENT_FOLDS}
            for name in CELLS
        }
        pairs = paired_readouts(context, paths, root / "diagnostics/gbdt_pairs")
        importance = {}
        for name in CELLS:
            if name == "a_slow":
                continue
            folds, samples, pooled, total_rows = {}, {}, {}, 0
            for fold in DEVELOPMENT_FOLDS:
                metadata = rr._read_json(paths[name][fold] / "score_manifest.json")[
                    "metadata"
                ]
                features = metadata["feature_names"]
                indices = context.evaluation[fold]
                active = context.store.read("active", indices)
                locations = np.argwhere(active)
                selected = locations[
                    np.linspace(
                        0, len(locations) - 1, min(4096, len(locations)), dtype=np.int64
                    )
                ]
                panel = rr._gbdt_features(context.store, indices, name)
                model = rr.MultiHorizonGBDT.load(
                    root / "gbdt" / name / "models" / fold,
                    expected_manifest_sha256=metadata["model"]["manifest_sha256"],
                )
                values = model.feature_importance(
                    panel[selected[:, 0], selected[:, 1]]
                )["mean_abs_tree_shap"]
                samples[fold] = {
                    "active_rows": len(locations),
                    "sample_rows": len(selected),
                    "sample_date_name_indices_sha256": hashlib.sha256(
                        selected.astype("<i8").tobytes()
                    ).hexdigest(),
                    "model_manifest_sha256": metadata["model"]["manifest_sha256"],
                }
                intraday = set(context.store.manifest["feature_names"]["intraday"])
                # Both value and age-channel attribution belong to their feature.
                scores = {
                    field: sum(
                        float(v)
                        for f, v in zip(features, values, strict=True)
                        if f in (field, field + "__age_sessions")
                    )
                    for field in intraday
                    if field in features
                }
                folds[fold] = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[
                    :15
                ]
                for feature, value in scores.items():
                    pooled[feature] = pooled.get(feature, 0.0) + value * len(selected)
                total_rows += len(selected)
                del model, panel
            importance[name] = {
                "population": "up_to_4096_active_evaluation_rows_per_fold_evenly_spaced_in_date_security_order",
                "selection_weight": 0,
                "method": "LightGBM_TreeSHAP_mean_absolute_across_heads_and_seeds_value_plus_age",
                "top15_by_fold": folds,
                "samples": samples,
                "top15_pooled": sorted(
                    (
                        (feature, value / total_rows)
                        for feature, value in pooled.items()
                    ),
                    key=lambda row: (-row[1], row[0]),
                )[:15],
                "gain_importance_retained_in_score_manifests": True,
            }
        return write_json_atomic(
            root / "checkpoint_diagnostics.json",
            {
                "status": "completed",
                "implementation": implementation,
                "cpu_result_sha256": sha256_file(root / "checkpoint_cpu_result.json"),
                "gbdt_pairs": pairs,
                "readouts": {
                    name: candidate_readout(
                        {
                            fold: root
                            / ("gbdt" if name in CELLS else "baselines")
                            / name
                            / fold
                            for fold in DEVELOPMENT_FOLDS
                        }
                    )
                    for name in (*rr._BASELINE_SIGNAL_NAMES, *CELLS)
                },
                "intraday_tree_shap": importance,
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def historical_b6_diagnostics(
    cpu_root: Path, b6_root: Path, output: Path, *, expected_inventory_sha256: str
) -> str:
    """Read B6's sealed three folds without refitting or overwriting its reports."""
    implementation = rr._git_identity()
    if output.exists():
        raise FileExistsError(output)
    sealed = b6_root / "artifact_inventory.json"
    if sha256_file(sealed) != expected_inventory_sha256:
        raise ValueError("B6 inventory differs from the accepted historical binding")
    artifact_inventory = rr._read_json(sealed)
    if (
        inventory(b6_root, exclude=set(artifact_inventory["excluded_self"]))
        != artifact_inventory["files"]
    ):
        raise ValueError("B6 sealed root differs from its inventory")
    context = rr._open_ledger_replay(rr._read_json(cpu_root / "frozen_design.json"))
    try:
        rows = {}
        hashes = {}
        for old, fold in zip(("F1", "F2", "F3"), DEVELOPMENT_FOLDS[-3:], strict=True):
            path = b6_root / "aggregates/B6" / old
            score, mask = rr._score_artifact(path, require_clean_transfer=True)
            metadata = rr._read_json(path / "score_manifest.json")["metadata"]
            if metadata["evaluation_date_indices"] != context.evaluation[fold].tolist():
                raise ValueError(
                    "B6 historical dates do not match the corresponding new fold"
                )
            momentum, momentum_mask = rr._score_artifact(
                cpu_root / "baselines/momentum_12_1" / fold, require_clean_transfer=True
            )
            inputs = rr._evaluation_inputs(
                context.store,
                context.evaluation[fold],
                score,
                mask,
                context.cdi,
                context.bova11.close_by_session,
                context.bova11_binding,
                context.lending_borrow,
                {
                    "v2_store_manifest": sha256_file(
                        context.store.root / "manifest.json"
                    )
                },
                transfer_chronology_clean=True,
            )
            rows[fold] = momentum_diagnostics(inputs, momentum, momentum_mask)
            hashes[old] = sha256_file(path / "score_manifest.json")
        return write_json_atomic(
            output,
            {
                "candidate": "historical_B6",
                "refitted": False,
                "fold_mapping": {"F1": "F12", "F2": "F13", "F3": "F14"},
                "folds": rows,
                "pooled": pooled_momentum_diagnostics(rows),
                "historical_inventory_sha256": expected_inventory_sha256,
                "historical_score_manifests": hashes,
                "implementation": implementation,
                "integrity_scan_disclosure": "historical_integrity_only_scans_of_later_old_store_rows_disclosed_in_Round3",
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gbdt", "b6", "seal"))
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--b6-root", type=Path)
    parser.add_argument("--b6-inventory-sha256")
    args = parser.parse_args(argv)
    if args.action == "gbdt":
        print(gbdt_diagnostics(args.cpu_root))
    elif args.action == "b6":
        if args.b6_root is None or args.b6_inventory_sha256 is None:
            parser.error("b6 requires its sealed root and accepted inventory hash")
        print(
            historical_b6_diagnostics(
                args.cpu_root,
                args.b6_root,
                args.cpu_root / "historical_b6_diagnostics.json",
                expected_inventory_sha256=args.b6_inventory_sha256,
            )
        )
    else:
        if (
            rr._read_json(args.cpu_root / "checkpoint_diagnostics.json")["status"]
            != "completed"
        ):
            raise ValueError("finish the CPU diagnostics before sealing")
        print(rr.seal_root(root=args.cpu_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
