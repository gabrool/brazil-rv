"""CPU R3.1: freeze first, replay immutable score panels, seal all readouts."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from brazil_rv.execution.stateful_ledger import (
    ledger_configurations,
    simulate_stateful_ledger,
)
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.config import PROJECT_ROOT
from brazil_rv.v2.evaluate import (
    _aligned_action_terms,
    _input_hashes,
    _ledger_inputs,
    _ledger_rows,
    _realized_beta_diagnostic,
)
from brazil_rv.v2.execution_policy import (
    ExecutionPolicy,
    original_trade_attribution,
    policy_grid,
    prior_median_volume,
    traded_readouts,
    traded_signal,
)
from brazil_rv.v2.research_rounds import (
    RESEARCH_FLAGS,
    ROUND1_SCHEMA,
    ROUND2_SCHEMA,
    _evaluation_from_artifacts,
    _folded_bootstrap,
    _git_identity,
    _open_ledger_replay,
    _read_json,
    _utc_now,
    _verify_sealed_root,
    seal_root,
)

REGISTRATION = PROJECT_ROOT / "research/preregistrations/v2_round3.md"
SCENARIOS = (
    "borrow_balance",
    "borrow_strict",
    "borrow_open",
    "comparator_sterile_proceeds",
)
CANDIDATES = ("arm_B", "ensemble", "gbdt", "momentum")
FOLDS = ("F1", "F2", "F3")


def freeze(*, output: Path, round1: Path, round2: Path) -> None:
    git = _git_identity()
    sources = {}
    for name, root, schema in (
        ("round1", round1, ROUND1_SCHEMA),
        ("round2", round2, ROUND2_SCHEMA),
    ):
        root = root.resolve(strict=True)
        _verify_sealed_root(root, expected_schema=schema)
        sources[name] = {
            "root": str(root),
            **{
                filename: sha256_file(root / filename)
                for filename in (
                    "frozen_design.json",
                    f"{name}_result.json",
                    "artifact_inventory.json",
                )
            },
        }
    design1 = _read_json(round1 / "frozen_design.json")
    design2 = _read_json(round2 / "frozen_design.json")
    if design1["store"] != design2["store"]:
        raise ValueError("execution sweep requires the same source store")
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        output / "frozen_design.json",
        {
            "schema": "BRAZIL_RV_V2_EXECUTION_SWEEP_V1",
            "status": "frozen_before_policy_results",
            **RESEARCH_FLAGS,
            "git": git,
            "created_at_utc": _utc_now(),
            "sources": sources,
            "preregistration": {
                "path": str(REGISTRATION),
                "sha256": sha256_file(REGISTRATION),
            },
            "grid": [asdict(policy) for policy in policy_grid()],
            "buffer_followups": [3, 9],
            "candidates": list(CANDIDATES),
            "scenarios": list(SCENARIOS),
            "selection": {
                "candidate": "arm_B",
                "paired_point_above_bps": 1.0,
                "paired_lower_95_above_bps": -1.0,
                "traded_ic_tolerance": 0.003,
                "ic_reference": "sealed_arm_B_pooled_raw_neutral_IC",
                "label": "execution_parameter_selected_in_sample",
            },
            "raw_score_or_model_recomputed": False,
        },
    )


def _source_path(sources, candidate, fold):
    if candidate == "momentum":
        return (
            Path(sources["round1"]["root"])
            / "baselines/momentum_12_1"
            / fold
            / "evaluation.json"
        )
    branch = "aggregates" if candidate == "arm_B" else "comparators"
    return (
        Path(sources["round2"]["root"]) / branch / candidate / fold / "evaluation.json"
    )


def _failures(summary, *, headline):
    failed = [key for key, count in summary["entry_defect_signatures"].items() if count]
    if headline:
        if not 1.5 <= summary["mean_gross_fraction_nav"] <= 2.25:
            failed.append("mean_gross_outside_1.5_to_2.25")
        if summary["mean_unresolved_stale_inventory_fraction_nav"] >= 0.02:
            failed.append("mean_unresolved_stale_inventory_at_least_0.02")
        if summary["insolvent"]:
            failed.append("insolvent")
        for side in ("long", "short"):
            if summary[f"mean_absolute_volatility_occupancy_deviation_{side}"] > 2:
                failed.append(f"mean_quintile_occupancy_deviation_{side}_above_two")
    return failed


def _finite_list(array):
    return [float(value) if np.isfinite(value) else None for value in array]


def _panel(inputs, original_report, policy, median_volume):
    scores, mask = traded_signal(inputs, policy)
    readouts = traded_readouts(inputs, scores, mask)
    result = {
        "policy": asdict(policy),
        "traded_signal_daily": {
            key: _finite_list(value) for key, value in readouts.items()
        },
        "scenarios": {},
        "failed_gates": [],
    }
    configurations = ledger_configurations()
    for scenario in SCENARIOS:
        config = replace(
            configurations[scenario], buffer_per_side=5 * policy.buffer_per_quintile
        )
        ledger = simulate_stateful_ledger(
            **_ledger_inputs(inputs, scores, mask),
            config=config,
            shortable=inputs.shortable_by_borrow_source[config.borrow_source],
            entry_sizing_volatility=inputs.target_scale_sigma
            if policy.inverse_volatility
            else None,
            capacity_buffer_per_side=30,
        )
        summary = ledger.summary()
        daily = _ledger_rows(
            ledger,
            cost_bps=config.cost_bps_per_side,
            annual_borrow_rate=config.annual_borrow_rate,
        )
        result["scenarios"][scenario] = {
            "summary": summary,
            "daily": daily,
            "realized_beta_bova11": _realized_beta_diagnostic(
                inputs, ledger, against_bova11=True
            ),
            "realized_beta_universe": _realized_beta_diagnostic(inputs, ledger),
            "quota_changed_from_six": bool(np.any(ledger.volatility_quota != 6)),
            "quota_exception_dates": [
                inputs.dates[day].isoformat()
                for day in np.flatnonzero((ledger.volatility_quota != 6).any(axis=1))
            ],
        }
        if policy == ExecutionPolicy():
            old = [
                row
                for row in original_report["economics"]["daily_table"]
                if row["scenario"] == scenario
            ]
            # The canonical row serializer has no scenario key; source evaluation adds it.
            if [
                {key: value for key, value in row.items() if key != "scenario"}
                for row in old
            ] != daily:
                result["failed_gates"].append(
                    f"{scenario}:baseline_daily_not_bit_identical"
                )
        failed = _failures(summary, headline=scenario == "borrow_balance")
        result["failed_gates"].extend(f"{scenario}:{name}" for name in failed)
        if scenario == "borrow_balance":
            result["liquidity_attribution"] = original_trade_attribution(
                ledger,
                entry_liquid=median_volume >= 20e6,
                entry_known=np.isfinite(median_volume),
                action_terms=_aligned_action_terms(inputs),
                annual_borrow_rate_by_name=inputs.annual_borrow_rate_by_name,
                config=config,
            )
        if failed:
            break
    return result


def _daily(panel, scenario="borrow_balance"):
    book = panel["scenarios"][scenario]
    values = np.asarray(
        [row["net_excess_all_cash_bps"] for row in book["daily"]], dtype=float
    )
    if book["summary"]["economics_unresolved"]:
        values[:] = np.nan
    return values


def pooled(panels, baseline):
    return {
        "traded_signal": {
            key: _folded_bootstrap(
                [
                    np.asarray(panel["traded_signal_daily"][key], dtype=float)
                    for panel in panels
                ]
            )
            for key in panels[0]["traded_signal_daily"]
        },
        "net_excess": {
            scenario: _folded_bootstrap([_daily(panel, scenario) for panel in panels])
            for scenario in SCENARIOS
        },
        "paired_net_excess": _folded_bootstrap(
            [
                _daily(panel) - _daily(base)
                for panel, base in zip(panels, baseline, strict=True)
            ]
        ),
        "quota_changed_from_six": any(
            panel["scenarios"]["borrow_balance"]["quota_changed_from_six"]
            for panel in panels
        ),
        "liquid_contribution": _folded_bootstrap(
            [
                np.asarray(
                    panel["liquidity_attribution"]["daily"][
                        "net_excess_contribution_bps"
                    ],
                    dtype=float,
                )
                if not panel["scenarios"]["borrow_balance"]["summary"][
                    "economics_unresolved"
                ]
                else np.full(
                    len(
                        panel["liquidity_attribution"]["daily"][
                            "net_excess_contribution_bps"
                        ]
                    ),
                    np.nan,
                )
                for panel in panels
            ]
        ),
    }


def choose_policy(policies, readouts, raw_ic):
    qualified = []
    for policy in policies:
        if policy == ExecutionPolicy():
            continue
        cell = readouts[policy.key]["arm_B"]
        paired = cell["paired_net_excess"]
        ic = cell["traded_signal"]["neutral_ic"]["estimate"]
        if (
            paired["estimate"] is not None
            and paired["lower_95"] is not None
            and ic is not None
            and paired["estimate"] > 1
            and paired["lower_95"] > -1
            and ic >= raw_ic - 0.003
            and not cell["quota_changed_from_six"]
        ):
            qualified.append(policy)
    joint = [policy for policy in qualified if policy.changed_dimensions > 1]
    options = joint or qualified
    selected = (
        max(
            options,
            key=lambda policy: readouts[policy.key]["arm_B"]["paired_net_excess"][
                "estimate"
            ],
        )
        if options
        else ExecutionPolicy()
    )
    return {
        "selected_policy": asdict(selected),
        "selected_key": selected.key,
        "qualified_keys": [policy.key for policy in qualified],
        "raw_arm_B_ic_reference": raw_ic,
        "label": "execution_parameter_selected_in_sample"
        if options
        else "current_policy_retained",
    }


def run(*, output: Path):
    design = _read_json(output / "frozen_design.json")
    if (
        _git_identity() != design["git"]
        or sha256_file(REGISTRATION) != design["preregistration"]["sha256"]
    ):
        raise ValueError(
            "execution sweep implementation or registration differs from freeze"
        )
    for name, source in design["sources"].items():
        root = Path(source["root"])
        _verify_sealed_root(
            root, expected_schema=ROUND1_SCHEMA if name == "round1" else ROUND2_SCHEMA
        )
        for filename, digest in source.items():
            if filename != "root" and sha256_file(root / filename) != digest:
                raise ValueError(f"source changed: {root / filename}")
    context = _open_ledger_replay(
        _read_json(Path(design["sources"]["round2"]["root"]) / "frozen_design.json")
    )
    retained, identities, volumes = {}, {}, {}
    for fold in FOLDS:
        indices = context.evaluation[fold]
        first = int(indices[0]) - 20
        history = np.arange(first, int(indices[-1]) + 1)
        volumes[fold] = prior_median_volume(
            context.store.read("volume_brl", history),
            context.store.read("activity_valid", history),
            indices - first,
        )
        for candidate in CANDIDATES:
            path = _source_path(design["sources"], candidate, fold)
            item = _evaluation_from_artifacts(
                path,
                store=context.store,
                indices=indices,
                cdi=context.cdi,
                bova11_close_by_index=context.bova11.close_by_session,
                bova11_binding=context.bova11_binding,
                lending_borrow=context.lending_borrow,
                expected_fold=fold,
            )
            retained[candidate, fold] = item
            identities[candidate, fold] = _input_hashes(item.inputs)
    panels, readouts = {}, {}
    policies = list(policy_grid())
    for cell_index in range(20):
        if cell_index == 18:
            best = max(
                policies,
                key=lambda policy: (
                    readouts[policy.key]["arm_B"]["paired_net_excess"]["estimate"]
                    if readouts[policy.key]["arm_B"]["paired_net_excess"]["estimate"]
                    is not None
                    else -np.inf
                ),
            )
            policies.extend(
                replace(best, buffer_per_quintile=buffer) for buffer in (3, 9)
            )
        policy = policies[cell_index]
        readouts[policy.key] = {}
        for candidate in CANDIDATES:
            folded = []
            for fold in FOLDS:
                print(
                    f"R3.1 {cell_index + 1}/20 {policy.key} {candidate} {fold}",
                    flush=True,
                )
                item = retained[candidate, fold]
                panel = _panel(item.inputs, item.result.report, policy, volumes[fold])
                source_path = _source_path(design["sources"], candidate, fold)
                panel.update(
                    {
                        "candidate": candidate,
                        "fold": fold,
                        **RESEARCH_FLAGS,
                        "source_evaluation": {
                            "path": str(source_path),
                            "sha256": sha256_file(source_path),
                        },
                        "raw_input_hashes": identities[candidate, fold],
                        "raw_primary_ic": item.result.report[
                            "mean_daily_primary_neutral_target_ic"
                        ],
                    }
                )
                if _input_hashes(item.inputs) != identities[candidate, fold]:
                    panel["failed_gates"].append("protected_input_mutation")
                path = output / "panels" / policy.key / candidate / f"{fold}.json"
                write_json_atomic(path, panel)
                if panel["failed_gates"]:
                    raise RuntimeError(
                        f"registered R3.1 stop: {path}: {panel['failed_gates']}"
                    )
                folded.append(panel)
            panels[policy.key, candidate] = folded
            readouts[policy.key][candidate] = pooled(
                folded, panels[ExecutionPolicy().key, candidate]
            )
        write_json_atomic(
            output / "pooled" / f"{policy.key}.json", readouts[policy.key]
        )
    raw_ic = float(
        np.nanmean(
            np.concatenate(
                [retained["arm_B", fold].result.daily_primary_ic for fold in FOLDS]
            )
        )
    )
    decision = choose_policy(policies, readouts, raw_ic)
    result = {
        "schema": "BRAZIL_RV_V2_EXECUTION_SWEEP_V1",
        "status": "complete",
        **RESEARCH_FLAGS,
        "frozen_design_sha256": sha256_file(output / "frozen_design.json"),
        "decision": decision,
        "cells": readouts,
        "policies": [asdict(policy) for policy in policies],
        "raw_scores_models_recomputed": False,
        "protected_inputs_exact": True,
        "source_evaluations_unchanged": True,
        "store_access": context.access,
    }
    for name, source in design["sources"].items():
        _verify_sealed_root(
            Path(source["root"]),
            expected_schema=ROUND1_SCHEMA if name == "round1" else ROUND2_SCHEMA,
        )
    write_json_atomic(output / "execution_sweep_result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round1", type=Path)
    parser.add_argument("--round2", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(output=args.output, round1=args.round1, round2=args.round2)
    else:
        if (args.output / "artifact_inventory.json").exists():
            raise FileExistsError("sealed execution roots are immutable")
        try:
            run(output=args.output)
        except Exception as error:
            write_json_atomic(
                args.output / "failure.json",
                {
                    "status": "stopped",
                    **RESEARCH_FLAGS,
                    "research_claim": False,
                    "error": str(error),
                    "created_at_utc": _utc_now(),
                },
            )
            seal_root(root=args.output, research_claim=False)
            raise
        seal_root(root=args.output)


if __name__ == "__main__":
    main()
