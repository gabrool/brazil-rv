"""Round-6 promotion rules and only the paired comparisons those rules require."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import paired_readouts
from .contract import ALLOWED_SEEDS, CONFIRMATION_SEEDS, DEVELOPMENT_FOLDS
from .research_checkpoint import _completed
from .round6 import SESSION1, SESSION1_FAMILIES, SESSION2, derive_session2_roster
from .round6_readouts import evaluation_design, informative_folds

IC = "primary_neutral_target_ic"
NET = "headline_net_excess_bps"


def eligible_arms(readouts: dict, excluded=()) -> list[str]:
    return [
        arm
        for arm, row in readouts.items()
        if arm not in excluded
        and all(
            row[key]["estimate"] is not None and math.isfinite(row[key]["estimate"])
            for key in (IC, NET)
        )
        and row[NET]["estimate"] >= 0
    ]


def promotion_trace(
    readouts: dict, pairs: dict, *, excluded=(), confirmation_complete=False
) -> dict:
    """All-fold leader; informative comparison determines the confirmation trigger."""
    eligible = eligible_arms(readouts, excluded)
    leader = max(eligible, key=lambda a: readouts[a][IC]["estimate"], default=None)
    qualifiers = []
    for arm in eligible:
        if arm == leader:
            continue
        pair = pairs[f"{arm}_minus_{leader}"]["pooled"]
        if rr._interval_includes_zero(pair[IC]) and rr._interval_is_positive(pair[NET]):
            qualifiers.append(arm)
    override = max(qualifiers, key=lambda a: readouts[a][NET]["estimate"], default=None)
    candidate = override or leader
    informative = (
        pairs[f"{candidate}_minus_S0"]["informative_subsets"][candidate]
        if candidate not in (None, "S0")
        else None
    )
    lower = informative["pooled"][IC]["lower_95"] if informative else None
    close = lower is not None and abs(lower) <= 0.001
    reasons = [
        reason
        for reason, applies in (
            ("informative_paired_IC_lower_bound_within_0.001", close),
            ("economics_override", override is not None),
        )
        if applies
    ]
    # An override is a candidate-versus-leader claim. Retain that matched pair
    # as well as the candidate-versus-S0 comparison required by the registration.
    needed = (
        {"S0", candidate, leader if override else candidate}
        if reasons and not confirmation_complete
        else set()
    )
    return {
        "eligible": eligible,
        "excluded_arms": list(excluded),
        "ic_leader": leader,
        "economics_override": override,
        "economics_override_qualifiers": qualifiers,
        "provisional_designation": candidate,
        "informative_comparison_to_S0": informative,
        "confirmation_reasons": reasons,
        "confirmation_arms": [a for a in readouts if a in needed],
        "confirmation_seeds": list(CONFIRMATION_SEEDS) if needed else [],
        "confirmation_complete": confirmation_complete,
        "tie_rule": "registration_order",
        "economics_basis": "original_rates_A1_full_calendar_fold_reset",
        "sensitivity_cost_promotion_weight": 0,
        "persistence_and_turnover": {
            a: {k: v for k, v in row.items() if "persistence" in k or "turnover" in k}
            for a, row in readouts.items()
        },
        "persistence_turnover_rule": "descriptive_tradeoff_review; no_unregistered_numeric_threshold",
        "read_2025_authorized": False,
    }


def seed_stability(traces: dict, seeds=ALLOWED_SEEDS) -> dict:
    panels = [traces["full"], *(traces[f"omit_{s}"] for s in seeds)]
    choices = [p["provisional_designation"] for p in panels]
    stable = choices[0] is not None and len(set(choices)) == 1
    return {
        "panel_order": ["full", *(f"omit_{s}" for s in seeds)],
        "provisional_designations": choices,
        "designation_stable": stable,
        "research_designation": choices[0] if stable else None,
        "working_research_comparator": choices[0] if stable else "S0",
        "comparator_economically_eligible_in_all_panels": all(
            (choices[0] if stable else "S0") in p["eligible"] for p in panels
        ),
        "parent_inconclusive": not stable,
        "confirmation_rules_still_apply": not traces["full"].get(
            "confirmation_complete", False
        ),
        "independent_replication": False,
    }


def roster_sensitivity(pairs: dict) -> dict:
    return derive_session2_roster(
        {
            arm: pairs[f"{arm}_minus_S0"]["informative_subsets"][arm]["pooled"][IC][
                "estimate"
            ]
            for arm in SESSION1_FAMILIES
        }
    )


def era_readout(paired_root: Path) -> dict:
    eras = {
        "2018_2019": (2018, 2019),
        "2020_2021": (2020, 2021),
        "2022_2024": (2022, 2024),
    }
    panels = {era: {} for era in eras}
    sources = {}
    for fold in DEVELOPMENT_FOLDS:
        path = paired_root / f"{fold}.json"
        sources[fold] = sha256_file(path)
        daily = rr._read_json(path)["population_audit"][fold]
        for era, (first, last) in eras.items():
            values = {
                metric: np.asarray(
                    [
                        row["delta"]
                        for row in rows
                        if first <= int(row["date"][:4]) <= last
                    ],
                    float,
                )
                for metric, rows in daily.items()
            }
            if len(values[IC]):
                panels[era][fold] = values
    return {
        "direction": "arm_minus_S0",
        "promotion_weight": 0,
        "source_paired_sha256": sources,
        "eras": {
            era: {
                "folds": list(folds),
                "paired": {
                    key: rr._folded_bootstrap(tuple(row[key] for row in folds.values()))
                    for key in next(iter(folds.values()))
                },
            }
            for era, folds in panels.items()
            if folds
        },
    }


def leader_comparisons(
    context, paths, readouts, pairs, output, support, *, excluded=()
):
    """Reuse parent comparisons; compute O(arms) additional exact-population pairs."""
    eligible = eligible_arms(readouts, excluded)
    leader = max(eligible, key=lambda a: readouts[a][IC]["estimate"], default=None)
    for arm in eligible:
        if arm == leader or f"{arm}_minus_{leader}" in pairs:
            continue
        pairs.update(
            paired_readouts(
                context,
                {arm: paths[arm], leader: paths[leader]},
                output,
                informative_folds=support,
            )
        )
    return pairs


def review(root: Path, output: Path, *, excluded=()) -> str:
    design = rr._read_json(root / "frozen_design.json")
    roster = rr._read_json(root / "session2_roster.json")
    order = (
        "S0",
        *SESSION1,
        *SESSION2,
        *(("C6",) if roster["c6_families"] else ()),
        "best_single_fresh_p",
        *(("C6_fresh_p",) if roster["c6_families"] else ()),
    )
    groups = {
        g: rr._read_json(root / f"{g}_result.json") for g in ("session1", "session2")
    }
    readouts, paths, pairs = {}, {}, {}
    for group, result in groups.items():
        if (
            result["status"] != "completed"
            or result["seeds"] != list(ALLOWED_SEEDS)
            or result["frozen_design_sha256"]
            != sha256_file(root / "frozen_design.json")
        ):
            raise ValueError("decision requires completed matched registered panels")
        pairs.update(result["paired"])
        for arm, row in result["readouts"].items():
            if arm in readouts and readouts[arm] != row["pooled"]:
                raise ValueError("repeated baseline readouts differ")
            readouts[arm] = row["pooled"]
            paths[arm] = {
                f: root / "aggregates" / group / arm / f for f in DEVELOPMENT_FOLDS
            }
            if not all(_completed(p) for p in paths[arm].values()):
                raise ValueError("full-panel decision requires accepted books")
    if set(readouts) != set(order):
        raise ValueError(
            "complete both registered sessions before a promotion decision"
        )
    readouts = {a: readouts[a] for a in order}
    output.mkdir(parents=True, exist_ok=False)
    context = rr._open_ledger_replay(evaluation_design(design))
    try:
        support = {a: informative_folds(design, a, roster) for a in order}
        pairs = leader_comparisons(
            context,
            paths,
            readouts,
            pairs,
            output / "paired",
            support,
            excluded=excluded,
        )
        return write_json_atomic(
            output / "result.json",
            {
                "status": "provisional_requires_fixed_seed_audit",
                "seeds": list(ALLOWED_SEEDS),
                "implementation": rr._git_identity(),
                "source_results": {
                    g: sha256_file(root / f"{g}_result.json") for g in groups
                },
                "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
                "readouts": readouts,
                "paired": pairs,
                "promotion_trace": promotion_trace(readouts, pairs, excluded=excluded),
                "registered_C6_roster": roster,
                "era_diagnostics": {
                    arm: era_readout(root / "paired/session1" / f"{arm}_minus_S0")
                    for arm in ("finetune_lr_1", "time_decay_756")
                },
                "temporal_modelling_premium": {
                    "direction": "S0_minus_mlp",
                    "comparison": pairs["S0_minus_mlp"],
                    "promotion_weight": 0,
                },
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def confirmation_review(root: Path, output: Path, decision_path: Path) -> str:
    """Reapply the registered rule only to the roster that triggered confirmation."""
    design_hash = sha256_file(root / "frozen_design.json")
    prior = rr._read_json(decision_path)
    trace = prior["decision_traces"]["full"]
    arms = trace["confirmation_arms"]
    result = rr._read_json(root / "confirmation_result.json")
    if (
        prior["status"] != "completed"
        or prior["frozen_design_sha256"] != design_hash
        or not trace["confirmation_reasons"]
        or trace["confirmation_seeds"] != list(CONFIRMATION_SEEDS)
        or result["status"] != "completed"
        or result["frozen_design_sha256"] != design_hash
        or result["seeds"] != [*ALLOWED_SEEDS, *CONFIRMATION_SEEDS]
        or set(result["readouts"]) != set(arms)
    ):
        raise ValueError(
            "confirmation review requires the triggered matched six-seed panel"
        )
    design = rr._read_json(root / "frozen_design.json")
    roster = rr._read_json(root / "session2_roster.json")
    readouts = {a: result["readouts"][a]["pooled"] for a in arms}
    paths = {
        a: {f: root / "aggregates/confirmation" / a / f for f in DEVELOPMENT_FOLDS}
        for a in arms
    }
    if not all(_completed(p) for folds in paths.values() for p in folds.values()):
        raise ValueError("confirmation review requires all accepted books")
    output.mkdir(parents=True, exist_ok=False)
    context = rr._open_ledger_replay(evaluation_design(design))
    try:
        pairs = leader_comparisons(
            context,
            paths,
            readouts,
            dict(result["paired"]),
            output / "paired",
            {a: informative_folds(design, a, roster) for a in arms},
        )
        return write_json_atomic(
            output / "result.json",
            {
                "status": "provisional_requires_fixed_seed_audit",
                "implementation": rr._git_identity(),
                "frozen_design_sha256": design_hash,
                "seeds": result["seeds"],
                "aggregate_group": "confirmation",
                "confirmation_complete": True,
                "screening_decision_sha256": sha256_file(decision_path),
                "confirmation_result_sha256": sha256_file(
                    root / "confirmation_result.json"
                ),
                "readouts": readouts,
                "paired": pairs,
                "promotion_trace": promotion_trace(
                    readouts, pairs, confirmation_complete=True
                ),
                "registered_C6_roster": roster,
                "roster_changes_applied": False,
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded", nargs="*", default=[])
    parser.add_argument("--confirmation-decision", type=Path)
    args = parser.parse_args()
    print(
        confirmation_review(args.root, args.output, args.confirmation_decision)
        if args.confirmation_decision
        else review(args.root, args.output, excluded=args.excluded)
    )


if __name__ == "__main__":
    main()
