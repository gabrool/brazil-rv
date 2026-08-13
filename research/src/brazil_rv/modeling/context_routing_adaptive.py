from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contract import ALLOWED_SEEDS, HORIZONS


MANDATORY_ROUTING_RUN_COUNT = 4
CONDITIONAL_ROUTING_RUN_COUNT_MAXIMUM = 5
ROUTING_RUN_COUNT_MINIMUM = MANDATORY_ROUTING_RUN_COUNT
ROUTING_RUN_COUNT_MAXIMUM = (
    MANDATORY_ROUTING_RUN_COUNT + CONDITIONAL_ROUTING_RUN_COUNT_MAXIMUM
)
ISSUER_RUN_COUNT_MINIMUM = 1
ISSUER_RUN_COUNT_MAXIMUM = 3
TOTAL_TRAINING_RUN_COUNT_MINIMUM = ROUTING_RUN_COUNT_MINIMUM + ISSUER_RUN_COUNT_MINIMUM
TOTAL_TRAINING_RUN_COUNT_MAXIMUM = ROUTING_RUN_COUNT_MAXIMUM + ISSUER_RUN_COUNT_MAXIMUM


def metric_delta(
    treatment: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    treatment_horizons = treatment["horizons"]
    control_horizons = control["horizons"]
    if not isinstance(treatment_horizons, Mapping) or not isinstance(
        control_horizons, Mapping
    ):
        raise ValueError("Validation horizon metrics are malformed")
    horizons: dict[str, dict[str, float]] = {}
    for horizon in (f"{value}m" for value in HORIZONS):
        current = treatment_horizons[horizon]
        baseline = control_horizons[horizon]
        horizons[horizon] = {
            "control_ic": float(baseline["spearman_ic"]),
            "treatment_ic": float(current["spearman_ic"]),
            "delta_ic": float(current["spearman_ic"]) - float(baseline["spearman_ic"]),
            "control_gross_spread": float(baseline["gross_top_minus_bottom"]),
            "treatment_gross_spread": float(current["gross_top_minus_bottom"]),
            "delta_gross_spread": float(current["gross_top_minus_bottom"])
            - float(baseline["gross_top_minus_bottom"]),
            "control_turnover": float(baseline["one_way_turnover"]),
            "treatment_turnover": float(current["one_way_turnover"]),
            "delta_turnover": float(current["one_way_turnover"])
            - float(baseline["one_way_turnover"]),
        }
    return {
        "control_primary_ic": float(control["primary_ic"]),
        "treatment_primary_ic": float(treatment["primary_ic"]),
        "delta_primary_ic": float(treatment["primary_ic"])
        - float(control["primary_ic"]),
        "control_mean_gross_spread": float(control["mean_gross_top_minus_bottom"]),
        "treatment_mean_gross_spread": float(treatment["mean_gross_top_minus_bottom"]),
        "delta_mean_gross_spread": float(treatment["mean_gross_top_minus_bottom"])
        - float(control["mean_gross_top_minus_bottom"]),
        "control_mean_turnover": float(control["mean_one_way_turnover"]),
        "treatment_mean_turnover": float(treatment["mean_one_way_turnover"]),
        "delta_mean_turnover": float(treatment["mean_one_way_turnover"])
        - float(control["mean_one_way_turnover"]),
        "horizons": horizons,
    }


def seed29_candidate_gate(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    delta = metric_delta(treatment, control)
    horizon_values = list(delta["horizons"].values())
    positive_horizons = sum(float(row["delta_ic"]) > 0.0 for row in horizon_values)
    deteriorated_spreads = sum(
        float(row["delta_gross_spread"]) < 0.0 for row in horizon_values
    )
    passed = (
        float(delta["delta_primary_ic"]) > 0.0
        and positive_horizons >= 2
        and deteriorated_spreads <= 1
    )
    return {
        "stage": "seed29_screen",
        "passed": passed,
        "criteria": {
            "positive_paired_primary_ic_delta": float(delta["delta_primary_ic"]) > 0.0,
            "positive_horizon_ic_delta_count": positive_horizons,
            "minimum_positive_horizon_count": 2,
            "gross_spread_deterioration_count": deteriorated_spreads,
            "maximum_gross_spread_deterioration_count": 1,
        },
        "paired_metrics": delta,
        "transaction_cost_modeling": False,
    }


def three_seed_candidate_gate(
    controls: Mapping[int, Mapping[str, Any]],
    treatments: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        tuple(sorted(controls)) != ALLOWED_SEEDS
        or tuple(sorted(treatments)) != ALLOWED_SEEDS
    ):
        raise ValueError("Confirmation requires matched seeds 11, 29, and 47")
    paired = {
        seed: metric_delta(treatments[seed], controls[seed]) for seed in ALLOWED_SEEDS
    }
    primary = [float(paired[seed]["delta_primary_ic"]) for seed in ALLOWED_SEEDS]
    horizon_means = {
        f"{horizon}m": sum(
            float(paired[seed]["horizons"][f"{horizon}m"]["delta_ic"])
            for seed in ALLOWED_SEEDS
        )
        / len(ALLOWED_SEEDS)
        for horizon in HORIZONS
    }
    mean_primary = sum(primary) / len(primary)
    positive_seed_count = sum(value > 0.0 for value in primary)
    positive_horizon_count = sum(value > 0.0 for value in horizon_means.values())
    return {
        "stage": "three_seed_confirmation",
        "passed": (
            mean_primary > 0.0
            and positive_seed_count >= 2
            and positive_horizon_count >= 2
        ),
        "criteria": {
            "mean_paired_primary_effect": mean_primary,
            "positive_paired_primary_seed_count": positive_seed_count,
            "minimum_positive_seed_count": 2,
            "mean_horizon_effects": horizon_means,
            "positive_mean_horizon_effect_count": positive_horizon_count,
            "minimum_positive_mean_horizon_count": 2,
        },
        "paired_by_seed": {str(seed): paired[seed] for seed in ALLOWED_SEEDS},
        "transaction_cost_modeling": False,
    }


def within_source_combination_gate(
    early_gate: Mapping[str, Any], film_gate: Mapping[str, Any]
) -> dict[str, Any]:
    early_passed = early_gate.get("passed") is True
    film_passed = film_gate.get("passed") is True
    should_run = early_passed and film_passed
    return {
        "decision": "run" if should_run else "skip",
        "should_run": should_run,
        "reason": (
            "both_individual_seed29_routes_eligible"
            if should_run
            else "requires_both_individual_seed29_routes_to_be_eligible"
        ),
        "inputs": {
            "early_concat_passed": early_passed,
            "film_passed": film_passed,
            "early_concat_paired_metrics": early_gate.get("paired_metrics"),
            "film_paired_metrics": film_gate.get("paired_metrics"),
        },
    }


def joint_synthesis_gate(
    slow_selection: Mapping[str, Any], macro_selection: Mapping[str, Any]
) -> dict[str, Any]:
    slow = slow_selection.get("selected")
    macro = macro_selection.get("selected")
    should_run = isinstance(slow, Mapping) and isinstance(macro, Mapping)
    return {
        "decision": "run" if should_run else "skip",
        "should_run": should_run,
        "reason": (
            "both_sources_have_an_eligible_selected_method"
            if should_run
            else "joint_requires_an_eligible_selected_method_for_each_source"
        ),
        "selected_slow": slow,
        "selected_macro_temporal": macro,
    }


def _active_route_count(candidate: Mapping[str, Any]) -> int:
    count = 0
    for route in (candidate["slow_routing"], candidate["macro_temporal_routing"]):
        count += int(route in ("early_concat", "early_concat_film"))
        count += int(route in ("film", "early_concat_film"))
    return count


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    identities = [str(candidate["identity"]) for candidate in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("Candidate identities must be unique")
    eligible = [candidate for candidate in candidates if candidate["gate"]["passed"]]

    def key(candidate: Mapping[str, Any]) -> tuple[float, int, float, float, int, str]:
        gate = candidate["gate"]
        metrics = gate["paired_metrics"]
        return (
            -float(metrics["delta_primary_ic"]),
            -int(gate["criteria"]["positive_horizon_ic_delta_count"]),
            -float(metrics["delta_mean_gross_spread"]),
            float(metrics["delta_mean_turnover"]),
            _active_route_count(candidate),
            str(candidate["identity"]),
        )

    ranked = sorted(eligible, key=key)
    ranking = [
        {
            "rank": position,
            "identity": candidate["identity"],
            "slow_routing": candidate["slow_routing"],
            "macro_temporal_routing": candidate["macro_temporal_routing"],
            "tiebreak_values": {
                "delta_primary_ic": candidate["gate"]["paired_metrics"][
                    "delta_primary_ic"
                ],
                "positive_horizon_ic_delta_count": candidate["gate"]["criteria"][
                    "positive_horizon_ic_delta_count"
                ],
                "delta_mean_gross_spread": candidate["gate"]["paired_metrics"][
                    "delta_mean_gross_spread"
                ],
                "delta_mean_turnover": candidate["gate"]["paired_metrics"][
                    "delta_mean_turnover"
                ],
                "active_route_count": _active_route_count(candidate),
                "lexical_identity": candidate["identity"],
            },
        }
        for position, candidate in enumerate(ranked, start=1)
    ]
    selected = None
    if ranked:
        winner = ranked[0]
        selected = {
            "identity": winner["identity"],
            "slow_routing": winner["slow_routing"],
            "macro_temporal_routing": winner["macro_temporal_routing"],
            "gate": winner["gate"],
        }
    return {
        "selected": selected,
        "eligible_candidate_count": len(ranked),
        "ineligible_candidates": [
            {
                "identity": candidate["identity"],
                "reason": "seed29_gate_failed",
                "gate": candidate["gate"],
            }
            for candidate in candidates
            if not candidate["gate"]["passed"]
        ],
        "ranking": ranking,
        "deterministic_tiebreak": [
            "descending_delta_primary_ic",
            "descending_positive_horizon_ic_delta_count",
            "descending_delta_mean_gross_spread",
            "ascending_delta_mean_turnover",
            "ascending_active_route_count",
            "ascending_lexical_identity",
        ],
    }
