from __future__ import annotations

from typing import Any

from .contract import HORIZONS


def _comparison_row(
    comparisons: list[dict[str, object]],
    name: str,
    horizon_minutes: int = 0,
) -> dict[str, object]:
    return next(
        row
        for row in comparisons
        if row["comparison"] == name and int(row["horizon_minutes"]) == horizon_minutes
    )


def _interval_result(row: dict[str, object]) -> str:
    if float(row["delta_lower_95"]) > 0.0:
        return "positive"
    if float(row["delta_upper_95"]) < 0.0:
        return "negative"
    return "inconclusive"


def _comparison_evidence(row: dict[str, object]) -> dict[str, object]:
    return {
        "delta_ic": row["delta_ic"],
        "delta_lower_95": row["delta_lower_95"],
        "delta_upper_95": row["delta_upper_95"],
        "interval_result": _interval_result(row),
    }


def build_context_training_summary(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    order = {minutes: index for index, minutes in enumerate((*HORIZONS, 0))}
    result: dict[str, object] = {}
    for family in ("wdo", "br_rates", "us_rates"):
        selected = sorted(
            (row for row in rows if row["context_family"] == family),
            key=lambda row: order[int(row["horizon_minutes"])],
        )
        result[family] = {
            "worst_horizon_delta": min(
                float(row["delta_ic"])
                for row in selected
                if int(row["horizon_minutes"]) in HORIZONS
            ),
            "results": selected,
        }
    return result


def stage_summary_markdown(summary: dict[str, Any]) -> str:
    hypotheses = summary["hypotheses"]
    trained = hypotheses["trained_multiscale_result"]
    lines = [
        "# Horizon multiscale stage summary",
        "",
        f"- Training runs: {summary['training_run_count']}",
        "- Held-out test accessed: no",
        f"- Representation: {summary['bottleneck_interpretation']}",
        f"- Trained multiscale supported: {trained['supported']}; "
        f"aggregate delta {float(trained['delta_ic']):.6f}.",
        f"- Shared aggregation delta (seed 29): "
        f"{float(hypotheses['shared_scale_aggregation']['delta_ic']):.6f}.",
        f"- Horizon specialization over shared (seed 29): "
        f"{float(hypotheses['horizon_scale_specialization']['delta_ic']):.6f}.",
        f"- Score capacity is a competing explanation: "
        f"{hypotheses['score_capacity_control']['competing_explanation']}.",
        "- Gradient conflict, single-horizon controls, context sources, and target "
        "structure remain separate hypotheses; see the artifact paths in "
        "stage_summary.json.",
        "- Architecture promotion: none.",
    ]
    return "\n".join(lines) + "\n"


def build_hypothesis_summary(
    comparisons: list[dict[str, object]],
    frozen_summary: dict[str, Any],
    gradient_summary: dict[str, Any],
    single_horizon_rows: list[dict[str, object]],
    context_summary: dict[str, Any],
    context_rows: list[dict[str, object]],
    oof_summary: dict[str, Any],
    target_summary: dict[str, Any],
) -> dict[str, object]:
    three_seed_rows = [
        row
        for row in comparisons
        if row["comparison"] == "horizon_multiscale_vs_final_three_seed"
    ]
    three_seed_aggregate = next(
        row for row in three_seed_rows if int(row["horizon_minutes"]) == 0
    )
    worst_horizon_delta = min(
        float(row["delta_ic"])
        for row in three_seed_rows
        if int(row["horizon_minutes"]) in HORIZONS
    )
    trained_supported = (
        float(three_seed_aggregate["delta_lower_95"]) > 0.0
        and worst_horizon_delta > 0.0
    )

    earlier = frozen_summary["earlier_tap_beats_final_post_fusion_by_horizon"]
    concatenated = frozen_summary["concatenated_beats_final_post_fusion_by_horizon"]
    probe_supported = any(
        bool(value) for value in (*earlier.values(), *concatenated.values())
    )
    if trained_supported:
        bottleneck_interpretation = (
            "The tested trained horizon-multiscale readout is supported by the "
            "paired three-seed result."
        )
    elif probe_supported:
        bottleneck_interpretation = (
            "Intermediate representations contain useful information diagnostically, "
            "but the selected global-mixture and training mechanism did not exploit "
            "it consistently."
        )
    else:
        bottleneck_interpretation = (
            "Neither frozen representation probes nor the trained multiscale result "
            "support a final-state information bottleneck."
        )

    shared = _comparison_row(comparisons, "shared_multiscale_vs_final_seed29")
    specialized = _comparison_row(comparisons, "horizon_multiscale_vs_shared_seed29")
    score = _comparison_row(comparisons, "final_score_mlp_vs_final_seed29")
    trained_seed29 = _comparison_row(comparisons, "horizon_multiscale_vs_final_seed29")
    score_competing = (
        float(trained_seed29["delta_ic"]) > 0.0
        and float(score["delta_ic"]) > 0.0
        and float(score["delta_ic"]) >= float(trained_seed29["delta_ic"])
    )

    single_deltas = {
        str(row["training_horizon"]): float(row["delta_from_control"])
        for row in single_horizon_rows
    }
    gradient_rows = gradient_summary["by_group_and_horizon_pair"]
    negative_gradient_cells = sum(
        row.get("fraction_negative") is not None
        and float(row["fraction_negative"]) > 0.0
        for row in gradient_rows
    )
    undefined_gradient_cells = sum(
        row.get("valid_samples") == 0 or row.get("fraction_negative") is None
        for row in gradient_rows
    )
    context_aggregate = {
        str(row["context_family"]): _comparison_evidence(row)
        for row in context_rows
        if int(row["horizon_minutes"]) == 0
    }
    positive_oof_cells = [
        {
            "probe": row["probe"],
            "horizon_minutes": row["horizon_minutes"],
            "delta_from_base": row["delta_from_base"],
        }
        for row in oof_summary["results"]
        if float(row["delta_from_base"]) > 0.0
    ]

    artifacts = {
        "representation_information_loss": [
            "audits/frozen_block/frozen_block_probes.csv",
            "audits/frozen_block/frozen_block_probe_summary.json",
        ],
        "shared_scale_aggregation": ["multiscale_comparison.csv"],
        "horizon_scale_specialization": [
            "multiscale_comparison.csv",
            "multiscale_gate_weights.csv",
        ],
        "trained_multiscale_result": [
            "multiscale_comparison.csv",
            "multiscale_paired_daily.parquet",
        ],
        "score_capacity_control": ["multiscale_comparison.csv"],
        "horizon_conflict": [
            "audits/gradient/horizon_gradient_audit.parquet",
            "audits/gradient/horizon_gradient_summary.json",
            "audits/gradient/single_horizon_controls.csv",
        ],
        "context_source_information": [
            "audits/context/context_inference_probes.csv",
            "audits/context/context_training_ablations.csv",
            "audits/oof/oof_residual_probes.csv",
        ],
        "target_structure": [
            "audits/target_basis/target_basis_summary.json",
            "audits/target_basis/target_pairwise.csv",
        ],
    }
    hypotheses = {
        "representation_information_loss": {
            "supported_diagnostically": probe_supported,
            "earlier_tap_beats_final_by_horizon": earlier,
            "concatenated_beats_final_by_horizon": concatenated,
            "best_tap_by_horizon": frozen_summary["best_tap_by_horizon"],
            "interpretation": bottleneck_interpretation,
            "artifacts": artifacts["representation_information_loss"],
        },
        "shared_scale_aggregation": {
            **_comparison_evidence(shared),
            "comparison": "shared_multiscale versus final, seed 29",
            "artifacts": artifacts["shared_scale_aggregation"],
        },
        "horizon_scale_specialization": {
            **_comparison_evidence(specialized),
            "comparison": "horizon_multiscale versus shared_multiscale, seed 29",
            "artifacts": artifacts["horizon_scale_specialization"],
        },
        "trained_multiscale_result": {
            **_comparison_evidence(three_seed_aggregate),
            "supported": trained_supported,
            "worst_horizon_delta": worst_horizon_delta,
            "per_horizon": {
                str(row["horizon_minutes"]): _comparison_evidence(row)
                for row in three_seed_rows
                if int(row["horizon_minutes"]) in HORIZONS
            },
            "artifacts": artifacts["trained_multiscale_result"],
        },
        "score_capacity_control": {
            **_comparison_evidence(score),
            "multiscale_delta_seed29": trained_seed29["delta_ic"],
            "score_minus_multiscale_delta_seed29": (
                float(score["delta_ic"]) - float(trained_seed29["delta_ic"])
            ),
            "competing_explanation": score_competing,
            "interpretation": (
                "Score-level capacity is a competing explanation for the seed-29 "
                "multiscale improvement."
                if score_competing
                else "The score-capacity control does not match or exceed the "
                "positive seed-29 multiscale point improvement."
            ),
            "artifacts": artifacts["score_capacity_control"],
        },
        "horizon_conflict": {
            "gradient_audit_is_descriptive": True,
            "gradient_sample_count": gradient_summary["sample_count"],
            "gradient_cells_with_any_negative_cosines": negative_gradient_cells,
            "single_horizon_deltas": single_deltas,
            "gradient_cells_without_defined_cosines": undefined_gradient_cells,
            "single_horizon_improvements": [
                horizon for horizon, delta in single_deltas.items() if delta > 0.0
            ],
            "interpretation": (
                "Gradient cosines and single-horizon improvements are reported "
                "separately; neither is treated as proof of a representation bottleneck."
            ),
            "artifacts": artifacts["horizon_conflict"],
        },
        "context_source_information": {
            "inference_modes": sorted(context_summary["inference"]),
            "retraining_aggregate": context_aggregate,
            "positive_oof_residual_cells": positive_oof_cells,
            "interpretation": (
                "Inference corruption measures checkpoint reliance or alignment; "
                "retraining ablation measures replaceable usefulness; OOF residual "
                "probes measure information remaining after an out-of-fold incumbent."
            ),
            "artifacts": artifacts["context_source_information"],
        },
        "target_structure": {
            "pooled_target_correlation": target_summary["pooled_target_correlation"],
            "eigenvalues": target_summary["eigenvalues"],
            "variance_shares": target_summary["variance_shares"],
            "fixed_basis_variance": target_summary["fixed_basis_variance"],
            "artifacts": artifacts["target_structure"],
        },
    }
    return {
        "hypotheses": hypotheses,
        "bottleneck_interpretation": bottleneck_interpretation,
        "artifacts": artifacts,
    }
