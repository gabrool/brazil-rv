"""One continuous development book over sealed, chronological fold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import evaluate as ev
from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .contract import DEVELOPMENT_END, DEVELOPMENT_FOLDS
from .execution_policy import ledger_gate_failures
from .round5_replay import ReplayInputs, _completed


def stitch_scores(
    panels: list[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Keep saved scores/masks literally; reject calendar gaps or overlaps."""
    spans = []
    end = None
    offset = 0
    for fold, indices, scores, mask in panels:
        indices = np.asarray(indices, np.int64)
        if (
            indices.ndim != 1
            or not indices.size
            or np.any(np.diff(indices) != 1)
            or (end is not None and indices[0] != end + 1)
        ):
            raise ValueError(
                "continuous scores require adjacent, nonoverlapping fold dates"
            )
        if scores.shape != mask.shape or scores.shape[0] != indices.size:
            raise ValueError("saved score and mask axes differ from their fold dates")
        spans.append(
            {
                "fold": fold,
                "start_offset": offset,
                "end_offset": offset + indices.size - 1,
                "first_session_index": int(indices[0]),
                "last_session_index": int(indices[-1]),
                "date_count": int(indices.size),
                "dates_with_any_score_by_head": mask.any(axis=1).sum(axis=0).tolist(),
                "final_date_valid_names_by_head": mask[-1].sum(axis=0).tolist(),
                "missing_entire_score_dates": indices[~mask.any(axis=(1, 2))].tolist(),
            }
        )
        offset += indices.size
        end = indices[-1]
    return (
        np.concatenate([p[1] for p in panels]),
        np.concatenate([p[2] for p in panels]),
        np.concatenate([p[3] for p in panels]),
        spans,
    )


def boundary_evidence(result, spans: list[dict]) -> list[dict]:
    """Audit observable opening continuity and cross-boundary order lifetimes."""
    if np.any(result.exit_instructions_terminal[:-1]) or any(
        row.reason == "evaluation_end"
        and row.cancellation_session != len(result.dates) - 1
        for row in result.cancellations
    ):
        raise ValueError(
            "an internal date triggered terminal liquidation or cancellation"
        )
    orders = {order.order_id: order for order in result.intended_orders}
    output = []
    for previous, following in zip(spans[:-1], spans[1:], strict=True):
        current = int(following["start_offset"])
        last = current - 1
        if result.nav[last] != result.start_nav[current]:
            raise ValueError("model switch reset the portfolio NAV")
        shares = result.signed_shares[last]
        if (
            np.count_nonzero(shares > 0) != result.held_count_start_of_day_long[current]
            or np.count_nonzero(shares < 0)
            != result.held_count_start_of_day_short[current]
        ):
            raise ValueError("model switch reset held positions")
        output.append(
            {
                "from_fold": previous["fold"],
                "to_fold": following["fold"],
                "previous_date": str(result.dates[last]),
                "new_model_date": str(result.dates[current]),
                "previous_close_nav": float(result.nav[last]),
                "next_opening_nav": float(result.start_nav[current]),
                "nav_continuity_exact": True,
                "opening_long_count": int(result.held_count_start_of_day_long[current]),
                "opening_short_count": int(
                    result.held_count_start_of_day_short[current]
                ),
                "prior_close_state_carried": {
                    name: float(getattr(result, name)[last])
                    for name in (
                        "free_cash",
                        "restricted_cash",
                        "hedge_restricted_cash",
                        "receivables",
                        "payables",
                        "hedge_signed_shares",
                    )
                },
                "next_close_state": {
                    name: float(getattr(result, name)[current])
                    for name in (
                        "free_cash",
                        "restricted_cash",
                        "hedge_restricted_cash",
                        "receivables",
                        "payables",
                        "hedge_signed_shares",
                    )
                },
                "prior_pending_entry_count": int(result.pending_entry_count[last]),
                "prior_pending_exit_count": int(result.pending_exit_count[last]),
                "next_pending_entry_count": int(result.pending_entry_count[current]),
                "next_pending_exit_count": int(result.pending_exit_count[current]),
                "pre_switch_orders_filled_after_switch": sorted(
                    {
                        fill.order_id
                        for fill in result.fills
                        if orders[fill.order_id].decision_session
                        < current
                        <= fill.fill_session
                    }
                ),
                "pre_switch_orders_cancelled_after_switch": sorted(
                    {
                        row.order_id
                        for row in result.cancellations
                        if orders[row.order_id].decision_session
                        < current
                        <= row.cancellation_session
                    }
                ),
            }
        )
    return output


def evaluate_continuous(inputs: ev.EvaluationInputs, spans: list[dict]):
    if inputs.dates[-1] > DEVELOPMENT_END or not inputs.transfer_chronology_clean:
        raise PermissionError(
            "continuous replay requires clean development-only scores"
        )
    economics, headline, _ = ev._evaluate_economics(
        inputs, settle_terminal_residuals=True
    )
    boundary = boundary_evidence(headline, spans)
    return economics, headline, boundary


def _reset_series(daily: list[dict], unresolved: list[str]) -> dict:
    net = np.array([row["net_return"] for row in daily], np.float64)
    excess = np.array([row["net_excess_all_cash_bps"] for row in daily], np.float64)
    return {
        "dates": [row["date"] for row in daily],
        "net_return": net.tolist(),
        "compounded_reset_return_index": np.cumprod(1 + net).tolist(),
        "net_excess_bps": excess.tolist(),
        "mean_net_excess_bps": float(np.mean(excess)),
        "unresolved_folds": unresolved,
        "interpretation": "fold-reset returns compounded for comparison; not one continuously carried ledger",
    }


def run(replay_root: Path, output: Path, strategy: str = "screening/S0") -> dict:
    if output.exists():
        raise FileExistsError(output)
    design_path = replay_root / "frozen_design.json"
    design = rr._read_json(design_path)
    design_sha = sha256_file(design_path)
    if design_sha != design_path.with_suffix(".json.sha256").read_text().split()[0]:
        raise ValueError("frozen replay design hash differs")
    books = sorted(
        [b for b in design["books"] if b["key"].rsplit("/", 1)[0] == strategy],
        key=lambda b: int(b["fold"][1:]),
    )
    if [b["fold"] for b in books] != list(DEVELOPMENT_FOLDS):
        raise ValueError("continuous strategy requires all fourteen registered folds")
    if any(b["historical_status"] != "accepted" for b in books):
        raise ValueError("continuous strategy contains a historically rejected fold")
    context = ReplayInputs(design)
    try:
        panels, comparator_sources = [], []
        reset_rows = {key: [] for key in ("sealed", "repaired")}
        unresolved = {key: [] for key in reset_rows}
        policy = None
        source_hashes = None
        treatment = "joint" if context.rates is not None else "bova_only"
        for book in books:
            original, inputs = context.original(book)
            if panels and policy != inputs.execution_policy:
                raise ValueError(
                    "continuous scores change execution policy at a fold boundary"
                )
            policy = inputs.execution_policy
            if (
                source_hashes is not None
                and source_hashes != original["source_artifact_hashes"]
            ):
                raise ValueError(
                    "continuous folds do not share one economic source contract"
                )
            source_hashes = dict(original["source_artifact_hashes"])
            panels.append(
                (book["fold"], inputs.session_indices, inputs.scores, inputs.score_mask)
            )
            destination = replay_root / treatment / book["key"]
            completed = _completed(destination, design_sha, book, treatment)
            if completed is None:
                raise FileNotFoundError(
                    f"missing repaired fold comparator: {destination}"
                )
            repaired = rr._evaluation_from_path(destination / "evaluation.json")
            if (
                ev._input_hashes(context.repaired(inputs, treatment))
                != repaired["input_hashes"]
            ):
                raise ValueError(
                    "repaired comparator differs from the continuous input treatment"
                )
            comparator_sources.append(
                {
                    "fold": book["fold"],
                    "path": str(destination),
                    "comparison_sha256": sha256_file(destination / "comparison.json"),
                    "output_hashes": completed["output_hashes"],
                }
            )
            for key, report in (("sealed", original), ("repaired", repaired)):
                reset_rows[key].extend(
                    {
                        field: row[field]
                        for field in ("date", "net_return", "net_excess_all_cash_bps")
                    }
                    for row in report["economics"]["headline_audit"]["daily_state"]
                )
                if report["economics"]["headline"]["economics_unresolved"]:
                    unresolved[key].append(book["fold"])
        indices, scores, mask, spans = stitch_scores(panels)
        source_hashes["continuous_score_sources"] = ev._strings_sha256(
            tuple(b["files"]["score_manifest.json"]["sha256"] for b in books)
        )
        context.templates.clear()
        inputs = rr._evaluation_inputs(
            context.context.store,
            indices,
            scores,
            mask,
            context.context.cdi,
            context.context.bova11.close_by_session,
            context.context.bova11_binding,
            context.context.lending_borrow,
            source_hashes,
            transfer_chronology_clean=True,
            execution_policy=policy,
        )
        inputs = context.repaired(inputs, treatment)
        output.mkdir(parents=True, exist_ok=False)
        frozen = {
            "schema": "ROUND5_CONTINUOUS_BOOK_DESIGN_V1",
            "strategy": strategy,
            "replay_design_path": str(design_path),
            "replay_design_sha256": sha256_file(design_path),
            "source_code_sha256": sha256_file(Path(__file__)),
            "source_program_sha256": {
                str(path.resolve()): sha256_file(path)
                for path in (
                    *(
                        Path(__file__).with_name(name + ".py")
                        for name in (
                            "round5_continuous",
                            "round5_replay",
                            "research_rounds",
                            "validate_pipeline",
                            "evaluate",
                            "execution_policy",
                            "corporate_actions",
                            "contract",
                        )
                    ),
                    Path(__file__).parents[1] / "execution" / "stateful_ledger.py",
                )
            },
            "source_books": books,
            "repaired_comparator_sources": comparator_sources,
            "score_coverage": spans,
            "input_hashes": ev._input_hashes(inputs),
            "switch_rule": "new fold scores at its first decision; all ledger and smoothing state carried",
            "terminal_rule": "single final2024 liquidation and registered uncertain-residual settlement",
            "missing_score_rule": "retain saved masks and registered policy missing-score handling; never fabricate predictions",
            "treatment": treatment,
            "official_validation_accessed": False,
            "test_accessed": False,
            "neural_fits": False,
        }
        frozen_hash = write_json_atomic(output / "frozen_design.json", frozen)
        economics, headline, boundaries = evaluate_continuous(inputs, spans)
        write_json_atomic(output / "economics.json", economics)
        reset = {
            key: _reset_series(rows, unresolved[key])
            for key, rows in reset_rows.items()
        }
        write_json_atomic(output / "fold_reset_comparators.json", reset)
        if any(v["dates"] != [str(d) for d in inputs.dates] for v in reset.values()):
            raise ValueError("continuous and reset comparator calendars differ")
        report = {
            "schema": "ROUND5_CONTINUOUS_BOOK_RESULT_V1",
            "strategy": strategy,
            "frozen_design_sha256": frozen_hash,
            "date_count": len(inputs.dates),
            "model_switch_count": len(boundaries),
            "ledger_state_initializations": 1,
            "action_terms_source": inputs.action_terms_source,
            "schedule_source": inputs.schedule_source,
            "borrow_source_label": inputs.borrow_source_label,
            "transfer_chronology_clean": True,
            "source_artifact_hashes": dict(inputs.source_artifact_hashes),
            "terminal_boundary": str(inputs.dates[-1]),
            "headline": economics["headline"],
            "boundary_evidence": boundaries,
            "score_coverage": spans,
            "engineering_gate_failures": ledger_gate_failures(
                economics["headline"], headline=True
            ),
            "fold_reset_comparison": {
                key: {
                    "mean_net_excess_bps": value["mean_net_excess_bps"],
                    "unresolved_folds": value["unresolved_folds"],
                    "continuous_minus_reset_mean_bps": float(
                        np.mean(
                            headline.net_excess_all_cash_bps - value["net_excess_bps"]
                        )
                    ),
                }
                for key, value in reset.items()
            },
            "limitations": [
                "Development-only descriptive replay of in-sample-selected S0/policy; no new selection or deployment claim.",
                "Registered inferred corporate-action and terminal-settlement uncertainty remains labelled; fold resets are historical comparators.",
            ],
            "official_validation_accessed": False,
            "test_accessed": False,
            "neural_fits": False,
            "research_claim": True,
            "deployment_changed": False,
            "files": {p.name: sha256_file(p) for p in output.glob("*.json")},
        }
        write_json_atomic(output / "result.json", report)
        return report
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", default="screening/S0")
    args = parser.parse_args()
    result = run(args.replay_root, args.output, args.strategy)
    print(
        json.dumps(
            {
                "date_count": result["date_count"],
                "model_switch_count": result["model_switch_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
