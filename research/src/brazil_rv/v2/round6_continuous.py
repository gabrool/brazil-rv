"""Descriptive continuous ledger for the completed Round-6 decision."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import evaluate as ev
from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .checkpoint_readouts import retained
from .contract import DEVELOPMENT_FOLDS
from .execution_policy import ledger_gate_failures
from .research_checkpoint import _completed
from .round5_continuous import _reset_series, evaluate_continuous, stitch_scores
from .round6 import SESSION1
from .round6_readouts import evaluation_design


def run(root: Path, audit: Path, output: Path) -> str:
    if output.exists():
        raise FileExistsError(output)
    decision = rr._read_json(audit)
    design_path = root / "frozen_design.json"
    design_sha = sha256_file(design_path)
    if (
        decision["status"] != "completed"
        or decision["frozen_design_sha256"] != design_sha
    ):
        raise ValueError(
            "continuous book requires this experiment's completed seed audit"
        )
    trace = decision["decision_traces"]["full"]
    if trace["confirmation_reasons"] and not trace.get("confirmation_complete", False):
        raise ValueError("registered confirmation must finish before the final readout")
    arm = decision["decision"]["working_research_comparator"]
    seeds = decision["seeds"]
    group = (
        "confirmation"
        if len(seeds) > 3
        else ("session1" if arm in ("S0", *SESSION1) else "session2")
    )
    design = evaluation_design(rr._read_json(design_path))
    context = rr._open_ledger_replay(design)
    try:
        panels, sources, daily, unresolved = [], [], [], []
        policy, source_hashes = None, None
        for fold in DEVELOPMENT_FOLDS:
            folder = root / "aggregates" / group / arm / fold
            if not _completed(folder):
                raise ValueError(f"continuous source is not an accepted book: {folder}")
            metadata = rr._read_json(folder / "score_manifest.json")["metadata"]
            if (
                metadata["arm"] != arm
                or metadata["seeds"] != seeds
                or metadata["round6_frozen_design_sha256"] != design_sha
            ):
                raise ValueError("continuous source differs from the final decision")
            saved = retained(context, folder, fold)
            inputs = saved.inputs
            if policy is not None and (
                policy != inputs.execution_policy
                or source_hashes != dict(inputs.source_artifact_hashes)
            ):
                raise ValueError("continuous folds change their economic contract")
            policy = inputs.execution_policy
            source_hashes = dict(inputs.source_artifact_hashes)
            panels.append(
                (fold, inputs.session_indices, inputs.scores, inputs.score_mask)
            )
            sources.append(
                {
                    "fold": fold,
                    "path": str(folder),
                    "accepted_sha256": sha256_file(folder / "accepted.json"),
                    "score_manifest_sha256": sha256_file(
                        folder / "score_manifest.json"
                    ),
                }
            )
            report = rr._read_json(folder / "evaluation.json")
            daily.extend(report["economics"]["headline_audit"]["daily_state"])
            if report["economics"]["headline"]["economics_unresolved"]:
                unresolved.append(fold)
            del saved, inputs, report
        indices, scores, mask, spans = stitch_scores(panels)
        source_hashes["continuous_score_sources"] = ev._strings_sha256(
            tuple(row["score_manifest_sha256"] for row in sources)
        )
        inputs = rr._evaluation_inputs(
            context.store,
            indices,
            scores,
            mask,
            context.cdi,
            context.bova11.close_by_session,
            context.bova11_binding,
            context.lending_borrow,
            source_hashes,
            transfer_chronology_clean=True,
            execution_policy=policy,
        )
        reset = _reset_series(daily, unresolved)
        if reset["dates"] != [str(d) for d in inputs.dates]:
            raise ValueError("continuous and fold-reset calendars differ")
        output.mkdir(parents=True)
        frozen_sha = write_json_atomic(
            output / "frozen_design.json",
            {
                "round6_frozen_design_sha256": design_sha,
                "seed_audit_sha256": sha256_file(audit),
                "arm": arm,
                "seeds": seeds,
                "source_books": sources,
                "input_hashes": ev._input_hashes(inputs),
                "source_code_sha256": sha256_file(Path(__file__)),
                "source_program_sha256": {
                    name: sha256_file(Path(__file__).with_name(name + ".py"))
                    for name in (
                        "round5_continuous",
                        "research_rounds",
                        "validate_pipeline",
                        "evaluate",
                        "execution_policy",
                        "corporate_actions",
                        "contract",
                    )
                },
                "switch_rule": "new fold scores; carry all portfolio and smoothing state",
                "terminal_rule": "one final development liquidation with registered settlement",
                "selection_weight": 0,
                **rr.RESEARCH_FLAGS,
            },
        )
        economics, headline, boundaries = evaluate_continuous(inputs, spans)
        write_json_atomic(output / "economics.json", economics)
        write_json_atomic(output / "fold_reset_comparator.json", reset)
        return write_json_atomic(
            output / "result.json",
            {
                "status": "completed",
                "arm": arm,
                "seeds": seeds,
                "frozen_design_sha256": frozen_sha,
                "seed_audit_sha256": sha256_file(audit),
                "parent_inconclusive": decision["decision"]["parent_inconclusive"],
                "ledger_state_initializations": 1,
                "date_count": len(inputs.dates),
                "model_switch_count": len(boundaries),
                "boundary_evidence": boundaries,
                "score_coverage": spans,
                "headline": economics["headline"],
                "engineering_gate_failures": ledger_gate_failures(
                    economics["headline"], headline=True
                ),
                "continuous_minus_fold_reset_mean_bps": float(
                    np.mean(headline.net_excess_all_cash_bps - reset["net_excess_bps"])
                ),
                "action_terms_source": inputs.action_terms_source,
                "schedule_source": inputs.schedule_source,
                "borrow_source_label": inputs.borrow_source_label,
                "selection_weight": 0,
                "limitations": [
                    "Development-selected descriptive book, not an independent test.",
                    "Inherited action and terminal-settlement uncertainty remains.",
                ],
                **rr.RESEARCH_FLAGS,
            },
        )
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.root, args.seed_audit, args.output))


if __name__ == "__main__":
    main()
