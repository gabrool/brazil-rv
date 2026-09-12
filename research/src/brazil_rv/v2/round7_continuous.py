"""One continuous development book for the completed Round-7 designation."""

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
from .round5_continuous import _reset_series, evaluate_continuous, stitch_scores
from .round7_program import read
from .round7_readouts import aggregate_path, economics_design


def run(root):
    decision_path = root / "decision.json"
    decision = read(decision_path)
    if decision["status"] != "complete":
        raise ValueError(
            "the final registered decision must precede its continuous book"
        )
    cell, seeds = decision["designation"], decision["seeds"]
    design = read(root / "frozen_design.json")
    context = rr._open_ledger_replay(economics_design(design))
    output = root / "continuous" / cell
    if output.exists():
        raise FileExistsError(output)
    try:
        panels, sources, daily, unresolved = [], [], [], []
        for fold in DEVELOPMENT_FOLDS:
            path = aggregate_path(root, cell, fold, seeds)
            saved = retained(context, path, fold)
            inputs = saved.inputs
            panels.append(
                (fold, inputs.session_indices, inputs.scores, inputs.score_mask)
            )
            sources.append(
                {
                    "fold": fold,
                    "score_manifest_sha256": sha256_file(path / "score_manifest.json"),
                    "accepted_sha256": sha256_file(path / "accepted.json"),
                }
            )
            economics = saved.result.report["economics"]
            daily.extend(economics["headline_audit"]["daily_state"])
            if economics["headline"]["economics_unresolved"]:
                unresolved.append(fold)
        indices, scores, mask, spans = stitch_scores(panels)
        hashes = dict(inputs.source_artifact_hashes)
        hashes["continuous_score_sources"] = ev._strings_sha256(
            tuple(s["score_manifest_sha256"] for s in sources)
        )
        combined = rr._evaluation_inputs(
            context.store,
            indices,
            scores,
            mask,
            context.cdi,
            context.bova11.close_by_session,
            context.bova11_binding,
            context.lending_borrow,
            hashes,
            transfer_chronology_clean=True,
            execution_policy=inputs.execution_policy,
        )
        reset = _reset_series(daily, unresolved)
        if reset["dates"] != [str(d) for d in combined.dates]:
            raise ValueError("continuous and fold-reset calendars differ")
        output.mkdir(parents=True)
        write_json_atomic(
            output / "source.json",
            {
                "decision_sha256": sha256_file(decision_path),
                "sources": sources,
                "input_hashes": ev._input_hashes(combined),
                "code_sha256": sha256_file(Path(__file__)),
            },
        )
        economics, headline, boundaries = evaluate_continuous(combined, spans)
        write_json_atomic(output / "economics.json", economics)
        write_json_atomic(output / "fold_reset_comparator.json", reset)
        result = {
            "status": "complete",
            "cell": cell,
            "seeds": seeds,
            "headline": economics["headline"],
            "engineering_gate_failures": ledger_gate_failures(
                economics["headline"], headline=True
            ),
            "ledger_state_initializations": 1,
            "model_switch_count": len(boundaries),
            "boundary_evidence": boundaries,
            "continuous_minus_reset_bps_per_day": float(
                np.mean(headline.net_excess_all_cash_bps - reset["net_excess_bps"])
            ),
            "decision_sha256": sha256_file(decision_path),
            "selection_weight": 0,
            "interpretation": "development-selected descriptive book; carries holdings across model changes; one final settlement",
            **rr.RESEARCH_FLAGS,
        }
        write_json_atomic(output / "result.json", result)
        return result
    finally:
        context.store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
