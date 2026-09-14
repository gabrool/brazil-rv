"""Hold sealed C6 scores/policy fixed and isolate the accepted accounting repair."""

import json
from pathlib import Path
import time

import numpy as np

from brazil_rv.v2 import research_rounds as rr
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.execution_policy import ledger_gate_failures
from brazil_rv.v2.round6_costs import headline_ledger
from brazil_rv.v2.round6_readouts import ensemble, evaluation_design


REPO = Path(__file__).resolve().parents[1]
FOLDS = ("F2", "F6", "F10", "F14")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    started = time.monotonic()
    source = Path(
        read(REPO / "docs/v2_round6_operations.json")["complete_arm_recoveries"]["S0"][
            "local_root"
        ]
    )
    design = read(source / "frozen_design.json")
    old = evaluation_design(design)
    binding = read(REPO / "docs/v2_round7_inputs.json")["evaluation_design"]
    assert sha256_file(Path(binding["path"])) == binding["sha256"]
    new = evaluation_design({"evaluation_sources": {"design": read(binding["path"])}})
    output = REPO / "docs/v2_c6_accounting_bridge.json"
    report = {
        "status": "running",
        "scope": "sealed C6, fixed three seeds, four existing screen folds, unchanged policy; accounting-only replay",
        "seeds": [11, 29, 47],
        "folds": list(FOLDS),
        "new_training": False,
        "official_validation_accessed": False,
        "test_accessed": False,
        "before_design_sha256": sha256_file(source / "frozen_design.json"),
        "after_design": binding,
        "before_store": old["store"],
        "after_store": new["store"],
        "results": {},
    }
    write_json_atomic(output, report)
    contexts = [rr._open_ledger_replay(d) for d in [old, new]]
    policy, _ = rr.load_selected_policy(
        Path(old["execution_policy"]["root"]),
        expected_result_sha256=old["execution_policy"]["result_sha256"],
    )
    roster = read(source / "session2_roster.json")
    try:
        for fold in FOLDS:
            ix = contexts[0].evaluation[fold]
            np.testing.assert_array_equal(ix, contexts[1].evaluation[fold])
            assert contexts[0].store.isins == contexts[1].store.isins
            scores, mask, sources = ensemble(
                design,
                source,
                "C6",
                fold,
                (11, 29, 47),
                contexts[0].store.dates[ix],
                contexts[0].store.isins,
                roster=roster,
            )
            records = []
            for context in contexts:
                inputs = rr._evaluation_inputs(
                    context.store,
                    ix,
                    scores,
                    mask,
                    context.cdi,
                    context.bova11.close_by_session,
                    context.bova11_binding,
                    context.lending_borrow,
                    {
                        "sealed_round6_training_store": design["store"][
                            "manifest_sha256"
                        ]
                    },
                    transfer_chronology_clean=True,
                    execution_policy=policy,
                )
                _, summary, daily = headline_ledger(inputs)
                records.append(
                    {
                        "summary": summary,
                        "daily": daily,
                        "gates": ledger_gate_failures(summary, headline=True),
                    }
                )
            reference_path = (
                source / "aggregates/session2/C6" / fold / "evaluation.json"
            )
            reference = read(reference_path)["economics"]["headline_audit"][
                "daily_state"
            ]
            # Full saved daily accounting/order-state equality protects this historical bridge.
            if records[0]["daily"] != reference:
                raise ValueError(
                    f"Current implementation did not exactly reproduce original C6/{fold}"
                )
            report["results"][fold] = {
                "score_sources": sources,
                "sealed_reference_sha256": sha256_file(reference_path),
                "original_daily_exact": True,
                "before": records[0],
                "after": records[1],
                "net_delta_bps": records[1]["summary"]["mean_net_excess_bps_per_day"]
                - records[0]["summary"]["mean_net_excess_bps_per_day"],
            }
            write_json_atomic(output, report)
            print(
                f"{fold}: {records[0]['summary']['mean_net_excess_bps_per_day']:.6f} -> {records[1]['summary']['mean_net_excess_bps_per_day']:.6f}; {time.monotonic() - started:.1f}s",
                flush=True,
            )
    finally:
        for context in contexts:
            context.store.close()
    report["status"] = "completed"
    report["elapsed_seconds"] = time.monotonic() - started
    report["pooled_before_bps"] = float(
        np.mean(
            [
                d["net_excess_all_cash_bps"]
                for f in FOLDS
                for d in report["results"][f]["before"]["daily"]
            ]
        )
    )
    report["pooled_after_bps"] = float(
        np.mean(
            [
                d["net_excess_all_cash_bps"]
                for f in FOLDS
                for d in report["results"][f]["after"]["daily"]
            ]
        )
    )
    write_json_atomic(output, report)


if __name__ == "__main__":
    main()
