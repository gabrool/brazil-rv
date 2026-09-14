"""Refresh only policy-composite diagnostics; retain all original result receipts."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.checkpoint_readouts import retained
from brazil_rv.v2.post_data_readouts import alignment, intervals, prepare_economics
from brazil_rv.v2 import post_data_readouts, research_rounds as rr
from brazil_rv.v2.research_checkpoint import _completed

root = Path(sys.argv[1])
receipts = root / "diagnostic_repair"
receipts.mkdir(exist_ok=False)
files = ["screen_result.json", "screen_with_attention_asam50.json"]
results = {name: json.loads((root / name).read_text()) for name in files}
for name in files:
    shutil.copyfile(root / name, receipts / name)
context = rr._open_ledger_replay(prepare_economics(root))
proof = []
corrected = {}
changed_metrics = {
    "composite_persistence_1",
    "composite_persistence_5",
    "D3_composite_neutral_ic_common",
    "D5_composite_neutral_ic_common",
    "D10_composite_neutral_ic_common",
}
try:
    for cell, paths in results[files[-1]]["aggregate_paths"].items():
        corrected[cell] = {}
        for fold, directory in paths.items():
            path = Path(directory)
            assert _completed(path)
            before = {
                n: sha256_file(path / n)
                for n in (
                    "evaluation.json",
                    "score_manifest.json",
                    "scores.npy",
                    "score_mask.npy",
                    "daily_readouts.json",
                    "accepted.json",
                    "seed_ic.json",
                )
            }
            inputs = retained(context, path, fold).inputs
            assert inputs.execution_policy is not None
            old = json.loads((path / "alignment.json").read_text())
            new = alignment(inputs)
            for key in old["series"].keys() - changed_metrics:
                assert old["series"][key] == new["series"][key]
            for key in old["populations"].keys() - {"score_available_to_book"}:
                assert old["populations"][key] == new["populations"][key]
            destination = receipts / cell / fold / "alignment.json"
            destination.parent.mkdir(parents=True)
            shutil.copyfile(path / "alignment.json", destination)
            write_json_atomic(path / "alignment.json", new)
            assert before == {n: sha256_file(path / n) for n in before}
            corrected[cell][fold] = new
            proof.append(
                {
                    "cell": cell,
                    "fold": fold,
                    "protected_artifacts": before,
                    "old_alignment_sha256": sha256_file(destination),
                    "new_alignment_sha256": sha256_file(path / "alignment.json"),
                    "policy": vars(inputs.execution_policy),
                }
            )
finally:
    context.store.close()
for name, result in results.items():
    folds = result["screen_folds"]
    result["alignment"] = {c: corrected[c] for c in result["alignment"]}
    for pair, fields in result["paired_traded_head_alignment"].items():
        left, right = pair.split("_minus_")
        for key in changed_metrics:
            fields[key] = intervals(
                [
                    np.asarray(corrected[left][f]["series"][key], float)
                    - np.asarray(corrected[right][f]["series"][key], float)
                    for f in folds
                ]
            )
    result["policy_composite_diagnostic_repair"] = {
        "previous_result_sha256": sha256_file(receipts / name),
        "repair_implementation_sha256": sha256_file(Path(__file__)),
        "readout_implementation_sha256": sha256_file(Path(post_data_readouts.__file__)),
        "scope": "Policy-specific composite IC, persistence and available-to-book population only; fits, model IC, scores, seed results and book economics unchanged",
    }
    write_json_atomic(root / name, result)
write_json_atomic(
    receipts / "verification.json",
    {
        "status": "passed",
        "protected_artifacts_exact": True,
        "aggregates": proof,
        "changed_series": sorted(changed_metrics),
        "previous_results": {name: sha256_file(receipts / name) for name in files},
    },
)
print(
    json.dumps(
        {
            "status": "passed",
            "aggregates": len(proof),
            "protected_artifacts_exact": True,
        }
    )
)
