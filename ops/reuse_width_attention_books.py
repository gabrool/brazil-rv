"""Compose already qualified attention books into the corrected width wave."""

import json
from pathlib import Path

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    old = bound_json(run["stage_d_width_original_evaluation_plan"])
    new = bound_json(run["stage_d_width_evaluation_plan"])
    root = Path(run["stage_d_width_plan"]["path"]).parent
    (root / "executed_book_reuse.py").write_bytes(Path(__file__).read_bytes())
    counts = {}
    for kind, expected_changes in (
        ("primary", {"fit_root", "refits"}),
        ("sensitivities", {"primary"}),
        ("funded_denial", {"denied_sensitivities", "primary"}),
        ("fraction_precision", {"primary", "producer", "qualifier"}),
    ):
        previous, current = bound_json(old[kind]), bound_json(new[kind])
        assert set(previous) == set(current)
        assert {k for k in previous if previous[k] != current[k]} == expected_changes
        if kind == "fraction_precision":
            for key in ("producer", "qualifier"):
                assert previous[key]["sha256"] == current[key]["sha256"]
        source = Path(old[kind]["path"]).parent
        destination = Path(new[kind]["path"]).parent
        progress = bound_json(binding(source / "replays.json"))
        assert not (destination / "replays.json").exists()
        result = dict(progress, plan=new[kind], status="waiting_for_fits")
        for key in ("completed", "skipped", "processed_primary"):
            if key in result:
                result[key] = [
                    r
                    for r in result[key]
                    if "/TE_128/" in (r if isinstance(r, str) else r["key"])
                ]
        if "pending_groups" in result:
            result["pending_groups"] = []
        proofs = []
        for record in result["completed"]:
            bound_json(record["book"])
            name = record["key"].replace("/", "_") + ".json"
            proof = source / "qualification" / name
            if proof.exists():
                content = bound_json(binding(proof))
                assert content["passed"] and content["book"] == record["book"]
                target = destination / "qualification" / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(proof.read_bytes())
                assert binding(target)["sha256"] == binding(proof)["sha256"]
                proofs.append(dict(original=binding(proof), composed=binding(target)))
        if kind in ("primary", "sensitivities"):
            assert len(proofs) == len(result["completed"])
        write_json_atomic(destination / "replays.json", result)
        counts[kind] = dict(
            completed=len(result["completed"]),
            skipped=len(result.get("skipped", [])),
            processed_primary=len(result.get("processed_primary", [])),
            source=binding(source / "replays.json"),
            composed=binding(destination / "replays.json"),
            copied_proofs=proofs,
        )
    report = dict(
        original=run["stage_d_width_original_evaluation_plan"],
        corrected=run["stage_d_width_evaluation_plan"],
        counts=counts,
        scope="Exact previously qualified TE128 books and source forecasts, unchanged account/scenario/qualification recipes. New progress references original book bytes; only small qualified receipts are copied. No account, forecast, fit or arithmetic replay. All original GRU96 books excluded from corrected progress and preserved in original evidence.",
    )
    write_json_atomic(root / "attention_book_reuse.json", report)
    run["stage_d_width_attention_book_reuse"] = binding(
        root / "attention_book_reuse.json"
    )
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                key: {
                    k: value[k] for k in ("completed", "skipped", "processed_primary")
                }
                for key, value in counts.items()
            }
        )
    )


if __name__ == "__main__":
    main()
