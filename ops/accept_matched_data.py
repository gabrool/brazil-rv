"""Bind qualified matched-data composition to fresh compatible refits."""

import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.contract import TARGET_NEUTRALIZATION_FEATURES
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    assembly = bound_json(run["stage_c_event_store_assembly"])
    audit = bound_json(run["stage_c_event_store_input_audit"])
    assert audit["status"] == "passed_complete_store_and_combined_consumers"
    assert audit["mismatches"] == 0 and audit["names"] == 933 and audit["history"] == 60
    folder = Path(run["stage_c_root"]) / "event_data/composition"
    attribution = binding(folder / "neutral_attribution.json")
    attributed = bound_json(attribution)
    bound_json(run["stage_c_event_index_reopening_qualification"])
    output = PROJECT / "docs/v2_matched_data_inputs.json"
    assert not output.exists()
    # Classify saved virtual support gains with its mask formula, without
    # repeating endpoint walks, SVD projection or dataset/collator samples.
    rows = np.load(folder / "neutral_affected_rows.npy")
    masks, raw = [], []
    for spec in (assembly["parent_store"], assembly["store"]):
        root = Path(spec["root"])
        manifest = json.loads((root / "manifest.json").read_text())

        def read(key):
            return np.load(root / manifest["arrays"][key]["path"], mmap_mode="r")[rows]

        fields = [
            manifest["feature_names"]["slow"].index(k)
            for k in TARGET_NEUTRALIZATION_FEATURES
        ]
        valid, sigma = read("target_valid"), read("target_scale_sigma")
        mask = valid & np.isfinite(read("target_shareholder_simple_return"))
        mask &= (
            read("slow_valid")[:, :, fields].all(axis=2)
            & np.isfinite(sigma)
            & (sigma > 1e-8)
        )[:, :, None]
        mask &= (mask.sum(axis=1) >= 20)[:, None, :]
        masks.append(mask)
        raw.append(valid)
    gains, losses = masks[1] & ~masks[0], masks[0] & ~masks[1]
    assert (
        int(gains.sum()) == attributed["gains"]
        and int(losses.sum()) == attributed["losses"]
    )
    support = dict(
        new_gross_outcomes_entering_neutral=int((gains & ~raw[0]).sum()),
        existing_gross_outcomes_with_restored_risk_support=int((gains & raw[0]).sum()),
        losses=int(losses.sum()),
        method="Mask-only classification of saved final virtual attribution; no numerical target or consumer rerun.",
    )
    write_json_atomic(folder / "neutral_support_attribution.json", support)
    evidence = {
        key: value
        for key, value in run.items()
        if key.startswith("stage_c_event_") and isinstance(value, dict)
    }
    evidence["stage_c_neutral_attribution"] = attribution
    evidence["neutral_support_attribution"] = binding(
        folder / "neutral_support_attribution.json"
    )
    acceptance = dict(
        status="accepted_complete_matched_data_contract",
        store=assembly["store"],
        parent=assembly["parent_store"],
        contract=assembly["contract"],
        earlier_baseline_preserved=run["economic_refit_inputs"],
        evidence=evidence,
        report=binding(PROJECT / "docs/v2_MATCHED_ECONOMIC_REPLAYS.md"),
        refit_required=True,
        neutral_support=support,
        accounting_source_replay="Frozen OLD PolicyData and previously qualified corporate candidate terms; no old checkpoint reads altered store coordinates.",
        account_terms=run["stage_c_event_candidate_terms"],
        boundaries="All933/full60/3717dates and original source assignments remain. Unit/peer support losses, missing M1, ALSC locked credit, ENAT null auction, and exposure-conditional Cielo/daytrade remain explicit. No revised profitability or completed fit is implied.",
        remaining_program="Fresh original-recipe matched C parents/children and economic comparisons; additional folds and D only under registration.",
    )
    write_json_atomic(output, acceptance)
    run["matched_data_inputs"] = binding(output)
    run["economic_refit_inputs"] = binding(output)
    run["stage_c_neutral_attribution"] = attribution
    run["stage_c_refit_root"] = str(Path(run["stage_c_root"]) / "data_refits")
    # The old interrupted plan remains a historical binding, never overwritten.
    run["stage_c_interrupted_refit_plan"] = run["stage_c_refit_plan"]
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                acceptance=binding(output),
                store=assembly["store"],
                refit_root=run["stage_c_refit_root"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
