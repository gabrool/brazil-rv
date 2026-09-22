"""Admit the composed added-period repairs after their complete consumer proof."""

import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    assembly = bound_json(run["scaling_data_store_assembly"])
    qualification = bound_json(run["scaling_data_store_input_audit"])
    assert qualification["status"] == "passed_complete_store_and_combined_consumers"
    assert qualification["mismatches"] == 0
    plan = bound_json(run["scaling_data_plan"])
    root = Path(assembly["store"]["root"])
    manifest = bound_json(
        dict(
            path=str(root / "manifest.json"),
            sha256=assembly["store"]["manifest_sha256"],
        )
    )
    parent = Path(plan["parent"]["root"])
    original = bound_json(
        dict(
            path=str(parent / "manifest.json"), sha256=plan["parent"]["manifest_sha256"]
        )
    )
    out = Path(run["scaling_data_store_assembly"]["path"]).parent
    dates, names = (
        np.load(root / "date_index.npy"),
        np.load(root / "isin_index.npy").tolist(),
    )
    wealth = bound_json(run["scaling_data_wealth"])
    # Public prebirth coordinates and post-effect claim wealth have different
    # meanings. Enumerate the latter without calling it a source observation.
    with np.load(wealth["deltas"]["path"]) as patch:
        ix = patch["shareholder_wealth_valid__indices"]
        values = patch["shareholder_wealth_valid__values"]
        assert values.all()
        owners = {names.index(e["isin"]): e for e in plan["history"]}
        assert set(ix[:, 1]) <= set(owners)
        observed = np.load(root / manifest["arrays"]["observed"]["path"], mmap_mode="r")
        active = np.load(root / manifest["arrays"]["active"]["path"], mmap_mode="r")
        assert not observed[tuple(ix.T)].any() and not active[tuple(ix.T)].any()
        for axis, event in owners.items():
            rows = ix[ix[:, 1] == axis, 0]
            assert (dates[rows] >= np.datetime64(event["effective_date"])).all()
            successor = names.index(event["successor_isin"])
            for key in patch.files:
                if key.endswith("__indices"):
                    cells = patch[key]
                    assert not (
                        (cells[:, 1] == successor)
                        & (dates[cells[:, 0]] < np.datetime64(event["effective_date"]))
                    ).any()
        wealth_scope = dict(
            post_effect_retired_claim_valid_gains=len(ix),
            observed_or_active_gains=0,
            successor_prebirth_changes=0,
            meaning="Same-class shareholder entitlement wealth follows the successor on the retired source coordinate under the existing wealth contract. It is not a quote, eligible name, fill or model observation. Successor public prebirth coordinates remain untouched.",
        )
    losses = bound_json(bound_json(run["scaling_data_daily_qualification"])["losses"])
    daily_losses = dict(
        total=len(losses), still_active=sum(x["still_active"] for x in losses)
    )
    daily_losses["retired"] = daily_losses["total"] - daily_losses["still_active"]
    assert all(x["field"] in range(27, 32) for x in losses if x["still_active"])
    # The complete-array proof already establishes these untouched hashes.
    for key in (
        "raw_close",
        "observed",
        "trade_observed",
        "fast_patch_valid",
        "fast_patch_mask",
    ):
        assert original["arrays"][key]["sha256"] == manifest["arrays"][key]["sha256"]
    scope = dict(
        wealth=wealth_scope,
        daily_losses=daily_losses,
        live_financial_losses=39,
        live_lending_losses=135,
        explanation="Unchanged three-other-peer rule accounts for live daily losses. Financial/lending losses retain the independently enumerated unit-uncertainty barriers; counts overlap dates and are not independent evidence.",
    )
    write_json_atomic(out / "support_disposition.json", scope)
    evidence = {
        k: v
        for k, v in run.items()
        if k.startswith("scaling_data_") and isinstance(v, dict) and "sha256" in v
    }
    evidence.update(
        neutral_attribution=binding(out / "neutral_attribution.json"),
        support_disposition=binding(out / "support_disposition.json"),
    )
    neutral = bound_json(evidence["neutral_attribution"])
    assert neutral["gains"] == 1032 and neutral["losses"] == 0
    result = dict(
        status="accepted_complete_scaling_data_contract",
        store=assembly["store"],
        parent=plan["parent"],
        contract=assembly["contract"],
        evidence=evidence,
        account_terms=bound_json(run["scaling_data_targets"])["account_terms"],
        earlier_baseline_preserved=run["economic_refit_inputs"],
        refit_required=True,
        report=binding(PROJECT / "docs/v2_SCALING_DATA_REPAIRS.md"),
        accounting_source_replay="Old forecasts retain their old model coordinates. Explicitly composed account terms are available separately; no old checkpoint reads this store.",
        boundaries="All933/full60/3717dates, original sources and accepted parents remain. Linx BDR/cash valuation and unknown delivery/fraction/lender terms remain explicit limitations. No new model performance or globally complete historical source coverage is claimed.",
        remaining_program="Matched attention parent stopping, common-date C6/GRU and policy comparison, then bounded architecture contrasts under the four-step registration.",
    )
    destination = PROJECT / "docs/v2_scaling_data_inputs.json"
    assert not destination.exists()
    write_json_atomic(destination, result)
    run["scaling_data_inputs"] = binding(destination)
    run["scaling_data_neutral_attribution"] = evidence["neutral_attribution"]
    run["economic_refit_inputs"] = binding(destination)
    write_json_atomic(pointer, run)
    print(
        json.dumps(dict(store=assembly["store"], support=scope, neutral=neutral)),
        flush=True,
    )


if __name__ == "__main__":
    main()
