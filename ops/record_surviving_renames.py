"""Record qualified dependency progress within the existing StageA closeout."""

import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text(encoding="utf8"))
    admission = bound_json(run["surviving_rename_admission"])
    daily = bound_json(run["surviving_rename_daily_qualification"])
    consumer = bound_json(run["surviving_rename_daily_input_audit"])
    out = Path(admission["plan"]["path"]).parent
    (out / "executed_record.py").write_bytes(Path(__file__).read_bytes())
    plan = bound_json(admission["plan"])
    snapshots = {}
    for key in ("registration", "closeout"):
        rec = plan[key]
        data = Path(rec["path"]).read_bytes()
        path = out / ("planning_" + key + ".md")
        path.write_bytes(data)
        assert binding(path)["sha256"] == rec["sha256"]
        snapshots[key] = {"original": rec, "retained": binding(path)}
    write_json_atomic(out / "planning_resolution.json", snapshots)
    root = Path(admission["parent"]["root"])
    dates, isins = np.load(root / "date_index.npy"), np.load(root / "isin_index.npy")
    active = np.load(root / "active.npy").copy()
    with np.load(admission["liquidity_deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    with np.load(daily["deltas"]["path"]) as z:
        ix = z["slow_valid__indices"][~z["slow_valid__values"]]
    losses = [
        dict(
            date=str(dates[t]),
            isin=str(isins[n]),
            field=int(f),
            active_after=bool(active[t, n]),
        )
        for t, n, f in ix
    ]
    assert all(not row["active_after"] or row["field"] >= 27 for row in losses)
    write_json_atomic(
        out / "validity_losses.json",
        {
            "rows": losses,
            "active_after": sum(x["active_after"] for x in losses),
            "disposition": "Retired predecessor cells or unchanged minimum-peer support under the new causal membership/clusters. Every loss retained and enumerated; no threshold relaxed or name excluded.",
        },
    )
    attempts = {
        "admission": "First invocation passed; old3link terms preserved, new2typed source joins,244liquidity control cells and delayed-knowledge checks. Existing700-row inventory and source PDFs reused, no download or census.",
        "wealth": "First invocation passed;7250standalone successor recurrence cells exact,2694normalized source rows selected. New two-chain recurrence and prior-close transition proof; oldstores/fits unchanged.",
        "daily_producer": "First invocation completed both cases and all30computed daily fields/clusters in123.6854401s. Exact executed code/features and all saved reducers retained. Expected NumPy empty/missing-window warnings are preserved in tool transcript, not a separately saved exact producer stdout. No repeated reducer campaign.",
        "initial_qualification": "Exact initial code, manifest and stdout retained in daily/qualification. Source masks used accepted raw-price availability instead of original normalized daily feature rows. Failed controls retained; no model/store was admitted from them.",
        "mask_qualification": "Exact code/manifest/stdout retained in daily/qualified. Source-row gap/ambiguity masks restore all price/wealth/risk/peer controls using saved reducers. Four volume/trade feature controls still fail because missing original normalized security rows must use the original complete-session zero-activity convention.",
        "activity_qualification": "Only fields17/18/20/21 recalculated,5.358068s. Every other successful control, wealth/book/source calculation reused. All94 finalchecks/83150470cells pass, including relevant sigma and clusters; unrelated inactive raw-sigma scratch differences remain enumerated and preserved.",
        "consumer": "First invocation185full933/full60actualdataset/collator samples,2009467410packed cells and future-read guard pass. This is slow-only, not a repeated native/target campaign or full-store acceptance.",
        "readonly_tooling": "Initial combined output truncated; a nonexistent source JSON/ops filename and a Windows rg wildcard failed read-only. These descriptions reconstruct the tool history; exact failed shell bytes were not separately retained. No source/store or model changed.",
        "runtime": "Research runtime unchanged. Canonical daily reproducer now restores original source masks/zero activity; initial executed bytes and saved qualification resolution are explicit. New ops are formatted/linted with Ruff.",
    }
    write_json_atomic(out / "attempts.json", attempts)
    totals = {
        k: sum(e[k] for e in daily["effects"])
        for k in ("values", "valid", "ages", "gains", "losses")
    }
    report = {
        "status": "qualified_identity_wealth_daily_progress_within_open_composition_block",
        "evidence": {
            k: run[k]
            for k in (
                "surviving_rename_admission",
                "surviving_rename_wealth",
                "surviving_rename_daily",
                "surviving_rename_daily_qualification",
                "surviving_rename_daily_input_audit",
            )
        },
        "parent": admission["parent"],
        "slow_effects": totals,
        "losses": binding(out / "validity_losses.json"),
        "attempts": binding(out / "attempts.json"),
        "checks": len(daily["checks"]),
        "control_cells": sum(c["cells"] for c in daily["checks"]),
        "consumer_cells": consumer["packed_cells"],
        "next": [
            "Reuse current saved daily deltas/controls; complete common state, issuer/financial and lending-denominator, native/scalar and auxiliary dependencies only where changed.",
            "Recompose72ready succession endpoints with final risks and new identity/eligibility target effects into a NEW complete store, preserving both accepted stores.",
            "January25 2017 distinct delivery/cash/rent disposition, integrated StageA admission, then registered C/D.",
        ],
        "prior_recovery_dependencies": {
            k: run[k]
            for k in (
                "event_composition_recovery",
                "natura_store_recovery",
                "economic_store_recovery",
                "held_event_source_recovery",
                "lending_feature_recovery",
            )
        },
        "not_claimed": [
            "StageA acceptance",
            "complete corporate/calendar/source block",
            "complete store",
            "corrected model profit or GPU fit",
            "loan alias/custody permission",
            "held-Cielo bound",
            "completion percentage",
        ],
    }
    write_json_atomic(out / "progress.json", report)
    run["surviving_rename_progress"] = binding(out / "progress.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                "totals": totals,
                "still_active_lost_cells": sum(x["active_after"] for x in losses),
                "progress": run["surviving_rename_progress"],
            }
        )
    )


if __name__ == "__main__":
    main()
