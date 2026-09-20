"""Bind completed remaining-family evidence without replaying numerical work."""

import json
from pathlib import Path
import subprocess

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    admission = bound_json(run["surviving_rename_admission"])
    source = Path(admission["plan"]["path"]).parent
    out = source / "market_progress"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    m1 = bound_json(run["surviving_rename_m1"])
    market = bound_json(run["surviving_rename_market"])
    consumer = bound_json(run["surviving_rename_market_input_audit"])
    parent = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(parent / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    dates = np.load(parent / "date_index.npy")
    isins = np.load(parent / "isin_index.npy")
    active = np.load(parent / "active.npy").copy()
    with np.load(admission["liquidity_deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    losses = []
    with np.load(market["deltas"]["path"]) as z:
        for family in market["families"]:
            key = "sidecar_" + family
            ix = z[key + "_valid__indices"]
            for t, n, f in ix[~z[key + "_valid__values"]]:
                losses.append(
                    dict(
                        family=family,
                        date=str(dates[t]),
                        isin=str(isins[n]),
                        field=m["feature_names"][key][f],
                        still_active=bool(active[t, n]),
                    )
                )
    assert len(losses) == 398 and not any(r["still_active"] for r in losses)
    write_json_atomic(
        out / "validity_losses.json",
        dict(
            rows=losses,
            interpretation="All398 new market/magnitude losses are retired ARZZ predecessor cells. No additional live loss; retain earlier226live unit-uncertainty and620daily peer losses separately.",
        ),
    )
    snapshots = {}
    for name, path in {
        "registration": PROJECT
        / "research/preregistrations/v2_economic_data_scaling.md",
        "closeout": PROJECT / "docs/v2_STAGE_A_CLOSEOUT.md",
    }.items():
        saved = out / ("planning_" + name + ".md")
        saved.write_bytes(path.read_bytes())
        snapshots[name] = dict(original=binding(path), retained=binding(saved))
    for folder in ["m1", "market"]:
        plan = bound_json(binding(source / folder / "plan.json"))
        assert (
            plan["registration"]["sha256"]
            == snapshots["registration"]["retained"]["sha256"]
        )
        assert plan["closeout"]["sha256"] == snapshots["closeout"]["retained"]["sha256"]
    write_json_atomic(out / "planning_resolution.json", snapshots)
    assert not subprocess.check_output(
        ["git", "diff", "HEAD", "--", "research/src"], cwd=PROJECT
    )
    write_json_atomic(
        out / "runtime_identity.json",
        dict(
            base_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
            ).strip(),
            research_runtime_unchanged=True,
            implementations=[
                binding(PROJECT / "research/src/brazil_rv/v2" / name)
                for name in [
                    "intraday_features.py",
                    "feature_spec.py",
                    "normalization.py",
                    "round5_exposures.py",
                    "round5_magnitude.py",
                    "hedge_beta.py",
                    "targets.py",
                    "data.py",
                    "store.py",
                ]
            ],
        ),
    )
    attempts = {
        "m1": "Native producer/243708controls/119858independent block coordinates passed first invocation20.5725098s. Scalar raw source reconstruction then completed; initial control failed17RRRP valid cells/1624ranked values and227RRRP/SEQL support cells because scratch reducers emitted past the final assigned COTAHIST date. Initial raw/code/stdout/control/corrected outputs preserved. Canonical boundary mask restored using saved reducers only; no source/native replay. Resume .7738721s passes3817836controls. First full invocation wall time was not separately recorded.",
        "market": "Initial magnitude reducers saved once, then old raw control mismatched14stale inactive ALSO cells. Expected raw-family rows had not yet been masked by accepted membership; canonical comparison now does so. Failed14x3comparison/code/stdout retained. Resume reuses reducers; all6control/corrected exposure regressions run once. OriginalFloat32 family cast, oil/non-oil and issuer shrinkage contracts preserved.259954326typed controls and67862628raw comparisons pass. Qualified producer161.978approximately; manifest records exact duration. Empty-history nanmean warning is expected missing-support arithmetic, not fabricated support.",
        "consumer": "First invocation877actualfull933/full60 consumers passed51.1102855s. New-family sample319340926cells/collated638681852comparisons; independent262269406typed/age cells and6432to-close outcomes; actual post-decision native mutation, scalar future prefix and future-feature-read guards. Slow history reused, not recounted. No repeated earlier185/303/124views.",
        "tooling": "Read-only wrong native_fast.py/natura-input filenames, Windows rg wildcard failures and truncated large tool output are reconstructed prose, not saved exact failed shell files. Pre-launch admission delta-key inspection prevented a bad input lookup before any producer. No raw/source/store/fit bytes changed.",
        "limits": "No original source census, new source retrieval, account book, neural forward, forecast scoring or fit. No accepted-store/oldfit mutation or new loan/option alias. Final primary-target/new-store composition, January25disposition and integratedStageA remain; C/Dunstarted.",
    }
    attempts["market"] = attempts["market"].replace(
        "161.978approximately", str(market["seconds"]) + "s"
    )
    write_json_atomic(out / "attempts.json", attempts)
    native = bound_json(m1["native"])
    native_stats = {
        k: sum(c[k] for c in native["cases"])
        for k in [
            "rows",
            "control_cells",
            "independent_block_cells",
            "changes",
            "gains",
            "losses",
        ]
    }
    report = dict(
        status="remaining_feature_dependencies_qualified_within_open_composition_block",
        parent=admission["parent"],
        evidence={
            k: run[k]
            for k in [
                "surviving_rename_m1",
                "surviving_rename_market",
                "surviving_rename_market_input_audit",
            ]
        },
        native=native_stats,
        scalar={
            k: m1[k]
            for k in [
                "effects",
                "scalar_gains",
                "scalar_losses",
                "target_gains",
                "target_losses",
                "also_eligible_without_m1",
            ]
        },
        market=market["families"],
        consumer={
            k: consumer[k]
            for k in [
                "samples",
                "sample_cells",
                "sample_and_collated_comparisons",
                "independent_typed_formula_cells",
                "independent_to_close_outcomes",
                "seconds",
            ]
        },
        losses=binding(out / "validity_losses.json"),
        attempts=binding(out / "attempts.json"),
        planning=binding(out / "planning_resolution.json"),
        prior_recovery_dependencies={
            k: run[k]
            for k in [
                "surviving_rename_dependency_recovery",
                "surviving_rename_recovery",
                "surviving_rename_lineage_recovery",
                "event_composition_recovery",
                "natura_store_recovery",
                "economic_store_recovery",
                "m1_scalar_recovery",
                "rename_m1_recovery",
            ]
        },
        remaining=[
            "Compose final new eligibility/same-class primary targets and existing72gross outcomes with final risk/slow coordinates",
            "New complete accepted store preserving earlier2; no old-fit coordinate substitution",
            "January25 2017 distinct delivery/cash/rent disposition",
            "IntegratedStageA, then registered C and conditional D",
        ],
    )
    write_json_atomic(out / "manifest.json", report)
    run["surviving_rename_market_progress"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                native=native_stats,
                scalar=report["scalar"],
                losses=len(losses),
                consumer=report["consumer"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
