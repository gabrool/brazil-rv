"""Bind the saved dependency evidence without replaying passed calculations."""

from collections import Counter
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
    out = source / "dependency_progress"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    paths = {
        "surviving_rename_issuers": "issuers/manifest.json",
        "surviving_rename_context": "context/manifest.json",
        "surviving_rename_lending": "lending/manifest.json",
        "surviving_rename_unit_history": "unit_history/manifest.json",
        "surviving_rename_dependency_input_audit": "dependency_consumer/manifest.json",
        "surviving_rename_auxiliaries": "auxiliaries/manifest.json",
        "surviving_rename_auxiliary_arithmetic": "auxiliaries/qualification/manifest.json",
        "surviving_rename_auxiliary_input_audit": "auxiliary_consumer/manifest.json",
    }
    evidence = {k: binding(source / v) for k, v in paths.items()}
    snapshots = {}
    for name, path in {
        "registration": PROJECT
        / "research/preregistrations/v2_economic_data_scaling.md",
        "closeout": PROJECT / "docs/v2_STAGE_A_CLOSEOUT.md",
    }.items():
        target = out / ("planning_" + name + ".md")
        target.write_bytes(path.read_bytes())
        snapshots[name] = dict(original=binding(path), retained=binding(target))
    for family in ("issuers", "auxiliaries"):
        assert (
            bound_json(binding(source / family / "plan.json"))["registration"]["sha256"]
            == snapshots["registration"]["retained"]["sha256"]
        )
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
            unchanged_runtime=[
                binding(p) for p in sorted((PROJECT / "research/src").rglob("*.py"))
            ],
            meaning="No production runtime edit. New ops reuse prior qualified algorithms. Exact initial/resumed executed recipes bind each saved result; current consumer adds the separately executed auxiliary mode after the issuer mode passed.",
        ),
    )
    parent = Path(admission["parent"]["root"])
    m = bound_json(
        dict(
            path=str(parent / "manifest.json"),
            sha256=admission["parent"]["manifest_sha256"],
        )
    )
    dates, isins = (
        np.load(parent / "date_index.npy"),
        np.load(parent / "isin_index.npy"),
    )
    active = np.load(parent / "active.npy").copy()
    with np.load(admission["liquidity_deltas"]["path"]) as z:
        active[tuple(z["active__indices"].T)] = z["active__values"]
    losses, family_stats, consumers = [], {}, []
    for key in (
        "surviving_rename_dependency_input_audit",
        "surviving_rename_auxiliary_input_audit",
    ):
        consumer = bound_json(evidence[key])
        consumers.append(
            {
                k: consumer[k]
                for k in (
                    "samples",
                    "sample_cells",
                    "sample_and_collated_comparisons",
                    "seconds",
                    "independent_transformed_cells",
                )
                if k in consumer
            }
        )
        with np.load(consumer["deltas"]["path"]) as z:
            for group, stats in consumer["families"].items():
                family = "sidecar_" + group
                names = m["feature_names"][family]
                ix = z[family + "_valid__indices"]
                supported = z[family + "_valid__values"]
                live_losses = 0
                for t, n, f in ix[~supported]:
                    live = bool(active[t, n])
                    if live:
                        assert (
                            group == "fundamentals"
                            and names[f]
                            in (
                                "log_market_cap",
                                "book_to_market",
                                "earnings_yield_ttm",
                            )
                        ) or (group == "lending" and names[f] == "utilization_proxy")
                    losses.append(
                        dict(
                            group=group,
                            date=str(dates[t]),
                            isin=str(isins[n]),
                            field=names[f],
                            active_after=live,
                        )
                    )
                    live_losses += live
                values = z[family + "_values__indices"]
                original_mask = np.load(parent / (family + "_valid.npy"), mmap_mode="r")
                new_mask = original_mask.copy()
                new_mask[tuple(ix.T)] = supported
                shared = original_mask[tuple(values.T)] & new_mask[tuple(values.T)]
                family_stats[group] = {
                    **stats,
                    "shared_valid_numeric_changes": int(shared.sum()),
                    "still_active_losses": live_losses,
                    "retired_losses": stats["lost"] - live_losses,
                }
    write_json_atomic(
        out / "validity_losses.json",
        dict(
            rows=losses,
            summary=dict(
                Counter(
                    (r["group"] + ("_live" if r["active_after"] else "_retired"))
                    for r in losses
                )
            ),
            interpretation="Live losses are source-selected capital/free-float intervals crossing existing inherited DISMES uncertainty; remaining losses are retired predecessor eligibility. No threshold relaxation or name deletion. Raw222valuation-field losses/74dates are not222new store losses; only shared previously supported eligible masks count here.",
        ),
    )
    attempts = dict(
        issuers="First invocation passed20.0149454s; own-version caches only, two legal issuers/5ISINs, no new source extraction. Exact executed recipe/CVM module/stdout retained.",
        context="Initial invocation saved common-state outputs then failed old sector alignment outside the bounded date window. Initial code/stdout retained. Qualified resume reuses common outputs and performs sector reducers once,25.667431s; initial wall time not separately recorded.",
        lending="First invocation passed2.388231s, qualified union/rate/float subsets reused. Economic rate/availability archive unchanged; no loan aliases.",
        unit_history="First invocation22.6558876s passed independent selected-document/FRE and ancestor interval arithmetic on every181utilization/74financial loss date. Four existing DISMES transitions, no newly proved split or erroneous capital.",
        dependency_consumer="First invocation303full933/full60 samples passed21.7621551s. New sidecar/common comparisons only; old slow history reused and not recounted as a new proof.",
        auxiliary_producer="First invocation4.5317548s passed. Two new bounded activity/oddlot windows and necessary index identity/ADV composition, no original census. 2019IN/PR source observations absent and remain unsupported.",
        auxiliary_consumer="First invocation124full933/full60 samples passed9.2228893s. Four new families only; independent typed transforms, no repeated issuer/slow/native checks.",
        auxiliary_arithmetic="Initial executed qualifier passed414activity/2312oddlot comparisons then failed first index comparison due to Float64 oracle versus original NumPy2 Float32 scalar division. Failed code/stdout retained. Counts recovered from saved outputs plus demonstrated execution boundary, explicitly labelled reconstructed, not a new arithmetic invocation. Qualified resume reuses them, checks22changed/new index rows at original Float32 scalar boundary7.753859s. Source/producer/consumer outputs unchanged. Initial full wall time unknown. A local event-date variable shadow was also corrected in the resumed oracle; no producer changed.",
        readonly_tooling="Read-only guessed ops filenames/Windows rg wildcard and truncated output failures are reconstructed prose, not retained exact failed shell files. No source/store/data result changed. Pre-execution Ruff unused names were fixed before producer invocation.",
        invariants="No neural forward, model scoring, fit, account replay, source census, accepted store/fit overwrite or heldout market consumer. No repeated V31-V41/cost/daily/72endpoint campaigns.",
    )
    write_json_atomic(out / "attempts.json", attempts)
    common = bound_json(evidence["surviving_rename_context"])
    progress = dict(
        status="qualified_dependencies_within_open_source_composition_block",
        evidence=evidence,
        parent=admission["parent"],
        families=family_stats,
        consumers=consumers,
        controls=sum(v["control_cells"] for v in family_stats.values()),
        common_effects={
            k: v for k, v in common["effects"].items() if k.startswith("common_state")
        },
        losses=binding(out / "validity_losses.json"),
        attempts=binding(out / "attempts.json"),
        planning=binding(out / "planning_resolution.json"),
        runtime=binding(out / "runtime_identity.json"),
        prior_recovery_dependencies={
            k: run[k]
            for k in (
                "surviving_rename_recovery",
                "surviving_rename_lineage_recovery",
                "event_composition_recovery",
                "natura_store_recovery",
                "economic_store_recovery",
                "lending_feature_recovery",
                "remaining_auxiliary_recovery",
            )
        },
        next=[
            "Native M1/scalar and affected magnitude/cross-market dependencies, preserving original source boundaries and full60history.",
            "Final115eligibility/same-class target effects plus ready72succession gross outcomes under final risks; new complete accepted store preserving both previous stores.",
            "January25 separate delivery/cash/rent disposition, integratedStageA admission, then registeredC/D.",
        ],
        limits="Intermediate sparse dependencies, not completed composition block/StageA or model profitability. Raw/family/transformed counts overlap and are not additive independent observations. Both accepted stores/oldfits remain sealed.",
    )
    write_json_atomic(out / "manifest.json", progress)
    run.update(evidence)
    run["surviving_rename_dependency_progress"] = binding(out / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: progress[k]
                for k in (
                    "status",
                    "families",
                    "consumers",
                    "controls",
                    "common_effects",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
