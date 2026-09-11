"""Round-6 actual adopted-policy ledgers under registered borrow-rate scenarios.

The original multi-scenario evaluation remains authoritative. These replays change
only borrow pricing and rerun the identical adopted-policy ledger, including NAV,
orders and costs. They do not recompute predictions or score statistics.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from brazil_rv.v2 import evaluate as ev
from brazil_rv.v2 import research_rounds as rr
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS
from brazil_rv.v2.execution_policy import ledger_gate_failures
from brazil_rv.v2.round6_readouts import ensemble, evaluation_design
from brazil_rv.execution.stateful_ledger import simulate_stateful_ledger


def headline_ledger(inputs):
    policy = inputs.execution_policy
    score, mask = ev.traded_signal(inputs, policy)
    arguments = ev._ledger_inputs(inputs, score, mask)
    arguments["capacity_buffer_per_side"] = 30
    if policy.inverse_volatility:
        arguments["entry_sizing_volatility"] = inputs.target_scale_sigma
    config = replace(policy.ledger_config(), settle_terminal_residuals=True)
    result = simulate_stateful_ledger(
        **arguments,
        config=config,
        shortable=inputs.shortable_by_borrow_source["borrow_balance"],
    )
    summary = {
        k: ev._finite_or_none(v) if isinstance(v, float) else v
        for k, v in result.summary().items()
    }
    daily = ev._ledger_rows(
        result,
        cost_bps=config.cost_bps_per_side,
        annual_borrow_rate=config.annual_borrow_rate,
    )
    return result, summary, daily


def load_rates(root, context):
    manifest = rr._read_json(root / "manifest.json")
    result = {}
    for label in ("original", "placeholder_v2", "latest_vintage"):
        path = root / f"{label}.npz"
        assert sha256_file(path) == manifest["files"][path.name]
        with np.load(path, allow_pickle=False) as archive:
            result[label] = {k: archive[k] for k in archive.files}
        # Latest-rate observation support can mechanically change the alignment
        # helper's availability arrays. A rate-only sensitivity must not use them.
        if label != "latest_vintage":
            for k in ("shortable_strict", "shortable_balance", "shortable_open"):
                np.testing.assert_array_equal(
                    result[label][k], getattr(context.lending_borrow, k)
                )
    for k in ("annual_taker_rate", "rate_imputed", "rate_placeholder"):
        np.testing.assert_array_equal(
            result["original"][k], getattr(context.lending_borrow, k)
        )
    manifest["latest_availability_differences_not_consumed"] = {
        k: int(
            np.count_nonzero(
                result["latest_vintage"][k] != getattr(context.lending_borrow, k)
            )
        )
        for k in ("shortable_strict", "shortable_balance", "shortable_open")
    }
    return result, manifest


def run(root, output, borrow_root, arms, folds, reference=None, *, seeds=ALLOWED_SEEDS):
    design = rr._read_json(root / "frozen_design.json")
    original = evaluation_design(design)
    context = rr._open_ledger_replay(original)
    policy, _ = rr.load_selected_policy(
        Path(original["execution_policy"]["root"]),
        expected_result_sha256=original["execution_policy"]["result_sha256"],
    )
    roster = (
        rr._read_json(root / "session2_roster.json")
        if (root / "session2_roster.json").exists()
        else None
    )
    panels, borrow_manifest = load_rates(borrow_root, context)
    started = time.monotonic()
    records = {a: {} for a in arms}
    try:
        for fold in folds:
            ix = context.evaluation[fold]
            for arm in arms:
                path = output / arm / f"{fold}.json"
                scores, mask, source = ensemble(
                    design,
                    root,
                    arm,
                    fold,
                    seeds,
                    context.store.dates[ix],
                    context.store.isins,
                    roster=roster,
                )
                if path.exists():
                    saved = rr._read_json(path)
                    assert saved["design_sha256"] == sha256_file(
                        root / "frozen_design.json"
                    ) and saved["borrow_manifest_sha256"] == sha256_file(
                        borrow_root / "manifest.json"
                    )
                    assert saved["seeds"] == list(seeds)
                    assert saved["source_score_manifests"] == source
                    records[arm][fold] = saved
                    continue
                inputs = rr._evaluation_inputs(
                    context.store,
                    ix,
                    scores,
                    mask,
                    context.cdi,
                    context.bova11.close_by_session,
                    context.bova11_binding,
                    context.lending_borrow,
                    {"round6_training_store": design["store"]["manifest_sha256"]},
                    transfer_chronology_clean=True,
                    execution_policy=policy,
                )
                ev._validate(inputs)
                scenarios = {}
                for label, arrays in panels.items():
                    changed = replace(
                        inputs,
                        annual_borrow_rate_by_name=arrays["annual_taker_rate"][ix],
                        borrow_rate_imputed=arrays["rate_imputed"][ix],
                        borrow_rate_placeholder=arrays["rate_placeholder"][ix],
                        borrow_source_label=label,
                    )
                    if label == "original":
                        for field in (
                            "annual_borrow_rate_by_name",
                            "borrow_rate_imputed",
                            "borrow_rate_placeholder",
                        ):
                            np.testing.assert_array_equal(
                                getattr(inputs, field), getattr(changed, field)
                            )
                    result, summary, daily = headline_ledger(changed)
                    if label == "original" and reference is not None:
                        prior = rr._read_json(
                            reference / arm / fold / "evaluation.json"
                        )["economics"]
                        for k, v in summary.items():
                            assert v == prior["headline"][k], (
                                arm,
                                fold,
                                k,
                                v,
                                prior["headline"][k],
                            )
                        old = prior["headline_audit"]["daily_state"]
                        assert len(old) == len(daily)
                        for before, after in zip(old, daily, strict=True):
                            assert before.keys() == after.keys()
                            for k in before:
                                if isinstance(before[k], float) and isinstance(
                                    after[k], float
                                ):
                                    difference = abs(before[k] - after[k])
                                    assert difference <= 1e-12, (k, difference)
                                else:
                                    assert before[k] == after[k], k
                    scenarios[label] = {
                        "summary": summary,
                        "daily": daily,
                        "gates": ledger_gate_failures(summary, headline=True),
                        "promotion_weight": 1 if label == "original" else 0,
                        "fills": ev._serialise_records(result.fills),
                        "date_count": len(result.dates),
                    }
                    del result
                saved = {
                    "schema": "ROUND6_ADOPTED_POLICY_BORROW_REPLAY_V1",
                    "arm": arm,
                    "fold": fold,
                    "seeds": list(seeds),
                    "design_sha256": sha256_file(root / "frozen_design.json"),
                    "borrow_manifest_sha256": sha256_file(
                        borrow_root / "manifest.json"
                    ),
                    "source_score_manifests": source,
                    "scenarios": scenarios,
                    "unchanged_shortability": True,
                    "latest_availability_differences_not_consumed": borrow_manifest[
                        "latest_availability_differences_not_consumed"
                    ],
                    "reference_original_headline_exact": reference is not None,
                    "new_predictions_or_fits": False,
                }
                write_json_atomic(path, saved)
                records[arm][fold] = saved
                print(f"accepted rate ledgers {arm}/{fold}", flush=True)
        summary = {}
        for arm, by_fold in records.items():
            summary[arm] = {}
            for label in panels:
                fields = {
                    key: []
                    for key in (
                        "net_excess_all_cash_bps",
                        "turnover_fraction_nav",
                        "borrow_cost_bps",
                    )
                }
                for fold, saved in by_fold.items():
                    for key in fields:
                        fields[key].append(
                            np.array(
                                [
                                    row[key]
                                    for row in saved["scenarios"][label]["daily"]
                                ],
                                dtype=float,
                            )
                        )
                summary[arm][label] = {
                    k: rr._folded_bootstrap(tuple(v)) for k, v in fields.items()
                }
        return write_json_atomic(
            output / "summary.json",
            {
                "schema": "ROUND6_BORROW_SENSITIVITY_SUMMARY_V1",
                "arms": summary,
                "folds": folds,
                "seeds": list(seeds),
                "wall_seconds": time.monotonic() - started,
                "rule": "Exact adopted-policy ledger for each rate scenario; fixed predictions/populations/shortability; all calendar sessions; hindsight/latest-vintage have zero promotion weight.",
            },
        )
    finally:
        context.store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--borrow-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["S0"])
    parser.add_argument("--folds", nargs="+", default=list(DEVELOPMENT_FOLDS))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(ALLOWED_SEEDS))
    a = parser.parse_args()
    print(
        run(
            a.root, a.output, a.borrow_root, a.arms, a.folds, a.reference, seeds=a.seeds
        )
    )
