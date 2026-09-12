"""Bind repaired economics and replay sealed S0 signals before new alpha fits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .execution_policy import ledger_gate_failures
from .hedge_beta import build_hedge_beta_sidecar
from .round6_costs import headline_ledger
from .round6_readouts import evaluation_design
from .round7_data import PROJECT, registered_sources
from .round5_derived import bind
from .train import rank_average_ensemble, _cli_stage_indices


def source_run_root():
    pointer = json.loads(
        (PROJECT / "docs/v2_model_design_checkpoint_audit.json").read_text(
            encoding="utf-8"
        )
    )
    return Path(pointer["source_root"])


def prepare(audit_root):
    repair = json.loads((audit_root / "store_repair.json").read_text(encoding="utf-8"))
    source_root = source_run_root()
    old = evaluation_design(rr._read_json(source_root / "frozen_design.json"))
    design = copy.deepcopy(old)
    design["store"] = repair["store"]
    beta_root = audit_root / "hedge_beta"
    if not (beta_root / "manifest.json").exists():
        build_hedge_beta_sidecar(
            store_root=Path(design["store"]["root"]),
            expected_store_manifest_sha256=design["store"]["manifest_sha256"],
            bova11_root=Path(design["bova11"]["root"]),
            expected_bova11_manifest_sha256=design["bova11"]["manifest_sha256"],
            output_root=beta_root,
        )
    beta = rr._read_json(beta_root / "manifest.json")
    old_beta = rr._read_json(Path(old["bova11"]["hedge_beta_root"]) / "manifest.json")
    for name, record in beta["arrays"].items():
        if record["sha256"] != old_beta["arrays"][name]["sha256"]:
            raise ValueError("protected causal economic beta changed")
    design["bova11"].update(
        hedge_beta_root=str(beta_root),
        hedge_beta_manifest_sha256=sha256_file(beta_root / "manifest.json"),
    )
    write_json_atomic(audit_root / "evaluation_design.json", design)
    return old, design, source_root


def run(audit_root, folds):
    old_design, new_design, source_root = prepare(audit_root)
    _, source_manifest, _, _ = registered_sources()
    contexts = [rr._open_ledger_replay(d) for d in (old_design, new_design)]
    policy, _ = rr.load_selected_policy(
        Path(old_design["execution_policy"]["root"]),
        expected_result_sha256=old_design["execution_policy"]["result_sha256"],
    )
    output = audit_root / "sealed_s0_replay"
    output.mkdir(exist_ok=True)
    try:
        for fold in folds:
            destination = output / f"{fold}.json"
            if destination.exists():
                continue
            indices = contexts[0].evaluation[fold]
            members, reference_mask, sources = [], None, {}
            for seed in (11, 29, 47):
                directory = (
                    source_root / "trajectories/S0" / f"{fold}_seed_{seed}" / "scores"
                )
                scores, mask = rr._score_artifact(
                    directory,
                    require_clean_transfer=True,
                    expected_dates=contexts[0].store.dates[indices],
                    expected_isins=contexts[0].store.isins,
                    expected_feature_schema_sha256=source_manifest[
                        "feature_schema_sha256"
                    ],
                )
                if reference_mask is not None and not np.array_equal(
                    mask, reference_mask
                ):
                    raise ValueError("sealed S0 seed masks differ")
                reference_mask = mask
                members.append(scores)
                sources[str(seed)] = sha256_file(directory / "score_manifest.json")
            scores = rank_average_ensemble(members, reference_mask)
            books, records = [], []
            for context in contexts:
                inputs = rr._evaluation_inputs(
                    context.store,
                    indices,
                    scores,
                    reference_mask,
                    context.cdi,
                    context.bova11.close_by_session,
                    context.bova11_binding,
                    context.lending_borrow,
                    {"sealed_s0_scores_" + k: v for k, v in sources.items()},
                    transfer_chronology_clean=True,
                    execution_policy=policy,
                )
                result, summary, daily = headline_ledger(inputs)
                books.append(result)
                records.append(
                    {
                        "summary": summary,
                        "daily": daily,
                        "gate_failures": ledger_gate_failures(summary, headline=True),
                    }
                )
            recovered = contexts[1].store.read("continuation_quote_mask", indices)
            affected = recovered & (
                (books[0].signed_shares != 0.0) | (books[1].signed_shares != 0.0)
            )
            # Include positions closed on a recovered print, whose closing shares are zero.
            for result in books:
                prior = np.vstack(
                    (
                        np.zeros((1, result.signed_shares.shape[1])),
                        result.signed_shares[:-1],
                    )
                )
                affected |= recovered & (prior != 0.0)
            positions = [
                {
                    "date": str(contexts[1].store.dates[indices[d]]),
                    "isin": contexts[1].store.isins[n],
                    "old_shares_end": float(books[0].signed_shares[d, n]),
                    "new_shares_end": float(books[1].signed_shares[d, n]),
                    "old_mark": float(books[0].mark_price[d, n])
                    if np.isfinite(books[0].mark_price[d, n])
                    else None,
                    "new_mark": float(books[1].mark_price[d, n])
                    if np.isfinite(books[1].mark_price[d, n])
                    else None,
                }
                for d, n in np.argwhere(affected)
            ]
            report = {
                "schema": "BRAZIL_RV_ROUND7_SEALED_SIGNAL_REPLAY_V1",
                "fold": fold,
                "fixed_signal_source_sha256": sources,
                "before_store": old_design["store"],
                "after_store": new_design["store"],
                "before": records[0],
                "after": records[1],
                "continued_print_held_positions": positions,
                "signal_retrained": False,
            }
            write_json_atomic(destination, report)
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "affected_position_days": len(positions),
                        "before_gates": records[0]["gate_failures"],
                        "after_gates": records[1]["gate_failures"],
                    }
                ),
                flush=True,
            )
    finally:
        for context in contexts:
            context.store.close()


def accept(audit_root):
    """Publish input acceptance only after matching every original sealed ledger."""
    repair = rr._read_json(audit_root / "store_repair.json")
    source_root = source_run_root()
    reports, before_daily, after_daily = [], [], []
    for index in range(1, 15):
        path = audit_root / "sealed_s0_replay" / f"F{index}.json"
        replay = rr._read_json(path)
        reference = source_root / "costs/session1_S0/S0" / f"F{index}.json"
        original = rr._read_json(reference)["scenarios"]["original"]
        if any(replay["before"][k] != original[k] for k in ("daily", "summary")):
            raise ValueError("original-accounting replay differs from sealed S0")
        if replay["before"]["gate_failures"] or replay["after"]["gate_failures"]:
            raise ValueError("sealed-score replay has an unresolved engineering gate")
        before_daily.extend(
            r["net_excess_all_cash_bps"] for r in replay["before"]["daily"]
        )
        after_daily.extend(
            r["net_excess_all_cash_bps"] for r in replay["after"]["daily"]
        )
        reports.append(
            {
                "fold": f"F{index}",
                "replay": bind(path),
                "sealed_reference": bind(reference),
                "original_daily_and_summary_exact": True,
                "affected_continued_print_positions": replay[
                    "continued_print_held_positions"
                ],
                "before_net_bps_per_day": replay["before"]["summary"][
                    "mean_net_excess_bps_per_day"
                ],
                "after_net_bps_per_day": replay["after"]["summary"][
                    "mean_net_excess_bps_per_day"
                ],
            }
        )
    if (
        not repair["protected_arrays_exact"]
        or not repair["existing_C1_terms_exact"]
        or repair["peak_rss_bytes"] > 8 * 1024**3
    ):
        raise ValueError("repaired store did not satisfy the protected-data contract")
    manifest = rr._read_json(Path(repair["store"]["root"]) / "manifest.json")
    store_root = Path(repair["store"]["root"])
    fit, selection, _, _ = _cli_stage_indices(store_root, "P", "pretrain_internal")
    active = np.load(store_root / manifest["arrays"]["active"]["path"], mmap_mode="r")
    pretrain_coverage = {}
    for family, fields in manifest["feature_names"].items():
        if not family.startswith("sidecar_"):
            continue
        validity = np.load(
            store_root / manifest["arrays"][family + "_valid"]["path"], mmap_mode="r"
        )
        pretrain_coverage[family] = {}
        for label, rows in (("fit", fit), ("selection", selection)):
            observed = validity[rows] & active[rows, :, None]
            pretrain_coverage[family][label] = {
                "active_stock_days": int(active[rows].sum()),
                "any_valid_stock_days": int(observed.any(axis=-1).sum()),
                "per_field_valid_stock_days": dict(
                    zip(
                        fields,
                        observed.sum(axis=(0, 1)).astype(int).tolist(),
                        strict=True,
                    )
                ),
            }
    result = {
        "schema": "BRAZIL_RV_ROUND7_INPUT_ACCEPTANCE_V1",
        "status": "accepted_cpu_preflight",
        "store": repair["store"],
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "evaluation_design": bind(audit_root / "evaluation_design.json"),
        "repair_evidence": bind(audit_root / "store_repair.json"),
        "source_audit": bind(audit_root / "manifest.json"),
        "family_coverage": bind(audit_root / "family_coverage.json"),
        "oddlot_coverage": bind(audit_root / "oddlot_coverage.json"),
        "native_support": json.loads(
            (audit_root / "native_support.json").read_text(encoding="utf-8")
        ),
        "pretraining_input_coverage": pretrain_coverage,
        "family_admission": "all registered families retained; absent observations masked, no minimum family-support gate",
        "foreign_clock": bind(audit_root / "foreign_clock/manifest.json"),
        "protected_arrays_exact": True,
        "existing_C1_terms_exact": True,
        "peak_rss_bytes": repair["peak_rss_bytes"],
        "target_changes_by_fold": repair["target_changes_by_fold"],
        "sealed_s0_replays": reports,
        "fixed_signal_economics": {
            "before_mean_net_bps_per_day": float(np.mean(before_daily)),
            "after_mean_net_bps_per_day": float(np.mean(after_daily)),
            "paired_delta_bps_per_day": float(
                np.mean(np.asarray(after_daily) - before_daily)
            ),
            "interpretation": "accounting repair only; unchanged sealed signals; not a new model comparison",
        },
        "source_limitations": [
            "action terms remain inferred; nearby unit-event metadata corroborates a candidate, not exact contractual terms",
            "retrospective labels and accounting repaired; historical decision-time wealth features remain protected",
            "foreign-flow earlier publication could not be established; current receipt bound retained",
            "other ranked families retain their registered transformations; coverage losses are disclosed for a separate data-representation comparison",
        ],
        "forward_capture": False,
        "heldout_access": False,
        "new_model_scores": False,
    }
    write_json_atomic(PROJECT / "docs/v2_round7_inputs.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", default=[f"F{i}" for i in range(1, 15)])
    parser.add_argument("--accept", action="store_true")
    args = parser.parse_args()
    if args.accept:
        result = accept(args.audit_root)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "store": result["store"],
                    "economics": result["fixed_signal_economics"],
                }
            )
        )
    else:
        run(args.audit_root, args.folds)


if __name__ == "__main__":
    main()
