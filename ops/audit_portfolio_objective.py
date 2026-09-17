"""Historical gradient/causality admission and source-bound closeout export."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch

from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.objective_readouts import calibration
from brazil_rv.v2.opportunity_research import bound, checked
from brazil_rv.v2.portfolio_objective import (
    TensorPreference,
    clone_account,
    utility_path,
)
from brazil_rv.v2.portfolio_program import PROJECT, read
from brazil_rv.v2.portfolio_training import load_data
from brazil_rv.v2.research_rounds import _git_identity


def audit_bridge(root):
    from brazil_rv.v2.portfolio_inputs import normalized_ranks
    from brazil_rv.v2.portfolio_objective_training import encoded, preparation
    from brazil_rv.v2.train import compile_forward, rank_average_ensemble

    torch.set_num_threads(1)
    design = read(root / "training_design.json")
    records = {}
    for arm in ("TE_all", "C6"):
        data, _ = load_data(Path(design["decision_root"]), arm)
        model, caches, axes, view, _ = preparation(design, data, arm, "F2", 11)
        model.eval()
        compiled = compile_forward(model)
        positions = np.arange(64)
        rows = axes["fit"][positions]
        with torch.no_grad():
            scores, soft, _ = encoded(
                compiled, caches["fit"], positions, len(data.inputs.security_ids)
            )
        active = caches["fit"].tensors["active_mask"][positions].cpu().numpy()
        mask = np.repeat(active[..., None], 3, -1)
        ranks = normalized_ranks(
            rank_average_ensemble([scores.cpu().numpy()], mask), mask
        )
        m = calibration(read(checked(design["mappings"]["F2"]))["arms"][arm])
        compact = (
            (ranks.mean(-1) - m.mean[0]) / m.scale[0] * m.coefficient[0] + m.intercept
        ) * active
        from brazil_rv.v2.data import restore_name_axis

        hard = restore_name_axis(
            compact,
            caches["fit"].names[positions].cpu().numpy(),
            len(data.inputs.security_ids),
        )
        outcomes = {}
        for name, preference in (("hard", torch.tensor(hard)), ("smooth", soft)):
            result, target, _ = exact_replay(
                view,
                TensorPreference(preference, int(rows[0])),
                int(rows[0]),
                int(rows[-1] + 1),
            )
            outcomes[name] = (result, target)
        difference = soft.numpy() - hard
        chosen = view.valid[rows]
        records[arm] = {
            "fit_only_rows": [int(rows[0]), int(rows[-1])],
            "mean_absolute_preference_error_bps": float(
                np.abs(difference[chosen]).mean() * 1e4
            ),
            "max_absolute_preference_error_bps": float(
                np.abs(difference[chosen]).max() * 1e4
            ),
            "mean_absolute_weight_difference": float(
                np.abs(outcomes["hard"][1] - outcomes["smooth"][1]).mean()
            ),
            "hard_mean_net_bps": float(
                outcomes["hard"][0].net_excess_all_cash_bps.mean()
            ),
            "smooth_mean_net_bps": float(
                outcomes["smooth"][0].net_excess_all_cash_bps.mean()
            ),
            "temperature_chosen_from_financial_outcomes": False,
            "source_parent": design["parents"][arm]["F2"]["11"],
        }
        del compiled, model, caches, data, view
        torch._dynamo.reset()
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    write_json_atomic(
        root / "rank_bridge.json",
        {
            "implementation": _git_identity(),
            "arms": records,
            "financial_selection_or_evaluation_read": False,
        },
    )
    print(records)


def audit_gradients(root):
    torch.set_num_threads(1)
    old = Path(read(PROJECT / "docs/v2_decision_run.json")["root"])
    data, binding = load_data(old, "C6")
    m = calibration(read(old / "phase3/mappings/F2.json")["arms"]["C6"])
    values = (data.ranks.mean(-1) - m.mean[0]) / m.scale[0] * m.coefficient[
        0
    ] + m.intercept
    results = []
    for start in (0, 120, 300):  # all within F2 fit, not financial evaluation
        rows = np.arange(start, start + 12)
        preference = torch.tensor(values[rows])
        initial = data.initial_account(start, policy_ledger_config())
        rng = np.random.default_rng(start + 20260917)
        direction = torch.tensor(rng.normal(size=preference.shape) * 0.0001)

        def value(coefficient):
            return utility_path(
                data,
                preference + coefficient * direction,
                clone_account(initial),
                rows,
                terminal=True,
            )

        t = time.monotonic()
        parameter = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        loss, _, plans, nav = value(parameter)
        loss.backward()
        analytic = parameter.grad.item()
        numerical = []
        for epsilon in (1e-4, 1e-5):
            plus, minus = value(epsilon)[0].item(), value(-epsilon)[0].item()
            numerical.append((plus - minus) / (2 * epsilon))
        exact, expected, _ = exact_replay(
            data, TensorPreference(preference, start), start, start + 12
        )
        result = {
            "first_row": start,
            "last_row": start + 11,
            "analytic": analytic,
            "finite_difference": numerical,
            "nav_error": float(np.max(np.abs(nav - exact.nav))),
            "target_error": float(np.max(np.abs(plans - expected))),
            "seconds": time.monotonic() - t,
        }
        results.append(result)
        write_json_atomic(
            root / "historical_gradient_audit.json",
            {
                "implementation": _git_identity(),
                "cache_binding": binding,
                "tests": results,
                "completed": False,
            },
        )
        if abs(analytic - numerical[-1]) > 0.02 * abs(numerical[-1]) + 0.001:
            raise ValueError(f"historical account derivative differs: {result}")
        if result["nav_error"] > 1e-7 or result["target_error"] > 1e-7:
            raise ValueError("historical account differs from independent ledger")
    write_json_atomic(
        root / "historical_gradient_audit.json",
        {
            "implementation": _git_identity(),
            "cache_binding": binding,
            "tests": results,
            "completed": True,
            "financial_evaluation_read": False,
        },
    )
    print(results)


def export_closeout(root):
    frozen = read(root / "closeout_design.json")
    store = Path(frozen["store"]["root"])
    manifest = read(store / "manifest.json")
    dates = np.load(store / "date_index.npy", allow_pickle=False)
    isins = np.load(store / "isin_index.npy", allow_pickle=False)
    arrays = {
        k: np.load(
            store / manifest["arrays"][k]["path"], mmap_mode="r", allow_pickle=False
        )
        for k in (
            "raw_close",
            "action_successor_index",
            "action_has_action",
            "action_session_resolved",
            "active",
        )
    }
    report = {
        "design": bound(root / "closeout_design.json"),
        "arms": {},
        "largest_events": {},
    }
    for arm in ("C6", "TE_all"):
        source = root / "closeout" / arm / "summary.json"
        summary = read(source)
        report["arms"][arm] = {
            "source": bound(source),
            "books": summary["books"],
            "intercept_daily_bps": summary["intercept_daily_bps"],
        }
        for scenario, record in summary["settlements"].items():
            checked(record["source"])
            for event in sorted(
                record["records"], key=lambda x: abs(x["gross_notional"]), reverse=True
            )[:10]:
                key = f"{event['security']}:{event['fill_date']}"
                if key in report["largest_events"]:
                    continue
                j = int(np.flatnonzero(isins == event["security"])[0])
                t = int(np.searchsorted(dates, np.datetime64(event["fill_date"])))
                observed = np.flatnonzero(
                    np.isfinite(arrays["raw_close"][:, j])
                    & (arrays["raw_close"][:, j] > 0)
                )
                before = observed[observed <= t]
                after = observed[observed > t]
                begin = max(0, int(before[-1]) - 5) if len(before) else max(0, t - 20)
                report["largest_events"][key] = {
                    "example_arm": arm,
                    "example_scenario": scenario,
                    "last_quote": str(dates[before[-1]]) if len(before) else None,
                    "next_quote_same_isin": str(dates[after[0]])
                    if len(after)
                    else None,
                    "observed_fill": False,
                    "example_settlement": event,
                    "action_rows": [
                        {
                            "date": str(dates[i]),
                            "resolved": bool(arrays["action_session_resolved"][i, j]),
                            "successor": str(
                                isins[arrays["action_successor_index"][i, j]]
                            )
                            if arrays["action_successor_index"][i, j] >= 0
                            else None,
                        }
                        for i in range(begin, min(t + 1, len(dates)))
                        if arrays["action_has_action"][i, j]
                    ],
                    "source_limit": "No verified contractual successor/cash settlement in accepted store; retain sensitivity flag.",
                }
    report["corporate_action_contract"] = manifest["metadata"][
        "corporate_action_contract"
    ]
    report["isin_succession_link_count"] = manifest["metadata"][
        "isin_succession_link_count"
    ]
    write_json_atomic(root / "closeout_summary.json", report)
    write_json_atomic(PROJECT / "docs/v2_portfolio_objective_closeout.json", report)


def export_acceptance(root, engineering_root):
    """Compact the already completed admission; never rerun financial screens."""
    records = {}
    evidence = []
    for arm in ("C6", "TE_all"):
        for fold in ("F2", "F14"):
            path = (
                engineering_root
                / "engineering"
                / arm
                / "hybrid"
                / f"{fold}_seed_11/acceptance.json"
            )
            record = read(path)
            evidence.append(path)
            checked(record["contract"]["design"])
            if (
                record["financial_selection_or_evaluation_read"]
                or record["nav_max_error"] > 1e-7
            ):
                raise ValueError("engineering admission failed")
            records[f"{arm}:{fold}"] = {
                "source": bound(path),
                "warm_step_seconds": record["timing"][-1]["seconds"],
                "peak_cuda_gib": record["peak_cuda_bytes"] / 2**30,
                "nav_max_error": record["nav_max_error"],
                "loss_scale_retries": sum(
                    x["loss_scale_retries"] for x in record["timing"]
                ),
                "shared_encoder_gradient_blocks": record["loss_scale"]["blocks"],
            }
    historical = read(root / "historical_gradient_audit.json")
    bridge = read(engineering_root / "rank_bridge.json")
    evidence.extend(
        [root / "historical_gradient_audit.json", engineering_root / "rank_bridge.json"]
    )
    if not historical["completed"] or bridge["financial_selection_or_evaluation_read"]:
        raise ValueError("historical admission or bridge incomplete")
    launch = datetime.fromisoformat(
        read(root / "local_execution.json")["launched_at"]
    ).timestamp()
    if any(path.stat().st_mtime > launch for path in evidence):
        raise ValueError("admission evidence was not completed before financial launch")
    report = {
        "status": "engineering_passed_financial_experiment_running",
        "training_design": bound(root / "training_design.json"),
        "engineering_design": bound(engineering_root / "training_design.json"),
        "local_execution": bound(root / "local_execution.json"),
        "all_admission_evidence_completed_before_launch": True,
        "gpu_cases": records,
        "historical_gradient_source": bound(root / "historical_gradient_audit.json"),
        "historical_gradients": historical["tests"],
        "rank_bridge_source": bound(engineering_root / "rank_bridge.json"),
        "rank_bridge": bridge["arms"],
        "scope": "Engineering admission only; it is not evidence of economic improvement.",
    }
    write_json_atomic(root / "engineering_acceptance.json", report)
    write_json_atomic(PROJECT / "docs/v2_portfolio_objective_engineering.json", report)


def audit_fits(root):
    """Audit completed trajectories only; partial fits remain resumable."""
    design = read(root / "training_design.json")
    records = {}
    for path in sorted((root / "fits").glob("*/*/*/run_manifest.json")):
        manifest = read(path)
        contract = manifest["contract"]
        if contract["design"] != bound(root / "training_design.json"):
            raise ValueError("fit belongs to another financial design")
        for file, digest in manifest["files"].items():
            if sha256_file(path.parent / file) != digest:
                raise ValueError(f"completed fit artifact changed: {file}")
        arm, fold, seed = contract["arm"], contract["fold"], str(contract["seed"])
        parent = torch.load(
            checked(design["parents"][arm][fold][seed]),
            map_location="cpu",
            weights_only=True,
        )
        initial = torch.load(
            path.parent / "initial.pt", map_location="cpu", weights_only=True
        )
        if any(
            not torch.equal(value, initial["model"][f"model.{name}"])
            for name, value in parent["model_state_dict"].items()
        ):
            raise ValueError("continuation did not start from its matched parent")
        if any(
            torch.count_nonzero(value)
            for name, value in initial["model"].items()
            if "economic_head" in name
        ):
            raise ValueError("initial cardinal head was not zero")
        history = read(path.parent / "history.json")
        best = {k: (history[0]["selection"][k], 0) for k in ("ic", "utility_bps")}
        epochs = {}
        for entry in history[1:]:
            number = entry["epoch"]
            epoch_path = path.parent / "epochs" / f"epoch_{number:03d}.pt"
            epochs[str(number)] = sha256_file(epoch_path)
            for selector in best:
                threshold = 1e-4 if selector == "ic" else 0.01
                if entry["selection"][selector] > best[selector][0] + threshold:
                    best[selector] = (entry["selection"][selector], number)
        for selector, (score, number) in best.items():
            state = torch.load(
                path.parent / f"selected_{selector}.pt",
                map_location="cpu",
                weights_only=True,
            )
            source = (
                path.parent / "epochs" / f"epoch_{number:03d}.pt"
                if number
                else path.parent / "initial.pt"
            )
            selected_source = torch.load(source, map_location="cpu", weights_only=True)
            if (
                state["epoch"] != number
                or list(manifest["selected"][selector]) != [score, number]
                or any(
                    not torch.equal(v, selected_source["model"][k])
                    for k, v in state["model"].items()
                )
            ):
                raise ValueError("selector does not match its shared saved trajectory")
        scale = read(path.parent / "loss_scale.json")
        if scale != read(root / "calibration" / arm / f"{fold}_seed_{seed}.json"):
            raise ValueError("objectives did not share the registered gradient scale")
        records[str(path.parent.relative_to(root))] = {
            "manifest": bound(path),
            "epoch_sha256": epochs,
            "selected": manifest["selected"],
            "zero_gradient_blocks": sum(
                x["training"]["zero_gradient_blocks"] for x in history[1:]
            ),
            "loss_scale_retries": sum(
                x["training"]["loss_scale_retries"] for x in history[1:]
            ),
        }
    write_json_atomic(
        root / "completed_fit_audit.json",
        {"completed_fits": len(records), "fits": records},
    )
    print({"completed_fits_audited": len(records)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("gradients", "closeout", "bridge", "acceptance", "fits")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--engineering-root", type=Path)
    args = parser.parse_args()
    if args.command == "gradients":
        audit_gradients(args.root)
    elif args.command == "closeout":
        export_closeout(args.root)
    elif args.command == "acceptance":
        export_acceptance(args.root, args.engineering_root)
    elif args.command == "fits":
        audit_fits(args.root)
    else:
        audit_bridge(args.root)


if __name__ == "__main__":
    main()
