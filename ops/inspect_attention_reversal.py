"""Saved-artifact audit of original versus corrected attention training."""

import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            result += differences(a.get(key), b.get(key), f"{prefix}.{key}".strip("."))
        return result
    return [] if a == b else [prefix]


def fit_detail(path):
    manifest = bound_json(binding(path))
    root = path.parent
    history = bound_json(binding(root / "history.json"))
    diagnostic = bound_json(binding(root / "diagnostics.json"))
    return manifest, dict(
        source=binding(path),
        history=binding(root / "history.json"),
        diagnostics=binding(root / "diagnostics.json"),
        epochs=manifest["epochs_completed"],
        selected_epoch=manifest["selected_epoch"],
        selected_clean_fit_ic=diagnostic["selected_clean_fit"]["mean_ic"],
        selection_ic=manifest["selection_ic"],
        ema_selection_ic=manifest.get("ema_selection_ic"),
        gradient_clip_fraction=float(
            np.mean([x["gradient_clip_fraction"] for x in history])
        ),
        loss_scale_retries=sum(x["loss_scale_retries"] for x in history),
        updates=sum(x["updates"] for x in history),
        selected_update=sum(
            x["updates"] for x in history[: manifest["selected_epoch"]]
        ),
        final_training_loss=history[-1]["training_loss"],
    )


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    source = bound_json(run["stage_c_plan"])
    refit = bound_json(run["stage_c_refit_plan"])
    out = Path(run["root"]) / "scaling_investigation"
    out.mkdir(exist_ok=True)
    plan_path = out / "plan.json"
    if not plan_path.exists():
        write_json_atomic(
            plan_path,
            dict(
                status="frozen_before_new_diagnostic_replays_or_additional_fold_outcomes",
                instruction=binding(PROJECT / "docs/v2_SCALING_INVESTIGATION.md"),
                old_plan=run["stage_c_plan"],
                new_fit_plan=run["stage_c_refit_plan"],
                old_result=run["stage_c_account_results"],
                new_result=run["stage_c_refit_results"],
                new_inputs=run["stage_c_refit_economics"],
                account=run["economic_account"],
                terms=run["stage_c_event_candidate_terms"],
                arms=["TE_full", "TE_wide"],
                screen_folds=source["folds"],
                seeds=[11, 29, 47],
                counterfactuals=[
                    "old_forecast_old_risk",
                    "old_forecast_new_risk",
                    "new_forecast_old_risk",
                ],
                counterfactual_account_inputs="new corrected inputs, frozen across all three contrasts",
                new_books=24,
                capital=10000000,
                policy="original equal-rank neutral calibration",
                expansion_folds=["F2", "F3", "F6", "F7", "F10", "F11", "F13", "F14"],
                reserved_development_folds=["F1", "F4", "F5", "F8", "F9", "F12"],
                reserve_limit="Prior research used these years; not pristine historical holdouts. No 2025/2026 consumer reads.",
                deferred="Frozen LSTM fits have not begun; explanation of reversal takes priority.",
            ),
        )
    plan = bound_json(binding(plan_path))
    assert not (out / "fit_audit.json").exists(), "Preserve prior result"
    (out / "executed_fit_audit.py").write_bytes(Path(__file__).read_bytes())
    dates = np.load(Path(refit["store"]["root"]) / "date_index.npy")
    rows, code_pairs = [], set()
    for arm in plan["arms"]:
        for slot in [
            *(f"P_seed_{seed}" for seed in plan["seeds"]),
            *(
                f"{fold}_seed_{seed}"
                for fold in plan["screen_folds"]
                for seed in plan["seeds"]
            ),
        ]:
            old, old_detail = fit_detail(
                Path(source["foundation_root"])
                / "fits"
                / arm
                / slot
                / "run_manifest.json"
            )
            new, new_detail = fit_detail(
                Path(run["stage_c_refit_root"])
                / "fits"
                / arm
                / slot
                / "run_manifest.json"
            )
            a, b = old["contract"], new["contract"]
            code_pairs.add((a["code"]["commit"], b["code"]["commit"]))
            conditioning = differences(a["preprocessing"], b["preprocessing"])
            window = b["fit_target_window"]
            rows.append(
                dict(
                    arm=arm,
                    slot=slot,
                    old=old_detail,
                    new=new_detail,
                    contract_changed_keys=[
                        key for key in a.keys() | b.keys() if a.get(key) != b.get(key)
                    ],
                    conditioning_changed_fields=conditioning,
                    fit_dates=dict(
                        count=len(window),
                        first=str(dates[window[0]]),
                        last=str(dates[window[-1]]),
                    ),
                    fit_window_exact=a["fit_target_window"] == window,
                    selection_policy_exact=a["selection_policy"]
                    == b["selection_policy"],
                    model_exact=a["config"] == b["config"],
                    recipe_exact=a["recipe"] == b["recipe"],
                    conditioning_date_changes=[
                        key for key in conditioning if "fit_date_indices" in key
                    ],
                )
            )
    code = []
    core = [
        "round7_training.py",
        "round7_preprocessing.py",
        "characteristic_model.py",
        "model.py",
        "round7.py",
        "splits.py",
        "normalization.py",
    ]
    for old, new in sorted(code_pairs):
        prefix = "research/src/brazil_rv/v2/"
        changed = subprocess.check_output(
            [
                "git",
                "diff",
                "--name-only",
                old,
                new,
                "--",
                *[prefix + name for name in core],
            ],
            cwd=PROJECT,
            text=True,
        ).splitlines()
        code.append(dict(old=old, new=new, inspected_files=core, changed_files=changed))
    aggregate = []
    for arm in plan["arms"]:
        for stage in ["P", "F"]:
            selected = [
                r for r in rows if r["arm"] == arm and r["slot"].startswith(stage)
            ]
            aggregate.append(
                dict(
                    arm=arm,
                    stage=stage,
                    fits=len(selected),
                    **{
                        version: {
                            key: float(np.mean([r[version][key] for r in selected]))
                            for key in [
                                "epochs",
                                "selected_clean_fit_ic",
                                "selection_ic",
                                "gradient_clip_fraction",
                                "loss_scale_retries",
                            ]
                        }
                        for version in ["old", "new"]
                    },
                )
            )
    old_report = bound_json(plan["old_result"])
    new_report = bound_json(plan["new_result"])
    old_books = bound_json(old_report["books"])
    new_books = bound_json(new_report["books"])
    economic = []
    for arm in plan["arms"]:
        for fold in plan["screen_folds"]:
            old = next(
                r
                for r in old_books
                if r["arm"] == arm
                and r["fold"] == fold
                and r["capital"] == plan["capital"]
                and r["member"] == "ensemble"
                and r["phase"] == "sources"
            )
            new = next(
                r
                for r in new_books
                if r["arm"] == arm
                and r["fold"] == fold
                and r["capital"] == plan["capital"]
                and r["member"] == "ensemble"
            )
            economic.append(
                dict(
                    arm=arm,
                    fold=fold,
                    old_corrected=old["mean"]["net_excess_bps"],
                    new_corrected=new["mean"]["net_excess_bps"],
                    delta=new["data_and_refit_delta_bps_day"],
                    new_forecast_ic=new["forecast_neutral_ic"],
                    shared_component_delta={
                        key: new["mean"][key] - old["mean"][key]
                        for key in old["mean"]
                        if key in new["mean"]
                    },
                )
            )
    report = dict(
        status="saved_artifacts_inspected",
        plan=binding(plan_path),
        pairs=len(rows),
        fits=rows,
        aggregate=aggregate,
        core_code=code,
        economics=economic,
        seconds=perf_counter() - tick,
        limits="No new fit, forecast, portfolio replay or source audit. Diagnostic fit IC describes raw selected checkpoints, not EMA. Numerical equality of a contract is not proof of data/consumer correctness.",
    )
    write_json_atomic(out / "fit_audit.json", report)
    run["scaling_investigation"] = binding(plan_path)
    run["scaling_fit_audit"] = binding(out / "fit_audit.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(pairs=len(rows), aggregates=aggregate, economics=economic, code=code),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
