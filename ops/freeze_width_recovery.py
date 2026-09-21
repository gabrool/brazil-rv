"""Preserve the failed width wave and bind only its necessary replacement fits."""

import json
from copy import deepcopy
from pathlib import Path
import subprocess

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def runtime(root):
    source = root / "research/src/brazil_rv/v2"
    return dict(
        git=dict(
            commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
        ),
        files={
            key: binding(source / filename)
            for key, filename in (
                ("training", "round7_training.py"),
                ("model", "characteristic_model.py"),
                ("temporal", "temporal_pathway.py"),
                ("configuration", "round7.py"),
                ("compiler", "train.py"),
            )
        },
    )


def main():
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    original = run["stage_d_width_plan"]
    old = bound_json(original)
    old_root = Path(original["path"]).parent
    failure = old_root / "scoring_failure"
    scope = bound_json(binding(failure / "architecture_scope/report.json"))
    for name in ("TE_full", "TE_wide", "GRU_early", "C6", "TE_128"):
        for comparison in ("saved_vs_eager", "compiled_vs_eager", "poisoned_vs_eager"):
            result = scope[name][comparison]
            assert (
                result["cells"] == result["finite"] and result["outside_tolerance"] == 0
            )
    assert scope["GRU_96"]["poisoned_vs_eager"]["finite"] == 0
    corrected_parent = bound_json(
        binding(failure / "parent_scope/qualified/report.json")
    )
    assert all(
        r["finite"] == r["cells"] and r["outside_tolerance"] == 0
        for r in corrected_parent["comparisons"].values()
    )
    old_progress = bound_json(binding(old_root / "refits.json"))
    reused = [r for r in old_progress["completed"] if r["key"].startswith("TE_128/")]
    assert len(reused) == 10
    for record in reused:
        assert bound_json(record["manifest"])["status"] == "completed"
    acceptance = dict(
        status="compiler_fix_qualified_gru96_replacement_required",
        architecture_scope=binding(failure / "architecture_scope/report.json"),
        parent_original=binding(failure / "parent_scope/original/report.json"),
        parent_qualified=binding(failure / "parent_scope/qualified/report.json"),
        fixed_forward=binding(
            failure / "allocation_state_probe/qualified/native_norm/report.json"
        ),
        gradient_test=binding(failure / "native_norm_tests/stdout.txt"),
        fixed_runtime=runtime(Path("C:/Brazil-RV/.worktrees/normalization")),
        preserved_original_plan=original,
        preserved_original_progress=binding(old_root / "refits.json"),
        invalidated_results="All GRU_96 width forecasts and dependent economics are excluded. Ten completed fits plus the failed nine-epoch F10/29 attempt remain original evidence. Fifteen new GRU96 P/F fits use identical cells/seeds/folds/budgets/selector with corrected compiled normalization. The tested original P batch passed; replacing its three parents is a conservative response to the same unsafe fused operation, not a claimed observed P mismatch.",
        retained_results="Stage C and TE128 remain usable: five architecture controls reproduce all original F10/11 forecasts and pass actual first-batch NaN allocation poisoning. The demonstrated failure is the GRU96 fused residual reduction. This purpose-limited scope test is not an exhaustive proof of every prior checkpoint/training graph, nor new profitability evidence. No source/store/account arithmetic is repeated.",
        correction="ATen native_layer_norm and backward remain in the compiled graph. No model equations, parameter shapes, training data, learning budget, optimizer, selector or installed package changed. Generated-kernel barrier was diagnostic only; disabling in-place buffers failed and is not the fix.",
    )
    write_json_atomic(failure / "acceptance.json", acceptance)
    root = Path(run["root"]) / "capacity_width_recovery"
    (root / "fits").mkdir(parents=True, exist_ok=False)
    (root / "executed_freezer.py").write_bytes(Path(__file__).read_bytes())
    plan = deepcopy(old)
    plan.update(
        status="frozen_before_compiler_corrected_width_outcomes",
        original_plan=original,
        compiler_acceptance=binding(failure / "acceptance.json"),
        driver=binding(PROJECT / "ops/run_capacity_width.py"),
        freezer=binding(Path(__file__)),
        runtime=acceptance["fixed_runtime"],
        runtime_by_arm={
            "GRU_96": acceptance["fixed_runtime"],
            "TE_128": runtime(Path("C:/Brazil-RV/.worktrees/matched-refits")),
        },
        reuse=dict(
            completed_fits=reused,
            directory_alias=dict(
                path=str(root / "fits/TE_128"), target=str(old_root / "fits/TE_128")
            ),
            original_evaluation=run["stage_d_width_evaluation_plan"],
            rule="Retain all TE128 completed fits and saved qualified books; complete five remaining TE128 fits with their unchanged runtime. Replace only GRU96 fits/books. Original width root and frozen plans stay intact; the explicit directory junction avoids duplicating attention fits.",
        ),
    )
    write_json_atomic(root / "plan.json", plan)
    reference = binding(root / "plan.json")
    write_json_atomic(root / "frozen_design.json", dict(store=old["store"]))
    write_json_atomic(
        root / "refits.json",
        dict(
            status="running",
            plan=reference,
            completed=reused,
            planned=old["planned_fits"],
        ),
    )
    run["stage_d_width_original_plan"] = original
    run["stage_d_width_original_evaluation_plan"] = run.pop(
        "stage_d_width_evaluation_plan"
    )
    if "stage_d_width_evaluation" in run:
        run["stage_d_width_original_evaluation"] = run.pop("stage_d_width_evaluation")
    run["stage_d_compiler_acceptance"] = binding(failure / "acceptance.json")
    run["stage_d_width_plan"] = reference
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(
                plan=reference,
                reused=len(reused),
                remaining=old["planned_fits"] - len(reused),
            )
        )
    )


if __name__ == "__main__":
    main()
