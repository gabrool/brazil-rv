"""Two cleared scalar actions retain their own index in the account contract."""

import copy
from dataclasses import fields, replace
import inspect
import json
from pathlib import Path
import pickle
import subprocess
from time import perf_counter

import numpy as np

from brazil_rv.execution.portfolio_policy import ledger_arguments
from brazil_rv.execution.stateful_ledger import (
    _validate_inputs,
    simulate_stateful_ledger,
)
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.portfolio_training import windows

PROJECT = Path(__file__).resolve().parents[1]


def main():
    tick = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    previous = run["stage_c_refit_economics"]
    source = bound_json(previous)
    out = Path(previous["path"]).parent / "action_identity"
    assert (out / "failure").is_dir() and not (out / "manifest.json").exists()
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    cache = Path(source["cache"]["path"])
    assert sha256_file(cache) == source["cache"]["sha256"]
    with cache.open("rb") as f:
        old = pickle.load(f)
    corrected = copy.copy(old)
    successor = old.inputs.action_successor_index.copy()
    rows = []
    for event in source["source_actions_represented_once"]:
        t = list(map(str, old.inputs.dates)).index(event["date"])
        n = list(old.inputs.security_ids).index(event["isin"])
        assert successor[t, n] == -1 and not old.inputs.action_has_action[t, n]
        assert old.inputs.action_shares_per_prior_share[t, n] == 1
        assert old.inputs.action_cash_per_prior_share[t, n] == 0
        assert old.inputs.action_payment_session[t, n] == -1
        successor[t, n] = n
        rows.append(
            dict(
                date=event["date"],
                isin=event["isin"],
                date_index=t,
                axis=n,
                before=-1,
                after=n,
            )
        )
    assert len(rows) == 2
    changed = np.argwhere(successor != old.inputs.action_successor_index)
    np.testing.assert_array_equal(changed, [[r["date_index"], r["axis"]] for r in rows])
    corrected.inputs = replace(old.inputs, action_successor_index=successor)
    unchanged_input_fields = []
    for field in fields(old.inputs):
        if field.name != "action_successor_index":
            assert getattr(corrected.inputs, field.name) is getattr(
                old.inputs, field.name
            )
            unchanged_input_fields.append(field.name)
    unchanged_policy_fields = []
    for key, value in vars(old).items():
        if key != "inputs":
            assert vars(corrected)[key] is value
            unchanged_policy_fields.append(key)
    plan = bound_json(run["stage_c_data_replay_plan"])
    defaults = {
        k: v.default
        for k, v in inspect.signature(simulate_stateful_ledger).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    config = bound_json(run["economic_account"])["primary_config"]
    controls = {
        k: config[k]
        for k in ("borrow_source", "volatility_balanced_entries", "beta_hedge")
    }
    validated = []
    for fold in plan["folds"]:
        ix = windows(Path(plan["prior_root"]), corrected, fold)["evaluation"]
        start, stop = int(ix[0]), int(ix[-1]) + 1
        args = defaults | ledger_arguments(corrected, start, stop) | controls
        args["shortable"] = corrected.shortable[start:stop]
        _validate_inputs(
            **{k: args[k] for k in inspect.signature(_validate_inputs).parameters}
        )
        before = ledger_arguments(old, start, stop)["action_terms"]
        after = args["action_terms"]
        for key in (
            "shares_per_prior_share",
            "cash_per_prior_share",
            "session_resolved",
            "has_action",
        ):
            np.testing.assert_array_equal(getattr(before, key), getattr(after, key))
        count = int(np.count_nonzero(before.successor_index != after.successor_index))
        assert count == (2 if fold == "F6" else 0)
        validated.append(
            dict(
                fold=fold,
                sessions=stop - start,
                changed_successor_cells=count,
                account_schema_passed=True,
            )
        )
    fixed_cache = out / "policy_data.pkl"
    with fixed_cache.open("xb") as f:
        pickle.dump(corrected, f, protocol=pickle.HIGHEST_PROTOCOL)
    fixed_hash = sha256_file(fixed_cache)
    compressed = subprocess.run(
        ["compact.exe", "/C", "/EXE:LZX", str(fixed_cache)],
        capture_output=True,
        text=True,
        check=True,
    )
    (out / "storage.txt").write_text(compressed.stdout)
    assert sha256_file(fixed_cache) == fixed_hash
    with fixed_cache.open("rb") as f:
        restored = pickle.load(f)
    np.testing.assert_array_equal(restored.inputs.action_successor_index, successor)
    write_json_atomic(
        out / "correction.json",
        dict(
            previous_input=previous,
            previous_qualification=run["stage_c_refit_economics_qualification"],
            rows=rows,
            unchanged_input_fields=unchanged_input_fields,
            unchanged_policy_fields=unchanged_policy_fields,
            account_arguments=validated,
            limits="The original qualification copied the same invalid -1 expectation; its other composition/risk/source proofs are reused. The failed F6 invocation stopped in whole-window input validation before any daily book state was simulated. No model store, fit, risk reducer, original cache or passed F2 book is changed or rerun. Static PolicyData is shallow-copied without invoking its constructor. Cleared ordinary actions use own security index; explicit distributions remain the sole conversion mechanism.",
        ),
    )
    manifest = dict(
        source,
        cache={**binding(fixed_cache), "bytes": fixed_cache.stat().st_size},
        previous_input=previous,
        action_identity_correction=binding(out / "correction.json"),
        status="qualified_refit_account_identity_correction",
    )
    write_json_atomic(out / "manifest.json", manifest)
    input_reference = binding(out / "manifest.json")
    write_json_atomic(
        out / "qualification.json",
        dict(
            passed=True,
            inputs=input_reference,
            previous_qualification=run["stage_c_refit_economics_qualification"],
            correction=binding(out / "correction.json"),
            seconds=perf_counter() - tick,
        ),
    )
    run["stage_c_refit_economics"] = input_reference
    run["stage_c_refit_economics_qualification"] = binding(out / "qualification.json")
    old_plans = {}
    for key in (
        "stage_c_data_replay_plan",
        "stage_c_refit_sensitivity_plan",
        "stage_c_refit_debit_plan",
    ):
        previous_plan = run[key]
        old_plans[key] = previous_plan
        value = bound_json(previous_plan)
        updated = dict(
            value,
            previous_plan=previous_plan,
            account_identity_correction=binding(out / "correction.json"),
        )
        if key == "stage_c_data_replay_plan":
            updated.update(
                inputs=input_reference,
                input_qualification=run["stage_c_refit_economics_qualification"],
            )
        else:
            updated["primary"] = run["stage_c_data_replay_plan"]
            if key == "stage_c_refit_debit_plan":
                updated["denied_sensitivities"] = run["stage_c_refit_sensitivity_plan"]
        target = Path(previous_plan["path"]).parent / "qualified_plan.json"
        assert not target.exists()
        write_json_atomic(target, updated)
        run[key] = binding(target)
    write_json_atomic(
        out / "plan_resolution.json",
        dict(
            previous=old_plans,
            corrected={k: run[k] for k in old_plans},
            scope="Only corrected portfolio-input bindings and provenance advance. All hypotheses, forecasts, driver bytes, graphs, seeds, folds and fit budgets remain frozen. Previously completed F2 books retain their original input/plan bindings and qualification because all their ledger coordinates are unchanged.",
        ),
    )
    run["stage_c_refit_action_identity"] = binding(out / "plan_resolution.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            dict(corrected=rows, validated=validated, seconds=perf_counter() - tick)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
