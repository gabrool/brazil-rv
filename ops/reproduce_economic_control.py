"""Reproduce one sealed account with its own original runtime before attribution."""

import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
RUN = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
ROOT = Path(RUN["stage_c_root"])
PLAN = json.loads((ROOT / "plan.json").read_text())
sys.path.insert(0, PLAN["baseline_control"]["runtime_path"])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from brazil_rv.execution.portfolio_policy import CalibratedPolicy, exact_replay  # noqa: E402
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic  # noqa: E402
from brazil_rv.v2.foundation_readouts import new_panel  # noqa: E402
from brazil_rv.v2.objective_readouts import calibration, rank_view  # noqa: E402
from brazil_rv.v2.portfolio_training import load_data, windows  # noqa: E402


def main():
    tick = perf_counter()
    torch.set_num_threads(1)
    out = ROOT / "baseline_control"
    out.mkdir(exist_ok=False)
    shutil.copyfile(__file__, out / "executed.py")
    original = PLAN["baseline_control"]
    source = Path(original["book"]["path"])
    assert sha256_file(source) == original["book"]["sha256"]
    meta = json.loads(source.read_text())
    prior = Path(PLAN["prior_root"])
    data, cache = load_data(prior, "C6")
    assert cache == meta["provenance"]["economic_cache_binding"]
    rows = windows(prior, data, "F2")["evaluation"]
    panels, valid, _ = new_panel(
        Path(PLAN["foundation_root"]), data, "TE_full", "F2", rows, "raw"
    )
    mapping = calibration(
        json.loads((prior / "phase3/mappings/F2.json").read_text())["arms"]["TE_all"]
    )
    view = rank_view(data, rows, panels["ensemble"], valid)
    result, targets, _ = exact_replay(
        view, CalibratedPolicy(mapping), int(rows[0]), int(rows[-1] + 1)
    )
    account = source.parent / "account.npz"
    assert sha256_file(account) == meta["files"]["account.npz"]
    with np.load(account) as old:
        errors = {
            k: float(np.nanmax(np.abs(getattr(result, k) - old[k])))
            for k in ("nav", "free_cash", "restricted_cash", "signed_shares")
        }
        errors["targets"] = float(np.max(np.abs(targets - old["targets"])))
    errors["net_excess_bps"] = float(
        np.max(np.abs(result.net_excess_all_cash_bps - meta["daily"]["net_excess_bps"]))
    )
    np.savez_compressed(
        out / "comparison.npz",
        nav=result.nav,
        targets=targets,
        net_excess_bps=result.net_excess_all_cash_bps,
    )
    assert max(errors.values()) < 1e-8, errors
    write_json_atomic(
        out / "manifest.json",
        dict(
            passed=True,
            source=original,
            errors=errors,
            sessions=len(rows),
            names=len(data.inputs.security_ids),
            old_policy_cache=cache,
            seconds=perf_counter() - tick,
            no_neural_forward_or_fit=True,
        ),
    )
    print(json.dumps(errors), flush=True)


if __name__ == "__main__":
    main()
