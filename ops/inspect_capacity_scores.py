"""Diagnose the saved GRU score failure without changing weights or training."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from brazil_rv.v2 import round7_score
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main(mode, seed):
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    plan = bound_json(run["stage_d_width_plan"])
    root = Path(run["stage_d_width_plan"]["path"]).parent
    fit = root / f"fits/GRU_96/F10_seed_{seed}"
    audit = root / "scoring_failure"
    out = audit / mode if seed == 29 else audit / f"control{seed}" / mode
    out.mkdir(parents=True, exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    checkpoint = binding(fit / "selected.pt")
    payload = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    finite = {k: bool(torch.isfinite(v).all()) for k, v in state.items()}
    assert all(finite.values()), "Non-finite saved weights"
    write_json_atomic(
        out / "plan.json",
        dict(
            mode=mode,
            checkpoint=checkpoint,
            frozen_width_plan=run["stage_d_width_plan"],
            contrast="Same selected weights, original data/preprocessing/membership/full history and autocast. Compare fresh eager then fresh compiled score export; no selector/training/input change or economic outcomes. Save any first failing batch before further diagnosis.",
        ),
    )
    original_batch = round7_score.model_batch
    original_panel = round7_score.canonical_head_panel
    last_cpu = None
    calls = 0

    def batch(cpu, device):
        nonlocal last_cpu
        last_cpu = cpu
        return original_batch(cpu, device)

    def panel(scores, active, horizons):
        nonlocal calls
        calls += 1
        bad = active[..., None] & ~np.isfinite(scores)
        if bad.any():
            torch.save(last_cpu, out / "first_failed_batch.pt")
            np.savez_compressed(
                out / "first_failed_predictions.npz",
                scores=scores,
                active=active,
                bad=bad,
            )
            write_json_atomic(
                out / "first_failure.json",
                dict(
                    batch=calls,
                    date_indices=last_cpu["date_index"].tolist(),
                    nonfinite=int(bad.sum()),
                    coordinates=np.argwhere(bad).tolist(),
                    all_names=last_cpu["name_index"].tolist(),
                ),
            )
        return original_panel(scores, active, horizons)

    round7_score.model_batch = batch
    round7_score.canonical_head_panel = panel
    tick = perf_counter()
    try:
        result = round7_score.score(
            Path(plan["store"]["root"]),
            Path(checkpoint["path"]),
            out / "scores",
            expected_sha256=checkpoint["sha256"],
            compiled=mode == "compiled",
            fixed_name_count=payload["contract"]["padded_name_count"],
        )
        outcome = dict(
            status="all_original_autocast_scores_finite",
            score_manifest=binding(out / "scores/score_manifest.json"),
            dates=result["axes"]["date_count"],
        )
    except FloatingPointError as error:
        outcome = dict(
            status="nonfinite_score_reproduced",
            error=str(error),
            first_failure=binding(out / "first_failure.json"),
        )
    write_json_atomic(
        out / "report.json",
        dict(
            outcome,
            checkpoint=checkpoint,
            batches=calls,
            seconds=perf_counter() - tick,
            weights_finite=finite,
        ),
    )
    print(json.dumps({k: v for k, v in outcome.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--seed", type=int, choices=(11, 29), default=29)
    args = parser.parse_args()
    main(args.mode, args.seed)
