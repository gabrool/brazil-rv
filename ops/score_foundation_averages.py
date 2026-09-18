"""Score original-trajectory averages with the original compatible inference package.

Run this script with PYTHONPATH pointing to the verified historical worktree,
after the active GPU fit worker finishes. No checkpoint is retrained or retuned.
"""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.config import ModelConfig
from brazil_rv.v2.contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS
from brazil_rv.v2.model import DailyMultiHorizonModel
from brazil_rv.v2.research_rounds import _git_identity
from brazil_rv.v2.round7_score import score
from brazil_rv.v2.train import compile_forward


def run(root):
    torch.set_num_threads(1)
    plan = json.loads((root / "averaging_plan.json").read_text())
    store = Path(plan["store"]["root"])
    if sha256_file(store / "manifest.json") != plan["store"]["manifest_sha256"]:
        raise ValueError("averaging inference store differs")
    implementation = _git_identity()
    complete = []
    for arm in ("C6", "TE_all"):
        torch._dynamo.reset()
        models, reference_config = None, None
        for fold in DEVELOPMENT_FOLDS:
            for seed in ALLOWED_SEEDS:
                key = f"{arm}/{fold}/{seed}"
                item = plan["checkpoints"][key]
                if "path" not in item:
                    complete.append({"key": key, "status": "unavailable"})
                    continue
                path = Path(item["path"])
                payload = torch.load(path, map_location="cpu", weights_only=True)
                contract = payload["contract"]
                if contract.get("economic_auxiliary"):
                    raise ValueError(
                        "original neutral averaging cannot use economic fits"
                    )
                characteristic = not contract["pretrain_key"].startswith("s0_")
                config = (CharacteristicConfig if characteristic else ModelConfig)(
                    **contract["config"]
                )
                if models is None:
                    model = (
                        (
                            CharacteristicModel(config)
                            if characteristic
                            else DailyMultiHorizonModel(config)
                        )
                        .cuda()
                        .eval()
                    )
                    models = (model, compile_forward(model))
                    reference_config = config
                elif config != reference_config:
                    raise ValueError("reused inference graph differs across averages")
                started = perf_counter()
                output = path.parent / "scores"
                score(
                    store,
                    path,
                    output,
                    expected_sha256=item["sha256"],
                    reusable_models=models,
                    fixed_name_count=contract["padded_name_count"],
                )
                complete.append(
                    {
                        "key": key,
                        "status": "scored",
                        "seconds": perf_counter() - started,
                        "manifest_sha256": sha256_file(output / "score_manifest.json"),
                    }
                )
                write_json_atomic(
                    root / "averaging_inference_progress.json",
                    {
                        "implementation": implementation,
                        "driver_sha256": sha256_file(Path(__file__)),
                        "plan_sha256": sha256_file(root / "averaging_plan.json"),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "completed": complete,
                        "planned": len(plan["checkpoints"]),
                    },
                )
                print(complete[-1], flush=True)
        del models, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
