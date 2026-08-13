from __future__ import annotations

import json
from datetime import datetime, timezone

import torch

from .contract import (
    BASELINE_TCN_SETTINGS,
    GH200_RUNTIME,
    RUN_OUTPUT_BASE,
    architecture_for_model,
)
from .data import (
    create_training_loaders,
    load_sample_index,
    resolve_feature_store,
    select_sample_split,
)
from .engine import compile_model, run_effective_batch_update
from .model import build_neural_model
from .optim import build_optimizer, build_scheduler


def main() -> None:
    store = resolve_feature_store()
    sample_index = load_sample_index(store)
    training = select_sample_split(sample_index, "train")
    validation = select_sample_split(sample_index, "validation")
    architecture = architecture_for_model("tcn", BASELINE_TCN_SETTINGS)
    train_loader, _, _ = create_training_loaders(
        store,
        training,
        validation,
        "tcn",
        "enabled",
        GH200_RUNTIME,
        29,
        architecture,
        "selected",
    )
    torch.manual_seed(29)
    torch.cuda.manual_seed_all(29)
    model = build_neural_model("tcn", architecture, "selected").cuda()
    optimizer, _ = build_optimizer(model)
    scheduler, _, _ = build_scheduler(optimizer, training.height)
    compiled = compile_model(model)
    iterator = iter(train_loader)
    batches = [next(iterator) for _ in range(GH200_RUNTIME.accumulation_steps)]
    update = run_effective_batch_update(
        compiled,
        batches,
        optimizer,
        scheduler,
        GH200_RUNTIME,
        "sam_adamw",
        "soft_spearman",
        0.50,
        0.125,
    )
    passed = bool(update["all_finite"] and update["rng_replay_exact"])
    output = RUN_OUTPUT_BASE / f"sanity_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "sanity_report.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "feature_store": str(store),
                "sam_update": update,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("GH200 sanity update failed")


if __name__ == "__main__":
    main()
