"""Bounded GH200 timing updates on fit inputs; no evaluation scores or research fit."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .config import ModelConfig
from .data import V2DailyDataset, collate_v2_daily, restore_name_axis, stage_name_count
from .losses import multi_horizon_loss
from .model import DailyMultiHorizonModel
from .train import (
    _cli_stage_indices,
    _model_forward,
    _to_device,
    build_optimizer,
    compile_forward,
    sam_step,
    set_deterministic_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--mode", choices=("dense_fp32", "compact_fp32", "compact_bf16"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if sha256_file(args.checkpoint) != args.expected_sha256:
        raise ValueError("benchmark initialization differs from sealed parent")
    fit, selection, _, target_window = _cli_stage_indices(args.store, "F", "F12")
    dataset = V2DailyDataset(
        args.store,
        fit,
        stage="finetune",
        include_intraday=False,
        include_fast=False,
        purpose="training",
        target_window_indices=target_window,
    )
    holdout = V2DailyDataset(
        args.store,
        selection,
        stage="finetune",
        include_intraday=False,
        include_fast=False,
        purpose="selection",
        target_window_indices=selection,
    )
    positions = np.linspace(0, len(fit) - 1, 16, dtype=int)
    width = None if args.mode == "dense_fp32" else stage_name_count(dataset, holdout)
    cpu = collate_v2_daily([dataset[int(i)] for i in positions], fixed_name_count=width)
    set_deterministic_seed(11)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = replace(
        ModelConfig(**payload["input_contract"]["model_config"]),
        use_bf16=args.mode.endswith("bf16"),
    )
    model = DailyMultiHorizonModel(config).cuda()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    batch = _to_device(cpu, torch.device("cuda"), omit_fast_stream=True)
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.use_bf16),
    ):
        prediction = _model_forward(model, batch)[..., :5].float().cpu().numpy()
    if width is not None:
        prediction = restore_name_axis(
            prediction, cpu["name_index"].numpy(), len(dataset.store.isins)
        )
    np.save(args.output / "fit_predictions.npy", prediction, allow_pickle=False)
    model.train()
    forward = compile_forward(model)
    optimizer = build_optimizer(model)

    def closure():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=config.use_bf16):
            score = _model_forward(forward, batch)[..., :5]
            return multi_horizon_loss(score, batch["targets"], batch["target_mask"])

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(3):
        sam_step(model, optimizer, closure)
    torch.cuda.synchronize()
    warmup = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(10):
        start = time.perf_counter()
        sam_step(model, optimizer, closure)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    record = dict(
        mode=args.mode,
        name_width=int(cpu["active_mask"].shape[1]),
        fit_date_indices=fit[positions].tolist(),
        active_names=cpu["active_mask"].sum(1).tolist(),
        seconds_per_sam_update=timings,
        median_seconds=float(np.median(timings)),
        warmup_seconds=warmup,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        checkpoint_sha256=args.expected_sha256,
        store_manifest_sha256=sha256_file(args.store / "manifest.json"),
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
    )
    write_json_atomic(args.output / "benchmark.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
