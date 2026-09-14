"""Selected-checkpoint precision readout on existing selection dates only."""

import argparse
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.post_data_program import fit_path, read
from brazil_rv.v2.relational_engineering import fp32_head
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import (
    daily_primary_ic,
    forward,
    model_batch,
    sequential_batches,
)
from brazil_rv.v2.train import _cli_stage_indices


def check(root):
    torch.set_num_threads(6)
    design, choices = (
        read(root / "frozen_design.json"),
        read(root / "calibration_choice.json"),
    )
    store = Path(design["store"]["root"])
    output = root / "calibration_precision.json"
    if output.exists():
        raise FileExistsError(output)
    records = []
    for cell in ("TE_slow", "C1_all"):
        for fold in ("F2", "F14"):
            for seed in (11, 29):
                directory = fit_path(root, cell, choices["recipes"][cell], fold, seed)
                run = read(directory / "run_manifest.json")
                checkpoint = directory / "selected.pt"
                assert sha256_file(checkpoint) == run["artifacts"]["selected.pt"]
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                config = CharacteristicConfig(**saved["contract"]["config"])
                model = CharacteristicModel(config).cuda().eval()
                model.load_state_dict(saved["model_state_dict"])
                _, rows, _, _ = _cli_stage_indices(store, "F", fold)
                preparation = Round7Preprocessing.from_payload(
                    saved["contract"]["preprocessing"]
                )
                data = V2DailyDataset(
                    store,
                    rows,
                    stage="finetune",
                    purpose="selection",
                    lookback=60,
                    enabled_sidecars=tuple(n for n, _ in config.family_counts),
                    include_fast=False,
                    include_intraday=False,
                    include_common_state=bool(config.family_counts),
                    compact_names=True,
                )
                metrics = {
                    name: []
                    for name in (
                        "fp32",
                        "bf16",
                        "bf16_fp32_head",
                        "bf16_fp32_score_rank",
                        "head_fp32_score_rank",
                    )
                }
                errors, head_errors = [], []
                try:
                    for positions in sequential_batches(len(data)):
                        cpu = preparation.collate(
                            [data[i] for i in positions],
                            fixed_name_count=saved["contract"]["padded_name_count"],
                        )
                        batch = model_batch(cpu, torch.device("cuda"))
                        predictions = {}
                        with torch.no_grad():
                            for mode in ("fp32", "bf16", "bf16_fp32_head"):
                                original = model.head.forward
                                if mode == "bf16_fp32_head":
                                    model.head.forward = MethodType(
                                        fp32_head, model.head
                                    )
                                try:
                                    with torch.autocast(
                                        "cuda",
                                        dtype=torch.bfloat16,
                                        enabled=mode != "fp32",
                                    ):
                                        predictions[mode] = (
                                            forward(model, batch, characteristic=True)
                                            .float()
                                            .mean(2)
                                            .cpu()
                                            .numpy()
                                        )
                                finally:
                                    model.head.forward = original
                        mask, active = (
                            cpu["target_mask"].numpy()[..., 2:],
                            cpu["active_mask"].numpy(),
                        )
                        targets = cpu["targets"].numpy()[..., 2:]
                        for mode, values in predictions.items():
                            metrics[mode].extend(
                                daily_primary_ic(values, targets, mask, active)
                            )
                        for mode, name, normalized in (
                            ("bf16", "bf16_fp32_score_rank", errors),
                            ("bf16_fp32_head", "head_fp32_score_rank", head_errors),
                        ):
                            metrics[name].extend(
                                daily_primary_ic(
                                    predictions[mode], predictions["fp32"], mask, active
                                )
                            )
                            for day in range(len(active)):
                                selected = active[day] & mask[day].all(-1)
                                if selected.sum() < 20:
                                    continue
                                ref, value = (
                                    predictions["fp32"][day, selected],
                                    predictions[mode][day, selected],
                                )
                                diff = (value - value.mean(0)) - (ref - ref.mean(0))
                                normalized.extend(
                                    (
                                        np.sqrt(np.mean(diff**2, axis=0))
                                        / ref.std(0).clip(1e-12)
                                    ).tolist()
                                )
                    records.append(
                        {
                            "cell": cell,
                            "fold": fold,
                            "seed": seed,
                            "checkpoint_sha256": run["artifacts"]["selected.pt"],
                            "registered_selected_ic": run["selection_ic"],
                            "eager_selection_metrics": {
                                k: float(np.nanmean(v)) for k, v in metrics.items()
                            },
                            "mean_bf16_error_over_cross_sectional_std": float(
                                np.mean(errors)
                            ),
                            "mean_fp32_head_error_over_cross_sectional_std": float(
                                np.mean(head_errors)
                            ),
                            "selection_access": data.access_ledger.payload(),
                        }
                    )
                finally:
                    data.store.close()
                print(records[-1], flush=True)
    result = {
        "status": "completed",
        "source_sha256": sha256_file(Path(__file__)),
        "frozen_design_sha256": sha256_file(root / "frozen_design.json"),
        "financial_evaluation_scores_read": False,
        "diagnostic_only_no_precision_change": True,
        "records": records,
    }
    write_json_atomic(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    check(parser.parse_args().root)
