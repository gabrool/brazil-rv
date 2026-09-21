"""One actual failing-fit input: isolate eager/compiled state and mode effects."""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.data import V2DailyDataset, stage_name_count
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import autocast_dtype, forward, model_batch
from brazil_rv.v2.train import (
    _cli_stage_indices,
    compile_forward,
    set_deterministic_seed,
)

PROJECT = Path(__file__).resolve().parents[1]


def compare(a, b):
    valid = torch.isfinite(a) & torch.isfinite(b)
    return dict(
        shape=list(a.shape),
        finite_a=int(torch.isfinite(a).sum()),
        finite_b=int(torch.isfinite(b).sum()),
        max_absolute=float((a - b).abs()[valid].max()) if valid.any() else None,
        exact=bool(torch.equal(a, b)),
    )


def main():
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    plan = bound_json(run["stage_d_width_plan"])
    root = Path(run["stage_d_width_plan"]["path"]).parent
    out = root / "scoring_failure/batch_probe"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    checkpoint = root / "fits/GRU_96/F10_seed_29/selected.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = CharacteristicConfig(**payload["contract"]["config"])
    store = Path(plan["store"]["root"])
    _, _, rows, _ = _cli_stage_indices(store, "F", "F10")
    dataset = V2DailyDataset(
        store,
        rows,
        stage="evaluation",
        lookback=60,
        enabled_sidecars=tuple(n for n, _ in config.family_counts),
        include_fast=False,
        include_intraday=False,
        include_common_state=True,
        compact_names=True,
        purpose="evaluation",
    )
    preparation = Round7Preprocessing.from_payload(payload["contract"]["preprocessing"])
    collate = partial(
        preparation.collate,
        fixed_name_count=max(
            payload["contract"]["padded_name_count"], stage_name_count(dataset)
        ),
    )
    cpu = next(iter(DataLoader(dataset, batch_sampler=[[0, 1]], collate_fn=collate)))
    write_json_atomic(
        out / "plan.json",
        dict(
            checkpoint=binding(checkpoint),
            date_indices=cpu["date_index"].tolist(),
            contrast="Single actual first evaluation batch, identical loaded model and original Float16 autocast. Eager repeated, compiled, then eager again; compare parameter/input immutability and prior saved score exports. No refit or altered graph/selector. Original training deterministic runtime flags restored before comparison.",
        ),
    )
    torch.save(cpu, out / "cpu_batch.pt")
    set_deterministic_seed(29)
    model = CharacteristicModel(config).cuda().eval()
    model.load_state_dict(payload["model_state_dict"])
    batch = model_batch(cpu, torch.device("cuda"))
    inputs = {k: v.clone() for k, v in batch.items()}
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    results = {}

    def invoke(label, instance):
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))),
        ):
            result = forward(instance, batch, characteristic=True).clone()
        torch.cuda.synchronize()
        results[label] = result.float().cpu()

    invoke("eager_first", model)
    invoke("eager_repeat", model)
    compiled = compile_forward(model)
    invoke("compiled", compiled)
    invoke("eager_after", model)
    report = dict(
        eager_repeat=compare(results["eager_first"], results["eager_repeat"]),
        compiled=compare(results["compiled"], results["eager_first"]),
        eager_after=compare(results["eager_after"], results["eager_first"]),
        changed_inputs=[k for k, v in inputs.items() if not torch.equal(v, batch[k])],
        changed_state=[
            k for k, v in weights.items() if not torch.equal(v, model.state_dict()[k])
        ],
        training_modules=[k for k, m in model.named_modules() if m.training],
        names=cpu["name_index"].tolist(),
        dates=cpu["date_index"].tolist(),
    )
    torch.save(results, out / "predictions.pt")
    active = cpu["active_mask"].numpy()
    names = cpu["name_index"].numpy()
    for mode in ("eager", "compiled"):
        saved = np.load(root / f"scoring_failure/{mode}/scores/scores.npy")
        original = np.stack([saved[i, names[i]][:, [2, 3, 4]] for i in range(2)])
        current = results[mode if mode == "compiled" else "eager_first"].mean(2).numpy()
        report["previous_" + mode] = compare(
            torch.from_numpy(current[active]), torch.from_numpy(original[active])
        )
    write_json_atomic(out / "report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"names", "dates"}}),
        flush=True,
    )
    dataset.store.close()


if __name__ == "__main__":
    main()
