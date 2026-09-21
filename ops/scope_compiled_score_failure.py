"""Bound the observed compiler failure across the six actual C/D architectures."""

import gc
import inspect
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.data import V2DailyDataset, stage_name_count
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.model import DailyMultiHorizonModel, ModelConfig
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import (
    autocast_dtype,
    forward,
    model_batch,
    sequential_batches,
)
from brazil_rv.v2.train import (
    _cli_stage_indices,
    compile_forward,
    set_deterministic_seed,
)

PROJECT = Path(__file__).resolve().parents[1]


def contrast(actual, expected, valid):
    a, b = actual[valid], expected[valid]
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[finite] - b[finite])
    return dict(
        cells=len(a),
        finite=int(finite.sum()),
        max_absolute=float(delta.max(initial=0)),
        outside_tolerance=int((delta > 0.002 + 0.02 * np.abs(b[finite])).sum()),
        rms=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
    )


def main():
    from torch._inductor.codecache import PyCodeCache

    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    plan = bound_json(run["stage_d_width_plan"])
    root = Path(run["stage_d_width_plan"]["path"]).parent
    out = root / "scoring_failure/architecture_scope"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    cases = {
        name: Path(run["stage_c_refit_root"]) / "fits" / name / "F10_seed_11"
        for name in ("TE_full", "TE_wide", "GRU_early", "C6")
    }
    cases.update(
        {name: root / "fits" / name / "F10_seed_11" for name in ("TE_128", "GRU_96")}
    )
    write_json_atomic(
        out / "plan.json",
        dict(
            cases={k: binding(v / "selected.pt") for k, v in cases.items()},
            compiler=binding(Path(inspect.getfile(compile_forward))),
            rtol=0.02,
            atol=0.002,
            scope="Six saved F10 seed11 checkpoints: every126 original evaluation date in eager mode and the first original chronological compiled export batch under NaN-initialized temporary buffers. Compare original saved forecasts. No refit, optimizer step, held-out data, accounting replay or source/store change. A representative shape test does not qualify every other checkpoint or training/selection graph.",
        ),
    )
    results = {}
    for name, fit in cases.items():
        tick = perf_counter()
        cell = out / name
        cell.mkdir()
        payload = torch.load(fit / "selected.pt", map_location="cpu", weights_only=True)
        contract = payload["contract"]
        characteristic = not contract["pretrain_key"].startswith("s0_")
        config = (CharacteristicConfig if characteristic else ModelConfig)(
            **contract["config"]
        )
        horizons = config.horizons if characteristic else HORIZONS
        families = tuple(
            n
            for n, _ in (
                config.family_counts
                if characteristic
                else config.sidecar_feature_counts
            )
        )
        store = Path(plan["store"]["root"])
        _, _, rows, _ = _cli_stage_indices(store, "F", "F10")
        dataset = V2DailyDataset(
            store,
            rows,
            stage="evaluation",
            lookback=60,
            enabled_sidecars=families,
            include_fast=False,
            include_intraday=False,
            include_common_state=characteristic and "cross_market" in families,
            compact_names=True,
            purpose="evaluation",
        )
        prep = Round7Preprocessing.from_payload(contract["preprocessing"])
        loader = DataLoader(
            dataset,
            batch_sampler=sequential_batches(len(dataset)),
            collate_fn=partial(
                prep.collate,
                fixed_name_count=max(
                    contract["padded_name_count"], stage_name_count(dataset)
                ),
            ),
        )
        set_deterministic_seed(11)
        model = (
            (
                CharacteristicModel(config)
                if characteristic
                else DailyMultiHorizonModel(config)
            )
            .cuda()
            .eval()
        )
        model.load_state_dict(payload["model_state_dict"])
        values = np.zeros(
            (len(rows), len(dataset.store.isins), len(HORIZONS)), dtype=np.float32
        )
        valid = np.zeros_like(values, dtype=bool)
        indices = [HORIZONS.index(h) for h in horizons]
        offset = 0
        first_cpu = None
        first_eager = None
        for cpu in loader:
            batch = model_batch(cpu, torch.device("cuda"))
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))),
            ):
                prediction = (
                    forward(model, batch, characteristic=characteristic)
                    .float()
                    .mean(2)
                    .cpu()
                    .numpy()
                )
            if first_cpu is None:
                first_cpu = cpu
                first_eager = prediction.copy()
            names = cpu["name_index"].numpy()
            active = cpu["active_mask"].numpy()
            for i in range(len(names)):
                for h, j in enumerate(indices):
                    values[offset + i, names[i, active[i]], j] = prediction[
                        i, active[i], h
                    ]
                    valid[offset + i, names[i, active[i]], j] = True
            offset += len(names)
        np.savez_compressed(
            cell / "eager.npz", scores=values, valid=valid, date_indices=rows
        )
        saved = np.load(fit / "scores/scores.npy")
        saved_valid = np.load(fit / "scores/score_mask.npy")
        assert np.array_equal(valid, saved_valid)
        saved_comparison = contrast(saved, values, valid)
        batch = model_batch(first_cpu, torch.device("cuda"))
        before_modules = len(PyCodeCache.modules)
        compiled = compile_forward(model)

        def invoke(instance, inputs):
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))),
            ):
                return (
                    forward(instance, inputs, characteristic=characteristic)
                    .float()
                    .mean(2)
                    .cpu()
                    .numpy()
                )

        compiled_values = invoke(compiled, batch)
        modules = [
            m
            for m in PyCodeCache.modules[before_modules:]
            if hasattr(m, "empty_strided_cuda")
        ]
        assert modules
        originals = [m.empty_strided_cuda for m in modules]
        for module, original in zip(modules, originals):

            def initialized(*args, _original=original, **kwargs):
                tensor = _original(*args, **kwargs)
                if tensor.is_floating_point():
                    tensor.fill_(float("nan"))
                return tensor

            module.empty_strided_cuda = initialized
        poisoned = invoke(compiled, batch)
        for module, original in zip(modules, originals):
            module.empty_strided_cuda = original
        first_valid = np.broadcast_to(
            first_cpu["active_mask"].numpy()[..., None], first_eager.shape
        )
        np.savez_compressed(
            cell / "batch.npz",
            eager=first_eager,
            compiled=compiled_values,
            poisoned=poisoned,
            active=first_cpu["active_mask"].numpy(),
            names=first_cpu["name_index"].numpy(),
            dates=first_cpu["date_index"].numpy(),
        )
        for i, module in enumerate(modules):
            (cell / f"generated_{i}.py").write_bytes(Path(module.__file__).read_bytes())
        result = dict(
            checkpoint=binding(fit / "selected.pt"),
            original_scores=binding(fit / "scores/score_manifest.json"),
            config=contract["config"],
            saved_vs_eager=saved_comparison,
            compiled_vs_eager=contrast(compiled_values, first_eager, first_valid),
            poisoned_vs_eager=contrast(poisoned, first_eager, first_valid),
            seconds=perf_counter() - tick,
        )
        write_json_atomic(cell / "report.json", result)
        results[name] = result
        print(
            json.dumps(
                dict(
                    cell=name,
                    **{
                        k: v
                        for k, v in result.items()
                        if k.endswith("eager") or k == "seconds"
                    },
                )
            ),
            flush=True,
        )
        dataset.store.close()
        del model, compiled, batch, loader, dataset
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    write_json_atomic(out / "report.json", results)


if __name__ == "__main__":
    main()
