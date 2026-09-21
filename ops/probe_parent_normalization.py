"""Check the affected parent selector's actual batch under allocator poisoning."""

import argparse
import inspect
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import autocast_dtype, forward, model_batch
from brazil_rv.v2.train import _cli_stage_indices, compile_forward
from scope_compiled_score_failure import contrast

PROJECT = Path(__file__).resolve().parents[1]


def main(qualified):
    from torch._inductor.codecache import PyCodeCache

    started = perf_counter()
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    plan = bound_json(run["stage_d_width_plan"])
    root = Path(run["stage_d_width_plan"]["path"]).parent
    out = (
        root
        / "scoring_failure/parent_scope"
        / ("qualified" if qualified else "original")
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    checkpoint = root / "fits/GRU_96/P_seed_11/selected.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    contract = payload["contract"]
    config = CharacteristicConfig(**contract["config"])
    store = Path(plan["store"]["root"])
    _, select, _, _ = _cli_stage_indices(store, "P", "pretrain_internal")
    rows = select[:16]
    write_json_atomic(
        out / "plan.json",
        dict(
            checkpoint=binding(checkpoint),
            compiler=binding(Path(inspect.getfile(compile_forward))),
            dates=rows.tolist(),
            rtol=0.02,
            atol=0.002,
            scope="First original 16-date parent selection batch, own fitted conditioning/full60/population and padded width. Eager and compiled forward with temporary NaN/zero/one allocation. No optimizer, selector replacement, new source or evaluation target consumption.",
        ),
    )
    dataset = V2DailyDataset(
        store,
        rows,
        stage="pretrain",
        purpose="selection",
        lookback=60,
        enabled_sidecars=tuple(n for n, _ in config.family_counts),
        include_fast=False,
        include_intraday=False,
        include_common_state=True,
        compact_names=True,
    )
    prep = Round7Preprocessing.from_payload(contract["preprocessing"])
    cpu = prep.collate(
        [dataset[i] for i in range(len(rows))],
        fixed_name_count=contract["padded_name_count"],
    )
    batch = model_batch(cpu, torch.device("cuda"))
    model = CharacteristicModel(config).cuda().eval()
    model.load_state_dict(payload["model_state_dict"])

    def invoke(instance):
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))),
        ):
            return (
                forward(instance, batch, characteristic=True)
                .float()
                .mean(2)
                .cpu()
                .numpy()
            )

    eager = invoke(model)
    before = len(PyCodeCache.modules)
    compiled = compile_forward(model)
    values = {"eager": eager, "compiled": invoke(compiled)}
    modules = [
        m for m in PyCodeCache.modules[before:] if hasattr(m, "empty_strided_cuda")
    ]
    assert modules
    for fill in (float("nan"), 0.0, 1.0):
        originals = [m.empty_strided_cuda for m in modules]
        for module, original in zip(modules, originals):

            def allocate(*args, _original=original, _fill=fill, **kwargs):
                result = _original(*args, **kwargs)
                if result.is_floating_point():
                    result.fill_(_fill)
                return result

            module.empty_strided_cuda = allocate
        values[str(fill)] = invoke(compiled)
        for module, original in zip(modules, originals):
            module.empty_strided_cuda = original
    active = np.broadcast_to(cpu["active_mask"].numpy()[..., None], eager.shape)
    results = {
        name: contrast(value, eager, active)
        for name, value in values.items()
        if name != "eager"
    }
    np.savez_compressed(out / "outputs.npz", **values, active=active)
    for i, module in enumerate(modules):
        (out / f"generated_{i}.py").write_bytes(Path(module.__file__).read_bytes())
    write_json_atomic(
        out / "report.json", dict(comparisons=results, seconds=perf_counter() - started)
    )
    print(json.dumps(results), flush=True)
    if qualified:
        assert all(
            r["finite"] == r["cells"] and r["outside_tolerance"] == 0
            for r in results.values()
        )
        assert np.array_equal(values["nan"], values["0.0"]) and np.array_equal(
            values["0.0"], values["1.0"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualified", action="store_true")
    main(parser.parse_args().qualified)
