"""Diagnose the two saved/eager tolerance crossings without changing forecasts."""

from functools import partial
import inspect
import json
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
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import forward, model_batch, sequential_batches
from brazil_rv.v2.train import compile_forward, set_deterministic_seed
from scope_compiled_score_failure import contrast

PROJECT = Path(__file__).resolve().parents[1]


def main():
    from torch._inductor.codecache import PyCodeCache

    started = perf_counter()
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    design = bound_json(run["stage_d_width_plan"])
    root = Path(run["stage_d_width_plan"]["path"]).parent
    initial = root / "forecast_qualification"
    old = np.load(initial / "GRU_96/eager.npz")
    fit = root / "fits/GRU_96/F2_seed_11"
    saved = np.load(fit / "scores/scores.npy")
    bad = old["valid"] & (
        np.abs(saved - old["scores"]) > 0.002 + 0.02 * np.abs(old["scores"])
    )
    locations = np.argwhere(bad)
    assert len(locations) == 2 and len(set(locations[:, 0])) == 1
    out = root / "forecast_roundoff"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    write_json_atomic(
        out / "plan.json",
        dict(
            initial=binding(initial / "report.json"),
            checkpoint=binding(fit / "selected.pt"),
            compiler=binding(Path(inspect.getfile(compile_forward))),
            affected_coordinates=locations.tolist(),
            contrast="Fresh fixed compiled inference on every original F2/11 evaluation batch; repeat each with NaN-initialized compiler temporaries. Require exact default/poison equality and original tolerance versus saved forecasts, with every valid value finite. On the single crossing batch also compare eager FP32, eager FP16 and compiled FP16 using the same inputs/weights. Preserve the original failed eager tolerance and all model/score bytes; do not enlarge its threshold, retrain or modify economic outcomes. This is a numerical diagnosis, not automatic admission.",
        ),
    )
    payload = torch.load(fit / "selected.pt", map_location="cpu", weights_only=True)
    contract = payload["contract"]
    config = CharacteristicConfig(**contract["config"])
    dataset = V2DailyDataset(
        Path(design["store"]["root"]),
        old["date_indices"],
        stage="evaluation",
        lookback=60,
        enabled_sidecars=tuple(n for n, _ in config.family_counts),
        include_fast=False,
        include_intraday=False,
        include_common_state=True,
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
    model = CharacteristicModel(config).cuda().eval()
    model.load_state_dict(payload["model_state_dict"])
    before = len(PyCodeCache.modules)
    compiled = compile_forward(model)
    panels = {name: np.zeros_like(saved) for name in ("compiled", "poisoned")}
    offset = 0
    focus = None

    def invoke(instance, batch, amp=True):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return (
                forward(instance, batch, characteristic=True)
                .float()
                .mean(2)
                .cpu()
                .numpy()
            )

    for cpu in loader:
        batch = model_batch(cpu, torch.device("cuda"))
        ordinary = invoke(compiled, batch)
        modules = [
            m for m in PyCodeCache.modules[before:] if hasattr(m, "empty_strided_cuda")
        ]
        assert modules
        originals = [m.empty_strided_cuda for m in modules]
        for module, original in zip(modules, originals, strict=True):

            def initialized(*args, _original=original, **kwargs):
                tensor = _original(*args, **kwargs)
                if tensor.is_floating_point():
                    tensor.fill_(float("nan"))
                return tensor

            module.empty_strided_cuda = initialized
        poisoned = invoke(compiled, batch)
        for module, original in zip(modules, originals, strict=True):
            module.empty_strided_cuda = original
        names = cpu["name_index"].numpy()
        active = cpu["active_mask"].numpy()
        for label, values in (("compiled", ordinary), ("poisoned", poisoned)):
            for i in range(len(names)):
                for h, horizon in enumerate(config.horizons):
                    panels[label][
                        offset + i, names[i, active[i]], HORIZONS.index(horizon)
                    ] = values[i, active[i], h]
        if offset <= locations[0, 0] < offset + len(names):
            full = invoke(model, batch, amp=False)
            eager = invoke(model, batch)
            np.savez_compressed(
                out / "focus_batch.npz",
                fp32=full,
                eager=eager,
                compiled=ordinary,
                poisoned=poisoned,
                names=names,
                active=active,
                date_indices=cpu["date_index"].numpy(),
            )
            focus = []
            for day, name, horizon in locations:
                i = int(day - offset)
                j = int(np.flatnonzero(names[i] == name)[0])
                h = config.horizons.index(HORIZONS[horizon])
                focus.append(
                    dict(
                        index=[int(day), int(name), int(horizon)],
                        date=str(dataset.store.dates[old["date_indices"][day]]),
                        isin=dataset.store.isins[name],
                        saved=float(saved[day, name, horizon]),
                        original_eager=float(old["scores"][day, name, horizon]),
                        eager=float(eager[i, j, h]),
                        fp32=float(full[i, j, h]),
                        compiled=float(ordinary[i, j, h]),
                        poisoned=float(poisoned[i, j, h]),
                    )
                )
        offset += len(names)
    np.savez_compressed(out / "panels.npz", **panels)
    report = dict(
        initial=binding(initial / "report.json"),
        saved_vs_fresh_compiled=contrast(saved, panels["compiled"], old["valid"]),
        poisoned_vs_compiled=contrast(
            panels["poisoned"], panels["compiled"], old["valid"]
        ),
        exact_poisoned=bool(np.array_equal(panels["compiled"], panels["poisoned"])),
        exact_saved=bool(np.array_equal(saved, panels["compiled"])),
        focus=focus,
        seconds=perf_counter() - started,
    )
    write_json_atomic(out / "report.json", report)
    print(json.dumps(report), flush=True)
    dataset.store.close()
    for name in ("saved_vs_fresh_compiled", "poisoned_vs_compiled"):
        check = report[name]
        assert check["finite"] == check["cells"] and check["outside_tolerance"] == 0, (
            name
        )
    assert report["exact_poisoned"]


if __name__ == "__main__":
    main()
