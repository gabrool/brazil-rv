"""Actual compiled ASAM update under controlled temporary-memory contents."""

from copy import deepcopy
from functools import partial
import gc
import inspect
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.data import V2DailyDataset
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import (
    TrainingObjective,
    model_batch,
    optimizer_step,
    recipe_optimizer,
)
from brazil_rv.v2.train import (
    DateBatchSampler,
    _cli_stage_indices,
    _restore_rng,
    _rng_state,
    compile_forward,
    set_deterministic_seed,
)

PROJECT = Path(__file__).resolve().parents[1]


def main():
    from torch._inductor.codecache import PyCodeCache

    torch.set_num_threads(1)
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    investigation = bound_json(run["scaling_investigation"])
    refit = bound_json(run["stage_c_refit_plan"])
    out = Path(run["scaling_investigation"]["path"]).parent / "training_probe"
    out.mkdir(exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    cases = {
        arm: Path(run["stage_c_refit_root"]) / "fits" / arm / "F10_seed_29"
        for arm in investigation["arms"]
    }
    write_json_atomic(
        out / "plan.json",
        dict(
            cases={k: binding(v / "selected.pt") for k, v in cases.items()},
            runtime={
                name: binding(Path(inspect.getfile(fn)))
                for name, fn in [
                    ("compile", compile_forward),
                    ("step", optimizer_step),
                    ("model", CharacteristicModel),
                ]
            },
            scope="Both actual corrected F10/29 attention checkpoints, selected because this is the largest observed reversal and seed29 has a changed parent stopping trajectory. First original epoch-one shuffled full-population training batch, own conditioning/full60/target window. Original C compiler runtime, actual two-pass ASAM and selected optimizer state. Restore weights, optimizer and identical RNG before each normal/repeat/NaN/zero/one temporary-allocation variant. No adopted optimizer update, refit, new selection, data change or held-out read. This bounds one actual training shape per width, not all historical training kernels.",
        ),
    )
    reports = []
    for arm, fit in cases.items():
        tick = perf_counter()
        payload = torch.load(fit / "selected.pt", map_location="cpu", weights_only=True)
        contract = payload["contract"]
        config = CharacteristicConfig(**contract["config"])
        store = Path(refit["store"]["root"])
        rows, _, _, window = _cli_stage_indices(store, "F", "F10")
        dataset = V2DailyDataset(
            store,
            rows,
            stage="finetune",
            target_window_indices=window,
            lookback=60,
            enabled_sidecars=tuple(k for k, _ in config.family_counts),
            include_fast=False,
            include_intraday=False,
            include_common_state=True,
            compact_names=True,
            purpose="training",
        )
        prep = Round7Preprocessing.from_payload(contract["preprocessing"])
        sampler = DateBatchSampler(rows, seed=29)
        sampler.set_epoch(1)
        cpu = next(
            iter(
                DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    collate_fn=partial(
                        prep.collate, fixed_name_count=contract["padded_name_count"]
                    ),
                )
            )
        )
        cell = out / arm
        cell.mkdir()
        torch.save(cpu, cell / "batch.pt")
        set_deterministic_seed(29)
        model = CharacteristicModel(config).cuda().train()
        batch = model_batch(cpu, torch.device("cuda"))
        optimizer = recipe_optimizer(
            model,
            cuda=True,
            learning_rate=contract["recipe"]["learning_rate"],
            transferred=contract["transferred_parameters"],
            transferred_multiplier=contract["recipe"]["transferred_multiplier"],
        )
        objective = TrainingObjective(
            model,
            characteristic=True,
            head_indices=[HORIZONS.index(h) for h in config.horizons],
            loss_kind=contract["loss"],
            cuda=True,
        )
        compiled = compile_forward(objective)

        def reset():
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(deepcopy(payload["optimizer_state_dict"]))
            optimizer.zero_grad(set_to_none=True)

        def invoke():
            diagnostic = {}
            scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
            loss, gap = optimizer_step(
                model,
                optimizer,
                lambda: compiled(batch),
                contract["rho"],
                adaptive=contract["recipe"]["adaptive"],
                eta=contract["recipe"]["eta"],
                diagnostics=diagnostic,
                scaler=scaler,
            )
            torch.cuda.synchronize()
            return dict(loss=loss, gap=gap, diagnostics=diagnostic)

        reset()
        before = len(PyCodeCache.modules)
        invoke()
        modules = [
            m for m in PyCodeCache.modules[before:] if hasattr(m, "empty_strided_cuda")
        ]
        assert modules
        originals = [m.empty_strided_cuda for m in modules]
        for i, module in enumerate(modules):
            (cell / f"generated_{i}.py").write_bytes(Path(module.__file__).read_bytes())
        rng = _rng_state()
        baseline = None
        results = []
        for label, fill in [
            ("normal", None),
            ("repeat", None),
            ("nan", float("nan")),
            ("zero", 0.0),
            ("one", 1.0),
        ]:
            reset()
            _restore_rng(rng)
            for module, original in zip(modules, originals, strict=True):
                if fill is None:
                    module.empty_strided_cuda = original
                else:

                    def initialized(*args, _original=original, _fill=fill, **kwargs):
                        value = _original(*args, **kwargs)
                        if value.is_floating_point():
                            value.fill_(_fill)
                        return value

                    module.empty_strided_cuda = initialized
            try:
                result = invoke()
                state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                if baseline is None:
                    baseline = state
                result.update(
                    label=label,
                    finite=all(bool(torch.isfinite(v).all()) for v in state.values()),
                    max_parameter_difference=max(
                        float((state[k] - baseline[k]).abs().max()) for k in state
                    ),
                    exact_parameters=all(
                        torch.equal(state[k], baseline[k]) for k in state
                    ),
                )
                torch.save(state, cell / f"{label}_after.pt")
            except Exception as error:
                result = dict(
                    label=label, error=repr(error), finite=False, exact_parameters=False
                )
            finally:
                for module, original in zip(modules, originals, strict=True):
                    module.empty_strided_cuda = original
            results.append(result)
            write_json_atomic(cell / "progress.json", results)
        report = dict(
            arm=arm,
            checkpoint=binding(fit / "selected.pt"),
            date_indices=cpu["date_index"].tolist(),
            results=results,
            generated_modules=len(modules),
            seconds=perf_counter() - tick,
            passed=all(r.get("finite") and r.get("exact_parameters") for r in results),
        )
        write_json_atomic(cell / "report.json", report)
        reports.append(report)
        print(json.dumps(report), flush=True)
        dataset.store.close()
        model = compiled = objective = optimizer = batch = dataset = None
        torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()
    write_json_atomic(
        out / "report.json",
        dict(results=reports, passed=all(r["passed"] for r in reports)),
    )
    run = json.loads(pointer.read_text())
    run["scaling_training_probe"] = binding(out / "report.json")
    write_json_atomic(pointer, run)


if __name__ == "__main__":
    main()
