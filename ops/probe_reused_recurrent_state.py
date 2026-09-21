"""Isolate checkpoint reload and the existing restored fit-only diagnostic."""

import argparse
import ast
import inspect
import json
from functools import partial
from pathlib import Path

import torch
import numpy as np

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.characteristic_model import CharacteristicConfig, CharacteristicModel
from brazil_rv.v2.data import V2DailyDataset, stage_name_count
from brazil_rv.v2.data_repair import binding
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import (
    autocast_dtype,
    forward,
    model_batch,
    member_loss,
    optimizer_step,
    recipe_optimizer,
)
from brazil_rv.v2.train import compile_forward, set_deterministic_seed
from brazil_rv.v2.training_diagnostics import probe
from probe_capacity_score_batch import compare

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--boundary",
        choices=(
            "diagnostic",
            "device",
            "training",
            "optimizer",
            "allocation",
            "allocation_trace",
        ),
        default="diagnostic",
    )
    parser.add_argument("--out-of-place", action="store_true")
    parser.add_argument("--barrier", action="store_true")
    parser.add_argument("--qualified-runtime", action="store_true")
    args = parser.parse_args()
    boundary = args.boundary
    torch.set_num_threads(1)
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["stage_d_width_plan"]["path"]).parent
    audit = root / "scoring_failure"
    out = audit / (
        "reused_state_probe" if boundary == "diagnostic" else boundary + "_state_probe"
    )
    if boundary == "allocation":
        out = out / "qualified"
    if boundary == "allocation_trace":
        out = out / "controlled_initialization"
    if args.out_of_place:
        out = out / "out_of_place"
    if args.barrier:
        out = out / "barrier"
    if args.qualified_runtime:
        out = out / "native_norm"
    out.mkdir(parents=True, exist_ok=False)
    (out / "executed.py").write_bytes(Path(__file__).read_bytes())
    fit = root / "fits/GRU_96/F10_seed_11"
    payload = torch.load(fit / "selected.pt", map_location="cpu", weights_only=True)
    terminal = torch.load(
        fit / "epochs/epoch_009.pt", map_location="cpu", weights_only=True
    )
    contract = payload["contract"]
    config = CharacteristicConfig(**contract["config"])
    cpu = torch.load(audit / "batch_probe/cpu_batch.pt", weights_only=True)
    # The other seed's preprocessed features must not enter this seed's model.
    # Reconstruct these same two dates with this checkpoint's own conditioning.
    store = Path(
        json.loads(Path(run["stage_d_width_plan"]["path"]).read_text())["store"]["root"]
    )
    kwargs = dict(
        lookback=60,
        enabled_sidecars=tuple(n for n, _ in config.family_counts),
        include_fast=False,
        include_intraday=False,
        include_common_state=True,
        compact_names=True,
    )
    dataset = V2DailyDataset(
        store,
        cpu["date_index"].numpy(),
        stage="evaluation",
        purpose="evaluation",
        **kwargs,
    )
    preparation = Round7Preprocessing.from_payload(contract["preprocessing"])
    collate = partial(
        preparation.collate,
        fixed_name_count=max(contract["padded_name_count"], stage_name_count(dataset)),
    )
    batch = model_batch(collate([dataset[0], dataset[1]]), torch.device("cuda"))
    dates = [contract["probe_date_indices"][0], contract["probe_date_indices"][-1]]
    training = V2DailyDataset(
        store,
        dates,
        stage="finetune",
        purpose="training",
        target_window_indices=np.array(contract["fit_target_window"]),
        **kwargs,
    )
    diagnostic_batch = model_batch(
        collate([training[0], training[1]]), torch.device("cuda")
    )
    write_json_atomic(
        out / "plan.json",
        dict(
            selected=binding(fit / "selected.pt"),
            terminal=binding(fit / "epochs/epoch_009.pt"),
            evaluation_dates=cpu["date_index"].tolist(),
            diagnostic_fit_dates=dates,
            boundary=boundary,
            compiler_inplace_buffers=not args.out_of_place,
            diagnostic_generated_kernel_barrier=args.barrier,
            imported_compiler_runtime=binding(Path(inspect.getfile(compile_forward))),
            allocation_trace_rule="Record every floating temporary allocation by generated-code location; initialize all other allocations to zero on every invocation so allocator contents cannot carry across contrasts. Poison selected groups, then bisect groups whose output has fewer finite cells than the eager control. Retain every test and joint-only interaction; no economic selection. Earlier unconstrained allocation tracing is not qualified to identify unique defective allocation sites because poisoned cache memory can survive between calls.",
            contrast=(
                "One same-object selected/terminal/selected reload sequence; then the original fit-only diagnostic with saved optimizer restored."
                if boundary == "diagnostic"
                else "Same checkpoint and compiled object before/after the selected isolated boundary: no-op device application, eager training-mode forward, restored fit-only optimizer update, or initial-content control of compiler temporary allocations (NaN/zero/one). No checkpoint reload sequence in this branch."
            )
            + " Compare eager/compiled outputs, parameter addresses and exact parameter restoration. No new epoch, fitted statistic, evaluated economics or source change.",
        ),
    )
    set_deterministic_seed(11)
    model = CharacteristicModel(config).cuda().eval()
    model.load_state_dict(payload["model_state_dict"])
    if args.out_of_place:
        torch._inductor.config.inplace_buffers = False
    compiled = compile_forward(model)
    values = {}
    addresses = {}

    def call(label, instance):
        addresses[label + "_before"] = {
            k: v.data_ptr() for k, v in model.named_parameters()
        }
        with (
            torch.no_grad(),
            torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))),
        ):
            values[label] = (
                forward(instance, batch, characteristic=True).clone().float().cpu()
            )
        addresses[label + "_after"] = {
            k: v.data_ptr() for k, v in model.named_parameters()
        }

    call("selected_eager", model)
    call("selected_compiled", compiled)
    if boundary != "diagnostic":
        before = {k: v.clone() for k, v in model.state_dict().items()}
        if boundary == "device":
            model.to(torch.device("cuda"))
        elif boundary == "optimizer":
            from brazil_rv.v2.contract import HORIZONS

            recipe = contract["recipe"]
            optimizer = recipe_optimizer(
                model,
                cuda=True,
                learning_rate=recipe["learning_rate"],
                transferred=contract["transferred_parameters"],
                transferred_multiplier=recipe["transferred_multiplier"],
            )
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            model.train()
            heads = [HORIZONS.index(h) for h in config.horizons]

            def closure():
                with torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))):
                    prediction = forward(
                        model, diagnostic_batch, characteristic=True
                    ).float()
                return member_loss(
                    prediction,
                    diagnostic_batch["targets"][..., heads],
                    diagnostic_batch["target_mask"][..., heads]
                    & diagnostic_batch["active_mask"][..., None],
                )

            optimizer_step(
                model,
                optimizer,
                closure,
                contract["rho"],
                adaptive=recipe["adaptive"],
                eta=recipe["eta"],
                scaler=torch.amp.GradScaler("cuda", init_scale=256.0),
            )
            model.load_state_dict(before)
            model.eval()
        elif boundary in ("allocation", "allocation_trace"):
            from torch._inductor.codecache import PyCodeCache

            modules = [
                m
                for m in PyCodeCache.modules
                if hasattr(m, "call") and hasattr(m, "empty_strided_cuda")
            ]
            assert modules
            if args.barrier:
                symbol = "triton_red_fused_add_native_layer_norm_scalar_tensor_where_73"
                original_module = next(m for m in modules if hasattr(m, symbol))
                source = Path(original_module.__file__).read_text()
                tree = ast.parse(source)
                node = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id == symbol
                )
                kernel = node.value.args[1].value
                marker = "    tmp28 = tmp28_tmp[:, None]\n    for roffset"
                assert kernel.count(marker) == 1
                fixed = kernel.replace(
                    marker,
                    "    tmp28 = tmp28_tmp[:, None]\n    tl.debug_barrier()\n    for roffset",
                )
                assert source.count(kernel) == 1
                fixed_source = source.replace(kernel, fixed)
                (out / "kernel_original.py").write_text(kernel)
                (out / "kernel_barrier.py").write_text(fixed)
                fixed_module = PyCodeCache.load(fixed_source)
                setattr(original_module, symbol, getattr(fixed_module, symbol))
            write_json_atomic(
                out / "generated_modules.json",
                [binding(Path(m.__file__)) for m in modules],
            )
            for i, module in enumerate(modules):
                (out / f"generated_{i}.py").write_bytes(
                    Path(module.__file__).read_bytes()
                )
            originals = [m.empty_strided_cuda for m in modules]
            if boundary == "allocation_trace":
                inventory, records, defects, interactions = {}, [], [], []
                selected = set()
                for module_index, (module, original) in enumerate(
                    zip(modules, originals)
                ):

                    def located(
                        *args, _original=original, _module_index=module_index, **kwargs
                    ):
                        location = (
                            f"{_module_index}:{inspect.currentframe().f_back.f_lineno}"
                        )
                        tensor = _original(*args, **kwargs)
                        if tensor.is_floating_point():
                            inventory[location] = dict(
                                shape=list(tensor.shape),
                                stride=list(tensor.stride()),
                                dtype=str(tensor.dtype),
                            )
                            tensor.fill_(float("nan") if location in selected else 0.0)
                        return tensor

                    module.empty_strided_cuda = located
                call("trace_inventory", compiled)

                def test(locations):
                    selected.clear()
                    selected.update(locations)
                    label = f"trace_{len(records)}"
                    call(label, compiled)
                    result = compare(values[label], values["selected_eager"])
                    records.append(dict(locations=locations, result=result))
                    return result["finite_a"] < result["finite_b"]

                def isolate(locations):
                    if len(locations) == 1:
                        defects.append(locations[0])
                        return
                    halves = (
                        locations[: len(locations) // 2],
                        locations[len(locations) // 2 :],
                    )
                    bad = [part for part in halves if test(part)]
                    if not bad:
                        interactions.append(locations)
                    for part in bad:
                        isolate(part)

                candidates = list(inventory)
                assert test(candidates)
                isolate(candidates)
                for module, original in zip(modules, originals):
                    module.empty_strided_cuda = original
                write_json_atomic(
                    out / "allocation_trace.json",
                    dict(
                        inventory=inventory,
                        tests=records,
                        defects=defects,
                        interactions=interactions,
                    ),
                )
                torch.save(values, out / "predictions.pt")
                print(
                    json.dumps(
                        dict(
                            allocations=len(inventory),
                            tests=len(records),
                            defects=defects,
                            interactions=interactions,
                        )
                    ),
                    flush=True,
                )
                dataset.store.close()
                training.store.close()
                return
            for label, fill in (("nan", float("nan")), ("zero", 0.0), ("one", 1.0)):
                for module, original in zip(modules, originals):

                    def initialized(*args, _original=original, _fill=fill, **kwargs):
                        tensor = _original(*args, **kwargs)
                        if tensor.is_floating_point():
                            tensor.fill_(_fill)
                        return tensor

                    module.empty_strided_cuda = initialized
                call("allocation_" + label, compiled)
            for module, original in zip(modules, originals):
                module.empty_strided_cuda = original
        else:
            model.train()
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=autocast_dtype(torch.device("cuda"))),
            ):
                forward(model, diagnostic_batch, characteristic=True)
            model.eval()
        call("after_eager", model)
        call("after_compiled", compiled)
        report = {
            "before": compare(values["selected_compiled"], values["selected_eager"]),
            "eager": compare(values["after_eager"], values["selected_eager"]),
            "compiled": compare(values["after_compiled"], values["selected_eager"]),
            "changed_state": [
                k
                for k, v in before.items()
                if not torch.equal(v, model.state_dict()[k])
            ],
        }
        if boundary == "allocation":
            report["allocation"] = {
                k: compare(v, values["selected_eager"])
                for k, v in values.items()
                if k.startswith("allocation_")
            }
        torch.save(values, out / "predictions.pt")
        write_json_atomic(out / "parameter_addresses.json", addresses)
        write_json_atomic(out / "report.json", report)
        print(json.dumps(report), flush=True)
        dataset.store.close()
        training.store.close()
        return
    model.load_state_dict(terminal["model_state_dict"])
    call("terminal_eager", model)
    call("terminal_compiled", compiled)
    model.load_state_dict(payload["model_state_dict"])
    call("reloaded_eager", model)
    call("reloaded_compiled", compiled)
    recipe = contract["recipe"]
    report = {
        "selected_compiled": compare(
            values["selected_compiled"], values["selected_eager"]
        ),
        "terminal_compiled": compare(
            values["terminal_compiled"], values["terminal_eager"]
        ),
        "reloaded_compiled": compare(
            values["reloaded_compiled"], values["reloaded_eager"]
        ),
        "eager_reload_exact": compare(
            values["reloaded_eager"], values["selected_eager"]
        ),
    }
    write_json_atomic(out / "reload_report.json", report)
    optimizer = recipe_optimizer(
        model,
        cuda=True,
        learning_rate=recipe["learning_rate"],
        transferred=contract["transferred_parameters"],
        transferred_multiplier=recipe["transferred_multiplier"],
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = probe(
        model,
        optimizer,
        diagnostic_batch,
        characteristic=True,
        horizons=config.horizons,
        rho=contract["rho"],
        adaptive=recipe["adaptive"],
        eta=recipe["eta"],
        amp_dtype=autocast_dtype(torch.device("cuda")),
    )
    call("after_probe_eager", model)
    call("after_probe_compiled", compiled)
    report.update(
        probe=result,
        changed_state=[
            k for k, v in before.items() if not torch.equal(v, model.state_dict()[k])
        ],
        after_probe_eager=compare(
            values["after_probe_eager"], values["selected_eager"]
        ),
        after_probe_compiled=compare(
            values["after_probe_compiled"], values["selected_eager"]
        ),
    )
    torch.save(values, out / "predictions.pt")
    write_json_atomic(out / "parameter_addresses.json", addresses)
    write_json_atomic(out / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "probe"}), flush=True)
    dataset.store.close()
    training.store.close()


if __name__ == "__main__":
    main()
