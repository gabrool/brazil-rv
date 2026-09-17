"""Matched, resumable local-GPU continuations through the portfolio account."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from brazil_rv.execution.allocation import AllocationConfig
from brazil_rv.execution.portfolio_policy import exact_replay, policy_ledger_config
from .artifacts import sha256_file, write_json_atomic
from .characteristic_model import CharacteristicModel
from .contract import ALLOWED_SEEDS, DEVELOPMENT_FOLDS, HORIZONS
from .data import V2DailyDataset, restore_name_axis, stage_name_count
from .model import DailyMultiHorizonModel
from .objective_program import CELLS, RECIPES
from .objective_readouts import calibration
from .opportunity_research import bound, checked
from .portfolio_objective import (
    NeuralPreference,
    TensorPreference,
    active_view,
    clone_account,
    full_preferences,
    utility_path,
)
from .portfolio_program import PROJECT, read
from .portfolio_training import load_data, save_checkpoint, utility_series
from .research_rounds import _git_identity
from .round7 import SCREEN_FOLDS, configuration
from .round7_preprocessing import Round7Preprocessing
from .round7_training import (
    DateTensorCache,
    autocast_dtype,
    daily_primary_ic,
    learning_rate_fraction,
    optimizer_step,
    recipe_optimizer,
    sequential_batches,
)
from .train import (
    _cli_stage_indices,
    _restore_rng,
    _rng_state,
    compile_forward,
    set_deterministic_seed,
)

VARIANTS = ("rank", "hybrid", "utility")
SETTINGS = {
    "maximum_epochs": 12,
    "minimum_epochs": 6,
    "patience": 5,
    "block_sessions": 32,
    "neural_batch_dates": 16,
    "learning_rate": 3e-5,
    "new_head_learning_rate": 1e-4,
    "smooth_rank_temperature": 0.1,
    "head_daily_return_unit": 1e-4,
    "loss_scale": "equal shared-encoder gradient norms, median of three initial fit-only blocks",
    "economic_selector": "mean daily net excess CDI minus 2.5*causal variance",
}


def source_checkpoint(old, arm, fold, seed):
    path = old / "phase3/fits" / arm / "neutral" / f"{fold}_seed_{seed}"
    manifest = read(path / "run_manifest.json")
    checkpoint = path / "selected.pt"
    if sha256_file(checkpoint) != manifest["artifacts"]["selected.pt"]:
        raise ValueError("neutral warm start changed")
    return checkpoint


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    old = Path(read(PROJECT / "docs/v2_decision_run.json")["root"])
    source = read(old / "phase3/frozen_design.json")
    accepted = read(PROJECT / "docs/v2_data_inputs.json")["store"]
    if source["store"]["manifest_sha256"] != accepted["manifest_sha256"]:
        raise ValueError("objective source is not the accepted repaired store")
    design = {
        "implementation": _git_identity(),
        "settings": SETTINGS,
        "store": accepted,
        "decision_root": str(old),
        "source_design": bound(old / "phase3/frozen_design.json"),
        "registration": bound(
            PROJECT / "research/preregistrations/v2_portfolio_objective.md"
        ),
        "allocation": asdict(AllocationConfig()),
        "ledger": asdict(policy_ledger_config()),
        "recipes": {a: asdict(r) for a, r in RECIPES.items()},
        "parents": {
            a: {
                f: {
                    str(s): bound(source_checkpoint(old, a, f, s))
                    for s in ALLOWED_SEEDS
                }
                for f in DEVELOPMENT_FOLDS
            }
            for a in CELLS
        },
        "mappings": {
            f: bound(old / "phase3/mappings" / f"{f}.json") for f in DEVELOPMENT_FOLDS
        },
        "caches": {a: bound(old / "cache" / a / "policy_data.json") for a in CELLS},
        "runtime": {
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
        },
        "heldout_accessed": False,
        "forward_capture": False,
    }
    write_json_atomic(root / "training_design.json", design)


def preparation(design, data, arm, fold, seed, *, include_evaluation=False):
    """One exact compact cache per warm start, shared across objective variants."""
    store = Path(design["store"]["root"])
    if sha256_file(store / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("objective store changed")
    path = checked(design["parents"][arm][fold][str(seed)])
    parent = torch.load(path, map_location="cpu", weights_only=True)
    contract = parent["contract"]
    if (parent["stage"], parent["fold"], parent["seed"]) != ("F", fold, seed):
        raise ValueError("warm start is not matched to this trajectory")
    if contract["store_manifest_sha256"] != design["store"]["manifest_sha256"]:
        raise ValueError("warm-start preprocessing binds another store")
    prep = Round7Preprocessing.from_payload(contract["preprocessing"])
    config = configuration(CELLS[arm], read(store / "manifest.json")["feature_names"])
    if asdict(config) != contract["config"]:
        raise ValueError("warm-start architecture changed")
    characteristic = arm != "C6"
    horizons = config.horizons if characteristic else HORIZONS
    families = tuple(
        n
        for n, _ in (
            config.family_counts if characteristic else config.sidecar_feature_counts
        )
    )
    options = dict(
        lookback=60,
        enabled_sidecars=families,
        include_intraday=False,
        include_fast=False,
        include_common_state=characteristic and bool(families),
        compact_names=True,
    )
    fit, select, evaluation, fit_window = _cli_stage_indices(store, "F", fold)
    # Burn-in begins after all fitting ranking labels have matured.
    # Never use a purge observation whose outcome trained the warm start.
    selection_start = int(fit_window[-1] + 1)
    if selection_start != int(select[0]):
        raise ValueError(
            "selection must start after the warm start's last label endpoint"
        )
    parts = {
        "fit": V2DailyDataset(
            store,
            fit,
            stage="finetune",
            purpose="training",
            target_window_indices=fit_window,
            **options,
        ),
        "selection": V2DailyDataset(
            store, select, stage="finetune", purpose="selection", **options
        ),
    }
    if include_evaluation:
        # Post-selection embargo is causal inventory burn-in. It is never used
        # in fitting, scaling or checkpoint selection.
        rows = np.arange(int(select[-1] + 1), int(evaluation[-1] + 1))
        parts["evaluation"] = V2DailyDataset(
            store, rows, stage="evaluation", purpose="evaluation", **options
        )
    width = stage_name_count(*parts.values())
    caches, axes = {}, {}
    device = torch.device("cuda")
    origin = int(data.inputs.session_indices[0])
    try:
        for name, dataset in parts.items():
            indices = fit if name == "fit" else select if name == "selection" else rows
            axes[name] = indices - origin
            if axes[name][0] < 0 or axes[name][-1] >= len(data.valid):
                raise ValueError("account cache does not cover neural dates")
            loader = DataLoader(
                dataset,
                batch_sampler=sequential_batches(len(dataset)),
                collate_fn=partial(prep.collate, fixed_name_count=width),
                pin_memory=True,
            )
            caches[name] = DateTensorCache(
                loader, indices, len(data.inputs.security_ids), device
            )
        isins = tuple(parts["fit"].store.isins)
        if isins != tuple(data.inputs.security_ids):
            raise ValueError("neural and economic permanent identities differ")
    finally:
        for dataset in parts.values():
            dataset.store.close()
    mapping = calibration(read(checked(design["mappings"][fold]))["arms"][arm])
    model = (
        CharacteristicModel(config)
        if characteristic
        else DailyMultiHorizonModel(config)
    ).to(device)
    model.load_state_dict(parent["model_state_dict"], strict=True)
    neural = NeuralPreference(
        model, characteristic=characteristic, horizons=horizons, calibration=mapping
    ).to(device)
    active = np.zeros_like(data.valid)
    for name, cache in caches.items():
        active[axes[name]] = restore_name_axis(
            cache.tensors["active_mask"].cpu().numpy(),
            cache.names.cpu().numpy(),
            len(isins),
        )
    view = active_view(data, np.arange(len(active)), active)
    return (
        neural,
        caches,
        axes,
        view,
        {
            "parent": bound(path),
            "padded_names": width,
            "cache_bytes": sum(c.bytes for c in caches.values()),
            "fit": axes["fit"].tolist(),
            "selection": axes["selection"].tolist(),
            "ranking_fit_target_window": fit_window.tolist(),
            "preprocessing": prep.payload(),
            "selection_inventory": "empty after all warm-start fit labels have matured",
        },
    )


def encoded(neural_forward, cache, positions, count):
    outputs, preferences, losses = [], [], []
    for local in sequential_batches(len(positions), SETTINGS["neural_batch_dates"]):
        chosen = np.asarray(positions)[local]
        scores, pref, loss = neural_forward(cache.gather(chosen))
        outputs.append(scores)
        preferences.append(full_preferences(pref, cache.names[chosen], count))
        losses.append(loss * len(chosen) / len(positions))
    return torch.cat(outputs), torch.cat(preferences), sum(losses)


def readout(model, neural_forward, cache, rows, data):
    model.eval()
    scores, prefs, daily = [], [], []
    target_heads = [HORIZONS.index(h) for h in (3, 5, 10)]
    with torch.no_grad():
        for positions in sequential_batches(len(rows)):
            batch = cache.gather(positions)
            predicted, preference, _ = neural_forward(batch)
            daily.extend(
                daily_primary_ic(
                    predicted.cpu().numpy(),
                    batch["targets"][..., target_heads].cpu().numpy(),
                    batch["target_mask"][..., target_heads].cpu().numpy(),
                    batch["active_mask"].cpu().numpy(),
                ).tolist()
            )
            scores.append(
                restore_name_axis(
                    predicted.cpu().numpy(),
                    cache.names[positions].cpu().numpy(),
                    len(data.inputs.security_ids),
                )
            )
            prefs.append(
                full_preferences(
                    preference, cache.names[positions], len(data.inputs.security_ids)
                ).numpy()
            )
    values = np.concatenate(prefs)
    result, targets, previous = exact_replay(
        data,
        TensorPreference(torch.from_numpy(values), int(rows[0])),
        int(rows[0]),
        int(rows[-1] + 1),
    )
    summary = {
        "ic": float(np.nanmean(daily)),
        "utility_bps": float(
            utility_series(data, result, previous, int(rows[0])).mean()
        ),
        "net_excess_bps": float(result.net_excess_all_cash_bps.mean()),
        "economics_unresolved": bool(result.economics_unresolved),
        "mean_gross": float(np.abs(targets).sum(1).mean()),
    }
    return summary, (np.concatenate(scores), values), (result, targets, previous)


def gradient_scale(model, neural_forward, cache, rows, data):
    """Fit-only magnitude calibration; no economic validation tuning of lambda."""
    model.eval()
    starts = sorted(set([0, (len(rows) // 2) // 32 * 32, (len(rows) - 32) // 32 * 32]))
    account = data.initial_account(int(rows[0]), policy_ledger_config())
    records = []
    parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad
        and "economic_head" not in name
        and not name.startswith("model.head")
    ]
    for first in range(0, len(rows), 32):
        positions = np.arange(first, min(first + 32, len(rows)))
        with torch.set_grad_enabled(first in starts):
            _, pref, ranking = encoded(
                neural_forward, cache, positions, len(data.inputs.security_ids)
            )
            utility, account, _, _ = utility_path(data, pref, account, rows[positions])
        if first in starts:
            gradients = []
            for loss in (ranking, utility):
                g = torch.autograd.grad(
                    loss, parameters, retain_graph=True, allow_unused=True
                )
                gradients.append(
                    float(
                        torch.sqrt(
                            sum(x.float().square().sum() for x in g if x is not None)
                        )
                    )
                )
            records.append(
                {
                    "first_row": int(rows[first]),
                    "ranking_gradient": gradients[0],
                    "utility_gradient": gradients[1],
                    "ratio": gradients[0] / max(gradients[1], 1e-30),
                }
            )
        account.detach()
    if any(
        r["utility_gradient"] < 1e-9 or not np.isfinite(r["ratio"]) for r in records
    ):
        raise ValueError("portfolio objective has no finite shared-encoder gradient")
    return {
        "weight": float(np.median([r["ratio"] for r in records])),
        "blocks": records,
    }


def epoch(
    model,
    neural_forward,
    cache,
    rows,
    data,
    optimizer,
    scaler,
    arm,
    variant,
    number,
    weight,
):
    model.train()
    account = data.initial_account(int(rows[0]), policy_ledger_config())
    blocks = [
        np.arange(i, min(i + SETTINGS["block_sessions"], len(rows)))
        for i in range(0, len(rows), SETTINGS["block_sessions"])
    ]
    recipe = RECIPES[arm]
    records = []
    for b, positions in enumerate(blocks):
        clean_state = []

        def closure():
            _, pref, ranking = encoded(
                neural_forward, cache, positions, len(data.inputs.security_ids)
            )
            if variant == "rank":
                return ranking
            utility, state, _, _ = utility_path(
                data,
                pref,
                clone_account(account),
                rows[positions],
                terminal=int(rows[positions[-1]]) == int(rows[-1]),
            )
            if not clean_state:
                state.detach()
                clean_state.append(state)
            value = utility.to(device=ranking.device, dtype=torch.float32) * weight
            return value + ranking if variant == "hybrid" else value

        multiplier = learning_rate_fraction(
            (number - 1) * len(blocks) + b, SETTINGS["maximum_epochs"] * len(blocks)
        )
        for group in optimizer.param_groups:
            group["lr"] = (
                SETTINGS["new_head_learning_rate"] * group["lr_multiplier"] * multiplier
            )
        diagnostic = {}
        loss, gap = optimizer_step(
            model,
            optimizer,
            closure,
            recipe.rho,
            adaptive=recipe.adaptive,
            eta=recipe.eta,
            diagnostics=diagnostic,
            scaler=scaler,
        )
        if clean_state:
            account = clean_state[0]
        records.append({"loss": loss, "sam_gap": gap, **diagnostic})
    return {
        "loss": float(np.mean([r["loss"] for r in records])),
        "sam_gap": float(np.mean([r["sam_gap"] for r in records])),
        "gradient_norm_median": float(
            np.median([r["descent_gradient_norm"] for r in records])
        ),
        "zero_gradient_blocks": sum(
            r["descent_gradient_norm"] < 1e-10 for r in records
        ),
        "loss_scale_retries": sum(r.get("loss_scale_retries", 0) for r in records),
        "blocks": len(records),
    }


def fit(root, design, data, arm, fold, seed, variant, *, engineering=False):
    set_deterministic_seed(seed)
    model, caches, axes, view, source = preparation(design, data, arm, fold, seed)
    directory = (
        root
        / ("engineering" if engineering else "fits")
        / arm
        / variant
        / f"{fold}_seed_{seed}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    contract = {
        "design": bound(root / "training_design.json"),
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "variant": variant,
        "source": source,
        "engineering": engineering,
    }
    complete = directory / "run_manifest.json"
    if complete.exists():
        old = read(complete)
        if old["contract"] != contract:
            raise ValueError("completed objective fit differs")
        for file, digest in old["files"].items():
            if sha256_file(directory / file) != digest:
                raise ValueError("completed objective artifact changed")
        return old
    neural_forward = compile_forward(model)
    optimizer = recipe_optimizer(
        model,
        cuda=True,
        learning_rate=SETTINGS["new_head_learning_rate"],
        transferred=[
            n for n, _ in model.named_parameters() if "economic_head" not in n
        ],
        transferred_multiplier=SETTINGS["learning_rate"]
        / SETTINGS["new_head_learning_rate"],
    )
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=256.0)
        if autocast_dtype(torch.device("cuda")) == torch.float16
        else None
    )
    start = 1
    history = []
    best = {"ic": (-float("inf"), 0), "utility_bps": (-float("inf"), 0)}
    resume = directory / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cuda", weights_only=True)
        if state["contract"] != contract:
            raise ValueError("resumable objective contract changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler:
            scaler.load_state_dict(state["scaler"])
        history, best, scale = state["history"], state["best"], state["scale"]
        start = len(history)
        _restore_rng((state["cpu_rng"].cpu(), [v.cpu() for v in state["cuda_rng"]]))
    else:
        # Engineering profiles initial fit gradients only. The financial launch
        # receives no selection/evaluation result from the engineering run.
        t = time.monotonic()
        scale = gradient_scale(model, neural_forward, caches["fit"], axes["fit"], view)
        write_json_atomic(directory / "loss_scale.json", scale)
        if engineering:
            positions = np.arange(min(32, len(axes["fit"])))
            model.train()
            timing = []
            for i in range(3):
                t = time.monotonic()
                # Full real-date block with both SAM passes and allocator gradients.
                account = view.initial_account(
                    int(axes["fit"][0]), policy_ledger_config()
                )

                def closure():
                    _, p, r = encoded(
                        neural_forward,
                        caches["fit"],
                        positions,
                        len(view.inputs.security_ids),
                    )
                    u, _, _, _ = utility_path(
                        view, p, clone_account(account), axes["fit"][positions]
                    )
                    return (
                        r + u.to(device=r.device, dtype=torch.float32) * scale["weight"]
                    )

                diagnostic = {}
                optimizer_step(
                    model,
                    optimizer,
                    closure,
                    RECIPES[arm].rho,
                    adaptive=RECIPES[arm].adaptive,
                    eta=RECIPES[arm].eta,
                    scaler=scaler,
                    diagnostics=diagnostic,
                )
                timing.append({"seconds": time.monotonic() - t, **diagnostic})
            # Compare the actual current neural preferences through both ledgers.
            model.eval()
            with torch.no_grad():
                _, p, _ = encoded(
                    neural_forward,
                    caches["fit"],
                    positions,
                    len(view.inputs.security_ids),
                )
                _, _, plans, nav = utility_path(
                    view,
                    p,
                    view.initial_account(int(axes["fit"][0]), policy_ledger_config()),
                    axes["fit"][positions],
                    terminal=True,
                )
                exact, actual, _ = exact_replay(
                    view,
                    TensorPreference(p, int(axes["fit"][0])),
                    int(axes["fit"][0]),
                    int(axes["fit"][positions[-1]] + 1),
                )
            error = float(np.max(np.abs(nav - exact.nav)))
            if error > 1e-7 or np.max(np.abs(plans - actual)) > 1e-7:
                raise ValueError(
                    "actual neural economic path fails independent accounting"
                )
            result = {
                "contract": contract,
                "loss_scale": scale,
                "timing": timing,
                "nav_max_error": error,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                "financial_selection_or_evaluation_read": False,
            }
            write_json_atomic(directory / "acceptance.json", result)
            return result
        initial, _, _ = readout(
            model, neural_forward, caches["selection"], axes["selection"], view
        )
        history = [{"epoch": 0, "selection": initial, "seconds": time.monotonic() - t}]
        for selector in best:
            best[selector] = (initial[selector], 0)
            save_checkpoint(
                directory / f"selected_{selector}.pt",
                {"model": model.state_dict(), "epoch": 0, "contract": contract},
            )
    (directory / "epochs").mkdir(exist_ok=True)
    for number in range(start, SETTINGS["maximum_epochs"] + 1):
        if number > SETTINGS["minimum_epochs"] and all(
            number - 1 - v[1] >= SETTINGS["patience"] for v in best.values()
        ):
            break
        t = time.monotonic()
        training = epoch(
            model,
            neural_forward,
            caches["fit"],
            axes["fit"],
            view,
            optimizer,
            scaler,
            arm,
            variant,
            number,
            scale["weight"],
        )
        selection, _, _ = readout(
            model, neural_forward, caches["selection"], axes["selection"], view
        )
        record = {
            "epoch": number,
            "training": training,
            "selection": selection,
            "seconds": time.monotonic() - t,
        }
        history.append(record)
        checkpoint = {
            "model": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "epoch": number,
            "contract": contract,
        }
        save_checkpoint(directory / "epochs" / f"epoch_{number:03d}.pt", checkpoint)
        for selector in best:
            improvement = 1e-4 if selector == "ic" else 0.01
            if selection[selector] > best[selector][0] + improvement:
                best[selector] = (selection[selector], number)
                save_checkpoint(directory / f"selected_{selector}.pt", checkpoint)
        cpu, cuda = _rng_state()
        save_checkpoint(
            resume,
            {
                **checkpoint,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "history": history,
                "best": best,
                "scale": scale,
                "cpu_rng": cpu,
                "cuda_rng": cuda,
            },
        )
        write_json_atomic(directory / "history.json", history)
        print(
            json.dumps(
                {"arm": arm, "fold": fold, "seed": seed, "objective": variant, **record}
            ),
            flush=True,
        )
    result = {
        "status": "completed",
        "contract": contract,
        "selected": best,
        "epochs_completed": len(history) - 1,
        "files": {p.name: sha256_file(p) for p in directory.glob("selected_*.pt")},
    }
    result["files"]["history.json"] = sha256_file(directory / "history.json")
    result["files"]["loss_scale.json"] = sha256_file(directory / "loss_scale.json")
    write_json_atomic(complete, result)
    return result


def run(root, *, engineering=False, confirmation=False):
    torch.set_num_threads(1)
    design = read(root / "training_design.json")
    if design["implementation"] != _git_identity() or design["runtime"]["torch"] != str(
        torch.__version__
    ):
        raise ValueError("financial worker differs from its frozen source/runtime")
    survivors = read(root / "screen_summary.json")["survivors"] if confirmation else {}
    folds = (
        [f for f in DEVELOPMENT_FOLDS if f not in SCREEN_FOLDS]
        if confirmation
        else SCREEN_FOLDS
    )
    for arm in CELLS:
        if confirmation and arm not in survivors:
            continue
        data, _ = load_data(Path(design["decision_root"]), arm)
        for fold in ("F2", "F14") if engineering else folds:
            for seed in (11,) if engineering else ALLOWED_SEEDS:
                for variant in (
                    ("hybrid",)
                    if engineering
                    else (VARIANTS if not confirmation else ("rank", *survivors[arm]))
                ):
                    torch._dynamo.reset()
                    torch.cuda.reset_peak_memory_stats()
                    fit(
                        root,
                        design,
                        data,
                        arm,
                        fold,
                        seed,
                        variant,
                        engineering=engineering,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "engineer", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    else:
        run(
            args.root,
            engineering=args.command == "engineer",
            confirmation=args.confirmation,
        )


if __name__ == "__main__":
    main()
