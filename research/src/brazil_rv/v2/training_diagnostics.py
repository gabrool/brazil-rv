"""Bounded fit-only model probes; no hooks or activation retention in training."""

from __future__ import annotations

import copy
from collections import defaultdict

import torch

from .temporal_pathway import PeerAttention
from .train import _restore_rng, _rng_state


def _rms(values):
    return float(values.detach().float().square().mean().sqrt())


def _module_norms(model, originals):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0])
    for name, parameter in model.named_parameters():
        # Retain individual family identity; group the rest by architectural part.
        part = (
            ".".join(name.split(".")[:2])
            if name.startswith("families.")
            else name.split(".")[0]
        )
        original = originals[name].to(parameter.device)
        totals[part][0] += float(original.square().sum())
        totals[part][1] += float((parameter.detach() - original).square().sum())
        if parameter.grad is not None:
            totals[part][2] += float(parameter.grad.detach().square().sum())
    return {
        name: {
            "weight_norm": weight**0.5,
            "update_norm": update**0.5,
            "update_to_weight": (update / max(weight, 1e-24)) ** 0.5,
            "descent_gradient_norm_after_clip": gradient**0.5,
        }
        for name, (weight, update, gradient) in totals.items()
    }


def probe(
    model,
    optimizer,
    batch,
    *,
    characteristic,
    horizons,
    rho,
    adaptive,
    eta,
    use_bf16=True,
):
    """One actual two-pass update, then restore model/optimizer/RNG exactly.

    Called on two fixed training dates at initialization, early and selected
    states. Hooks summarize detached activations; nothing is serialized at
    name/date resolution. Interventions are sensitivity checks, not importance.
    """
    from .round7_training import forward, member_loss, optimizer_step
    from .contract import HORIZONS

    rng, mode = _rng_state(), model.training
    state = {n: p.detach().clone() for n, p in model.state_dict().items()}
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    gradients = {
        n: None if p.grad is None else p.grad.detach().clone()
        for n, p in model.named_parameters()
    }
    handles, activations, prediction, result = [], {}, [], {}
    head_indices = [HORIZONS.index(h) for h in horizons]
    mask = batch["active_mask"][..., None, None]

    def observe(name):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                activations.setdefault(name, []).append(_rms(value))
            if name == "film" and isinstance(value, torch.Tensor):
                gamma, beta = value.detach().float().chunk(2, -1)
                result.setdefault("film", []).append(
                    {"gamma_rms": _rms(gamma), "beta_rms": _rms(beta)}
                )

        return hook

    def attention_observer(name):
        def hook(module, args, output):
            # Eight queries on one nonempty sample: never materialize all T*N²
            # attention matrices or interfere with the fused training operator.
            with torch.no_grad():
                values, valid = args
                counts = valid.sum(-1)
                index = int(counts.argmax())
                if counts[index] < 2:
                    return
                # An early left-padding position can have just a few names;
                # inspect a fully populated slice, not the first nonempty one.
                values, valid = values[index : index + 1], valid[index : index + 1]
                names, width = values.shape[-2:]
                q, k, _ = (
                    module.qkv(module.norm(values))
                    .float()
                    .reshape(1, names, 3, 4, width // 4)
                    .permute(2, 0, 3, 1, 4)
                    .unbind(0)
                )
                queries = torch.nonzero(valid[0]).flatten()[:8]
                logits = q[:, :, queries] @ k.transpose(-1, -2) / (width // 4) ** 0.5
                weights = torch.softmax(
                    logits.masked_fill(~valid[:, None, None, :], -torch.inf), -1
                )
                entropy = -(weights * weights.clamp_min(1e-20).log()).sum(-1)
                result.setdefault("attention", {}).setdefault(name, []).append(
                    {
                        "valid_keys": int(valid.sum()),
                        "qk_logit_rms": _rms(logits[..., valid[0]]),
                        "entropy_over_log_keys": float(
                            entropy.mean() / valid.sum().float().log()
                        ),
                        "head_weight_dispersion": float(weights.std(dim=1).mean()),
                    }
                )

        return hook

    def predict():
        with torch.autocast(
            device_type=mask.device.type,
            dtype=torch.bfloat16,
            enabled=mask.is_cuda and use_bf16,
        ):
            return forward(model, batch, characteristic=characteristic).float()

    def closure():
        scores = predict()
        prediction.append(scores.detach())
        return member_loss(
            scores,
            batch["targets"][..., head_indices],
            batch["target_mask"][..., head_indices] & batch["active_mask"][..., None],
        )

    try:
        for name, module in model.named_modules():
            if name and (
                "." not in name or name.startswith("families.") and name.count(".") == 1
            ):
                handles.append(module.register_forward_hook(observe(name)))
            if isinstance(module, PeerAttention):
                handles.append(module.register_forward_hook(attention_observer(name)))
        model.train()
        clean, gap = optimizer_step(
            model,
            optimizer,
            closure,
            rho,
            adaptive=adaptive,
            eta=eta,
            diagnostics=result,
        )
        result.update(
            clean_loss=clean,
            perturbed_loss=clean + gap,
            modules=_module_norms(model, state),
            activation_rms=activations,
        )
        if len(prediction) == 2:
            valid = mask.expand_as(prediction[0])
            first, second = prediction[0][valid], prediction[1][valid]
            result["sam_score_movement_over_clean_std"] = float(
                (second - first).square().mean().sqrt() / first.std().clamp_min(1e-12)
            )
        for handle in handles:
            handle.remove()
        handles.clear()
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            clean_scores = predict()
            result["cross_sectional_score_resolution"] = {}
            population = (
                batch["active_mask"][..., None]
                & batch["target_mask"][..., head_indices]
            )
            for h, horizon in enumerate(horizons):
                rows = [
                    clean_scores[d, population[d, :, h], :, h].mean(-1)
                    for d in range(len(clean_scores))
                    if population[d, :, h].sum() >= 2
                ]
                if rows:
                    result["cross_sectional_score_resolution"][str(horizon)] = {
                        "mean_std": sum(float(x.std()) for x in rows) / len(rows),
                        "mean_absolute_level": sum(float(x.mean().abs()) for x in rows)
                        / len(rows),
                        "mean_unique_fraction": sum(
                            len(torch.unique(x)) / len(x) for x in rows
                        )
                        / len(rows),
                    }
            peers = [
                (n, m)
                for n, m in model.named_modules()
                if isinstance(m, PeerAttention) and "temporal_peer.peer" in n
            ]
            for name, module in peers:
                handle = module.register_forward_hook(
                    lambda m, args, output: torch.where(
                        args[1][..., None], m.output_norm(args[0]), 0.0
                    )
                )
                handles.append(handle)
                bypassed = predict()
                valid = mask.expand_as(clean_scores)
                result.setdefault("peer_bypass", {})[name] = float(
                    (bypassed[valid] - clean_scores[valid]).square().mean().sqrt()
                    / clean_scores[valid].std().clamp_min(1e-12)
                )
                handle.remove()
                handles.clear()
    finally:
        for handle in handles:
            handle.remove()
        model.load_state_dict(state)
        optimizer.load_state_dict(optimizer_state)
        for name, parameter in model.named_parameters():
            parameter.grad = gradients[name]
        model.train(mode)
        _restore_rng(rng)
    return result
