from __future__ import annotations

import torch


def _capped_simplex(
    values: torch.Tensor,
    caps: torch.Tensor,
    target: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Project nonnegative rows onto a capped simplex."""
    # Bisection identifies the locally fixed free/capped sets. Reconstructing the
    # threshold analytically from those sets gives the exact piecewise Jacobian;
    # differentiating through the branch history of bisection would not.
    with torch.no_grad():
        detached_values = values.detach()
        detached_caps = caps.detach()
        detached_target = target.detach()
        infinity = torch.full_like(detached_values, torch.inf)
        lower = torch.where(active, detached_values - detached_caps, infinity).amin(
            dim=-1
        )
        upper = torch.where(active, detached_values, -infinity).amax(dim=-1)
        for _ in range(96):
            threshold = (lower + upper) / 2
            allocated = torch.minimum(
                (detached_values - threshold.unsqueeze(-1)).clamp_min(0),
                detached_caps,
            )
            above_target = allocated.sum(dim=-1) > detached_target
            lower = torch.where(above_target, threshold, lower)
            upper = torch.where(above_target, upper, threshold)
        provisional = torch.minimum(
            (detached_values - ((lower + upper) / 2).unsqueeze(-1)).clamp_min(0),
            detached_caps,
        )
        free = active & (provisional > 0) & (provisional < detached_caps)
        saturated = active & (provisional == detached_caps) & (detached_caps > 0)

    free_count = free.sum(dim=-1)
    safe_count = free_count.clamp_min(1)
    threshold = (
        torch.where(free, values, 0).sum(dim=-1)
        + torch.where(saturated, caps, 0).sum(dim=-1)
        - target
    ) / safe_count
    allocated = torch.where(
        saturated,
        caps,
        torch.where(free, values - threshold.unsqueeze(-1), torch.zeros_like(values)),
    )
    # The no-free case lies on a simplex kink and is locally constant. Preserve
    # its exact value while retaining an explicit zero derivative.
    no_free = free_count == 0
    allocated = torch.where(no_free.unsqueeze(-1), provisional + values * 0, allocated)
    residual = target - allocated.sum(dim=-1)
    allocated = allocated + torch.where(
        free,
        residual.unsqueeze(-1) / safe_count.unsqueeze(-1),
        torch.zeros_like(allocated),
    )
    return allocated


def project_weights(
    raw: torch.Tensor,
    valid_mask: torch.Tensor,
    caps: torch.Tensor | float,
    gross_target: torch.Tensor | float,
) -> torch.Tensor:
    """Project raw signed weights to a capped, neutral, exact-gross portfolio.

    Positive raw values remain long, negative values remain short, and zero or
    masked values receive no allocation. ``gross_target`` is total absolute
    exposure, so each signed side receives half of it.
    """
    if not raw.is_floating_point() or raw.ndim == 0:
        raise ValueError("Raw weights must be a floating tensor with a name axis")

    try:
        mask = torch.broadcast_to(
            valid_mask.to(device=raw.device, dtype=torch.bool), raw.shape
        )
        cap = torch.broadcast_to(
            torch.as_tensor(caps, device=raw.device, dtype=raw.dtype), raw.shape
        )
        gross = torch.broadcast_to(
            torch.as_tensor(gross_target, device=raw.device, dtype=raw.dtype),
            raw.shape[:-1],
        )
    except RuntimeError as error:
        raise ValueError(
            "Projection inputs do not broadcast to the raw-weight rows"
        ) from error

    if not torch.isfinite(gross).all() or not (gross > 0).all():
        raise ValueError("Gross target must be finite and strictly positive")
    if not torch.isfinite(torch.where(mask, raw, torch.zeros_like(raw))).all():
        raise ValueError("Valid raw weights must be finite")
    valid_caps = torch.where(mask, cap, torch.zeros_like(cap))
    if not torch.isfinite(valid_caps).all() or not (valid_caps >= 0).all():
        raise ValueError("Valid per-name caps must be finite and nonnegative")

    candidate = torch.where(mask, raw, torch.zeros_like(raw))
    flat_candidate = candidate.reshape(-1, candidate.shape[-1])
    flat_caps = valid_caps.reshape_as(flat_candidate)
    scale = torch.maximum(gross.abs(), torch.ones_like(gross))
    tolerance = 16 * torch.finfo(raw.dtype).eps * scale
    already_feasible = (
        (candidate.sum(dim=-1).abs() <= tolerance)
        & ((candidate.abs().sum(dim=-1) - gross).abs() <= tolerance)
        & (candidate.abs() <= valid_caps + tolerance.unsqueeze(-1)).all(dim=-1)
    )
    if already_feasible.all():
        return candidate

    pending = torch.nonzero(~already_feasible.reshape(-1), as_tuple=False).flatten()
    selected = flat_candidate[pending]
    selected_caps = flat_caps[pending]
    selected_mask = mask.reshape_as(flat_candidate)[pending]
    positive = selected_mask & (selected > 0)
    negative = selected_mask & (selected < 0)
    side_target = (gross / 2).reshape(-1)[pending]
    positive_capacity = torch.where(positive, selected_caps, 0).sum(dim=-1)
    negative_capacity = torch.where(negative, selected_caps, 0).sum(dim=-1)
    feasible = (positive_capacity >= side_target) & (negative_capacity >= side_target)
    if not feasible.all():
        rows = pending[~feasible].tolist()
        raise ValueError(
            "Exact neutral gross is infeasible for signed cap capacity in flattened "
            f"rows {rows}"
        )

    long = _capped_simplex(
        selected.clamp_min(0),
        torch.where(positive, selected_caps, 0),
        side_target,
        positive,
    )
    short = _capped_simplex(
        (-selected).clamp_min(0),
        torch.where(negative, selected_caps, 0),
        side_target,
        negative,
    )
    projected = long - short
    return flat_candidate.index_copy(0, pending, projected).reshape_as(candidate)
