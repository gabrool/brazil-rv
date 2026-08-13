from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .contract import PRICE_FEATURE_CLIP
from .transforms import centered_midranks, leave_one_out_medians

VALIDITY_TO_FEATURE_CHANNELS = ((0, 2), (1, 3), (4,), (5,))


@dataclass(frozen=True)
class PeerFeatureResult:
    features: NDArray[np.float32]
    valid: NDArray[np.bool_]
    usable_peer_count: NDArray[np.int16]


def build_peer_features(
    normalized_returns: NDArray[np.floating],
    return_valid: NDArray[np.bool_],
    active: NDArray[np.bool_],
    selected_relation: NDArray[np.object_],
    selected_group_id: NDArray[np.integer],
    sector_group_id: NDArray[np.integer],
    subsector_group_id: NDArray[np.integer],
    issuer_ids: Sequence[str | None],
) -> PeerFeatureResult:
    """Build causal peer features for one date, independently at each minute."""
    returns = np.asarray(normalized_returns)
    source_valid = np.asarray(return_valid)
    active = np.asarray(active)
    relation = np.asarray(selected_relation, dtype=object)
    selected_group = np.asarray(selected_group_id)
    sector_group = np.asarray(sector_group_id)
    subsector_group = np.asarray(subsector_group_id)
    if returns.ndim != 3 or returns.shape[2] != 2:
        raise ValueError("normalized_returns must have shape [equity, minute, 2]")
    equity_count, minute_count, _ = returns.shape
    if source_valid.shape != returns.shape:
        raise ValueError("return_valid must align with normalized_returns")
    for name, values in (
        ("active", active),
        ("selected_relation", relation),
        ("selected_group_id", selected_group),
        ("sector_group_id", sector_group),
        ("subsector_group_id", subsector_group),
    ):
        if values.shape != (equity_count,):
            raise ValueError(f"{name} must have shape [equity]")
    if len(issuer_ids) != equity_count:
        raise ValueError("issuer_ids must align with the equity axis")
    if source_valid.dtype != np.dtype(bool) or active.dtype != np.dtype(bool):
        raise ValueError("return_valid and active must have boolean dtype")
    if not np.isfinite(returns).all():
        raise ValueError("normalized_returns contains a non-finite value")

    features = np.zeros((equity_count, minute_count, 6), dtype=np.float32)
    valid = np.zeros((equity_count, minute_count, 4), dtype=bool)
    usable_peer_count = np.zeros((equity_count, minute_count, 4), dtype=np.int16)

    for window, (difference_channel, rank_channel, valid_channel) in enumerate(
        ((0, 2, 0), (1, 3, 1))
    ):
        values = returns[:, :, window].astype(np.float64, copy=False)
        window_valid = source_valid[:, :, window]
        for relation_name, static_groups in (
            ("SECTOR", sector_group),
            ("SUBSECTOR", subsector_group),
        ):
            focal_policy = active & (relation == relation_name)
            group_ids = np.unique(selected_group[focal_policy & (selected_group >= 0)])
            for group_id in group_ids:
                members = active & (static_groups == group_id)
                focals = focal_policy & (selected_group == group_id)
                for minute_idx in range(minute_count):
                    usable = members & window_valid[:, minute_idx]
                    slots = np.flatnonzero(usable)
                    if slots.size < 3:
                        continue
                    focal_slots = np.flatnonzero(focals & usable)
                    if focal_slots.size == 0:
                        continue
                    group_values = values[slots, minute_idx]
                    positions = np.full(equity_count, -1, dtype=np.int32)
                    positions[slots] = np.arange(slots.size, dtype=np.int32)
                    focal_positions = positions[focal_slots]
                    differences = group_values - leave_one_out_medians(group_values)
                    ranks = centered_midranks(group_values)
                    features[focal_slots, minute_idx, difference_channel] = np.clip(
                        differences[focal_positions],
                        -PRICE_FEATURE_CLIP,
                        PRICE_FEATURE_CLIP,
                    ).astype(np.float32)
                    features[focal_slots, minute_idx, rank_channel] = ranks[
                        focal_positions
                    ]
                    valid[focal_slots, minute_idx, valid_channel] = True
                    usable_peer_count[focal_slots, minute_idx, valid_channel] = (
                        slots.size - 1
                    )

    issuer_groups: dict[str, list[int]] = {}
    for slot, issuer_id in enumerate(issuer_ids):
        if issuer_id:
            issuer_groups.setdefault(issuer_id, []).append(slot)
    for window, (feature_channel, valid_channel) in enumerate(((4, 2), (5, 3))):
        values = returns[:, :, window].astype(np.float64, copy=False)
        window_valid = source_valid[:, :, window]
        for members_list in issuer_groups.values():
            members = np.zeros(equity_count, dtype=bool)
            members[members_list] = True
            members &= active
            if int(members.sum()) < 2:
                continue
            for minute_idx in range(minute_count):
                slots = np.flatnonzero(members & window_valid[:, minute_idx])
                if slots.size < 2:
                    continue
                group_values = values[slots, minute_idx]
                differences = group_values - leave_one_out_medians(group_values)
                features[slots, minute_idx, feature_channel] = np.clip(
                    differences, -PRICE_FEATURE_CLIP, PRICE_FEATURE_CLIP
                ).astype(np.float32)
                valid[slots, minute_idx, valid_channel] = True
                usable_peer_count[slots, minute_idx, valid_channel] = slots.size - 1

    return PeerFeatureResult(features, valid, usable_peer_count)


def validate_peer_arrays(
    features: NDArray[np.float32],
    valid: NDArray[np.bool_],
    *,
    date_chunk: int = 16,
) -> None:
    """Validate dtype, finiteness, bounds, and false-mask zero filling."""
    if features.ndim != 4 or features.shape[-1] != 6:
        raise ValueError(
            "equity_peer_features.npy must have four axes and six channels"
        )
    if valid.shape != (*features.shape[:-1], 4):
        raise ValueError("equity_peer_valid.npy must align and have four channels")
    if features.dtype != np.dtype(np.float32):
        raise ValueError("equity_peer_features.npy must have float32 dtype")
    if valid.dtype != np.dtype(bool):
        raise ValueError("equity_peer_valid.npy must have boolean dtype")
    for start in range(0, features.shape[0], date_chunk):
        stop = min(start + date_chunk, features.shape[0])
        values = np.asarray(features[start:stop])
        masks = np.asarray(valid[start:stop])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite peer feature in dates {start}:{stop}")
        if np.any(np.abs(values) > PRICE_FEATURE_CLIP + 1e-6):
            raise ValueError(
                f"Peer feature outside clipping bounds in dates {start}:{stop}"
            )
        ranks = values[..., 2:4]
        rank_valid = masks[..., :2]
        if np.any((ranks <= -1.0) & rank_valid) or np.any((ranks >= 1.0) & rank_valid):
            raise ValueError(
                f"Selected-peer rank outside (-1, 1) in dates {start}:{stop}"
            )
        for valid_channel, feature_channels in enumerate(VALIDITY_TO_FEATURE_CHANNELS):
            invalid = ~masks[..., valid_channel]
            if np.any(values[..., feature_channels][invalid] != 0):
                raise ValueError(
                    "Peer numeric features must be exactly zero under false validity"
                )
