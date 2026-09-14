"""Compact, causal opportunity information for frozen-forecast controllers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.data_roots import resolve_external_root
from brazil_rv.v2.portfolio_inputs import HEADS, normalized_ranks
from brazil_rv.v2.portfolio_program import ARMS, PROJECT, read
from brazil_rv.v2.round7_preprocessing import common_snapshot, cross_market_partition
from brazil_rv.v2.store import open_store_for_dates
from brazil_rv.v2.train import rank_average_ensemble


def recover_seed_agreement(root):
    """Read only required score members; retain their rank dispersion, not logits."""
    caches = {arm: read(root / "cache" / arm / "manifest.json") for arm in ARMS}
    requested = {}
    for arm, cache in caches.items():
        for fold, seeds in cache["sources"].items():
            for source in seeds:
                stem, suffix = (
                    source["manifest"].split("/model_runs/", 1)[1].split("/", 1)
                )
                directory = str(Path(suffix).parent).replace("\\", "/")
                requested[stem, directory] = (arm, fold, source)
    source_records = {}
    recovery = read(PROJECT / "docs/v2_portfolio_policy_recovery.json")
    if sha256_file(Path(recovery["inventory"])) != recovery["inventory_sha256"]:
        raise ValueError("source forecast inventory changed")
    origin = recovery["source"].rsplit("/", 1)[-1]
    archives = {
        r["archive"].rsplit("/", 1)[-1].split(".")[0]: Path(r["local_archive"])
        for r in recovery["archives"]
    }
    for record in read(recovery["inventory"]):
        if record["group"] == "forecasts":
            source_records[origin, record["path"]] = (
                archives["forecasts"],
                record["sha256"],
            )
    old = read(PROJECT / "docs/v2_post_data_recovery.json")
    for archive in old["archives_in_restore_order"]:
        manifest = Path(archive["manifest"])
        if sha256_file(manifest) != archive["manifest_sha256"]:
            raise ValueError("earlier forecast recovery manifest changed")
        record = read(manifest)
        origin = record["root"].rsplit("/", 1)[-1]
        for member in record["files"]:
            source_records[origin, member["path"]] = (
                Path(archive["archive"]),
                member["sha256"],
            )
    needed = {}
    for origin, directory in requested:
        for name in (
            "score_manifest.json",
            "scores.npy",
            "score_mask.npy",
            "date_index.npy",
            "isin_index.npy",
        ):
            member = directory + "/" + name
            archive, digest = source_records[origin, member]
            needed.setdefault(archive, {})[member] = (origin, digest)
    payloads = {}
    for archive, members in needed.items():
        with tarfile.open(archive, "r|*") as stream:
            remaining = set(members)
            for member in stream:
                name = member.name.removeprefix("./")
                if name not in remaining:
                    continue
                payload = stream.extractfile(member).read()
                origin, digest = members[name]
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError(f"sealed seed artifact changed: {name}")
                payloads[origin, name] = payload
                remaining.remove(name)
                if not remaining:
                    break
            if remaining:
                raise ValueError("required seed source is absent from recovery")
        print(
            {"seed_source_archive": str(archive), "members": len(members)}, flush=True
        )
    blocks = {}
    for (origin, directory), (arm, fold, source) in requested.items():
        raw = payloads.pop((origin, directory + "/score_manifest.json"))
        if hashlib.sha256(raw).hexdigest() != source["sha256"]:
            raise ValueError("seed source differs from ensemble cache")
        manifest = json.loads(raw)
        arrays = {}
        for name in (
            "scores.npy",
            "score_mask.npy",
            "date_index.npy",
            "isin_index.npy",
        ):
            raw = payloads.pop((origin, directory + "/" + name))
            if hashlib.sha256(raw).hexdigest() != manifest["artifacts"][name]["sha256"]:
                raise ValueError("score payload differs from bound seed manifest")
            arrays[name] = np.load(io.BytesIO(raw), allow_pickle=False)
        scores, mask = arrays["scores.npy"], arrays["score_mask.npy"]
        ranks = normalized_ranks(rank_average_ensemble([scores], mask), mask)
        block = blocks.setdefault((arm, fold), [])
        block.append(
            (
                ranks[..., HEADS],
                mask[..., HEADS],
                arrays["date_index.npy"],
                arrays["isin_index.npy"],
            )
        )
    destination = root / "phase2/context"
    destination.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        values, dates = [], []
        keys = sorted(
            (k for k in blocks if k[0] == arm), key=lambda k: blocks[k][0][2][0]
        )
        for key in keys:
            members = blocks.pop(key)
            reference = members[0]
            for member in members[1:]:
                for a, b in zip(reference[1:], member[1:]):
                    if not np.array_equal(a, b):
                        raise ValueError(
                            "seed date, identity or valid population differs"
                        )
            values.append(
                np.std([m[0] for m in members], axis=0).mean(-1).astype(np.float32)
            )
            dates.extend(reference[2])
        output = destination / f"{arm}_agreement.npz"
        np.savez_compressed(
            output,
            disagreement=np.concatenate(values),
            dates=np.asarray(dates),
            isins=reference[3],
        )
        write_json_atomic(
            destination / f"{arm}_agreement.json",
            {
                "sha256": sha256_file(output),
                "source_cache_sha256": sha256_file(
                    root / "cache" / arm / "manifest.json"
                ),
                "definition": "population standard deviation across three normalized seed ranks, averaged across D3/D5/D10",
                "seeds": [11, 29, 47],
                "source_members_verified": True,
            },
        )


@dataclass
class ControllerContext:
    common: np.ndarray
    valid: np.ndarray
    passthrough: tuple[bool, ...]
    names: tuple[str, ...]
    disagreement: np.ndarray


def matured_shadow(ranks, valid, residual5, target_valid):
    """At t update from t-6, whose close(t-1) five-session label is now known.

    Fixed equal-rank shadow forecasts are observed even while the actual account
    is in cash. EWMA half-life is 60 available updates; no fitted or future alpha
    is used to construct this state. Invalid days do not manufacture an outcome.
    """
    days = len(ranks)
    values, seen = np.zeros((days, 3)), np.zeros((days, 3), bool)
    level, second, mass = 0.0, 0.0, 0.0
    decay = np.exp(-np.log(2) / 60)
    for day in range(days):
        origin = day - 6
        if origin >= 0:
            mask = valid[origin] & target_valid[origin] & np.isfinite(residual5[origin])
            if mask.sum() >= 2:
                x = ranks[origin, mask].mean(-1)
                x = x - x.mean()
                y = residual5[origin, mask] / 5
                payoff = float(x @ y / max(np.abs(x).sum(), 1e-12))
                level = decay * level + (1 - decay) * payoff
                second = decay * second + (1 - decay) * payoff**2
                mass = decay * mass + (1 - decay)
        if mass > 0:
            mean = level / mass
            values[day] = (
                mean * 1e4,
                np.sqrt(max(second / mass - mean**2, 0)) * 1e4,
                mass,
            )
            seen[day] = True
    return values, seen


def build_context(root, data, arm, market):
    design = read(root / "frozen_design.json")
    indices = data.inputs.session_indices
    store_root, _ = resolve_external_root(design["store"]["root"])
    if sha256_file(store_root / "manifest.json") != design["store"]["manifest_sha256"]:
        raise ValueError("controller source store changed")
    store, access = open_store_for_dates(
        store_root, indices, purpose="training", verify_hashes=False
    )
    # Verify consumed arrays once when building the compact cache. Do not hash
    # unrelated M1/target arrays for every independent small controller fit.
    used_arrays = {}
    for name in (
        "sidecar_cross_market_values",
        "sidecar_cross_market_valid",
        "sidecar_cross_market_age_sessions",
        "common_state_diagnostic_values",
        "common_state_diagnostic_valid",
    ):
        record = store.manifest["arrays"][name]
        digest = sha256_file(store_root / record["path"])
        if digest != record["sha256"]:
            raise ValueError(f"controller source array changed: {name}")
        used_arrays[name] = digest
    all_names = store.manifest["feature_names"]["sidecar_cross_market"]
    shared, _ = cross_market_partition(all_names)
    diagnostic_names = store.manifest["feature_names"]["common_state_diagnostic"]
    columns = [all_names.index(n) for n in shared]
    chunks = []
    try:
        for start in range(0, len(indices), 64):
            rows = indices[start : start + 64]
            x = store.read("sidecar_cross_market_values", rows)[..., columns]
            valid = store.read("sidecar_cross_market_valid", rows)[..., columns]
            valid &= data.inputs.active[start : start + 64, :, None]
            age = store.read("sidecar_cross_market_age_sessions", rows)[..., columns]
            chunks.append(common_snapshot(x, valid, age))
        x, valid, age = [np.concatenate([c[i] for c in chunks]) for i in range(3)]
        diagnostic = store.read("common_state_diagnostic_values", indices)
        diagnostic_valid = store.read("common_state_diagnostic_valid", indices)
    finally:
        store.close()
    agreement_path = root / "phase2/context" / f"{arm}_agreement.npz"
    binding = read(agreement_path.with_suffix(".json"))
    if sha256_file(agreement_path) != binding["sha256"]:
        raise ValueError("seed agreement artifact changed")
    with np.load(agreement_path, allow_pickle=False) as source:
        if not np.array_equal(
            source["dates"].astype("datetime64[D]"),
            np.asarray(data.inputs.dates, dtype="datetime64[D]"),
        ) or tuple(source["isins"].tolist()) != tuple(data.inputs.security_ids):
            raise ValueError("agreement dates or security identities differ")
        disagreement = source["disagreement"].copy()
    h5 = HORIZONS.index(5)
    cash5 = np.full(len(indices), np.nan)
    for t in range(len(indices) - 5):
        cash5[t] = np.prod(1 + data.inputs.cdi_returns[t + 1 : t + 6]) - 1
    residual = (
        data.inputs.shareholder_simple_returns[..., h5]
        - cash5[:, None]
        - data.beta * market[:, None]
    )
    shadow, shadow_valid = matured_shadow(
        data.ranks, data.valid, residual, data.inputs.shareholder_target_mask[..., h5]
    )
    age_known = np.isfinite(age) & (age >= 0)
    age_value = np.log1p(np.where(age_known, age, 0))
    age_value /= age_value + np.log1p(252)
    count = data.valid.sum(1).clip(1)
    opportunity = np.column_stack(
        (
            (disagreement * data.valid).sum(1) / count,
            (np.std(data.ranks, axis=-1) * data.valid).sum(1) / count,
        )
    )
    names = (
        *shared,
        *(n + "_valid" for n in shared),
        *(n + "_age" for n in shared),
        *(n + "_age_known" for n in shared),
        *diagnostic_names,
        *(n + "_valid" for n in diagnostic_names),
        "shadow_payoff_bps",
        "shadow_volatility_bps",
        "shadow_mass",
        "shadow_payoff_valid",
        "shadow_volatility_valid",
        "shadow_mass_valid",
        "mean_seed_disagreement",
        "mean_horizon_disagreement",
    )
    common = np.column_stack(
        (
            x,
            valid,
            age_value,
            age_known,
            diagnostic,
            diagnostic_valid,
            shadow,
            shadow_valid,
            opportunity,
        )
    ).astype(np.float32)
    masks = np.column_stack(
        (
            valid,
            np.ones_like(valid),
            age_known,
            np.ones_like(valid),
            diagnostic_valid,
            np.ones_like(diagnostic_valid),
            shadow_valid,
            np.ones_like(shadow_valid),
            np.ones_like(opportunity, bool),
        )
    )
    passthrough = tuple(
        n.endswith(("_valid", "_known", "_age"))
        or n
        in (
            "shadow_mass",
            "foreign_flow_month_reset",
            "foreign_flow_methodology_change",
        )
        for n in names
    )
    result = ControllerContext(common, masks, passthrough, names, disagreement)
    payload = root / "phase2/context" / f"{arm}_features.npz"
    np.savez_compressed(payload, common=common, valid=masks, disagreement=disagreement)
    write_json_atomic(
        root / "phase2/context" / f"{arm}_features.json",
        {
            "names": names,
            "passthrough": passthrough,
            "sha256": sha256_file(payload),
            "consumed_array_sha256": used_arrays,
            "source_store": design["store"],
            "source_cache_sha256": sha256_file(
                root / "cache" / arm / "policy_data.json"
            ),
            "agreement_sha256": binding["sha256"],
            "shadow_first_available_origin_lag": 6,
            "shadow_half_life": 60,
            "valid_dates_by_field": masks.sum(0).tolist(),
            "access": access.payload(),
            "full_dates": len(indices),
            "active_stock_days": int(data.inputs.active.sum()),
            "heldout_accessed": False,
        },
    )
    return result


def load_context(root, arm, binding):
    directory = root / "phase2/context"
    metadata = read(directory / f"{arm}_features.json")
    path = directory / f"{arm}_features.npz"
    if (
        metadata["source_cache_sha256"] != binding
        or sha256_file(path) != metadata["sha256"]
    ):
        raise ValueError("controller context differs from its forecast population")
    with np.load(path, allow_pickle=False) as values:
        return ControllerContext(
            values["common"].copy(),
            values["valid"].copy(),
            tuple(metadata["passthrough"]),
            tuple(metadata["names"]),
            values["disagreement"].copy(),
        )
