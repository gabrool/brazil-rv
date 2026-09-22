"""Post-hoc forecast variants from completed round-7 fits.

A completed F fit keeps every epoch's model state, its per-epoch selection
history and the exported scores of the epoch the trainer selected. A *variant*
changes only which saved epochs supply forecasts (a selection view) or how
several saved forecasts are rank-averaged (an ensemble). It retrains nothing
and never reads evaluation labels. An alternative epoch is scored through the
registered scoring path on a derived selection checkpoint that binds the saved
epoch file by hash, so the same store, fit-only conditioning, inference code
and provenance checks apply as for the trainer's own export.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from .artifacts import sha256_file, write_json_atomic
from .portfolio_inputs import HEADS, normalized_ranks
from .research_rounds import _score_artifact
from .selection_rules import epoch_views
from .train import rank_average_ensemble


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def binding(path):
    return {"path": str(path), "sha256": sha256_file(Path(path))}


def completed_fit(fit):
    """The sealed manifest and hash-checked history of one completed fit."""
    fit = Path(fit)
    manifest = read_json(fit / "run_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"fit is not completed: {fit}")
    history_path = fit / "history.json"
    expected = manifest["artifacts"].get("history.json")
    if expected is None or sha256_file(history_path) != expected:
        raise ValueError(f"fit history differs from its sealed hash: {fit}")
    history = read_json(history_path)
    if history[-1]["epoch"] != manifest["epochs_completed"]:
        raise ValueError(f"fit history is shorter than its manifest: {fit}")
    return manifest, history


def epoch_checkpoint(fit, manifest, epoch):
    """The sealed checkpoint file holding ``epoch`` and its manifest hash."""
    fit = Path(fit)
    relative = (
        "selected.pt"
        if epoch == manifest["selected_epoch"]
        else f"epochs/epoch_{epoch:03d}.pt"
    )
    expected = manifest["artifacts"].get(relative)
    if expected is None:
        raise ValueError(f"{relative} is not a sealed artifact of {fit}")
    path = fit / relative
    if sha256_file(path) != expected:
        raise ValueError(f"saved checkpoint differs from its sealed hash: {path}")
    return path, expected


def _atomic_torch_save(payload, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def derived_checkpoint(fit, manifest, epoch, destination, *, view):
    """Write ``destination/selected.pt``: the saved epoch plus the view's score.

    The payload is the sealed epoch file's own payload; only ``selection_ic``
    (the view's selection score, which the scorer requires) and an explicit
    ``selection_view`` provenance record are added. Existing derivations are
    reused after a hash check, never rewritten.
    """
    fit, destination = Path(fit), Path(destination)
    source, source_sha256 = epoch_checkpoint(fit, manifest, epoch)
    target = destination / "selected.pt"
    record_path = destination / "view.json"
    if target.exists() or record_path.exists():
        record = read_json(record_path)
        if (
            record["source"]["sha256"] != source_sha256
            or record["view"]["epochs"] != view["epochs"]
            or sha256_file(target) != record["derived"]["sha256"]
        ):
            raise ValueError(f"existing derived checkpoint differs: {destination}")
        return target, record["derived"]["sha256"]
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("epoch") != epoch or payload.get("stage") != "F":
        raise ValueError("saved epoch payload does not describe the requested epoch")
    if "selection_ic" in payload and source.name == "selected.pt":
        raise ValueError("the trainer's own selection needs no derived checkpoint")
    derived = {
        **payload,
        "selection_ic": float(view.get("selection_score", float("nan"))),
        "selection_view": {
            "rule": view["rule"],
            "epochs": [int(e) for e in view["epochs"]],
            "epoch": int(epoch),
            "source_checkpoint": {"path": str(source), "sha256": source_sha256},
            "fit_manifest": binding(fit / "run_manifest.json"),
            "definition": (
                "post-hoc selection view over one executed trajectory; the weights "
                "are the sealed epoch file's weights, no retraining"
            ),
        },
    }
    _atomic_torch_save(derived, target)
    record = {
        "source": {"path": str(source), "sha256": source_sha256},
        "derived": binding(target),
        "view": {k: v for k, v in view.items() if k != "scores"},
        "epoch": int(epoch),
        "fit_manifest": binding(fit / "run_manifest.json"),
    }
    write_json_atomic(record_path, record)
    return target, record["derived"]["sha256"]


def scores_for_epoch(
    store_root, fit, manifest, epoch, cache_root, *, view, device, compiled=False
):
    """The scores directory of one saved epoch, exported once on demand.

    The trainer's own selected epoch reuses its sealed export. Any other epoch
    is scored through ``round7_score.score`` on a derived checkpoint under
    ``cache_root/epoch_XXX`` with the fit's own padded name count.
    """
    fit, cache_root = Path(fit), Path(cache_root)
    if epoch == manifest["selected_epoch"]:
        scores = fit / "scores"
        exported = read_json(scores / "score_manifest.json")
        if exported["checkpoint"]["sha256"] != manifest["artifacts"]["selected.pt"]:
            raise ValueError(f"exported scores bind another checkpoint: {fit}")
        return scores, {"source": "trainer_export", "epoch": int(epoch)}
    destination = cache_root / f"epoch_{epoch:03d}"
    checkpoint, checkpoint_sha256 = derived_checkpoint(
        fit, manifest, epoch, destination, view=view
    )
    scores = destination / "scores"
    if (scores / "score_manifest.json").exists():
        existing = read_json(scores / "score_manifest.json")
        if existing["checkpoint"]["sha256"] != checkpoint_sha256:
            raise ValueError(f"cached scores bind another checkpoint: {scores}")
    else:
        from .round7_score import score

        score(
            Path(store_root),
            checkpoint,
            scores,
            expected_sha256=checkpoint_sha256,
            compiled=compiled,
            device=device,
            fixed_name_count=manifest["contract"].get("padded_name_count"),
        )
    return scores, {
        "source": "derived_view_export",
        "epoch": int(epoch),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha256},
    }


def load_panel(scores, *, dates, isins, feature_schema_sha256, active):
    """Normalized D3/D5/D10 rank panel of one score export on the evaluation rows."""
    values, mask = _score_artifact(
        Path(scores),
        require_clean_transfer=True,
        expected_dates=np.asarray(dates, dtype="datetime64[D]"),
        expected_isins=isins,
        expected_feature_schema_sha256=feature_schema_sha256,
    )
    mask = mask[..., HEADS]
    active = np.asarray(active, dtype=bool)
    valid = mask.all(-1) & active
    if not np.array_equal(valid, active):
        raise ValueError("forecasts lost eligible names")
    panel = normalized_ranks(rank_average_ensemble([values[..., HEADS]], mask), mask)
    return panel.astype(np.float32), valid


def compose(panels):
    """Equal-weight mean of normalized rank panels: the repository's rank ensemble."""
    arrays = [np.asarray(p, dtype=np.float32) for p in panels]
    if not arrays or any(a.shape != arrays[0].shape for a in arrays[1:]):
        raise ValueError("ensemble members must share one panel shape")
    return np.mean(arrays, axis=0, dtype=np.float64).astype(np.float32)


def member_epochs(history, rule):
    """The saved epochs one post-hoc rule rank-averages, with its detail record."""
    return epoch_views(history, rules=(rule,))[rule]


def parse_member(text):
    """``arm@rule`` -> (arm, rule); a bare arm means the trainer's own selection."""
    arm, _, rule = text.partition("@")
    if not arm:
        raise ValueError(f"member lacks an arm: {text!r}")
    return arm, (rule or "raw")


def parse_variant(text):
    """``NAME=arm@rule+arm@rule`` or ``arm@rule`` -> (name, [(arm, rule), ...])."""
    name, separator, members = text.partition("=")
    if not separator:
        members, name = name, None
    parsed = [parse_member(m) for m in members.split("+") if m]
    if not parsed:
        raise ValueError(f"variant has no members: {text!r}")
    if name is None:
        if len(parsed) != 1:
            raise ValueError("a multi-member variant needs an explicit name")
        name = f"{parsed[0][0]}@{parsed[0][1]}"
    return name, parsed
