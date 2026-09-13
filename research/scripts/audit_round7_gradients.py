"""Frozen-checkpoint, fit-only CPU probes; no optimization or score selection."""

import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from brazil_rv.v2.characteristic_model import CharacteristicModel
from brazil_rv.v2.contract import HORIZONS
from brazil_rv.v2.data import V2DailyDataset, stage_name_count
from brazil_rv.v2.round7 import CELLS, PATHWAY_CELLS, configuration
from brazil_rv.v2.round7_preprocessing import Round7Preprocessing
from brazil_rv.v2.round7_training import forward, model_batch, member_loss
from brazil_rv.v2.train import _cli_stage_indices
from brazil_rv.v2.model import DailyMultiHorizonModel

torch.set_num_threads(6)
project = Path(__file__).resolve().parents[2]
root = Path(
    json.loads((project / "docs/v2_round7_inputs.json").read_text())["store"]["root"]
)
manifest = json.loads((root / "manifest.json").read_text())
fit, _, _, window = _cli_stage_indices(root, "F", "F14")
indices = fit[np.linspace(max(0, len(fit) - 120), len(fit) - 1, 4, dtype=int)]
results = []
for cell_name in ("A1", "B4", "B5", "B9", "B11", "GE"):
    cell = next(c for c in (*CELLS, *PATHWAY_CELLS) if c["cell"] == cell_name)
    config = configuration(cell, manifest["feature_names"])
    characteristic = cell["graph"] != "s0"
    horizons = config.horizons if characteristic else HORIZONS
    families = tuple(
        n
        for n, _ in (
            config.family_counts if characteristic else config.sidecar_feature_counts
        )
    )
    run = (
        "v2_round7_pathway_20260913T031500Z"
        if cell_name == "GE"
        else "v2_round7_20260913T012300Z"
    )
    checkpoint = (
        Path("D:/quant-data/b3/processed/model_runs")
        / run
        / "trajectories"
        / cell_name
        / "F14_seed_11/tail_average.pt"
    )
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    prep = Round7Preprocessing.from_payload(payload["contract"]["preprocessing"])
    dataset = V2DailyDataset(
        root,
        indices,
        stage="finetune",
        lookback=60,
        enabled_sidecars=families,
        include_intraday=False,
        include_fast=False,
        include_common_state=characteristic and bool(families),
        compact_names=True,
        target_window_indices=window,
        purpose="training",
    )
    try:
        batch = model_batch(
            prep.collate(
                [dataset[i] for i in range(len(dataset))],
                fixed_name_count=stage_name_count(dataset),
            ),
            torch.device("cpu"),
        )
        model = (
            CharacteristicModel(config)
            if characteristic
            else DailyMultiHorizonModel(config)
        )
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        targets = batch["targets"][..., [HORIZONS.index(h) for h in horizons]]
        mask = (
            batch["target_mask"][..., [HORIZONS.index(h) for h in horizons]]
            & batch["active_mask"][..., None]
        )

        def predict(b=batch):
            return forward(model, b, characteristic=characteristic)

        baseline = predict()
        loss = member_loss(baseline, targets, mask)
        loss.backward()
        gradients = {
            n: p.grad.detach().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        norm = sum(g.square().sum() for g in gradients.values()).sqrt()
        groups = {}
        for n, g in gradients.items():
            key = (
                ".".join(n.split(".")[:2])
                if n.startswith("families.")
                else n.split(".")[0]
            )
            groups[key] = groups.get(key, 0) + float(g.square().sum() / norm.square())
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        use = batch["active_mask"][..., None, None].expand_as(baseline)

        def corr(x, y):
            return float(
                torch.corrcoef(torch.stack((x[use].flatten(), y[use].flatten())))[0, 1]
            )

        shocks = []
        with torch.no_grad():
            for rho in (0.005, 0.01, 0.05, 0.125):
                for n, p in model.named_parameters():
                    if n in gradients:
                        p.copy_(original[n] + rho * gradients[n] / norm)
                changed = predict()
                shocks.append(
                    dict(
                        rho=rho,
                        loss=float(member_loss(changed, targets, mask)),
                        score_correlation=corr(baseline, changed),
                    )
                )
            for n, p in model.named_parameters():
                p.copy_(original[n])
            ablations = []
            for family in families:
                changed = dict(batch)
                changed["sidecar_" + family + "_values"] = torch.zeros_like(
                    batch["sidecar_" + family + "_values"]
                )
                ablations.append(
                    dict(
                        family=family,
                        zero_values_keep_masks_score_correlation=corr(
                            baseline, predict(changed)
                        ),
                    )
                )
        result = dict(
            cell=cell_name,
            checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            dates=indices.tolist(),
            loss=float(loss),
            gradient_norm=float(norm),
            gradient_energy_by_group=groups,
            perturbations=shocks,
            ablations=ablations,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
    finally:
        dataset.store.close()
Path(
    "D:/quant-data/b3/interim/round7_postmortem_20260913/gradient_probes.json"
).write_text(json.dumps(results, indent=2))
