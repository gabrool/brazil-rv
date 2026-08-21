from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .analyze import select_trajectory_rule
from .contract import ALLOWED_SEEDS
from .data import feature_store_identity, load_external_sidecar, resolve_feature_store
from .engine import objective_metadata
from .provenance import repository_commit
from .train import run_training

DISCOVERY_FOLDS = ("fold_a", "fold_b")
EXTERNAL_DATA_READOUT_CONTRACT = {
    "primary": "bidirectional_odd_even_crossfit_patience3_raw",
    "secondary": "final_ema_0995",
    "trajectory_rule_reselection": False,
}


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _completed_run_matches(
    run_dir: Path,
    *,
    store: Path,
    fold: str,
    seed: int,
    commit: str,
    external_sidecar: dict[str, object] | None,
) -> bool:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return bool(
        manifest.get("status") == "completed"
        and manifest.get("repository_commit") == commit
        and Path(str(manifest.get("feature_store"))).resolve() == store.resolve()
        and manifest.get("seed") == seed
        and manifest.get("split", {}).get("training") == fold
        and manifest.get("split", {}).get("test_accessed") is False
        and manifest.get("objective") == objective_metadata()
        and manifest.get("external_sidecar") == external_sidecar
    )


def run_campaign(
    store: Path,
    output_dir: Path,
    *,
    sidecar_dir: Path | None = None,
) -> Path:
    commit = repository_commit()
    identity = feature_store_identity(store)
    sidecar = None if sidecar_dir is None else load_external_sidecar(sidecar_dir, store)
    sidecar_identity = None if sidecar is None else sidecar.identity
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "campaign_manifest.json"
    manifest = {
        "schema": "DISCOVERY_TRAJECTORY_CAMPAIGN",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "feature_store": str(store.resolve()),
        "feature_store_identity": identity,
        "external_sidecar": sidecar_identity,
        "folds": list(DISCOVERY_FOLDS),
        "seeds": list(ALLOWED_SEEDS),
        "objective": objective_metadata(),
        "official_validation_accessed": False,
        "test_accessed": False,
    }
    if sidecar is not None:
        manifest["external_data_readout_contract"] = EXTERNAL_DATA_READOUT_CONTRACT
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = {
            key: manifest[key]
            for key in (
                "schema",
                "repository_commit",
                "feature_store",
                "feature_store_identity",
                "external_sidecar",
                *(("external_data_readout_contract",) if sidecar is not None else ()),
                "folds",
                "seeds",
                "objective",
                "official_validation_accessed",
                "test_accessed",
            )
        }
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise ValueError("Existing campaign manifest has a different contract")
        manifest["created_at"] = existing["created_at"]
    _atomic_json(manifest_path, manifest)
    fold_runs: dict[str, list[Path]] = {fold: [] for fold in DISCOVERY_FOLDS}
    for fold in DISCOVERY_FOLDS:
        for seed in ALLOWED_SEEDS:
            run_dir = output_dir / fold / f"seed_{seed}"
            if run_dir.exists():
                if not _completed_run_matches(
                    run_dir,
                    store=store,
                    fold=fold,
                    seed=seed,
                    commit=commit,
                    external_sidecar=sidecar_identity,
                ):
                    raise ValueError(f"Existing run does not match campaign: {run_dir}")
            else:
                run_training(
                    store=store,
                    seed=seed,
                    selection_window=fold,
                    run_dir=run_dir,
                    sidecar_dir=sidecar_dir,
                )
            fold_runs[fold].append(run_dir)
    selection = (
        None
        if sidecar is not None
        else select_trajectory_rule(fold_runs, output_dir / "trajectory_selection.json")
    )
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "trajectory_selection": (
                None if selection is None else str(selection.resolve())
            ),
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-fold, three-seed trajectory screen"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sidecar-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        run_campaign(
            resolve_feature_store(),
            args.output_dir,
            sidecar_dir=args.sidecar_dir,
        )
    )


if __name__ == "__main__":
    main()
