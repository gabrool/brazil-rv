from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from brazil_rv.preprocessing.b3_options_open_interest import (
    FEATURES as OPTIONS_FEATURES,
)
from brazil_rv.preprocessing.bdi_lending_strong import FEATURES as LENDING_FEATURES
from brazil_rv.preprocessing.dce_iron_ore import FEATURES as DCE_FEATURES
from brazil_rv.preprocessing.external_sidecar import materialize_external_sidecar
from brazil_rv.preprocessing.external_sidecar_subset import subset_external_sidecar

from .data import feature_store_identity, load_external_sidecar
from .provenance import repository_commit
from .three_fold_sidecar_screen import run_three_fold_sidecar_screen

PAIR_FEATURES = (
    "late_market_momentum_beta",
    "hks_same_interval_return_lag5",
)
CANDIDATE_ORDER = (
    "feature_full_eight",
    "feature_orthogonal_pair",
    "p2_lending_rates_flows",
    "p2_options_open_interest",
    "p2_dce_iron_ore",
)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_sidecar(
    *,
    store: Path,
    source: Path,
    output_dir: Path,
    features: Sequence[str],
) -> Path:
    return materialize_external_sidecar(
        store=store,
        source=source,
        output_dir=output_dir,
        cadence="daily",
        features=features,
        date_column="available_date",
        source_date_column="source_trade_date",
    )


def run_program(
    *,
    store: Path,
    selected_feature_sidecar: Path,
    lending_source: Path,
    options_source: Path,
    dce_source: Path,
    parent_campaign: Path,
    fold_c_parent: Path,
    fold_c_parent_replay_report: Path,
    run_root: Path,
    output_dir: Path,
    parallel_processes: int = 2,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if parallel_processes != 2:
        raise ValueError("The preregistered program uses exactly two GH200 processes")
    commit = repository_commit()
    selected = load_external_sidecar(selected_feature_sidecar, store)
    expected_full = (
        "vwap_reversal_15m_cs",
        "late_market_momentum_beta",
        "overnight_minus_intraday_20d_cs",
        "signed_semivariance_1d",
        "edge_spread_60m_cs",
        "vwap_reversal_volume_flip",
        "amihud_30m_cs",
        "hks_same_interval_return_lag5",
    )
    if selected.feature_names != expected_full:
        raise ValueError(
            "The frozen F2 eight-feature sidecar differs from Experiment 39"
        )

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "program_manifest.json"
    manifest: dict[str, object] = {
        "schema": "FINAL_FEATURE_P2_PROGRAM_V1",
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository_commit": commit,
        "feature_store": str(store.resolve()),
        "feature_store_identity": feature_store_identity(store),
        "candidate_order": list(CANDIDATE_ORDER),
        "feature_program_contract": {
            "full_eight": "features only; no incumbent-field pruning",
            "orthogonal_pair": list(PAIR_FEATURES),
            "pair_role": "fixed confirmatory closure; not selected after full-eight result",
            "this_is_the_final_feature_program_test": True,
        },
        "p2_contract": {
            "candidates": [
                "B3 registered lending rates and flows",
                "B3 listed-equity option open interest",
                "DCE iron-ore settlement/open-interest state",
            ],
            "each_source_screened_independently": True,
        },
        "folds": ["fold_c", "fold_a", "fold_b"],
        "primary_gate": "mean delta >= +0.001 and every fold delta >= 0",
        "diversity_guardrail": "standalone delta >= -0.001 on every fold",
        "retention_comparator": "canonical_parent_only",
        "designated_challenger_role": "informational_only_on_folds_a_b",
        "base_feature_pruning": False,
        "official_validation_accessed": False,
        "test_accessed": False,
        "candidates": {name: {"status": "pending"} for name in CANDIDATE_ORDER},
    }
    _atomic_json(manifest_path, manifest)

    try:
        sidecar_root = output_dir / "sidecars"
        pair_sidecar = subset_external_sidecar(
            store=store,
            source_dir=selected_feature_sidecar,
            output_dir=sidecar_root / "feature_orthogonal_pair",
            features=PAIR_FEATURES,
        )
        lending_sidecar = _source_sidecar(
            store=store,
            source=lending_source,
            output_dir=sidecar_root / "p2_lending_rates_flows",
            features=LENDING_FEATURES,
        )
        options_sidecar = _source_sidecar(
            store=store,
            source=options_source,
            output_dir=sidecar_root / "p2_options_open_interest",
            features=OPTIONS_FEATURES,
        )
        dce_sidecar = _source_sidecar(
            store=store,
            source=dce_source,
            output_dir=sidecar_root / "p2_dce_iron_ore",
            features=DCE_FEATURES,
        )
        specifications = (
            (
                "feature_full_eight",
                selected_feature_sidecar,
                "final_feature_program_features_only_no_pruning",
            ),
            (
                "feature_orthogonal_pair",
                pair_sidecar,
                "fixed_near_orthogonal_pair_confirmatory_closure",
            ),
            (
                "p2_lending_rates_flows",
                lending_sidecar,
                "p2_strong_form_lending_rates_and_registered_flows",
            ),
            (
                "p2_options_open_interest",
                options_sidecar,
                "p2_strong_form_listed_equity_option_open_interest",
            ),
            (
                "p2_dce_iron_ore",
                dce_sidecar,
                "p2_contract_specific_iron_ore_settlement_and_open_interest",
            ),
        )
        for name, sidecar, role in specifications:
            manifest["candidates"][name] = {
                "status": "running",
                "sidecar": load_external_sidecar(sidecar, store).identity,
            }
            _atomic_json(manifest_path, manifest)
            screen = run_three_fold_sidecar_screen(
                store=store,
                sidecar_dir=sidecar,
                candidate_name=name,
                parent_campaign=parent_campaign,
                fold_c_parent=fold_c_parent,
                fold_c_parent_replay_report=fold_c_parent_replay_report,
                run_root=run_root,
                output_dir=output_dir / "candidates" / name,
                experiment_role=role,
                parallel_processes=parallel_processes,
            )
            summary = json.loads(
                (screen / "screen_summary.json").read_text(encoding="utf-8")
            )
            manifest["candidates"][name] = {
                "status": "completed",
                "sidecar": load_external_sidecar(sidecar, store).identity,
                "screen_summary": str((screen / "screen_summary.json").resolve()),
                "candidate_retained": summary["candidate_retained"],
            }
            _atomic_json(manifest_path, manifest)
        program_summary = {
            "schema": "FINAL_FEATURE_P2_PROGRAM_SUMMARY_V1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "candidate_order": list(CANDIDATE_ORDER),
            "candidates": manifest["candidates"],
            "feature_program_closed_after_this_screen": True,
            "official_validation_accessed": False,
            "test_accessed": False,
        }
        summary_path = output_dir / "program_summary.json"
        _atomic_json(summary_path, program_summary)
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "program_summary": str(summary_path.resolve()),
            }
        )
        _atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the final features-only closure and P2 strong-source screens"
    )
    for name in (
        "store",
        "selected_feature_sidecar",
        "lending_source",
        "options_source",
        "dce_source",
        "parent_campaign",
        "fold_c_parent",
        "fold_c_parent_replay_report",
        "run_root",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--parallel-processes", type=int, default=2)
    print(run_program(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
