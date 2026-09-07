from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from brazil_rv.execution.stateful_ledger import LedgerConfig, simulate_stateful_ledger
from brazil_rv.v2.artifacts import inventory, sha256_file, write_json_atomic
from brazil_rv.v2.evaluate import _aligned_action_terms, _economics_signal
from brazil_rv.v2.store import open_store_for_samples
from brazil_rv.v2.validate_pipeline import (
    _evaluation_inputs,
    _load_development_cdi,
    _read_store_header,
)


def _read_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _verify_bound_rows(root: Path, payload: dict[str, Any]) -> None:
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ValueError("bound inventory lacks file rows")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("bound inventory row is malformed")
        path = root / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"bound inventory size mismatch: {path}")
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError(f"bound inventory hash mismatch: {path}")


def _load_array(path: Path, record: dict[str, Any]) -> np.ndarray:
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"array SHA-256 mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record["shape"] or values.dtype.str != record["dtype"]:
        raise ValueError(f"array shape/dtype mismatch: {path}")
    return values


def _git_identity(repo: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    if dirty:
        raise RuntimeError("diagnostic requires a clean Git worktree")
    return {"commit": commit, "worktree_clean": True}


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else math.nan


def _replayed_headline(result: Any) -> dict[str, object]:
    config = LedgerConfig()
    signed_values = np.where(
        (result.signed_shares != 0.0) & np.isfinite(result.mark_price),
        result.signed_shares * result.mark_price,
        0.0,
    )
    deployed_net = np.divide(
        signed_values.sum(axis=1),
        result.nav,
        out=np.full(len(result.dates), np.nan, dtype=np.float64),
        where=result.nav != 0.0,
    )
    return {
        "scenario": "cost_4_borrow_0.02",
        "cost_bps_per_side": config.cost_bps_per_side,
        "annual_borrow_rate": config.annual_borrow_rate,
        "buffer_per_side": config.buffer_per_side,
        "short_proceeds_remuneration": config.short_proceeds_remuneration,
        "terminal_settlement_convention": "last_mark_after_10_sessions",
        "settlement_grace_sessions": config.settlement_grace_sessions,
        "settlement_haircut": config.settlement_haircut,
        "path_model_count": 1,
        **result.summary(),
        "mean_deployed_net_fraction_nav": _finite_mean(deployed_net),
        "terminal_unresolved_inventory_fraction_nav": (
            result.unresolved_inventory_notional / result.nav[-1]
            if result.nav[-1] != 0.0
            else None
        ),
    }


def _assert_headline_identity(
    *, label: str, sealed: dict[str, Any], replayed: dict[str, object]
) -> None:
    missing = sorted(set(sealed) - set(replayed))
    changed = sorted(
        key for key in sealed if key in replayed and sealed[key] != replayed[key]
    )
    if missing or changed:
        raise RuntimeError(
            f"pass-4f headline identity failed for {label}: "
            f"missing={missing}, changed={changed}"
        )


def _daily_rows(result: Any) -> list[dict[str, object]]:
    return [
        {
            "date": day.isoformat(),
            "nav": float(result.nav[index]),
            "gross_fraction_nav": float(result.gross_fraction_nav[index]),
            "k_eff_per_side": int(result.k_eff_per_side[index]),
            "held_count_start_of_day_long": int(
                result.held_count_start_of_day_long[index]
            ),
            "held_count_start_of_day_short": int(
                result.held_count_start_of_day_short[index]
            ),
            "held_count_end_of_day_long": int(result.held_count_end_of_day_long[index]),
            "held_count_end_of_day_short": int(
                result.held_count_end_of_day_short[index]
            ),
            "occupied_after_submission_long": int(
                result.occupied_after_submission_long[index]
            ),
            "occupied_after_submission_short": int(
                result.occupied_after_submission_short[index]
            ),
            "open_slots_after_submission_long": int(
                result.open_slots_after_submission_long[index]
            ),
            "open_slots_after_submission_short": int(
                result.open_slots_after_submission_short[index]
            ),
            "band_candidates_long": int(result.band_candidates_long[index]),
            "band_candidates_short": int(result.band_candidates_short[index]),
            "band_excluded_unresolved_long": int(
                result.band_excluded_unresolved_long[index]
            ),
            "band_excluded_unresolved_short": int(
                result.band_excluded_unresolved_short[index]
            ),
            "band_excluded_settled_long": int(result.band_excluded_settled_long[index]),
            "band_excluded_settled_short": int(
                result.band_excluded_settled_short[index]
            ),
            "band_candidates_without_prior_session_print_long": int(
                result.band_candidates_without_prior_session_print_long[index]
            ),
            "band_candidates_without_prior_session_print_short": int(
                result.band_candidates_without_prior_session_print_short[index]
            ),
            "band_exhausted_open_slots_long": int(
                result.band_exhausted_open_slots_long[index]
            ),
            "band_exhausted_open_slots_short": int(
                result.band_exhausted_open_slots_short[index]
            ),
            "blocked_open_slots_long": int(result.blocked_open_slots_long[index]),
            "blocked_open_slots_short": int(result.blocked_open_slots_short[index]),
            "pending_entries_end_of_day_long": int(
                result.pending_entries_end_of_day_long[index]
            ),
            "pending_entries_end_of_day_short": int(
                result.pending_entries_end_of_day_short[index]
            ),
            "pending_entries_without_print_today_long": int(
                result.pending_entries_without_print_today_long[index]
            ),
            "pending_entries_without_print_today_short": int(
                result.pending_entries_without_print_today_short[index]
            ),
            "slots_freed_by_exit_fill_today_long": int(
                result.slots_freed_by_exit_fill_today_long[index]
            ),
            "slots_freed_by_exit_fill_today_short": int(
                result.slots_freed_by_exit_fill_today_short[index]
            ),
            "exit_instructions_ineligible": int(
                result.exit_instructions_ineligible[index]
            ),
            "exit_instructions_settlement_grace": int(
                result.exit_instructions_settlement_grace[index]
            ),
            "exit_instructions_rank_out_of_retention": int(
                result.exit_instructions_rank_out_of_retention[index]
            ),
            "exit_instructions_terminal": int(result.exit_instructions_terminal[index]),
            "net_cap_block_with_balanced_book": int(
                result.net_cap_block_with_balanced_book[index]
            ),
            "gross_cap_block_below_target": int(
                result.gross_cap_block_below_target[index]
            ),
            "name_cap_block_on_fresh_entry": int(
                result.name_cap_block_on_fresh_entry[index]
            ),
            "entry_pending_printed_unblocked_unfilled": int(
                result.entry_pending_printed_unblocked_unfilled[index]
            ),
            "entry_fill_quantity_short": int(result.entry_fill_quantity_short[index]),
            "pending_entry_count": int(result.pending_entry_count[index]),
            "pending_exit_count": int(result.pending_exit_count[index]),
            "shortfall_small_universe": float(
                result.gross_shortfall_small_universe[index]
            ),
            "shortfall_occupancy_pending": float(
                result.gross_shortfall_occupancy_pending[index]
            ),
            "shortfall_occupancy_band_exhausted": float(
                result.gross_shortfall_occupancy_band_exhausted[index]
            ),
            "shortfall_occupancy_blocked": float(
                result.gross_shortfall_occupancy_blocked[index]
            ),
            "shortfall_occupancy_exit_gap": float(
                result.gross_shortfall_occupancy_exit_gap[index]
            ),
            "shortfall_occupancy_other": float(
                result.gross_shortfall_occupancy_other[index]
            ),
            "shortfall_sizing_fill": float(result.gross_shortfall_sizing_fill[index]),
            "shortfall_sizing_mark_drift": float(
                result.gross_shortfall_sizing_mark_drift[index]
            ),
            "shortfall_sizing_nav_drift": float(
                result.gross_shortfall_sizing_nav_drift[index]
            ),
        }
        for index, day in enumerate(result.dates)
    ]


def _entry_bands(
    *, scores: np.ndarray, mask: np.ndarray, k_per_side: int
) -> np.ndarray:
    bands = np.zeros(scores.shape, dtype=np.int8)
    for day in range(scores.shape[0]):
        names = np.flatnonzero(mask[day] & np.isfinite(scores[day]))
        order = (
            names[np.argsort(scores[day, names], kind="stable")]
            if names.size
            else names
        )
        k_eff = min(k_per_side, len(order) // 2)
        if k_eff:
            bands[day, order[-k_eff:]] = 1
            bands[day, order[:k_eff]] = -1
    return bands


def _band_rows(
    *, result: Any, inputs: Any, economics_score: np.ndarray, economics_mask: np.ndarray
) -> list[dict[str, object]]:
    bands = _entry_bands(
        scores=economics_score,
        mask=economics_mask,
        k_per_side=LedgerConfig().k_per_side,
    )
    in_band = bands != 0
    printed = np.isfinite(inputs.raw_close) & (inputs.raw_close > 0.0)
    prior_stale = np.zeros_like(in_band)
    last_print = np.where(
        np.isfinite(inputs.initial_reference_price)
        & (inputs.initial_reference_price > 0.0),
        -1,
        -2,
    ).astype(np.int64)
    for day in range(len(inputs.dates)):
        prior_stale[day] = in_band[day] & (last_print < day - 1)
        last_print[printed[day]] = day
    entry_orders = [
        order for order in result.intended_orders if order.purpose == "entry"
    ]
    entry_fills = [fill for fill in result.fills if fill.purpose == "entry"]
    expired = [
        cancellation
        for cancellation in result.cancellations
        if cancellation.reason == "expired"
    ]
    minimum_band_sessions = 0.10 * len(inputs.dates)
    rows: list[dict[str, object]] = []
    for name in np.flatnonzero(in_band.sum(axis=0) >= minimum_band_sessions):
        rows.append(
            {
                "security_index": int(name),
                "security_id": str(inputs.security_ids[name]),
                "sessions_in_band": int(in_band[:, name].sum()),
                "sessions_in_long_band": int((bands[:, name] > 0).sum()),
                "sessions_in_short_band": int((bands[:, name] < 0).sum()),
                "sessions_held": int((result.position_sign[:, name] != 0).sum()),
                "entries_submitted": sum(
                    order.security_index == name for order in entry_orders
                ),
                "entry_fills": sum(fill.security_index == name for fill in entry_fills),
                "expiry_cancellations": sum(
                    cancellation.security_index == name for cancellation in expired
                ),
                "sessions_printed_in_window": int(printed[:, name].sum()),
                "sessions_in_window": len(inputs.dates),
                "sessions_without_prior_print_while_in_band": int(
                    prior_stale[:, name].sum()
                ),
                "unresolved_action_sessions_while_in_band": int(
                    (in_band[:, name] & ~inputs.action_session_resolved).sum()
                ),
            }
        )
    return rows


def _largest_term(
    payload: dict[str, float], names: tuple[str, ...]
) -> dict[str, object]:
    name = max(names, key=lambda item: abs(payload[item]))
    return {"name": name, "value": payload[name]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--store-manifest-sha256", required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--acceptance-manifest-sha256", required=True)
    parser.add_argument("--acceptance-inventory-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--stderr-log", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    store_root = args.store_root.resolve(strict=True)
    acceptance_root = args.acceptance_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    code = _git_identity(repo)
    if sha256_file(store_root / "manifest.json") != args.store_manifest_sha256:
        raise ValueError("sealed store manifest mismatch")
    acceptance = _read_json(
        acceptance_root / "pipeline_validation_manifest.json",
        args.acceptance_manifest_sha256,
    )
    runner_inventory = _read_json(
        acceptance_root / "inventory.json", args.acceptance_inventory_sha256
    )
    _verify_bound_rows(acceptance_root, runner_inventory)
    if acceptance.get("official_validation_accessed") or acceptance.get(
        "test_accessed"
    ):
        raise PermissionError("diagnostic refuses protected-data access")

    store_manifest, store_dates = _read_store_header(store_root)
    cdi_source = acceptance["sources"]["cdi"]
    cdi_by_index, _ = _load_development_cdi(
        dates=store_dates,
        cdi_path=Path(cdi_source["development_extension"]["path"]),
        expected_sha256=cdi_source["development_extension"]["sha256"],
        experiment52_cdi_path=Path(cdi_source["experiment52_reference"]["path"]),
        experiment52_expected_sha256=cdi_source["experiment52_reference"]["sha256"],
    )
    records = [
        *acceptance["results"]["baselines"],
        *acceptance["results"]["gbdt_triage"],
    ]
    if len(records) != 16:
        raise ValueError("pass-4g diagnostic requires exactly 16 sealed books")

    output_root.mkdir(parents=True, exist_ok=False)
    summary_rows: list[dict[str, object]] = []
    access_rows: list[dict[str, object]] = []
    f2_never_held: list[dict[str, object]] = []
    for record in records:
        fold = str(record["fold"])
        book = str(record.get("name", "gbdt_ensemble"))
        label = f"{fold}:{book}"
        score_manifest_path = Path(record["score_manifest"]).resolve(strict=True)
        score_manifest = _read_json(
            score_manifest_path, record["score_manifest_sha256"]
        )
        score_root = score_manifest_path.parent
        scores = _load_array(
            score_root / "scores.npy", score_manifest["artifacts"]["scores.npy"]
        )
        score_mask = _load_array(
            score_root / "score_mask.npy",
            score_manifest["artifacts"]["score_mask.npy"],
        ).astype(np.bool_, copy=False)
        metadata = score_manifest["metadata"]
        raw_indices = metadata.get(
            "date_indices", metadata.get("evaluation_date_indices")
        )
        if raw_indices is None:
            raise ValueError("score manifest lacks evaluation date indices")
        indices = np.asarray(raw_indices, dtype=np.int64)
        store, access = open_store_for_samples(
            store_root,
            indices,
            purpose="evaluation",
            history_lookbacks=np.full(indices.size, 253, dtype=np.int64),
            history_end_offsets=np.full(indices.size, -1, dtype=np.int64),
        )
        try:
            if access.official_validation_accessed or access.test_accessed:
                raise PermissionError("protected data was unexpectedly authorized")
            inputs = _evaluation_inputs(
                store,
                indices,
                scores,
                score_mask,
                cdi_by_index,
                {"v2_store_manifest": args.store_manifest_sha256},
                transfer_chronology_clean=True,
            )
            if any(day.year >= 2025 for day in inputs.dates):
                raise PermissionError("diagnostic refuses every 2025/2026 session")
            economics_score, economics_mask = _economics_signal(inputs)
            result = simulate_stateful_ledger(
                dates=inputs.dates,
                scores=economics_score,
                score_mask=economics_mask,
                active=np.asarray(inputs.active, dtype=np.bool_),
                raw_close=inputs.raw_close,
                action_terms=_aligned_action_terms(inputs),
                action_payment_session=inputs.action_payment_session,
                cdi_returns=inputs.cdi_returns,
                security_ids=inputs.security_ids,
                config=LedgerConfig(),
                initial_reference_price=inputs.initial_reference_price,
            )
        finally:
            store.close()
        access_rows.append(access.payload())

        sealed = record["evaluation"]["headline_economics"]
        replayed = _replayed_headline(result)
        _assert_headline_identity(label=label, sealed=sealed, replayed=replayed)
        summary = result.summary()
        decomposition = summary["gross_shortfall_decomposition"]
        signatures = summary["entry_defect_signatures"]
        assert isinstance(decomposition, dict) and isinstance(signatures, dict)
        book_root = output_root / "books" / fold / book
        book_root.mkdir(parents=True, exist_ok=False)
        daily_rows = _daily_rows(result)
        band_rows = _band_rows(
            result=result,
            inputs=inputs,
            economics_score=economics_score,
            economics_mask=economics_mask,
        )
        pl.DataFrame(daily_rows).write_csv(book_root / "occupancy_daily.csv")
        pl.DataFrame(band_rows).write_csv(book_root / "band_names.csv")
        write_json_atomic(
            book_root / "decomposition.json",
            {
                "schema": "BRAZIL_RV_V2_PASS4G_DECOMPOSITION_V1",
                "fold": fold,
                "book": book,
                "pass4f_headline_bit_identical": True,
                "pass4f_headline": sealed,
                "gross_shortfall_decomposition": decomposition,
                "entry_defect_signatures": signatures,
                "exit_instructions_by_cause": summary["exit_instructions_by_cause"],
            },
        )
        occupancy_names = (
            "occupancy_pending",
            "occupancy_band_exhausted",
            "occupancy_blocked",
            "occupancy_exit_gap",
            "occupancy_other",
        )
        sizing_names = ("sizing_fill", "sizing_mark_drift", "sizing_nav_drift")
        largest_occupancy = _largest_term(decomposition, occupancy_names)
        largest_sizing = _largest_term(decomposition, sizing_names)
        row = {
            "fold": fold,
            "book": book,
            "mean_gross_fraction_nav": summary["mean_gross_fraction_nav"],
            "total_shortfall": decomposition["total"],
            "occupancy_share": decomposition["occupancy_share"],
            "largest_occupancy_subterm": largest_occupancy["name"],
            "largest_occupancy_subterm_value": largest_occupancy["value"],
            "largest_sizing_subterm": largest_sizing["name"],
            "largest_sizing_subterm_value": largest_sizing["value"],
            **signatures,
            "pass4f_headline_bit_identical": True,
        }
        summary_rows.append(row)
        if fold == "F2" and book == "inverse_volatility_20":
            f2_never_held = sorted(
                (row for row in band_rows if int(row["sessions_held"]) == 0),
                key=lambda row: (
                    -int(row["sessions_in_band"]),
                    int(row["security_index"]),
                ),
            )[:10]
        print(json.dumps({"completed": label, "summary": row}), flush=True)

    p0 = any(
        int(row[f"D{number}_{name}"]) > 0
        for row in summary_rows
        for number, name in (
            (1, "entry_pending_printed_unblocked_unfilled"),
            (2, "entry_fill_quantity_short"),
            (3, "blocked_open_slots"),
            (4, "cap_block_defects"),
        )
    ) or any(
        float(row["D5_eligibility_flicker_exit_share"]) > 0.10 for row in summary_rows
    )
    target = next(
        row
        for row in summary_rows
        if row["fold"] == "F2" and row["book"] == "inverse_volatility_20"
    )
    disposition = (
        "P0" if p0 else "P1" if float(target["occupancy_share"]) >= 0.30 else "P2"
    )
    pl.DataFrame(summary_rows).write_csv(output_root / "summary_by_book.csv")
    shutil.copyfile(Path(__file__), output_root / "diagnostic_source.py")
    manifest = {
        "schema": "BRAZIL_RV_V2_PASS4G_OCCUPANCY_DIAGNOSTIC_V1",
        "status": "stop_defect_signature" if disposition == "P0" else "passed",
        "disposition": disposition,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "score-free occupancy reconstruction from sealed pass-4f panels",
        "code": code,
        "sources": {
            "store_root": str(store_root),
            "store_manifest_sha256": args.store_manifest_sha256,
            "acceptance_root": str(acceptance_root),
            "acceptance_manifest_sha256": args.acceptance_manifest_sha256,
            "acceptance_inventory_sha256": args.acceptance_inventory_sha256,
        },
        "source_tier_labels": {
            "action_terms_source": store_manifest["metadata"]["action_terms_source"],
            "schedule_source": store_manifest["metadata"]["schedule_source"],
        },
        "bit_identity": {
            "book_count": len(summary_rows),
            "all_pass4f_headlines_bit_identical": True,
        },
        "f2_inverse_volatility": {
            "occupancy_share": target["occupancy_share"],
            "largest_occupancy_subterm": {
                "name": target["largest_occupancy_subterm"],
                "value": target["largest_occupancy_subterm_value"],
            },
            "largest_sizing_subterm": {
                "name": target["largest_sizing_subterm"],
                "value": target["largest_sizing_subterm_value"],
            },
            "top_ten_band_names_never_held": f2_never_held,
        },
        "book_summaries": summary_rows,
        "official_validation_accessed": False,
        "test_accessed": False,
        "research_claim": False,
        "transfer_chronology_clean": True,
        "deployment_changed": False,
    }
    manifest_sha = write_json_atomic(output_root / "diagnostic_manifest.json", manifest)
    access_audit = {
        "schema": "BRAZIL_RV_V2_PASS4G_ACCESS_AUDIT_V1",
        "status": "passed",
        "diagnostic_manifest_sha256": manifest_sha,
        "source_access_ledgers": access_rows,
        "official_validation_accessed": False,
        "test_accessed": False,
        "research_claim": False,
        "deployment_changed": False,
    }
    write_json_atomic(output_root / "access_audit.json", access_audit)
    print(
        json.dumps(
            {
                "root": str(output_root),
                "manifest_sha256": manifest_sha,
                "disposition": disposition,
            }
        ),
        flush=True,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    operational = output_root / "operational_logs"
    operational.mkdir()
    shutil.copyfile(args.stdout_log.resolve(strict=True), operational / "stdout.log")
    shutil.copyfile(args.stderr_log.resolve(strict=True), operational / "stderr.log")
    excluded = {"artifact_inventory.json", "artifact_inventory.json.sha256"}
    rows = inventory(output_root, exclude=excluded)
    inventory_payload = {
        "schema": "BRAZIL_RV_V2_RESEARCH_INVENTORY_V1",
        "status": "passed",
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "excluded_self": sorted(excluded),
        "official_validation_accessed": False,
        "test_accessed": False,
        "research_claim": False,
        "transfer_chronology_clean": True,
        "deployment_changed": False,
    }
    write_json_atomic(output_root / "artifact_inventory.json", inventory_payload)
    _verify_bound_rows(output_root, inventory_payload)


if __name__ == "__main__":
    main()
