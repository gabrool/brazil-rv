from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from brazil_rv.execution.stateful_ledger import LedgerConfig, simulate_stateful_ledger
from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.evaluate import _aligned_action_terms, _economics_signal
from brazil_rv.v2.store import open_store_for_samples
from brazil_rv.v2.validate_pipeline import (
    _evaluation_inputs,
    _load_development_cdi,
    _read_store_header,
)


MATERIAL_LATER_PRINT_SHARE = 0.10
TENDER_FLAT_RANGE_FRACTION = 0.005
TENDER_PREMIUM_FRACTION = 0.05


def _read_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


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


def _positive(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0.0)


def _tender_metrics(
    *, close: np.ndarray, dates: np.ndarray, last_print_index: int
) -> dict[str, object]:
    start = max(0, last_print_index - 19)
    path = np.asarray(close[start : last_print_index + 1], dtype=np.float64)
    path_dates = dates[start : last_print_index + 1]
    observed = path[_positive(path)]
    terminal = observed[-5:]
    prior = observed[:-5]
    terminal_range = (
        float((terminal.max() - terminal.min()) / terminal.mean())
        if terminal.size >= 2 and terminal.mean() > 0.0
        else None
    )
    premium = (
        float(observed[-1] / prior.mean() - 1.0)
        if prior.size and prior.mean() > 0.0
        else None
    )
    tender_like = bool(
        terminal_range is not None
        and premium is not None
        and terminal_range <= TENDER_FLAT_RANGE_FRACTION
        and premium >= TENDER_PREMIUM_FRACTION
    )
    return {
        "close_path_dates": [str(value)[:10] for value in path_dates],
        "close_path": [float(value) if np.isfinite(value) else None for value in path],
        "close_path_observed_count": int(observed.size),
        "terminal_five_range_fraction": terminal_range,
        "last_close_premium_to_prior_path_mean": premium,
        "tender_like_signature": tender_like,
    }


def _episodes(signs: np.ndarray) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    start: int | None = None
    side = 0
    for index, value in enumerate(signs.tolist() + [0]):
        current = int(np.sign(value))
        if start is None and current:
            start, side = index, current
        elif start is not None and current != side:
            rows.append((start, index - 1, side))
            start = index if current else None
            side = current
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--store-manifest-sha256", required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--acceptance-manifest-sha256", required=True)
    parser.add_argument("--acceptance-inventory-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    _read_json(
        acceptance_root / "artifact_inventory.json",
        args.acceptance_inventory_sha256,
    )
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
    full_close = np.load(
        store_root / "raw_close.npy", mmap_mode="r", allow_pickle=False
    )
    isins = np.load(store_root / "isin_index.npy", allow_pickle=False).astype(str)
    master = pl.read_parquet(store_root / "security_master.parquet")
    ticker_by_isin = dict(
        zip(master["isin"].to_list(), master["ticker"].to_list(), strict=True)
    )
    candidates = pl.read_parquet(store_root / "isin_succession_candidates.parquet")
    succession_isins = set(candidates["predecessor_isin"].to_list()) | set(
        candidates["successor_isin"].to_list()
    )

    position_rows: list[dict[str, object]] = []
    book_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    records = [
        *acceptance["results"]["baselines"],
        *acceptance["results"]["gbdt_triage"],
    ]
    for record in records:
        score_manifest_path = Path(record["score_manifest"]).resolve(strict=True)
        score_manifest = _read_json(
            score_manifest_path, record["score_manifest_sha256"]
        )
        score_root = score_manifest_path.parent
        scores = _load_array(
            score_root / "scores.npy", score_manifest["artifacts"]["scores.npy"]
        )
        score_mask = _load_array(
            score_root / "score_mask.npy", score_manifest["artifacts"]["score_mask.npy"]
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

        sealed = record["evaluation"]["headline_economics"]
        replayed = result.summary()["mean_unresolved_stale_inventory_fraction_nav"]
        if not np.isclose(
            replayed,
            sealed["mean_unresolved_stale_inventory_fraction_nav"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise RuntimeError(
                "diagnostic replay differs from sealed headline economics"
            )
        shares = np.asarray(result.signed_shares)
        marks = np.asarray(result.mark_price)
        printed = _positive(np.asarray(inputs.raw_close))
        held = shares != 0.0
        stale = held & ~printed
        fold = str(record["fold"])
        book = str(record.get("name", "gbdt_ensemble"))
        later_printed_name_days = 0
        book_positions: list[dict[str, object]] = []
        for name in np.flatnonzero(stale.any(axis=0)):
            for entry, end, side in _episodes(shares[:, name]):
                stale_days = np.flatnonzero(stale[entry : end + 1, name]) + entry
                if not stale_days.size:
                    continue
                first_stale = int(stale_days[0])
                last_stale = int(stale_days[-1])
                global_first_stale = int(indices[first_stale])
                earlier = np.flatnonzero(
                    _positive(np.asarray(full_close[:global_first_stale, name]))
                )
                if not earlier.size:
                    raise RuntimeError("stale position lacks a prior positive mark")
                last_print = int(earlier[-1])
                later_global = np.flatnonzero(
                    _positive(np.asarray(full_close[global_first_stale + 1 :, name]))
                )
                next_print = (
                    global_first_stale + 1 + int(later_global[0])
                    if later_global.size
                    else None
                )
                later_eval = bool(printed[first_stale + 1 : end + 1, name].any())
                later_printed_days_for_position = (
                    int(stale_days.size) if later_eval else 0
                )
                later_printed_name_days += later_printed_days_for_position
                isin = str(isins[name])
                first_notional = float(
                    abs(shares[first_stale, name] * marks[first_stale, name])
                )
                first_fraction = float(first_notional / result.nav[first_stale])
                row = {
                    "fold": fold,
                    "book": book,
                    "security_index": int(name),
                    "isin": isin,
                    "ticker": ticker_by_isin.get(isin),
                    "side": "long" if side > 0 else "short",
                    "entry_date": inputs.dates[entry],
                    "position_end_date": inputs.dates[end],
                    "first_stale_date": inputs.dates[first_stale],
                    "last_stale_date": inputs.dates[last_stale],
                    "stale_name_days": int(stale_days.size),
                    "last_print_before_stale_date": store_dates[last_print]
                    .astype("datetime64[D]")
                    .item(),
                    "last_mark": float(full_close[last_print, name]),
                    "notional_at_last_mark": first_notional,
                    "notional_fraction_nav_at_first_stale": first_fraction,
                    "prints_again_within_position_window": later_eval,
                    "first_later_print_anywhere_date": (
                        store_dates[next_print].astype("datetime64[D]").item()
                        if next_print is not None
                        else None
                    ),
                    "prints_again_anywhere_in_store": next_print is not None,
                    "succession_candidate": isin in succession_isins,
                    **_tender_metrics(
                        close=np.asarray(full_close[:, name]),
                        dates=store_dates,
                        last_print_index=last_print,
                    ),
                }
                position_rows.append(row)
                book_positions.append(row)
        stale_count = int(stale.sum())
        book_rows.append(
            {
                "fold": fold,
                "book": book,
                "position_count": len(book_positions),
                "distinct_isin_count": len(
                    {str(row["isin"]) for row in book_positions}
                ),
                "stale_name_days": stale_count,
                "stale_name_days_from_positions_that_print_again": later_printed_name_days,
                "later_printed_stale_name_day_share": (
                    later_printed_name_days / stale_count if stale_count else 0.0
                ),
                "summed_position_notional_fraction_nav_at_first_stale": float(
                    sum(
                        float(row["notional_fraction_nav_at_first_stale"])
                        for row in book_positions
                    )
                ),
                "tender_like_position_count": sum(
                    bool(row["tender_like_signature"]) for row in book_positions
                ),
                "succession_candidate_position_count": sum(
                    bool(row["succession_candidate"]) for row in book_positions
                ),
            }
        )
        replay_rows.append(
            {
                "fold": fold,
                "book": book,
                "sealed_mean_unresolved_stale_fraction": float(
                    sealed["mean_unresolved_stale_inventory_fraction_nav"]
                ),
                "replayed_mean_unresolved_stale_fraction": float(replayed),
            }
        )

    stale_name_days = sum(int(row["stale_name_days"]) for row in book_rows)
    later_name_days = sum(
        int(row["stale_name_days_from_positions_that_print_again"]) for row in book_rows
    )
    later_share = later_name_days / stale_name_days if stale_name_days else 0.0
    tender_count = sum(bool(row["tender_like_signature"]) for row in position_rows)
    payload = {
        "schema": "BRAZIL_RV_V2_PASS4F_STALE_HOLDING_DIAGNOSTIC_V1",
        "status": "stop_fill_defect"
        if later_share >= MATERIAL_LATER_PRINT_SHARE
        else "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "score-free reconstruction of stale holdings from sealed Pass-4e panels",
        "code": code,
        "sources": {
            "store_root": str(store_root),
            "store_manifest_sha256": args.store_manifest_sha256,
            "acceptance_root": str(acceptance_root),
            "acceptance_manifest_sha256": args.acceptance_manifest_sha256,
            "acceptance_inventory_sha256": args.acceptance_inventory_sha256,
        },
        "definitions": {
            "material_later_print_share": MATERIAL_LATER_PRINT_SHARE,
            "material_share_unit": "stale name-days attributable to a position that prints again within its evaluation holding episode",
            "tender_like_signature": (
                "last five observed closes span <=0.5% of their mean and the last close "
                "is >=5% above the mean of earlier observed closes in the 20-session path"
            ),
        },
        "summary": {
            "book_count": len(book_rows),
            "position_count": len(position_rows),
            "distinct_isin_count": len({str(row["isin"]) for row in position_rows}),
            "stale_name_days": stale_name_days,
            "later_printed_stale_name_days": later_name_days,
            "later_printed_stale_name_day_share": later_share,
            "positions_printing_again_anywhere": sum(
                bool(row["prints_again_anywhere_in_store"]) for row in position_rows
            ),
            "positions_printing_again_within_window": sum(
                bool(row["prints_again_within_position_window"])
                for row in position_rows
            ),
            "tender_like_position_count": tender_count,
            "tender_like_position_fraction": (
                tender_count / len(position_rows) if position_rows else 0.0
            ),
            "succession_candidate_position_count": sum(
                bool(row["succession_candidate"]) for row in position_rows
            ),
            "fill_defect_stop_required": later_share >= MATERIAL_LATER_PRINT_SHARE,
        },
        "replay_identity": replay_rows,
        "official_validation_accessed": False,
        "test_accessed": False,
        "research_claim": False,
        "transfer_chronology_clean": True,
        "deployment_changed": False,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output_root / "diagnostic_source.py")
    pl.DataFrame(position_rows).write_parquet(output_root / "stale_positions.parquet")
    pl.DataFrame(book_rows).write_parquet(output_root / "summary_by_book.parquet")
    payload["artifacts"] = {
        name: {
            "bytes": (output_root / name).stat().st_size,
            "sha256": sha256_file(output_root / name),
        }
        for name in (
            "diagnostic_source.py",
            "stale_positions.parquet",
            "summary_by_book.parquet",
        )
    }
    digest = write_json_atomic(output_root / "diagnostic_manifest.json", payload)
    print(
        json.dumps(
            {"root": str(output_root), "manifest_sha256": digest, **payload["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
