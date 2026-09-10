"""Hash-bound ledger-only repairs of the 465 final Round-4 books.

Scores, targets, populations and score-only statistics are retained exactly. The
only replaceable inputs are BOVA closes/economic betas and independently admitted
observed lending rates. Shortability, balances and forecast sidecars stay frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import evaluate as ev
from . import research_rounds as rr
from .artifacts import sha256_file, write_json_atomic
from .bova11 import load_bova11_series
from .contract import DEVELOPMENT_END
from .execution_policy import ExecutionPolicy, ledger_gate_failures
from .hedge_beta import HEDGE_BETA_MAX_AGE, load_hedge_beta_sidecar
from .lending_archive import load_lending_borrow_panels
from .store import peak_rss_bytes

SCHEMA = "BRAZIL_RV_V2_ROUND5_LEDGER_REPLAY_V1"
REGISTRATION = (
    Path(__file__).resolve().parents[3] / "preregistrations/v2_round5_data.json"
)
BOVA_INPUT_HASHES = (
    "bova11_close",
    "bova11_manifest",
    "bova11_data",
    "hedge_beta_manifest",
    "hedge_beta",
    "hedge_beta_valid",
    "hedge_beta_history_values",
    "hedge_beta_history_valid",
    "initial_hedge_reference_price",
)
RATE_INPUT_HASHES = (
    "annual_borrow_rate_by_name",
    "borrow_rate_imputed",
    "borrow_rate_placeholder",
)
BOVA_SOURCE_HASHES = ("bova11_manifest", "bova11_data", "hedge_beta_manifest")
RATE_SOURCE_HASHES = ("round5_lending_rates_manifest", "round5_lending_rates")
# These diagnostics depend on the realized book, not its forecast or population.
LEDGER_REPORT_PATHS = (
    "economics",
    "diagnostics.realized_beta",
    "diagnostics.realized_beta_bova11",
    "mask_coverage.stale_mark_name_days",
    "mask_coverage.unresolved_action_name_days",
    "mask_coverage.valuation_scenario_count",
    "mask_coverage.actual_risk_breach_dates",
)
RATE_REPORT_PATHS = (
    *RATE_INPUT_HASHES,
    "borrow_source_label",
    "diagnostics.lending_coverage",
)
BOOK_PATTERNS = {
    "screening": (r"aggregates/screening/([^/]+)/(F\d+)/evaluation.json", 87),
    "seed_audit": (r"(omit_(?:11|29|47))/([^/]+)/(F\d+)/evaluation.json", 252),
    "cpu_replay": (r"cpu_replay/(baselines|gbdt)/([^/]+)/(F\d+)/evaluation.json", 126),
}


def _rate_treatment(treatment: str) -> bool:
    return treatment in {"lending_only", "joint"}


def _bova_treatment(treatment: str) -> bool:
    return treatment in {"bova_only", "joint"}


def _allowed_inputs(treatment: str) -> tuple[str, ...]:
    return (
        *(BOVA_INPUT_HASHES if _bova_treatment(treatment) else ()),
        *(RATE_INPUT_HASHES if _rate_treatment(treatment) else ()),
    )


def protected_projection(report: dict, treatment: str) -> dict:
    """Explicit exemptions; every other report field must remain exactly equal."""
    result = {
        key: copy.deepcopy(value) for key, value in report.items() if key != "economics"
    }
    paths = (
        *LEDGER_REPORT_PATHS,
        *(RATE_REPORT_PATHS if _rate_treatment(treatment) else ()),
        *(f"input_hashes.{key}" for key in _allowed_inputs(treatment)),
        *(
            f"source_artifact_hashes.{key}"
            for key in (
                *(BOVA_SOURCE_HASHES if _bova_treatment(treatment) else ()),
                *(RATE_SOURCE_HASHES if _rate_treatment(treatment) else ()),
            )
        ),
    )
    for path in paths:
        parts = path.split(".")
        node = result
        for part in parts[:-1]:
            node = node.get(part, {})
        node.pop(parts[-1], None)
    return result


def _changed_hashes(before: dict, after: dict) -> list[str]:
    return sorted(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )


def _gate_failures(report: dict) -> dict:
    economics = report["economics"]
    failures = {
        row["scenario"]: ledger_gate_failures(
            row, headline=row["scenario"] == "borrow_balance"
        )
        for row in economics["summaries"]
    }
    failures["d5_only_diagnostic"] = ledger_gate_failures(
        economics["d5_only_diagnostic"]["summary"], headline=False
    )
    return {key: value for key, value in failures.items() if value}


def replay_report(
    original: dict, inputs: ev.EvaluationInputs, *, treatment: str
) -> tuple[dict, dict]:
    """Recompute accounting only after verifying the exact protected input hashes."""
    hashes = ev._input_hashes(inputs)
    changed = _changed_hashes(original["input_hashes"], hashes)
    forbidden = set(changed) - set(_allowed_inputs(treatment))
    if forbidden:
        raise ValueError(f"replay changed protected input fields: {sorted(forbidden)}")
    shortability = {
        name: ev._array_sha256(np.asarray(values, dtype=np.bool_))
        for name, values in sorted(inputs.shortable_by_borrow_source.items())
    }
    if shortability != original["shortable_by_borrow_source"]:
        raise ValueError("replay changed protected shortability masks")
    if (
        not _rate_treatment(treatment)
        and inputs.borrow_source_label != original["borrow_source_label"]
    ):
        raise ValueError("BOVA-only replay changed the lending source label")
    numerical = set(changed) - {
        "bova11_manifest",
        "bova11_data",
        "hedge_beta_manifest",
    }
    if (
        treatment == "bova_only"
        and numerical
        and original["window"]["name"] not in {"F4", "F5"}
    ):
        raise ValueError("BOVA repair changed a numerical input outside F4/F5")
    report = copy.deepcopy(original)
    if numerical:
        ev._validate(inputs)
        economics, headline, _ = ev._evaluate_economics(
            inputs,
            settle_terminal_residuals=original["economics"]["contract"].get(
                "terminal_residuals_settled", False
            ),
        )
        report["economics"] = economics
        report["diagnostics"]["realized_beta"] = ev._realized_beta_diagnostic(
            inputs, headline
        )
        report["diagnostics"]["realized_beta_bova11"] = ev._realized_beta_diagnostic(
            inputs, headline, against_bova11=True
        )
        report["mask_coverage"].update(
            stale_mark_name_days=int(headline.stale_mark_name_days.sum()),
            unresolved_action_name_days=int(headline.unresolved_action_name_days.sum()),
            valuation_scenario_count=int(headline.valuation_scenario_count.sum()),
            actual_risk_breach_dates=int(headline.actual_risk_breach.sum()),
        )
    # Provenance changes even in folds whose numerical accounting inputs are exact.
    if _bova_treatment(treatment):
        report["economics"]["contract"]["hedge_beta_manifest_sha256"] = (
            inputs.hedge_beta_manifest_sha256
        )
    if _rate_treatment(treatment):
        for key in RATE_INPUT_HASHES:
            report[key] = hashes[key]
        report["borrow_source_label"] = inputs.borrow_source_label
        report["diagnostics"]["lending_coverage"] = ev._lending_coverage(inputs)
    report["input_hashes"] = hashes
    report["source_artifact_hashes"] = dict(
        sorted(inputs.source_artifact_hashes.items())
    )
    if protected_projection(original, treatment) != protected_projection(
        report, treatment
    ):
        changed_fields = rr._changed_field_paths(
            protected_projection(original, treatment),
            protected_projection(report, treatment),
            "report",
        )
        raise ValueError(
            f"replay changed protected non-ledger fields: {sorted(changed_fields)}"
        )
    old_failures, new_failures = _gate_failures(original), _gate_failures(report)
    new_flags = {
        key: sorted(set(flags) - set(old_failures.get(key, ())))
        for key, flags in new_failures.items()
    }
    new_flags = {key: flags for key, flags in new_flags.items() if flags}
    if new_flags:
        raise RuntimeError(f"new registered book failures: {new_flags}")
    return report, {
        "changed_input_hashes": changed,
        "numerical_ledger_inputs_changed": sorted(numerical),
        "accounting_recomputed": bool(numerical),
        "accounting_reused_by_exact_input_identity": not numerical,
        "protected_non_ledger_fields_bit_identical": True,
        "historical_gate_failures": old_failures,
        "replay_gate_failures": new_failures,
    }


def _verify_file(path: Path, record: dict) -> None:
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"sealed artifact differs: {path}")


def summarize_replay(root: Path, treatment: str, output: Path) -> dict:
    """Pool matched daily economics by strategy, never across overlapping arms."""
    if output.exists():
        raise FileExistsError(output)
    design_path = root / "frozen_design.json"
    design = rr._read_json(design_path)
    groups: dict[str, list[dict]] = {}
    books, sources = [], []
    affected = set()
    for book in design["books"]:
        path = root / treatment / book["key"] / "comparison.json"
        comparison = rr._read_json(path)
        actual = sha256_file(path)
        recorded = path.with_suffix(".json.sha256").read_text().split()[0]
        if actual != recorded or comparison["frozen_design_sha256"] != sha256_file(
            design_path
        ):
            raise ValueError("replay comparison identity differs")
        if not comparison["protected_non_ledger_fields_bit_identical"]:
            raise ValueError("replay has a protected-field change")
        sources.append({"path": str(path), "sha256": actual})
        pair = comparison["headline_net_excess_bps"]
        before, after = (np.asarray(pair[k], np.float64) for k in ("before", "after"))
        matched = np.isfinite(before) & np.isfinite(after)
        if before.shape != after.shape:
            raise ValueError("replay daily economics axes differ")
        before, after = (
            np.where(matched, before, np.nan),
            np.where(matched, after, np.nan),
        )
        row = {
            "key": book["key"],
            "fold": book["fold"],
            "historical_status": book["historical_status"],
            "before_bps_per_day": float(np.nanmean(before)) if matched.any() else None,
            "after_bps_per_day": float(np.nanmean(after)) if matched.any() else None,
            "paired_delta_bps_per_day": float(np.nanmean(after - before))
            if matched.any()
            else None,
            "matched_days": int(matched.sum()),
            "accounting_recomputed": comparison["accounting_recomputed"],
            "protected_fields_exact": True,
        }
        books.append(row)
        if comparison["accounting_recomputed"]:
            affected.add(book["fold"])
        if book["historical_status"] == "accepted":
            groups.setdefault(book["key"].rsplit("/", 1)[0], []).append(
                {"fold": book["fold"], "before": before, "after": after}
            )
    pooled = {}
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["fold"][1:]))
        pooled[key] = {
            "folds": [row["fold"] for row in rows],
            **{
                label: rr._folded_bootstrap(
                    tuple(
                        row["after"] - row["before"]
                        if label == "paired_delta"
                        else row[label]
                        for row in rows
                    )
                )
                for label in ("before", "after", "paired_delta")
            },
        }
    result = {
        "schema": "BRAZIL_RV_ROUND5_ECONOMIC_READOUT_V1",
        "replay_root": str(root),
        "treatment": treatment,
        "frozen_design_sha256": sha256_file(design_path),
        "book_count": len(books),
        "all_protected_fields_exact": True,
        "accounting_recomputed_folds": sorted(
            affected, key=lambda value: int(value[1:])
        ),
        "historically_rejected_books_excluded_from_pools": [
            b["key"] for b in books if b["historical_status"] != "accepted"
        ],
        "pooling": "matched resolved daily observations; each strategy pooled separately; original folded block bootstrap; no independent resampling of paired outcomes",
        "pooled": pooled,
        "books": books,
        "comparisons": sources,
    }
    write_json_atomic(output, result)
    return result


def enumerate_books(registration: dict) -> tuple[list[dict], dict]:
    """Use sealed inventories rather than recursive directory discovery."""
    books, original_design = [], None
    for group, (pattern, expected_count) in BOOK_PATTERNS.items():
        binding = registration["round4_roots"][group]
        root = Path(binding["root"]).resolve(strict=True)
        inventory_path = root / "artifact_inventory.json"
        if sha256_file(inventory_path) != binding["inventory_sha256"]:
            raise ValueError(f"Round-4 inventory identity differs: {group}")
        files = {row["path"]: row for row in rr._read_json(inventory_path)["files"]}
        selected = [(relative, re.fullmatch(pattern, relative)) for relative in files]
        selected = [(relative, match) for relative, match in selected if match]
        if len(selected) != expected_count:
            raise ValueError(
                f"{group} has {len(selected)} sealed books, expected {expected_count}"
            )
        for relative, match in selected:
            source = Path(relative).parent
            parts = match.groups()
            key = (
                "/".join(("screening", *parts))
                if group == "screening"
                else "/".join(parts)
            )
            if group == "cpu_replay":
                key = "cpu/" + key
            records = {}
            for filename in (
                "evaluation.json",
                "scores.npy",
                "score_mask.npy",
                "score_manifest.json",
                "score_manifest.json.sha256",
                "accepted.json",
                "rejected.json",
            ):
                record = files.get((source / filename).as_posix())
                if record is not None:
                    records[filename] = {k: record[k] for k in ("bytes", "sha256")}
            if (
                not {
                    "evaluation.json",
                    "scores.npy",
                    "score_mask.npy",
                    "score_manifest.json",
                }
                <= records.keys()
            ):
                raise ValueError(f"sealed book is incomplete: {group}/{source}")
            books.append(
                {
                    "key": key,
                    "fold": parts[-1],
                    "source_root": str(root),
                    "relative_path": source.as_posix(),
                    "files": records,
                    "historical_status": "rejected"
                    if "rejected.json" in records
                    else "accepted",
                }
            )
        if group == "cpu_replay":
            _verify_file(root / "frozen_design.json", files["frozen_design.json"])
            original_design = rr._read_json(root / "frozen_design.json")
    if len({row["key"] for row in books}) != len(books):
        raise ValueError("duplicate final Round-4 replay book")
    return sorted(
        books, key=lambda row: (int(row["fold"][1:]), row["key"])
    ), original_design


def freeze(
    root: Path, sources_path: Path, registration_path: Path = REGISTRATION
) -> str:
    if root.exists():
        raise FileExistsError(root)
    registration = rr._read_json(registration_path)
    sources = rr._read_json(sources_path)
    if sources["registration_sha256"] != sha256_file(registration_path):
        raise ValueError("economic sources bind a different Round-5 registration")
    books, original_design = enumerate_books(registration)
    if (
        original_design["store"]["manifest_sha256"]
        != registration["base_store"]["manifest_sha256"]
    ):
        raise ValueError("Round-4 replay source differs from the registered base store")
    inputs = [
        Path(original_design["store"]["root"]),
        Path(original_design["bova11"]["root"]),
        Path(original_design["bova11"]["hedge_beta_root"]),
        Path(original_design["lending_archive"]["root"]),
        *(Path(row["source_root"]) for row in books),
    ]
    for name in ("bova11", "hedge_beta", "new_lending_rates"):
        record = sources.get(name)
        if record is None:
            continue
        source = Path(record["root"]).resolve(strict=True)
        inputs.append(source)
        if sha256_file(source / "manifest.json") != record["manifest_sha256"]:
            raise ValueError(f"repaired source manifest differs: {name}")
        if name == "new_lending_rates":
            proof = record["availability_proof"]
            proof_path = Path(proof["path"])
            if sha256_file(proof_path) != proof["sha256"]:
                raise ValueError("lending availability proof hash differs")
            evidence = rr._read_json(proof_path)
            if (
                evidence.get("source_manifest_sha256") != record["manifest_sha256"]
                or evidence.get("causality_passed") is not True
                or evidence.get("point_in_time") is not True
            ):
                raise ValueError(
                    "replacement rates lack an admitted availability proof"
                )
    if any(root.resolve().is_relative_to(source.resolve()) for source in inputs):
        raise ValueError("replay output must be outside immutable source roots")
    implementation = rr._git_identity()
    files = (Path(__file__), Path(ev.__file__), Path(rr.__file__))
    implementation["files"] = {str(path.resolve()): sha256_file(path) for path in files}
    design = {
        "schema": SCHEMA,
        "created_at_utc": rr._utc_now(),
        "implementation": implementation,
        "registration": {
            "path": str(registration_path.resolve()),
            "sha256": sha256_file(registration_path),
        },
        "economic_sources": sources,
        "economic_sources_sha256": sha256_file(sources_path),
        "original_design": original_design,
        "books": books,
        "treatments": ["bova_only", "lending_only", "joint"]
        if sources.get("new_lending_rates")
        else ["bova_only"],
        "exemptions": {
            "bova_input_hashes": BOVA_INPUT_HASHES,
            "rate_input_hashes": RATE_INPUT_HASHES,
            "ledger_report_paths": LEDGER_REPORT_PATHS,
            "rate_report_paths": RATE_REPORT_PATHS,
            "bova_source_hashes": BOVA_SOURCE_HASHES,
            "rate_source_hashes": RATE_SOURCE_HASHES,
        },
        "unchanged_input_policy": "reuse exact accounting; always verify the sealed book and all input hashes",
        "official_validation_accessed": False,
        "test_accessed": False,
        "neural_fits": False,
    }
    root.mkdir(parents=True, exist_ok=False)
    return write_json_atomic(root / "frozen_design.json", design)


class ReplayInputs:
    """One verified source context; at most one fold of expensive store views."""

    def __init__(self, design: dict):
        self.context = rr._open_ledger_replay(design["original_design"])
        self.templates = {}
        self.fold = None
        sources = design["economic_sources"]
        dates = self.context.store.dates.astype(object).tolist()
        if dates[-1] > DEVELOPMENT_END:
            raise PermissionError("Round-5 replay refuses held-out consumer rows")
        bova = sources["bova11"]
        self.bova = load_bova11_series(
            Path(bova["root"]),
            expected_manifest_sha256=bova["manifest_sha256"],
            canonical_dates=dates,
        )
        beta = sources["hedge_beta"]
        self.beta = load_hedge_beta_sidecar(
            Path(beta["root"]),
            expected_manifest_sha256=beta["manifest_sha256"],
            expected_store_manifest_sha256=design["original_design"]["store"][
                "manifest_sha256"
            ],
            expected_bova11_manifest_sha256=self.bova.manifest_sha256,
            canonical_dates=dates,
            isins=self.context.store.isins,
        )
        rates = sources.get("new_lending_rates")
        self.rates = (
            None
            if rates is None
            else load_lending_borrow_panels(
                Path(rates["root"]),
                expected_manifest_sha256=rates["manifest_sha256"],
                canonical_dates=dates,
                canonical_isins=self.context.store.isins,
            )
        )

    def original(self, book: dict) -> tuple[dict, ev.EvaluationInputs]:
        source = Path(book["source_root"]) / book["relative_path"]
        for filename, record in book["files"].items():
            _verify_file(source / filename, record)
        report = rr._evaluation_from_path(source / "evaluation.json")
        if report.get("transfer_chronology_clean") is not True:
            raise PermissionError("replay source lacks clean transfer chronology")
        scores, mask = rr._score_artifact(source, require_clean_transfer=True)
        metadata = rr._read_json(source / "score_manifest.json")["metadata"]
        indices = self.context.evaluation[book["fold"]]
        if (
            metadata.get("evaluation_date_indices") != indices.tolist()
            or metadata.get("fold") != book["fold"]
        ):
            raise ValueError("sealed scores differ from their registered fold axis")
        policy_record = report["economics"].get("execution_policy")
        policy = (
            None
            if policy_record is None
            else ExecutionPolicy(
                **{
                    key: tuple(policy_record[key])
                    if key == "horizons"
                    else policy_record[key]
                    for key in (
                        "theta",
                        "horizons",
                        "inverse_volatility",
                        "buffer_per_quintile",
                    )
                }
            )
        )
        if self.fold != book["fold"]:
            self.templates.clear()
            self.fold = book["fold"]
        key = json.dumps(policy_record, sort_keys=True)
        source_hashes = report["source_artifact_hashes"]
        if key not in self.templates:
            self.templates[key] = rr._evaluation_inputs(
                self.context.store,
                indices,
                scores,
                mask,
                self.context.cdi,
                self.context.bova11.close_by_session,
                self.context.bova11_binding,
                self.context.lending_borrow,
                source_hashes,
                transfer_chronology_clean=True,
                execution_policy=policy,
            )
        inputs = replace(
            self.templates[key],
            scores=scores,
            score_mask=mask,
            source_artifact_hashes=source_hashes,
        )
        if ev._input_hashes(inputs) != report["input_hashes"]:
            raise ValueError(
                f"original replay inputs differ from sealed inputs: {book['key']}"
            )
        return report, inputs

    def repaired(
        self, original: ev.EvaluationInputs, treatment: str
    ) -> ev.EvaluationInputs:
        indices = np.asarray(original.session_indices, dtype=np.int64)
        changes = {}
        provenance = dict(original.source_artifact_hashes)
        if _bova_treatment(treatment):
            start = int(indices[0])
            history_start = max(0, start - HEDGE_BETA_MAX_AGE)
            changes.update(
                bova11_close=self.bova.close_by_session[indices],
                bova11_manifest_sha256=self.bova.manifest_sha256,
                bova11_data_sha256=self.bova.data_sha256,
                hedge_beta=self.beta.values[indices],
                hedge_beta_valid=self.beta.valid[indices],
                hedge_beta_history=(
                    self.beta.values[history_start:start],
                    self.beta.valid[history_start:start],
                ),
                hedge_beta_manifest_sha256=self.beta.manifest_sha256,
                initial_hedge_reference_price=float(
                    self.bova.close_by_session[start - 1]
                )
                if start
                else np.nan,
            )
            provenance.update(
                bova11_manifest=self.bova.manifest_sha256,
                bova11_data=self.bova.data_sha256,
                hedge_beta_manifest=self.beta.manifest_sha256,
            )
        if _rate_treatment(treatment):
            if self.rates is None:
                raise ValueError("no admitted replacement lending rates")
            changes.update(
                annual_borrow_rate_by_name=self.rates.annual_taker_rate[indices],
                borrow_rate_imputed=self.rates.rate_imputed[indices],
                borrow_rate_placeholder=self.rates.rate_placeholder[indices],
                borrow_source_label="sealed_shortability_with_round5_observed_rates",
            )
            provenance.update(
                round5_lending_rates_manifest=self.rates.manifest_sha256,
                round5_lending_rates=self.rates.rate_sha256,
            )
        return replace(original, **changes, source_artifact_hashes=provenance)

    def close(self) -> None:
        self.templates.clear()
        self.context.store.close()


def _headline_values(report: dict) -> list[float | None]:
    return [
        row["net_excess_all_cash_bps"]
        for row in report["economics"]["daily_table"]
        if row["scenario"] == "borrow_balance"
    ]


def _comparison(original: dict, report: dict, record: dict) -> dict:
    before, after = _headline_values(original), _headline_values(report)
    delta = [
        None if a is None or b is None else b - a
        for a, b in zip(before, after, strict=True)
    ]
    return {
        **record,
        "before_headline": original["economics"]["headline"],
        "after_headline": report["economics"]["headline"],
        "headline_net_excess_bps": {"before": before, "after": after, "delta": delta},
        "mean_delta_bps": float(
            np.mean([value for value in delta if value is not None])
        )
        if any(value is not None for value in delta)
        else None,
    }


def _completed(
    destination: Path, design_sha: str, book: dict, treatment: str
) -> dict | None:
    path = destination / "comparison.json"
    if not path.exists():
        return None
    if path.with_suffix(".json.sha256").read_text().split()[0] != sha256_file(path):
        raise ValueError("completed replay comparison hash differs")
    record = rr._read_json(path)
    if (
        record["frozen_design_sha256"] != design_sha
        or record["source_evaluation_sha256"]
        != book["files"]["evaluation.json"]["sha256"]
        or record["treatment"] != treatment
    ):
        raise ValueError("replay resume contract differs")
    for filename, expected in record["output_hashes"].items():
        if sha256_file(destination / filename) != expected:
            raise ValueError(f"completed replay file differs: {destination / filename}")
    return record


def run(
    root: Path,
    *,
    limit: int | None = None,
    book_key: str | None = None,
    treatment: str | None = None,
) -> dict:
    design_path = root / "frozen_design.json"
    design_sha = sha256_file(design_path)
    if design_path.with_suffix(".json.sha256").read_text().split()[0] != design_sha:
        raise ValueError("frozen replay design hash differs")
    design = rr._read_json(design_path)
    for filename, expected in design["implementation"]["files"].items():
        if sha256_file(Path(filename)) != expected:
            raise ValueError(
                f"replay implementation differs from its frozen source: {filename}"
            )
    books = [
        row for row in design["books"] if book_key is None or row["key"] == book_key
    ]
    treatments = design["treatments"] if treatment is None else [treatment]
    if not books or not set(treatments) <= set(design["treatments"]):
        raise ValueError("requested replay book or treatment is not registered")
    context = None
    started = time.perf_counter()
    newly_completed = 0
    try:
        for book in books:
            original = inputs = None
            for selected in treatments:
                destination = root / selected / book["key"]
                completed = _completed(destination, design_sha, book, selected)
                if completed is not None:
                    continue
                if limit is not None and newly_completed >= limit:
                    break
                write_json_atomic(
                    root / "progress.json",
                    {
                        "status": "running",
                        "active_book": book["key"],
                        "treatment": selected,
                        "newly_completed": newly_completed,
                        "elapsed_seconds": time.perf_counter() - started,
                        "updated_at_utc": rr._utc_now(),
                    },
                )
                cell_start = time.perf_counter()
                if context is None:
                    context = ReplayInputs(design)
                context_seconds = time.perf_counter() - cell_start
                verification_start = time.perf_counter()
                if original is None:
                    original, inputs = context.original(book)
                verification_seconds = time.perf_counter() - verification_start
                accounting_start = time.perf_counter()
                repaired = context.repaired(inputs, selected)
                report, evidence = replay_report(original, repaired, treatment=selected)
                record = _comparison(
                    original,
                    report,
                    {
                        **evidence,
                        "key": book["key"],
                        "fold": book["fold"],
                        "treatment": selected,
                        "historical_status": book["historical_status"],
                        "historically_rejected_remains_excluded": book[
                            "historical_status"
                        ]
                        == "rejected",
                        "frozen_design_sha256": design_sha,
                        "source_evaluation_sha256": book["files"]["evaluation.json"][
                            "sha256"
                        ],
                        "elapsed_seconds": time.perf_counter() - cell_start,
                        "timings": {
                            "context_open_seconds": context_seconds,
                            "source_verification_seconds": verification_seconds,
                            "repair_and_accounting_seconds": time.perf_counter()
                            - accounting_start,
                        },
                        "process_peak_rss_bytes": peak_rss_bytes(),
                    },
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = Path(
                    tempfile.mkdtemp(
                        prefix=f".{destination.name}.", dir=destination.parent
                    )
                )
                try:
                    source = Path(book["source_root"]) / book["relative_path"]
                    for filename in (
                        "scores.npy",
                        "score_mask.npy",
                        "score_manifest.json",
                        "score_manifest.json.sha256",
                    ):
                        if filename in book["files"]:
                            shutil.copyfile(source / filename, temporary / filename)
                    write_json_atomic(temporary / "evaluation.json", report)
                    record["output_hashes"] = {
                        path.name: sha256_file(path)
                        for path in temporary.iterdir()
                        if path.is_file()
                    }
                    write_json_atomic(temporary / "comparison.json", record)
                    os.replace(temporary, destination)
                except BaseException:
                    if temporary.exists() and temporary.parent == destination.parent:
                        shutil.rmtree(temporary)
                    raise
                newly_completed += 1
                message = {
                    "book": book["key"],
                    "treatment": selected,
                    "seconds": record["elapsed_seconds"],
                    "accounting_recomputed": record["accounting_recomputed"],
                    "mean_delta_bps": record["mean_delta_bps"],
                }
                with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(message) + "\n")
                print(json.dumps(message), flush=True)
            if limit is not None and newly_completed >= limit:
                break
        total = len(design["books"]) * len(design["treatments"])
        completed_count = sum(
            (root / selected / row["key"] / "comparison.json").exists()
            for row in design["books"]
            for selected in design["treatments"]
        )
        result = {
            "schema": SCHEMA,
            "status": "completed" if completed_count == total else "checkpoint",
            "frozen_design_sha256": design_sha,
            "completed_books": completed_count,
            "expected_books": total,
            "newly_completed_books": newly_completed,
            "invocation_elapsed_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes": peak_rss_bytes(),
            "protected_non_ledger_fields_bit_identical": True,
            "official_validation_accessed": False,
            "test_accessed": False,
            "paid_compute_launched": False,
            "updated_at_utc": rr._utc_now(),
        }
        write_json_atomic(root / "progress.json", result)
        if completed_count == total:
            result["books"] = [
                {
                    key: record[key]
                    for key in (
                        "key",
                        "fold",
                        "treatment",
                        "historical_status",
                        "accounting_recomputed",
                        "mean_delta_bps",
                        "elapsed_seconds",
                        "source_evaluation_sha256",
                        "output_hashes",
                    )
                }
                for row in design["books"]
                for selected in design["treatments"]
                if (
                    record := _completed(
                        root / selected / row["key"], design_sha, row, selected
                    )
                )
            ]
            write_json_atomic(root / "result.json", result)
        return result
    except BaseException as error:
        write_json_atomic(
            root / "stop.json",
            {
                "error": str(error),
                "frozen_design_sha256": design_sha,
                "at_utc": rr._utc_now(),
            },
        )
        raise
    finally:
        if context is not None:
            context.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--book")
    parser.add_argument("--treatment", choices=("bova_only", "lending_only", "joint"))
    args = parser.parse_args()
    if args.action == "freeze":
        if args.sources is None:
            parser.error("freeze requires --sources")
        print(freeze(args.root, args.sources, args.registration))
    else:
        print(
            json.dumps(
                run(
                    args.root,
                    limit=args.limit,
                    book_key=args.book,
                    treatment=args.treatment,
                )
            )
        )


if __name__ == "__main__":
    main()
