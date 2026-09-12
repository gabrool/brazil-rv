"""One immutable Round-7 store: repaired outcomes and native fundamentals."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .build_store import _eventual_survival_groups
from .corporate_actions import (
    AlignedActionTerms,
    detect_distribution_changes,
    infer_cotahist_action_terms,
    verified_action_terms_to_table,
)
from .feature_spec import FeatureSpec, feature_schema_sha256, feature_specs
from .data_foundation import continuation_identity_axis
from .research_rounds import _git_identity
from .round5_cvm import rad_rows
from .round5_derived import bind
from .round5_store import align_family
from .round7_data import (
    NATIVE_FIELDS,
    corroborate_u2,
    registered_sources,
    unit_action_evidence,
)
from .splits import development_evaluation_windows
from .store import StoreStaging, close_memmap, peak_rss_bytes
from .targets import build_economic_multi_day_targets

TARGET_FIELDS = {
    "target_primary": "primary",
    "target_valid": "primary_valid",
    "target_normalized_residual": "normalized_residual",
    "target_normalized_cross_section_valid": "normalized_cross_section_valid",
    "target_shareholder_midrank": "shareholder_midrank",
    "target_shareholder_valid": "shareholder_valid",
    "target_shareholder_simple_return": "shareholder_simple_return",
    "target_terminal_wealth": "terminal_wealth",
    "target_terminal_loss": "terminal_loss",
    "target_price_midrank": "price_midrank",
    "target_price_valid": "price_valid",
    "target_price_simple_return": "price_simple_return",
}
QUOTE_FIELDS = {
    "raw_open": "open_brl",
    "raw_high": "high_brl",
    "raw_low": "low_brl",
    "raw_close": "close_brl",
    "volume_brl": "volume_brl",
    "quantity": "quantity",
    "trade_count": "trades",
    "distribution_number": "distribution_number",
}


def different(a, b):
    return ~(np.equal(a, b) | (np.isnan(a) & np.isnan(b)))


def build(audit_root: Path, output: Path):
    code = _git_identity()
    source, original, cvm_manifest, accepted = registered_sources()
    if (
        output.resolve().is_relative_to(source.resolve())
        or "raw" in output.resolve().parts
    ):
        raise ValueError("repair output must be outside immutable source directories")
    audit_manifest = audit_root / "manifest.json"
    audit = json.loads(audit_manifest.read_text(encoding="utf-8"))
    if audit["base_store"] != accepted["store"]:
        raise ValueError("source audit binds a different parent")
    clock_path = audit_root / "foreign_clock/manifest.json"
    clock = json.loads(clock_path.read_text(encoding="utf-8"))
    if clock["earlier_admission_automatically_authorized"]:
        raise ValueError(
            "an earlier foreign-flow clock requires an explicit source decision"
        )
    for key in ("u2_events", "native_family"):
        if sha256_file(Path(audit[key]["path"])) != audit[key]["sha256"]:
            raise ValueError(f"audited source changed: {key}")
    if (
        sha256_file(audit_root / "continued_quotes.parquet")
        != audit["continued_prints"]["data"]["sha256"]
    ):
        raise ValueError("audited continuation quotes changed")
    dates = np.load(source / "date_index.npy", allow_pickle=False)
    isins = tuple(np.load(source / "isin_index.npy", allow_pickle=False).tolist())
    date_lookup, name_lookup = (
        {d: i for i, d in enumerate(dates.astype(object))},
        {n: i for i, n in enumerate(isins)},
    )
    mapped = {}

    def old(name):
        if name not in mapped:
            mapped[name] = np.load(
                source / original["arrays"][name]["path"],
                mmap_mode="r",
                allow_pickle=False,
            )
        return mapped[name]

    try:
        check_rows = np.unique(np.linspace(0, len(dates) - 1, 24, dtype=np.int64))
        baseline = build_economic_multi_day_targets(
            old("raw_close"),
            old("observed"),
            old("active"),
            old("target_scale_sigma"),
            AlignedActionTerms(
                old("action_shares_per_prior_share"),
                old("action_cash_per_prior_share"),
                old("action_session_resolved"),
                old("action_has_action"),
                old("action_successor_index"),
            ),
            source_rows=check_rows,
        )
        for name, attr in TARGET_FIELDS.items():
            if different(getattr(baseline, attr), old(name)[check_rows]).any():
                raise ValueError(
                    f"current target builder does not reproduce parent: {name}"
                )
        del baseline
        updates = {name: np.array(old(name)) for name in QUOTE_FIELDS}
        for name in (
            "observed",
            "trade_observed",
            "activity_valid",
            "action_shares_per_prior_share",
            "action_cash_per_prior_share",
            "action_has_action",
            "action_session_resolved",
            "action_payment_session",
            "inferred_action_u1_mask",
            "inferred_action_c1_mask",
            "inferred_action_u2_mask",
            "inferred_action_large_move_no_action_mask",
        ):
            updates[name] = np.array(old(name))
        quotes = pl.read_parquet(audit_root / "continued_quotes.parquet")
        recovered = np.zeros(old("observed").shape, bool)
        for row in quotes.iter_rows(named=True):
            i, n = date_lookup[row["trade_date"]], name_lookup[row["isin"]]
            if old("observed")[i, n]:
                raise ValueError(
                    "continuation recovery would overwrite an existing quote"
                )
            recovered[i, n] = True
            for array, field in QUOTE_FIELDS.items():
                updates[array][i, n] = row[field]
            updates["observed"][i, n] = True
            updates["trade_observed"][i, n] = row["trades"] > 0
            updates["activity_valid"][i, n] = True
            updates["action_session_resolved"][i, n] = True
        updates["continuation_quote_mask"] = recovered
        # A recovered end-of-day print can value/exit a claim, but must not create
        # a new opening fill. Do not use its later BDI code to alter orders at 15:45.
        updates["entry_fill_allowed"] = ~recovered
        updates["distribution_change_mask"] = detect_distribution_changes(
            updates["distribution_number"], updates["observed"]
        )

        def table(name):
            return pl.read_parquet(source / original["tables"][name]["path"])

        original_terms = table("corporate_actions_verified_terms")
        records = json.loads(
            (audit_root / "u2_events.json").read_text(encoding="utf-8")
        )
        rejected_keys = set()
        for row in records:
            if row["classification"] != "large_move_no_action":
                continue
            i, n = row["date_index"], row["name_index"]
            rejected_keys.add((row["date"], row["isin"]))
            updates["action_shares_per_prior_share"][i, n] = 1.0
            updates["action_has_action"][i, n] = False
            updates["action_payment_session"][i, n] = -1
            updates["inferred_action_u2_mask"][i, n] = False
            updates["inferred_action_large_move_no_action_mask"][i, n] = True
        retained = original_terms.filter(
            pl.Series(
                [
                    (str(r["effective_date"]), r["isin"]) not in rejected_keys
                    for r in original_terms.iter_rows(named=True)
                ]
            )
        )
        cvm = json.loads(cvm_manifest.read_text(encoding="utf-8"))
        provider = table("corporate_actions_provider_observations")
        identity = pl.read_parquet(cvm_manifest.parent / "identity.parquet")
        filings = rad_rows(Path(cvm["source_root"]))
        quote_evidence = unit_action_evidence(
            quotes.select(
                pl.col("trade_date").alias("effective_date"), "isin"
            ).to_dicts(),
            provider,
            identity,
            filings,
            dates,
            isins,
            updates["distribution_change_mask"],
        )
        corroborated = np.zeros(recovered.shape, bool)
        for row in quote_evidence:
            corroborated[row["date_index"], row["name_index"]] = bool(
                row["corroboration"]
            )
        # Fixed corrected factors keep rejected old crashes out of the running
        # unit history. Newly recovered prints infer only corroborated U2 terms.
        fixed_factors = np.where(
            recovered, np.nan, updates["action_shares_per_prior_share"]
        )
        inferred = infer_cotahist_action_terms(
            dates,
            isins,
            updates["raw_close"],
            updates["quantity"],
            updates["trade_count"],
            updates["distribution_number"],
            updates["observed"],
            old("active"),
            fixed_share_factors=fixed_factors,
            u2_corroborated=corroborated,
        )
        new_terms = inferred.terms
        new_table = verified_action_terms_to_table(new_terms)
        extra_u2 = corroborate_u2(
            new_table,
            provider,
            identity,
            filings,
            dates,
            isins,
            updates["distribution_change_mask"],
        )
        kept_new = []
        updates["inferred_action_large_move_no_action_mask"][recovered] = (
            inferred.large_move_no_action[recovered]
        )
        for term in new_terms:
            i, n = date_lookup[term.effective_date], name_lookup[term.isin]
            kept_new.append(term)
            updates["action_shares_per_prior_share"][i, n] = term.shares_per_prior_share
            updates["action_cash_per_prior_share"][i, n] = term.cash_per_prior_share
            updates["action_has_action"][i, n] = True
            updates["action_payment_session"][i, n] = (
                i if term.cash_per_prior_share else -1
            )
            updates[f"inferred_action_{term.evidence[:2].lower()}_mask"][i, n] = True
        final_terms = pl.concat(
            [retained, verified_action_terms_to_table(kept_new)]
        ).sort("isin", "effective_date", "sequence")
        original_c1 = original_terms.filter(
            pl.col("evidence").str.starts_with("C1:")
        ).sort("isin", "effective_date")
        if (
            not final_terms.join(
                original_c1.select("isin", "effective_date"),
                on=["isin", "effective_date"],
                how="inner",
            )
            .sort("isin", "effective_date")
            .equals(original_c1)
        ):
            raise ValueError("repair changed an existing C1 cash term")
        actions = AlignedActionTerms(
            updates["action_shares_per_prior_share"],
            updates["action_cash_per_prior_share"],
            updates["action_session_resolved"],
            updates["action_has_action"],
            old("action_successor_index"),
        )
        identities = continuation_identity_axis(isins, table("isin_succession_links"))
        survival = _eventual_survival_groups(dates, updates["observed"], identities)
        updates["audit_eventual_survives_to_final_year"] = np.broadcast_to(
            survival["survives_to_final_year"][None, :], recovered.shape
        )
        print(
            json.dumps(
                {
                    "phase": "rebuild_targets",
                    "recovered_quotes": int(recovered.sum()),
                    "new_action_terms": len(kept_new),
                }
            ),
            flush=True,
        )
        targets = build_economic_multi_day_targets(
            updates["raw_close"],
            updates["observed"],
            old("active"),
            old("target_scale_sigma"),
            actions,
        )
        for array, attr in TARGET_FIELDS.items():
            updates[array] = getattr(targets, attr)
        # Retrospective action corroboration must never revise historical model
        # inputs, including the old decision-causal shareholder-wealth chain.
        # The repaired ledger prices claims from raw quotes and q/d directly.
        native = pl.read_parquet(audit_root / "fundamentals_native.parquet")
        values, valid, ages = align_family(
            native, dates.astype(object).tolist(), isins, NATIVE_FIELDS
        )
        valid &= np.asarray(old("active"))[..., None]
        values = np.where(valid, values, 0.0).astype(np.float32)
        ages = np.where(valid, ages, -1.0).astype(np.float32)
        for suffix, data in (
            ("values", values),
            ("valid", valid),
            ("age_sessions", ages),
        ):
            updates["sidecar_fundamentals_native_" + suffix] = data
        affected_events = (
            recovered
            | different(
                updates["action_shares_per_prior_share"],
                old("action_shares_per_prior_share"),
            )
            | different(
                updates["action_cash_per_prior_share"],
                old("action_cash_per_prior_share"),
            )
            | different(
                updates["action_session_resolved"], old("action_session_resolved")
            )
        )
        expected_dates = np.zeros((len(dates), 5), bool)
        for h_index, h in enumerate((1, 2, 3, 5, 10)):
            for shift in range(h + 1):
                expected_dates[: len(dates) - shift, h_index] |= affected_events[
                    shift:
                ].any(axis=1)
        difference_counts, by_fold = {}, {}
        for name, value in updates.items():
            if name not in original["arrays"]:
                continue
            changed = different(value, old(name))
            difference_counts[name] = int(changed.sum())
            if name in TARGET_FIELDS:
                date_head = changed.any(axis=1) if changed.ndim == 3 else changed
                if (date_head & ~expected_dates).any():
                    raise ValueError(
                        f"target changes extend beyond repaired outcome windows: {name}"
                    )
        for fold, (start, end) in development_evaluation_windows().items():
            selection = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
            by_fold[fold] = {
                name: int(
                    different(updates[name][selection], old(name)[selection]).sum()
                )
                for name in TARGET_FIELDS
            }
        names = copy.deepcopy(original["feature_names"])
        names["sidecar_fundamentals_native"] = list(NATIVE_FIELDS)
        metadata = copy.deepcopy(original["metadata"])
        specs = [FeatureSpec(**s) for s in metadata["feature_schema"]["specifications"]]
        specs.extend(feature_specs("sidecar_fundamentals_native", NATIVE_FIELDS))
        order = [f for f in ("slow", "intraday", "native_fast") if f in names] + sorted(
            f for f in names if f.startswith("sidecar_")
        )
        specs = tuple(s for family in order for s in specs if s.family == family)
        metadata["feature_schema"] = {
            "minimum_rank_names": 20,
            "specifications": [asdict(s) for s in specs],
            "sha256": feature_schema_sha256(specs),
        }
        metadata["sidecar_capabilities"]["fundamentals_native"] = {
            "enabled": list(NATIVE_FIELDS),
            "source_missing": [],
        }
        report = {
            "schema": "BRAZIL_RV_ROUND7_STORE_REPAIR_V1",
            "code": code,
            "base_store": accepted["store"],
            "source_audit": bind(audit_manifest),
            "foreign_clock": bind(clock_path),
            "foreign_flow_arrays_unchanged": True,
            "changed_cells": difference_counts,
            "target_changes_by_fold": by_fold,
            "additional_u2": extra_u2,
            "additional_action_terms": len(kept_new),
            "existing_C1_terms_exact": True,
            "parent_target_reproduction_dates": len(check_rows),
            "survival_flag_reconstructed": True,
            "new_entry_suppressed_active_quote_rows": int(
                (recovered & old("active")).sum()
            ),
            "protected_features": "All old slow, sidecar, risk, common-state and decision-causal wealth arrays remain byte-identical. Outcomes and accounting use repaired q/d and recovered raw quotes. Retrospective corroboration is not a feature.",
            "audit_table_scope": "Inherited tables describe the old store; round7 repair tables and this metadata describe changed observations, actions and targets.",
            "forward_capture": False,
            "heldout_access": False,
        }
        metadata["round7_repair"] = report
        tables = {name: source / r["path"] for name, r in original["tables"].items()}
        tables.update(
            corporate_actions_verified_terms=final_terms,
            round7_continued_quotes=quotes,
            round7_continued_quote_summary=audit_root
            / "continued_quote_summary.parquet",
            round7_target_changes=pl.DataFrame(
                [
                    {"fold": f, "array": n, "changed_cells": v}
                    for f, counts in by_fold.items()
                    for n, v in counts.items()
                ]
            ),
        )
        with StoreStaging(output, dates=dates, isins=isins) as staging:
            for name, record in original["arrays"].items():
                if name not in updates or difference_counts.get(name) == 0:
                    staging.copy_array(name, source / record["path"])
                else:
                    staging.write_array(name, updates[name], dtype=old(name).dtype)
            for name, value in updates.items():
                if name not in original["arrays"]:
                    staging.write_array(name, value)
            staging.seal(
                feature_names=names,
                metadata=metadata,
                sources=[
                    *original["sources"],
                    bind(audit_manifest),
                    bind(clock_path),
                    bind(cvm_manifest),
                ],
                tables=tables,
                maximum_peak_rss_bytes=8 * 1024**3,
            )
        sealed = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for name, record in original["arrays"].items():
            if (
                name not in updates
                and sealed["arrays"][name]["sha256"] != record["sha256"]
            ):
                raise ValueError(f"protected array differs: {name}")
        report.update(
            store={
                "root": str(output),
                "manifest_sha256": sha256_file(output / "manifest.json"),
            },
            peak_rss_bytes=peak_rss_bytes(),
            protected_arrays_exact=True,
            status="built_pending_economic_and_source_acceptance",
        )
        write_json_atomic(audit_root / "store_repair.json", report)
        return report
    finally:
        for array in mapped.values():
            close_memmap(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.audit_root, args.output)
    print(json.dumps({k: result[k] for k in ("status", "store", "peak_rss_bytes")}))


if __name__ == "__main__":
    main()
