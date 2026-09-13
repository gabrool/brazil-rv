"""Rebuild model inputs from bound sources without changing the research book."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import hashlib
import inspect
import pickle
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .build_store import _route_decision_known_continuations
from .corporate_actions import (
    align_decision_known_action_terms,
    build_shareholder_wealth_ohlc_into,
    infer_cotahist_action_terms,
    verified_action_terms_from_table,
)
from .data_foundation import panel_from_daily, prepare_cash_equities
from .decision_clock import load_session_schedule, next_session_decision_cutoffs
from .contract import DEVELOPMENT_END, FEATURE_AGE_CONTRACT
from .feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    feature_specs,
    transform_feature_panel_into,
)
from .features import _ambiguous_interval_clear, _rolling_moments, exact_log_return
from .round5_cvm import (
    fundamental_features,
    load_financial_documents,
    rad_rows,
    valuation_market,
)
from .round5_store import align_family
from .research_rounds import _git_identity
from .sidecars import derive_known_archive_features, materialize_known_archive
from .store import StoreStaging, close_memmap
from .universe import build_daily_universe

PROJECT = Path(__file__).resolve().parents[4]


def bound_json(record):
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"source identity differs: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def binding(path):
    return {"path": str(path), "sha256": sha256_file(path)}


def parent_store():
    accepted = json.loads(
        (PROJECT / "docs/v2_round7_inputs.json").read_text(encoding="utf-8")
    )["store"]
    root = Path(accepted["root"])
    manifest = bound_json(
        {"path": str(root / "manifest.json"), "sha256": accepted["manifest_sha256"]}
    )
    if manifest["axes"]["date_end"] > DEVELOPMENT_END.isoformat():
        raise PermissionError("data repair is restricted to development history")
    return root, manifest, accepted


def build_financial_family(repair_root: Path, *, reuse_extracted_documents=False):
    source_store, _, parent = parent_store()
    pointer = json.loads(
        (PROJECT / "docs/v2_round5_cvm_final_acceptance.json").read_text(
            encoding="utf-8"
        )
    )["family_manifest"]
    original = bound_json(pointer)
    family = Path(pointer["path"]).parent
    root = Path(original["source_root"])
    identity = pl.read_parquet(family / "identity.parquet")
    cache_path = repair_root / "financial_documents.pkl"
    cache_manifest = repair_root / "financial_documents.json"
    cache_identity = {
        "source": pointer,
        "implementation": [
            binding(Path(__file__).with_name(n))
            for n in ("round5_cvm.py", "round5_cvm_xml.py", "round5_cvm_capital.py")
        ],
        "repairs": [
            binding(repair_root / n)
            for n in (
                "account_unit_dispositions.json",
                "capital_source_dispositions.json",
            )
        ],
    }
    previous = (
        json.loads(cache_manifest.read_text(encoding="utf-8"))
        if cache_manifest.exists()
        else {}
    )
    if reuse_extracted_documents:
        # Explicit upstream artifact reuse after a downstream feature-only edit.
        # It cannot substitute documents from different sources or dispositions.
        for key in ("source", "repairs"):
            if previous.get("identity", {}).get(key) != cache_identity[key]:
                raise ValueError(
                    "extracted documents do not bind current sources/repairs"
                )
        if previous["cache"] != binding(cache_path):
            raise ValueError("extracted document artifact changed")
    if reuse_extracted_documents or (
        previous.get("identity") == cache_identity
        and previous["cache"] == binding(cache_path)
    ):
        with cache_path.open("rb") as handle:
            documents, source_evidence = pickle.load(handle)
    else:
        documents, source_evidence = load_financial_documents(
            root, set(identity["cnpj"]), repair_root=repair_root
        )
        with cache_path.open("wb") as handle:
            pickle.dump((documents, source_evidence), handle)
        write_json_atomic(
            cache_manifest, {"identity": cache_identity, "cache": binding(cache_path)}
        )
    generation = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode()
    ).hexdigest()[:12]
    output = repair_root / ("financial_family_" + generation)
    output.mkdir(exist_ok=False)
    dates = np.load(source_store / "date_index.npy").astype(object).tolist()
    capital_changes = json.loads(
        (family / "capital_change_observations.json").read_text(encoding="utf-8")
    )
    for event in capital_changes:
        event["effective"] = date.fromisoformat(event["effective"])
    print(
        json.dumps({"stage": "financial_features", "documents": len(documents)}),
        flush=True,
    )
    frame, audit = fundamental_features(
        documents,
        rad_rows(root),
        identity,
        dates,
        valuation_market(source_store),
        capital_changes,
    )
    frame.write_parquet(output / "fundamentals.parquet")
    manifest = {
        "parent": parent,
        "original_family": pointer,
        "source_identity": binding(family / "identity.parquet"),
        "capital_changes": binding(family / "capital_change_observations.json"),
        "consumed_sources": source_evidence,
        "extraction": bound_json(binding(cache_manifest)),
        "feature_implementation": cache_identity["implementation"],
        "audit": audit,
        "data": binding(output / "fundamentals.parquet"),
        "age_rule": "newest contributing public receipt; oldest dependency and economic reference ages retained separately",
        "fallback_rule": "latest coherent calculation per field from versions public by that decision; no stale cutoff",
        "forward_capture": False,
        "heldout_access": False,
    }
    write_json_atomic(output / "manifest.json", manifest)
    write_json_atomic(
        repair_root / "financial_family.json", binding(output / "manifest.json")
    )
    return manifest


def source_families(manifest):
    """Walk bound store ancestry, taking the newest source for each family."""
    result = {}
    current = manifest
    while "round5_extension" in current["metadata"]:
        extension = current["metadata"]["round5_extension"]
        for family in extension["families"]:
            result.setdefault(family["family"], family)
        parent = extension["base_store"]
        current = bound_json(
            {
                "path": str(Path(parent["root"]) / "manifest.json"),
                "sha256": parent["manifest_sha256"],
            }
        )
    return result


def slow_source_fields(source, manifest, dates, isins):
    """Reproduce the original causal wealth chain including the 2009 warmup."""
    records = [
        r
        for r in manifest["sources"]
        if Path(r.get("path", "")).name.startswith("equities_daily_")
    ]
    for record in records:
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("COTAHIST source changed")
    daily = pl.concat(
        [pl.read_parquet(r["path"]) for r in records], how="diagonal_relaxed"
    )
    daily = daily.filter(pl.col("trade_date") <= DEVELOPMENT_END)
    validation = prepare_cash_equities(
        daily, v1_isins=isins, require_units=True, maximum_rejection_fraction=0.005
    )
    schedule_record = next(
        r
        for r in manifest["sources"]
        if Path(r.get("path", "")).name == "b3_session_schedule_reconstructed_v1.csv"
    )
    if sha256_file(Path(schedule_record["path"])) != schedule_record["sha256"]:
        raise ValueError("source calendar changed")
    full_schedule = load_session_schedule(Path(schedule_record["path"]))
    start = validation.accepted["trade_date"].min()
    schedule = tuple(
        s for s in full_schedule if start <= s.trade_date <= DEVELOPMENT_END
    )
    following = next(
        s.decision_at for s in full_schedule if s.trade_date > DEVELOPMENT_END
    )
    calendar = [s.trade_date for s in schedule]
    observed_dates = set(validation.accepted["trade_date"])
    panel = panel_from_daily(
        validation.accepted,
        dates=calendar,
        isins=isins,
        source_session_complete=np.array([d in observed_dates for d in calendar]),
        invalid_observations=validation.rejected,
    )
    # The current store's retrospective Round-7 terms did not rewrite features.
    # Follow the bound ancestry to the original feature-producing terms.
    ancestor, ancestor_root = manifest, source
    while "round5_extension" in ancestor["metadata"]:
        parent = ancestor["metadata"]["round5_extension"]["base_store"]
        ancestor_root = Path(parent["root"])
        ancestor = bound_json(
            {
                "path": str(ancestor_root / "manifest.json"),
                "sha256": parent["manifest_sha256"],
            }
        )
    terms = verified_action_terms_from_table(
        pl.read_parquet(
            ancestor_root
            / ancestor["tables"]["corporate_actions_verified_terms"]["path"]
        )
    )
    universe = build_daily_universe(
        panel.close_brl,
        panel.volume_brl,
        panel.observed,
        trade_observed=panel.trade_observed,
        activity_valid=panel.activity_valid,
        source_session_complete=panel.source_session_complete,
    )
    inferred = infer_cotahist_action_terms(
        panel.dates,
        panel.isins,
        panel.close_brl,
        panel.quantity,
        panel.trades,
        panel.distribution_number,
        panel.observed,
        universe.active,
    )
    actions = align_decision_known_action_terms(
        terms,
        panel.dates,
        panel.isins,
        coverage_resolved=inferred.coverage_resolved,
        decision_timestamps=next_session_decision_cutoffs(
            schedule, following_decision_at=following
        ),
    )
    wealth = [np.empty(panel.observed.shape, np.float32) for _ in range(4)]
    valid = np.empty(panel.observed.shape, bool)
    build_shareholder_wealth_ohlc_into(
        panel.open_brl,
        panel.high_brl,
        panel.low_brl,
        panel.close_brl,
        panel.observed,
        actions,
        wealth_open=wealth[0],
        wealth_high=wealth[1],
        wealth_low=wealth[2],
        wealth_close=wealth[3],
        wealth_valid=valid,
    )
    linked = _route_decision_known_continuations(
        dates=panel.dates,
        isins=panel.isins,
        links=pl.read_parquet(
            source / manifest["tables"]["isin_succession_links"]["path"]
        ),
        decision_timestamps=[s.decision_at for s in schedule],
        raw_close=panel.close_brl,
        volume_brl=panel.volume_brl,
        trades=panel.trades,
        observed=panel.observed,
        trade_observed=panel.trade_observed,
        activity_valid=panel.activity_valid,
        ambiguous_action=~actions.session_resolved,
        shareholder_wealth_arrays=(*wealth, valid),
    )
    kept = np.searchsorted(panel.dates, np.asarray(dates, dtype="datetime64[D]"))
    old_wealth = np.load(source / "shareholder_wealth_close.npy", mmap_mode="r")
    if not np.array_equal(
        wealth[3][kept], old_wealth, equal_nan=True
    ) or not np.array_equal(
        valid[kept], np.load(source / "shareholder_wealth_valid.npy", mmap_mode="r")
    ):
        raise ValueError(
            "source reconstruction does not reproduce the sealed causal wealth chain"
        )
    returns, _ = exact_log_return(
        wealth[3], 1, linked.ambiguous_action, shareholder_wealth_valid=valid
    )
    skew, kurtosis, moment_valid = _rolling_moments(returns, 60)
    moment_valid &= _ambiguous_interval_clear(linked.ambiguous_action, 60)
    first = np.where(linked.observed.any(axis=0), linked.observed.argmax(axis=0), -1)
    age = kept[:, None] - 1 - first
    return {
        "realized_skew_60": skew[kept - 1],
        "realized_kurtosis_60": kurtosis[kept - 1],
        "observed_history_age_sessions": age,
    }, moment_valid[kept - 1]


def build_store(repair_root: Path, output: Path):
    source, original, parent = parent_store()
    if (
        output.resolve().is_relative_to(source.resolve())
        or "raw" in output.resolve().parts
    ):
        raise ValueError("derived repair must be outside immutable sources")
    financial_pointer = json.loads(
        (repair_root / "financial_family.json").read_text(encoding="utf-8")
    )
    financial_manifest = Path(financial_pointer["path"])
    financial = bound_json(financial_pointer)
    if financial["parent"] != parent:
        raise ValueError("financial repair binds a different research store")
    dates = np.load(source / "date_index.npy").astype(object).tolist()
    isins = tuple(np.load(source / "isin_index.npy").tolist())
    names = copy.deepcopy(original["feature_names"])
    del names["sidecar_fundamentals_native"]
    names["sidecar_fundamentals"] += [
        "earnings_negative_flag",
        "incomplete_latest_statement_flag",
    ]
    sources = source_families(original)
    protected = {
        n
        for n in original["arrays"]
        if not n.startswith("sidecar_") and n not in {"slow_values", "slow_valid"}
    }
    mapped = {}

    def old(name):
        if name not in mapped:
            mapped[name] = np.load(
                source / original["arrays"][name]["path"], mmap_mode="r"
            )
        return mapped[name]

    active = old("active")
    metadata = copy.deepcopy(original["metadata"])
    metadata["implementation_git_commit"] = _git_identity()["commit"]
    metadata["feature_age_contract"] = dict(FEATURE_AGE_CONTRACT)
    metadata["sidecar_capabilities"].pop("fundamentals_native", None)
    specifications = [
        FeatureSpec(**s)
        for s in metadata["feature_schema"]["specifications"]
        if not s["family"].startswith("sidecar_")
    ]
    evidence = []
    try:
        slow_path = repair_root / "slow_source_fields.npz"
        slow_manifest_path = repair_root / "slow_source_manifest.json"
        slow_identity = {
            "parent": parent,
            "implementation_sha256": hashlib.sha256(
                ast.dump(ast.parse(inspect.getsource(slow_source_fields))).encode()
            ).hexdigest(),
        }
        if slow_manifest_path.exists():
            cached = json.loads(slow_manifest_path.read_text(encoding="utf-8"))
            if {k: cached[k] for k in slow_identity} != slow_identity or sha256_file(
                slow_path
            ) != cached["data"]["sha256"]:
                raise ValueError("cached slow source derivation differs")
            with np.load(slow_path) as archive:
                slow_fields = {
                    n: archive[n] for n in archive.files if n != "moment_valid"
                }
                moment_valid = archive["moment_valid"]
        else:
            slow_fields, moment_valid = slow_source_fields(
                source, original, dates, isins
            )
            np.savez(slow_path, **slow_fields, moment_valid=moment_valid)
            write_json_atomic(
                slow_manifest_path, {**slow_identity, "data": binding(slow_path)}
            )
        with StoreStaging(output, dates=dates, isins=isins) as staging:
            for name in protected:
                staging.copy_array(name, source / original["arrays"][name]["path"])
            slow = staging.create_array(
                "slow_values", old("slow_values").shape, np.float32
            )
            slow_valid = staging.create_array(
                "slow_valid", old("slow_valid").shape, np.bool_
            )
            slow[:] = old("slow_values")
            slow_valid[:] = old("slow_valid")
            # The stored snapshot is t-1. Keep every other slow coordinate,
            # especially risk/target normalization fields, byte-identical.
            for name in ("realized_skew_60", "realized_kurtosis_60"):
                f = names["slow"].index(name)
                usable = moment_valid & active
                if not np.array_equal(usable, old("slow_valid")[:, :, f]):
                    raise ValueError(f"recomputed {name} source mask differs")
                slow[:, :, f] = np.where(usable, np.arcsinh(slow_fields[name]), 0)
                specifications = [
                    feature_specs("slow", (name,))[0]
                    if s.family == "slow" and s.name == name
                    else s
                    for s in specifications
                ]
            # Reconstruct the linked source history; never invert clipped ages.
            f = names["slow"].index("observed_history_age_sessions")
            logs = np.log1p(np.maximum(slow_fields["observed_history_age_sessions"], 0))
            slow[:, :, f] = np.where(
                slow_valid[:, :, f], logs / (logs + np.log1p(252.0)), 0
            )
            specifications = [
                feature_specs("slow", (s.name,))[0]
                if s.family == "slow" and s.name == "observed_history_age_sessions"
                else s
                for s in specifications
            ]
            del slow_fields, moment_valid, logs
            close_memmap(slow)
            close_memmap(slow_valid)
            for family in sorted(n for n in names if n.startswith("sidecar_")):
                short = family.removeprefix("sidecar_")
                field_names = tuple(names[family])
                specs = feature_specs(family, field_names)
                record = None
                if short == "fundamentals":
                    record = financial["data"]
                elif short in sources:
                    record = sources[short]["data"]
                if record is not None:
                    path = Path(record["path"])
                    if sha256_file(path) != record["sha256"]:
                        raise ValueError(f"family source differs: {short}")
                    frame = pl.read_parquet(path)
                    raw, mask, ages = align_family(frame, dates, isins, field_names)
                    del frame
                elif short == "oddlot":
                    record = next(
                        s
                        for s in original["sources"]
                        if str(s.get("path", "")).endswith("odd_lot_activity.parquet")
                    )
                    path = Path(record["path"])
                    if sha256_file(path) != record["sha256"]:
                        raise ValueError("oddlot source differs")
                    frame = derive_known_archive_features(
                        pl.read_parquet(path), dates, isins, group="oddlot"
                    )
                    panel = materialize_known_archive(
                        frame, dates, isins, group="oddlot"
                    )
                    if panel.feature_names != field_names:
                        raise ValueError("oddlot field order differs")
                    raw, mask, ages = panel.values, panel.valid, panel.age_sessions
                    del frame, panel
                else:
                    raise ValueError(f"missing physical source for {short}")
                values = staging.create_array(family + "_values", raw.shape, np.float32)
                valid = staging.create_array(family + "_valid", mask.shape, np.bool_)
                transform_feature_panel_into(raw, mask, active, specs, values, valid)
                ages[~active] = -1
                staging.write_array(family + "_age_sessions", ages)
                counts = []
                for f, name in enumerate(field_names):
                    previous = (
                        old(family + "_valid")[
                            ..., original["feature_names"][family].index(name)
                        ]
                        if name in original["feature_names"].get(family, ())
                        else np.zeros(active.shape, bool)
                    )
                    counts.append(
                        {
                            "field": name,
                            "source_valid_active": int((mask[..., f] & active).sum()),
                            "valid_active": int(valid[..., f].sum()),
                            "previous_valid_active": int((previous & active).sum()),
                            "gained": int((valid[..., f] & ~previous).sum()),
                            "lost": int((previous & active & ~valid[..., f]).sum()),
                        }
                    )
                evidence.append({"family": short, "source": record, "fields": counts})
                metadata["sidecar_capabilities"][short] = {
                    "enabled": [r["field"] for r in counts if r["valid_active"]],
                    "source_missing": [
                        r["field"] for r in counts if not r["valid_active"]
                    ],
                }
                specifications.extend(specs)
                close_memmap(values)
                close_memmap(valid)
                del raw, mask, ages
                print(
                    json.dumps({"stage": "family_rebuilt", "family": short}), flush=True
                )
            order = [
                f for f in ("slow", "intraday", "native_fast") if f in names
            ] + sorted(f for f in names if f.startswith("sidecar_"))
            specifications = tuple(
                s for f in order for s in specifications if s.family == f
            )
            metadata["feature_schema"] = {
                "minimum_rank_names": 20,
                "specifications": [asdict(s) for s in specifications],
                "sha256": feature_schema_sha256(specifications),
            }
            report = {
                "parent": parent,
                "financial_family": binding(financial_manifest),
                "families": evidence,
                "protected_arrays": sorted(protected),
                "conditioning": "fit-only robust median/IQR followed by asinh; common fields once per date; established P scales inherited by F",
                "forward_capture": False,
                "heldout_access": False,
            }
            metadata["data_repair"] = report
            staging.seal(
                feature_names=names,
                metadata=metadata,
                sources=[*original["sources"], binding(financial_manifest)],
                tables={n: source / r["path"] for n, r in original["tables"].items()},
                maximum_peak_rss_bytes=8 * 1024**3,
            )
        sealed = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for name in protected:
            if sealed["arrays"][name]["sha256"] != original["arrays"][name]["sha256"]:
                raise ValueError(f"protected array differs: {name}")
        report.update(
            store={
                "root": str(output),
                "manifest_sha256": sha256_file(output / "manifest.json"),
            },
            protected_arrays_exact=True,
        )
        write_json_atomic(repair_root / "store_repair.json", report)
        return report
    finally:
        for array in mapped.values():
            close_memmap(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-extracted-documents", action="store_true")
    args = parser.parse_args()
    if args.output:
        build_store(args.repair_root, args.output)
    else:
        build_financial_family(
            args.repair_root, reuse_extracted_documents=args.reuse_extracted_documents
        )


if __name__ == "__main__":
    main()
