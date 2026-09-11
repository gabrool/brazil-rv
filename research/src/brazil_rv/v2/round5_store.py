"""Extend the sealed development store with admitted, separately masked families."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from .artifacts import sha256_file, write_json_atomic
from .build_store import (
    _external_feature_validity_by_survival_liquidity,
    _feature_validity_by_survival,
    _prior_adv20,
)
from .contract import DEVELOPMENT_END, FINETUNE_START, PRETRAIN_END
from .data_foundation import continuation_identity_axis
from .feature_spec import (
    FeatureSpec,
    feature_schema_sha256,
    feature_specs,
    transform_feature_panel_into,
)
from .research_rounds import _git_identity
from .store import StoreStaging, close_memmap, open_store_for_samples, peak_rss_bytes
from .store_comparison import _survival_audit

MAXIMUM_RSS = 8 * 1024**3


def align_family(
    frame: pl.DataFrame, dates: list, isins: tuple[str, ...], names: tuple[str, ...]
):
    """Align already decision-dated observations; never apply another lag/fill."""
    shape = (len(dates), len(isins), len(names))
    values = np.zeros(shape, np.float32)
    valid = np.zeros(shape, bool)
    ages = np.full(shape, -1, np.float32)
    if frame.is_empty():
        return values, valid, ages
    if frame["date"].max() > DEVELOPMENT_END:
        raise PermissionError("Round-5 family contains held-out consumer rows")
    if frame.select(pl.struct("date", "isin").is_duplicated().any()).item():
        raise ValueError("family has duplicate decision/security rows")
    date_axis = pl.DataFrame(
        {"date": dates, "__d": np.arange(len(dates))},
        schema={"date": pl.Date, "__d": pl.Int64},
    )
    name_axis = pl.DataFrame({"isin": isins, "__n": np.arange(len(isins))})
    aligned = frame.join(date_axis, on="date", how="inner").join(
        name_axis, on="isin", how="inner"
    )
    if aligned.height != frame.height:
        raise ValueError(
            "family includes dates or securities outside the bound store axes"
        )
    d, n = aligned["__d"].to_numpy(), aligned["__n"].to_numpy()
    for index, name in enumerate(names):
        if name not in aligned.columns:
            continue  # Explicitly unavailable members of an otherwise useful family.
        payload = aligned[name].cast(pl.Float32).to_numpy()
        known = np.isfinite(payload)
        if f"{name}_mask" in aligned.columns:
            known &= aligned[f"{name}_mask"].fill_null(False).to_numpy()
        age_name = f"{name}_age_sessions"
        if known.any() and age_name not in aligned.columns:
            raise ValueError(f"{name} has values but no source-age provenance")
        age = (
            aligned[age_name].cast(pl.Float32).to_numpy()
            if age_name in aligned.columns
            else np.full(payload.shape, -1)
        )
        if np.any(known & (~np.isfinite(age) | (age < 0))):
            raise ValueError(f"{name} marks unknown source age as available")
        values[d[known], n[known], index] = payload[known]
        valid[d[known], n[known], index] = True
        ages[d[known], n[known], index] = age[known]
    return values, valid, ages


def _verified_json(record: dict) -> dict:
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"source manifest identity differs: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build(plan_path: Path, output: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    code = _git_identity()
    registration = _verified_json(plan["registration"])
    for amendment in plan["amendments"]:
        if sha256_file(Path(amendment["path"])) != amendment["sha256"]:
            raise ValueError("Round-5 formula amendment identity differs")
    base = registration["base_store"]
    source = Path(base["root"])
    if sha256_file(source / "manifest.json") != base["manifest_sha256"]:
        raise ValueError("base store differs from the Round-5 registration")
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if original["axes"]["date_end"] > DEVELOPMENT_END.isoformat():
        raise PermissionError("base store extends into held-out history")
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("new store must be outside its immutable source")
    rows = np.arange(original["axes"]["date_count"])
    axis = np.load(source / "date_index.npy", allow_pickle=False)
    samples = rows[
        (axis <= np.datetime64(PRETRAIN_END)) | (axis >= np.datetime64(FINETUNE_START))
    ]
    # The ten-session embargo is causal context, never a fit/evaluation sample.
    store, access = open_store_for_samples(
        source,
        samples,
        purpose="training",
        history_lookbacks=60,
        history_end_offsets=0,
    )
    dates = store.dates.astype(object).tolist()
    isins = store.isins
    active = store.read("active", rows)
    observed = store.read("observed", rows)
    links = pl.read_parquet(
        source / original["tables"]["isin_succession_links"]["path"]
    )
    survival_identities = continuation_identity_axis(isins, links)
    survival = _survival_audit(store, axis)
    if survival and survival["different_cells"]:
        raise ValueError("Base survival audit flag differs from dated observations")
    prior_adv = _prior_adv20(
        store.read("volume_brl", rows), store.read("activity_valid", rows)
    )
    families = plan["families"]
    if len({item["family"] for item in families}) != len(families):
        raise ValueError("duplicate planned sidecar family")
    if any(item["family"] == "oddlot" for item in families):
        raise ValueError("Round-5 preserves the existing oddlot arrays exactly")
    replacements = {f"sidecar_{item['family']}" for item in families}
    allowed = {
        f"{family}_{suffix}"
        for family in replacements
        for suffix in ("values", "valid", "age_sessions")
    }
    metadata = copy.deepcopy(original["metadata"])
    feature_names = copy.deepcopy(original["feature_names"])
    specifications = [
        FeatureSpec(**item)
        for item in metadata["feature_schema"]["specifications"]
        if item["family"] not in replacements
    ]
    evidence = []
    try:
        with StoreStaging(output, dates=dates, isins=isins) as staging:
            for name, record in original["arrays"].items():
                if name not in allowed:
                    staging.copy_array(name, source / record["path"])
            for item in families:
                family = f"sidecar_{item['family']}"
                names = tuple(item["feature_names"])
                specs = feature_specs(family, names)
                if item["status"] == "admitted":
                    proof = _verified_json(item["availability_proof"])
                    if not (
                        proof.get("causality_passed") is True
                        and proof.get("first_available_decision_passed") is True
                    ):
                        raise ValueError(
                            f"{family} lacks its joined availability proof"
                        )
                    if proof.get("data_sha256") != item["data"]["sha256"]:
                        raise ValueError(f"{family} proof binds different source data")
                    _verified_json(item["source_manifest"])
                    path = Path(item["data"]["path"])
                    if sha256_file(path) != item["data"]["sha256"]:
                        raise ValueError(f"{family} source data identity differs")
                    frame = pl.read_parquet(path)
                elif item["status"] == "unavailable" and item.get("reason"):
                    frame = pl.DataFrame(schema={"date": pl.Date, "isin": pl.String})
                else:
                    raise ValueError(f"{family} has no admission decision")
                raw, mask, ages = align_family(frame, dates, isins, names)
                del frame
                present = mask.any(axis=2)
                composition = {
                    "population": "active observed name-days with at least one raw family field",
                    "uses_outcomes_for_features": False,
                    "survival": _feature_validity_by_survival(
                        axis,
                        active,
                        observed,
                        {family: (mask, present)},
                        survival_identities,
                        maximum_gap=None,
                    ).to_dicts(),
                    "liquidity_strata": [],
                    "pooled_liquidity_strata": [],
                    "status": "source_unavailable_no_observable_population",
                }
                if np.any(active & observed & present & np.isfinite(prior_adv)):
                    pooled = _external_feature_validity_by_survival_liquidity(
                        axis,
                        active,
                        observed,
                        prior_adv,
                        family,
                        mask,
                        present,
                        survival_identities,
                        enforce=False,
                    )
                    composition["pooled_liquidity_strata"] = pooled.drop(
                        "coverage_note"
                    ).to_dicts()
                    strata = _external_feature_validity_by_survival_liquidity(
                        axis,
                        active,
                        observed,
                        prior_adv,
                        family,
                        mask,
                        present,
                        survival_identities,
                        calendar_standardized=True,
                    )
                    # The base helper carries an old fixed lending note. The
                    # actual counts above describe this extended population.
                    composition["liquidity_strata"] = strata.drop(
                        "coverage_note"
                    ).to_dicts()
                    composition["status"] = "completed_no_binding_failure"
                    composition["comparison"] = (
                        "equal calendar-session weight within common observed survival-group support; fixed-calendar-weight name bootstrap"
                    )
                values = staging.create_array(f"{family}_values", raw.shape, np.float32)
                valid = staging.create_array(f"{family}_valid", mask.shape, np.bool_)
                transform_feature_panel_into(raw, mask, active, specs, values, valid)
                ages[~valid] = -1
                staging.write_array(f"{family}_age_sessions", ages)
                coverage = []
                years = np.array([day.year for day in dates])
                for year in np.unique(years):
                    selected = years == year
                    coverage.append(
                        {
                            "year": int(year),
                            "active_name_days": int(active[selected].sum()),
                            "informative_active_name_days": int(
                                valid[selected].any(axis=-1).sum()
                            ),
                            "valid_by_feature": dict(
                                zip(
                                    names,
                                    valid[selected]
                                    .sum(axis=(0, 1))
                                    .astype(int)
                                    .tolist(),
                                    strict=True,
                                )
                            ),
                        }
                    )
                evidence.append(
                    {**item, "coverage": coverage, "composition_audit": composition}
                )
                observed_features = valid.any(axis=(0, 1))
                metadata.setdefault("sidecar_capabilities", {})[item["family"]] = {
                    "enabled": [
                        name
                        for name, observed in zip(names, observed_features, strict=True)
                        if observed
                    ],
                    "source_missing": [
                        name
                        for name, observed in zip(names, observed_features, strict=True)
                        if not observed
                    ],
                }
                feature_names[family] = list(names)
                specifications.extend(specs)
                close_memmap(values)
                close_memmap(valid)
                del raw, mask, ages
                if peak_rss_bytes() > MAXIMUM_RSS:
                    raise MemoryError("Round-5 extension exceeded 8 GiB")
            by_family = {
                family: [spec for spec in specifications if spec.family == family]
                for family in feature_names
            }
            order = [
                family
                for family in ("slow", "intraday", "native_fast")
                if family in by_family
            ]
            order += sorted(
                family for family in by_family if family.startswith("sidecar_")
            )
            ordered_specs = tuple(
                spec for family in order for spec in by_family[family]
            )
            metadata["feature_schema"] = {
                "minimum_rank_names": 20,
                "specifications": [asdict(spec) for spec in ordered_specs],
                "sha256": feature_schema_sha256(ordered_specs),
            }
            metadata["round5_extension"] = {
                "code": code,
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "amendments": plan["amendments"],
                "base_store": base,
                "families": evidence,
                "protected_arrays_copied_byte_for_byte": True,
                "fit_dependent_clipping_in_store": False,
                "survival_flag_reconstruction": survival,
                "audit_table_scope": "Copied base tables describe the original store; current sidecar coverage and composition are recorded per family here.",
                "provider_invariance": "Provider observations and verified terms are copied unchanged; the extension consumes only admitted feature archives and does not reconstruct protected features or targets from providers.",
            }
            staging.seal(
                feature_names=feature_names,
                metadata=metadata,
                sources=[
                    *original["sources"],
                    {"path": str(source), "manifest_sha256": base["manifest_sha256"]},
                ],
                tables={
                    name: source / record["path"]
                    for name, record in original["tables"].items()
                },
                maximum_peak_rss_bytes=MAXIMUM_RSS,
            )
        result = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        changed = [
            name
            for name, record in original["arrays"].items()
            if name not in allowed
            and result["arrays"][name]["sha256"] != record["sha256"]
        ]
        if (
            changed
            or result["indices"] != original["indices"]
            or result["tables"] != original["tables"]
        ):
            raise ValueError(
                f"Round-5 changed protected arrays, axes or tables: {changed}"
            )
        acceptance = {
            "status": "passed",
            "store_root": str(output),
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "protected_array_count": len(original["arrays"].keys() - allowed),
            "protected_arrays_exact": True,
            "indices_and_tables_exact": True,
            "peak_rss_bytes": peak_rss_bytes(),
            "families": evidence,
            "s0_input_identity": {
                name: result["arrays"][name]["sha256"]
                for name in (
                    "slow_values",
                    "slow_valid",
                    "slow_age_sessions",
                    "slow_timestep_valid",
                    "active",
                )
                if name in result["arrays"]
            },
            "access": access.payload(),
            "neural_fits": False,
        }
        write_json_atomic(output / "round5_acceptance.json", acceptance)
        return acceptance
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.plan, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "store_root", "manifest_sha256", "peak_rss_bytes")
            }
        )
    )


if __name__ == "__main__":
    main()
