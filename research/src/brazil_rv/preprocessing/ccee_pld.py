from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import fastexcel
import polars as pl

CONTRACT_VERSION = "CCEE_PLD_DAILY_POWER_EXPOSURE_V1"
EXPECTED_SOURCE_SHA256 = (
    "08a0aebfd3112ca65954ff3b623e66c67f3740943fe7c5ca6d219cc26f5b5ff6"
)
SHEET_NAME = "Test Sheet"
SUBMARKETS = ("SUDESTE", "SUL", "NORDESTE", "NORTE")
SECO = "SUDESTE"
B3_SESSION_HOURS = frozenset(range(10, 17))
PRIOR_WINDOW = 60
PRIOR_MIN_OBSERVATIONS = 20
LEVEL_SCALE_FLOOR = 0.05
CHANGE_SCALE_FLOOR = 0.02
ROBUST_Z_CLIP = 6.0
LOG_CHANGE_CLIP = 3.0

POWER_ROLES = ("hydro", "thermal", "renewable", "distribution", "transmission")
POWER_ROLE_EXPOSURES: dict[str, tuple[str, ...]] = {
    "ISIN:BRALUPCDAM15": ("hydro", "transmission"),
    "ISIN:BRAUREACNOR9": ("hydro", "renewable"),
    "ISIN:BRAXIAACNOR0": ("hydro", "thermal", "renewable", "transmission"),
    "ISIN:BRAXIAACNPC9": ("hydro", "thermal", "renewable", "transmission"),
    "ISIN:BRCMIGACNPR3": ("hydro", "renewable", "distribution", "transmission"),
    "ISIN:BRCPFEACNOR0": ("hydro", "renewable", "distribution", "transmission"),
    "ISIN:BRCPLEACNOR8": (
        "hydro",
        "thermal",
        "renewable",
        "distribution",
        "transmission",
    ),
    "ISIN:BREGIEACNOR9": ("hydro", "thermal", "renewable"),
    "ISIN:BRELETACNOR6": ("hydro", "thermal", "renewable", "transmission"),
    "ISIN:BRENEVACNOR8": ("thermal",),
    "ISIN:BRENGICDAM16": ("distribution", "transmission"),
    "ISIN:BREQTLACNOR0": ("distribution", "transmission"),
    "ISIN:BRISAEACNPR9": ("transmission",),
    "ISIN:BRLIGTACNOR2": ("hydro", "distribution"),
    "ISIN:BRTAEECDAM10": ("transmission",),
    "ISIN:BRTRPLACNPR1": ("transmission",),
}
ROLE_MAPPING_RATIONALE = (
    "Conservative issuer operating-role taxonomy keyed only by permanent B3 ISIN. "
    "It includes direct generators, distributors, and transmission operators; indirect "
    "equipment, fuel, and industrial cogeneration names are excluded. A role is included "
    "only when it was an enduring material activity over the security's observed research "
    "membership, rather than inferred from a later project or current ticker."
)
ROLE_MAPPING_UNCERTAINTY = (
    "Issuer-level multi-hot roles are not asset-weighted and do not encode acquisitions, "
    "disposals, contract coverage, or subsidiary ownership percentages. Effective use is "
    "the conjunction of this permanent-ID map and the canonical point-in-time membership "
    "mask; candidates outside the map must remain unavailable, not observed zero."
)
ROLE_EVIDENCE_URLS = (
    "https://ri.alupar.com.br/a-companhia/perfil-corporativo/",
    "https://ri.aurenenergia.com.br/a-companhia/mapa-de-ativos/",
    "https://ri.copel.com/a-copel/perfil-corporativo/",
    "https://ri.energisa.com.br/a-energisa/perfil-corporativo/",
    "https://ri.equatorialenergia.com.br/a-companhia/quem-somos/",
    "https://ri.isaenergiabrasil.com.br/pt/a-isa-energia/perfil-corporativo",
    "https://ri.light.com.br/a-companhia-/historico-e-perfil-corporativo/",
    "https://ri.taesa.com.br/",
)

# ANEEL's annual homologated PLD minima. The two post-2020 upper limits are
# kept only for years independently verified in official CCEE/ANEEL material.
PLD_MINIMUM_BRL_MWH = {
    2018: 40.16,
    2019: 42.35,
    2020: 39.68,
    2021: 49.77,
    2022: 55.70,
    2023: 69.04,
    2024: 61.07,
    2025: 58.60,
}
PLD_UPPER_LIMITS_BRL_MWH = {
    2021: (583.88, 1197.87),
    2022: (646.58, 1326.50),
    2023: (684.73, 1404.77),
    2024: (716.80, 1470.57),
}
REGULATORY_LIMIT_EVIDENCE_URLS = (
    "https://www.ccee.org.br/o/ccee/documentos/CCEE_1124403",
    "https://www.ccee.org.br/documents/80415/919464/25%C2%BA%20Encontro%20do%20PLD-%20Janeiro%20de%202022.pdf/fe773ade-54c6-942f-3ed9-e0185cb3fa76",
)

PLD_FEATURES = (
    "pld_seco_daily_level_z60",
    "pld_seco_b3_session_level_z60",
    "pld_session_rest_spread_ratio",
    "pld_daily_range_ratio",
    "pld_submarket_dispersion_ratio",
    "pld_change_1d_log",
    "pld_change_5d_log",
    "pld_change_1d_surprise_z60",
    "pld_floor_binding_flag",
    "pld_structural_cap_binding_flag",
    "pld_hourly_cap_binding_flag",
)
ROLE_FEATURES = tuple(f"power_role_{role}" for role in POWER_ROLES)
FEATURES = (*PLD_FEATURES, *ROLE_FEATURES)


@dataclass(frozen=True)
class PldDay:
    trade_date: date
    # Hour-major, then SUBMARKETS order. Missing values represent a missing
    # complete daylight-saving hour, never an imputed observation.
    values: tuple[tuple[float | None, ...], ...]


@dataclass
class WorkbookAudit:
    source_file: str
    source_sha256: str
    sheet_name: str
    workbook_rows: int
    workbook_columns: int
    canonical_data_rows: int
    ignored_extra_rows: int
    ignored_nondate_columns: int
    date_columns: int
    first_date: str
    last_date: str
    missing_calendar_dates: int
    complete_24_hour_days: int
    complete_23_hour_days: int
    ignored_extra_rows_with_date_values: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).strip()).date()
    except ValueError:
        return None


def _number(value: object) -> float | None:
    if value is None or not str(value).strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"Invalid PLD value: {value!r}")
    return parsed


def parse_wide_rows(
    rows: list[tuple[object, ...]], source_file: Path, source_sha256: str
) -> tuple[list[PldDay], WorkbookAudit]:
    if len(rows) < 97 or len(rows[0]) < 3:
        raise ValueError("CCEE PLD sheet is smaller than its 24x4 wide contract")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("CCEE PLD sheet rows have inconsistent widths")
    if tuple(str(value).strip() for value in rows[0][:2]) != ("Hora", "Submercado"):
        raise ValueError("CCEE PLD sheet has unexpected first columns")

    canonical = rows[1:97]
    expected_labels = [
        (str(hour), submarket) for hour in range(24) for submarket in SUBMARKETS
    ]
    actual_labels = [(str(row[0]).strip(), str(row[1]).strip()) for row in canonical]
    if actual_labels != expected_labels:
        raise ValueError("CCEE PLD canonical rows are not 24 hours x 4 submarkets")

    dated_columns = [
        (column, parsed)
        for column, value in enumerate(rows[0])
        if (parsed := _header_date(value)) is not None
    ]
    if len({parsed for _, parsed in dated_columns}) != len(dated_columns):
        raise ValueError("CCEE PLD sheet contains duplicate date columns")
    days: list[PldDay] = []
    day_hour_counts: list[int] = []
    for column, trade_date in dated_columns:
        values: list[tuple[float | None, ...]] = []
        for hour in range(24):
            hour_values = tuple(
                _number(canonical[hour * len(SUBMARKETS) + offset][column])
                for offset in range(len(SUBMARKETS))
            )
            present = sum(value is not None for value in hour_values)
            if present not in {0, len(SUBMARKETS)}:
                raise ValueError(
                    f"Partial submarket PLD hour on {trade_date} hour {hour}"
                )
            values.append(hour_values)
        observed_hours = sum(
            all(value is not None for value in hour) for hour in values
        )
        if observed_hours not in {23, 24}:
            raise ValueError(
                f"Expected 23 or 24 complete PLD hours on {trade_date}, got {observed_hours}"
            )
        day_hour_counts.append(observed_hours)
        days.append(PldDay(trade_date=trade_date, values=tuple(values)))
    days.sort(key=lambda day: day.trade_date)
    if not days:
        raise ValueError("CCEE PLD sheet contains no date columns")

    present_dates = {day.trade_date for day in days}
    span_days = (days[-1].trade_date - days[0].trade_date).days + 1
    extra_date_columns = {column for column, _ in dated_columns}
    ignored_extra_with_values = sum(
        any(
            column < len(row) and _number(row[column]) is not None
            for column in extra_date_columns
        )
        for row in rows[97:]
        if row[0] is not None and str(row[0]).strip().isdigit()
    )
    audit = WorkbookAudit(
        source_file=str(source_file.resolve()),
        source_sha256=source_sha256,
        sheet_name=SHEET_NAME,
        workbook_rows=len(rows),
        workbook_columns=width,
        canonical_data_rows=len(canonical),
        ignored_extra_rows=max(len(rows) - 97, 0),
        ignored_nondate_columns=width - len(dated_columns),
        date_columns=len(days),
        first_date=days[0].trade_date.isoformat(),
        last_date=days[-1].trade_date.isoformat(),
        missing_calendar_dates=span_days - len(present_dates),
        complete_24_hour_days=day_hour_counts.count(24),
        complete_23_hour_days=day_hour_counts.count(23),
        ignored_extra_rows_with_date_values=ignored_extra_with_values,
    )
    return days, audit


def read_pld_workbook(
    source_file: Path, *, expected_sha256: str = EXPECTED_SOURCE_SHA256
) -> tuple[list[PldDay], WorkbookAudit]:
    source_hash = _sha256(source_file)
    if source_hash.casefold() != expected_sha256.casefold():
        raise ValueError(
            f"Frozen CCEE workbook SHA256 mismatch: {source_hash} != {expected_sha256}"
        )
    workbook = fastexcel.read_excel(source_file)
    if workbook.sheet_names != [SHEET_NAME]:
        raise ValueError(
            f"Expected only sheet {SHEET_NAME!r}, got {workbook.sheet_names}"
        )
    frame = workbook.load_sheet(SHEET_NAME, header_row=None).to_polars()
    return parse_wide_rows(frame.rows(), source_file, source_hash)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty PLD slice")
    return statistics.fmean(values)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _prior_robust_z(
    current: float,
    prior: list[float],
    *,
    scale_floor: float,
) -> tuple[float, bool]:
    if len(prior) < PRIOR_MIN_OBSERVATIONS:
        return 0.0, False
    window = prior[-PRIOR_WINDOW:]
    center = statistics.median(window)
    mad = statistics.median(abs(value - center) for value in window)
    scale = max(1.4826 * mad, scale_floor)
    return _clamp((current - center) / scale, -ROBUST_Z_CLIP, ROBUST_Z_CLIP), True


def _daily_base(day: PldDay) -> dict[str, float | int]:
    seco_values = [
        hour_values[0] for hour_values in day.values if hour_values[0] is not None
    ]
    assert all(value is not None for value in seco_values)
    seco = [float(value) for value in seco_values]
    session = [
        float(day.values[hour][0])
        for hour in B3_SESSION_HOURS
        if day.values[hour][0] is not None
    ]
    rest = [
        float(hour_values[0])
        for hour, hour_values in enumerate(day.values)
        if hour not in B3_SESSION_HOURS and hour_values[0] is not None
    ]
    dispersions = [
        statistics.pstdev(float(value) for value in hour_values)
        for hour_values in day.values
        if all(value is not None for value in hour_values)
    ]
    all_values = [
        float(value)
        for hour_values in day.values
        for value in hour_values
        if value is not None
    ]
    daily_mean = _mean(seco)
    session_mean = _mean(session)
    rest_mean = _mean(rest)
    return {
        "pld_observed_hours": len(seco),
        "pld_seco_daily_mean_brl_mwh": daily_mean,
        "pld_seco_b3_session_mean_brl_mwh": session_mean,
        "pld_seco_rest_mean_brl_mwh": rest_mean,
        "pld_seco_daily_min_brl_mwh": min(seco),
        "pld_seco_daily_max_brl_mwh": max(seco),
        "pld_all_submarket_min_brl_mwh": min(all_values),
        "pld_all_submarket_max_brl_mwh": max(all_values),
        "pld_submarket_hourly_dispersion_mean_brl_mwh": _mean(dispersions),
    }


def derive_daily_features(days: list[PldDay]) -> list[dict[str, object]]:
    ordered = sorted(days, key=lambda day: day.trade_date)
    if len({day.trade_date for day in ordered}) != len(ordered):
        raise ValueError("Duplicate PLD trade dates")
    base_by_date = {day.trade_date: _daily_base(day) for day in ordered}
    daily_log_history: list[float] = []
    session_log_history: list[float] = []
    change_history: list[float] = []
    rows: list[dict[str, object]] = []
    for day in ordered:
        base = base_by_date[day.trade_date]
        daily_mean = float(base["pld_seco_daily_mean_brl_mwh"])
        session_mean = float(base["pld_seco_b3_session_mean_brl_mwh"])
        rest_mean = float(base["pld_seco_rest_mean_brl_mwh"])
        daily_log = math.log(daily_mean)
        session_log = math.log(session_mean)
        daily_z, daily_z_mask = _prior_robust_z(
            daily_log, daily_log_history, scale_floor=LEVEL_SCALE_FLOOR
        )
        session_z, session_z_mask = _prior_robust_z(
            session_log, session_log_history, scale_floor=LEVEL_SCALE_FLOOR
        )
        row: dict[str, object] = {
            "trade_date": day.trade_date,
            **base,
            "pld_seco_daily_level_z60": daily_z,
            "pld_seco_daily_level_z60_mask": daily_z_mask,
            "pld_seco_b3_session_level_z60": session_z,
            "pld_seco_b3_session_level_z60_mask": session_z_mask,
            "pld_session_rest_spread_ratio": _clamp(
                (session_mean - rest_mean) / daily_mean, -3.0, 3.0
            ),
            "pld_session_rest_spread_ratio_mask": True,
            "pld_daily_range_ratio": _clamp(
                (
                    float(base["pld_seco_daily_max_brl_mwh"])
                    - float(base["pld_seco_daily_min_brl_mwh"])
                )
                / daily_mean,
                0.0,
                5.0,
            ),
            "pld_daily_range_ratio_mask": True,
            "pld_submarket_dispersion_ratio": _clamp(
                float(base["pld_submarket_hourly_dispersion_mean_brl_mwh"])
                / daily_mean,
                0.0,
                3.0,
            ),
            "pld_submarket_dispersion_ratio_mask": True,
        }
        lag1 = base_by_date.get(day.trade_date - timedelta(days=1))
        lag5 = base_by_date.get(day.trade_date - timedelta(days=5))
        change1 = (
            math.log(daily_mean / float(lag1["pld_seco_daily_mean_brl_mwh"]))
            if lag1 is not None
            else 0.0
        )
        change5 = (
            math.log(daily_mean / float(lag5["pld_seco_daily_mean_brl_mwh"]))
            if lag5 is not None
            else 0.0
        )
        row["pld_change_1d_log"] = _clamp(change1, -LOG_CHANGE_CLIP, LOG_CHANGE_CLIP)
        row["pld_change_1d_log_mask"] = lag1 is not None
        row["pld_change_5d_log"] = _clamp(change5, -LOG_CHANGE_CLIP, LOG_CHANGE_CLIP)
        row["pld_change_5d_log_mask"] = lag5 is not None
        surprise, surprise_mask = (
            _prior_robust_z(change1, change_history, scale_floor=CHANGE_SCALE_FLOOR)
            if lag1 is not None
            else (0.0, False)
        )
        row["pld_change_1d_surprise_z60"] = surprise
        row["pld_change_1d_surprise_z60_mask"] = surprise_mask

        minimum = PLD_MINIMUM_BRL_MWH.get(day.trade_date.year)
        row["pld_floor_binding_flag"] = float(
            minimum is not None
            and float(base["pld_all_submarket_min_brl_mwh"]) <= minimum + 0.011
        )
        row["pld_floor_binding_flag_mask"] = minimum is not None
        upper_limits = PLD_UPPER_LIMITS_BRL_MWH.get(day.trade_date.year)
        if upper_limits is None:
            row["pld_structural_cap_binding_flag"] = 0.0
            row["pld_structural_cap_binding_flag_mask"] = False
            row["pld_hourly_cap_binding_flag"] = 0.0
            row["pld_hourly_cap_binding_flag_mask"] = False
        else:
            structural, hourly = upper_limits
            submarket_daily_means = [
                _mean(
                    [
                        float(hour_values[index])
                        for hour_values in day.values
                        if hour_values[index] is not None
                    ]
                )
                for index in range(len(SUBMARKETS))
            ]
            row["pld_structural_cap_binding_flag"] = float(
                max(submarket_daily_means) >= structural - 0.10
            )
            row["pld_structural_cap_binding_flag_mask"] = True
            row["pld_hourly_cap_binding_flag"] = float(
                float(base["pld_all_submarket_max_brl_mwh"]) >= hourly - 0.011
            )
            row["pld_hourly_cap_binding_flag_mask"] = True

        # Every normalizer above saw only history ending at D-1. Append D last.
        daily_log_history.append(daily_log)
        session_log_history.append(session_log)
        if lag1 is not None:
            change_history.append(change1)
        rows.append(row)
    return rows


def build_power_rows(
    days: list[PldDay], *, start: date | None = None, end: date | None = None
) -> list[dict[str, object]]:
    daily_rows = derive_daily_features(days)
    output: list[dict[str, object]] = []
    for daily in daily_rows:
        trade_date = daily["trade_date"]
        if start is not None and trade_date < start:
            continue
        if end is not None and trade_date > end:
            continue
        for security_id, roles in sorted(POWER_ROLE_EXPOSURES.items()):
            row = {"trade_date": trade_date, "security_id": security_id, **daily}
            for role, feature in zip(POWER_ROLES, ROLE_FEATURES, strict=True):
                row[feature] = float(role in roles)
                row[f"{feature}_mask"] = True
            output.append(row)
    return output


def _rows_to_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        raise ValueError("CCEE PLD sidecar contains no rows")
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .with_columns(
            pl.col("trade_date").cast(pl.Date),
            *[pl.col(feature).cast(pl.Float32) for feature in FEATURES],
            *[pl.col(f"{feature}_mask").cast(pl.Boolean) for feature in FEATURES],
        )
        .sort("trade_date", "security_id")
    )


def build_sidecar(
    source_file: Path,
    output_dir: Path,
    *,
    start: date | None = None,
    end: date | None = None,
    expected_sha256: str = EXPECTED_SOURCE_SHA256,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    days, workbook_audit = read_pld_workbook(
        source_file, expected_sha256=expected_sha256
    )
    frame = _rows_to_frame(build_power_rows(days, start=start, end=end))
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "ccee_pld_daily_power.parquet"
    frame.write_parquet(data_path, compression="zstd", statistics=True)
    feature_valid_rows = {
        feature: int(frame.get_column(f"{feature}_mask").sum()) for feature in FEATURES
    }
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "workbook_audit": asdict(workbook_audit),
        "availability_rule": (
            "The complete PLD curve for trade date D is available at D-1 20:00 "
            "America/Sao_Paulo and is usable by every B3 decision on D"
        ),
        "source_geometry_rule": (
            "row 0 date columns; rows 1-96 exactly 24 hours x 4 submarkets; "
            "later DST/note rows and non-date columns excluded"
        ),
        "normalization_rule": (
            "log levels and 1d changes use shifted prior-60 median/MAD with 20-prior "
            "minimum, fixed scale floors, and +/-6 clipping; raw log changes and "
            "dimensionless spreads are fixed-clipped; no fitted full-sample statistics"
        ),
        "b3_session_hours_local": sorted(B3_SESSION_HOURS),
        "features": list(FEATURES),
        "feature_valid_rows": feature_valid_rows,
        "power_role_exposures": {
            security_id: list(roles)
            for security_id, roles in sorted(POWER_ROLE_EXPOSURES.items())
        },
        "role_mapping_rationale": ROLE_MAPPING_RATIONALE,
        "role_mapping_uncertainty": ROLE_MAPPING_UNCERTAINTY,
        "role_evidence_urls": list(ROLE_EVIDENCE_URLS),
        "non_power_missingness_rule": (
            "No source row is emitted outside the conservative permanent-ID power map; "
            "the shared aligner must leave those names masked, not fill observed zeros"
        ),
        "pld_minimum_brl_mwh": PLD_MINIMUM_BRL_MWH,
        "pld_upper_limits_brl_mwh": {
            year: {"structural": values[0], "hourly": values[1]}
            for year, values in PLD_UPPER_LIMITS_BRL_MWH.items()
        },
        "regulatory_limit_evidence_urls": list(REGULATORY_LIMIT_EVIDENCE_URLS),
        "cap_mask_rule": (
            "Upper-limit features are unavailable outside independently verified "
            "2021-2024 structural/hourly regimes; limits are never inferred from sample maxima"
        ),
        "start_filter": start.isoformat() if start else None,
        "end_filter": end.isoformat() if end else None,
        "output_rows": frame.height,
        "distinct_dates": frame.get_column("trade_date").n_unique(),
        "distinct_power_securities": frame.get_column("security_id").n_unique(),
        "first_trade_date": frame.get_column("trade_date").min().isoformat(),
        "last_trade_date": frame.get_column("trade_date").max().isoformat(),
        "output_file": data_path.name,
        "output_sha256": _sha256(data_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def _iso_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe CCEE day-ahead PLD power-exposure sidecar"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=_iso_date)
    parser.add_argument("--end", type=_iso_date)
    args = parser.parse_args()
    manifest = build_sidecar(args.source, args.out, start=args.start, end=args.end)
    print(
        f"Wrote {manifest['output_rows']:,} rows across "
        f"{manifest['distinct_dates']:,} dates and "
        f"{manifest['distinct_power_securities']:,} power securities to {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
