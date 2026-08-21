from __future__ import annotations

import argparse
import bisect
import email.utils
import hashlib
import html
import io
import json
import math
import re
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import fastexcel
import polars as pl

from .contract import DECISION_TIMES

CONTRACT_VERSION = "B3_INDEX_REBALANCE_V1"
INDEXES = ("IBOV", "IBXX", "SMLL")
FEATURE_SUFFIXES = (
    "current_weight_sqrt",
    "preview_delta_signed_sqrt",
    "preview_add",
    "preview_delete",
    "preview_pressure",
    "pre_effective_ramp",
    "post_effective_reversal",
)
FEATURES = tuple(
    f"{index.lower()}_{suffix}" for index in INDEXES for suffix in FEATURE_SUFFIXES
)
TICKER_PATTERN = re.compile(r"^[A-Z0-9]{4,8}$")
SAO_PAULO = ZoneInfo("America/Sao_Paulo")


@dataclass(frozen=True)
class SourceEvent:
    disclosure_date: date
    effective_date: date
    stage: str
    page_url: str
    asset_url: str


@dataclass(frozen=True)
class Portfolio:
    index: str
    weights: dict[str, float]
    quantities: dict[str, float]


@dataclass(frozen=True)
class Disclosure:
    disclosure_date: date
    effective_date: date
    stage: str
    available_at_utc: datetime
    source_asset: Path


# Official contemporaneous B3 release pages and immutable attachments. The first
# current portfolio is needed to compare the August previews with what was then
# held; the remaining events span both discovery selection windows.
SOURCE_EVENTS = (
    SourceEvent(
        date(2023, 5, 2),
        date(2023, 5, 2),
        "effective",
        "https://www.b3.com.br/pt_br/noticias/nova-carteira-do-ibovespa-b3-tem-86-papeis.htm",
        "https://www.b3.com.br/data/files/11/F1/D5/7E/1ECD781064456178AC094EA8/Carteira%20Final.2.zip",
    ),
    SourceEvent(
        date(2023, 8, 1),
        date(2023, 9, 4),
        "preview_1",
        "https://www.b3.com.br/pt_br/noticias/primeira-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-setembro-conta-com-86-ativos.htm",
        "https://www.b3.com.br/data/files/DE/B6/5D/E8/4B1B98101DBF7498AC094EA8/Primeira%20Previa.zip",
    ),
    SourceEvent(
        date(2023, 8, 16),
        date(2023, 9, 4),
        "preview_2",
        "https://www.b3.com.br/pt_br/noticias/segunda-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-setembro-conta-com-86-ativos.htm",
        "https://www.b3.com.br/data/files/EA/71/CB/F8/DCEF9810746C7D98AC094EA8/2PREVIA_3TRI23.zip",
    ),
    SourceEvent(
        date(2023, 8, 31),
        date(2023, 9, 4),
        "preview_3",
        "https://www.b3.com.br/pt_br/noticias/terceira-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-setembro-conta-com-86-ativos.htm",
        "https://www.b3.com.br/data/files/67/A5/88/C2/BFB4A8103234E0A8AC094EA8/3PREVIA_3TRI2023.zip",
    ),
    SourceEvent(
        date(2023, 9, 4),
        date(2023, 9, 4),
        "effective",
        "https://www.b3.com.br/pt_br/noticias/b3-anuncia-nova-carteira-do-ibovespa-ate-dezembro.htm",
        "https://www.b3.com.br/data/files/7B/47/9C/C8/6856A8103234E0A8AC094EA8/Carteira%20Definitiva.zip",
    ),
    SourceEvent(
        date(2023, 12, 1),
        date(2024, 1, 2),
        "preview_1",
        "https://www.b3.com.br/pt_br/noticias/primeira-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-janeiro-conta-com-86-ativos.htm",
        "https://www.b3.com.br/data/files/CF/11/FE/26/FE54C810DC9DE3C8AC094EA8/Primeira%20Previa.zip",
    ),
    SourceEvent(
        date(2023, 12, 18),
        date(2024, 1, 2),
        "preview_2",
        "https://www.b3.com.br/pt_br/noticias/segunda-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-janeiro-conta-com-87-ativos.htm",
        "https://www.b3.com.br/data/files/AF/44/EF/10/62D7C8103152D4C8AC094EA8/Arquivos%20Segunda%20Previa.zip",
    ),
    SourceEvent(
        date(2023, 12, 27),
        date(2024, 1, 2),
        "preview_3",
        "https://www.b3.com.br/pt_br/noticias/terceira-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-janeiro-conta-com-87-ativos.htm",
        "https://www.b3.com.br/data/files/86/E7/97/D9/F8BAC8103152D4C8AC094EA8/Terceira%20Previa.Jan24.zip",
    ),
    SourceEvent(
        date(2024, 1, 2),
        date(2024, 1, 2),
        "effective",
        "https://www.b3.com.br/pt_br/noticias/novas-carteiras-indices.htm",
        "https://www.b3.com.br/data/files/F7/51/8B/A4/8ABCC8103152D4C8AC094EA8/Novas%20carteiras%20indices.zip",
    ),
    SourceEvent(
        date(2024, 4, 1),
        date(2024, 5, 6),
        "preview_1",
        "https://www.b3.com.br/pt_br/noticias/primeira-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-maio-conta-com-88-ativos.htm",
        "https://www.b3.com.br/data/files/54/43/ED/8F/70B9E810D34843E8AC094EA8/Arquivos%20Primeira%20Previa%20Maio%202024.zip",
    ),
    SourceEvent(
        date(2024, 4, 16),
        date(2024, 5, 6),
        "preview_2",
        "https://www.b3.com.br/pt_br/noticias/segunda-previa-da-carteira-do-ibovespa-b3-que-entra-em-vigor-em-maio-conta-com-87-ativos.htm",
        "https://www.b3.com.br/data/files/CC/F1/8D/BF/F08EE8100E866AE8AC094EA8/Arquivos%20Segunda%20Previa.zip",
    ),
    SourceEvent(
        date(2024, 5, 2),
        date(2024, 5, 6),
        "preview_3",
        "https://www.b3.com.br/pt_br/noticias/terceira-previa-da-carteira-do-ibovespa-b3.htm",
        "https://www.b3.com.br/data/files/1C/64/FC/78/4CE3F8100E866AE8AC094EA8/Previa%20Carteira%20Indices.zip",
    ),
    SourceEvent(
        date(2024, 5, 6),
        date(2024, 5, 6),
        "effective",
        "https://www.b3.com.br/pt_br/noticias/ibovespa-e-demais-indices.htm",
        "https://www.b3.com.br/data/files/3D/12/7F/D1/4EE4F8100A0DD4F8AC094EA8/VIRADA%20-%20MAI2024.xlsx",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(url: str) -> tuple[bytes, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": "Brazil-RV research"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read(), response.headers.get("Last-Modified")


def _page_links_to_asset(page_url: str, asset_url: str) -> bool:
    body = _get(page_url)[0].decode("latin-1", errors="ignore")
    links = {
        urllib.parse.urljoin(page_url, html.unescape(value))
        for value in re.findall(r'href=["\']([^"\']+)["\']', body, flags=re.IGNORECASE)
    }
    normalized = urllib.parse.unquote(asset_url).lower()
    return any(urllib.parse.unquote(link).lower() == normalized for link in links)


def acquire_sources(output_dir: Path) -> list[dict[str, object]]:
    if output_dir.exists():
        raise FileExistsError(f"Raw output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, object]] = []
    for event in SOURCE_EVENTS:
        if not _page_links_to_asset(event.page_url, event.asset_url):
            raise ValueError(
                f"B3 release page no longer links its recorded asset: {event.page_url}"
            )
        payload, last_modified = _get(event.asset_url)
        if last_modified is None:
            raise ValueError(
                f"B3 attachment has no Last-Modified timestamp: {event.asset_url}"
            )
        available_at = email.utils.parsedate_to_datetime(last_modified)
        if available_at.tzinfo is None:
            raise ValueError(
                f"B3 attachment timestamp has no timezone: {last_modified}"
            )
        available_at_utc = available_at.astimezone(timezone.utc)
        suffix = Path(urllib.parse.urlparse(event.asset_url).path).suffix.lower()
        if suffix == ".zip" and not payload.startswith(b"PK"):
            raise ValueError(f"B3 attachment is not a ZIP: {event.asset_url}")
        if suffix == ".xlsx" and not payload.startswith(b"PK"):
            raise ValueError(f"B3 attachment is not XLSX: {event.asset_url}")
        filename = f"{event.disclosure_date:%Y%m%d}_{event.stage}{suffix}"
        path = output_dir / filename
        path.write_bytes(payload)
        records.append(
            {
                "disclosure_date": event.disclosure_date.isoformat(),
                "effective_date": event.effective_date.isoformat(),
                "stage": event.stage,
                "page_url": event.page_url,
                "asset_url": event.asset_url,
                "asset_last_modified_http": last_modified,
                "available_at_utc": available_at_utc.isoformat(),
                "filename": filename,
                "bytes": len(payload),
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "contract_version": "B3_INDEX_RELEASE_RAW_SNAPSHOT_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "availability_rule": (
            "Each exact attachment is first available at its preserved HTTP "
            "Last-Modified timestamp; source bytes are never backdated to the "
            "date printed on the associated release page"
        ),
        "files": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return records


def _composition_workbook(path: Path) -> bytes:
    payload = path.read_bytes()
    if path.suffix.lower() == ".xlsx":
        return payload
    candidates: list[bytes] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            if (
                not name.lower().endswith(".xlsx")
                or name.startswith("__MACOSX/")
                or Path(name).name.startswith("._")
            ):
                continue
            workbook = archive.read(name)
            reader = fastexcel.read_excel(workbook)
            if set(INDEXES).issubset(reader.sheet_names):
                candidates.append(workbook)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one index-composition workbook in {path}, found {len(candidates)}"
        )
    return candidates[0]


def parse_composition(path: Path) -> list[Portfolio]:
    workbook = fastexcel.read_excel(_composition_workbook(path))
    portfolios: list[Portfolio] = []
    for index in INDEXES:
        frame = workbook.load_sheet(index, header_row=None).to_polars()
        if frame.width < 5:
            raise ValueError(f"Malformed {index} sheet in {path}")
        columns = frame.columns
        weights: dict[str, float] = {}
        quantities: dict[str, float] = {}
        for row in frame.select(columns[0], columns[3], columns[4]).iter_rows():
            ticker = str(row[0]).strip() if row[0] is not None else ""
            if TICKER_PATTERN.fullmatch(ticker) is None:
                continue
            try:
                quantity = _number(row[1], thousands=True)
                weight = _number(row[2], thousands=False) / 100.0
            except (TypeError, ValueError):
                continue
            if ticker in weights:
                raise ValueError(f"Duplicate {index} ticker {ticker} in {path}")
            weights[ticker] = weight
            quantities[ticker] = quantity
        total = sum(weights.values())
        if len(weights) < 20 or not 0.998 <= total <= 1.002:
            raise ValueError(
                f"Incomplete {index} composition in {path}: "
                f"rows={len(weights)}, total={total:.6f}"
            )
        portfolios.append(Portfolio(index, weights, quantities))
    return portfolios


def _number(value: object, *, thousands: bool) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip()
    if thousands:
        text = text.replace(".", "")
    return float(text.replace(",", "."))


def _load_disclosures(raw_dir: Path) -> list[Disclosure]:
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    disclosures: list[Disclosure] = []
    for record in manifest["files"]:
        source = raw_dir / record["filename"]
        if _sha256(source) != record["sha256"]:
            raise ValueError(f"Source hash mismatch: {source}")
        disclosures.append(
            Disclosure(
                date.fromisoformat(record["disclosure_date"]),
                date.fromisoformat(record["effective_date"]),
                str(record["stage"]),
                datetime.fromisoformat(record["available_at_utc"]),
                source,
            )
        )
    return sorted(disclosures, key=lambda item: item.available_at_utc)


def _first_available_decision(
    available_at_utc: datetime,
    market_dates: list[date],
) -> tuple[date, int] | None:
    local = available_at_utc.astimezone(SAO_PAULO)
    position = bisect.bisect_left(market_dates, local.date())
    if position == len(market_dates):
        return None
    if market_dates[position] == local.date():
        decision = bisect.bisect_right(
            DECISION_TIMES, local.time().replace(tzinfo=None)
        )
        if decision < len(DECISION_TIMES):
            return market_dates[position], decision
        position += 1
        if position == len(market_dates):
            return None
    return market_dates[position], 0


def load_market_dates(cotahist_dir: Path) -> list[date]:
    sources = sorted(cotahist_dir.glob("year=*/equities_daily_*.parquet"))
    if not sources:
        raise FileNotFoundError(f"No parsed COTAHIST files under {cotahist_dir}")
    return sorted(
        pl.scan_parquet(sources)
        .select(pl.col("trade_date").unique())
        .collect()
        .get_column("trade_date")
        .to_list()
    )


def _load_cotahist(cotahist_dir: Path) -> pl.DataFrame:
    sources = sorted(cotahist_dir.glob("year=*/equities_daily_*.parquet"))
    return (
        pl.scan_parquet(sources)
        .select("trade_date", "security_id", "ticker", "close_brl", "volume_brl")
        .collect()
        .sort("trade_date", "ticker")
    )


def _market_state(
    cotahist: pl.DataFrame,
    canonical_ids: set[str],
) -> tuple[
    dict[date, dict[str, str]],
    dict[date, dict[str, float]],
    dict[date, dict[str, float]],
]:
    canonical_tickers: dict[str, tuple[date, str]] = {}
    latest_close: dict[str, float] = {}
    volume_history: dict[str, list[float]] = {}
    identity_by_date: dict[date, dict[str, str]] = {}
    prior_close_by_date: dict[date, dict[str, float]] = {}
    adv_share_by_date: dict[date, dict[str, float]] = {}
    for key, group in cotahist.group_by("trade_date", maintain_order=True):
        current_date = key[0] if isinstance(key, tuple) else key
        identity_by_date[current_date] = {
            ticker: value[1] for ticker, value in canonical_tickers.items()
        }
        prior_close_by_date[current_date] = dict(latest_close)
        prior_adv = {
            security_id: sum(values[-20:]) / min(len(values), 20)
            for security_id, values in volume_history.items()
            if values
        }
        total_adv = sum(prior_adv.values())
        adv_share_by_date[current_date] = (
            {key: value / total_adv for key, value in prior_adv.items()}
            if total_adv > 0.0
            else {}
        )
        for row in group.select(
            "ticker", "security_id", "close_brl", "volume_brl"
        ).to_dicts():
            ticker = str(row["ticker"])
            security_id = str(row["security_id"])
            latest_close[ticker] = float(row["close_brl"])
            if security_id in canonical_ids:
                canonical_tickers[ticker] = (current_date, security_id)
                volume_history.setdefault(security_id, []).append(
                    float(row["volume_brl"])
                )
    return identity_by_date, prior_close_by_date, adv_share_by_date


def _reconstructed_weights(
    quantities: dict[str, float], prior_closes: dict[str, float]
) -> tuple[dict[str, float], float]:
    notionals = {
        ticker: quantity * prior_closes[ticker]
        for ticker, quantity in quantities.items()
        if ticker in prior_closes and quantity > 0.0 and prior_closes[ticker] > 0.0
    }
    total = sum(notionals.values())
    if total <= 0.0:
        return {}, 0.0
    return {ticker: value / total for ticker, value in notionals.items()}, len(
        notionals
    ) / len(quantities)


def _signed_sqrt(value: float) -> float:
    return math.copysign(math.sqrt(abs(value)), value) if value else 0.0


def build_frame(
    disclosures: list[Disclosure],
    market_dates: list[date],
    cotahist: pl.DataFrame,
    canonical_ids: list[str],
    *,
    available_start: date,
    available_end: date,
) -> tuple[pl.DataFrame, dict[str, object]]:
    canonical_set = set(canonical_ids)
    identity, prior_closes, adv_shares = _market_state(cotahist, canonical_set)
    session_pos = {value: index for index, value in enumerate(market_dates)}
    events_by_coordinate: dict[tuple[date, int], Disclosure] = {}
    parsed: dict[Path, dict[str, Portfolio]] = {}
    for event in disclosures:
        coordinate = _first_available_decision(event.available_at_utc, market_dates)
        if coordinate is None:
            continue
        if coordinate in events_by_coordinate:
            raise ValueError(f"Two index disclosures share availability {coordinate}")
        events_by_coordinate[coordinate] = event
        parsed[event.source_asset] = {
            item.index: item for item in parse_composition(event.source_asset)
        }
    current_quantities: dict[str, dict[str, float]] = {}
    frozen_delta: dict[str, dict[str, float]] = {}
    frozen_pressure: dict[str, dict[str, float]] = {}
    preview_weights: dict[str, dict[str, float]] = {}
    preview_quantities: dict[str, dict[str, float]] = {}
    preview_effective: date | None = None
    last_effective_delta: dict[str, dict[str, float]] = {}
    last_effective_date: dict[str, date] = {}
    frames: list[pl.DataFrame] = []
    event_audit: list[dict[str, object]] = []

    for market_date in market_dates:
        if preview_effective == market_date:
            last_effective_delta = {
                index: dict(values) for index, values in frozen_delta.items()
            }
            last_effective_date = {index: market_date for index in INDEXES}
            current_quantities = {
                index: dict(values) for index, values in preview_quantities.items()
            }
            preview_weights = {}
            preview_quantities = {}
            preview_effective = None
            frozen_delta = {}
            frozen_pressure = {}

        current_weights: dict[str, dict[str, float]] = {}
        coverage: dict[str, float] = {}
        for index, quantities in current_quantities.items():
            weights, price_coverage = _reconstructed_weights(
                quantities, prior_closes.get(market_date, {})
            )
            current_weights[index] = weights
            coverage[index] = price_coverage

        day_identity = identity.get(market_date, {})
        inverse_identity = {
            security_id: ticker for ticker, security_id in day_identity.items()
        }
        day_rows: list[dict[str, object]] = []
        for decision_idx in range(len(DECISION_TIMES)):
            event = events_by_coordinate.get((market_date, decision_idx))
            if event is not None:
                portfolios = parsed[event.source_asset]
                if event.stage == "effective":
                    current_quantities = {
                        index: dict(portfolios[index].quantities) for index in INDEXES
                    }
                    current_weights = {
                        index: dict(portfolios[index].weights) for index in INDEXES
                    }
                    coverage = {index: 1.0 for index in INDEXES}
                    if (
                        preview_effective is not None
                        and preview_effective <= market_date
                    ):
                        preview_weights = {}
                        preview_quantities = {}
                        preview_effective = None
                        frozen_delta = {}
                        frozen_pressure = {}
                else:
                    if set(current_weights) != set(INDEXES):
                        raise ValueError(
                            f"Preview {market_date} has no preceding effective composition"
                        )
                    if any(value < 0.98 for value in coverage.values()):
                        raise ValueError(
                            "Current-portfolio prior-close coverage below 98% on "
                            f"{market_date}: {coverage}"
                        )
                    preview_weights = {
                        index: dict(portfolios[index].weights) for index in INDEXES
                    }
                    preview_quantities = {
                        index: dict(portfolios[index].quantities) for index in INDEXES
                    }
                    preview_effective = event.effective_date
                    day_adv = adv_shares.get(market_date, {})
                    frozen_delta = {}
                    frozen_pressure = {}
                    for index in INDEXES:
                        deltas: dict[str, float] = {}
                        pressures: dict[str, float] = {}
                        tickers = set(current_weights[index]) | set(
                            preview_weights[index]
                        )
                        for ticker in tickers:
                            security_id = day_identity.get(ticker)
                            if security_id is None:
                                continue
                            delta = preview_weights[index].get(
                                ticker, 0.0
                            ) - current_weights[index].get(ticker, 0.0)
                            deltas[security_id] = delta
                            scaled = delta / max(day_adv.get(security_id, 0.0), 1e-6)
                            pressures[security_id] = (
                                max(min(scaled, 10.0), -10.0) / 10.0
                            )
                        frozen_delta[index] = deltas
                        frozen_pressure[index] = pressures
                event_audit.append(
                    {
                        "disclosure_date": event.disclosure_date.isoformat(),
                        "effective_date": event.effective_date.isoformat(),
                        "stage": event.stage,
                        "available_at_utc": event.available_at_utc.isoformat(),
                        "available_date": market_date.isoformat(),
                        "decision_idx": decision_idx,
                        "source_asset": event.source_asset.name,
                        "source_sha256": _sha256(event.source_asset),
                        "current_price_coverage": coverage,
                    }
                )

            if not available_start <= market_date <= available_end or set(
                current_weights
            ) != set(INDEXES):
                continue
            for security_id in canonical_ids:
                ticker = inverse_identity.get(security_id)
                row: dict[str, object] = {
                    "available_date": market_date,
                    "decision_idx": decision_idx,
                    "security_id": security_id,
                }
                for index in INDEXES:
                    current_weight = (
                        current_weights[index].get(ticker, 0.0)
                        if ticker is not None
                        else 0.0
                    )
                    preview_available = (
                        preview_effective is not None and index in preview_weights
                    )
                    preview_weight = (
                        preview_weights[index].get(ticker, 0.0)
                        if preview_available and ticker is not None
                        else 0.0
                    )
                    delta = frozen_delta.get(index, {}).get(security_id, 0.0)
                    days_to_effective = (
                        session_pos[preview_effective] - session_pos[market_date]
                        if preview_available
                        else 0
                    )
                    post_age = (
                        session_pos[market_date]
                        - session_pos[last_effective_date[index]]
                        if index in last_effective_date
                        else -1
                    )
                    values = {
                        "current_weight_sqrt": math.sqrt(max(current_weight, 0.0)),
                        "preview_delta_signed_sqrt": _signed_sqrt(delta),
                        "preview_add": float(
                            preview_available
                            and current_weight == 0.0
                            and preview_weight > 0.0
                        ),
                        "preview_delete": float(
                            preview_available
                            and current_weight > 0.0
                            and preview_weight == 0.0
                        ),
                        "preview_pressure": (
                            frozen_pressure.get(index, {}).get(security_id, 0.0)
                            if preview_available
                            else 0.0
                        ),
                        "pre_effective_ramp": (
                            _signed_sqrt(delta)
                            * math.exp(-max(days_to_effective, 0) / 5.0)
                            if preview_available
                            else 0.0
                        ),
                        "post_effective_reversal": (
                            -_signed_sqrt(
                                last_effective_delta.get(index, {}).get(
                                    security_id, 0.0
                                )
                            )
                            * math.exp(-post_age / 10.0)
                            if 1 <= post_age <= 20
                            else 0.0
                        ),
                    }
                    for suffix, value in values.items():
                        name = f"{index.lower()}_{suffix}"
                        row[name] = value
                        row[f"{name}_mask"] = True
                day_rows.append(row)
        if day_rows:
            frames.append(pl.DataFrame(day_rows, infer_schema_length=None))

    if not frames:
        raise ValueError("Index-rebalance builder produced no rows")
    frame = pl.concat(frames, how="vertical", rechunk=True).with_columns(
        pl.col("available_date").cast(pl.Date),
        pl.col("decision_idx").cast(pl.Int8),
        *[pl.col(name).cast(pl.Float32) for name in FEATURES],
        *[pl.col(f"{name}_mask").cast(pl.Boolean) for name in FEATURES],
    )
    return frame, {
        "events": event_audit,
        "rows": frame.height,
        "dates": frame.get_column("available_date").n_unique(),
        "decision_count": frame.get_column("decision_idx").n_unique(),
        "securities": frame.get_column("security_id").n_unique(),
        "feature_valid_rows": {
            name: int(frame.get_column(f"{name}_mask").sum()) for name in FEATURES
        },
    }


def build_sidecar(
    raw_dir: Path,
    cotahist_dir: Path,
    canonical_ids: list[str],
    output_dir: Path,
    *,
    available_start: date,
    available_end: date,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    frame, audit = build_frame(
        _load_disclosures(raw_dir),
        load_market_dates(cotahist_dir),
        _load_cotahist(cotahist_dir),
        canonical_ids,
        available_start=available_start,
        available_end=available_end,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    data_path = output_dir / "index_rebalance.parquet"
    frame.write_parquet(data_path, compression="zstd", statistics=True)
    manifest: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "raw_manifest": str((raw_dir / "manifest.json").resolve()),
        "raw_manifest_sha256": _sha256(raw_dir / "manifest.json"),
        "cotahist_source": str(cotahist_dir.resolve()),
        "availability_rule": (
            "Exact preview and definitive bytes are first used at the first model "
            "decision strictly after the attachment's preserved HTTP Last-Modified "
            "timestamp; a known preview becomes the current portfolio at its stated "
            "effective-session open"
        ),
        "identity_rule": (
            "Dated B3 ticker maps only through COTAHIST observations available before "
            "that session to a permanent canonical security_id"
        ),
        "normalization_rule": (
            "Current weights are causally reconstructed from the last official "
            "theoretical quantities and prior B3 closes; weights use square roots, "
            "deltas signed square roots, and pressure prior-20 ADV-share scaling "
            "with a fixed clip"
        ),
        "features": list(FEATURES),
        "cadence": "intraday",
        "available_start": available_start.isoformat(),
        "available_end": available_end.isoformat(),
        "output_file": data_path.name,
        "output_sha256": _sha256(data_path),
        **audit,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def _canonical_ids(assignments: Path) -> list[str]:
    return (
        pl.read_parquet(assignments)
        .sort("security_id")
        .get_column("security_id")
        .cast(pl.String)
        .to_list()
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Acquire or build the PIT B3 index-rebalance source sidecar"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--out", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-dir", type=Path, required=True)
    build.add_argument("--cotahist-dir", type=Path, required=True)
    build.add_argument("--assignments", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--available-start", type=date.fromisoformat, required=True)
    build.add_argument("--available-end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.command == "acquire":
        records = acquire_sources(args.out)
        print(f"Acquired {len(records)} B3 release assets to {args.out}")
    else:
        manifest = build_sidecar(
            args.raw_dir,
            args.cotahist_dir,
            _canonical_ids(args.assignments),
            args.out,
            available_start=args.available_start,
            available_end=args.available_end,
        )
        print(
            f"Wrote {manifest['rows']:,} rows across {manifest['dates']} dates to {args.out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
