"""Trace selected extreme stored ratios into accepted own-version source books."""

import json
import hashlib
import io
import zipfile
from pathlib import Path
import numpy as np
import polars as pl
from pypdf import PdfReader
from brazil_rv.v2.round5_cvm import (
    load_accounts,
    attach_viewer_accounts,
    apply_account_unit_dispositions,
)
from brazil_rv.v2.round5_cvm_xml import original_accounts
from brazil_rv.v2.round5_cvm_capital import load_capital, load_capital_dispositions

project = Path(__file__).resolve().parents[2]
pointer = json.loads(
    (project / "docs/v2_round5_cvm_final_acceptance.json").read_text()
)["family_manifest"]
manifest = json.loads(Path(pointer["path"]).read_text())
source = Path(manifest["source_root"])
identity = pl.read_parquet(Path(pointer["path"]).parent / "identity.parquet")
names = ["BRODPVACNOR4", "BRTTENACNOR0", "BRAUREACNOR9", "BRBKBRACNOR4"]
issuers = (
    identity.filter(pl.col("isin").is_in(names))
    .select("isin", "cnpj", "cvm_code")
    .unique()
)
documents = load_accounts(source, set(issuers["cnpj"]))
dispositions = load_capital_dispositions(
    source / "capital_source_dispositions.json", documents
)
errors = []
for d in documents:
    viewer = source / "originals" / d["id"]
    if not d.get("accounts") and (viewer / "manifest.json").exists():
        try:
            attach_viewer_accounts(d, viewer)
        except (ValueError, KeyError) as error:
            errors.append({"id": d["id"], "stage": "viewer", "error": str(error)})
    archive = source / "original_zips" / d["id"] / "source.zip"
    if (not d.get("accounts") or d.get("shares") is None) and archive.exists():
        try:
            p = original_accounts(d, archive)
        except (ValueError, KeyError) as error:
            errors.append(
                {
                    "id": d["id"],
                    "error": str(error),
                    "has_csv_accounts": bool(d.get("accounts")),
                }
            )
            p = {}
        if not d.get("accounts"):
            d.update(p)
        if d.get("shares") is None:
            d["shares"] = p.get("shares")
    capital = source / "capital" / d["id"]
    if d.get("shares") is None and (capital / "manifest.json").exists():
        d["shares"] = load_capital(d, capital)["shares"]
    if d["id"] in dispositions:
        d["shares"] = dispositions[d["id"]]["shares"]
apply_account_unit_dispositions(source, documents)
root = Path(
    json.loads((project / "docs/v2_round7_inputs.json").read_text())["store"]["root"]
)
dates = np.load(root / "date_index.npy")
isins = np.load(root / "isin_index.npy")
active = np.load(root / "active.npy", mmap_mode="r")
frame = pl.read_parquet(Path(pointer["path"]).parent / "fundamentals.parquet")
day_index, name_index = np.nonzero(active)
eligible = pl.DataFrame(
    {"date": dates[day_index], "isin": isins[name_index]}
).with_columns(pl.col("date").cast(pl.Date))
active_frame = frame.join(eligible, on=["date", "isin"], how="inner")
candidates = []
for field, threshold in [
    ("earnings_yield_ttm", 10),
    ("book_to_market", 100),
    ("gross_profitability", 10),
    ("revenue_growth_yoy", 100),
]:
    suspect = active_frame.filter(pl.col(field).abs() > threshold)
    candidates.append(
        dict(
            field=field,
            absolute_threshold=threshold,
            stock_days=suspect.height,
            issuers=suspect["isin"].n_unique(),
            interpretation="Triage only; these thresholds do not establish an error or authorize masking.",
        )
    )
footprints = []
for name, field, lower in [
    ("BRBKBRACNOR4", "revenue_growth_yoy", 100),
    ("BRODPVACNOR4", "gross_profitability", 100),
    ("BRTTENACNOR0", "earnings_yield_ttm", 10),
    ("BRAUREACNOR9", "earnings_yield_ttm", 10),
]:
    allowed = dates[active[:, list(isins).index(name)]].astype(object).tolist()
    rows = frame.filter(
        (pl.col("isin") == name)
        & pl.col("date").is_in(allowed)
        & (pl.col(field) > lower)
    )
    footprints.append(
        dict(
            isin=name,
            field=field,
            screen_threshold=lower,
            active_stock_days=rows.height,
            dates=rows["date"].cast(pl.String).to_list(),
            minimum=rows[field].min(),
            maximum=rows[field].max(),
        )
    )
filings = []
for doc, pages in [("70549", [94]), ("134143", [2, 120]), ("125769", [2, 57])]:
    archive = source / "original_zips" / doc / "source.zip"
    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist() if n.endswith(".pdf"))
        body = z.read(member)
    reader = PdfReader(io.BytesIO(body))
    filings.append(
        dict(
            document=doc,
            archive=str(archive),
            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            pdf_member=member,
            pdf_sha256=hashlib.sha256(body).hexdigest(),
            pages=[
                dict(pdf_page=p, text=reader.pages[p - 1].extract_text()) for p in pages
            ],
        )
    )
output = {
    "issuers": issuers.to_dicts(),
    "documents": documents,
    "audit_parse_errors": errors,
    "scope": "Selected own-version account/capital trace with accepted dispositions; unparsed XML does not replace existing CSV accounts.",
    "active_outlier_footprints": footprints,
    "filing_evidence": filings,
    "candidate_screens": candidates,
}
path = Path(
    "D:/quant-data/b3/interim/round7_postmortem_20260913/cvm_outlier_sources.json"
)
path.write_text(json.dumps(output, default=str, indent=2))
print(json.dumps({"documents": len(documents), "output": str(path)}))
