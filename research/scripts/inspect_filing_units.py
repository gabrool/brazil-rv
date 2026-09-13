"""Exact-filing unit triage: preserve originals and extract their own evidence."""

import argparse
import io
import json
import pickle
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from pypdf import PdfReader

from brazil_rv.v2.round5_cvm import fetch, normalized, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ids", nargs="+")
    args = parser.parse_args()
    root = args.root
    project = Path(__file__).resolve().parents[2]
    pointer = json.loads(
        (project / "docs/v2_round5_cvm_final_acceptance.json").read_text()
    )["family_manifest"]
    source = Path(json.loads(Path(pointer["path"]).read_text())["source_root"])
    with (root / "accepted_documents.pkl").open("rb") as handle:
        documents = {d["id"]: d for d in pickle.load(handle)}
    candidates = json.loads((root / "scale_candidates.json").read_text())
    store = Path(
        json.loads((project / "docs/v2_round7_inputs.json").read_text())["store"][
            "root"
        ]
    )
    dates = np.load(store / "date_index.npy").astype(object)
    isins = np.load(store / "isin_index.npy")
    d, n = np.nonzero(np.load(store / "active.npy", mmap_mode="r"))
    active = pl.DataFrame(
        {"date": dates[d].tolist(), "isin": isins[n]},
        schema={"date": pl.Date, "isin": pl.String},
    )
    identity = pl.read_parquet(Path(pointer["path"]).parent / "identity.parquet").join(
        active, on=["date", "isin"]
    )
    spans = {
        k: (f["date"].min(), f["date"].max())
        for k, f in identity.partition_by("cnpj", as_dict=True).items()
    }
    wanted = {}
    for candidate in candidates:
        for endpoint in ("before", "after"):
            doc = documents[candidate[endpoint]["id"]]
            span = spans.get((doc["cnpj"],))
            # A discontinuity can be caused by either understated OR overstated
            # units. Inspect both exact originals; neither direction is a fix.
            if (
                span
                and doc["reference"].year >= span[0].year - 4
                and doc["receipt"] <= span[1]
            ):
                wanted.setdefault(doc["id"], set()).update(
                    c["field"] for c in candidate["changes"]
                )
    for identifier in ("125769", "70549", "134143", "106749", "123449"):
        wanted.setdefault(identifier, set()).add("prior_audit")
    if args.ids:
        wanted = {i: wanted.get(i, {"manual_review"}) for i in args.ids}
    evidence_root = root / "filing_evidence"
    evidence_root.mkdir(exist_ok=True)

    def inspect(identifier):
        destination = evidence_root / identifier
        destination.mkdir(exist_ok=True)
        output = destination / "evidence.json"
        if output.exists():
            cached = json.loads(output.read_text(encoding="utf-8"))
            if not cached.get("error", "").startswith("UnicodeEncodeError"):
                return cached
        doc = documents[identifier]
        record = {
            "document": {
                k: str(doc[k])
                for k in ("id", "cnpj", "cvm_code", "reference", "kind", "version")
            },
            "fields": sorted(wanted[identifier]),
            "accepted_shares": doc.get("shares"),
            "accounts": doc.get("accounts"),
        }
        archive = source / "original_zips" / identifier / "source.zip"
        try:
            if not archive.exists():
                archive = destination / "source.zip"
                if not archive.exists():
                    url = (
                        "https://www.rad.cvm.gov.br/ENETCONSULTA/frmDownloadDocumento.aspx?CodigoInstituicao=1&NumeroSequencialDocumento="
                        + identifier
                    )
                    body = fetch(url, attempts=2)
                    if not zipfile.is_zipfile(io.BytesIO(body)):
                        raise ValueError("exact original unavailable: non-ZIP response")
                    archive.write_bytes(body)
            with zipfile.ZipFile(archive) as outer:
                pdf_member = next(
                    (n for n in outer.namelist() if n.lower().endswith(".pdf")), None
                )
                if pdf_member is None:
                    raise ValueError("exact original has no embedded PDF")
                body = outer.read(pdf_member)
            reader = PdfReader(io.BytesIO(body))
            texts = [p.extract_text() or "" for p in reader.pages]
            (destination / "pages.json").write_text(
                json.dumps(texts, ensure_ascii=False), encoding="utf-8"
            )
            # Report source excerpts, not automatic corrections from magnitudes.
            selected = []
            for p, text in enumerate(texts):
                clean = normalized(text)
                if (
                    (
                        "capital social" in clean
                        and ("acoes" in clean or "acao" in clean)
                    )
                    or ("unidade" in clean and "milhares" in clean)
                    or (p < 10)
                ):
                    selected.append({"page": p + 1, "text": text[:13000]})
            record.update(
                status="extracted",
                archive=str(archive),
                archive_sha256=sha256(archive),
                pdf_member=pdf_member,
                pages=len(texts),
                excerpts=selected,
            )
        except Exception as exc:
            record.update(status="unresolved", error=f"{type(exc).__name__}: {exc}")
        output.write_text(
            json.dumps(record, default=str, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    print(json.dumps({"candidate_documents": len(wanted)}), flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(inspect, sorted(wanted, key=int))):
            results.append(
                {k: v for k, v in result.items() if k not in {"accounts", "excerpts"}}
            )
            if (i + 1) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "completed": i + 1,
                            "status": dict(Counter(r["status"] for r in results)),
                        }
                    ),
                    flush=True,
                )
    if not args.ids:
        (root / "filing_unit_inventory.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "completed": len(results),
                "status": dict(Counter(r["status"] for r in results)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
