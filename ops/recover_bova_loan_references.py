"""Recover published average quotes from the already bound hedge archives.

This is source recovery, not admission into an accepted store or policy cache.
"""

from datetime import datetime
import json
from pathlib import Path
import zipfile

import polars as pl

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic
from brazil_rv.v2.bova11 import BOVA11_BDI_CODES, BOVA11_ISIN


PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads(
        (PROJECT / "docs/v2_round5_economic_sources.json").read_text()
    )["bova11"]
    source_root = Path(pointer["root"])
    manifest_path = source_root / "manifest.json"
    if sha256_file(manifest_path) != pointer["manifest_sha256"]:
        raise ValueError("accepted hedge manifest changed")
    manifest = json.loads(manifest_path.read_text())
    root = Path(
        json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())[
            "root"
        ]
    )
    output = root / "bova_loan_reference_audit"
    output.mkdir(exist_ok=False)
    rows = []
    for source in manifest["sources"]:
        path = Path(source["path"])
        if int(path.stem[-4:]) > 2024:
            raise PermissionError("development hedge archives only")
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"bound source changed: {path}")
        found = 0
        with zipfile.ZipFile(path) as archive, archive.open(source["txt_member"]) as f:
            for line in f:
                if line[:2] != b"01" or line[12:24].strip() != b"BOVA11":
                    continue
                if (
                    line[230:242].decode("ascii") != BOVA11_ISIN
                    or int(line[24:27]) != 10
                    or line[39:49].split()[0] != b"CI"
                    or line[10:12].decode("ascii") not in BOVA11_BDI_CODES
                ):
                    continue
                if len(line.rstrip(b"\r\n")) != 245:
                    raise ValueError("malformed exact hedge row")
                day = datetime.strptime(line[2:10].decode("ascii"), "%Y%m%d").date()
                if day.year > 2024:
                    raise PermissionError("protected hedge observation")
                factor = int(line[210:217])
                if factor <= 0:
                    raise ValueError("published hedge quote factor is not positive")
                average, close = [
                    int(line[a:b]) / (100 * factor) for a, b in ((95, 108), (108, 121))
                ]
                if average <= 0 or close <= 0:
                    raise ValueError("nonpositive exact hedge quote")
                rows.append((day, BOVA11_ISIN, average, close))
                found += 1
        if found != source["bova11_row_count"]:
            raise ValueError(f"hedge row census changed: {path}")
    quotes = pl.DataFrame(
        rows,
        schema=[
            ("trade_date", pl.Date),
            ("isin", pl.String),
            ("average_brl", pl.Float64),
            ("close_brl", pl.Float64),
        ],
        orient="row",
    ).sort("trade_date")
    if quotes["trade_date"].n_unique() != len(quotes):
        raise ValueError("ambiguous published hedge date")
    accepted_path = source_root / manifest["data_file"]
    if sha256_file(accepted_path) != manifest["data_sha256"]:
        raise ValueError("accepted hedge close series changed")
    if not quotes.select("trade_date", "close_brl").equals(
        pl.read_parquet(accepted_path)
    ):
        raise ValueError("recovered hedge quotes do not reproduce accepted closes")
    path = output / "loan_reference_quotes.parquet"
    quotes.write_parquet(path)
    report = {
        "source": pointer,
        "sources": manifest["sources"],
        "rows": quotes.height,
        "first_date": str(quotes["trade_date"][0]),
        "last_date": str(quotes["trade_date"][-1]),
        "data": {"path": str(path), "sha256": sha256_file(path)},
        "accepted_close_equality": True,
        "heldout_accessed": False,
        "status": "recovered_published_averages_not_admitted",
        "availability": "report-day quotes; loan input must use a prior-session publication, never the current close",
    }
    write_json_atomic(output / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
