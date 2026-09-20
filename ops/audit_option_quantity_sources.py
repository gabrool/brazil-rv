"""Original fixed-width option quantities and the auxiliary cash-source join.

Only option records are decoded from the 15 bound 2010--2024 ZIPs. Cash rows
reuse the separately audited equity normalization, without another quote census.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
from time import perf_counter
import zipfile

import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json

PROJECT = Path(__file__).resolve().parents[1]


def main():
    start = perf_counter()
    pointer = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer.read_text())
    families = bound_json(run["remaining_auxiliaries"])
    root = Path(
        families["parent_families"]["options"]["source_manifest"]["path"]
    ).parent.parent
    source_manifest = root / "cotahist_option_volume/manifest.json"
    manifest = json.loads(source_manifest.read_text())
    master = json.loads((root / "b3_identity_axis_audit.json").read_text())[
        "master_file"
    ]
    assert binding(Path(master["path"]))["sha256"] == master["sha256"]
    bounds = defaultdict(list)
    for row in pl.read_parquet(master["path"]).to_dicts():
        bounds[row["isin"]].append(
            (
                str(row["first_date"]).replace("-", ""),
                str(row["last_date"]).replace("-", ""),
            )
        )
    output = Path(run["root"]) / "option_quantity_sources"
    output.mkdir(exist_ok=True)
    (
        output / ("executed_" + binding(Path(__file__))["sha256"][:12] + ".py")
    ).write_bytes(Path(__file__).read_bytes())
    results = []
    for source in manifest["files"]:
        path = Path(source["path"])
        saved = output / (path.stem + ".json")
        if saved.exists():
            prior = json.loads(saved.read_text())
            assert prior["source"]["sha256"] == source["source_sha256"]
            assert prior["output"]["sha256"] == source["output_sha256"]
            assert prior["mismatches"] == 0
            results.append(prior)
            continue
        assert binding(path)["sha256"] == source["source_sha256"]
        assert 2010 <= int(path.stem[-4:]) <= 2024
        totals = defaultdict(lambda: [0, 0, 0])
        currencies, factors = Counter(), Counter()
        used, maximum = 0, None
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.upper().endswith(".TXT")]
            assert len(names) == 1
            with archive.open(names[0]) as handle:
                for raw in handle:
                    if raw[:2] != b"01" or raw[24:27] not in (b"070", b"080"):
                        continue
                    assert len(raw.rstrip(b"\r\n")) == 245
                    isin = raw[230:242].decode("ascii")
                    day = raw[2:10].decode("ascii")
                    if not any(a <= day <= b for a, b in bounds.get(isin, [])):
                        continue
                    quantity = int(raw[152:170])
                    values = totals[(day, isin)]
                    values[raw[24:27] == b"080"] += quantity
                    values[2] += 1
                    used += 1
                    currencies[raw[52:56].decode("ascii").strip()] += 1
                    factors[raw[210:217].decode("ascii")] += 1
                    if maximum is None or quantity > maximum["quantity"]:
                        maximum = {
                            "date": day,
                            "isin": isin,
                            "ticker": raw[12:24].decode("ascii").strip(),
                            "quantity": quantity,
                            "source_row": raw.decode("ascii").strip(),
                        }
        frame_path = root / "cotahist_option_volume" / source["output"]
        assert binding(frame_path)["sha256"] == source["output_sha256"]
        frame = pl.read_parquet(frame_path)
        assert frame.height == len(totals) == source["rows"]
        for row in frame.to_dicts():
            key = (row["source_trade_date"].strftime("%Y%m%d"), row["isin"])
            assert totals.pop(key) == [
                row[k]
                for k in ["call_quantity", "put_quantity", "traded_option_series"]
            ], key
        assert not totals
        receipt = {
            "year": path.stem[-4:],
            "source": {"path": str(path), "sha256": source["source_sha256"]},
            "output": binding(frame_path),
            "source_option_rows": used,
            "aggregate_rows": frame.height,
            "comparisons": frame.height * 3,
            "currencies": dict(currencies),
            "price_factors": dict(factors),
            "largest_printed_quantity": maximum,
            "mismatches": 0,
        }
        results.append(receipt)
        write_json_atomic(output / (path.stem + ".json"), receipt)
    # Independent join/units check from the already original-source-audited
    # cash-equity annual tables to this family's separate normalized cash axis.
    cash_manifest = json.loads((root / "b3_cash_axis/manifest.json").read_text())
    cash = pl.read_parquet(root / "b3_cash_axis/cash.parquet")
    cash_checks = 0
    for source in cash_manifest["sources"]:
        frame = pl.read_parquet(source["path"])
        year = int(Path(source["path"]).stem[-4:])
        assert 2010 <= year <= 2024
        old = cash.filter(pl.col("source_trade_date").dt.year() == year)
        mapped = {
            (r["trade_date"], r["isin"], r["ticker"]): r
            for r in frame.filter(pl.col("market_type") == 10).to_dicts()
        }
        for row in old.to_dicts():
            raw = mapped[(row["source_trade_date"], row["isin"], row["ticker"])]
            for dst, src in [
                ("quantity", "quantity"),
                ("trades", "trades"),
                ("volume_brl", "volume_brl"),
                ("quote_factor", "quote_factor"),
            ]:
                assert row[dst] == raw[src], (
                    year,
                    row["isin"],
                    dst,
                    row[dst],
                    raw[src],
                )
                cash_checks += 1
            # The audited equity normalization already applies the quote factor.
            assert row["close_brl"] == raw["close_brl"]
            cash_checks += 1
    receipt = {
        "status": "passed",
        "source_manifest": binding(source_manifest),
        "master": master,
        "annual": results,
        "option_source_rows": sum(x["source_option_rows"] for x in results),
        "aggregate_rows": sum(x["aggregate_rows"] for x in results),
        "option_comparisons": sum(x["comparisons"] for x in results),
        "cash_axis_rows": cash.height,
        "cash_comparisons": cash_checks,
        "cash_original_quote_census_repeated": False,
        "cash_manifest": binding(root / "b3_cash_axis/manifest.json"),
        "seconds": perf_counter() - start,
        "code": binding(Path(__file__)),
        "production_or_heldout_changes": False,
        "limitations": "One preserved annual COTAHIST vintage; historical revision share unknown. Option quantities are printed units, not lots times 100. Existing source-date/next-decision clocks are preserved.",
    }
    write_json_atomic(output / "manifest.json", receipt)
    run["option_quantity_source_audit"] = binding(output / "manifest.json")
    write_json_atomic(pointer, run)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in [
                    "status",
                    "option_source_rows",
                    "aggregate_rows",
                    "option_comparisons",
                    "cash_axis_rows",
                    "cash_comparisons",
                    "seconds",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
