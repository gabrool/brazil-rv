"""Independent bounded reducer arithmetic, index workbooks and XML clocks."""

import io
import json
import re
import zipfile
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import fastexcel
import numpy as np
import polars as pl

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding, bound_json
from brazil_rv.v2.round5_derived import verified

PROJECT = Path(__file__).resolve().parents[1]


def workbooks(path):
    if path.suffix.lower() != ".zip":
        return [(path.name, fastexcel.read_excel(path))]
    with zipfile.ZipFile(path) as z:
        return [
            (name, fastexcel.read_excel(z.read(name)))
            for name in z.namelist()
            if name.lower().endswith((".xlsx", ".xls"))
            and not name.startswith("__MACOSX/")
            and not Path(name).name.startswith("._")
        ]


def main():
    start = perf_counter()
    pointer_path = PROJECT / "docs/v2_economic_data_scaling_run.json"
    run = json.loads(pointer_path.read_text(encoding="utf8"))
    admission = bound_json(run["remaining_auxiliaries"])
    root = Path(run["remaining_auxiliaries"]["path"]).parent
    output = root / "source_audit"
    output.mkdir(exist_ok=True)
    (
        output / ("executed_" + binding(Path(__file__))["sha256"][:12] + ".py")
    ).write_bytes(Path(__file__).read_bytes())
    store = Path(admission["parent"]["root"])
    sessions = np.load(store / "date_index.npy").astype(object).tolist()
    units, odd_checks = [], 0
    scopes = admission["scopes"]
    reducer_path = output / "reducer_results.json"
    if reducer_path.exists():
        saved = json.loads(reducer_path.read_text(encoding="utf8"))
        units, odd_checks = saved["activity"], saved["oddlot_comparisons"]
        scopes = []
    for scope in scopes:
        folder = root / scope["successor"]
        frames = {
            n: pl.read_parquet(folder / (n + ".parquet"))
            for n in ["cash", "quantities", "snapshots", "nonregular"]
        }
        maps = {
            n: {(r["source_trade_date"], r["isin"]): r for r in frame.to_dicts()}
            for n, frame in frames.items()
        }
        outputs = {
            n: {
                r["date"]: r
                for r in pl.read_parquet(folder / (n + ".parquet")).to_dicts()
            }
            for n in ["microstructure", "options"]
        }
        link = scope["link"]
        boundary = link["effective_index"]

        def row(name, t):
            isin = scope["predecessor"] if t < boundary else scope["successor"]
            return maps[name].get((sessions[t], isin), {})

        checked, supported = 0, 0

        def compare(day, name, value, age):
            nonlocal checked, supported
            family = (
                "microstructure"
                if name in ["avg_trade_size_20", "after_hours_volume_share_5"]
                else "options"
            )
            r = outputs[family].get(day, {})
            got = r.get(name)
            if value is None:
                assert got is None, (day, name, got)
            else:
                np.testing.assert_allclose(got, value, rtol=3e-15, atol=1e-12)
                assert r[name + "_age_sessions"] == age
                supported += 1
            checked += 1

        def total(values):
            return sum(values) if all(v is not None for v in values) else None

        for t in range(
            max(boundary, link["known_index"]),
            sessions.index(date.fromisoformat(scope["end"])) + 1,
        ):
            day = sessions[t]
            stock = [row("cash", d) for d in range(t - 20, t)]
            brl, trades = (
                total([r.get("volume_brl") for r in stock]),
                total([r.get("trades") for r in stock]),
            )
            compare(
                day,
                "avg_trade_size_20",
                brl / trades if brl is not None and trades and trades > 0 else None,
                1,
            )
            qcall, qput = [], []
            after, reported = [], []
            for d in range(t - 20, t):
                q, oi = row("quantities", d), row("snapshots", d)
                for name, dest in [("call_quantity", qcall), ("put_quantity", qput)]:
                    dest.append(
                        q.get(name, 0.0 if oi.get("listed_series", 0) > 0 else None)
                    )
                if d >= t - 5:
                    pr = row("nonregular", d)
                    quantity, regular = pr.get("quantity"), pr.get("regular_quantity")
                    nonregular = pr.get("nonregular_quantity")
                    value = (
                        nonregular
                        if nonregular is not None
                        else quantity - regular
                        if quantity is not None and regular is not None
                        else None
                    )
                    after.append(
                        value
                        if value is not None
                        and quantity is not None
                        and 0 <= value <= quantity
                        else None
                    )
                    reported.append(quantity)
            quantity = total([r.get("quantity") for r in stock])
            call, put = total(qcall), total(qput)
            compare(
                day,
                "option_to_stock_volume_20",
                (call + put) / quantity
                if call is not None and put is not None and quantity and quantity > 0
                else None,
                1,
            )
            call5, put5 = total(qcall[-5:]), total(qput[-5:])
            compare(
                day,
                "put_call_volume_ratio_5",
                put5 / call5 if call5 and call5 > 0 and put5 is not None else None,
                1,
            )
            a, b = total(after), total(reported)
            compare(
                day,
                "after_hours_volume_share_5",
                a / b if a is not None and b and b > 0 else None,
                1,
            )
            oi = row("snapshots", t - 1)
            assert not oi.get("oi_all_listed_observed", False), (
                "new full-OI support needs independent delta arithmetic"
            )
            for name in [
                "put_call_oi_log_ratio",
                "delta_oi_to_volume_1",
                "uncovered_call_share",
            ]:
                compare(day, name, None, None)
            call, put = oi.get("call_oi"), oi.get("put_oi")
            compare(
                day,
                "observed_series_put_call_oi_log_ratio",
                np.log(put / call) if call and put and call > 0 and put > 0 else None,
                2,
            )
            compare(
                day,
                "observed_series_oi_coverage",
                oi["oi_observed_series"] / oi["listed_series"]
                if oi.get("listed_series", 0) > 0
                else None,
                2,
            )
        odd = pl.read_parquet(folder / "odd_source.parquet")
        oddmap = {(r["source_trade_date"], r["isin"]): r for r in odd.to_dicts()}

        def share(t, decision):
            isin = scope["predecessor"] if t < boundary else scope["successor"]
            r = oddmap.get((sessions[t], isin))
            if not r or r["available_date"] > decision:
                return None
            regular, odd = r["regular_volume_brl"], r["odd_lot_volume_brl"]
            return (
                odd / (regular + odd)
                if regular is not None
                and odd is not None
                and regular >= 0
                and odd >= 0
                and regular + odd > 0
                else None
            )

        for r in pl.read_parquet(folder / "oddlot.parquet").to_dicts():
            t = sessions.index(r["date"]) - 1
            current, prior = share(t, r["date"]), share(t - 5, r["date"])
            expected = [
                current,
                current - prior if current is not None and prior is not None else None,
            ]
            for name, value in zip(
                ["oddlot_volume_share", "oddlot_volume_share_change_5"], expected
            ):
                assert (r[name] is None) == (value is None)
                if value is not None:
                    np.testing.assert_allclose(r[name], value, rtol=0, atol=1e-15)
                    assert r[name + "_age_sessions"] == 1
                odd_checks += 1
        units.append(
            {
                "isin": scope["successor"],
                "field_comparisons": checked,
                "supported": supported,
                "complete_oi_stays_unknown": True,
            }
        )

    write_json_atomic(
        output / "reducer_results.json",
        {"activity": units, "oddlot_comparisons": odd_checks},
    )
    index = bound_json(admission["parent_families"]["rebalance"]["source_manifest"])
    index_sources = bound_json(index["snapshots"])
    expected = pl.read_parquet(verified(index_sources["output"]))
    progress = output / "workbook_progress.json"
    workbook_results = (
        json.loads(progress.read_text(encoding="utf8")) if progress.exists() else []
    )
    receipts = []
    # Independently read literal stock code, theoretical quantity and percent
    # from each original worksheet, without the normalized portfolio parser.
    for event in index_sources["events"]:
        path = verified(event["composition_source"])
        receipts.append(event["composition_source"])
        if any(
            r["date"] == event["disclosure_date"]
            and r["stage"] == event["stage"]
            and r["source"] == event["composition_source"]
            for r in workbook_results
        ):
            continue
        actual = {}
        for member, workbook in workbooks(path):
            for sheet in workbook.sheet_names:
                name = {"IBRX": "IBXX"}.get(sheet.upper(), sheet.upper())
                if name not in ["IBOV", "IBXX", "SMLL"]:
                    continue
                for row in (
                    workbook.load_sheet(sheet, header_row=None, dtype_coercion="coerce")
                    .to_polars()
                    .iter_rows()
                ):
                    if len(row) < 5 or not re.fullmatch(
                        r"[A-Z0-9]{4}\d{1,2}", str(row[0])
                    ):
                        continue
                    key = name, row[0]
                    assert key not in actual
                    actual[key] = float(row[4]) / 100, float(row[3])
        subset = expected.filter(
            (pl.col("disclosure_date") == date.fromisoformat(event["disclosure_date"]))
            & (pl.col("stage") == event["stage"])
        )
        assert len(actual) == len(subset), (
            event["disclosure_date"],
            len(actual),
            len(subset),
        )
        for r in subset.to_dicts():
            weight, quantity = actual[(r["index"], r["ticker"])]
            np.testing.assert_allclose(weight, r["weight_fraction"], rtol=0, atol=2e-17)
            assert quantity == r["quantity"]
        workbook_results.append(
            {
                "date": event["disclosure_date"],
                "stage": event["stage"],
                "rows": len(actual),
                "source": event["composition_source"],
                "available_at": event["available_at"],
                "clock_evidence": event["availability_evidence"],
            }
        )
        write_json_atomic(progress, workbook_results)
    write_json_atomic(output / "workbook_results.json", workbook_results)
    # Original nested ZIP headers, not HTTP metadata, independently confirm the
    # selected vintage and availability on both sides of the two renames.
    activity = bound_json(admission["parent_families"]["options"]["source_manifest"])
    inventory = bound_json(activity["source_inventory"])
    by_name = {Path(r["path"]).name: r for r in inventory}
    headers = []
    for day in ["2023-10-24", "2023-10-25", "2024-11-14", "2024-11-18"]:
        receipt = bound_json(by_name[day + ".json"])
        receipts.append(by_name[day + ".json"])
        cutoff = datetime.fromisoformat(
            receipt["available_date"] + "T15:45:00"
        ).replace(tzinfo=ZoneInfo("America/Sao_Paulo"))
        for kind in ["IN", "PR"]:
            record = receipt["raw_files"][kind]
            path = verified(record)
            receipts.append(record)
            with zipfile.ZipFile(path) as outer:
                children = [r for r in outer.infolist() if not r.is_dir()]
                assert len(children) == 1
                with zipfile.ZipFile(io.BytesIO(outer.read(children[0]))) as inner:
                    versions = []
                    for member in inner.namelist():
                        if member.endswith(".xml"):
                            with inner.open(member) as f:
                                head = f.read(8192)
                            match = re.search(rb"<(?:\w+:)?CreDtAndTm>([^<]+)", head)
                            timestamp = datetime.fromisoformat(match[1].decode())
                            if timestamp.tzinfo is None:
                                timestamp = timestamp.replace(
                                    tzinfo=ZoneInfo("America/Sao_Paulo")
                                )
                            versions.append((timestamp, member))
            chosen = max(v for v in versions if v[0] <= cutoff)
            old = receipt[kind.lower()]
            assert chosen[1] == old["selected_member"]
            assert chosen[0] == datetime.fromisoformat(old["selected_creation_time"])
            assert chosen[0].date() == date.fromisoformat(day) and chosen[0].hour >= 15
            headers.append(
                {
                    "date": day,
                    "kind": kind,
                    "versions": len(versions),
                    "selected": chosen[1],
                    "created": chosen[0].isoformat(),
                    "available_decision": cutoff.isoformat(),
                }
            )
    # Correct the eligibility/history attribution for index fields: unlike the
    # other archives, the old index producer omitted inactive rows altogether.
    control = pl.read_parquet(root / "index_eligibility_only.parquet")
    changed = pl.read_parquet(root / "index_amended.parquet")
    old = pl.read_parquet(verified(admission["parent_families"]["rebalance"]["data"]))
    joined = changed.join(
        control, on=["date", "isin"], how="left", suffix="_control"
    ).join(
        old.select("date", "isin").with_columns(pl.lit(True).alias("old_row")),
        on=["date", "isin"],
        how="left",
    )
    newrows = joined.filter(pl.col("old_row").is_null())
    index_attribution = {
        "new_rows": len(newrows),
        "eligibility_only_new_rows": newrows["index_pressure_control"].count(),
        "new_pressure_values_changed_by_history_or_weight_routing": int(
            (newrows["index_pressure"] != newrows["index_pressure_control"])
            .fill_null(False)
            .sum()
        ),
        "new_pressure_values_without_old_adv": newrows[
            "index_pressure_control"
        ].null_count(),
        "note": "Supersedes the index eligibility/history counts in the initial consumer receipt; arrays and successful tensor comparisons are unchanged.",
    }
    write_json_atomic(output / "index_attribution.json", index_attribution)
    result = {
        "status": "passed",
        "input": run["remaining_auxiliaries"],
        "bounded_activity_oracles": units,
        "oddlot_comparisons": odd_checks,
        "index_original_workbooks": len(workbook_results),
        "index_original_rows": sum(r["rows"] for r in workbook_results),
        "workbooks": binding(output / "workbook_results.json"),
        "original_xml_headers": headers,
        "index_attribution": index_attribution,
        "receipts": receipts,
        "seconds": perf_counter() - start,
        "code": binding(Path(__file__)),
        "limitations": [
            "B3 XML creation times bound existence, not retrieval by a particular broker; preserved single-vintage revision share is unknown.",
            "Original option series/cash numerical raw reconciliation outside these normalized bounded histories remains a separate upstream boundary.",
            "The index weight difference is a pressure proxy with price drift, not inferred passive fund order flow.",
            "No model forecast or held-out consumer was read.",
        ],
    }
    write_json_atomic(output / "manifest.json", result)
    run["remaining_auxiliary_source_audit"] = binding(output / "manifest.json")
    write_json_atomic(pointer_path, run)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "index_original_workbooks",
                    "index_original_rows",
                    "oddlot_comparisons",
                    "index_attribution",
                    "seconds",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
