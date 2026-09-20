"""Independently read saved monthly XMLs and bind the historical cost source scope."""

from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
from time import perf_counter
from xml.dom import minidom
from zipfile import ZipFile

import numpy as np

from brazil_rv.execution.spot_costs import MonthlySpotTariff
from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.corporate_replay import load_corporate_replay
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]


def children(node, name):
    return [
        x
        for x in node.childNodes
        if x.nodeType == x.ELEMENT_NODE and x.localName == name
    ]


def one(node, name):
    values = children(node, name)
    assert len(values) == 1, (name, len(values))
    return values[0]


def content(node):
    return "".join(x.data for x in node.childNodes if x.nodeType == x.TEXT_NODE).strip()


def main():
    started = perf_counter()
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "tariff_coverage"
    output = root / "qualification.json"
    assert not output.exists()
    shutil.copyfile(__file__, root / "qualification_executed.py")
    candidate = json.loads((root / "monthly/candidates.json").read_text())
    _, dates = load_corporate_replay(
        run["enat_settlement_terms"]["path"], run["enat_settlement_terms"]["sha256"]
    )
    rows, tariffs, late = [], [], []
    for row in candidate["rows"]:
        archive = Path(row["archive"]["path"])
        assert binding(archive) == row["archive"]
        with ZipFile(archive) as outer:
            assert len(outer.infolist()) == 1
            nested = outer.read(outer.infolist()[0])
        with ZipFile(io.BytesIO(nested)) as inner:
            payload = inner.read("BVBG.072.01.xml")
        assert sha256(payload).hexdigest() == row["xml_sha256"]
        xml = minidom.parseString(payload)
        reports = xml.getElementsByTagNameNS("urn:bvmf.181.01.xsd", "EqtsFeePblcInf")
        assert len(reports) == 1
        report = reports[0]
        creation = content(
            xml.getElementsByTagNameNS("urn:bvmf.052.01.xsd", "CreDtAndTm")[0]
        )
        validity = one(report, "VldtyPrd")
        start, end = content(one(validity, "FrDt")), content(one(validity, "ToDt"))
        params = one(report, "RptParams")
        assert content(one(params, "Frqcy")) == "MNTH"
        report_date = content(one(one(params, "RptDtAndTm"), "Dt"))
        normal = []
        for group in children(report, "FeeInf"):
            if content(one(group, "FeeGrpMkt")) == "1":
                normal += [
                    fee
                    for fee in children(group, "Fee")
                    if content(one(fee, "DayTradInd")) == "false"
                ]
        assert len(normal) == 1
        tier = [
            t
            for t in children(normal[0], "TierAndCost")
            if content(one(t, "FeeTp")) == "1"
        ]
        assert len(tier) == 1 and not children(tier[0], "TierInitlVal")
        costs = children(tier[0], "CostInf")
        assert len(costs) == 1
        fee = one(costs[0], "FeeCostVal")
        categories = [
            content(one(c, "ClntCtgy")) for c in children(costs[0], "ClntCtgyDtls")
        ]
        assert categories == ["1", "2"] and fee.getAttribute("Ccy") == "BRL"
        rate = Decimal(content(fee)) * Decimal(10000)
        assert Decimal(".2") <= rate <= Decimal(".5")
        assert (creation, start, end, report_date, rate) == (
            row["creation_time"],
            row["valid_from"],
            row["valid_to"],
            row["report_date"],
            Decimal(row["trading_bps"]),
        )
        # Creation is a lower bound, not proof of first web publication. Use a
        # conservative next-equity-date convention AND the requested archive date.
        next_date = dates[
            np.searchsorted(dates, np.datetime64(creation[:10]), side="right")
        ]
        available = max(str(next_date), row["request_date"], report_date)
        tariff = MonthlySpotTariff(
            start, end, available, float(rate), row["xml_sha256"]
        )
        tariffs.append(asdict(tariff))
        rows.append(dict(**row, available_date=available))
        if available > start:
            late.append(
                dict(valid_from=start, creation=creation, available_date=available)
            )
    assert len(rows) == 50 and len(late) == 3
    calendar_rows = []
    for day in dates[
        (dates >= np.datetime64("2016-07-18")) & (dates <= np.datetime64("2024-12-30"))
    ]:
        matched = [
            t
            for t in tariffs
            if max(t["valid_from"], t["available_date"]) <= str(day) <= t["valid_to"]
        ]
        assert len(matched) <= 1
        if day >= np.datetime64("2021-02-02"):
            status, rate, source = "177_2020_fixed", 0.5, "177/2020;017/2023"
        elif matched:
            status, rate, source = (
                "saved_monthly",
                matched[0]["trading_bps"],
                matched[0]["source_sha256"],
            )
        else:
            status, rate, source = (
                "unrecovered_or_not_yet_known_bound",
                None,
                "018/2013;061/2013",
            )
        calendar_rows.append(
            dict(date=str(day), status=status, trading_bps=rate, source=source)
        )
    source_pages = {
        "b3_082_2009": [1, 2],
        "b3_018_2013": list(range(1, 9)),
        "b3_061_2013": [1, 2, 3, 4],
        "b3_144_2015": [1, 2, 3],
        "b3_120_2016": [1, 2, 3],
        "b3_011_2017": [1, 2, 3],
        "b3_101_2018": [1, 2, 3, 4],
        "b3_tariff_catalogue_2024": [2, 75, 77, 78, 79],
        "b3_tariff_schema_presentation_2025": [14],
    }
    sources = {
        name: dict(
            receipt=binding(root / f"{name}_receipt.json"),
            original=binding(root / f"{name}.pdf"),
            visually_reviewed_pages=pages,
        )
        for name, pages in source_pages.items()
    }
    result = dict(
        producer=binding(Path(__file__)),
        candidate=binding(root / "monthly/candidates.json"),
        tariffs=tariffs,
        monthly_sources=rows,
        calendar=calendar_rows,
        late_creation=late,
        counts={
            s: sum(r["status"] == s for r in calendar_rows)
            for s in sorted({r["status"] for r in calendar_rows})
        },
        sources=sources,
        visual_pages=sum(map(len, source_pages.values())),
        interpretation=(
            "Historical fee rates come from the 50 own XML payloads and dated 018/2013/061/2013 notices, never the 2025 examples. The recovered 2024 catalogue describes the successor .02 schema; its field descriptions and the public 2025 technical slide (group1=spot, false=normal, type1=trading, decimal rate units) corroborate the mapping of matching .01 tags. This is an explicit cross-version semantic inference, not recovery of an own-vintage external code list. The external list on the public 2021 index requires login and was not accessed. The unique normal trading cost applies equally to both printed category codes; no category-specific fund discount is selected. Historical web-publication time/revision completeness remain unknown. Creation-date and first archive-request date gates avoid observed backdating but do not establish historical availability completeness. No 2025/2026 market consumer is read."
        ),
        missing_contract="0.5bp primary upper tariff bound; 0.2bp separate lower limiting bound, only unrecovered/not-yet-known regular trading dates. 0.2 is not an observed finite-ADTV rate. Ordinary clearing2.75bp before2021-02-02,2.5 thereafter. Auction0.7bp remains a separate execution hypothesis.",
        custody_contract="082/2009/101/2018 old brackets with literal <=300000 before2019 and <300000 thereafter; this source-wording boundary is not claimed to prove an intended economic rule change. 144/2015,120/2016,011/2017,101/2018 dated active resident maintenance 7.59/8.02,8.18/8.65,8.40/8.88,8.78/9.28 at <=5000/>5000. Active one-account/custodian hypothesis includes zero-stock month end. 177/2020 active exemption starts2021-02-02; prior accepted custody sources reused. Client payment/own last-mark valuation remain hypotheses; corporate physical admission still separate.",
        seconds=perf_counter() - started,
    )
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("counts", "late_creation", "visual_pages", "seconds")
            }
        )
    )


if __name__ == "__main__":
    main()
