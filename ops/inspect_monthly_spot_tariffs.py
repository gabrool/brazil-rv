"""Extract candidate historical tariff coordinates from the saved TX responses."""

from decimal import Decimal
from hashlib import sha256
import io
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from brazil_rv.v2.artifacts import write_json_atomic
from brazil_rv.v2.data_repair import binding

PROJECT = Path(__file__).resolve().parents[1]
NS = {"fee": "urn:bvmf.181.01.xsd", "file": "urn:bvmf.052.01.xsd"}


def main():
    run = json.loads((PROJECT / "docs/v2_economic_data_scaling_run.json").read_text())
    root = Path(run["root"]) / "tariff_coverage"
    receipts = json.loads((root / "monthly/retrieval.json").read_text())["rows"]
    initial = json.loads((root / "tx_20200102_receipt.json").read_text())
    receipts.append(dict(initial, status="recovered", request_date="2020-01-02"))
    rows = []
    for receipt in receipts:
        if receipt["status"] != "recovered":
            continue
        path = Path(receipt["path"])
        assert binding(path)["sha256"] == receipt["sha256"]
        with ZipFile(path) as outer:
            assert len(outer.namelist()) == 1
            member = outer.namelist()[0]
            inner_bytes = outer.read(member)
        with ZipFile(io.BytesIO(inner_bytes)) as inner:
            assert inner.namelist() == ["BVBG.072.01.xml"]
            payload = inner.read("BVBG.072.01.xml")
        xml = ET.fromstring(payload)
        reports = xml.findall(".//fee:EqtsFeePblcInf", NS)
        assert len(reports) == 1
        report = reports[0]
        candidates = []
        for market in report.findall("fee:FeeInf", NS):
            if market.findtext("fee:FeeGrpMkt", namespaces=NS) != "1":
                continue
            for fee in market.findall("fee:Fee", NS):
                if fee.findtext("fee:DayTradInd", namespaces=NS) != "false":
                    continue
                for tier in fee.findall("fee:TierAndCost", NS):
                    if tier.findtext("fee:FeeTp", namespaces=NS) == "1":
                        candidates.extend(tier.findall("fee:CostInf", NS))
        assert len(candidates) == 1
        cost = candidates[0]
        rate = cost.find("fee:FeeCostVal", NS)
        rows.append(
            dict(
                archive=binding(path),
                member=member,
                inner_sha256=sha256(inner_bytes).hexdigest(),
                xml_sha256=sha256(payload).hexdigest(),
                xml_bytes=len(payload),
                request_date=receipt["request_date"],
                creation_time=xml.findtext(".//file:CreDtAndTm", namespaces=NS),
                report_date=report.findtext(
                    "fee:RptParams/fee:RptDtAndTm/fee:Dt", namespaces=NS
                ),
                valid_from=report.findtext("fee:VldtyPrd/fee:FrDt", namespaces=NS),
                valid_to=report.findtext("fee:VldtyPrd/fee:ToDt", namespaces=NS),
                fee_value=rate.text,
                currency=rate.attrib["Ccy"],
                trading_bps=str(Decimal(rate.text) * 10000),
                categories=[
                    x.text for x in cost.findall("fee:ClntCtgyDtls/fee:ClntCtgy", NS)
                ],
            )
        )
    rows.sort(key=lambda row: row["valid_from"])
    write_json_atomic(
        root / "monthly/candidates.json",
        dict(
            producer=binding(Path(__file__)),
            rows=rows,
            status="Saved-source coordinates only; code-list and chronology qualification precedes admission.",
        ),
    )
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "request_date",
                        "creation_time",
                        "valid_from",
                        "valid_to",
                        "trading_bps",
                    )
                }
                for r in rows
            ]
        )
    )


if __name__ == "__main__":
    main()
