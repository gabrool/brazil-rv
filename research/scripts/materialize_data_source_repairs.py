"""Materialize the reviewed exact-filing decisions, never magnitude heuristics."""

import argparse
import json
import pickle
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from brazil_rv.v2.round5_cvm import sha256
from brazil_rv.v2.round5_cvm_xml import validate_original_identity

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
root = args.root
documents = {
    d["id"]: d for d in pickle.load((root / "accepted_documents.pkl").open("rb"))
}
# These IDs were reviewed against their own dated/basis-specific statements and
# notes. The text matcher proposes candidates; it does not admit new IDs here.
currency_ids = set(
    "100841 102051 102053 102601 104154 105870 106245 10642 106749 10727 107648 109780 17879 17913 20229 5264 57623 5768 64416 6728 67679 7054 70549 74380 74832 7636 76875 76910 82496 82499 83778 86198 87073 89113 91677 97958 98998".split()
)
proposals = {
    r["document"]["id"]: r
    for r in json.loads(
        (root / "currency_note_candidates.json").read_text(encoding="utf-8")
    )
}


def identity(identifier):
    return {
        k: str(documents[identifier][k])
        for k in ("id", "cnpj", "cvm_code", "kind", "reference", "version")
    }


def evidence(identifier):
    folder = root / "filing_evidence" / identifier
    extracted = json.loads((folder / "evidence.json").read_text(encoding="utf-8"))
    archive = Path(extracted["archive"])
    if sha256(archive) != extracted["archive_sha256"]:
        raise ValueError("Reviewed archive changed")
    with zipfile.ZipFile(archive) as z:
        member = next(
            n
            for n in z.namelist()
            if n.startswith("FormularioDemonstracaoFinanceira") and n.endswith(".xml")
        )
        validate_original_identity(
            documents[identifier], z, ET.fromstring(z.read(member))
        )
    return [
        {"path": str(p), "sha256": sha256(p)} for p in (archive, folder / "pages.json")
    ]


currency = []
for identifier in sorted(currency_ids, key=int):
    proposal = proposals[identifier]
    matches = proposal["matches"]
    if identifier == "5264":
        matches = [
            m for m in matches if m["page"] == 101
        ]  # current consolidated segment totals
    currency.append(
        {
            "document": identity(identifier),
            "multiplier": 1000,
            "accounts_before": json.loads(
                json.dumps(documents[identifier]["accounts"], default=str)
            ),
            "reason": "Own-period monetary amounts reconcile to the issuer's statements/notes explicitly in thousands of BRL; the submitted unit-scale heading is inconsistent. Preserve signs, periods, bases, shares and the original receipt.",
            "reconciliation": matches,
            "evidence": evidence(identifier),
        }
    )

# Paid-in and treasury counts are reviewed separately. Counts never come from a
# later filing, a weighted-average EPS denominator, or a guessed 1,000 multiplier.
capital_decisions = {
    "55631": (
        (316684999, 0),
        (9337178, 0),
        [2, 67, 68],
        "Own note18 capital and treasury tables reconcile316684999 minus9337178; the rounded front paid-in row already excludes treasury.",
    ),
    "58976": (
        (317178517, 0),
        (9498058, 0),
        [2, 79, 80],
        "Own version3 notes give exact paid-in and treasury counts; replace rounded printed quantities with their same-filing exact values.",
    ),
    "60672": (
        (317896418, 0),
        (9498058, 0),
        [2, 76, 77],
        "Own note17 gives exact paid-in and treasury quantities for2016Q3; preserve those rather than the rounded front table.",
    ),
    "63235": (
        (317896418, 0),
        (9498058, 0),
        [3, 112, 113, 115],
        "Own version2 capital, treasury and dividend-denominator reconciliations agree; the front table's paid-in quantity already excludes treasury.",
    ),
    "58890": (
        (317178517, 0),
        (9498058, 0),
        [2, 79, 80],
        "Own note18 capital and treasury rollforwards reconcile317178517 paid-in minus9498058. The front table both mislabels units and puts already-net shares in its paid-in row.",
    ),
    "58924": (
        (317178517, 0),
        (9498058, 0),
        [2, 79, 80],
        "Own version2 note18 independently reconciles317178517 paid-in minus9498058; do not double-subtract treasury from the front table.",
    ),
    "63187": (
        (317896418, 0),
        (9498058, 0),
        [3, 112, 113],
        "Own note18 capital and treasury rollforwards reconcile317896418 paid-in minus9498058. The front table both mislabels units and puts already-net shares in its paid-in row.",
    ),
    "71393": (
        (1105826145, 0),
        (13842004, 0),
        [3, 104, 105],
        "Own notes25.1 and25.3 reconcile absolute paid-in and treasury shares, including the completed PN-to-ON conversion; the front-table thousand scale is misdeclared.",
    ),
    "83884": (
        (11057992, 7313964),
        (0, 167896),
        [2, 77, 79, 83],
        "Own version2 notes independently reconcile paid-in counts and unweighted available shares. The front table reports net shares as paid-in, causing a second treasury subtraction.",
    ),
    "5523": (
        (197461211, 183792282),
        (0, 0),
        [3, 40],
        "Own-date shareholder table sums to exact ON/PN counts; the rounded unit-labelled capital table understates both classes.",
    ),
    "14984": (
        (451669063, 349996554),
        (411, 1542258),
        [3, 164, 166],
        "Own note29a sums PN classes A+B; note29 treasury table gives parent-company ON411 and PN1542258. Its consolidated treasury includes subsidiary holdings and is a different basis.",
    ),
    "20227": (
        (161371285, 0),
        (0, 0),
        [2, 35],
        "Own-date capital rollforward and prose report161371285 ON; source table reports zero treasury. Its quantity-scale label is inconsistent with those absolute counts.",
    ),
    "74525": (
        (560000000, 0),
        (0, 0),
        [2, 57],
        "Own note states the split had already taken effect by2017-12-31 and560000000 ON at2018-03-31; source table reports zero treasury.",
    ),
    "83778": (
        (11057992, 7313964),
        (0, 167896),
        [2, 77, 79, 83],
        "Own-date capital note gives paid-in counts. Note28 separately gives unweighted available ON11057992 and PN32861+7113207, implying treasury167896. Do not subtract treasury twice from the already-net rounded front table or use its weighted average.",
    ),
    "89113": (
        (1200000000, 0),
        (0, 0),
        [2, 67],
        "Own-date note15 shareholder table explicitly totals1200000000 ON held100% by Caixa; book equity/BVPS corroborates the unit reconciliation.",
    ),
    "123449": (
        (497018000, 0),
        (0, 0),
        [2, 114],
        "Own2022 note23 capital rollforward labels quantities in thousands and ends497018; source table treasury is zero. No2023 filing is used.",
    ),
    "125769": (
        (1000000000, 0),
        (0, 0),
        [2, 57],
        "Own note19.1 explicitly labels shares in units and totals1000000000 ON; shareholder rows sum100%, with zero source treasury.",
    ),
    "134143": (
        (498298000, 0),
        (135000, 0),
        [2, 120, 121],
        "Own note23 spells out498298000 shares and separately reconciles treasury85000+50000=135000; retain both capital and repurchase information.",
    ),
}
capital = []
for identifier, (paid, treasury, pages, reason) in capital_decisions.items():
    capital.append(
        {
            "document": identity(identifier),
            "disposition": "reconciled",
            "paid_in_shares": dict(zip(("ON", "PN"), paid)),
            "treasury_shares": dict(zip(("ON", "PN"), treasury)),
            "pages": pages,
            "reason": reason,
            "evidence": evidence(identifier),
        }
    )
for name, rows in (
    ("account_unit_dispositions", currency),
    ("capital_source_dispositions", capital),
):
    (root / f"{name}.json").write_text(
        json.dumps({"documents": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
print(
    json.dumps(
        {"currency_corrections": len(currency), "capital_corrections": len(capital)}
    )
)
