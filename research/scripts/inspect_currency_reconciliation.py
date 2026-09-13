"""Propose exact own-note currency reconciliations for human/source review."""

import argparse
import json
import re
from pathlib import Path

from brazil_rv.v2.round5_cvm import normalized

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
results = []
for path in args.root.glob("filing_evidence/*/evidence.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["status"] != "extracted":
        continue
    pages = json.loads((path.parent / "pages.json").read_text(encoding="utf-8"))
    accounts = record.get("accounts") or {}
    for factor in (1000, 0.001):
        matches = []
        for p, text in enumerate(pages):
            clean = normalized(text)
            if p < 12 or not re.search(r"milhares\s+de\s+reais|em\s+r\$\s*mil", clean):
                continue
            if "codigo da conta" in clean[:600] or "descricao da conta" in clean[:600]:
                continue
            for basis, book in accounts.items():
                for field in (
                    "assets",
                    "equity",
                    "revenue",
                    "net_income",
                    "parent_income",
                    "gross_profit",
                    "cash_flow",
                ):
                    if field not in book:
                        continue
                    value = book[field]["value"] * factor / 1000
                    if abs(value) < 1000 or abs(value - round(value)) > 0.0001:
                        continue
                    printed = f"{abs(round(value)):,}".replace(",", ".")
                    hits = list(
                        re.finditer(
                            r"(?<![\d.,])" + re.escape(printed) + r"(?![\d.,])", text
                        )
                    )
                    if hits:
                        hit = hits[0]
                        matches.append(
                            dict(
                                page=p + 1,
                                basis=basis,
                                field=field,
                                value=book[field]["value"],
                                heading=text[:400],
                                excerpt=text[
                                    max(0, hit.start() - 160) : hit.end() + 180
                                ],
                            )
                        )
        unique = {(m["field"], m["value"]) for m in matches}
        if len(unique) >= 2:
            results.append(
                dict(
                    document=record["document"],
                    multiplier=factor,
                    matches=matches,
                    archive=record["archive"],
                    archive_sha256=record["archive_sha256"],
                )
            )
(args.root / "currency_note_candidates.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(
    json.dumps(
        {
            "candidate_documents": len(results),
            "ids": [r["document"]["id"] for r in results],
        }
    )
)
