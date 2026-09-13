"""Compact exact-filing note candidates for source review, never admission."""

import argparse
import json
import re
from datetime import date
from pathlib import Path

from brazil_rv.v2.round5_cvm import normalized
from brazil_rv.v2.round5_cvm_notes import propose_paid_in_note


parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
summaries = []
for path in args.root.glob("filing_evidence/*/evidence.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["status"] != "extracted":
        summaries.append(
            {
                "id": record["document"]["id"],
                "status": record["status"],
                "error": record["error"],
            }
        )
        continue
    text_path = path.parent / "pages.json"
    texts = json.loads(text_path.read_text(encoding="utf-8"))
    shares = record["accepted_shares"] or {"ON": None, "PN": None}
    proposals, matches = [], []
    counts = {str(int(v * 1000)): code for code, v in shares.items() if v and v < 5e7}
    for page, text in enumerate(texts):
        proposal = propose_paid_in_note(
            text, date.fromisoformat(record["document"]["reference"]), shares
        )
        if proposal and proposal["paid_in_shares"] != shares:
            proposals.append({"page": page + 1, **proposal})
        clean = normalized(text)
        if page > 20 and "capital social" in clean:
            for match in re.finditer(r"\b\d{1,3}(?:\.\d{3}){2,}\b", clean):
                number = match.group().replace(".", "")
                if number in counts:
                    matches.append(
                        {
                            "page": page + 1,
                            "class": counts[number],
                            "passage": clean[
                                max(0, match.start() - 250) : match.end() + 320
                            ],
                        }
                    )
    currency = []
    for page, text in enumerate(texts):
        clean = normalized(text)
        if page > 15 and "milhares de reais" in clean:
            currency.append({"page": page + 1, "heading": clean[:330]})
    summary_accounts = {
        b: {
            k: a["value"]
            for k, a in book.items()
            if k in {"assets", "equity", "revenue", "net_income"}
        }
        for b, book in (record.get("accounts") or {}).items()
    }
    summaries.append(
        {
            "id": record["document"]["id"],
            "reference": record["document"]["reference"],
            "fields": record["fields"],
            "status": "extracted",
            "shares": shares,
            "proposals": proposals,
            "thousandfold_literal": matches,
            "accounts": summary_accounts,
            "notes_thousands": currency[:1],
        }
    )
(args.root / "unit_review_summary.json").write_text(
    json.dumps(summaries, ensure_ascii=True, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "records": len(summaries),
            "note_proposals": sum(bool(r.get("proposals")) for r in summaries),
            "literal_thousandfold": sum(
                bool(r.get("thousandfold_literal")) for r in summaries
            ),
        }
    )
)
