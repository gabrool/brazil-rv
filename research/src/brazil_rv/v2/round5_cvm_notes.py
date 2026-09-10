"""Narrow own-period capital-note extraction for a separately reviewed audit."""

from __future__ import annotations

from datetime import date
import re

from .round5_cvm import normalized

MONTHS = (
    "janeiro",
    "fevereiro",
    "marco",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)


def propose_paid_in_note(
    text: str, reference: date, paid_in: dict, treasury: dict
) -> dict | None:
    """Locate literal share-count proposals for review, without source admission.

    This intentionally handles only prose with an explicit own-period date,
    named ON/PN quantities and zero treasury in the source capital table. Tables,
    lot units, attributed/subsidiary capital and ambiguous periods need separate
    review. The source page and the filing's other capital/treasury notes must be
    reviewed before accepting a proposal. A non-match is not an unavailable
    disposition or a consumer mask. No treasury quantity is inferred here.
    """
    if any(treasury.get(cls) != 0 for cls in ("ON", "PN") if paid_in[cls] > 0):
        return None
    key = re.sub(r"\s+", " ", normalized(text))
    if re.search(r"\blote de mil acoes\b|\bacoes.{0,45}\bem milhares\b", key):
        return None
    day, month, year = reference.day, reference.month, reference.year
    own_date = (
        rf"\bem\s+(?:0?{day}/0?{month}/{year}|"
        rf"0?{day}\s+de\s+{MONTHS[month - 1]}\s+de\s+{year})\b"
    )
    candidates = []
    for match in re.finditer(own_date, key):
        # Stay within the same sentence. Decimal/grouping periods are not stops.
        passage = re.split(
            r"(?<!\d)\.(?:\s|$)", key[match.end() : match.end() + 850], 1
        )[0]
        declaration = re.search(
            r"\bcapital (?:social|subscrito|integralizado)\b.{0,300}?"
            r"\b(?:representad[oa]|dividid[oa]|compost[oa])\b",
            passage,
        )
        if declaration is None or re.search(
            r"\b(?:controlada|subsidiaria|investida)\b|ponderad", passage
        ):
            continue
        prelude = passage[: declaration.end()]
        if any(int(y) != year for y in re.findall(r"\b(?:19|20)\d{2}\b", prelude)):
            continue
        quantities = {}
        for cls, label in (("ON", "ordinarias"), ("PN", "preferenciais")):
            values = {
                int(value.replace(".", ""))
                for value in re.findall(
                    rf"(?<![\d.,])((?:\d{{1,3}}(?:\.\d{{3}})+|\d+))\s+acoes\s+{label}\b",
                    passage[declaration.end() :],
                )
            }
            if len(values) == 1:
                quantities[cls] = values.pop()
            elif not values and paid_in[cls] == 0:
                quantities[cls] = 0
            else:
                break
        if len(quantities) == 2 and any(quantities.values()):
            candidates.append(
                {
                    "paid_in_shares": quantities,
                    "normalized_evidence": key[match.start() : match.end()] + passage,
                }
            )
    unique = {tuple(c["paid_in_shares"].items()) for c in candidates}
    return candidates[0] if len(unique) == 1 else None
