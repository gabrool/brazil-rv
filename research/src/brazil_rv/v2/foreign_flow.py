"""Historical B3 published MTD totals, with explicit difference semantics."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time

import numpy as np
import polars as pl

from .contract import DEVELOPMENT_END
from .round5_market import SAO_PAULO, first_available_decision


def parse_mtd(text: str, publication_date: date) -> dict:
    """Read the first MTD participation table, never the previous-month table.

    B3 has used both Brazilian and US thousands separators in these PDFs.
    The amounts in this table are integer thousands of BRL in either locale.
    """
    match = re.search(r"Dados acumulados[^\n]+dia\s+(\d{2}/\d{2}/\d{4})", text)
    if match is None:
        raise ValueError("no dated MTD participation table")
    tail = text[match.end() :]
    header, marker, table = tail.partition("Investidor Estrangeiro")
    if not marker or not re.search(r"Compras.*Mil.*Vendas.*Mil", header):
        raise ValueError("foreign totals lack the thousand-BRL unit header")
    number = r"(\d+(?:[.,]\d{3})*)"
    row = re.match(rf"\s*{number}\s+[\d.,]+\s+{number}\s+[\d.,]+", table)
    if row is None:
        raise ValueError("unrecognized foreign MTD amount row")
    reference = datetime.strptime(match[1], "%d/%m/%Y").date()
    if reference > publication_date or publication_date > DEVELOPMENT_END:
        raise ValueError("foreign source chronology escapes historical boundary")
    buy, sell = (int(re.sub(r"[.,]", "", value)) for value in row.groups())
    return {
        "publication_date": publication_date,
        "reference_date": reference,
        "buy_mtd_thousand_brl": buy,
        "sell_mtd_thousand_brl": sell,
        "net_mtd_billion_brl": (buy - sell) / 1e6,
        "methodology": "post_20220520_published_participation"
        if publication_date >= date(2022, 5, 20)
        else "transition_20220401_20220519"
        if publication_date >= date(2022, 4, 1)
        else "pre_20220401_published_participation",
    }


def published_differences(observations: list[dict], sessions: list[date]) -> list[dict]:
    """Difference successive comparable publications, keeping unsupported gaps.

    A new month starts at zero only for its actual first B3 session. Five-session
    flow sums five separately observed consecutive changes, possibly across a
    verified month reset. A revised same-reference total updates the baseline
    but is not a new daily flow. No future publication repairs an earlier gap.
    """
    positions = {day: i for i, day in enumerate(sessions)}
    previous = None
    changes: dict[date, float] = {}
    result = []
    for row in sorted(observations, key=lambda x: x["publication_date"]):
        day, publication = row["reference_date"], row["publication_date"]
        if publication > DEVELOPMENT_END or day > publication:
            raise ValueError("foreign source escapes the development chronology")
        if day not in positions:
            continue
        index = positions[day]
        month_start = index == 0 or sessions[index - 1].month != day.month
        changed_method = (
            previous is not None and previous["methodology"] != row["methodology"]
        )
        same_month = previous is not None and (
            previous["reference_date"].year,
            previous["reference_date"].month,
        ) == (day.year, day.month)
        consecutive = (
            previous is not None
            and positions.get(previous["reference_date"]) == index - 1
        )
        flow = None
        if not changed_method:
            if month_start and (previous is None or previous["reference_date"] < day):
                flow = row["net_mtd_billion_brl"]
            elif same_month and consecutive:
                flow = row["net_mtd_billion_brl"] - previous["net_mtd_billion_brl"]
        if changed_method:
            changes.clear()
        if flow is not None:
            changes[day] = flow
        else:
            changes.pop(day, None)
        five = sessions[max(0, index - 4) : index + 1]
        flow5 = (
            sum(changes[d] for d in five)
            if len(five) == 5 and all(d in changes for d in five)
            else None
        )
        result.append(
            {
                **row,
                "foreign_flow_1": flow,
                "foreign_flow_5": flow5,
                "foreign_flow_month_reset": float(month_start),
                "foreign_flow_methodology_change": float(changed_method),
                "interval_sessions": index - positions[previous["reference_date"]]
                if previous is not None
                else None,
                "semantics": "published_total_difference",
                # A dated BDI is not proof of an intraday publication. Its
                # date-only bound admits it at the next eligible decision.
                "available_at": datetime.combine(
                    publication, time(23, 59, 59), SAO_PAULO
                ).astimezone(UTC),
            }
        )
        previous = row
    return result


def decision_panel(observations: list[dict], sessions: list[date]):
    """Latest published state and reference age; invalid new releases clear it."""
    names = (
        "foreign_flow_1",
        "foreign_flow_5",
        "foreign_flow_month_reset",
        "foreign_flow_methodology_change",
    )
    values = np.zeros((len(sessions), len(names)), np.float32)
    valid = np.zeros_like(values, bool)
    ages = np.full_like(values, -1)
    records = published_differences(observations, sessions)
    records.sort(key=lambda x: x["available_at"])
    positions = {day: i for i, day in enumerate(sessions)}
    cursor, latest = 0, None
    for i in range(len(sessions)):
        while (
            cursor < len(records)
            and first_available_decision(records[cursor]["available_at"], sessions) <= i
        ):
            latest = records[cursor]
            cursor += 1
        if latest is not None:
            for j, name in enumerate(names):
                if latest[name] is not None:
                    values[i, j] = latest[name]
                    valid[i, j] = True
                    ages[i, j] = i - positions[latest["reference_date"]]
    return names, values, valid, ages, pl.DataFrame(records)
