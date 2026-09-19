"""Bind the money-only-date audit to dated B3 clearing/custody receipts."""

import json
from pathlib import Path

import numpy as np

from brazil_rv.v2.artifacts import sha256_file, write_json_atomic


PROJECT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads(
        (PROJECT / "docs/v2_economic_data_scaling_run.json").read_text()
    )
    root = Path(pointer["root"])
    sources = json.loads((root / "clearing_calendar_sources.json").read_text())
    by_name = {row["id"]: row for row in sources}
    for row in sources:
        assert sha256_file(Path(row["path"])) == row["sha256"]
    cash = json.loads((PROJECT / "docs/v2_cash_calendar_audit.json").read_text())
    dates = np.load(cash["calendar"]["path"], allow_pickle=False).astype(
        "datetime64[D]"
    )
    assert sha256_file(Path(cash["calendar"]["path"])) == cash["calendar"]["sha256"]
    rows = []
    for date in cash["omitted_monetary_dates"]:
        year = int(date[:4])
        source = by_name[
            "b3_calendar_" + ("2021_2022" if year in (2021, 2022) else str(year))
        ]
        if year == 2016:
            status = "equity_closed_clearing_not_independently_resolved"
            rule = "2016 news table proves trading closure; detailed clearing circular remains unrecovered"
            pages = []
        elif year == 2017 and date[5:] != "12-29":
            status = "split_clearing_era_requires_specific_equity_confirmation"
            rule = "111/2016-DP p1 names Camara BMFBOVESPA for local closures but does not separately state the old Camara de Acoes; do not equate the two"
            pages = [1, 2]
        else:
            status = "no_additional_equity_delivery_day"
            pages = {
                2017: [1, 2],
                2018: [1, 2, 3],
                2019: [1, 2, 3, 4],
                2020: [1, 2, 3, 4],
                2021: [2, 4, 5],
                2022: [3, 4],
                2023: [3, 4],
                2024: [3, 4],
            }[year]
            rule = "dated national/year-end or Sao Paulo local table closes equity clearing or central-depository movements; OTC/FX exceptions are different markets"
        assert np.datetime64(date) not in dates
        previous = int(np.searchsorted(dates, np.datetime64(date))) - 1
        rows.append(
            {
                "date": date,
                "status": status,
                "rule": rule,
                "source_sha256": source["sha256"],
                "source_id": source["id"],
                "pdf_pages": pages,
                "previous_equity_session": str(dates[previous]),
                "next_equity_session": str(dates[previous + 1]),
            }
        )
    controls = []
    # The June 2020 dated notice expressly retains Corpus Christi and opens the
    # original July/November holidays after their extraordinary anticipation.
    for value, expected in [
        ("2020-06-11", False),
        ("2020-07-09", True),
        ("2020-11-20", True),
        ("2021-01-25", False),
        ("2021-07-09", False),
        ("2022-01-25", True),
        ("2023-01-25", True),
        ("2023-11-20", True),
        ("2024-11-20", False),
    ]:
        actual = bool(np.datetime64(value) in dates)
        assert actual == expected
        controls.append({"date": value, "equity_session": actual})
    term_spans = {}
    for term in (30, 63, 126):
        span = (dates[term:] - dates[:-term]).astype(int)
        term_spans[str(term)] = {
            "min_calendar_days": int(span.min()),
            "max_calendar_days": int(span.max()),
        }
        assert span.max() < 365 * 2
    destination = root / "clearing_calendar_audit.json"
    if destination.exists():
        raise FileExistsError("completed calendar audit must not be repeated")
    write_json_atomic(
        destination,
        {
            "calendar": cash["calendar"],
            "sources": sources,
            "money_only_dates": rows,
            "controls": controls,
            "contract_term_calendar_spans": term_spans,
            "changed_quote_or_model_arrays": 0,
            "changed_settlement_indices": 0,
            "conclusion": "cash interest uses the corrected monetary interval panel; do not mechanically add all 23 monetary-only dates to stock or loan settlement. Twenty dates have direct clearing/custody closure evidence; three older dates retain specific source ambiguity.",
            "accrual_boundary": "B3-session day count retained as explicit convention; closed equity delivery does not alone prove every loan-rent day-count rule. D0 registration versus D1 electronic accrual and dated tariff endpoints still require their separate source/sensitivity treatment.",
            "limits": "not an audit of every historical clearing day; 2016-12-30, 2017-01-25 and 2017-11-20 require older equity-specific receipts. No synthetic transactions or changed historical tensors.",
        },
    )
    print(
        json.dumps(
            {
                "money_only_dates": len(rows),
                "delivery_closures": sum(
                    r["status"] == "no_additional_equity_delivery_day" for r in rows
                ),
                "controls": len(controls),
                "term_spans": term_spans,
                "sha256": sha256_file(destination),
            }
        )
    )


if __name__ == "__main__":
    main()
