from datetime import date, timedelta

import numpy as np
import pytest

from brazil_rv.v2.foreign_flow import decision_panel, parse_mtd, published_differences


@pytest.mark.parametrize(
    "amounts",
    ["31.848.177 26,57 33.173.763 27,68", "31,848,177 26.57 33,173,763 27.68"],
)
def test_published_units_and_locale(amounts):
    text = (
        "Dados acumulados do início do mês até o dia 05/03/2024\nTipos de Investidores Compras (R$) Mil Participação (%) Vendas (R$) Mil Participação (%)\nInvestidor Estrangeiro "
        + amounts
    )
    result = parse_mtd(text, date(2024, 3, 7))
    assert result["net_mtd_billion_brl"] == pytest.approx(-1.325586)
    with pytest.raises(ValueError, match="chronology"):
        parse_mtd(text, date(2024, 3, 4))


def test_resets_gaps_revisions_and_methodology_are_not_fake_daily_flows():
    days = [date(2024, 1, 29) + timedelta(days=i) for i in range(12)]
    rows = [
        {
            "publication_date": d + timedelta(days=2),
            "reference_date": d,
            "net_mtd_billion_brl": float(i + 1 if d.month == 1 else i - 2),
            "methodology": "a",
        }
        for i, d in enumerate(days[:8])
    ]
    result = published_differences(rows, days)
    assert [x["foreign_flow_1"] for x in result] == [1.0] * 8
    assert result[3]["foreign_flow_month_reset"] == 1
    assert result[4]["foreign_flow_5"] == 5
    missing = published_differences(rows[:4] + rows[5:], days)
    assert missing[4]["foreign_flow_1"] is None
    assert missing[4]["foreign_flow_5"] is None
    revised = rows[:4] + [
        {
            **rows[3],
            "publication_date": rows[3]["publication_date"] + timedelta(days=1),
            "net_mtd_billion_brl": 10.0,
        }
    ]
    assert published_differences(revised, days)[-1]["foreign_flow_1"] is None
    changed = published_differences(
        rows[:5] + [{**x, "methodology": "b"} for x in rows[5:]], days
    )
    assert changed[5]["foreign_flow_1"] is None
    assert changed[5]["foreign_flow_methodology_change"] == 1


def test_future_publications_do_not_repair_past_and_date_only_waits_for_next_decision():
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(10)]
    row = {
        "publication_date": days[2],
        "reference_date": days[0],
        "net_mtd_billion_brl": 2.0,
        "methodology": "a",
    }
    before = decision_panel([row], days)
    after = decision_panel(
        [
            row,
            {
                **row,
                "publication_date": days[6],
                "reference_date": days[4],
                "net_mtd_billion_brl": 100,
            },
        ],
        days,
    )
    for a, b in zip(before[1:4], after[1:4]):
        np.testing.assert_array_equal(a[:7], b[:7])
    assert not before[2][2, 0]
    assert before[2][3, 0] and before[1][3, 0] == 2.0 and before[3][3, 0] == 3
    assert not after[2][7, 0]  # Unsupported multi-session interval clears flow1.
