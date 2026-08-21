from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from brazil_rv.preprocessing.bdi_lending import (
    Bulletin,
    Identity,
    Position,
    StockDay,
    _map_bulletin,
    build_rows,
    parse_bdi_pages,
)

PETR_ISIN = "BRPETRACNPR6"
PETR_ID = f"ISIN:{PETR_ISIN}"
VALE_ISIN = "BRVALEACNOR0"
VALE_ID = f"ISIN:{VALE_ISIN}"


def test_parser_stitches_legacy_and_modern_layouts_without_double_counting() -> None:
    legacy = parse_bdi_pages(
        [
            "\n".join(
                (
                    "Banco de Títulos",
                    "informamos o saldo acumulado de ações emprestadas em 01/04/2022.",
                    "Ação Empresa Tipo Saldo em número de ações Balanço (R$)",
                    "PETR4 PETROLEO BRASILEIRO S.A. PN N2 1.234 45.678,90",
                    "VALE3 VALE S.A. ON NM 2.000 150.000,00",
                    "Mercado Número de contratos Valor referencial (R$)",
                )
            )
        ],
        date(2022, 4, 4),
    )
    assert legacy is not None
    assert legacy.layout == "legacy_ticker"
    assert legacy.position_date == date(2022, 4, 1)
    assert legacy.positions[0].quantity == 1_234
    assert legacy.positions[0].isin is None

    modern = parse_bdi_pages(
        [
            "\n".join(
                (
                    "Empréstimos de Ativos – Posição em Aberto",
                    "Data Ticker ISIN Empresa Tipo Mercado Saldo Preço Saldo em R$",
                    "28/06/2024 PETR4 BRPETRACNPR6 PETROBRAS PN N2 Registro 100 30,0000 3.000,00",
                    "28/06/2024 PETR4 BRPETRACNPR6 PETROBRAS PN N2 Neg. Eletrônica D+1 200 30,0000 6.000,00",
                    "28/06/2024 PETR4 BRPETRACNPR6 PETROBRAS PN N2 Total 300 - 9.000,00",
                    "28/06/2024 VALE3 BRVALEACNOR0 VALE ON NM Registro 10 60.0000 600.00",
                    "Empréstimos de Ativos – Empréstimos Registrados",
                )
            )
        ],
        date(2024, 6, 28),
    )
    assert modern is not None
    assert modern.layout == "modern_isin"
    assert modern.source_row_count == 4
    assert modern.used_total_row_count == 1
    by_isin = {position.isin: position for position in modern.positions}
    assert by_isin[PETR_ISIN].quantity == 300
    assert by_isin[VALE_ISIN].quantity == 10


def test_legacy_identity_is_exact_same_date_ticker_not_current_ticker() -> None:
    position_date = date(2022, 4, 1)
    identity = Identity(PETR_ID, PETR_ISIN, date(2021, 1, 1), date(2024, 12, 31))
    bulletin = Bulletin(
        report_date=date(2022, 4, 4),
        position_date=position_date,
        layout="legacy_ticker",
        positions=(Position(position_date, "OLDT4", None, 100, 3_000.0),),
        source_row_count=1,
        used_total_row_count=0,
    )
    mapped, audit = _map_bulletin(
        bulletin,
        {PETR_ISIN: identity},
        {(position_date, PETR_ID): StockDay("OLDT4", 1_000)},
    )
    assert mapped[PETR_ID].ticker == "OLDT4"
    assert audit == {"unmapped": 0, "outside": 0}

    unmatched, audit = _map_bulletin(
        bulletin,
        {PETR_ISIN: identity},
        {(position_date, PETR_ID): StockDay("PETR4", 1_000)},
    )
    assert unmatched == {}
    assert audit["unmapped"] == 1


def test_rows_use_next_session_exact_lags_and_future_mutation_isolated() -> None:
    first = date(2023, 1, 2)
    sessions = [first + timedelta(days=index) for index in range(43)]
    identities = {
        PETR_ISIN: Identity(PETR_ID, PETR_ISIN, first, sessions[-1]),
        VALE_ISIN: Identity(VALE_ID, VALE_ISIN, first, sessions[-1]),
    }
    stock_days = {
        (day, security_id): StockDay(ticker, 1_000)
        for day in sessions
        for security_id, ticker in ((PETR_ID, "PETR4"), (VALE_ID, "VALE3"))
    }
    bulletins = [
        Bulletin(
            report_date=day,
            position_date=day,
            layout="modern_isin",
            positions=(Position(day, "PETR4", PETR_ISIN, 100 + index, 3_000.0),),
            source_row_count=1,
            used_total_row_count=0,
        )
        for index, day in enumerate(sessions[:-1])
        if index != 34
    ]
    rows = build_rows(bulletins, identities, stock_days, sessions)
    keyed = {(row["source_position_date"], row["security_id"]): row for row in rows}
    first_valid = keyed[(sessions[19], PETR_ID)]
    assert first_valid["available_date"] == sessions[20]
    assert first_valid["lending_balance_days_to_cover_log_tanh_mask"]
    assert not keyed[(sessions[18], PETR_ID)][
        "lending_balance_days_to_cover_log_tanh_mask"
    ]
    assert keyed[(sessions[19], VALE_ID)]["lending_balance_quantity"] == 0
    assert keyed[(sessions[24], PETR_ID)][
        "lending_balance_days_to_cover_change_5_tanh_mask"
    ]
    assert not keyed[(sessions[39], PETR_ID)][
        "lending_balance_days_to_cover_change_5_tanh_mask"
    ]
    assert keyed[(sessions[39], PETR_ID)][
        "lending_balance_days_to_cover_change_20_tanh_mask"
    ]

    cutoff = sessions[40]
    baseline = {
        (row["source_position_date"], row["security_id"]): {
            key: value
            for key, value in row.items()
            if key.startswith("lending_balance_days_to_cover")
        }
        for row in rows
        if row["source_position_date"] < cutoff
    }
    mutated = bulletins.copy()
    mutated[-1] = replace(
        mutated[-1],
        positions=(replace(mutated[-1].positions[0], quantity=10_000_000_000),),
    )
    rerun = build_rows(mutated, identities, stock_days, sessions)
    assert {
        (row["source_position_date"], row["security_id"]): {
            key: value
            for key, value in row.items()
            if key.startswith("lending_balance_days_to_cover")
        }
        for row in rerun
        if row["source_position_date"] < cutoff
    } == baseline
