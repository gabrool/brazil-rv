from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .config import ExecutionConfig

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class DailyExecutionResult:
    trade_date: date
    net_pnl_brl: float
    gross_pnl_brl: float
    spread_cost_brl: float
    fees_brl: float
    cdi_earned_brl: float
    turnover_brl: float
    max_intraday_gross_brl: float
    forced_fill_count: int

    def __post_init__(self) -> None:
        components = (
            self.net_pnl_brl,
            self.gross_pnl_brl,
            self.spread_cost_brl,
            self.fees_brl,
            self.cdi_earned_brl,
            self.turnover_brl,
            self.max_intraday_gross_brl,
        )
        if not all(math.isfinite(value) for value in components):
            raise ValueError("Daily execution results must be finite")
        expected = (
            self.gross_pnl_brl
            - self.spread_cost_brl
            - self.fees_brl
            + self.cdi_earned_brl
        )
        if not math.isclose(self.net_pnl_brl, expected, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Daily execution accounting identity does not hold")
        if (
            self.spread_cost_brl < 0.0
            or self.fees_brl < 0.0
            or self.turnover_brl < 0.0
            or self.max_intraday_gross_brl < 0.0
            or self.forced_fill_count < 0
        ):
            raise ValueError("Execution costs, exposure, and counts cannot be negative")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["trade_date"] = self.trade_date.isoformat()
        return values


def execution_report_payload(
    *,
    config: ExecutionConfig,
    input_sha256: Mapping[str, str],
    daily: Sequence[DailyExecutionResult],
) -> dict[str, object]:
    inputs = dict(sorted(input_sha256.items()))
    if not inputs or any(
        not name or _SHA256.fullmatch(value) is None for name, value in inputs.items()
    ):
        raise ValueError("Input identities must be named lowercase SHA-256 values")
    rows = sorted(daily, key=lambda value: value.trade_date)
    if len({value.trade_date for value in rows}) != len(rows):
        raise ValueError("Daily execution results contain duplicate dates")

    aggregate = {
        "date_count": len(rows),
        "net_pnl_brl": sum(value.net_pnl_brl for value in rows),
        "gross_pnl_brl": sum(value.gross_pnl_brl for value in rows),
        "spread_cost_brl": sum(value.spread_cost_brl for value in rows),
        "fees_brl": sum(value.fees_brl for value in rows),
        "cdi_earned_brl": sum(value.cdi_earned_brl for value in rows),
        "turnover_brl": sum(value.turnover_brl for value in rows),
        "max_intraday_gross_brl": max(
            (value.max_intraday_gross_brl for value in rows), default=0.0
        ),
        "forced_fill_count": sum(value.forced_fill_count for value in rows),
    }
    return {
        "schema": "B3_EXECUTION_BACKTEST_REPORT_V1",
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "input_sha256": inputs,
        "daily": [value.to_dict() for value in rows],
        "aggregate": aggregate,
    }


def write_execution_report(
    path: Path,
    *,
    config: ExecutionConfig,
    input_sha256: Mapping[str, str],
    daily: Sequence[DailyExecutionResult],
) -> dict[str, str]:
    payload = execution_report_payload(
        config=config, input_sha256=input_sha256, daily=daily
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    os.replace(sidecar_temporary, sidecar)
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "sha256_path": str(sidecar.resolve()),
    }
