from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    """Economic assumptions shared by one execution-backtest run."""

    nav_brl: float = 10_000_000.0
    gross_target: float = 2.0
    participation_rate: float = 0.10
    name_cap_fraction_of_gross: float = 0.025
    adv_cap_fraction: float = 0.05
    fee_bps: float = 2.0
    max_spread_bps: float = 75.0
    min_adv_brl: float = 1_000_000.0
    taper_minutes: int = 30
    force_spread_multiplier: float = 2.0
    margin_fraction_of_gross: float = 0.5
    causal_lookback_sessions: int = 20
    horizon_blend: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    band: float = 0.0
    position_tolerance_brl: float = 1e-8

    def __post_init__(self) -> None:
        positive = (
            self.nav_brl,
            self.gross_target,
            self.participation_rate,
            self.name_cap_fraction_of_gross,
            self.adv_cap_fraction,
            self.max_spread_bps,
            self.min_adv_brl,
            self.force_spread_multiplier,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Execution scale and cap settings must be positive")
        if self.participation_rate > 1.0:
            raise ValueError("Participation rate cannot exceed one")
        if not math.isfinite(self.fee_bps) or self.fee_bps < 0.0:
            raise ValueError("Fee bps must be finite and non-negative")
        if (
            not math.isfinite(self.margin_fraction_of_gross)
            or self.margin_fraction_of_gross < 0.0
        ):
            raise ValueError("Margin fraction must be finite and non-negative")
        if self.taper_minutes <= 0 or self.causal_lookback_sessions <= 0:
            raise ValueError("Taper and causal lookback must be positive")
        if not self.horizon_blend or any(
            not math.isfinite(value) or value < 0.0 for value in self.horizon_blend
        ):
            raise ValueError("Horizon blend weights must be finite and non-negative")
        if sum(self.horizon_blend) <= 0.0:
            raise ValueError("At least one horizon blend weight must be positive")
        if math.isnan(self.band) or self.band < 0.0:
            raise ValueError("Band must be non-negative")
        if (
            not math.isfinite(self.position_tolerance_brl)
            or self.position_tolerance_brl < 0.0
        ):
            raise ValueError("Position tolerance must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["horizon_blend"] = list(self.horizon_blend)
        if math.isinf(self.band):
            values["band"] = "infinity"
        return values

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
