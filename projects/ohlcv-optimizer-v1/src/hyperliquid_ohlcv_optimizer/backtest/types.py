from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCosts:
    fee_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class RiskConfig:
    risk_pct: float
    max_position_notional_pct_equity: float


@dataclass(frozen=True)
class CommonTradeParams:
    tp_mode: str  # pct|rr
    tp_pct: float
    tp_rr: float
    tp_mgmt: str  # full|partial_trailing
    tp1_close_frac: float
    tp_trail_pct: float
    sl_type: str  # pct|atr
    sl_pct: float
    sl_atr_period: int
    sl_atr_mult: float
    sl_trailing: bool
    exit_on_flat: bool


@dataclass(frozen=True)
class GridConfig:
    max_adds: int
    spacing_pct: float
    size_multiplier: float


@dataclass(frozen=True)
class MartingaleConfig:
    multiplier: float
    max_steps: int


@dataclass(frozen=True)
class PositionManagerConfig:
    mode: str  # none|grid|martingale
    grid: GridConfig | None = None
    martingale: MartingaleConfig | None = None


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float
    costs: ExecutionCosts
    risk: RiskConfig
    common: CommonTradeParams
    pm: PositionManagerConfig
