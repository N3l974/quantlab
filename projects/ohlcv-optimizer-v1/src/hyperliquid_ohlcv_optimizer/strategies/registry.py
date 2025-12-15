from __future__ import annotations

from hyperliquid_ohlcv_optimizer.strategies.base import Strategy
from hyperliquid_ohlcv_optimizer.strategies.bollinger_breakout import BollingerBreakoutStrategy
from hyperliquid_ohlcv_optimizer.strategies.ma_cross import MovingAverageCrossStrategy
from hyperliquid_ohlcv_optimizer.strategies.range_breakout import RangeBreakoutStrategy
from hyperliquid_ohlcv_optimizer.strategies.rsi_reversion import RsiReversionStrategy
from hyperliquid_ohlcv_optimizer.strategies.stochastic_strategy import StochasticStrategy


def builtin_strategies() -> list[Strategy]:
    return [
        MovingAverageCrossStrategy(),
        RsiReversionStrategy(),
        BollingerBreakoutStrategy(),
        RangeBreakoutStrategy(),
        StochasticStrategy(),
    ]
