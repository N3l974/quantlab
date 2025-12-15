from __future__ import annotations

import pandas as pd

from hyperliquid_ohlcv_optimizer.strategies.base import Strategy, StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.indicators import ema


class MovingAverageCrossStrategy(Strategy):
    name = "ma_cross"

    def sample_params(self, trial) -> dict:
        fast = int(trial.suggest_int("fast_period", 5, 80))
        slow = int(trial.suggest_int("slow_period", max(fast + 1, 10), 250))
        return {"fast_period": fast, "slow_period": slow}

    def compute_signal(self, df: pd.DataFrame, params: dict, ctx: StrategyContext) -> pd.Series:
        close = df["close"]
        fast = ema(close, int(params["fast_period"]))
        slow = ema(close, int(params["slow_period"]))
        sig = (fast > slow).astype("int8") - (fast < slow).astype("int8")
        sig = sig.fillna(0).astype("int8")
        return sig
