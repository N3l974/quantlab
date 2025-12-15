from __future__ import annotations

import pandas as pd

from hyperliquid_ohlcv_optimizer.strategies.base import Strategy, StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.indicators import bollinger


class BollingerBreakoutStrategy(Strategy):
    name = "bollinger_breakout"

    def sample_params(self, trial) -> dict:
        period = int(trial.suggest_int("bb_period", 10, 80))
        n_std = float(trial.suggest_float("bb_std", 1.5, 4.0))
        return {"bb_period": period, "bb_std": n_std}

    def compute_signal(self, df: pd.DataFrame, params: dict, ctx: StrategyContext) -> pd.Series:
        lower, mid, upper = bollinger(df["close"], int(params["bb_period"]), float(params["bb_std"]))
        close = df["close"]
        long_sig = (close > upper).astype("int8")
        short_sig = (close < lower).astype("int8")
        sig = long_sig - short_sig
        return sig.fillna(0).astype("int8")
