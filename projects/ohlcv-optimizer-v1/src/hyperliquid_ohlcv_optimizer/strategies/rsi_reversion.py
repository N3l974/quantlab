from __future__ import annotations

import pandas as pd

from hyperliquid_ohlcv_optimizer.strategies.base import Strategy, StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.indicators import rsi


class RsiReversionStrategy(Strategy):
    name = "rsi_reversion"

    def sample_params(self, trial) -> dict:
        period = int(trial.suggest_int("rsi_period", 7, 40))
        low_th = float(trial.suggest_float("rsi_low_th", 10.0, 45.0))
        high_th = float(trial.suggest_float("rsi_high_th", 55.0, 90.0))
        if high_th <= low_th:
            high_th = low_th + 10.0
        return {"rsi_period": period, "rsi_low_th": low_th, "rsi_high_th": high_th}

    def compute_signal(self, df: pd.DataFrame, params: dict, ctx: StrategyContext) -> pd.Series:
        r = rsi(df["close"], int(params["rsi_period"]))
        long_sig = (r < float(params["rsi_low_th"])).astype("int8")
        short_sig = (r > float(params["rsi_high_th"])).astype("int8")
        sig = long_sig - short_sig
        return sig.fillna(0).astype("int8")
