from __future__ import annotations

import pandas as pd

from hyperliquid_ohlcv_optimizer.strategies.base import Strategy, StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.indicators import sma, stochastic_k


class StochasticStrategy(Strategy):
    name = "stochastic"

    def sample_params(self, trial) -> dict:
        k_period = int(trial.suggest_int("stoch_k_period", 5, 40))
        d_period = int(trial.suggest_int("stoch_d_period", 2, 15))
        oversold = float(trial.suggest_float("stoch_oversold", 5.0, 35.0))
        overbought = float(trial.suggest_float("stoch_overbought", 65.0, 95.0))
        if overbought <= oversold:
            overbought = oversold + 30.0
        return {
            "stoch_k_period": k_period,
            "stoch_d_period": d_period,
            "stoch_oversold": oversold,
            "stoch_overbought": overbought,
        }

    def compute_signal(self, df: pd.DataFrame, params: dict, ctx: StrategyContext) -> pd.Series:
        k = stochastic_k(df["close"], df["high"], df["low"], int(params["stoch_k_period"]))
        d = sma(k, int(params["stoch_d_period"]))

        k_prev = k.shift(1)
        d_prev = d.shift(1)

        cross_up = (k_prev <= d_prev) & (k > d)
        cross_down = (k_prev >= d_prev) & (k < d)

        long_sig = (cross_up & (k < float(params["stoch_oversold"])) & (d < float(params["stoch_oversold"])) ).astype("int8")
        short_sig = (cross_down & (k > float(params["stoch_overbought"])) & (d > float(params["stoch_overbought"])) ).astype("int8")
        sig = long_sig - short_sig
        return sig.fillna(0).astype("int8")
