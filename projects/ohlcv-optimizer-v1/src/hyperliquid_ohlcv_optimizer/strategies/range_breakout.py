from __future__ import annotations

import pandas as pd

from hyperliquid_ohlcv_optimizer.strategies.base import Strategy, StrategyContext


class RangeBreakoutStrategy(Strategy):
    name = "range_breakout"

    def sample_params(self, trial) -> dict:
        lookback = int(trial.suggest_int("range_lookback", 10, 250))
        buffer_pct = float(trial.suggest_float("range_buffer_pct", 0.0, 0.5))
        return {"range_lookback": lookback, "range_buffer_pct": buffer_pct}

    def compute_signal(self, df: pd.DataFrame, params: dict, ctx: StrategyContext) -> pd.Series:
        lb = int(params["range_lookback"])
        buf = float(params["range_buffer_pct"]) / 100.0

        prev_high = df["high"].shift(1).rolling(lb, min_periods=lb).max()
        prev_low = df["low"].shift(1).rolling(lb, min_periods=lb).min()

        close = df["close"]
        long_sig = (close > (prev_high * (1.0 + buf))).astype("int8")
        short_sig = (close < (prev_low * (1.0 - buf))).astype("int8")
        sig = long_sig - short_sig
        return sig.fillna(0).astype("int8")
