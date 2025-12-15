from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class StrategyContext:
    timeframe: str


class Strategy(Protocol):
    name: str

    def sample_params(self, trial) -> dict:  # optuna.Trial
        raise NotImplementedError

    def compute_signal(self, df: pd.DataFrame, params: dict, ctx: StrategyContext) -> pd.Series:
        raise NotImplementedError
