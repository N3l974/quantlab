from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SourceAdapter(ABC):
    name: str

    @abstractmethod
    def list_markets(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def list_timeframes(self, symbol: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        raise NotImplementedError

    def ohlcv_available_from_ms(self, symbol: str, timeframe: str) -> tuple[int | None, bool]:
        return (None, False)
