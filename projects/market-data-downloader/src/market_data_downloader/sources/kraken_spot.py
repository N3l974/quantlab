from __future__ import annotations

import logging

import ccxt
import pandas as pd

from market_data_downloader.core.timeframes import sort_timeframes_desc, timeframe_to_ms
from market_data_downloader.sources.base import SourceAdapter

logger = logging.getLogger(__name__)


class KrakenSpotSource(SourceAdapter):
    name = "kraken_spot"

    def __init__(self) -> None:
        self._exchange = ccxt.kraken({"enableRateLimit": True})
        self._markets_loaded = False

    def _load_markets(self) -> None:
        if self._markets_loaded:
            return
        self._exchange.load_markets()
        self._markets_loaded = True

    def list_markets(self) -> list[str]:
        self._load_markets()
        symbols: list[str] = []
        for market in self._exchange.markets.values():
            if market.get("active") is False:
                continue
            if not market.get("spot"):
                continue
            symbols.append(market["symbol"])

        return sorted(set(symbols))

    def list_timeframes(self, symbol: str) -> list[str]:
        self._load_markets()
        timeframes = list((self._exchange.timeframes or {}).keys())
        return sort_timeframes_desc(timeframes)

    def ohlcv_available_from_ms(self, symbol: str, timeframe: str) -> tuple[int | None, bool]:
        self._load_markets()
        try:
            batch = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=0, limit=1)
            if batch:
                return (int(batch[0][0]), False)
        except Exception:
            logger.exception("Failed to estimate Kraken OHLCV availability for %s %s", symbol, timeframe)

        return (None, False)

    def fetch_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        self._load_markets()
        tf_ms = timeframe_to_ms(timeframe)

        since = start_ms
        rows: list[list[int | float]] = []

        while since < end_ms:
            batch = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=720)
            if not batch:
                break

            for ts, o, h, l, c, v in batch:
                if ts < start_ms:
                    continue
                if ts >= end_ms:
                    continue
                rows.append([int(ts), float(o), float(h), float(l), float(c), float(v)])

            last_ts = int(batch[-1][0])
            next_since = last_ts + tf_ms
            if next_since <= since:
                break
            since = next_since

        df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        if not df.empty:
            df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")

        return df
