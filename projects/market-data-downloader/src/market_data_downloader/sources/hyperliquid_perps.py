from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from market_data_downloader.core.timeframes import sort_timeframes_desc
from market_data_downloader.core.timeframes import timeframe_to_ms
from market_data_downloader.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

_SUPPORTED_INTERVALS = [
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
]


class HyperliquidPerpsSource(SourceAdapter):
    name = "hyperliquid_perps"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._info_url = "https://api.hyperliquid.xyz/info"
        self._meta: dict | None = None

    def _post(self, payload: dict) -> object:
        r = self._session.post(self._info_url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def _load_meta(self) -> None:
        if self._meta is None:
            self._meta = self._post({"type": "meta"})  # type: ignore[assignment]

    def list_markets(self) -> list[str]:
        self._load_meta()
        universe = self._meta.get("universe", []) if self._meta else []
        names = [a["name"] for a in universe if isinstance(a, dict) and "name" in a]
        return sorted(set(names))

    def list_timeframes(self, symbol: str) -> list[str]:
        return sort_timeframes_desc(list(_SUPPORTED_INTERVALS))

    def ohlcv_available_from_ms(self, symbol: str, timeframe: str) -> tuple[int | None, bool]:
        try:
            tf_ms = timeframe_to_ms(timeframe)
        except Exception:
            return (None, True)

        now_ms = int(time.time() * 1000)
        earliest_ms = max(0, now_ms - 5000 * tf_ms)
        return (earliest_ms, True)

    def ohlcv_max_candles_per_request(self, symbol: str, timeframe: str) -> int | None:
        return 5000

    def fetch_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        payload = {
            "type": "candleSnapshot",
            "req": {"coin": symbol, "interval": timeframe, "startTime": start_ms, "endTime": end_ms},
        }

        candles = self._post(payload)
        rows: list[list[int | float]] = []

        for c in candles or []:  # type: ignore[union-attr]
            ts = c.get("t")
            if ts is None:
                ts = c.get("T")
            if ts is None:
                continue

            rows.append(
                [
                    int(ts),
                    float(c["o"]),
                    float(c["h"]),
                    float(c["l"]),
                    float(c["c"]),
                    float(c["v"]),
                ]
            )

        df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        if not df.empty:
            df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")

        return df
