from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from market_data_downloader.core.timeframes import sort_timeframes_desc
from market_data_downloader.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None


_TF_TO_MT5_ATTR: dict[str, str] = {
    "1m": "TIMEFRAME_M1",
    "3m": "TIMEFRAME_M3",
    "5m": "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h": "TIMEFRAME_H1",
    "2h": "TIMEFRAME_H2",
    "4h": "TIMEFRAME_H4",
    "6h": "TIMEFRAME_H6",
    "8h": "TIMEFRAME_H8",
    "12h": "TIMEFRAME_H12",
    "1d": "TIMEFRAME_D1",
    "1w": "TIMEFRAME_W1",
    "1M": "TIMEFRAME_MN1",
}


def _dt_utc_from_ms(ts_ms: int) -> datetime:
    dt = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
    return dt.replace(tzinfo=None)


def _mt5_timeframe(timeframe: str) -> int:
    if mt5 is None:
        raise RuntimeError("MetaTrader5 is not installed")
    tf = str(timeframe)
    attr = _TF_TO_MT5_ATTR.get(tf)
    if not attr:
        raise ValueError(f"Unsupported timeframe for MT5: {timeframe}")
    try:
        return int(getattr(mt5, attr))
    except Exception as e:
        raise ValueError(f"Unsupported timeframe for MT5 (missing {attr}): {timeframe}") from e


class Mt5IcmarketsSource(SourceAdapter):
    name = "mt5_icmarkets"

    def __init__(self) -> None:
        if mt5 is None:
            raise RuntimeError(
                "MetaTrader5 is not installed. Install it with: pip install MetaTrader5 (Windows only)."
            )

        ok = bool(mt5.initialize())
        if not ok:
            err = None
            try:
                err = mt5.last_error()
            except Exception:
                err = None
            raise RuntimeError(
                f"Failed to initialize MetaTrader5. Ensure MT5 terminal is installed, running, and logged into ICMarkets. last_error={err}"
            )

        try:
            info = mt5.account_info()
        except Exception:
            info = None
        if info is None:
            raise RuntimeError(
                "MetaTrader5 initialized but account_info() is None. Ensure the MT5 terminal is logged in (ICMarkets demo/real)."
            )

    def list_markets(self) -> list[str]:
        if mt5 is None:
            return []

        syms = mt5.symbols_get()
        if not syms:
            return []

        names: list[str] = []
        for s in syms:
            n = getattr(s, "name", None)
            if n:
                names.append(str(n))
        return sorted(set(names))

    def list_timeframes(self, symbol: str) -> list[str]:
        if mt5 is None:
            return []
        out = []
        for tf, attr in _TF_TO_MT5_ATTR.items():
            if hasattr(mt5, attr):
                out.append(tf)
        return sort_timeframes_desc(out)

    def ohlcv_available_from_ms(self, symbol: str, timeframe: str) -> tuple[int | None, bool]:
        if mt5 is None:
            return (None, True)

        try:
            tf = _mt5_timeframe(str(timeframe))
        except Exception:
            return (None, True)

        if not bool(mt5.symbol_select(str(symbol), True)):
            return (None, True)

        try:
            # Request the earliest available bar by using a very old start date.
            # MT5 returns timestamps in seconds since epoch.
            rates = mt5.copy_rates_from(str(symbol), int(tf), datetime(1970, 1, 1), 1)
        except Exception:
            rates = None

        if rates is None or len(rates) == 0:
            return (None, True)

        try:
            ts_s = int(rates[0]["time"])
        except Exception:
            try:
                ts_s = int(getattr(rates[0], "time"))
            except Exception:
                return (None, True)

        return (ts_s * 1000, True)

    def fetch_ohlcv(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        if mt5 is None:
            raise RuntimeError("MetaTrader5 is not installed")

        tf = _mt5_timeframe(str(timeframe))

        if not bool(mt5.symbol_select(str(symbol), True)):
            err = None
            try:
                err = mt5.last_error()
            except Exception:
                err = None
            raise ValueError(f"Symbol not found or cannot be selected in MT5: {symbol}. last_error={err}")

        dt_start = _dt_utc_from_ms(int(start_ms))
        dt_end = _dt_utc_from_ms(int(end_ms))

        rates = mt5.copy_rates_range(str(symbol), int(tf), dt_start, dt_end)
        if rates is None:
            return pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rates)
        if df.empty:
            return pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

        if "time" not in df.columns:
            return pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

        df["timestamp_ms"] = (pd.to_numeric(df["time"], errors="coerce").fillna(0).astype("int64") * 1000).astype(
            "int64"
        )

        if "real_volume" in df.columns:
            rv = pd.to_numeric(df["real_volume"], errors="coerce").fillna(0)
            if "tick_volume" in df.columns:
                tv = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0)
                df["volume"] = rv.where(rv > 0, tv)
            else:
                df["volume"] = rv
        elif "tick_volume" in df.columns:
            df["volume"] = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0.0

        out = df[["timestamp_ms", "open", "high", "low", "close", "volume"]].copy()
        for c in ["open", "high", "low", "close", "volume"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

        out = out.dropna(subset=["timestamp_ms"]).drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
        out = out[(out["timestamp_ms"] >= int(start_ms)) & (out["timestamp_ms"] < int(end_ms))]
        return out.reset_index(drop=True)
