from __future__ import annotations

import re

TIMEFRAME_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    "1M": 30 * 24 * 60 * 60_000,
}

_TF_RE = re.compile(r"^(\d+)([mhdwM])$")


def timeframe_ms_or_none(timeframe: str) -> int | None:
    tf = timeframe.strip()
    if not tf:
        return None

    if tf in TIMEFRAME_TO_MS:
        return TIMEFRAME_TO_MS[tf]

    m = _TF_RE.match(tf)
    if not m:
        return None

    n = int(m.group(1))
    unit = m.group(2)
    unit_ms = {
        "m": 60_000,
        "h": 60 * 60_000,
        "d": 24 * 60 * 60_000,
        "w": 7 * 24 * 60 * 60_000,
        "M": 30 * 24 * 60 * 60_000,
    }[unit]

    return n * unit_ms


def timeframe_to_ms(timeframe: str) -> int:
    ms = timeframe_ms_or_none(timeframe)
    if ms is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return ms


def align_up(ts_ms: int, timeframe_ms: int) -> int:
    return ((ts_ms + timeframe_ms - 1) // timeframe_ms) * timeframe_ms


def sort_timeframes_desc(timeframes: list[str]) -> list[str]:
    def key(tf: str) -> tuple[int, str]:
        ms = timeframe_ms_or_none(tf)
        return (ms if ms is not None else -1, tf)

    # Larger timeframes first.
    return sorted(timeframes, key=key, reverse=True)
