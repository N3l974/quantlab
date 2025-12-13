from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_data_downloader.core.timeframes import align_up, timeframe_to_ms
from market_data_downloader.utils.strings import to_path_component

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OhlcvLocation:
    source: str
    symbol: str
    timeframe: str


def _utc_datetime_from_ms(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _month_start(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, 1, tzinfo=timezone.utc)


def _add_month(dt: datetime) -> datetime:
    year = dt.year + (dt.month // 12)
    month = (dt.month % 12) + 1
    return datetime(year, month, 1, tzinfo=timezone.utc)


def iter_months(start_ms: int, end_ms: int) -> list[tuple[int, int, int, int]]:
    start_dt = _utc_datetime_from_ms(start_ms)
    end_dt = _utc_datetime_from_ms(end_ms - 1)

    cur = _month_start(start_dt)
    end_month = _month_start(end_dt)

    months: list[tuple[int, int, int, int]] = []
    while cur <= end_month:
        nxt = _add_month(cur)
        months.append((cur.year, cur.month, int(cur.timestamp() * 1000), int(nxt.timestamp() * 1000)))
        cur = nxt

    return months


def parquet_path(data_root: Path, loc: OhlcvLocation, year: int, month: int) -> Path:
    sym = to_path_component(loc.symbol)
    tf = to_path_component(loc.timeframe)
    src = to_path_component(loc.source)
    return data_root / "ohlcv" / src / sym / tf / f"{year:04d}" / f"{month:02d}.parquet"


def legacy_parquet_path(data_root: Path, loc: OhlcvLocation, year: int, month: int) -> Path:
    sym = to_path_component(loc.symbol)
    tf = to_path_component(loc.timeframe)
    src = to_path_component(loc.source)
    return data_root / src / sym / tf / f"{year:04d}" / f"{month:02d}.parquet"


def read_existing_timestamps(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=np.int64)

    try:
        s = pd.read_parquet(path, columns=["timestamp_ms"])["timestamp_ms"]
        return s.astype("int64").to_numpy()
    except Exception:
        logger.exception("Failed reading existing parquet timestamps: %s", path)
        return np.array([], dtype=np.int64)


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def compute_missing_ranges(
    existing_ts: np.ndarray,
    start_ms: int,
    end_ms: int,
    timeframe: str,
) -> list[tuple[int, int]]:
    tf_ms = timeframe_to_ms(timeframe)
    aligned_start = align_up(start_ms, tf_ms)
    if aligned_start >= end_ms:
        return []

    expected = np.arange(aligned_start, end_ms, tf_ms, dtype=np.int64)
    if expected.size == 0:
        return []

    if existing_ts.size == 0:
        return [(int(expected[0]), int(end_ms))]

    existing_set = np.unique(existing_ts[(existing_ts >= aligned_start) & (existing_ts < end_ms)])
    missing = np.setdiff1d(expected, existing_set, assume_unique=False)
    if missing.size == 0:
        return []

    ranges: list[tuple[int, int]] = []
    range_start = int(missing[0])
    prev = int(missing[0])
    for ts in missing[1:]:
        ts_int = int(ts)
        if ts_int == prev + tf_ms:
            prev = ts_int
            continue

        ranges.append((range_start, prev + tf_ms))
        range_start = ts_int
        prev = ts_int

    ranges.append((range_start, prev + tf_ms))
    return ranges
