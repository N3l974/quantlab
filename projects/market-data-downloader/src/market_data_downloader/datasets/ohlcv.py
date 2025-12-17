from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from market_data_downloader.core.timeframes import timeframe_to_ms
from market_data_downloader.sources.base import SourceAdapter
from market_data_downloader.storage.ohlcv_store import (
    OhlcvLocation,
    compute_missing_ranges,
    iter_months,
    legacy_parquet_path,
    parquet_path,
    read_existing_timestamps,
    write_parquet,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DownloadSummary:
    files_written: int
    candles_added: int
    candles_total: int


ProgressCallback = Callable[[int, int, str], None]


def _read_existing_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp_ms", "open", "high", "low", "close", "volume"])

    return pd.read_parquet(path)


def download_ohlcv(
    *,
    adapter: SourceAdapter,
    data_root: Path,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    progress: ProgressCallback | None = None,
) -> DownloadSummary:
    if start_ms >= end_ms:
        raise ValueError("start must be < end")

    now_ms = int(time.time() * 1000)
    if end_ms > now_ms:
        end_ms = now_ms

    loc = OhlcvLocation(source=adapter.name, symbol=symbol, timeframe=timeframe)

    files_written = 0
    candles_added = 0
    candles_total = 0

    month_jobs: list[
        tuple[
            int,
            int,
            int,
            int,
            int,
            int,
            Path,
            Path,
            list[tuple[int, int]],
            bool,
        ]
    ] = []
    for year, month, month_start_ms, month_end_ms in iter_months(start_ms, end_ms):
        sub_start = max(start_ms, month_start_ms)
        sub_end = min(end_ms, month_end_ms)
        if sub_start >= sub_end:
            continue

        path = parquet_path(data_root, loc, year, month)
        legacy_path = legacy_parquet_path(data_root, loc, year, month)
        needs_migration = (not path.exists()) and legacy_path.exists()
        existing_path = path if path.exists() else legacy_path
        existing_ts = read_existing_timestamps(existing_path)

        missing_ranges = compute_missing_ranges(existing_ts, sub_start, sub_end, timeframe)

        if not missing_ranges and existing_path.exists() and not needs_migration:
            try:
                candles_total += int(pd.read_parquet(existing_path, columns=["timestamp_ms"]).shape[0])
            except Exception:
                pass

        month_jobs.append(
            (
                year,
                month,
                month_start_ms,
                month_end_ms,
                sub_start,
                sub_end,
                path,
                legacy_path,
                missing_ranges,
                needs_migration,
            )
        )

    download_jobs = [j for j in month_jobs if j[8] or j[9]]
    total_jobs = len(download_jobs)
    done_jobs = 0
    if progress is not None:
        progress(done_jobs, total_jobs, "Starting")

    for year, month, month_start_ms, month_end_ms, sub_start, sub_end, path, legacy_path, missing_ranges, needs_migration in download_jobs:
        if progress is not None:
            progress(done_jobs, total_jobs, f"Downloading {year:04d}-{month:02d}")

        existing_df_path = path if path.exists() else legacy_path

        if needs_migration and not missing_ranges:
            df_existing = _read_existing_df(existing_df_path)
            if not df_existing.empty:
                write_parquet(path, df_existing)
                files_written += 1
                candles_total += int(df_existing.shape[0])

            done_jobs += 1
            if progress is not None:
                progress(done_jobs, total_jobs, f"Migrated {year:04d}-{month:02d}")
            continue

        new_parts: list[pd.DataFrame] = []
        tf_ms = timeframe_to_ms(timeframe)
        max_candles = adapter.ohlcv_max_candles_per_request(symbol=symbol, timeframe=timeframe)
        available_from_ms, _ = adapter.ohlcv_available_from_ms(symbol=symbol, timeframe=timeframe)

        def _align_down(ts_ms: int) -> int:
            return (ts_ms // tf_ms) * tf_ms

        def _iter_windows(r_start: int, r_end: int) -> list[tuple[int, int]]:
            if max_candles is None:
                return [(r_start, r_end)]

            step_ms = int(max_candles) * tf_ms
            if step_ms <= 0:
                return [(r_start, r_end)]

            windows: list[tuple[int, int]] = []
            cur = int(r_start)
            while cur < r_end:
                nxt = min(int(r_end), cur + step_ms)
                windows.append((cur, int(nxt)))
                cur = int(nxt)
            return windows

        for r_start, r_end in missing_ranges:
            rr_start = int(r_start)
            rr_end = int(r_end)

            if available_from_ms is not None:
                rr_start = max(rr_start, int(available_from_ms))
            rr_end = min(rr_end, _align_down(now_ms + tf_ms))
            if rr_start >= rr_end:
                continue

            for w_start, w_end in _iter_windows(rr_start, rr_end):
                logger.info("Downloading %s %s %s %s..%s", adapter.name, symbol, timeframe, w_start, w_end)
                df_part = adapter.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start_ms=w_start, end_ms=w_end)
                if df_part is None or df_part.empty:
                    continue

                df_part = df_part[(df_part["timestamp_ms"] >= w_start) & (df_part["timestamp_ms"] < w_end)]
                if df_part.empty:
                    continue

                new_parts.append(df_part)

        if not new_parts:
            try:
                df_existing = _read_existing_df(existing_df_path)
                candles_total += int(df_existing.shape[0])
            except Exception:
                pass

            done_jobs += 1
            if progress is not None:
                progress(done_jobs, total_jobs, f"No new data for {year:04d}-{month:02d}")
            continue

        df_new = pd.concat(new_parts, ignore_index=True)
        df_new = df_new.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")

        df_existing = _read_existing_df(existing_df_path)
        if not df_existing.empty:
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all = df_all.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
        df_all = df_all[(df_all["timestamp_ms"] >= month_start_ms) & (df_all["timestamp_ms"] < month_end_ms)]

        before = int(df_existing.shape[0])
        after = int(df_all.shape[0])

        write_parquet(path, df_all)
        files_written += 1
        candles_added += max(0, after - before)
        candles_total += after

        done_jobs += 1
        if progress is not None:
            progress(done_jobs, total_jobs, f"Wrote {year:04d}-{month:02d}")

    return DownloadSummary(files_written=files_written, candles_added=candles_added, candles_total=candles_total)
