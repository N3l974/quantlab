from __future__ import annotations

from pathlib import Path

import pandas as pd


def _timeframe_dir(data_root: Path, source: str, symbol: str, timeframe: str) -> Path:
    return data_root / "ohlcv" / source / symbol / timeframe


def discover_symbols(*, data_root: Path, source: str) -> list[str]:
    root = data_root / "ohlcv" / source
    if not root.exists():
        return []

    symbols: list[str] = []
    for p in root.iterdir():
        if p.is_dir():
            symbols.append(p.name)

    return sorted(set(symbols))


def load_ohlcv(*, data_root: Path, source: str, symbol: str, timeframe: str) -> pd.DataFrame:
    tf_dir = _timeframe_dir(data_root, source, symbol, timeframe)
    if not tf_dir.exists():
        raise FileNotFoundError(f"Missing timeframe folder: {tf_dir}")

    files = sorted(tf_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {tf_dir}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        dfs.append(pd.read_parquet(f))

    df = pd.concat(dfs, ignore_index=True)
    expected_cols = ["timestamp_ms", "open", "high", "low", "close", "volume"]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    df = df[expected_cols].copy()

    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
    df["timestamp_ms"] = df["timestamp_ms"].astype("int64")

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype("float64")

    return df.reset_index(drop=True)
