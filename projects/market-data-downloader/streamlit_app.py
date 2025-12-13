from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st

from market_data_downloader.core.registry import create_source, list_sources
from market_data_downloader.core.timeframes import sort_timeframes_desc
from market_data_downloader.datasets.ohlcv import download_ohlcv
from market_data_downloader.sources.builtins import register_builtin_sources
from market_data_downloader.utils.repo import default_data_root


def _date_to_start_ms(d) -> int:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _date_to_end_ms_inclusive(d) -> int:
    dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=1)
    return int(dt.timestamp() * 1000)


@st.cache_data(show_spinner=False)
def _cached_markets(source_name: str) -> list[str]:
    register_builtin_sources()
    adapter = create_source(source_name)
    return adapter.list_markets()


@st.cache_data(show_spinner=False)
def _cached_timeframes(source_name: str, symbol: str) -> list[str]:
    register_builtin_sources()
    adapter = create_source(source_name)
    return sort_timeframes_desc(adapter.list_timeframes(symbol))


@st.cache_data(show_spinner=False)
def _cached_ohlcv_available_from_ms(source_name: str, symbol: str, timeframe: str) -> tuple[int | None, bool]:
    register_builtin_sources()
    adapter = create_source(source_name)
    return adapter.ohlcv_available_from_ms(symbol, timeframe)


def main() -> None:
    st.set_page_config(page_title="Market Data Downloader", layout="centered")
    st.title("Market Data Downloader")

    register_builtin_sources()

    st.caption("Downloads market datasets and stores Parquet in data/market_data (repo root).")

    dataset = st.selectbox("Dataset", options=["ohlcv"])

    default_root = default_data_root()
    data_root_str = st.text_input("Data root", value=str(default_root))
    data_root = Path(data_root_str)

    source = st.selectbox("Source", options=list_sources())

    st.subheader("Instrument")

    markets = _cached_markets(source)

    query = st.text_input("Search symbol", value="BTC")
    q = query.strip()
    if q:
        filtered = [m for m in markets if q.upper() in m.upper()]
    else:
        filtered = markets

    if not filtered:
        st.error("No symbols match your search.")
        st.stop()

    if len(filtered) > 200:
        st.info("Too many results; showing first 200. Refine search to narrow down.")
        filtered = filtered[:200]

    symbol = st.selectbox("Symbol", options=filtered)

    timeframes = _cached_timeframes(source, symbol)
    timeframe = st.selectbox("Timeframe", options=timeframes, index=(timeframes.index("1h") if "1h" in timeframes else 0))

    st.subheader("Date range")

    selection_key = f"{dataset}|{source}|{symbol}|{timeframe}"
    if dataset == "ohlcv":
        available_from_ms, is_estimated = _cached_ohlcv_available_from_ms(source, symbol, timeframe)
        if available_from_ms is not None:
            available_from_date = datetime.fromtimestamp(available_from_ms / 1000.0, tz=timezone.utc).date()
            qualifier = "~ " if is_estimated else ""
            st.info(f"Available from {qualifier}{available_from_date} UTC")

            if st.session_state.get("_auto_date_key") != selection_key:
                st.session_state["_auto_date_key"] = selection_key
                st.session_state["start_date"] = available_from_date
                st.session_state["end_date"] = datetime.now(timezone.utc).date()

    start_date = st.date_input("Start date (UTC)", key="start_date")
    end_date = st.date_input("End date (UTC)", key="end_date")

    if start_date is None or end_date is None:
        st.stop()

    start_ms = _date_to_start_ms(start_date)
    end_ms = _date_to_end_ms_inclusive(end_date)

    if start_ms >= end_ms:
        st.error("Start must be before end.")
        st.stop()

    st.subheader("Download")
    if dataset == "ohlcv":
        st.write(f"Output folder: `{data_root / dataset / source / symbol / timeframe}`")
    else:
        st.write(f"Output folder: `{data_root / dataset}`")

    if st.button("Download", type="primary"):
        if dataset != "ohlcv":
            st.error("Only dataset 'ohlcv' is implemented for now")
            st.stop()

        adapter = create_source(source)
        progress_bar = st.progress(0.0)
        status = st.empty()
        t0 = time.time()

        def on_progress(done: int, total: int, message: str) -> None:
            if total <= 0:
                progress_bar.progress(0.0)
                status.write(message)
                return

            frac = min(1.0, max(0.0, done / total))
            progress_bar.progress(frac)

            elapsed = time.time() - t0
            if done > 0:
                remaining = elapsed * (total - done) / done
                status.write(f"{message} ({done}/{total}) — ETA ~ {int(remaining)}s")
            else:
                status.write(f"{message} ({done}/{total})")

        with st.spinner("Downloading..."):
            try:
                summary = download_ohlcv(
                    adapter=adapter,
                    data_root=data_root,
                    symbol=symbol,
                    timeframe=timeframe,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    progress=on_progress,
                )
            except Exception as e:
                st.exception(e)
                st.stop()

        progress_bar.progress(1.0)
        status.empty()

        st.success("Done")
        st.json(
            {
                "files_written": summary.files_written,
                "candles_added": summary.candles_added,
                "candles_total": summary.candles_total,
            }
        )


if __name__ == "__main__":
    main()
