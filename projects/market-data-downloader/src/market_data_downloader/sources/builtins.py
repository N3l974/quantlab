from __future__ import annotations

from market_data_downloader.core.registry import register_source
from market_data_downloader.sources.binance_futures import BinanceFuturesSource
from market_data_downloader.sources.hyperliquid_perps import HyperliquidPerpsSource
from market_data_downloader.sources.kraken_futures import KrakenFuturesSource
from market_data_downloader.sources.kraken_spot import KrakenSpotSource
from market_data_downloader.sources.mt5_icmarkets import Mt5IcmarketsSource


_registered = False


def register_builtin_sources() -> None:
    global _registered
    if _registered:
        return

    register_source(BinanceFuturesSource.name, BinanceFuturesSource)
    register_source(HyperliquidPerpsSource.name, HyperliquidPerpsSource)
    register_source(KrakenFuturesSource.name, KrakenFuturesSource)
    register_source(KrakenSpotSource.name, KrakenSpotSource)
    register_source(Mt5IcmarketsSource.name, Mt5IcmarketsSource)
    _registered = True
