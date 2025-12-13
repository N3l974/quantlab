from __future__ import annotations

from pathlib import Path

import questionary
import typer

from market_data_downloader.cli.parsing import parse_date_to_ms
from market_data_downloader.core.registry import create_source, list_sources
from market_data_downloader.core.timeframes import sort_timeframes_desc
from market_data_downloader.datasets.ohlcv import download_ohlcv
from market_data_downloader.utils.repo import default_data_root


def run_interactive() -> None:
    dataset = questionary.select("Dataset", choices=["ohlcv"]).ask()
    if not dataset:
        raise typer.Abort()

    source = questionary.select("Source", choices=list_sources()).ask()
    if not source:
        raise typer.Abort()

    adapter = create_source(source)

    markets = adapter.list_markets()

    query = questionary.text(
        "Search symbol (ex: BTC, ETH). Leave empty to show first results",
    ).ask()
    if query is None:
        raise typer.Abort()

    q = query.strip()
    if q:
        filtered = [m for m in markets if q.upper() in m.upper()]
    else:
        filtered = markets

    if not filtered:
        typer.echo("No symbol matches your search.")
        raise typer.Abort()

    if len(filtered) > 200:
        filtered = filtered[:200]
        typer.echo("Too many results; showing first 200. Refine search to narrow down.")

    symbol = questionary.select("Symbol", choices=filtered).ask()
    if not symbol:
        raise typer.Abort()

    timeframes = sort_timeframes_desc(adapter.list_timeframes(symbol))
    timeframe = questionary.select("Timeframe", choices=timeframes).ask()
    if not timeframe:
        raise typer.Abort()

    start = questionary.text("Start date (YYYY-MM-DD)").ask()
    end = questionary.text("End date (YYYY-MM-DD)").ask()
    if not start or not end:
        raise typer.Abort()

    start_ms = parse_date_to_ms(start, is_end=False)
    end_ms = parse_date_to_ms(end, is_end=True)

    data_root = default_data_root()

    confirmed = questionary.confirm(
        f"Download {source} {symbol} {timeframe} from {start} to {end} into {data_root}?",
        default=True,
    ).ask()

    if not confirmed:
        raise typer.Abort()

    if dataset != "ohlcv":
        raise typer.Abort()

    summary = download_ohlcv(
        adapter=adapter,
        data_root=Path(data_root),
        symbol=symbol,
        timeframe=timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    typer.echo(
        f"Done. files_written={summary.files_written} candles_added={summary.candles_added} candles_total={summary.candles_total}"
    )
