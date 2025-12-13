from __future__ import annotations

import logging
from pathlib import Path

import typer

from market_data_downloader.cli.interactive import run_interactive
from market_data_downloader.cli.parsing import parse_date_to_ms
from market_data_downloader.core.registry import create_source, list_sources
from market_data_downloader.datasets.ohlcv import download_ohlcv
from market_data_downloader.sources.builtins import register_builtin_sources
from market_data_downloader.utils.repo import default_data_root

app = typer.Typer(add_completion=False)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    register_builtin_sources()

    if ctx.invoked_subcommand is None:
        run_interactive()


@app.command()
def sources() -> None:
    """List available sources."""

    register_builtin_sources()
    for s in list_sources():
        typer.echo(s)


@app.command()
def download(
    dataset: str = typer.Option("ohlcv", "--dataset", "-d"),
    source: str = typer.Option(..., "--source"),
    symbol: str = typer.Option(..., "--symbol"),
    timeframe: str = typer.Option(..., "--timeframe", "--tf"),
    start: str = typer.Option(..., "--start"),
    end: str = typer.Option(..., "--end"),
    data_root: Path | None = typer.Option(None, "--data-root"),
) -> None:
    """Download a dataset and store in Parquet partitioned by month."""

    register_builtin_sources()
    adapter = create_source(source)

    start_ms = parse_date_to_ms(start, is_end=False)
    end_ms = parse_date_to_ms(end, is_end=True)

    root = data_root or default_data_root()

    if dataset != "ohlcv":
        raise typer.BadParameter("Only dataset 'ohlcv' is implemented for now")

    summary = download_ohlcv(
        adapter=adapter,
        data_root=root,
        symbol=symbol,
        timeframe=timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    typer.echo(
        f"Done. files_written={summary.files_written} candles_added={summary.candles_added} candles_total={summary.candles_total}"
    )
