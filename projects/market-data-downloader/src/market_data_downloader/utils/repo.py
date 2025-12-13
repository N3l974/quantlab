from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()

    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p

    raise RuntimeError("Could not locate repo root (missing .git directory).")


def default_data_root() -> Path:
    return find_repo_root() / "data" / "market_data"
