from __future__ import annotations

from dataclasses import dataclass

from market_data_downloader.sources.base import SourceAdapter


@dataclass(frozen=True)
class SourceSpec:
    name: str
    adapter_cls: type[SourceAdapter]


_SOURCES: dict[str, SourceSpec] = {}


def register_source(name: str, adapter_cls: type[SourceAdapter]) -> None:
    key = name.strip()
    if not key:
        raise ValueError("Source name cannot be empty")
    if key in _SOURCES:
        raise ValueError(f"Source already registered: {key}")

    _SOURCES[key] = SourceSpec(name=key, adapter_cls=adapter_cls)


def list_sources() -> list[str]:
    return sorted(_SOURCES.keys())


def get_source(name: str) -> SourceSpec:
    key = name.strip()
    try:
        return _SOURCES[key]
    except KeyError as e:
        raise ValueError(f"Unknown source: {key}") from e


def create_source(name: str) -> SourceAdapter:
    spec = get_source(name)
    return spec.adapter_cls()
