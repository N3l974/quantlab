from __future__ import annotations

import re


def to_path_component(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Empty value cannot be converted to a path component")

    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)
