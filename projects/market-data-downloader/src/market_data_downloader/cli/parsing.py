from __future__ import annotations

from datetime import datetime, timedelta, timezone

from dateutil import parser


def _is_date_only(value: str) -> bool:
    v = value.strip()
    return ("T" not in v) and (":" not in v)


def parse_date_to_ms(value: str, *, is_end: bool) -> int:
    if not value or not value.strip():
        raise ValueError("Empty date")

    dt = parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    if is_end and _is_date_only(value):
        dt = dt + timedelta(days=1)

    return int(dt.timestamp() * 1000)
