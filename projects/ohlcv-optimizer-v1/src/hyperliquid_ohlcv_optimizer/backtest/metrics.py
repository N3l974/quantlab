from __future__ import annotations

import numpy as np


def max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0

    peaks = np.maximum.accumulate(equity)
    dd = (peaks - equity) / np.maximum(peaks, 1e-12)
    return float(dd.max() * 100.0)


def net_return_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0

    start = float(equity[0])
    end = float(equity[-1])
    if start <= 0:
        return 0.0

    return (end / start - 1.0) * 100.0
