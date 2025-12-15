from __future__ import annotations

from dataclasses import dataclass

from hyperliquid_ohlcv_optimizer.backtest.types import GridConfig, MartingaleConfig, PositionManagerConfig


@dataclass
class MartingaleState:
    step: int = 0


@dataclass
class GridState:
    adds_done: int = 0
    last_fill_price: float | None = None


class PositionManager:
    def __init__(self, config: PositionManagerConfig) -> None:
        self._cfg = config
        self._martingale = MartingaleState()
        self._grid = GridState()

    def reset_for_new_position(self, entry_price: float) -> None:
        self._grid = GridState(adds_done=0, last_fill_price=entry_price)

    def base_size_multiplier(self) -> float:
        if self._cfg.mode == "martingale":
            mc: MartingaleConfig = self._cfg.martingale  # type: ignore[assignment]
            step = min(self._martingale.step, mc.max_steps)
            return float(mc.multiplier) ** int(step)

        return 1.0

    def on_trade_closed(self, pnl: float) -> None:
        if self._cfg.mode != "martingale":
            return

        mc: MartingaleConfig = self._cfg.martingale  # type: ignore[assignment]
        if pnl < 0:
            self._martingale.step = min(self._martingale.step + 1, mc.max_steps)
        else:
            self._martingale.step = 0

    def can_add_grid(self) -> bool:
        if self._cfg.mode != "grid":
            return False
        gc: GridConfig = self._cfg.grid  # type: ignore[assignment]
        return self._grid.adds_done < int(gc.max_adds)

    def next_grid_price(self, *, direction: int) -> float | None:
        if self._cfg.mode != "grid":
            return None

        last = self._grid.last_fill_price
        if last is None:
            return None

        gc: GridConfig = self._cfg.grid  # type: ignore[assignment]
        spacing = float(gc.spacing_pct) / 100.0

        if direction > 0:
            return last * (1.0 - spacing)
        else:
            return last * (1.0 + spacing)

    def grid_add_qty_multiplier(self) -> float:
        if self._cfg.mode != "grid":
            return 1.0

        gc: GridConfig = self._cfg.grid  # type: ignore[assignment]
        i = self._grid.adds_done
        return float(gc.size_multiplier) ** int(i)

    def on_grid_filled(self, fill_price: float) -> None:
        if self._cfg.mode != "grid":
            return

        self._grid.adds_done += 1
        self._grid.last_fill_price = float(fill_price)
