from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hyperliquid_ohlcv_optimizer.backtest.metrics import max_drawdown_pct, net_return_pct
from hyperliquid_ohlcv_optimizer.backtest.position_manager import PositionManager
from hyperliquid_ohlcv_optimizer.backtest.types import BacktestConfig
from hyperliquid_ohlcv_optimizer.strategies.indicators import atr


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: np.ndarray
    net_return_pct: float
    max_drawdown_pct: float
    trades: pd.DataFrame


def _apply_slippage(price: float, *, side: str, slippage_bps: float) -> float:
    slip = float(slippage_bps) / 10_000.0
    if side == "buy":
        return price * (1.0 + slip)
    if side == "sell":
        return price * (1.0 - slip)
    raise ValueError(f"Unknown side: {side}")


def _fee(amount_notional: float, fee_bps: float) -> float:
    return abs(float(amount_notional)) * (float(fee_bps) / 10_000.0)


def run_backtest(*, df: pd.DataFrame, signal: pd.Series, config: BacktestConfig) -> BacktestResult:
    if df.empty:
        eq = np.array([config.initial_equity], dtype=np.float64)
        return BacktestResult(equity_curve=eq, net_return_pct=0.0, max_drawdown_pct=0.0, trades=pd.DataFrame())

    sig = signal.reindex(df.index).fillna(0).astype("int8")
    entry_intent = sig.shift(1).fillna(0).astype("int8")

    atr_series: pd.Series | None = None
    if str(config.common.sl_type) == "atr":
        atr_series = atr(df["high"], df["low"], df["close"], int(config.common.sl_atr_period))

    pm = PositionManager(config.pm)

    cash = float(config.initial_equity)

    direction = 0  # -1 short, +1 long
    qty = 0.0
    avg_entry = 0.0

    entry_qty0 = 0.0
    peak_qty = 0.0

    tp_price = 0.0
    sl_price = 0.0

    atr_ref = np.nan

    tp1_filled = False
    tp_trail_extreme = np.nan
    position_realized_pnl = 0.0

    equity_curve: list[float] = []
    trades_rows: list[dict] = []

    entry_ts = None

    position_id = 0

    def mark_to_market(close_price: float) -> float:
        if direction == 0:
            return cash
        pnl_unreal = qty * (close_price - avg_entry) * float(direction)
        return cash + pnl_unreal

    def _pm_telemetry() -> dict:
        pm_mode = str(getattr(config.pm, "mode", "none"))
        if pm_mode == "grid":
            try:
                grid_adds_done = int(pm.grid_adds_done())
            except Exception:
                grid_adds_done = 0
        else:
            grid_adds_done = 0

        out = {
            "pm_mode": pm_mode,
            "grid_adds_done": grid_adds_done,
            "entry_qty0": float(entry_qty0),
            "peak_qty": float(peak_qty),
        }
        try:
            out["peak_qty_mult"] = float(peak_qty) / max(float(entry_qty0), 1e-12)
        except Exception:
            out["peak_qty_mult"] = 0.0
        return out

    def recalc_tp_sl() -> None:
        nonlocal tp_price, sl_price

        sl_type = str(config.common.sl_type)
        if sl_type == "atr":
            if atr_series is None:
                sl_price = np.nan
                return
            atr_val = float(atr_ref)
            if not np.isfinite(atr_val):
                sl_price = np.nan
                return
            dist = float(config.common.sl_atr_mult) * atr_val
        else:
            dist = avg_entry * (float(config.common.sl_pct) / 100.0)

        if direction > 0:
            sl_price = avg_entry - dist
        else:
            sl_price = avg_entry + dist

        tp_mode = str(config.common.tp_mode)
        if tp_mode == "rr":
            risk_dist = abs(float(avg_entry) - float(sl_price))
            if (not np.isfinite(risk_dist)) or risk_dist <= 0:
                tp_price = np.nan
                return
            rr = float(config.common.tp_rr)
            if direction > 0:
                tp_price = avg_entry + rr * risk_dist
            else:
                tp_price = avg_entry - rr * risk_dist
        else:
            tp = float(config.common.tp_pct) / 100.0
            if direction > 0:
                tp_price = avg_entry * (1.0 + tp)
            else:
                tp_price = avg_entry * (1.0 - tp)

    def risk_qty_for_entry(entry_price: float, *, atr_value: float | None) -> float:
        risk_cap = cash * float(config.risk.risk_pct)

        sl_type = str(config.common.sl_type)
        if sl_type == "atr":
            if atr_value is None or (not np.isfinite(float(atr_value))):
                return 0.0
            sl_dist = float(config.common.sl_atr_mult) * float(atr_value)
        else:
            sl_dist = entry_price * (float(config.common.sl_pct) / 100.0)

        if (not np.isfinite(sl_dist)) or sl_dist <= 0:
            return 0.0
        q = risk_cap / sl_dist

        max_notional = cash * float(config.risk.max_position_notional_pct_equity) / 100.0
        if max_notional > 0:
            q = min(q, max_notional / max(entry_price, 1e-12))
        return float(max(q, 0.0))

    def cap_qty_by_max_notional(q: float, *, price: float, existing_qty: float = 0.0) -> float:
        max_notional = cash * float(config.risk.max_position_notional_pct_equity) / 100.0
        if max_notional <= 0:
            return float(max(q, 0.0))
        max_total_qty = max_notional / max(float(price), 1e-12)
        remaining = max_total_qty - float(existing_qty)
        return float(max(0.0, min(float(q), float(remaining))))

    for i in range(len(df)):
        row = df.iloc[i]
        ts = int(row["timestamp_ms"])
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        atr_prev = float(atr_series.iat[i - 1]) if (atr_series is not None and i > 0) else None
        atr_ref = float(atr_prev) if atr_prev is not None else np.nan

        intent = int(entry_intent.iat[i])

        if direction == 0 and intent != 0:
            position_id += 1
            direction = 1 if intent > 0 else -1
            side = "buy" if direction > 0 else "sell"
            fill_price = _apply_slippage(o, side=side, slippage_bps=config.costs.slippage_bps)

            base_mult = pm.base_size_multiplier()
            q0 = risk_qty_for_entry(fill_price, atr_value=atr_prev) * base_mult
            q0 = cap_qty_by_max_notional(q0, price=fill_price, existing_qty=0.0)
            qty = float(q0)
            avg_entry = float(fill_price)
            entry_ts = ts

            entry_qty0 = float(qty)
            peak_qty = float(qty)

            tp1_filled = False
            tp_trail_extreme = np.nan
            position_realized_pnl = 0.0

            pm.reset_for_new_position(avg_entry)

            fee = _fee(qty * fill_price, config.costs.fee_bps)
            recalc_tp_sl()
            if fee > cash or qty <= 0 or (not np.isfinite(sl_price)) or (not np.isfinite(tp_price)):
                direction = 0
                qty = 0.0
                avg_entry = 0.0
                entry_ts = None
                entry_qty0 = 0.0
                peak_qty = 0.0
            else:
                cash -= fee

        elif direction != 0 and intent != 0 and (1 if intent > 0 else -1) != direction:
            exit_side = "sell" if direction > 0 else "buy"
            exit_price = _apply_slippage(o, side=exit_side, slippage_bps=config.costs.slippage_bps)
            fee_out = _fee(qty * exit_price, config.costs.fee_bps)
            pnl = qty * (exit_price - avg_entry) * float(direction)
            cash += pnl
            cash -= fee_out
            position_realized_pnl += float(pnl)

            trades_rows.append(
                {
                    "position_id": position_id,
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "exit_reason": "flip",
                    "direction": direction,
                    "qty": qty,
                    "avg_entry": avg_entry,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    **_pm_telemetry(),
                }
            )
            pm.on_trade_closed(position_realized_pnl)

            direction = 0
            qty = 0.0
            avg_entry = 0.0
            tp_price = 0.0
            sl_price = 0.0
            entry_ts = None

            entry_qty0 = 0.0
            peak_qty = 0.0

            tp1_filled = False
            tp_trail_extreme = np.nan
            position_realized_pnl = 0.0

        elif direction != 0 and int(intent) == 0 and bool(config.common.exit_on_flat):
            exit_side = "sell" if direction > 0 else "buy"
            exit_price = _apply_slippage(o, side=exit_side, slippage_bps=config.costs.slippage_bps)
            fee_out = _fee(qty * exit_price, config.costs.fee_bps)
            pnl = qty * (exit_price - avg_entry) * float(direction)
            cash += pnl
            cash -= fee_out
            position_realized_pnl += float(pnl)

            trades_rows.append(
                {
                    "position_id": position_id,
                    "entry_ts": entry_ts,
                    "exit_ts": ts,
                    "exit_reason": "exit_on_flat",
                    "direction": direction,
                    "qty": qty,
                    "avg_entry": avg_entry,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    **_pm_telemetry(),
                }
            )
            pm.on_trade_closed(position_realized_pnl)

            direction = 0
            qty = 0.0
            avg_entry = 0.0
            tp_price = 0.0
            sl_price = 0.0
            entry_ts = None

            tp1_filled = False
            tp_trail_extreme = np.nan
            position_realized_pnl = 0.0

        if direction != 0:
            if bool(config.common.sl_trailing) and i > 0:
                ref_price = float(df.iloc[i - 1]["close"])
                sl_type = str(config.common.sl_type)
                if sl_type == "atr":
                    if atr_prev is not None and np.isfinite(float(atr_prev)):
                        dist = float(config.common.sl_atr_mult) * float(atr_prev)
                    else:
                        dist = None
                else:
                    dist = ref_price * (float(config.common.sl_pct) / 100.0)

                if dist is not None and np.isfinite(float(dist)):
                    sl_candidate = ref_price - float(dist) if direction > 0 else ref_price + float(dist)
                    if direction > 0:
                        sl_price = max(float(sl_price), float(sl_candidate))
                    else:
                        sl_price = min(float(sl_price), float(sl_candidate))

            adverse = l if direction > 0 else h
            favorable = h if direction > 0 else l

            if direction > 0:
                if adverse <= sl_price:
                    exit_price = _apply_slippage(sl_price, side="sell", slippage_bps=config.costs.slippage_bps)
                    fee_out = _fee(qty * exit_price, config.costs.fee_bps)
                    pnl = qty * (exit_price - avg_entry)
                    cash += pnl
                    cash -= fee_out
                    position_realized_pnl += float(pnl)
                    trades_rows.append(
                        {
                            "position_id": position_id,
                            "entry_ts": entry_ts,
                            "exit_ts": ts,
                            "exit_reason": "stop_loss",
                            "direction": direction,
                            "qty": qty,
                            "avg_entry": avg_entry,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            **_pm_telemetry(),
                        }
                    )
                    pm.on_trade_closed(position_realized_pnl)
                    direction = 0
                    qty = 0.0
                    avg_entry = 0.0
                    entry_ts = None
                    entry_qty0 = 0.0
                    peak_qty = 0.0
                else:
                    tp_mgmt = str(config.common.tp_mgmt)
                    if not (tp_mgmt == "partial_trailing" and tp1_filled):
                        while pm.can_add_grid():
                            ngp = pm.next_grid_price(direction=direction)
                            if ngp is None or adverse > ngp:
                                break

                            add_price = _apply_slippage(float(ngp), side="buy", slippage_bps=config.costs.slippage_bps)
                            add_qty = risk_qty_for_entry(add_price, atr_value=atr_prev) * pm.grid_add_qty_multiplier()
                            add_qty = cap_qty_by_max_notional(add_qty, price=add_price, existing_qty=qty)
                            if add_qty <= 0:
                                break

                            new_qty = qty + float(add_qty)
                            avg_entry = (avg_entry * qty + add_price * float(add_qty)) / max(new_qty, 1e-12)
                            qty = new_qty

                            peak_qty = max(float(peak_qty), float(qty))
                            pm.on_grid_filled(add_price)

                            fee_add = _fee(add_qty * add_price, config.costs.fee_bps)
                            if fee_add > cash:
                                break
                            cash -= fee_add
                            recalc_tp_sl()

                            if adverse <= sl_price:
                                break

                    if direction != 0 and tp_mgmt == "partial_trailing":
                        if (not tp1_filled) and favorable >= tp_price:
                            close_frac = float(config.common.tp1_close_frac)
                            close_qty = float(qty) * close_frac
                            close_qty = float(max(0.0, min(float(close_qty), float(qty))))
                            if close_qty >= float(qty) or close_qty <= 0:
                                close_qty = float(qty)

                            exit_price = _apply_slippage(tp_price, side="sell", slippage_bps=config.costs.slippage_bps)
                            fee_out = _fee(close_qty * exit_price, config.costs.fee_bps)
                            pnl = close_qty * (exit_price - avg_entry)
                            cash += pnl
                            cash -= fee_out
                            position_realized_pnl += float(pnl)

                            trades_rows.append(
                                {
                                    "position_id": position_id,
                                    "entry_ts": entry_ts,
                                    "exit_ts": ts,
                                    "exit_reason": "tp1_partial",
                                    "direction": direction,
                                    "qty": close_qty,
                                    "avg_entry": avg_entry,
                                    "exit_price": exit_price,
                                    "pnl": pnl,
                                    **_pm_telemetry(),
                                }
                            )

                            qty = float(qty) - float(close_qty)
                            if qty <= 0:
                                pm.on_trade_closed(position_realized_pnl)
                                direction = 0
                                qty = 0.0
                                avg_entry = 0.0
                                entry_ts = None
                                tp1_filled = False
                                tp_trail_extreme = np.nan
                                position_realized_pnl = 0.0
                                entry_qty0 = 0.0
                                peak_qty = 0.0
                            else:
                                tp1_filled = True
                                tp_trail_extreme = float(tp_price)

                        elif tp1_filled and i > 0:
                            ref_price = float(df.iloc[i - 1]["close"])
                            tp_trail_extreme = (
                                max(float(tp_trail_extreme), float(ref_price))
                                if np.isfinite(float(tp_trail_extreme))
                                else float(ref_price)
                            )
                            trail_pct = float(config.common.tp_trail_pct) / 100.0
                            trail_price = float(tp_trail_extreme) * (1.0 - trail_pct)
                            if adverse <= trail_price:
                                exit_price = _apply_slippage(trail_price, side="sell", slippage_bps=config.costs.slippage_bps)
                                fee_out = _fee(qty * exit_price, config.costs.fee_bps)
                                pnl = qty * (exit_price - avg_entry)
                                cash += pnl
                                cash -= fee_out
                                position_realized_pnl += float(pnl)
                                trades_rows.append(
                                    {
                                        "position_id": position_id,
                                        "entry_ts": entry_ts,
                                        "exit_ts": ts,
                                        "exit_reason": "tp_trailing",
                                        "direction": direction,
                                        "qty": qty,
                                        "avg_entry": avg_entry,
                                        "exit_price": exit_price,
                                        "pnl": pnl,
                                        **_pm_telemetry(),
                                    }
                                )
                                pm.on_trade_closed(position_realized_pnl)
                                direction = 0
                                qty = 0.0
                                avg_entry = 0.0
                                entry_ts = None
                                tp1_filled = False
                                tp_trail_extreme = np.nan
                                position_realized_pnl = 0.0
                                entry_qty0 = 0.0
                                peak_qty = 0.0

                    if direction != 0 and tp_mgmt != "partial_trailing" and favorable >= tp_price:
                        exit_price = _apply_slippage(tp_price, side="sell", slippage_bps=config.costs.slippage_bps)
                        fee_out = _fee(qty * exit_price, config.costs.fee_bps)
                        pnl = qty * (exit_price - avg_entry)
                        cash += pnl
                        cash -= fee_out
                        position_realized_pnl += float(pnl)
                        trades_rows.append(
                            {
                                "position_id": position_id,
                                "entry_ts": entry_ts,
                                "exit_ts": ts,
                                "exit_reason": "tp_full",
                                "direction": direction,
                                "qty": qty,
                                "avg_entry": avg_entry,
                                "exit_price": exit_price,
                                "pnl": pnl,
                                **_pm_telemetry(),
                            }
                        )
                        pm.on_trade_closed(position_realized_pnl)
                        direction = 0
                        qty = 0.0
                        avg_entry = 0.0
                        entry_ts = None
                        tp1_filled = False
                        tp_trail_extreme = np.nan
                        entry_qty0 = 0.0
                        peak_qty = 0.0
                        position_realized_pnl = 0.0

            else:
                if adverse >= sl_price:
                    exit_price = _apply_slippage(sl_price, side="buy", slippage_bps=config.costs.slippage_bps)
                    fee_out = _fee(qty * exit_price, config.costs.fee_bps)
                    pnl = qty * (avg_entry - exit_price)
                    cash += pnl
                    cash -= fee_out
                    position_realized_pnl += float(pnl)
                    trades_rows.append(
                        {
                            "position_id": position_id,
                            "entry_ts": entry_ts,
                            "exit_ts": ts,
                            "exit_reason": "stop_loss",
                            "direction": direction,
                            "qty": qty,
                            "avg_entry": avg_entry,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            **_pm_telemetry(),
                        }
                    )
                    pm.on_trade_closed(position_realized_pnl)
                    direction = 0
                    qty = 0.0
                    avg_entry = 0.0
                    entry_ts = None

                    entry_qty0 = 0.0
                    peak_qty = 0.0

                    tp1_filled = False
                    tp_trail_extreme = np.nan
                    position_realized_pnl = 0.0
                else:
                    tp_mgmt = str(config.common.tp_mgmt)
                    if not (tp_mgmt == "partial_trailing" and tp1_filled):
                        while pm.can_add_grid():
                            ngp = pm.next_grid_price(direction=direction)
                            if ngp is None or adverse < ngp:
                                break

                            add_price = _apply_slippage(float(ngp), side="sell", slippage_bps=config.costs.slippage_bps)
                            add_qty = risk_qty_for_entry(add_price, atr_value=atr_prev) * pm.grid_add_qty_multiplier()
                            add_qty = cap_qty_by_max_notional(add_qty, price=add_price, existing_qty=qty)
                            if add_qty <= 0:
                                break

                            new_qty = qty + float(add_qty)
                            avg_entry = (avg_entry * qty + add_price * float(add_qty)) / max(new_qty, 1e-12)
                            qty = new_qty

                            peak_qty = max(float(peak_qty), float(qty))
                            pm.on_grid_filled(add_price)

                            fee_add = _fee(add_qty * add_price, config.costs.fee_bps)
                            if fee_add > cash:
                                break
                            cash -= fee_add
                            recalc_tp_sl()

                            if adverse >= sl_price:
                                break

                    if direction != 0 and tp_mgmt == "partial_trailing":
                        if (not tp1_filled) and favorable <= tp_price:
                            close_frac = float(config.common.tp1_close_frac)
                            close_qty = float(qty) * close_frac
                            close_qty = float(max(0.0, min(float(close_qty), float(qty))))
                            if close_qty >= float(qty) or close_qty <= 0:
                                close_qty = float(qty)

                            exit_price = _apply_slippage(tp_price, side="buy", slippage_bps=config.costs.slippage_bps)
                            fee_out = _fee(close_qty * exit_price, config.costs.fee_bps)
                            pnl = close_qty * (avg_entry - exit_price)
                            cash += pnl
                            cash -= fee_out
                            position_realized_pnl += float(pnl)

                            trades_rows.append(
                                {
                                    "position_id": position_id,
                                    "entry_ts": entry_ts,
                                    "exit_ts": ts,
                                    "exit_reason": "tp1_partial",
                                    "direction": direction,
                                    "qty": close_qty,
                                    "avg_entry": avg_entry,
                                    "exit_price": exit_price,
                                    "pnl": pnl,
                                    **_pm_telemetry(),
                                }
                            )

                            qty = float(qty) - float(close_qty)
                            if qty <= 0:
                                pm.on_trade_closed(position_realized_pnl)
                                direction = 0
                                qty = 0.0
                                avg_entry = 0.0
                                entry_ts = None
                                tp1_filled = False
                                tp_trail_extreme = np.nan
                                position_realized_pnl = 0.0
                            else:
                                tp1_filled = True
                                tp_trail_extreme = float(tp_price)

                        elif tp1_filled and i > 0:
                            ref_price = float(df.iloc[i - 1]["close"])
                            tp_trail_extreme = (
                                min(float(tp_trail_extreme), float(ref_price))
                                if np.isfinite(float(tp_trail_extreme))
                                else float(ref_price)
                            )
                            trail_pct = float(config.common.tp_trail_pct) / 100.0
                            trail_price = float(tp_trail_extreme) * (1.0 + trail_pct)
                            if adverse >= trail_price:
                                exit_price = _apply_slippage(trail_price, side="buy", slippage_bps=config.costs.slippage_bps)
                                fee_out = _fee(qty * exit_price, config.costs.fee_bps)
                                pnl = qty * (avg_entry - exit_price)
                                cash += pnl
                                cash -= fee_out
                                position_realized_pnl += float(pnl)
                                trades_rows.append(
                                    {
                                        "position_id": position_id,
                                        "entry_ts": entry_ts,
                                        "exit_ts": ts,
                                        "exit_reason": "tp_trailing",
                                        "direction": direction,
                                        "qty": qty,
                                        "avg_entry": avg_entry,
                                        "exit_price": exit_price,
                                        "pnl": pnl,
                                        **_pm_telemetry(),
                                    }
                                )
                                pm.on_trade_closed(position_realized_pnl)
                                direction = 0
                                qty = 0.0
                                avg_entry = 0.0
                                entry_ts = None
                                tp1_filled = False
                                tp_trail_extreme = np.nan
                                position_realized_pnl = 0.0
                                entry_qty0 = 0.0
                                peak_qty = 0.0

                    if direction != 0 and tp_mgmt != "partial_trailing" and favorable <= tp_price:
                        exit_price = _apply_slippage(tp_price, side="buy", slippage_bps=config.costs.slippage_bps)
                        fee_out = _fee(qty * exit_price, config.costs.fee_bps)
                        pnl = qty * (avg_entry - exit_price)
                        cash += pnl
                        cash -= fee_out
                        position_realized_pnl += float(pnl)
                        trades_rows.append(
                            {
                                "position_id": position_id,
                                "entry_ts": entry_ts,
                                "exit_ts": ts,
                                "exit_reason": "tp_full",
                                "direction": direction,
                                "qty": qty,
                                "avg_entry": avg_entry,
                                "exit_price": exit_price,
                                "pnl": pnl,
                                **_pm_telemetry(),
                            }
                        )
                        pm.on_trade_closed(position_realized_pnl)
                        direction = 0
                        qty = 0.0
                        avg_entry = 0.0
                        entry_ts = None
                        tp1_filled = False
                        tp_trail_extreme = np.nan
                        entry_qty0 = 0.0
                        peak_qty = 0.0
                        position_realized_pnl = 0.0

        equity_curve.append(mark_to_market(c))

    eq = np.array(equity_curve, dtype=np.float64)
    trades = pd.DataFrame(trades_rows)

    res = BacktestResult(
        equity_curve=eq,
        net_return_pct=net_return_pct(eq),
        max_drawdown_pct=max_drawdown_pct(eq),
        trades=trades,
    )
    return res
