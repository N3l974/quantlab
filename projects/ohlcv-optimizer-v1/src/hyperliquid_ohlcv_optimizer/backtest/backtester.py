from __future__ import annotations

from dataclasses import dataclass
import math

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
    max_drawdown_intrabar_pct: float
    exec_reject_rate: float
    exec_round_rate: float
    liquidated: bool
    liquidation_ts: int | None
    peak_notional: float
    peak_notional_pct_equity: float
    peak_qty_mult: float
    cap_hit_rate: float
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
        return BacktestResult(
            equity_curve=eq,
            net_return_pct=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_intrabar_pct=0.0,
            exec_reject_rate=0.0,
            exec_round_rate=0.0,
            liquidated=False,
            liquidation_ts=None,
            peak_notional=0.0,
            peak_notional_pct_equity=0.0,
            peak_qty_mult=0.0,
            cap_hit_rate=0.0,
            trades=pd.DataFrame(),
        )

    sig = signal.reindex(df.index).fillna(0).astype("int8")
    entry_intent = sig.shift(1).fillna(0).astype("int8")

    atr_series: pd.Series | None = None
    if str(config.common.sl_type) == "atr":
        atr_series = atr(df["high"], df["low"], df["close"], int(config.common.sl_atr_period))

    pm = PositionManager(config.pm)

    cash = float(config.initial_equity)

    exec_ops = 0
    exec_rejects = 0
    exec_rounds = 0

    cap_ops = 0
    cap_hits = 0

    direction = 0  # -1 short, +1 long
    qty = 0.0
    avg_entry = 0.0

    def risk_qty_for_entry(entry_price: float, *, atr_value: float | None) -> float:
        mode = str(getattr(config.risk, "mode", "risk") or "risk").strip().lower()
        if mode in {"fixed", "notional", "fixed_notional"}:
            notional = cash * (float(getattr(config.risk, "fixed_notional_pct_equity", 0.0) or 0.0) / 100.0)
            if (not np.isfinite(float(notional))) or float(notional) <= 0:
                return 0.0
            q = float(notional) / max(float(entry_price), 1e-12)
        else:
            risk_cap = cash * float(getattr(config.risk, "risk_pct", 0.0) or 0.0)

            sl_type = str(config.common.sl_type)
            if sl_type == "atr":
                if atr_value is None or (not np.isfinite(float(atr_value))):
                    return 0.0
                sl_dist = float(config.common.sl_atr_mult) * float(atr_value)
            else:
                sl_dist = entry_price * (float(config.common.sl_pct) / 100.0)

            if (not np.isfinite(sl_dist)) or sl_dist <= 0:
                return 0.0
            if (not np.isfinite(float(risk_cap))) or float(risk_cap) <= 0:
                return 0.0
            q = risk_cap / sl_dist

        max_notional = cash * float(config.risk.max_position_notional_pct_equity) / 100.0
        if max_notional > 0:
            q = min(q, max_notional / max(entry_price, 1e-12))
        return float(max(q, 0.0))

    def cap_qty_by_max_notional(q: float, *, price: float, existing_qty: float = 0.0) -> float:
        nonlocal cap_ops, cap_hits
        cap_ops += 1
        max_notional = cash * float(config.risk.max_position_notional_pct_equity) / 100.0
        if max_notional <= 0:
            return float(max(q, 0.0))
        max_total_qty = max_notional / max(float(price), 1e-12)
        remaining = max_total_qty - float(existing_qty)
        out = float(max(0.0, min(float(q), float(remaining))))
        if float(out) + 1e-12 < float(q):
            cap_hits += 1
        return float(out)

    def apply_execution_constraints(q: float, *, price: float) -> float:
        nonlocal exec_ops, exec_rejects, exec_rounds
        exec_ops += 1

        out = float(max(q, 0.0))
        if out <= 0:
            exec_rejects += 1
            return 0.0

        step = float(getattr(config.execution, "qty_step", 0.0) or 0.0)
        if np.isfinite(step) and step > 0:
            out2 = math.floor(out / step) * step
            if out2 + 1e-12 < out:
                exec_rounds += 1
            out = float(max(out2, 0.0))

        min_qty = float(getattr(config.execution, "min_qty", 0.0) or 0.0)
        if np.isfinite(min_qty) and min_qty > 0 and out + 1e-12 < min_qty:
            exec_rejects += 1
            return 0.0

        min_notional = float(getattr(config.execution, "min_notional", 0.0) or 0.0)
        notional = out * float(price)
        if np.isfinite(min_notional) and min_notional > 0 and notional + 1e-8 < min_notional:
            exec_rejects += 1
            return 0.0

        return float(out)

    position_id = 0
    entry_ts: int | None = None
    tp_price = np.nan
    sl_price = np.nan

    entry_qty0 = 0.0
    peak_qty = 0.0

    global_peak_qty_mult = 0.0
    peak_notional = 0.0
    peak_notional_pct_equity = 0.0

    liquidated = False
    liquidation_ts: int | None = None

    equity_curve: list[float] = []
    equity_worst_curve: list[float] = []
    trades_rows: list[dict] = []

    tp1_filled = False
    tp_trail_extreme = np.nan
    position_realized_pnl = 0.0

    atr_ref = np.nan

    def mark_to_market(price: float) -> float:
        if direction == 0 or qty <= 0:
            return float(cash)
        pnl_u = float(qty) * (float(price) - float(avg_entry)) * float(direction)
        return float(cash) + float(pnl_u)

    def worst_mtm(high: float, low: float) -> float:
        if direction == 0 or qty <= 0:
            return float(cash)
        adverse_price = float(low) if direction > 0 else float(high)
        return float(mark_to_market(adverse_price))

    def recalc_tp_sl() -> None:
        nonlocal tp_price, sl_price

        if direction == 0 or qty <= 0 or (not np.isfinite(float(avg_entry))):
            tp_price = np.nan
            sl_price = np.nan
            return

        sl_type = str(config.common.sl_type)
        sl_dist: float | None
        if sl_type == "atr":
            if not np.isfinite(float(atr_ref)):
                sl_dist = None
            else:
                sl_dist = float(config.common.sl_atr_mult) * float(atr_ref)
        else:
            sl_dist = float(avg_entry) * (float(config.common.sl_pct) / 100.0)

        if sl_dist is None or (not np.isfinite(float(sl_dist))) or float(sl_dist) <= 0:
            tp_price = np.nan
            sl_price = np.nan
            return

        if direction > 0:
            sl_price = float(avg_entry) - float(sl_dist)
        else:
            sl_price = float(avg_entry) + float(sl_dist)

        tp_mode = str(config.common.tp_mode)
        if tp_mode == "rr":
            tp_dist = float(sl_dist) * float(config.common.tp_rr)
        else:
            tp_dist = float(avg_entry) * (float(config.common.tp_pct) / 100.0)

        if (not np.isfinite(float(tp_dist))) or float(tp_dist) <= 0:
            tp_price = np.nan
            return

        if direction > 0:
            tp_price = float(avg_entry) + float(tp_dist)
        else:
            tp_price = float(avg_entry) - float(tp_dist)

    def _pm_telemetry() -> dict:
        try:
            return {
                "pm_mode": str(getattr(config.pm, "mode", "none")),
                "grid_adds_done": int(pm.grid_adds_done()),
            }
        except Exception:
            return {}

    def _update_exposure_metrics(*, price: float, eq_ref: float) -> None:
        nonlocal peak_notional, peak_notional_pct_equity, global_peak_qty_mult, peak_qty
        if direction == 0 or qty <= 0:
            return
        notional = abs(float(qty)) * float(price)
        if np.isfinite(float(notional)):
            peak_notional = float(max(float(peak_notional), float(notional)))
            if np.isfinite(float(eq_ref)) and float(eq_ref) > 0:
                peak_notional_pct_equity = float(
                    max(float(peak_notional_pct_equity), (float(notional) / float(eq_ref)) * 100.0)
                )

        peak_qty = float(max(float(peak_qty), float(qty)))
        if float(entry_qty0) > 0:
            qm = float(qty) / float(entry_qty0)
            if np.isfinite(float(qm)):
                global_peak_qty_mult = float(max(float(global_peak_qty_mult), abs(float(qm))))

    def _should_liquidate(*, eq_worst: float, price: float) -> bool:
        if direction == 0 or qty <= 0:
            return False
        notional = abs(float(qty)) * float(price)
        if (not np.isfinite(float(notional))) or float(notional) <= 0:
            return False

        prof = str(getattr(config.broker, "profile", "perps") or "perps").strip().lower()
        if prof == "cfd":
            used_margin = float(getattr(config.broker, "cfd_initial_margin_rate", 0.01) or 0.01) * float(notional)
            if used_margin <= 0:
                return False
            ml = float(eq_worst) / float(used_margin)
            return float(ml) <= float(getattr(config.broker, "cfd_stopout_margin_level", 0.5) or 0.5)

        if prof == "spot":
            return float(eq_worst) <= 0.0

        mmr = float(getattr(config.broker, "perps_maintenance_margin_rate", 0.01) or 0.01)
        return float(eq_worst) <= float(mmr) * float(notional)

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
            q0 = apply_execution_constraints(q0, price=fill_price)
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
                            add_qty = apply_execution_constraints(add_qty, price=add_price)
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
                            close_qty = apply_execution_constraints(close_qty, price=tp_price)
                            if close_qty >= float(qty) or close_qty <= 0:
                                tp1_filled = True
                            else:
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
                            add_qty = apply_execution_constraints(add_qty, price=add_price)
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
                            close_qty = apply_execution_constraints(close_qty, price=tp_price)
                            if close_qty >= float(qty) or close_qty <= 0:
                                tp1_filled = True
                            else:
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

        eq_close = float(mark_to_market(c))
        eq_worst = float(worst_mtm(h, l))

        _update_exposure_metrics(price=c, eq_ref=eq_close)
        _update_exposure_metrics(price=(float(l) if direction > 0 else float(h)), eq_ref=eq_worst)

        if (not liquidated) and direction != 0 and _should_liquidate(eq_worst=eq_worst, price=(float(l) if direction > 0 else float(h))):
            liquidated = True
            liquidation_ts = int(ts)
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
            cash = max(0.0, float(eq_worst))

            eq_close = float(cash)
            eq_worst = float(cash)
            equity_curve.append(eq_close)
            equity_worst_curve.append(eq_worst)
            break

        equity_curve.append(eq_close)
        equity_worst_curve.append(eq_worst)

    eq = np.array(equity_curve, dtype=np.float64)
    eq_w = np.array(equity_worst_curve, dtype=np.float64)
    trades = pd.DataFrame(trades_rows)

    exec_reject_rate = 0.0
    exec_round_rate = 0.0
    if int(exec_ops) > 0:
        exec_reject_rate = float(exec_rejects) / float(exec_ops)
        exec_round_rate = float(exec_rounds) / float(exec_ops)

    cap_hit_rate = 0.0
    if int(cap_ops) > 0:
        cap_hit_rate = float(cap_hits) / float(cap_ops)

    res = BacktestResult(
        equity_curve=eq,
        net_return_pct=net_return_pct(eq),
        max_drawdown_pct=max_drawdown_pct(eq),
        max_drawdown_intrabar_pct=max_drawdown_pct(eq_w),
        exec_reject_rate=float(exec_reject_rate),
        exec_round_rate=float(exec_round_rate),
        liquidated=bool(liquidated),
        liquidation_ts=liquidation_ts,
        peak_notional=float(peak_notional),
        peak_notional_pct_equity=float(peak_notional_pct_equity),
        peak_qty_mult=float(global_peak_qty_mult),
        cap_hit_rate=float(cap_hit_rate),
        trades=trades,
    )
    return res
