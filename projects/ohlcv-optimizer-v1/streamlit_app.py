from __future__ import annotations

import json
import queue
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st
import pandas as pd
import optuna
from dateutil import parser

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

from hyperliquid_ohlcv_optimizer.backtest.backtester import run_backtest
from hyperliquid_ohlcv_optimizer.backtest.trade_analysis import summarize_positions
from hyperliquid_ohlcv_optimizer.data.ohlcv_loader import discover_symbols, load_ohlcv
from hyperliquid_ohlcv_optimizer.optimize.optuna_runner import (
    OptimizationConfig,
    build_report_from_storage,
    build_backtest_config_from_params,
    run_optimization,
)
from hyperliquid_ohlcv_optimizer.strategies.base import StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.registry import builtin_strategies
from hyperliquid_ohlcv_optimizer.utils.repo import default_data_root


def _parse_date_to_ms(value: str, *, is_end: bool) -> int:
    dt = parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    if is_end:
        dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    return int(dt.timestamp() * 1000)


def _ms_to_yyyy_mm_dd(ms: int) -> str:
    return pd.to_datetime(int(ms), unit="ms", utc=True).strftime("%Y-%m-%d")


def _count_trials_for_run(*, storage_url: str, study_name_prefix: str) -> int | None:
    storage_url = str(storage_url or "").strip()
    prefix = f"{str(study_name_prefix).strip()}."
    try:
        summaries = optuna.study.get_all_study_summaries(storage=storage_url)
    except Exception:
        return None

    total = 0
    matched = 0
    for s in summaries:
        try:
            name = str(s.study_name)
            if name.startswith(prefix) or name == str(study_name_prefix):
                total += int(s.n_trials or 0)
                matched += 1
        except Exception:
            pass

    if matched > 0:
        return int(total)

    if storage_url.lower().startswith("sqlite"):
        try:
            return int(sum(int(s.n_trials or 0) for s in summaries))
        except Exception:
            return None

    return None


def _recommend_candidate_settings(*, max_trials: int, n_strats: int, stop_mode: str) -> tuple[str, int]:
    n_strats = max(1, int(n_strats) if n_strats is not None else 1)
    sm = str(stop_mode or "").strip().lower()

    per = int(round(400.0 / float(n_strats)))
    per = max(30, min(200, per))

    pool = "pareto"
    if sm != "time":
        try:
            mt = int(max_trials)
        except Exception:
            mt = 0
        total = mt * n_strats if mt > 0 else 0
        if (mt > 0) and (mt <= 300) and (total <= 1200):
            pool = "complete"

        if mt > 0:
            per = min(per, mt)
    else:
        per = min(per, 100)

    return pool, per


def _set_period(*, start: str, end: str) -> None:
    st.session_state["start_date"] = start
    st.session_state["end_date"] = end


@st.cache_data(show_spinner=False)
def _load_ohlcv_cached(*, data_root: str, source: str, symbol: str, timeframe: str) -> pd.DataFrame:
    return load_ohlcv(data_root=Path(data_root), source=source, symbol=symbol, timeframe=timeframe)


def _discover_saved_runs() -> list[Path]:
    runs_root = _PROJECT_ROOT / "runs"
    if not runs_root.exists():
        return []
    out: list[Path] = []
    for p in runs_root.iterdir():
        if p.is_dir():
            out.append(p)
    out.sort(key=lambda x: x.name, reverse=True)
    return out


def _run_display_name(p: Path) -> str:
    if (p / "report.json").exists():
        return p.name
    if (p / "optuna.db").exists() and (p / "context.json").exists():
        return f"{p.name} (incomplete)"
    missing: list[str] = []
    if not (p / "report.json").exists():
        missing.append("report.json")
    if not (p / "optuna.db").exists():
        missing.append("optuna.db")
    if not (p / "context.json").exists():
        missing.append("context.json")
    return f"{p.name} (missing: {', '.join(missing)})"


def _load_report_json(run_dir: Path) -> dict | None:
    p = run_dir / "report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_context_json(run_dir: Path) -> dict | None:
    p = run_dir / "context.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_status_json(run_dir: Path) -> dict | None:
    p = run_dir / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _discover_running_runs() -> list[Path]:
    runs_root = _PROJECT_ROOT / "runs"
    if not runs_root.exists():
        return []
    out: list[Path] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        st_payload = _load_status_json(p)
        if not isinstance(st_payload, dict):
            continue
        state = str(st_payload.get("state") or "").strip().lower()
        if state in {"running", "stopping", "building_report"}:
            out.append(p)
    out.sort(key=lambda x: x.name, reverse=True)
    return out


def _get_query_param(name: str) -> str | None:
    if hasattr(st, "query_params"):
        try:
            v = st.query_params.get(name)
            if isinstance(v, list):
                return str(v[0]) if v else None
            if v is None:
                return None
            return str(v)
        except Exception:
            pass
    if hasattr(st, "experimental_get_query_params"):
        try:
            d = st.experimental_get_query_params()
            v = d.get(name)
            if isinstance(v, list):
                return str(v[0]) if v else None
            if v is None:
                return None
            return str(v)
        except Exception:
            return None
    return None


def _sharpe_from_equity(equity) -> float:
    try:
        eq = pd.Series(equity).astype("float64").to_numpy()
    except Exception:
        return 0.0
    if eq.size < 2:
        return 0.0
    rets = (eq[1:] - eq[:-1]) / (pd.Series(eq[:-1]).replace(0, 1e-12).to_numpy())
    mu = float(pd.Series(rets).mean()) if rets.size else 0.0
    sigma = float(pd.Series(rets).std(ddof=0)) if rets.size else 0.0
    if sigma <= 1e-12:
        return 0.0
    return float((mu / sigma) * (float(rets.size) ** 0.5))


def _load_df_for_report(*, report_payload: dict, data_root_override: Path | None) -> pd.DataFrame:
    data = report_payload.get("data") or {}
    data_root = str(data_root_override) if data_root_override is not None else str(data.get("data_root") or default_data_root())
    source = str(data.get("source") or "hyperliquid_perps")
    symbol = str(data.get("symbol_storage") or data.get("symbol") or "BTC")
    timeframe = str(data.get("timeframe") or "5m")

    df = _load_ohlcv_cached(data_root=data_root, source=source, symbol=symbol, timeframe=timeframe)

    start_ms = data.get("start_ms")
    end_ms = data.get("end_ms")

    if start_ms is None and str(data.get("start") or "").strip():
        start_ms = _parse_date_to_ms(str(data.get("start")), is_end=False)
    if end_ms is None and str(data.get("end") or "").strip():
        end_ms = _parse_date_to_ms(str(data.get("end")), is_end=True)

    if start_ms is None and str(data.get("optimized_start") or "").strip():
        start_ms = _parse_date_to_ms(str(data.get("optimized_start")), is_end=False)
    if end_ms is None and str(data.get("optimized_end") or "").strip():
        end_ms = _parse_date_to_ms(str(data.get("optimized_end")), is_end=True)

    if start_ms is not None:
        df = df[df["timestamp_ms"] >= int(start_ms)]
    if end_ms is not None:
        df = df[df["timestamp_ms"] < int(end_ms)]

    return df.reset_index(drop=True)


def _build_manifest(
    *,
    champion: dict,
    data_root: str,
    source: str,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cfg: OptimizationConfig,
) -> dict:
    meta_keys = {
        "strategy",
        "return_train_pct",
        "dd_train_pct",
        "return_test_pct",
        "dd_test_pct",
        "trades_train",
        "trades_test",
        "eligible",
        "trial",
        "seconds",
        "pm_mode",
    }
    common_keys = {
        "tp_mode",
        "tp_pct",
        "tp_rr",
        "tp_mgmt",
        "tp1_close_frac",
        "tp_trail_pct",
        "sl_type",
        "sl_pct",
        "sl_atr_period",
        "sl_atr_mult",
        "sl_trailing",
        "exit_on_flat",
    }
    grid_keys = {"grid_max_adds", "grid_spacing_pct", "grid_size_multiplier"}
    martingale_keys = {"martingale_multiplier", "martingale_max_steps"}

    params = {k: v for k, v in champion.items() if k not in meta_keys}

    common = {k: params.pop(k) for k in list(params.keys()) if k in common_keys}
    grid = {k: params.pop(k) for k in list(params.keys()) if k in grid_keys}
    martingale = {k: params.pop(k) for k in list(params.keys()) if k in martingale_keys}
    strategy_params = params

    pm_mode = str(champion.get("pm_mode", cfg.pm_mode))
    pm: dict = {"mode": pm_mode}
    if pm_mode == "grid":
        pm["grid"] = grid
    elif pm_mode == "martingale":
        pm["martingale"] = martingale

    return {
        "project": "ohlcv-optimizer-v1",
        "version": 1,
        "data": {
            "data_root": data_root,
            "source": source,
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start,
            "end": end,
        },
        "strategy": {
            "name": champion.get("strategy"),
            "params": strategy_params,
        },
        "backtest": {
            "initial_equity": cfg.initial_equity,
            "costs": {"fee_bps": cfg.fee_bps, "slippage_bps": cfg.slippage_bps},
            "risk": {
                "risk_mode": cfg.risk_mode,
                "risk_pct": cfg.risk_pct,
                "fixed_notional_pct_equity": cfg.fixed_notional_pct_equity,
                "max_position_notional_pct_equity": cfg.max_position_notional_pct_equity,
                "max_leverage": getattr(cfg, "max_leverage", None),
            },
            "execution": {
                "min_qty": float(getattr(cfg, "min_qty", 0.0) or 0.0),
                "qty_step": float(getattr(cfg, "qty_step", 0.0) or 0.0),
                "min_notional": float(getattr(cfg, "min_notional", 0.0) or 0.0),
            },
            "broker": {
                "profile": str(getattr(cfg, "broker_profile", "perps")),
                "perps_maintenance_margin_rate": float(getattr(cfg, "perps_maintenance_margin_rate", 0.01) or 0.01),
                "cfd_initial_margin_rate": float(getattr(cfg, "cfd_initial_margin_rate", 0.01) or 0.01),
                "cfd_stopout_margin_level": float(getattr(cfg, "cfd_stopout_margin_level", 0.5) or 0.5),
            },
            "common": common,
            "position_manager": pm,
        },
        "selection": {
            "dd_threshold_pct": cfg.dd_threshold_pct,
            "min_trades_train": cfg.min_trades_train,
            "min_trades_test": cfg.min_trades_test,
        },
        "result": {
            "return_train_pct": champion.get("return_train_pct"),
            "dd_train_pct": champion.get("dd_train_pct"),
            "return_test_pct": champion.get("return_test_pct"),
            "dd_test_pct": champion.get("dd_test_pct"),
            "trades_train": champion.get("trades_train"),
            "trades_test": champion.get("trades_test"),
        },
    }


def main() -> None:
    st.set_page_config(page_title="OHLCV Optimizer v1", layout="wide")
    st.title("OHLCV Optimizer v1")

    if "start_date" not in st.session_state:
        st.session_state["start_date"] = ""
    if "end_date" not in st.session_state:
        st.session_state["end_date"] = ""

    if "opt_running" not in st.session_state:
        st.session_state["opt_running"] = False
    if "opt_stop_event" not in st.session_state:
        st.session_state["opt_stop_event"] = None
    if "opt_queue" not in st.session_state:
        st.session_state["opt_queue"] = None
    if "opt_thread" not in st.session_state:
        st.session_state["opt_thread"] = None
    if "opt_last_payload" not in st.session_state:
        st.session_state["opt_last_payload"] = None
    if "opt_report" not in st.session_state:
        st.session_state["opt_report"] = None
    if "opt_error" not in st.session_state:
        st.session_state["opt_error"] = None
    if "opt_started_at" not in st.session_state:
        st.session_state["opt_started_at"] = None
    if "opt_saved_dir" not in st.session_state:
        st.session_state["opt_saved_dir"] = None
    if "opt_context" not in st.session_state:
        st.session_state["opt_context"] = None

    mode_options = ["Optimize", "Backtest", "Analyze"]
    qp_mode = _get_query_param("mode")
    mode_index = 0
    if qp_mode and qp_mode in mode_options:
        mode_index = mode_options.index(qp_mode)
    mode = st.sidebar.selectbox("Mode", options=mode_options, index=mode_index)

    train_frac = 0.75
    broker_profile = "hyperliquid_perps"
    selected_run_dir: Path | None = None
    selected_report_payload: dict | None = None
    vault_dir = _PROJECT_ROOT / "champion_vault"
    selected_vault_champion_path: Path | None = None
    selected_vault_champion_payload: dict | None = None

    with st.sidebar.expander("Settings", expanded=True):
        data_root_default = default_data_root()
        data_root = Path(st.text_input("Data root", value=str(data_root_default)))

        broker_profile_cfg = "perps"
        broker_preset = "hyperliquid_perps"

        if mode == "Backtest":
            try:
                vault_files = sorted([p for p in vault_dir.rglob("*.json") if p.is_file()])
            except Exception:
                vault_files = []

            if not vault_files:
                st.warning(f"No champions found in vault: {vault_dir}")
                selected_vault_champion_path = None
                selected_vault_champion_payload = None
            else:
                qp_champion = _get_query_param("champion")
                champ_index = 0
                if qp_champion:
                    for i, p in enumerate(vault_files):
                        if p.name == qp_champion:
                            champ_index = i
                            break

                def _champ_label(p: Path) -> str:
                    try:
                        return str(p.relative_to(vault_dir))
                    except Exception:
                        return str(p)

                selected_vault_champion_path = st.selectbox(
                    "Champion (vault)",
                    options=vault_files,
                    index=champ_index,
                    format_func=_champ_label,
                    key="selected_vault_champion_path",
                )

                try:
                    selected_vault_champion_payload = json.loads(
                        selected_vault_champion_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    selected_vault_champion_payload = None

            champ_source = "hyperliquid_perps"
            if isinstance(selected_vault_champion_payload, dict):
                data_snap = selected_vault_champion_payload.get("data_snapshot") or {}
                champ_source = str(data_snap.get("source") or champ_source)

                cfg_snap = selected_vault_champion_payload.get("config_snapshot") or {}
                broker_profile_cfg = str(cfg_snap.get("broker_profile") or broker_profile_cfg)
                broker_preset = str(cfg_snap.get("broker_preset") or broker_preset)

            broker_profile = str(
                st.selectbox(
                    "Data source",
                    options=["hyperliquid_perps", "binance_futures", "mt5_icmarkets"],
                    index=(
                        1
                        if champ_source == "binance_futures"
                        else (2 if champ_source == "mt5_icmarkets" else 0)
                    ),
                    disabled=True,
                    help="Backtest follows the champion snapshot source.",
                )
            )

            st.selectbox(
                "Risk model",
                options=["perps", "cfd", "spot"],
                index=0 if broker_profile_cfg == "perps" else (1 if broker_profile_cfg == "cfd" else 2),
                disabled=True,
                help="Backtest follows the champion config snapshot.",
            )
        else:
            if mode == "Optimize":
                broker_profile = str(
                    st.selectbox(
                        "Data source",
                        options=["hyperliquid_perps", "binance_futures", "mt5_icmarkets"],
                        index=0,
                        help="Chooses the OHLCV source folder under data/market_data/ohlcv/<source>/...",
                    )
                )

                if str(st.session_state.get("broker_preset") or "") == "custom":
                    st.session_state["broker_preset"] = "hyperliquid_perps"

                c_preset, c_custom = st.columns([3, 1])
                with c_preset:
                    broker_preset = str(
                        st.selectbox(
                            "Broker profile",
                            options=["hyperliquid_perps", "binance_futures", "mt5_icmarkets"],
                            index=0,
                            key="broker_preset",
                            help="Preset that controls risk model + default fees/slippage. Data source remains independent.",
                        )
                    )
                with c_custom:
                    if "broker_custom_overrides" not in st.session_state:
                        st.session_state["broker_custom_overrides"] = False
                    broker_custom_overrides = bool(
                        st.checkbox(
                            "Custom",
                            key="broker_custom_overrides",
                            help="Show and override broker-dependent settings.",
                        )
                    )
            else:
                broker_profile = str(
                    st.selectbox(
                        "Data source",
                        options=["hyperliquid_perps", "binance_futures", "mt5_icmarkets"],
                        index=0,
                        help="Chooses the OHLCV source folder under data/market_data/ohlcv/<source>/..."
                    )
                )

        if mode == "Optimize":
            st.subheader("Execution")
            broker_profile_cfg_preset = str(broker_profile_cfg)
            fee_default = 4.5
            slippage_default = 1.0
            min_qty_default = 0.0
            qty_step_default = 0.0
            min_notional_default = 0.0
            perps_maintenance_margin_rate_default = 0.01
            cfd_initial_margin_rate_default = 0.01
            cfd_stopout_margin_level_default = 0.5
            max_leverage_default_x = 0.0

            if str(broker_preset) == "hyperliquid_perps":
                broker_profile_cfg_preset = "perps"
                fee_default = 4.5
                slippage_default = 1.0
                min_qty_default = 0.0
                qty_step_default = 0.001
                min_notional_default = 10.0
                max_leverage_default_x = 50.0
            elif str(broker_preset) == "binance_futures":
                broker_profile_cfg_preset = "perps"
                fee_default = 4.0
                slippage_default = 0.5
                min_qty_default = 0.0
                qty_step_default = 0.001
                min_notional_default = 5.0
                max_leverage_default_x = 50.0
            elif str(broker_preset) == "mt5_icmarkets":
                broker_profile_cfg_preset = "cfd"
                fee_default = 0.0
                slippage_default = 1.0
                min_qty_default = 0.01
                qty_step_default = 0.01
                min_notional_default = 0.0
                max_leverage_default_x = 2.0

            broker_custom_overrides = bool(st.session_state.get("broker_custom_overrides") or False)

            if "last_broker_preset" not in st.session_state:
                st.session_state["last_broker_preset"] = str(broker_preset)
            if "last_broker_custom_overrides" not in st.session_state:
                st.session_state["last_broker_custom_overrides"] = bool(broker_custom_overrides)

            if "last_max_leverage_default_x" not in st.session_state:
                st.session_state["last_max_leverage_default_x"] = float(max_leverage_default_x)
            if "max_leverage_x" not in st.session_state:
                st.session_state["max_leverage_x"] = float(max_leverage_default_x)

            if "fee_bps" not in st.session_state:
                st.session_state["fee_bps"] = float(fee_default)
            if "slippage_bps" not in st.session_state:
                st.session_state["slippage_bps"] = float(slippage_default)
            if "min_qty" not in st.session_state:
                st.session_state["min_qty"] = float(min_qty_default)
            if "qty_step" not in st.session_state:
                st.session_state["qty_step"] = float(qty_step_default)
            if "min_notional" not in st.session_state:
                st.session_state["min_notional"] = float(min_notional_default)
            if "broker_profile_cfg" not in st.session_state:
                st.session_state["broker_profile_cfg"] = str(broker_profile_cfg_preset)
            if "perps_maintenance_margin_rate" not in st.session_state:
                st.session_state["perps_maintenance_margin_rate"] = float(perps_maintenance_margin_rate_default)
            if "cfd_initial_margin_rate" not in st.session_state:
                st.session_state["cfd_initial_margin_rate"] = float(cfd_initial_margin_rate_default)
            if "cfd_stopout_margin_level" not in st.session_state:
                st.session_state["cfd_stopout_margin_level"] = float(cfd_stopout_margin_level_default)

            preset_changed = str(st.session_state.get("last_broker_preset")) != str(broker_preset)
            custom_turned_off = bool(st.session_state.get("last_broker_custom_overrides")) and (not broker_custom_overrides)
            if preset_changed or custom_turned_off:
                try:
                    last_default = float(st.session_state.get("last_max_leverage_default_x") or 0.0)
                    cur = float(st.session_state.get("max_leverage_x") or 0.0)
                    if abs(cur - last_default) <= 1e-12:
                        st.session_state["max_leverage_x"] = float(max_leverage_default_x)
                except Exception:
                    pass
                st.session_state["last_max_leverage_default_x"] = float(max_leverage_default_x)
                if not broker_custom_overrides:
                    st.session_state["fee_bps"] = float(fee_default)
                    st.session_state["slippage_bps"] = float(slippage_default)
                    st.session_state["min_qty"] = float(min_qty_default)
                    st.session_state["qty_step"] = float(qty_step_default)
                    st.session_state["min_notional"] = float(min_notional_default)
                    st.session_state["broker_profile_cfg"] = str(broker_profile_cfg_preset)
                    st.session_state["perps_maintenance_margin_rate"] = float(perps_maintenance_margin_rate_default)
                    st.session_state["cfd_initial_margin_rate"] = float(cfd_initial_margin_rate_default)
                    st.session_state["cfd_stopout_margin_level"] = float(cfd_stopout_margin_level_default)
                st.session_state["last_broker_preset"] = str(broker_preset)
                st.session_state["last_broker_custom_overrides"] = bool(broker_custom_overrides)

            if not broker_custom_overrides:
                broker_profile_cfg = str(broker_profile_cfg_preset)
                st.session_state["fee_bps"] = float(fee_default)
                st.session_state["slippage_bps"] = float(slippage_default)
                st.session_state["min_qty"] = float(min_qty_default)
                st.session_state["qty_step"] = float(qty_step_default)
                st.session_state["min_notional"] = float(min_notional_default)
                st.session_state["broker_profile_cfg"] = str(broker_profile_cfg_preset)
                st.session_state["perps_maintenance_margin_rate"] = float(perps_maintenance_margin_rate_default)
                st.session_state["cfd_initial_margin_rate"] = float(cfd_initial_margin_rate_default)
                st.session_state["cfd_stopout_margin_level"] = float(cfd_stopout_margin_level_default)

                fee_bps = float(fee_default)
                slippage_bps = float(slippage_default)
                min_qty = float(min_qty_default)
                qty_step = float(qty_step_default)
                min_notional = float(min_notional_default)

                st.caption(f"Risk model: {str(broker_profile_cfg)}")
                st.caption(f"Fee/slippage: {float(fee_bps):g} bps / {float(slippage_bps):g} bps")
                st.caption(
                    f"Exec constraints: min_qty={float(min_qty):g}, qty_step={float(qty_step):g}, min_notional={float(min_notional):g}"
                )
            else:
                rm = str(st.session_state.get("broker_profile_cfg") or broker_profile_cfg_preset)
                rm_index = 0 if rm == "perps" else (1 if rm == "cfd" else 2)
                broker_profile_cfg = str(
                    st.selectbox(
                        "Risk model",
                        options=["perps", "cfd", "spot"],
                        index=rm_index,
                        key="broker_profile_cfg",
                        help="Controls liquidation/margin model used by the backtester.",
                    )
                )

                fee_bps = float(
                    st.number_input(
                        "Fee (taker) bps",
                        min_value=0.0,
                        step=0.1,
                        key="fee_bps",
                    )
                )
                slippage_bps = float(
                    st.number_input(
                        "Slippage bps",
                        min_value=0.0,
                        step=0.1,
                        key="slippage_bps",
                    )
                )

                st.subheader("Execution constraints")
                min_qty = float(
                    st.number_input(
                        "Min qty",
                        min_value=0.0,
                        step=0.01,
                        key="min_qty",
                        help="Minimum order quantity (min lot). 0 disables.",
                    )
                )
                qty_step = float(
                    st.number_input(
                        "Qty step",
                        min_value=0.0,
                        step=0.001,
                        key="qty_step",
                        help="Quantity step size. Quantity is rounded down to the step. 0 disables.",
                    )
                )
                min_notional = float(
                    st.number_input(
                        "Min notional",
                        min_value=0.0,
                        step=1.0,
                        key="min_notional",
                        help="Minimum order value (qty*price). 0 disables.",
                    )
                )

            st.subheader("Selection filters")
            dd_threshold_pct = float(
                st.number_input("Max drawdown threshold (test) %", min_value=0.0, value=40.0, step=1.0)
            )
            min_trades_train = int(st.number_input("Min trades (train)", min_value=0, value=0, step=1))
            min_trades_test = int(st.number_input("Min trades (test)", min_value=0, value=0, step=1))

            st.subheader("Stopping conditions")
            stop_mode = st.radio(
                "Stop mode",
                options=["Trials", "Time"],
                index=0,
                horizontal=True,
                help="Choose ONE stopping condition. The other one will be ignored.",
            )

            max_trials_ui = int(
                st.number_input(
                    "Stop after max trials (per strategy)",
                    min_value=10,
                    value=300,
                    step=50,
                    disabled=(stop_mode != "Trials"),
                    key="max_trials",
                )
            )

            time_budget_minutes_ui = int(
                st.number_input(
                    "Stop after time budget (minutes per strategy)",
                    min_value=1,
                    value=30,
                    step=5,
                    disabled=(stop_mode != "Time"),
                    key="time_budget_minutes",
                )
            )

            ctx_tag = str(st.session_state.get("opt_context_tag") or "")
            if str(st.session_state.get("opt_context_applied_tag") or "") != ctx_tag:
                try:
                    v = (st.session_state.get("opt_context") or {}).get("max_position_notional_pct_equity")
                    st.session_state["max_position_notional_pct_equity_manual"] = float(v) if v is not None else 100.0
                except Exception:
                    st.session_state["max_position_notional_pct_equity_manual"] = 100.0

                try:
                    v = (st.session_state.get("opt_context") or {}).get("max_leverage")
                    st.session_state["max_leverage_x"] = float(v) if v is not None else 0.0
                except Exception:
                    st.session_state["max_leverage_x"] = 0.0

                try:
                    v = (st.session_state.get("opt_context") or {}).get("broker_preset")
                    if v is not None and str(v) != "custom":
                        st.session_state["broker_preset"] = str(v)
                except Exception:
                    pass

                try:
                    v = (st.session_state.get("opt_context") or {}).get("broker_custom_overrides")
                    if v is None:
                        v = bool(
                            str((st.session_state.get("opt_context") or {}).get("broker_preset") or "")
                            == "custom"
                        )
                    st.session_state["broker_custom_overrides"] = bool(v)
                except Exception:
                    pass

                try:
                    v = (st.session_state.get("opt_context") or {}).get("broker_profile")
                    if v is not None:
                        st.session_state["broker_profile_cfg"] = str(v)
                except Exception:
                    pass

                for k in [
                    "fee_bps",
                    "slippage_bps",
                    "min_qty",
                    "qty_step",
                    "min_notional",
                    "perps_maintenance_margin_rate",
                    "cfd_initial_margin_rate",
                    "cfd_stopout_margin_level",
                ]:
                    try:
                        v = (st.session_state.get("opt_context") or {}).get(k)
                        if v is None:
                            continue
                        if k in {"fee_bps", "slippage_bps", "min_qty", "qty_step", "min_notional"}:
                            st.session_state[k] = float(v)
                        elif k in {"perps_maintenance_margin_rate", "cfd_initial_margin_rate", "cfd_stopout_margin_level"}:
                            st.session_state[k] = float(v)
                    except Exception:
                        pass

                try:
                    v = (st.session_state.get("opt_context") or {}).get("storage_url")
                    if v is None:
                        v = os.environ.get("OPTUNA_STORAGE_URL")
                    st.session_state["storage_url"] = str(v or "")
                except Exception:
                    st.session_state["storage_url"] = str(os.environ.get("OPTUNA_STORAGE_URL") or "")

                st.session_state["opt_context_applied_tag"] = ctx_tag

            if "storage_url" not in st.session_state:
                try:
                    v = (st.session_state.get("opt_context") or {}).get("storage_url")
                    if v is None:
                        v = os.environ.get("OPTUNA_STORAGE_URL")
                    st.session_state["storage_url"] = str(v or "")
                except Exception:
                    st.session_state["storage_url"] = str(os.environ.get("OPTUNA_STORAGE_URL") or "")

            storage_url_ui = str(
                st.text_input(
                    "Optuna storage_url",
                    key="storage_url",
                    help="Leave empty to use local sqlite in the run folder. Use a postgres URL for shared/multiprocess storage.",
                )
            ).strip()

            if stop_mode == "Time":
                max_trials = 10_000_000
                time_budget_minutes = time_budget_minutes_ui
            else:
                max_trials = max_trials_ui
                time_budget_minutes = 0

            workers = int(
                st.number_input(
                    "Worker processes (SQLite)",
                    min_value=1,
                    value=8,
                    step=1,
                    help=(
                        "Number of OS worker processes per strategy (SQLite storage). "
                        "Each worker runs trials with n_jobs=1. Increase up to your CPU cores."
                    ),
                )
            )

            multiprocess = True
            if stop_mode == "Time" and int(workers) > 1:
                st.warning("Time stop with multiple workers can overshoot the budget because already-started trials must finish.")

            st.subheader("Backtest")
            initial_equity = float(st.number_input("Initial equity", min_value=10.0, value=10_000.0, step=100.0))

            risk_mode_default = str((st.session_state.get("opt_context") or {}).get("risk_mode") or "risk").strip().lower()
            risk_mode = str(
                st.selectbox(
                    "Risk management mode",
                    options=["risk", "fixed"],
                    index=(1 if risk_mode_default == "fixed" else 0),
                )
            )

            if risk_mode == "fixed":
                fixed_notional_pct_equity = float(
                    st.number_input(
                        "Fixed notional (% equity)",
                        min_value=0.0,
                        value=float((st.session_state.get("opt_context") or {}).get("fixed_notional_pct_equity") or 10.0),
                        step=1.0,
                    )
                )
                risk_pct = 0.0
            else:
                fixed_notional_pct_equity = 0.0
                risk_pct_ui = float(
                    st.number_input(
                        "Risk per trade (%)",
                        min_value=0.0,
                        value=1.0,
                        step=0.1,
                        help="Sizing uses risk_pct as a fraction (1.0% => 0.01).",
                    )
                )
                risk_pct = float(risk_pct_ui) / 100.0

            if "max_position_notional_pct_equity_manual" not in st.session_state:
                try:
                    v = (st.session_state.get("opt_context") or {}).get("max_position_notional_pct_equity")
                    st.session_state["max_position_notional_pct_equity_manual"] = float(v) if v is not None else 100.0
                except Exception:
                    st.session_state["max_position_notional_pct_equity_manual"] = 100.0

            if "max_leverage_x" not in st.session_state:
                try:
                    v = (st.session_state.get("opt_context") or {}).get("max_leverage")
                    st.session_state["max_leverage_x"] = float(v) if v is not None else 0.0
                except Exception:
                    st.session_state["max_leverage_x"] = 0.0

            if "max_leverage_initialized" not in st.session_state:
                st.session_state["max_leverage_initialized"] = False
            if not bool(st.session_state.get("max_leverage_initialized")):
                try:
                    v = (st.session_state.get("opt_context") or {}).get("max_leverage")
                    if v is not None and float(v) > 0:
                        st.session_state["max_position_notional_pct_equity_manual"] = float(v) * 100.0
                except Exception:
                    pass
                st.session_state["max_leverage_initialized"] = True

            max_leverage_ui = float(
                st.number_input(
                    "Max leverage (x)",
                    min_value=0.0,
                    step=0.5,
                    key="max_leverage_x",
                    help="0 disables. When set, overrides max position notional cap: max_notional_pct_equity = max_leverage * 100.",
                )
            )
            max_leverage = float(max_leverage_ui)
            if max_leverage <= 0:
                max_leverage = None

            if max_leverage is None:
                max_position_notional_pct_equity = float(
                    st.number_input(
                        "Max position notional (% equity)",
                        min_value=0.0,
                        value=float(st.session_state.get("max_position_notional_pct_equity_manual") or 100.0),
                        step=10.0,
                        help="Caps abs(qty)*price as a % of equity. 100% ~ 1x notional.",
                        key="max_position_notional_pct_equity_manual",
                    )
                )
            else:
                max_position_notional_pct_equity = float(max_leverage) * 100.0
                st.number_input(
                    "Max position notional (% equity)",
                    min_value=0.0,
                    value=float(max_position_notional_pct_equity),
                    step=10.0,
                    disabled=True,
                    help="Derived from max leverage.",
                )
                st.caption(f"Effective notional cap: {float(max_position_notional_pct_equity):.1f}% (~{float(max_leverage):.2f}x)")

            broker_custom_overrides = bool(st.session_state.get("broker_custom_overrides") or False)

            if "perps_maintenance_margin_rate" not in st.session_state:
                st.session_state["perps_maintenance_margin_rate"] = 0.01
            if "cfd_initial_margin_rate" not in st.session_state:
                st.session_state["cfd_initial_margin_rate"] = 0.01
            if "cfd_stopout_margin_level" not in st.session_state:
                st.session_state["cfd_stopout_margin_level"] = 0.5

            if broker_custom_overrides:
                st.subheader("Broker")
                perps_maintenance_margin_rate = float(
                    st.number_input(
                        "Perps maintenance margin rate",
                        min_value=0.0,
                        step=0.001,
                        format="%.3f",
                        key="perps_maintenance_margin_rate",
                        disabled=(broker_profile_cfg != "perps"),
                        help="Used for liquidation: liquidate if equity_worst <= mmr * notional.",
                    )
                )
                cfd_initial_margin_rate = float(
                    st.number_input(
                        "CFD initial margin rate",
                        min_value=0.0,
                        step=0.001,
                        format="%.3f",
                        key="cfd_initial_margin_rate",
                        disabled=(broker_profile_cfg != "cfd"),
                        help="Used margin = imr * notional.",
                    )
                )
                cfd_stopout_margin_level = float(
                    st.number_input(
                        "CFD stopout margin level",
                        min_value=0.0,
                        step=0.05,
                        format="%.2f",
                        key="cfd_stopout_margin_level",
                        disabled=(broker_profile_cfg != "cfd"),
                        help="Liquidate if equity_worst / used_margin <= stopout.",
                    )
                )
            else:
                perps_maintenance_margin_rate = float(st.session_state.get("perps_maintenance_margin_rate") or 0.01)
                cfd_initial_margin_rate = float(st.session_state.get("cfd_initial_margin_rate") or 0.01)
                cfd_stopout_margin_level = float(st.session_state.get("cfd_stopout_margin_level") or 0.5)

                if broker_profile_cfg == "perps":
                    st.caption(f"Perps maintenance margin rate: {float(perps_maintenance_margin_rate):.3f}")
                elif broker_profile_cfg == "cfd":
                    st.caption(
                        f"CFD margin: imr={float(cfd_initial_margin_rate):.3f}, stopout={float(cfd_stopout_margin_level):.2f}"
                    )

            st.subheader("Walkforward")
            train_frac = float(
                st.slider(
                    "Train fraction",
                    min_value=0.50,
                    max_value=0.95,
                    value=0.75,
                    step=0.01,
                    help="Train/test split for walkforward. Example: 0.75 = first 75% train, last 25% test.",
                )
            )

            st.subheader("Méthode d'optimisation")
            pm_enabled_default = True
            try:
                pm_enabled_default = str((st.session_state.get("opt_context") or {}).get("pm_mode") or "auto").strip().lower() != "none"
            except Exception:
                pm_enabled_default = True
            pm_enabled = bool(
                st.checkbox(
                    "Enable position management optimization (grid/martingale)",
                    value=bool(pm_enabled_default),
                    help="If enabled, Optuna will choose pm_mode among none/grid/martingale. If disabled, pm_mode is forced to none.",
                )
            )
            pm_mode = "auto" if pm_enabled else "none"

            tp_force_rr = bool(
                st.checkbox(
                    "Force TP in RR mode",
                    value=bool((st.session_state.get("opt_context") or {}).get("tp_mode_policy") in {"rr_fixed", "force_rr", "rr"}),
                    help="If enabled, TP is always computed as TP = SL * RR, with a fixed RR multiple you choose.",
                )
            )
            tp_rr_fixed = float(
                st.number_input(
                    "RR multiple (TP = SL × RR)",
                    min_value=0.05,
                    value=float((st.session_state.get("opt_context") or {}).get("tp_rr_fixed") or 2.0),
                    step=0.1,
                    disabled=not tp_force_rr,
                )
            )
            tp_mode_policy = "rr_fixed" if tp_force_rr else "auto"

            objective_options = [
                "win_rate_train",
                "return_train_pct",
                "sharpe_train",
                "median_pnl_per_position_train",
                "median_pnl_per_position_train_pct",
                "avg_pnl_per_position_train",
                "avg_pnl_per_position_train_pct",
                "sharpe_pnl_per_position_train",
            ]
            objective_default = str((st.session_state.get("opt_context") or {}).get("optuna_objective_metric") or "win_rate_train")
            try:
                obj_index = objective_options.index(objective_default)
            except Exception:
                obj_index = 0

            optuna_objective_metric = str(
                st.selectbox(
                    "Optuna objective metric (train)",
                    options=objective_options,
                    index=int(obj_index),
                    help="This metric is optimized on TRAIN during Optuna trials (first objective). Drawdown is still minimized as the second objective.",
                )
            )

            st.subheader("Ranking")
            candidate_pool = str(
                st.selectbox(
                    "Candidate pool",
                    options=["pareto", "complete"],
                    index=0,
                    help="pareto = fast (best_trials), complete = slower (all COMPLETE trials, capped by pareto_candidates_max).",
                )
            )
            pareto_candidates_max = int(
                st.number_input(
                    "Max candidates per strategy",
                    min_value=10,
                    value=100,
                    step=10,
                    help="Hard cap applied before computing test metrics (keeps report build fast).",
                )
            )

            global_top_k = int(
                st.number_input(
                    "Global Top-K",
                    min_value=5,
                    value=50,
                    step=5,
                    help="Number of candidates kept in the global leaderboard (cross-strategy).",
                )
            )
            ranking_metric = str(
                st.selectbox(
                    "Ranking metric (post-walkforward)",
                    options=[
                        "median_pnl_per_position_test",
                        "median_pnl_per_position_test_pct",
                        "avg_pnl_per_position_test",
                        "avg_pnl_per_position_test_pct",
                        "sharpe_pnl_per_position_test",
                        "return_test_pct",
                        "sharpe_test",
                    ],
                    index=5,
                    help="This metric is used to rank candidates on TEST (post-walkforward). The same metric with _train is used for pre-walkforward ranks.",
                )
            )

            require_positive_train_metric_for_test = bool(
                st.checkbox(
                    "Only test candidates with positive train metric",
                    value=True,
                    help=(
                        "If enabled, a candidate is backtested on TEST only if the TRAIN version of the ranking metric is > 0. "
                        "Example: ranking_metric=return_test_pct => require return_train_pct > 0."
                    ),
                )
            )
        else:
            st.caption("This mode loads settings from a saved run (report.json).")
            optuna_objective_metric = str(
                (st.session_state.get("opt_context") or {}).get("optuna_objective_metric") or "return_train_pct"
            )
            pm_mode = str((st.session_state.get("opt_context") or {}).get("pm_mode") or "auto")
            tp_mode_policy = str((st.session_state.get("opt_context") or {}).get("tp_mode_policy") or "auto")
            tp_rr_fixed = float((st.session_state.get("opt_context") or {}).get("tp_rr_fixed") or 2.0)
            require_positive_train_metric_for_test = bool(
                (st.session_state.get("opt_context") or {}).get("require_positive_train_metric_for_test")
                if (st.session_state.get("opt_context") or {}).get("require_positive_train_metric_for_test") is not None
                else True
            )

    with st.sidebar.expander("Info", expanded=False):
        st.caption("Position management is optimized automatically (none/grid/martingale).")

    def _cfg_from_report(cfg_payload: dict) -> OptimizationConfig:
        return OptimizationConfig(
            initial_equity=float(cfg_payload.get("initial_equity", 10_000.0)),
            fee_bps=float(cfg_payload.get("fee_bps", 4.5)),
            slippage_bps=float(cfg_payload.get("slippage_bps", 1.0)),
            dd_threshold_pct=float(cfg_payload.get("dd_threshold_pct", 40.0)),
            max_trials=int(cfg_payload.get("max_trials", 300)),
            time_budget_seconds=cfg_payload.get("time_budget_seconds"),
            n_jobs=int(cfg_payload.get("n_jobs", 1)),
            min_trades_train=int(cfg_payload.get("min_trades_train", 0)),
            min_trades_test=int(cfg_payload.get("min_trades_test", 0)),
            timeframe=str(cfg_payload.get("timeframe", "5m")),
            pm_mode=str(cfg_payload.get("pm_mode", "auto")),
            strategies=cfg_payload.get("strategies"),
            risk_mode=str(cfg_payload.get("risk_mode", "risk")),
            risk_pct=float(cfg_payload.get("risk_pct", 0.01)),
            fixed_notional_pct_equity=float(cfg_payload.get("fixed_notional_pct_equity", 0.0)),
            max_position_notional_pct_equity=float(cfg_payload.get("max_position_notional_pct_equity", 100.0)),
            max_leverage=float(cfg_payload["max_leverage"]) if cfg_payload.get("max_leverage") is not None else None,
            min_qty=float(cfg_payload.get("min_qty", 0.0)),
            qty_step=float(cfg_payload.get("qty_step", 0.0)),
            min_notional=float(cfg_payload.get("min_notional", 0.0)),
            broker_profile=str(cfg_payload.get("broker_profile", "perps")),
            perps_maintenance_margin_rate=float(cfg_payload.get("perps_maintenance_margin_rate", 0.01)),
            cfd_initial_margin_rate=float(cfg_payload.get("cfd_initial_margin_rate", 0.01)),
            cfd_stopout_margin_level=float(cfg_payload.get("cfd_stopout_margin_level", 0.5)),
            pareto_candidates_max=int(cfg_payload.get("pareto_candidates_max", 50)),
            candidate_pool=str(cfg_payload.get("candidate_pool", "pareto")),
            global_top_k=int(cfg_payload.get("global_top_k", 50)),
            ranking_metric=str(cfg_payload.get("ranking_metric", "return_test_pct")),
            train_frac=float(cfg_payload.get("train_frac", 0.75)),
            require_positive_train_metric_for_test=bool(cfg_payload.get("require_positive_train_metric_for_test", True)),
            optuna_objective_metric=str(cfg_payload.get("optuna_objective_metric", "return_train_pct")),
            tp_mode_policy=str(cfg_payload.get("tp_mode_policy", "auto")),
            tp_rr_fixed=float(cfg_payload.get("tp_rr_fixed", 2.0)),
        )

    def _rebuild_report_from_optuna_db(*, run_dir: Path) -> dict | None:
        ctx = _load_context_json(run_dir)
        if not isinstance(ctx, dict):
            st.error("Missing context.json")
            return None

        ctx_data = ctx.get("data") or {}
        ctx_cfg = ctx.get("config") or {}

        data_root_used = str(data_root) if data_root is not None else str(ctx_data.get("data_root") or default_data_root())
        source_used = str(ctx_data.get("source") or "hyperliquid_perps")
        symbol_storage_used = str(ctx_data.get("symbol") or "BTC")
        timeframe_used = str(ctx_data.get("timeframe") or "5m")

        try:
            df = _load_ohlcv_cached(
                data_root=str(data_root_used),
                source=source_used,
                symbol=symbol_storage_used,
                timeframe=timeframe_used,
            )
        except FileNotFoundError as e:
            st.error(str(e))
            return None

        start_ms = ctx_data.get("start_ms")
        end_ms = ctx_data.get("end_ms")
        if start_ms is not None:
            df = df[df["timestamp_ms"] >= int(start_ms)]
        if end_ms is not None:
            df = df[df["timestamp_ms"] < int(end_ms)]
        df = df.reset_index(drop=True)

        optimized_start = None
        optimized_end = None
        train_start = None
        train_end = None
        test_start = None
        test_end = None
        try:
            if not df.empty and "timestamp_ms" in df.columns:
                optimized_start = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[0]))
                optimized_end = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[-1]))
        except Exception:
            optimized_start = None
            optimized_end = None

        cfg = _cfg_from_report(ctx_cfg)

        cfg_storage_url = str((ctx_cfg or {}).get("storage_url") or "").strip()
        if cfg_storage_url:
            storage_url = cfg_storage_url
        else:
            db_path = run_dir / "optuna.db"
            storage_url = f"sqlite:///{db_path.as_posix()}"

        study_name_prefix = run_dir.name
        report = build_report_from_storage(df=df, config=cfg, storage_url=storage_url, study_name_prefix=study_name_prefix)

        leaderboard_path = run_dir / "leaderboard.csv"
        global_leaderboard_path = run_dir / "global_leaderboard.csv"
        candidates_path = run_dir / "candidates.csv"
        champion_path = run_dir / "champion.json"
        report_path = run_dir / "report.json"

        try:
            report.leaderboard.to_csv(leaderboard_path, index=False)
        except Exception:
            pass
        if getattr(report, "global_leaderboard", None) is not None:
            try:
                report.global_leaderboard.to_csv(global_leaderboard_path, index=False)
            except Exception:
                pass
        if getattr(report, "candidates", None) is not None:
            try:
                report.candidates.to_csv(candidates_path, index=False)
            except Exception:
                pass
        if report.champion is not None:
            try:
                champion_path.write_text(json.dumps(report.champion, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        n = int(len(df))
        train_frac_used = float(getattr(cfg, "train_frac", 0.75) or 0.75)
        cut = int(n * train_frac_used) if n else 0
        if n >= 10:
            cut = max(1, min(n - 1, int(cut)))
        else:
            cut = max(0, min(n, int(cut)))

        if optimized_start is not None and optimized_end is not None and n > 0:
            train_start = optimized_start
            try:
                train_end = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[max(0, int(cut) - 1)]))
            except Exception:
                train_end = optimized_end

            try:
                test_start = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[min(int(cut), n - 1)]))
            except Exception:
                test_start = train_end
            test_end = optimized_end

        report_payload = {
            "project": "ohlcv-optimizer-v1",
            "version": 1,
            "saved_at": run_dir.name,
            "meta": getattr(report, "meta", None),
            "strategies_skipped": getattr(report, "strategies_skipped", None),
            "champion_global": getattr(report, "champion_global", None),
            "champions_by_strategy": getattr(report, "champions_by_strategy", None),
            "data": {
                "data_root": str(data_root_used),
                "source": source_used,
                "symbol": symbol_storage_used,
                "symbol_display": symbol_storage_used,
                "symbol_storage": symbol_storage_used,
                "timeframe": timeframe_used,
                "start": None,
                "end": None,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "optimized_start": optimized_start,
                "optimized_end": optimized_end,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "candles_total": int(len(df)),
                "candles_train": int(cut),
                "candles_test": int(max(0, int(len(df)) - int(cut))),
            },
            "config": {
                "initial_equity": cfg.initial_equity,
                "fee_bps": cfg.fee_bps,
                "slippage_bps": cfg.slippage_bps,
                "broker_preset": str(
                    st.session_state.get("broker_preset")
                    or (st.session_state.get("opt_context") or {}).get("broker_preset")
                    or ""
                ),
                "broker_custom_overrides": bool(st.session_state.get("broker_custom_overrides") or False),
                "dd_threshold_pct": cfg.dd_threshold_pct,
                "optuna_objective_metric": str(getattr(cfg, "optuna_objective_metric", "return_train_pct")),
                "min_trades_train": cfg.min_trades_train,
                "min_trades_test": cfg.min_trades_test,
                "max_trials": cfg.max_trials,
                "time_budget_seconds": cfg.time_budget_seconds,
                "n_jobs": cfg.n_jobs,
                "timeframe": cfg.timeframe,
                "pm_mode": cfg.pm_mode,
                "strategies": cfg.strategies,
                "risk_mode": str(getattr(cfg, "risk_mode", "risk")),
                "risk_pct": float(getattr(cfg, "risk_pct", 0.0) or 0.0),
                "fixed_notional_pct_equity": float(getattr(cfg, "fixed_notional_pct_equity", 0.0) or 0.0),
                "max_position_notional_pct_equity": float(getattr(cfg, "max_position_notional_pct_equity", 0.0) or 0.0),
                "max_leverage": getattr(cfg, "max_leverage", None),
                "min_qty": float(getattr(cfg, "min_qty", 0.0) or 0.0),
                "qty_step": float(getattr(cfg, "qty_step", 0.0) or 0.0),
                "min_notional": float(getattr(cfg, "min_notional", 0.0) or 0.0),
                "broker_profile": str(getattr(cfg, "broker_profile", "perps")),
                "perps_maintenance_margin_rate": float(getattr(cfg, "perps_maintenance_margin_rate", 0.01) or 0.01),
                "cfd_initial_margin_rate": float(getattr(cfg, "cfd_initial_margin_rate", 0.01) or 0.01),
                "cfd_stopout_margin_level": float(getattr(cfg, "cfd_stopout_margin_level", 0.5) or 0.5),
                "pareto_candidates_max": cfg.pareto_candidates_max,
                "candidate_pool": cfg.candidate_pool,
                "global_top_k": cfg.global_top_k,
                "ranking_metric": cfg.ranking_metric,
                "train_frac": cfg.train_frac,
                "require_positive_train_metric_for_test": bool(
                    getattr(cfg, "require_positive_train_metric_for_test", True)
                ),
                "tp_mode_policy": str(getattr(cfg, "tp_mode_policy", "auto") or "auto"),
                "tp_rr_fixed": float(getattr(cfg, "tp_rr_fixed", 2.0) or 2.0),
                "multiprocess": True,
                "workers": None,
            },
            "leaderboard": report.leaderboard.to_dict(orient="records"),
            "global_leaderboard": (
                report.global_leaderboard.to_dict(orient="records")
                if getattr(report, "global_leaderboard", None) is not None
                else None
            ),
            "global_leaderboard_file": "global_leaderboard.csv" if global_leaderboard_path.exists() else None,
            "candidates_file": "candidates.csv" if candidates_path.exists() else None,
            "champion": report.champion,
        }

        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        try:
            (run_dir / "open_analyze.url").write_text(
                "[InternetShortcut]\n" f"URL=http://localhost:8501/?mode=Analyze&run={run_dir.name}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        return report_payload

    def _run_champion_backtest(*, report_payload: dict) -> None:
        champion = report_payload.get("champion")
        if not isinstance(champion, dict):
            st.warning("No champion found in this run.")
            return

        cfg_payload = report_payload.get("config") or {}
        cfg = _cfg_from_report(cfg_payload)
        try:
            df = _load_df_for_report(report_payload=report_payload, data_root_override=data_root)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("The run references a symbol/timeframe folder that is missing locally. Download it with projects/market-data-downloader.")
            return
        except Exception as e:
            st.error(f"Failed to load run candles: {type(e).__name__}: {e}")
            return
        if df.empty:
            st.error("No candles for this run.")
            return

        all_strats = builtin_strategies()
        strat_name = str(champion.get("strategy"))
        strat = next((s for s in all_strats if s.name == strat_name), None)
        if strat is None:
            st.error(f"Strategy not found: {strat_name}")
            return

        ctx = StrategyContext(timeframe=str(report_payload.get("data", {}).get("timeframe") or cfg.timeframe))
        bt_cfg = build_backtest_config_from_params(params=champion, base=cfg)
        sig = strat.compute_signal(df, champion, ctx)
        res = run_backtest(df=df, signal=sig, config=bt_cfg)
        pos_df, overview = summarize_positions(res.trades)

        eq = res.equity_curve
        start_e = float(eq[0]) if eq.size else float(cfg.initial_equity)
        end_e = float(eq[-1]) if eq.size else float(cfg.initial_equity)
        total_return = (end_e / max(start_e, 1e-12)) - 1.0

        start_ms = int(df["timestamp_ms"].iloc[0])
        end_ms = int(df["timestamp_ms"].iloc[-1])
        period_days = max(1e-9, float(end_ms - start_ms) / 86_400_000.0)

        ann_return = (1.0 + total_return) ** (365.0 / period_days) - 1.0
        mon_days = 365.25 / 12.0
        mon_return = (1.0 + total_return) ** (mon_days / period_days) - 1.0

        sharpe_eq = _sharpe_from_equity(eq)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Net return", f"{res.net_return_pct:.2f}%")
        c2.metric("Max DD (close)", f"{res.max_drawdown_pct:.2f}%")
        c3.metric("Sharpe (equity)", f"{sharpe_eq:.2f}")
        c4.metric("Return annualized", f"{ann_return * 100.0:.2f}%")
        c5.metric("Return monthly", f"{mon_return * 100.0:.2f}%")

        c10, c11, c12, c13 = st.columns(4)
        c10.metric("Max DD (intrabar)", f"{float(getattr(res, 'max_drawdown_intrabar_pct', 0.0)):.2f}%")
        c11.metric("Liquidated", str(bool(getattr(res, "liquidated", False))))
        c12.metric("Peak notional % eq", f"{float(getattr(res, 'peak_notional_pct_equity', 0.0)):.2f}%")
        c13.metric("Cap hit rate", f"{100.0 * float(getattr(res, 'cap_hit_rate', 0.0)):.1f}%")

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Positions", int(overview.get("positions", 0)))
        c7.metric("Win rate", f"{100.0 * float(overview.get('win_rate', 0.0)):.1f}%")
        c8.metric("TP1 hit rate", f"{100.0 * float(overview.get('tp1_hit_rate', 0.0)):.1f}%")
        c9.metric("Period days", f"{period_days:.2f}")

        with st.expander("Overview JSON", expanded=False):
            st.json(overview)

        with st.expander("Positions table", expanded=False):
            if not pos_df.empty:
                st.dataframe(pos_df, width="stretch")

    def _safe_name(value: str) -> str:
        s = str(value or "").strip()
        out = []
        for ch in s:
            if ch.isalnum() or ch in {"_", "-", "."}:
                out.append(ch)
            else:
                out.append("_")
        return "".join(out) or "unknown"

    def _now_utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _candidate_params(candidate: dict) -> dict:
        drop_keys = {
            "return_train_pct",
            "dd_train_pct",
            "dd_train_intrabar_pct",
            "exec_reject_rate_train",
            "exec_round_rate_train",
            "return_test_pct",
            "dd_test_pct",
            "dd_test_intrabar_pct",
            "exec_reject_rate_test",
            "exec_round_rate_test",
            "sharpe_train",
            "sharpe_test",
            "median_pnl_per_position_train",
            "median_pnl_per_position_test",
            "avg_pnl_per_position_train",
            "avg_pnl_per_position_test",
            "sharpe_pnl_per_position_train",
            "sharpe_pnl_per_position_test",
            "median_pnl_per_position_train_pct",
            "median_pnl_per_position_test_pct",
            "trades_train",
            "trades_test",
            "martingale_max_loss_streak_train",
            "martingale_max_loss_streak_test",
            "martingale_max_step_used_train",
            "martingale_max_step_used_test",
            "martingale_max_multiplier_used_train",
            "martingale_max_multiplier_used_test",
            "grid_max_adds_used_train",
            "grid_max_adds_used_test",
            "grid_max_multiplier_used_train",
            "grid_max_multiplier_used_test",
            "liquidated_train",
            "liquidated_test",
            "peak_notional_pct_equity_train",
            "peak_notional_pct_equity_test",
            "peak_qty_mult_train",
            "peak_qty_mult_test",
            "cap_hit_rate_train",
            "cap_hit_rate_test",
            "eligible",
            "trial",
            "seconds",
            "rank_pre_wf",
            "rank_post_wf",
            "rank_delta",
        }

        params: dict = {}
        for k, v in (candidate or {}).items():
            if k in drop_keys:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            params[k] = v
        return params

    def _load_candidate_analytics(*, run_dir: Path, picked: dict) -> dict | None:
        analytics_dir = run_dir / "candidate_analytics"
        try:
            strat_name = str(picked.get("strategy") or "")
            trial_num = int(picked.get("trial") or 0)
        except Exception:
            strat_name = str(picked.get("strategy") or "")
            trial_num = 0

        if not strat_name or (not analytics_dir.exists()):
            return None

        p = analytics_dir / f"{_safe_name(strat_name)}__trial_{int(trial_num)}.json"
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _render_candidate_overview(*, picked: dict, params: dict, cfg_info: dict) -> None:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Return (test)", f"{float(picked.get('return_test_pct') or 0.0):.2f}%")
        k2.metric("Max DD (test)", f"{float(picked.get('dd_test_pct') or 0.0):.2f}%")
        k3.metric("Sharpe (test)", f"{float(picked.get('sharpe_test') or 0.0):.2f}")
        k4.metric("Trades (test)", int(picked.get("trades_test") or 0))

        k9, k10, k11, k12 = st.columns(4)
        k9.metric("Max DD intrabar (test)", f"{float(picked.get('dd_test_intrabar_pct') or 0.0):.2f}%")
        k10.metric("Liquidated (test)", str(bool(picked.get("liquidated_test") or False)))
        k11.metric("Peak notional % eq (test)", f"{float(picked.get('peak_notional_pct_equity_test') or 0.0):.2f}%")
        k12.metric("Cap hit rate (test)", f"{100.0 * float(picked.get('cap_hit_rate_test') or 0.0):.1f}%")

        k17, k18, k19, k20 = st.columns(4)
        k17.metric("Exec reject (test)", f"{100.0 * float(picked.get('exec_reject_rate_test') or 0.0):.1f}%")
        k18.metric("Exec round (test)", f"{100.0 * float(picked.get('exec_round_rate_test') or 0.0):.1f}%")
        k19.metric("Min qty", f"{float(cfg_info.get('min_qty', 0.0) or 0.0):g}")
        k20.metric("Min notional", f"{float(cfg_info.get('min_notional', 0.0) or 0.0):g}")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Return (train)", f"{float(picked.get('return_train_pct') or 0.0):.2f}%")
        k6.metric("Max DD (train)", f"{float(picked.get('dd_train_pct') or 0.0):.2f}%")
        k7.metric("Sharpe (train)", f"{float(picked.get('sharpe_train') or 0.0):.2f}")
        k8.metric("Trades (train)", int(picked.get("trades_train") or 0))

        k13, k14, k15, k16 = st.columns(4)
        k13.metric("Max DD intrabar (train)", f"{float(picked.get('dd_train_intrabar_pct') or 0.0):.2f}%")
        k14.metric("Liquidated (train)", str(bool(picked.get("liquidated_train") or False)))
        k15.metric("Peak notional % eq (train)", f"{float(picked.get('peak_notional_pct_equity_train') or 0.0):.2f}%")
        k16.metric("Cap hit rate (train)", f"{100.0 * float(picked.get('cap_hit_rate_train') or 0.0):.1f}%")

        k21, k22, k23, k24 = st.columns(4)
        k21.metric("Exec reject (train)", f"{100.0 * float(picked.get('exec_reject_rate_train') or 0.0):.1f}%")
        k22.metric("Exec round (train)", f"{100.0 * float(picked.get('exec_round_rate_train') or 0.0):.1f}%")
        k23.metric("Qty step", f"{float(cfg_info.get('qty_step', 0.0) or 0.0):g}")
        k24.metric("Broker preset", str((st.session_state.get('last_broker_preset') or '')))

        pm_mode_val = str(params.get("pm_mode") or picked.get("pm_mode") or "")
        if pm_mode_val:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("PM mode", pm_mode_val)
            if pm_mode_val == "martingale":
                p2.metric("Max loss streak (test)", int(picked.get("martingale_max_loss_streak_test") or 0))
                p3.metric("Max mult used (test)", f"{float(picked.get('martingale_max_multiplier_used_test') or 1.0):.2f}x")
                p4.metric("Max loss streak (train)", int(picked.get("martingale_max_loss_streak_train") or 0))
            elif pm_mode_val == "grid":
                p2.metric("Max adds used (test)", int(picked.get("grid_max_adds_used_test") or 0))
                p3.metric("Max size mult (test)", f"{float(picked.get('grid_max_multiplier_used_test') or 1.0):.2f}x")
                p4.metric("Max adds used (train)", int(picked.get("grid_max_adds_used_train") or 0))

    def _render_candidate_analytics(*, analytics_payload: dict) -> None:
        def _equity_df(ts_ms: list[int], eq: list[float]) -> pd.DataFrame:
            if not ts_ms or not eq:
                return pd.DataFrame()
            n = min(int(len(ts_ms)), int(len(eq)))
            if n <= 0:
                return pd.DataFrame()
            ts = pd.to_datetime(pd.Series(ts_ms[:n]).astype("int64"), unit="ms", utc=True)
            out = pd.DataFrame({"timestamp": ts.to_numpy(), "equity": pd.Series(eq[:n]).astype("float64").to_numpy()})
            out = out.set_index("timestamp")
            out["dd_pct"] = (out["equity"] / out["equity"].cummax().replace(0, 1e-12) - 1.0) * 100.0
            return out

        def _render_seg(seg_key: str, title: str) -> None:
            seg = (analytics_payload.get("segments") or {}).get(seg_key) or {}
            if seg.get("empty"):
                st.info("Empty segment")
                return
            if seg.get("error"):
                st.error("Analytics error")
                return

            period_start_ms = seg.get("period_start_ms")
            period_end_ms = seg.get("period_end_ms")

            ts_ms = seg.get("ts_ms") or []
            eq = seg.get("equity") or []
            kpis = seg.get("kpis") or {}
            overview = seg.get("overview") or {}
            positions_rows = seg.get("positions") or []

            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Return (period)", f"{float(kpis.get('return_period_pct') or 0.0):.2f}%")
            r2.metric("Return annualized", f"{float(kpis.get('return_annualized_pct') or 0.0):.2f}%")
            r3.metric("Max DD (close)", f"{float(kpis.get('max_drawdown_pct') or 0.0):.2f}%")
            pf = kpis.get("profit_factor")
            try:
                pf_f = float(pf)
            except Exception:
                pf_f = 0.0
            r4.metric("Profit factor", "∞" if pf_f == float("inf") else f"{pf_f:.2f}")
            r5.metric("Sharpe (equity)", f"{float(kpis.get('sharpe_equity') or 0.0):.2f}")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Max DD (intrabar)", f"{float(kpis.get('max_drawdown_intrabar_pct') or 0.0):.2f}%")
            s2.metric("Liquidated", str(bool(kpis.get("liquidated") or False)))
            s3.metric("Peak notional % eq", f"{float(kpis.get('peak_notional_pct_equity') or 0.0):.2f}%")
            s4.metric("Cap hit rate", f"{100.0 * float(kpis.get('cap_hit_rate') or 0.0):.1f}%")

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Positions", int(overview.get("positions", 0) or 0))
            s2.metric("Taux de réussite", f"{100.0 * float(overview.get('win_rate', 0.0) or 0.0):.1f}%")
            s3.metric("TP1 hit rate", f"{100.0 * float(overview.get('tp1_hit_rate', 0.0) or 0.0):.1f}%")
            s4.metric("Avg fills/pos", f"{float(overview.get('avg_fills_per_position', 0.0) or 0.0):.2f}")
            s5.metric("Period days", f"{float(kpis.get('period_days') or 0.0):.2f}")

            if period_start_ms is not None and period_end_ms is not None:
                try:
                    p_start = pd.to_datetime(int(period_start_ms), unit="ms", utc=True)
                    p_end = pd.to_datetime(int(period_end_ms), unit="ms", utc=True)
                    p_days = max(0.0, float((p_end - p_start).total_seconds()) / 86400.0)
                    st.caption(f"Period: {p_start.isoformat()} → {p_end.isoformat()} (days={p_days:.2f})")
                except Exception:
                    pass

            t1, t2, t3, t4, t5 = st.columns(5)
            t1.metric("Median PnL/pos", f"{float(overview.get('median_pnl_per_position', 0.0) or 0.0):.4f}")
            t2.metric("Avg PnL/pos", f"{float(overview.get('avg_pnl_per_position', 0.0) or 0.0):.4f}")
            t3.metric("TP1 PnL share", f"{100.0 * float(overview.get('tp1_pnl_share', 0.0) or 0.0):.1f}%")
            t4.metric("Sharpe (pos PnL)", f"{float(overview.get('sharpe_pnl_per_position', 0.0) or 0.0):.2f}")
            t5.metric("Candles (downsampled)", int(min(len(ts_ms), len(eq))))

            eq_df = _equity_df(ts_ms, eq)
            if not eq_df.empty:
                st.subheader("Equity curve")
                st.line_chart(eq_df[["equity"]], height=240)
                st.subheader("Drawdown (%)")
                st.line_chart(eq_df[["dd_pct"]], height=180)

            with st.expander("Exit reasons (final)", expanded=False):
                st.json(overview.get("final_exit_reason_dist", {}))

            with st.expander("Positions table", expanded=False):
                try:
                    pos_df = pd.DataFrame(positions_rows)
                except Exception:
                    pos_df = pd.DataFrame()
                if not pos_df.empty:
                    st.dataframe(pos_df, width="stretch", hide_index=True)

        seg_tabs = st.tabs(["Test (post-WF)", "Train (pre-WF)", "Full"])
        with seg_tabs[0]:
            _render_seg("test", "Test")
        with seg_tabs[1]:
            _render_seg("train", "Train")
        with seg_tabs[2]:
            _render_seg("full", "Full")

    class _ParamCaptureTrial:
        def __init__(self):
            self.keys: set[str] = set()

        def suggest_int(self, name, low, high, *args, **kwargs):
            self.keys.add(str(name))
            return int(low)

        def suggest_float(self, name, low, high, *args, **kwargs):
            self.keys.add(str(name))
            return float(low)

        def suggest_categorical(self, name, choices, *args, **kwargs):
            self.keys.add(str(name))
            try:
                return list(choices)[0]
            except Exception:
                return None

    def _allowed_param_keys_for_strategy(strat_name: str) -> set[str] | None:
        all_strats = builtin_strategies()
        strat = next((s for s in all_strats if s.name == str(strat_name)), None)
        if strat is None:
            return None

        t = _ParamCaptureTrial()
        try:
            _ = strat.sample_params(t)
        except Exception:
            pass

        common_keys = {
            "strategy",
            "pm_mode",
            "tp_mode",
            "tp_pct",
            "tp_rr",
            "tp_mgmt",
            "tp1_close_frac",
            "tp_trail_pct",
            "sl_type",
            "sl_pct",
            "sl_atr_period",
            "sl_atr_mult",
            "sl_trailing",
            "exit_on_flat",
        }
        pm_keys = {
            "grid_max_adds",
            "grid_spacing_pct",
            "grid_size_multiplier",
            "martingale_multiplier",
            "martingale_max_steps",
        }
        return set(common_keys) | set(pm_keys) | set(t.keys)

    def _save_retained_champion_to_vault(*, run_dir: Path, candidate: dict, report_payload: dict) -> Path | None:
        if not isinstance(candidate, dict):
            return None

        data_info = report_payload.get("data") or {}
        cfg_info = report_payload.get("config") or {}

        strat_name = str(candidate.get("strategy") or "")
        if not strat_name:
            return None

        trial_n = candidate.get("trial")
        try:
            trial_n_int = int(trial_n) if trial_n is not None else None
        except Exception:
            trial_n_int = None

        params = _candidate_params(candidate)
        payload = {
            "schema_version": 1,
            "saved_at": _now_utc_iso(),
            "origin": {"run_id": str(run_dir.name)},
            "strategy": strat_name,
            "trial": trial_n_int,
            "ranking_metric": str(cfg_info.get("ranking_metric") or ""),
            "candidate": candidate,
            "params": params,
            "data_snapshot": data_info,
            "config_snapshot": cfg_info,
        }

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sym = _safe_name(str(data_info.get("symbol_storage") or data_info.get("symbol") or ""))
        tf = _safe_name(str(data_info.get("timeframe") or ""))
        strat_dir = _safe_name(strat_name)
        tr_part = f"trial_{trial_n_int}" if trial_n_int is not None else "trial_unknown"
        base_fn = f"retained__{_safe_name(run_dir.name)}__{strat_dir}__{tr_part}__{ts}.json"

        run_out_dir = run_dir / "retained_champions"
        run_out_dir.mkdir(parents=True, exist_ok=True)
        run_path = run_out_dir / base_fn
        try:
            run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return None

        vault_out_dir = vault_dir / strat_dir / sym / tf
        vault_out_dir.mkdir(parents=True, exist_ok=True)
        vault_path = vault_out_dir / base_fn
        try:
            shutil.copy2(run_path, vault_path)
        except Exception:
            try:
                vault_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                return None

        try:
            index_path = vault_dir / "index.jsonl"
            index_row = {
                "saved_at": payload.get("saved_at"),
                "run_id": str(run_dir.name),
                "strategy": strat_name,
                "trial": trial_n_int,
                "path": str(vault_path),
                "symbol": str(data_info.get("symbol_storage") or data_info.get("symbol") or ""),
                "timeframe": str(data_info.get("timeframe") or ""),
                "source": str(data_info.get("source") or ""),
            }
            index_path.parent.mkdir(parents=True, exist_ok=True)
            with index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(index_row, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return vault_path

    def _run_vault_champion_backtest(*, champion_payload: dict) -> None:
        if not isinstance(champion_payload, dict):
            st.error("Invalid champion payload")
            return

        cfg_payload = champion_payload.get("config_snapshot") or {}
        data_payload = champion_payload.get("data_snapshot") or {}
        params = champion_payload.get("params")
        strat_name = str(champion_payload.get("strategy") or "")

        if not isinstance(params, dict):
            st.error("Champion is missing params")
            return
        if not strat_name:
            st.error("Champion is missing strategy")
            return

        allowed = _allowed_param_keys_for_strategy(strat_name)
        if allowed is None:
            st.error(f"Strategy not found: {strat_name}")
            return

        unknown = sorted([k for k in params.keys() if k not in allowed])
        if unknown:
            st.error(f"Champion params contain unknown keys (strict): {unknown}")
            return

        cfg = _cfg_from_report(cfg_payload)
        try:
            df = _load_df_for_report(report_payload={"data": data_payload}, data_root_override=data_root)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("The champion references a symbol/timeframe folder that is missing locally. Download it with projects/market-data-downloader.")
            return
        except Exception as e:
            st.error(f"Failed to load candles: {type(e).__name__}: {e}")
            return
        if df.empty:
            st.error("No candles.")
            return

        all_strats = builtin_strategies()
        strat = next((s for s in all_strats if s.name == strat_name), None)
        if strat is None:
            st.error(f"Strategy not found: {strat_name}")
            return

        ctx = StrategyContext(timeframe=str(data_payload.get("timeframe") or cfg.timeframe))
        bt_cfg = build_backtest_config_from_params(params=params, base=cfg)
        try:
            sig = strat.compute_signal(df, params, ctx)
        except Exception as e:
            st.error(f"Strategy failed (strict): {type(e).__name__}: {e}")
            return

        res = run_backtest(df=df, signal=sig, config=bt_cfg)
        pos_df, overview = summarize_positions(res.trades)
        overview["sharpe"] = _sharpe_from_equity(res.equity_curve)

        st.subheader("Vault champion backtest")
        st.caption(
            f"{strat_name} | trial {champion_payload.get('trial')} | {data_payload.get('symbol_storage') or data_payload.get('symbol')} {data_payload.get('timeframe')}"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net return (%)", f"{res.net_return_pct:.2f}")
        c2.metric("Max drawdown (%)", f"{res.max_drawdown_pct:.2f}")
        c3.metric("Positions", int(overview.get("positions", 0)))
        c4.metric("Sharpe", f"{float(overview.get('sharpe', 0.0)):.3f}")

        with st.expander("Overview JSON", expanded=False):
            st.json(overview)

        with st.expander("Positions table", expanded=False):
            if not pos_df.empty:
                st.dataframe(pos_df, width="stretch")

    def _run_candidate_backtest(*, candidate: dict, report_payload: dict) -> None:
        if not isinstance(candidate, dict):
            st.error("Invalid candidate")
            return

        cfg_payload = report_payload.get("config") or {}
        cfg = _cfg_from_report(cfg_payload)
        try:
            df = _load_df_for_report(report_payload=report_payload, data_root_override=data_root)
        except FileNotFoundError as e:
            st.error(str(e))
            st.info("The run references a symbol/timeframe folder that is missing locally. Download it with projects/market-data-downloader.")
            return
        except Exception as e:
            st.error(f"Failed to load run candles: {type(e).__name__}: {e}")
            return
        if df.empty:
            st.error("No candles for this run.")
            return

        params = _candidate_params(candidate)

        all_strats = builtin_strategies()
        strat_name = str(params.get("strategy") or candidate.get("strategy") or "")
        strat = next((s for s in all_strats if s.name == strat_name), None)
        if strat is None:
            st.error(f"Strategy not found: {strat_name}")
            return

        ctx = StrategyContext(timeframe=str(report_payload.get("data", {}).get("timeframe") or cfg.timeframe))
        bt_cfg = build_backtest_config_from_params(params=params, base=cfg)
        sig = strat.compute_signal(df, params, ctx)
        res = run_backtest(df=df, signal=sig, config=bt_cfg)
        pos_df, overview = summarize_positions(res.trades)
        overview["sharpe"] = _sharpe_from_equity(res.equity_curve)

        st.subheader("Candidate backtest")
        st.caption(
            f"{strat_name} | trial {candidate.get('trial')} | post#{candidate.get('rank_post_wf')} pre#{candidate.get('rank_pre_wf')}"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net return (%)", f"{res.net_return_pct:.2f}")
        c2.metric("Max drawdown (%)", f"{res.max_drawdown_pct:.2f}")
        c3.metric("Positions", int(overview.get("positions", 0)))
        c4.metric("Sharpe", f"{float(overview.get('sharpe', 0.0)):.3f}")

        with st.expander("Overview JSON", expanded=False):
            st.json(overview)

        with st.expander("Positions table", expanded=False):
            if not pos_df.empty:
                st.dataframe(pos_df, width="stretch")

    if mode in {"Backtest", "Analyze"}:
        if mode == "Backtest":
            st.subheader("Backtest")
            st.caption(f"Vault: {vault_dir}")
            if selected_vault_champion_path is not None:
                st.caption(str(selected_vault_champion_path))
            if not isinstance(selected_vault_champion_payload, dict):
                st.warning("No champion selected.")
                return

            with st.expander("Champion payload", expanded=False):
                st.json(selected_vault_champion_payload)

            if st.button("Run vault champion backtest", type="primary"):
                with st.spinner("Backtesting vault champion..."):
                    _run_vault_champion_backtest(champion_payload=selected_vault_champion_payload)
            return

        if mode == "Analyze":
            st.subheader("Analyze")

            run_dirs = _discover_saved_runs()
            if not run_dirs:
                st.warning("No saved runs found under runs/.")
                return

            qp_run = _get_query_param("run")
            run_index = 0
            if qp_run:
                for i, p in enumerate(run_dirs):
                    if p.name == qp_run:
                        run_index = i
                        break

            selected_run_dir = st.selectbox(
                "Run",
                options=run_dirs,
                index=run_index,
                format_func=_run_display_name,
                key="selected_run_dir",
            )
            selected_report_payload = _load_report_json(selected_run_dir)

        if selected_run_dir is None:
            st.warning("No saved runs found under runs/.")
            return

        if selected_report_payload is None:
            run_dir = selected_run_dir
            if run_dir is None:
                st.warning("No run selected.")
                return
            if (run_dir / "optuna.db").exists() and (run_dir / "context.json").exists():
                st.warning("This run has optuna.db but no report.json (incomplete).")
                if st.button("Rebuild report from optuna.db", type="primary"):
                    with st.spinner("Rebuilding report (this can take a while)..."):
                        rebuilt = _rebuild_report_from_optuna_db(run_dir=run_dir)
                    if isinstance(rebuilt, dict):
                        st.success("Rebuild done. Reloading...")
                        st.rerun()
                return

            missing = []
            for fn in ["report.json", "optuna.db", "context.json"]:
                if not (run_dir / fn).exists():
                    missing.append(fn)
            st.warning(f"Selected run is not analyzable yet (missing: {', '.join(missing)}).")
            return

        run_dir = selected_run_dir
        report_payload = selected_report_payload

        try:
            st.session_state["opt_context"] = report_payload.get("config") or {}
            st.session_state["opt_context_tag"] = str(run_dir)
        except Exception:
            pass

        if mode == "Analyze":
            st.caption(str(run_dir))

            with st.expander("Maintenance", expanded=False):
                ctx_exists = bool((run_dir / "context.json").exists())
                local_db_exists = bool((run_dir / "optuna.db").exists())
                storage_url_hint = str((report_payload.get("config") or {}).get("storage_url") or "").strip()
                storage_url_ok = False
                if storage_url_hint:
                    u = storage_url_hint.strip().lower()
                    if u.startswith("postgres"):
                        storage_url_ok = True
                    elif u.startswith("sqlite"):
                        p = storage_url_hint
                        p_low = p.lower()
                        if p_low.startswith("sqlite:///"):
                            p = p[len("sqlite:///"):]
                        elif p_low.startswith("sqlite://"):
                            p = p[len("sqlite://"):]
                        p = str(p).lstrip("/")
                        try:
                            storage_url_ok = Path(p).exists()
                        except Exception:
                            storage_url_ok = False
                    else:
                        storage_url_ok = True

                can_rebuild = bool(ctx_exists and (local_db_exists or storage_url_ok))
                clean_before_rebuild = bool(
                    st.checkbox(
                        "Clean derived artifacts before rebuild",
                        value=False,
                        disabled=not can_rebuild,
                        key=f"an_clean_before_rebuild__{run_dir.name}",
                    )
                )

                if not can_rebuild:
                    if not ctx_exists:
                        st.warning("Cannot rebuild: missing context.json")
                    elif storage_url_hint and (not storage_url_ok):
                        st.warning(f"Cannot rebuild: storage_url points to a missing file: {storage_url_hint}")
                    else:
                        st.warning("Cannot rebuild: missing optuna.db and no valid storage_url")
                else:
                    if storage_url_hint and (not local_db_exists):
                        st.caption(f"Using storage_url from report/context: {storage_url_hint}")
                    if st.button(
                        "Force rebuild report from optuna.db",
                        type="primary",
                        key=f"an_force_rebuild__{run_dir.name}",
                    ):
                        if clean_before_rebuild:
                            derived_files = [
                                "report.json",
                                "leaderboard.csv",
                                "global_leaderboard.csv",
                                "candidates.csv",
                                "champion.json",
                                "open_analyze.url",
                            ]
                            for fn in derived_files:
                                p = run_dir / fn
                                if p.exists() and p.is_file():
                                    try:
                                        p.unlink()
                                    except Exception:
                                        pass

                            for p in run_dir.glob("context.*.json"):
                                if p.is_file() and p.name != "context.json":
                                    try:
                                        p.unlink()
                                    except Exception:
                                        pass

                            analytics_dir = run_dir / "candidate_analytics"
                            if analytics_dir.exists() and analytics_dir.is_dir():
                                try:
                                    shutil.rmtree(analytics_dir)
                                except Exception:
                                    pass

                            logs_dir = run_dir / "logs"
                            if logs_dir.exists() and logs_dir.is_dir():
                                try:
                                    shutil.rmtree(logs_dir)
                                except Exception:
                                    pass

                        with st.spinner("Rebuilding report (this can take a while)..."):
                            rebuilt = _rebuild_report_from_optuna_db(run_dir=run_dir)
                        if isinstance(rebuilt, dict):
                            st.success("Rebuild done. Reloading...")
                            st.rerun()
                        else:
                            st.error("Rebuild failed")

            data_info = report_payload.get("data") or {}
            cfg_info = report_payload.get("config") or {}

            storage_url = str(cfg_info.get("storage_url") or "").strip()
            if not storage_url:
                db_path = run_dir / "optuna.db"
                storage_url = f"sqlite:///{db_path.as_posix()}"

            trials_total = _count_trials_for_run(storage_url=storage_url, study_name_prefix=run_dir.name)

            start_ms = data_info.get("start_ms")
            end_ms = data_info.get("end_ms")
            period_days = None
            period_label = None
            if start_ms is not None and end_ms is not None:
                try:
                    start_dt = pd.to_datetime(int(start_ms), unit="ms", utc=True)
                    end_dt = pd.to_datetime(int(end_ms), unit="ms", utc=True)
                    days = max(0.0, float((end_dt - start_dt).total_seconds()) / 86400.0)
                    period_days = days
                    end_disp = end_dt
                    if end_dt == end_dt.normalize() and end_dt > start_dt:
                        end_disp = end_dt - pd.Timedelta(milliseconds=1)
                    period_label = f"{start_dt.strftime('%Y-%m-%d')} → {end_disp.strftime('%Y-%m-%d')}"
                except Exception:
                    period_days = None
                    period_label = None

            opt_start = str(data_info.get("optimized_start") or "").strip()
            opt_end = str(data_info.get("optimized_end") or "").strip()
            if period_label is None and opt_start and opt_end:
                period_label = f"{opt_start} → {opt_end}"

            train_start = str(data_info.get("train_start") or "").strip()
            train_end = str(data_info.get("train_end") or "").strip()
            test_start = str(data_info.get("test_start") or "").strip()
            test_end = str(data_info.get("test_end") or "").strip()
            train_label = f"{train_start} → {train_end}" if (train_start and train_end) else None
            test_label = f"{test_start} → {test_end}" if (test_start and test_end) else None

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Symbol", str(data_info.get("symbol")))
            c2.metric("Timeframe", str(data_info.get("timeframe")))
            c3.metric("Trials", "—" if trials_total is None else int(trials_total))
            c4.metric("Candles", int(data_info.get("candles_total", 0) or 0))
            c5.metric("Days", "—" if period_days is None else f"{float(period_days):.2f}")
            c6.metric("Saved at", str(report_payload.get("saved_at")))

            if period_label:
                st.caption(f"Optimization period: {period_label}")

            if train_label or test_label:
                st.caption(f"Train: {train_label or '—'} | Test: {test_label or '—'}")

            st.subheader("Méthode d'optimisation")
            pm_mode_used = str(cfg_info.get("pm_mode") or "auto")
            pm_enabled_used = str(pm_mode_used).strip().lower() != "none"
            tp_mode_policy_used = str(cfg_info.get("tp_mode_policy") or "auto").strip().lower()
            rr_forced_used = tp_mode_policy_used in {"rr_fixed", "force_rr", "rr"}
            rr_value_used = None
            try:
                rr_value_used = float(cfg_info.get("tp_rr_fixed") or 2.0)
            except Exception:
                rr_value_used = None

            objective_used = str(cfg_info.get("optuna_objective_metric") or "return_train_pct")
            require_positive_used = bool(
                cfg_info.get("require_positive_train_metric_for_test")
                if cfg_info.get("require_positive_train_metric_for_test") is not None
                else True
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Objective", objective_used)
            m2.metric("PM", "ON" if pm_enabled_used else "OFF")
            m3.metric("pm_mode", pm_mode_used)
            m4.metric("Force RR", "ON" if rr_forced_used else "OFF")
            m5.metric("RR", "—" if (not rr_forced_used or rr_value_used is None) else f"{float(rr_value_used):g}")

            st.caption(
                "Require positive train metric for test: "
                + ("ON" if bool(require_positive_used) else "OFF")
            )

            with st.expander("Report config", expanded=False):
                if start_ms is not None and end_ms is not None:
                    try:
                        start_dt = pd.to_datetime(int(start_ms), unit="ms", utc=True)
                        end_dt = pd.to_datetime(int(end_ms), unit="ms", utc=True)
                        days = max(0.0, float((end_dt - start_dt).total_seconds()) / 86400.0)
                        st.caption(f"Data period: {start_dt.isoformat()} → {end_dt.isoformat()} (days={days:.2f})")
                    except Exception:
                        pass

                train_frac_used = float(cfg_info.get("train_frac", 0.75) or 0.75)
                candles_total = int(data_info.get("candles_total", 0) or 0)
                candles_train = int(candles_total * float(train_frac_used)) if candles_total > 0 else 0
                candles_test = int(max(0, candles_total - candles_train)) if candles_total > 0 else 0

                k1, k2, k3 = st.columns(3)
                k1.metric("Train fraction", f"{train_frac_used:.2f}")
                k2.metric("Candles train (est.)", int(candles_train))
                k3.metric("Candles test (est.)", int(candles_test))

                st.json(cfg_info)

            st.subheader("Global leaderboard")
            ranking_metric_used = str(cfg_info.get("ranking_metric", report_payload.get("meta", {}).get("ranking_metric", "median_pnl_per_position_test")))
            st.caption(f"Ranking metric: {ranking_metric_used}")

            global_lb = None
            glb_path = run_dir / "global_leaderboard.csv"
            if glb_path.exists():
                try:
                    global_lb = pd.read_csv(glb_path)
                except Exception:
                    global_lb = None
            if global_lb is None:
                glb_rows = report_payload.get("global_leaderboard") or []
                global_lb = pd.DataFrame(glb_rows)

            sort_mode = st.radio(
                "Sort",
                options=["Post-WF (test)", "Pre-WF (train)"],
                index=0,
                horizontal=True,
                help="Post-WF uses ranking metric on test. Pre-WF uses the same metric on train.",
            )

            if global_lb is None or global_lb.empty:
                st.write("No global leaderboard.")
            else:
                metric_train = ranking_metric_used.replace("_test", "_train")
                base_cols = [
                    "rank_post_wf",
                    "rank_pre_wf",
                    "rank_delta",
                    "strategy",
                    metric_train,
                    ranking_metric_used,
                    "sharpe_train",
                    "sharpe_test",
                    "trades_train",
                    "trades_test",
                    "return_train_pct",
                    "dd_train_pct",
                    "return_test_pct",
                    "dd_test_pct",
                    "trial",
                ]
                cols = []
                seen = set()
                for c in base_cols:
                    if c in global_lb.columns and c not in seen:
                        cols.append(c)
                        seen.add(c)
                glb_display = global_lb[cols].copy()

                if sort_mode == "Pre-WF (train)" and "rank_pre_wf" in glb_display.columns:
                    glb_display = glb_display.sort_values(["rank_pre_wf"], ascending=True)
                elif sort_mode == "Post-WF (test)" and "rank_post_wf" in glb_display.columns:
                    glb_display = glb_display.sort_values(["rank_post_wf"], ascending=True)

                for c in [
                    metric_train,
                    ranking_metric_used,
                    "return_train_pct",
                    "dd_train_pct",
                    "return_test_pct",
                    "dd_test_pct",
                    "sharpe_train",
                    "sharpe_test",
                ]:
                    if c in glb_display.columns:
                        glb_display[c] = pd.to_numeric(glb_display[c], errors="coerce").round(6)
                for c in ["trades_train", "trades_test", "trial", "rank_pre_wf", "rank_post_wf", "rank_delta"]:
                    if c in glb_display.columns:
                        glb_display[c] = pd.to_numeric(glb_display[c], errors="coerce").fillna(0).astype("int64")

                st.dataframe(glb_display, width="stretch", hide_index=True)

                pick_rows = global_lb.to_dict(orient="records")
                pick_label = lambda r: f"post#{r.get('rank_post_wf')} pre#{r.get('rank_pre_wf')} | {r.get('strategy')} | trial {r.get('trial')}"
                picked = st.selectbox("Inspect candidate", options=pick_rows, format_func=pick_label)

                try:
                    picked_strategy = str(picked.get("strategy") or "")
                    picked_trial = int(pd.to_numeric(picked.get("trial"), errors="coerce"))
                except Exception:
                    picked_strategy = str(picked.get("strategy") or "")
                    picked_trial = int(picked.get("trial") or 0)

                cand_path = run_dir / "candidates.csv"
                if cand_path.exists() and picked_strategy:
                    try:
                        cand_df = pd.read_csv(cand_path)
                    except Exception:
                        cand_df = pd.DataFrame()
                    if not cand_df.empty and "strategy" in cand_df.columns and "trial" in cand_df.columns:
                        try:
                            cand_df["trial"] = pd.to_numeric(cand_df["trial"], errors="coerce")
                        except Exception:
                            pass
                        try:
                            row_match = cand_df[
                                (cand_df["strategy"].astype(str) == str(picked_strategy))
                                & (pd.to_numeric(cand_df["trial"], errors="coerce").fillna(-1).astype("int64") == int(picked_trial))
                            ]
                            if not row_match.empty:
                                picked_full = dict(picked)
                                picked_full.update(row_match.iloc[0].to_dict())
                                picked = picked_full
                        except Exception:
                            pass

                params = _candidate_params(picked)

                st.subheader("Candidate")
                st.caption(
                    f"{str(picked.get('strategy'))} | trial {picked.get('trial')} | post#{picked.get('rank_post_wf')} pre#{picked.get('rank_pre_wf')}"
                )

                tab_overview, tab_analytics, tab_params, tab_json = st.tabs(["Overview", "Analytics", "Params", "JSON"])
                with tab_overview:
                    _render_candidate_overview(picked=picked, params=params, cfg_info=cfg_info)

                with tab_analytics:
                    analytics_payload = _load_candidate_analytics(run_dir=run_dir, picked=picked)
                    if not isinstance(analytics_payload, dict):
                        st.caption("No precomputed candidate analytics for this run (older run).")
                    else:
                        _render_candidate_analytics(analytics_payload=analytics_payload)

                with tab_params:
                    st.json(params)

                with tab_json:
                    st.json(picked)

                if st.button("Save retained champion (export to vault)", type="primary"):
                    vault_path = _save_retained_champion_to_vault(run_dir=run_dir, candidate=picked, report_payload=report_payload)
                    if vault_path is None:
                        st.error("Failed to save retained champion")
                    else:
                        st.success("Retained champion saved to vault")
                        st.code(str(vault_path), language="text")

            retained_dir = run_dir / "retained_champions"
            if retained_dir.exists():
                try:
                    retained_files = sorted([p for p in retained_dir.glob("*.json") if p.is_file()], key=lambda p: p.name, reverse=True)
                except Exception:
                    retained_files = []
                if retained_files:
                    st.subheader("Retained champions (this run)")
                    st.caption(str(retained_dir))
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "file": p.name,
                                    "modified": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                                }
                                for p in retained_files
                            ]
                        ),
                        width="stretch",
                        hide_index=True,
                    )
            return

    st.subheader("Dataset")
    source = str(broker_profile)
    st.caption(f"Source: {source}")
    timeframe = st.text_input("Timeframe", value="5m")

    symbols_storage = discover_symbols(data_root=data_root, source=source)
    if not symbols_storage:
        st.warning(
            f"No symbols found under: {Path(data_root) / 'ohlcv' / source}. "
            "Download data first using projects/market-data-downloader."
        )

    if source == "binance_futures":
        base_to_variants: dict[str, list[str]] = {}
        for s in symbols_storage:
            ss = str(s)
            if not ss:
                continue
            base = ss.split("_")[0]
            if not base:
                continue
            base_to_variants.setdefault(base, []).append(ss)

        symbols_display = sorted(base_to_variants.keys()) if base_to_variants else []
        symbol_display = st.selectbox("Symbol", options=symbols_display if symbols_display else ["BTC"], index=0)

        variants = sorted(base_to_variants.get(str(symbol_display), []))
        if not variants:
            symbol_storage = f"{str(symbol_display)}_USDT"
        elif len(variants) == 1:
            symbol_storage = variants[0]
        else:
            def _variant_label(v: str) -> str:
                if v.endswith("_USDT"):
                    return "USDT"
                if v.endswith("_USDT_USDT"):
                    return "USDT:USDT"
                if v.endswith("_USDC_USDC"):
                    return "USDC:USDC"
                suffix = "_".join(v.split("_")[1:])
                return suffix or v

            variant_options = [{"storage": v, "label": _variant_label(v)} for v in variants]
            picked = st.selectbox(
                "Contract",
                options=variant_options,
                format_func=lambda x: str(x.get("label")),
            )
            symbol_storage = str(picked.get("storage"))

            if str(picked.get("label")) != "USDT":
                symbol_display = f"{symbol_display}_{picked.get('label')}"
    else:
        symbol_storage = st.selectbox("Symbol", options=symbols_storage if symbols_storage else ["BTC"], index=0)
        symbol_display = str(symbol_storage)

    symbol = str(symbol_storage)

    st.subheader("Period (optional)")
    start_str = st.text_input("Start (YYYY-MM-DD)", key="start_date")
    end_str = st.text_input("End (YYYY-MM-DD)", key="end_date")

    show_available_period = st.checkbox("Show available period", value=False)
    if show_available_period:
        df_preview = pd.DataFrame()
        with st.spinner("Loading data for period preview..."):
            try:
                df_preview = _load_ohlcv_cached(
                    data_root=str(data_root),
                    source=source,
                    symbol=symbol,
                    timeframe=timeframe,
                )
            except FileNotFoundError:
                tf_dir = Path(data_root) / "ohlcv" / str(source) / str(symbol) / str(timeframe)
                st.error(f"Missing data folder: {tf_dir}")
                st.info("Download the missing timeframe using projects/market-data-downloader (Binance: symbol BTC/USDT, timeframe 5m).")
            except Exception as e:
                st.error(f"Failed to load OHLCV preview: {type(e).__name__}: {e}")
        if not df_preview.empty:
            start_avail = _ms_to_yyyy_mm_dd(int(df_preview["timestamp_ms"].iloc[0]))
            end_avail = _ms_to_yyyy_mm_dd(int(df_preview["timestamp_ms"].iloc[-1]))
            st.write(f"Available: {start_avail} → {end_avail} (UTC)")
            st.button(
                "Set Start/End to available period",
                on_click=_set_period,
                kwargs={"start": start_avail, "end": end_avail},
            )

    st.subheader("Strategies")
    all_strats = builtin_strategies()
    all_names = [s.name for s in all_strats]
    selected = st.multiselect("Enabled strategies", options=all_names, default=all_names, key="enabled_strategies")

    if stop_mode == "Trials":
        n_strats = len(selected) if selected else len(all_names)
        st.caption(f"Max trials is applied per strategy. With {n_strats} strategies enabled: up to {int(max_trials) * int(n_strats):,} trials total.")
        st.caption("Strategies run sequentially, but use multiple worker processes per strategy.")

    if "opt_active_run_dir" not in st.session_state:
        st.session_state["opt_active_run_dir"] = None

    if not st.session_state.get("opt_running"):
        if st.button("Run optimization", type="primary"):
            with st.spinner("Loading data..."):
                try:
                    df = _load_ohlcv_cached(data_root=str(data_root), source=source, symbol=symbol, timeframe=timeframe)
                except FileNotFoundError:
                    tf_dir = Path(data_root) / "ohlcv" / str(source) / str(symbol) / str(timeframe)
                    st.error(f"Missing data folder: {tf_dir}")
                    st.info("Download the missing timeframe using projects/market-data-downloader.")
                    return
                except Exception as e:
                    st.error(f"Failed to load OHLCV: {type(e).__name__}: {e}")
                    return

            if start_str.strip():
                start_ms = _parse_date_to_ms(start_str, is_end=False)
                df = df[df["timestamp_ms"] >= start_ms]
            else:
                start_ms = None
            if end_str.strip():
                end_ms = _parse_date_to_ms(end_str, is_end=True)
                df = df[df["timestamp_ms"] < end_ms]
            else:
                end_ms = None
            df = df.reset_index(drop=True)

            if df.empty:
                st.error("No candles in selected period.")
                return

            optimized_start = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[0]))
            optimized_end = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[-1]))
            n = len(df)
            cut = max(1, min(n - 1, int(n * float(train_frac)))) if n >= 10 else n

            train_start = optimized_start
            train_end = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[max(0, cut - 1)]))
            test_start = _ms_to_yyyy_mm_dd(int(df["timestamp_ms"].iloc[cut])) if cut < n else train_end
            test_end = optimized_end

            st.write(f"Loaded {len(df):,} candles")
            st.write(f"Optimized period: {optimized_start} → {optimized_end} (UTC)")
            if n >= 2 and cut < n:
                train_pct = int(round(float(train_frac) * 100.0))
                test_pct = 100 - train_pct
                st.write(f"Train ({train_pct}%): {train_start} → {train_end} ({cut:,} candles)")
                st.write(f"Test ({test_pct}%): {test_start} → {test_end} ({(n - cut):,} candles)")

            cfg = OptimizationConfig(
                initial_equity=initial_equity,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                dd_threshold_pct=dd_threshold_pct,
                min_trades_train=min_trades_train,
                min_trades_test=min_trades_test,
                max_trials=max_trials,
                time_budget_seconds=int(time_budget_minutes * 60) if time_budget_minutes > 0 else None,
                n_jobs=1,
                timeframe=timeframe,
                pm_mode=str(pm_mode),
                strategies=selected if selected else None,
                risk_mode=str(risk_mode),
                risk_pct=float(risk_pct),
                fixed_notional_pct_equity=float(fixed_notional_pct_equity),
                max_position_notional_pct_equity=float(max_position_notional_pct_equity),
                max_leverage=max_leverage,
                min_qty=float(min_qty),
                qty_step=float(qty_step),
                min_notional=float(min_notional),
                train_frac=float(train_frac),
                pareto_candidates_max=int(pareto_candidates_max),
                candidate_pool=str(candidate_pool),
                global_top_k=int(global_top_k),
                ranking_metric=str(ranking_metric),
                require_positive_train_metric_for_test=bool(require_positive_train_metric_for_test),
                optuna_objective_metric=str(optuna_objective_metric),
                tp_mode_policy=str(tp_mode_policy),
                tp_rr_fixed=float(tp_rr_fixed),
                broker_profile=str(broker_profile_cfg),
                perps_maintenance_margin_rate=float(perps_maintenance_margin_rate),
                cfd_initial_margin_rate=float(cfd_initial_margin_rate),
                cfd_stopout_margin_level=float(cfd_stopout_margin_level),
            )

            run_id = time.strftime("%Y%m%d_%H%M%S")
            run_dir = _PROJECT_ROOT / "runs" / f"{run_id}_{symbol_display}_{timeframe}"
            run_dir.mkdir(parents=True, exist_ok=True)

            storage_url_ctx = str(storage_url_ui or "").strip()
            try:
                u = storage_url_ctx.lower()
            except Exception:
                u = ""
            if u.startswith("sqlite"):
                storage_url_ctx = ""

            ctx_payload = {
                "data_root": str(data_root),
                "source": source,
                "symbol": symbol,
                "symbol_display": str(symbol_display),
                "symbol_storage": str(symbol_storage),
                "timeframe": timeframe,
                "start_str": start_str,
                "end_str": end_str,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "optimized_start": optimized_start,
                "optimized_end": optimized_end,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "cut": cut,
                "train_frac": float(train_frac),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "multiprocess": True,
                "workers": workers,
                "stop_mode": stop_mode,
            }

            run_dir.mkdir(parents=True, exist_ok=True)
            stop_flag = run_dir / "stop.flag"
            if stop_flag.exists():
                try:
                    stop_flag.unlink()
                except Exception:
                    pass

            ctx_path = run_dir / "context.json"
            ctx_path.write_text(
                json.dumps(
                    {
                        "data": {
                            "data_root": str(ctx_payload.get("data_root")),
                            "source": str(ctx_payload.get("source")),
                            "symbol": str(ctx_payload.get("symbol_storage")),
                            "timeframe": str(ctx_payload.get("timeframe")),
                            "start_ms": ctx_payload.get("start_ms"),
                            "end_ms": ctx_payload.get("end_ms"),
                        },
                        "config": {
                            "initial_equity": cfg.initial_equity,
                            "fee_bps": cfg.fee_bps,
                            "slippage_bps": cfg.slippage_bps,
                            "dd_threshold_pct": cfg.dd_threshold_pct,
                            "broker_preset": str(broker_preset),
                            "broker_custom_overrides": bool(st.session_state.get("broker_custom_overrides") or False),
                            "storage_url": str(storage_url_ctx),
                            "optuna_objective_metric": str(optuna_objective_metric),
                            "require_positive_train_metric_for_test": bool(require_positive_train_metric_for_test),
                            "min_trades_train": cfg.min_trades_train,
                            "min_trades_test": cfg.min_trades_test,
                            "max_trials": cfg.max_trials,
                            "time_budget_seconds": cfg.time_budget_seconds,
                            "stop_mode": str(stop_mode),
                            "workers": int(workers),
                            "timeframe": cfg.timeframe,
                            "pm_mode": cfg.pm_mode,
                            "strategies": cfg.strategies,
                            "risk_mode": cfg.risk_mode,
                            "risk_pct": cfg.risk_pct,
                            "fixed_notional_pct_equity": cfg.fixed_notional_pct_equity,
                            "max_position_notional_pct_equity": cfg.max_position_notional_pct_equity,
                            "max_leverage": getattr(cfg, "max_leverage", None),
                            "min_qty": float(getattr(cfg, "min_qty", 0.0) or 0.0),
                            "qty_step": float(getattr(cfg, "qty_step", 0.0) or 0.0),
                            "min_notional": float(getattr(cfg, "min_notional", 0.0) or 0.0),
                            "broker_profile": str(getattr(cfg, "broker_profile", "perps")),
                            "perps_maintenance_margin_rate": float(getattr(cfg, "perps_maintenance_margin_rate", 0.01) or 0.01),
                            "cfd_initial_margin_rate": float(getattr(cfg, "cfd_initial_margin_rate", 0.01) or 0.01),
                            "cfd_stopout_margin_level": float(getattr(cfg, "cfd_stopout_margin_level", 0.5) or 0.5),
                            "pareto_candidates_max": cfg.pareto_candidates_max,
                            "candidate_pool": cfg.candidate_pool,
                            "global_top_k": cfg.global_top_k,
                            "ranking_metric": cfg.ranking_metric,
                            "train_frac": cfg.train_frac,
                            "tp_mode_policy": str(getattr(cfg, "tp_mode_policy", "auto") or "auto"),
                            "tp_rr_fixed": float(getattr(cfg, "tp_rr_fixed", 2.0) or 2.0),
                        },
                        "run_id": str(run_id),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(_SRC_DIR) + os.pathsep + str(env.get("PYTHONPATH", ""))
            cmd = [
                sys.executable,
                "-m",
                "hyperliquid_ohlcv_optimizer.optimize.run_optimize",
                "--run-dir",
                str(run_dir),
                "--workers",
                str(int(workers)),
            ]
            subprocess.Popen(cmd, env=env)

            st.session_state["opt_launched_run_dir"] = str(run_dir)

            st.info("Optimization launched in a separate process. Follow progress in the terminal.")
            st.code(str(run_dir), language="text")
            st.code(
                f"python -m hyperliquid_ohlcv_optimizer.optimize.stop_run --run-dir \"{run_dir}\" --reason stop",
                language="bash",
            )

    if st.session_state.get("opt_launched_run_dir"):
        st.caption("This UI does not control running optimizations. Use the CLI to stop if needed.")

    report = st.session_state.get("opt_report")
    err = st.session_state.get("opt_error")
    ctx_payload = st.session_state.get("opt_context") or {}
    if err:
        st.error(err)

    if report is not None and not st.session_state.get("opt_running"):
        df = ctx_payload.get("df")
        cfg = ctx_payload.get("cfg")

        st.success("Done")

        if st.session_state.get("opt_saved_dir") is None:
            run_id = str(ctx_payload.get("run_id") or time.strftime("%Y%m%d_%H%M%S"))
            run_dir = Path(str(ctx_payload.get("run_dir") or (_PROJECT_ROOT / "runs" / f"{run_id}_{ctx_payload.get('symbol')}_{ctx_payload.get('timeframe')}")))
            run_dir.mkdir(parents=True, exist_ok=True)

            leaderboard_path = run_dir / "leaderboard.csv"
            global_leaderboard_path = run_dir / "global_leaderboard.csv"
            candidates_path = run_dir / "candidates.csv"
            champion_path = run_dir / "champion.json"
            report_path = run_dir / "report.json"

            try:
                report.leaderboard.to_csv(leaderboard_path, index=False)

                if getattr(report, "global_leaderboard", None) is not None:
                    try:
                        report.global_leaderboard.to_csv(global_leaderboard_path, index=False)
                    except Exception:
                        pass

                if getattr(report, "candidates", None) is not None:
                    try:
                        report.candidates.to_csv(candidates_path, index=False)
                    except Exception:
                        pass

                if report.champion is not None:
                    champion_path.write_text(
                        json.dumps(report.champion, indent=2, ensure_ascii=False), encoding="utf-8"
                    )

                report_payload = {
                    "project": "ohlcv-optimizer-v1",
                    "version": 1,
                    "saved_at": run_id,
                    "meta": getattr(report, "meta", None),
                    "strategies_skipped": getattr(report, "strategies_skipped", None),
                    "champion_global": getattr(report, "champion_global", None),
                    "champions_by_strategy": getattr(report, "champions_by_strategy", None),
                    "data": {
                        "data_root": str(ctx_payload.get("data_root")),
                        "source": ctx_payload.get("source"),
                        "symbol": ctx_payload.get("symbol_display") or ctx_payload.get("symbol"),
                        "symbol_display": ctx_payload.get("symbol_display"),
                        "symbol_storage": ctx_payload.get("symbol_storage") or ctx_payload.get("symbol"),
                        "timeframe": ctx_payload.get("timeframe"),
                        "start": ctx_payload.get("start_str"),
                        "end": ctx_payload.get("end_str"),
                        "start_ms": ctx_payload.get("start_ms"),
                        "end_ms": ctx_payload.get("end_ms"),
                        "optimized_start": ctx_payload.get("optimized_start"),
                        "optimized_end": ctx_payload.get("optimized_end"),
                        "train_start": ctx_payload.get("train_start"),
                        "train_end": ctx_payload.get("train_end"),
                        "test_start": ctx_payload.get("test_start"),
                        "test_end": ctx_payload.get("test_end"),
                        "candles_total": int(len(df)) if df is not None else None,
                        "candles_train": int(ctx_payload.get("cut", 0)),
                        "candles_test": int(max(0, int(len(df)) - int(ctx_payload.get("cut", 0)))) if df is not None else None,
                    },
                    "config": {
                        "initial_equity": cfg.initial_equity if cfg is not None else None,
                        "fee_bps": cfg.fee_bps if cfg is not None else None,
                        "slippage_bps": cfg.slippage_bps if cfg is not None else None,
                        "broker_preset": str(broker_preset),
                        "broker_custom_overrides": bool(st.session_state.get("broker_custom_overrides") or False),
                        "dd_threshold_pct": cfg.dd_threshold_pct if cfg is not None else None,
                        "min_trades_train": cfg.min_trades_train if cfg is not None else None,
                        "min_trades_test": cfg.min_trades_test if cfg is not None else None,
                        "max_trials": cfg.max_trials if cfg is not None else None,
                        "time_budget_seconds": cfg.time_budget_seconds if cfg is not None else None,
                        "n_jobs": cfg.n_jobs if cfg is not None else None,
                        "timeframe": cfg.timeframe if cfg is not None else None,
                        "pm_mode": cfg.pm_mode if cfg is not None else None,
                        "strategies": cfg.strategies if cfg is not None else None,
                        "pareto_candidates_max": cfg.pareto_candidates_max if cfg is not None else None,
                        "candidate_pool": cfg.candidate_pool if cfg is not None else None,
                        "global_top_k": cfg.global_top_k if cfg is not None else None,
                        "ranking_metric": cfg.ranking_metric if cfg is not None else None,
                        "train_frac": cfg.train_frac if cfg is not None else None,
                        "multiprocess": bool(ctx_payload.get("multiprocess")),
                        "workers": int(ctx_payload.get("workers", 1)),
                    },
                    "leaderboard": report.leaderboard.to_dict(orient="records"),
                    "global_leaderboard": (
                        report.global_leaderboard.to_dict(orient="records")
                        if getattr(report, "global_leaderboard", None) is not None
                        else None
                    ),
                    "global_leaderboard_file": "global_leaderboard.csv" if global_leaderboard_path.exists() else None,
                    "candidates_file": "candidates.csv" if candidates_path.exists() else None,
                    "champion": report.champion,
                }

                report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

                try:
                    (run_dir / "open_analyze.url").write_text(
                        "[InternetShortcut]\n"
                        f"URL=http://localhost:8501/?mode=Analyze&run={run_dir.name}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass

                st.session_state["opt_saved_dir"] = str(run_dir)
            except Exception as e:
                st.warning(f"Failed to save run artifacts: {type(e).__name__}: {e}")

        if st.session_state.get("opt_saved_dir") is not None:
            st.caption(f"Saved run to: {st.session_state.get('opt_saved_dir')}")

        st.subheader("Leaderboard")
        lb = report.leaderboard
        if lb is None or lb.empty:
            st.write("No leaderboard rows.")
        else:
            requested = []
            if cfg is not None and getattr(cfg, "strategies", None):
                requested = list(cfg.strategies)
            else:
                requested = [s.name for s in builtin_strategies()]

            missing = [s for s in requested if s not in set(lb.get("strategy", pd.Series(dtype="object")))]
            if missing:
                st.warning(
                    "Some selected strategies are missing from the leaderboard. "
                    "This usually means the optimization was stopped before reaching them, or the strategy had no COMPLETE trials. "
                    f"Missing: {', '.join(missing)}"
                )

            base_cols = [
                "strategy",
                "return_test_pct",
                "dd_test_pct",
                "sharpe_test",
                "trades_test",
                "return_train_pct",
                "dd_train_pct",
                "trades_train",
                "trial",
                "seconds",
            ]
            cols = [c for c in base_cols if c in lb.columns]
            lb_display = lb[cols].copy()

            for c in ["return_test_pct", "dd_test_pct", "return_train_pct", "dd_train_pct", "seconds", "sharpe_test"]:
                if c in lb_display.columns:
                    lb_display[c] = pd.to_numeric(lb_display[c], errors="coerce").round(3)

            for c in ["trades_train", "trades_test", "trial"]:
                if c in lb_display.columns:
                    lb_display[c] = pd.to_numeric(lb_display[c], errors="coerce").fillna(0).astype("int64")

            lb_display = lb_display.rename(
                columns={
                    "strategy": "Strategy",
                    "return_test_pct": "Return Test (%)",
                    "dd_test_pct": "DD Test (%)",
                    "sharpe_test": "Sharpe Test",
                    "trades_test": "Trades Test",
                    "return_train_pct": "Return Train (%)",
                    "dd_train_pct": "DD Train (%)",
                    "trades_train": "Trades Train",
                    "trial": "Trial",
                    "seconds": "Seconds",
                }
            )

            st.dataframe(lb_display, width="stretch", hide_index=True)

            with st.expander("Leaderboard (full)", expanded=False):
                st.dataframe(lb, width="stretch")

        if report.champion is not None and df is not None and cfg is not None:
            st.subheader("Champion")
            st.json(report.champion)

            st.subheader("Champion analytics (positions)")
            bt_cfg_full = build_backtest_config_from_params(params=report.champion, base=cfg)
            all_strats = builtin_strategies()
            strat = next((s for s in all_strats if s.name == str(report.champion.get("strategy"))), None)
            if strat is None:
                st.warning("Cannot compute champion analytics: strategy not found.")
            else:
                ctx = StrategyContext(timeframe=str(ctx_payload.get("timeframe")))
                sig_all = strat.compute_signal(df, report.champion, ctx)
                res_full = run_backtest(df=df, signal=sig_all, config=bt_cfg_full)
                pos_df, overview = summarize_positions(res_full.trades)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Positions", int(overview.get("positions", 0)))
                c2.metric("Win rate", f"{100.0 * float(overview.get('win_rate', 0.0)):.1f}%")
                c3.metric("TP1 hit rate", f"{100.0 * float(overview.get('tp1_hit_rate', 0.0)):.1f}%")
                c4.metric("Avg fills/pos", f"{float(overview.get('avg_fills_per_position', 0.0)):.2f}")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Median PnL/pos", f"{float(overview.get('median_pnl_per_position', 0.0)):.4f}")
                c6.metric("Avg PnL/pos", f"{float(overview.get('avg_pnl_per_position', 0.0)):.4f}")
                c7.metric("TP1 PnL share", f"{100.0 * float(overview.get('tp1_pnl_share', 0.0)):.1f}%")
                c8.metric("Sharpe (pos PnL)", f"{float(overview.get('sharpe_pnl_per_position', 0.0)):.2f}")

                eq = res_full.equity_curve
                start_e = float(eq[0]) if eq.size else float(cfg.initial_equity)
                end_e = float(eq[-1]) if eq.size else float(cfg.initial_equity)
                total_return = (end_e / max(start_e, 1e-12)) - 1.0

                start_ms = int(df["timestamp_ms"].iloc[0])
                end_ms = int(df["timestamp_ms"].iloc[-1])
                period_days = max(1e-9, float(end_ms - start_ms) / 86_400_000.0)

                ann_return = (1.0 + total_return) ** (365.0 / period_days) - 1.0
                mon_days = 365.25 / 12.0
                mon_return = (1.0 + total_return) ** (mon_days / period_days) - 1.0

                c9, c10, c11, c12 = st.columns(4)
                c9.metric("Return (period)", f"{total_return * 100.0:.2f}%")
                c10.metric("Return annualized", f"{ann_return * 100.0:.2f}%")
                c11.metric("Return monthly", f"{mon_return * 100.0:.2f}%")
                c12.metric("Period days", f"{period_days:.2f}")

                with st.expander("Exit reasons (final)", expanded=False):
                    st.json(overview.get("final_exit_reason_dist", {}))

                with st.expander("Overview JSON", expanded=False):
                    st.json(overview)

                with st.expander("Positions table", expanded=False):
                    if not pos_df.empty:
                        st.dataframe(pos_df)

            manifest = _build_manifest(
                champion=report.champion,
                data_root=str(ctx_payload.get("data_root")),
                source=str(ctx_payload.get("source")),
                symbol=str(ctx_payload.get("symbol")),
                timeframe=str(ctx_payload.get("timeframe")),
                start=str(ctx_payload.get("start_str")),
                end=str(ctx_payload.get("end_str")),
                cfg=cfg,
            )
            manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
            st.download_button(
                "Download paper manifest (JSON)",
                data=manifest_json,
                file_name="paper_manifest.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
