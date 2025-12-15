from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
import threading
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from hyperliquid_ohlcv_optimizer.backtest.backtester import run_backtest
from hyperliquid_ohlcv_optimizer.backtest.trade_analysis import summarize_positions
from hyperliquid_ohlcv_optimizer.backtest.types import (
    BacktestConfig,
    CommonTradeParams,
    ExecutionCosts,
    GridConfig,
    MartingaleConfig,
    PositionManagerConfig,
    RiskConfig,
)
from hyperliquid_ohlcv_optimizer.strategies.base import StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.registry import builtin_strategies


@dataclass(frozen=True)
class OptimizationConfig:
    initial_equity: float
    fee_bps: float
    slippage_bps: float
    dd_threshold_pct: float
    max_trials: int
    time_budget_seconds: int | None
    n_jobs: int
    min_trades_train: int = 0
    min_trades_test: int = 0
    timeframe: str = "5m"
    pm_mode: str = "auto"  # auto|none|grid|martingale
    strategies: list[str] | None = None
    risk_pct: float = 0.01
    max_position_notional_pct_equity: float = 100.0
    pareto_candidates_max: int = 50
    candidate_pool: str = "pareto"  # pareto|complete
    global_top_k: int = 50
    ranking_metric: str = "median_pnl_per_position_test"
    train_frac: float = 0.75


@dataclass(frozen=True)
class OptimizationReport:
    leaderboard: pd.DataFrame
    champion: dict | None
    global_leaderboard: pd.DataFrame | None = None
    candidates: pd.DataFrame | None = None
    champions_by_strategy: dict[str, dict] | None = None
    champion_global: dict | None = None
    strategies_skipped: dict[str, str] | None = None
    meta: dict | None = None


def _sharpe_from_equity(equity: np.ndarray) -> float:
    if equity.size < 2:
        return 0.0
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    mu = float(np.mean(rets)) if rets.size else 0.0
    sigma = float(np.std(rets, ddof=0)) if rets.size else 0.0
    if sigma <= 1e-12:
        return 0.0
    return float((mu / sigma) * np.sqrt(float(rets.size)))


def _metric_train_name(metric_test_name: str) -> str:
    return str(metric_test_name).replace("_test", "_train")


def _metric_sort_ascending(metric_name: str) -> bool:
    m = str(metric_name)
    return ("dd_" in m) or m.startswith("dd_") or m.endswith("_dd")


def _trial_pool(*, study: optuna.Study, config: OptimizationConfig) -> list[optuna.trial.FrozenTrial]:
    mode = str(getattr(config, "candidate_pool", "pareto") or "pareto").strip().lower()
    if mode == "complete":
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    else:
        trials = list(study.best_trials)
        if not trials:
            trials = [
                t
                for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
            ]

    trials = sorted(trials, key=lambda t: (t.values[1], -t.values[0]) if t.values is not None else (0.0, 0.0))
    trials = trials[: int(getattr(config, "pareto_candidates_max", 50) or 50)]
    return trials


def _add_rank_columns(*, df: pd.DataFrame, config: OptimizationConfig) -> pd.DataFrame:
    if df.empty:
        df["rank_pre_wf"] = pd.Series(dtype="int64")
        df["rank_post_wf"] = pd.Series(dtype="int64")
        df["rank_delta"] = pd.Series(dtype="int64")
        return df

    metric_test = str(getattr(config, "ranking_metric", "median_pnl_per_position_test"))
    metric_train = _metric_train_name(metric_test)

    df = df.copy()
    df["rank_pre_wf"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df["rank_post_wf"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    eligible_mask = df.get("eligible")
    if eligible_mask is None:
        eligible_mask = pd.Series([True] * len(df))

    eligible_mask = eligible_mask.fillna(False).astype(bool)

    if metric_train in df.columns:
        asc = _metric_sort_ascending(metric_train)
        ranks = df.loc[eligible_mask, metric_train].rank(method="min", ascending=asc)
        df.loc[eligible_mask, "rank_pre_wf"] = ranks.astype("Int64")

    dd_ok_mask = eligible_mask
    if "dd_test_pct" in df.columns:
        dd_ok_mask = dd_ok_mask & (pd.to_numeric(df["dd_test_pct"], errors="coerce") <= float(config.dd_threshold_pct))

    if metric_test in df.columns:
        asc = _metric_sort_ascending(metric_test)
        ranks = df.loc[dd_ok_mask, metric_test].rank(method="min", ascending=asc)
        df.loc[dd_ok_mask, "rank_post_wf"] = ranks.astype("Int64")

    df["rank_delta"] = (df["rank_post_wf"].astype("Int64") - df["rank_pre_wf"].astype("Int64")).astype("Int64")
    return df


def _build_global_outputs(*, candidates: pd.DataFrame, config: OptimizationConfig) -> tuple[pd.DataFrame, dict | None, dict[str, dict]]:
    if candidates is None or candidates.empty:
        return (pd.DataFrame(), None, {})

    metric_test = str(getattr(config, "ranking_metric", "median_pnl_per_position_test"))
    asc = _metric_sort_ascending(metric_test)

    df = candidates.copy()
    eligible_mask = df.get("eligible")
    if eligible_mask is None:
        eligible_mask = pd.Series([True] * len(df))
    eligible_mask = eligible_mask.fillna(False).astype(bool)

    dd_mask = pd.Series([True] * len(df))
    if "dd_test_pct" in df.columns:
        dd_mask = pd.to_numeric(df["dd_test_pct"], errors="coerce") <= float(config.dd_threshold_pct)

    pool = df[eligible_mask & dd_mask].copy()
    if pool.empty:
        return (pd.DataFrame(), None, {})

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if metric_test in pool.columns:
        sort_cols.append(metric_test)
        ascending.append(asc)
    if "dd_test_pct" in pool.columns:
        sort_cols.append("dd_test_pct")
        ascending.append(True)
    if "return_test_pct" in pool.columns:
        sort_cols.append("return_test_pct")
        ascending.append(False)

    pool = pool.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    top_k = int(getattr(config, "global_top_k", 50) or 50)
    global_lb = pool.iloc[:top_k].reset_index(drop=True)
    champion_global = global_lb.iloc[0].to_dict() if not global_lb.empty else None

    champions_by_strategy: dict[str, dict] = {}
    if "strategy" in pool.columns:
        for strat_name, grp in pool.groupby("strategy", dropna=False):
            grp_sorted = grp.sort_values(sort_cols, ascending=ascending)
            if not grp_sorted.empty:
                champions_by_strategy[str(strat_name)] = grp_sorted.iloc[0].to_dict()

    return (global_lb, champion_global, champions_by_strategy)


def build_report_from_storage(
    *,
    df: pd.DataFrame,
    config: OptimizationConfig,
    storage_url: str,
) -> OptimizationReport:
    df_train, df_test = _train_test_split(df, train_frac=float(getattr(config, "train_frac", 0.75)))

    min_trades_train = int(config.min_trades_train)
    min_trades_test = int(config.min_trades_test)

    strategies = builtin_strategies()
    if config.strategies is not None:
        allowed = set([s.strip() for s in config.strategies if s and s.strip()])
        if allowed:
            strategies = [s for s in strategies if s.name in allowed]

    candidates_rows: list[dict] = []
    strategies_skipped: dict[str, str] = {}

    storage = optuna.storages.RDBStorage(
        str(storage_url),
        engine_kwargs={"connect_args": {"timeout": 60}},
    )

    for strat in strategies:
        try:
            study = optuna.load_study(study_name=strat.name, storage=storage)
        except Exception:
            strategies_skipped[strat.name] = "no_study"
            continue

        ctx = StrategyContext(timeframe=config.timeframe)

        trials = _trial_pool(study=study, config=config)
        if not trials:
            strategies_skipped[strat.name] = "no_complete_trials"
            continue

        sig_all_cache: dict[int, pd.Series] = {}

        for tr in trials:
            bt_cfg = _build_backtest_config_from_params(params=tr.params, base=config)

            if tr.number not in sig_all_cache:
                sig_all_cache[tr.number] = strat.compute_signal(df, tr.params, ctx)

            sig_all = sig_all_cache[tr.number]
            sig_train = sig_all.iloc[: len(df_train)].reset_index(drop=True)
            sig_test = sig_all.iloc[len(df_train) :].reset_index(drop=True)

            res_train = run_backtest(df=df_train, signal=sig_train, config=bt_cfg)
            res_test = run_backtest(df=df_test, signal=sig_test, config=bt_cfg)

            sharpe_train = _sharpe_from_equity(res_train.equity_curve)
            sharpe_test = _sharpe_from_equity(res_test.equity_curve)

            _, overview_train = summarize_positions(res_train.trades)
            _, overview_test = summarize_positions(res_test.trades)

            trades_train = _count_positions(res_train.trades)
            trades_test = _count_positions(res_test.trades)
            eligible = (trades_train >= min_trades_train) and (trades_test >= min_trades_test)

            median_pnl_train = float(overview_train.get("median_pnl_per_position", 0.0))
            median_pnl_test = float(overview_test.get("median_pnl_per_position", 0.0))
            avg_pnl_train = float(overview_train.get("avg_pnl_per_position", 0.0))
            avg_pnl_test = float(overview_test.get("avg_pnl_per_position", 0.0))
            sharpe_pnl_train = float(overview_train.get("sharpe_pnl_per_position", 0.0))
            sharpe_pnl_test = float(overview_test.get("sharpe_pnl_per_position", 0.0))

            row = {
                "strategy": strat.name,
                "return_train_pct": res_train.net_return_pct,
                "dd_train_pct": res_train.max_drawdown_pct,
                "return_test_pct": res_test.net_return_pct,
                "dd_test_pct": res_test.max_drawdown_pct,
                "sharpe_train": sharpe_train,
                "sharpe_test": sharpe_test,
                "median_pnl_per_position_train": median_pnl_train,
                "median_pnl_per_position_test": median_pnl_test,
                "avg_pnl_per_position_train": avg_pnl_train,
                "avg_pnl_per_position_test": avg_pnl_test,
                "sharpe_pnl_per_position_train": sharpe_pnl_train,
                "sharpe_pnl_per_position_test": sharpe_pnl_test,
                "median_pnl_per_position_train_pct": (median_pnl_train / max(float(config.initial_equity), 1e-12)) * 100.0,
                "median_pnl_per_position_test_pct": (median_pnl_test / max(float(config.initial_equity), 1e-12)) * 100.0,
                "trades_train": trades_train,
                "trades_test": trades_test,
                "eligible": eligible,
                "trial": tr.number,
            }
            for k, v in tr.params.items():
                row[k] = v

            candidates_rows.append(row)

    candidates = pd.DataFrame(candidates_rows)
    if not candidates.empty:
        candidates = _add_rank_columns(df=candidates, config=config)

    global_lb, champion_global, champions_by_strategy = _build_global_outputs(candidates=candidates, config=config)
    leaderboard = pd.DataFrame(list(champions_by_strategy.values())) if champions_by_strategy else pd.DataFrame()

    return OptimizationReport(
        leaderboard=leaderboard,
        champion=champion_global,
        global_leaderboard=global_lb,
        candidates=candidates,
        champions_by_strategy=champions_by_strategy,
        champion_global=champion_global,
        strategies_skipped=strategies_skipped,
        meta={
            "ranking_metric": str(getattr(config, "ranking_metric", "median_pnl_per_position_test")),
            "candidate_pool": str(getattr(config, "candidate_pool", "pareto")),
            "global_top_k": int(getattr(config, "global_top_k", 50) or 50),
        },
    )


def _count_positions(trades: pd.DataFrame) -> int:
    if trades is None or trades.empty:
        return 0
    if "position_id" in trades.columns:
        s = trades["position_id"].dropna()
    elif "entry_ts" in trades.columns:
        s = trades["entry_ts"].dropna()
    else:
        return int(len(trades))
    if s.empty:
        return 0
    return int(s.nunique(dropna=True))


def _make_sampler(*, seed: int):
    samplers = optuna.samplers
    if hasattr(samplers, "MOTPESampler"):
        return samplers.MOTPESampler(seed=seed)
    if hasattr(samplers, "NSGAIISampler"):
        return samplers.NSGAIISampler(seed=seed)
    return samplers.RandomSampler(seed=seed)


def _train_test_split(df: pd.DataFrame, train_frac: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 10:
        return (df.copy(), df.copy())
    cut = max(1, min(n - 1, int(n * train_frac)))
    return (df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True))


def _build_backtest_config(*, trial: optuna.Trial, base: OptimizationConfig) -> BacktestConfig:
    tp_mode = str(trial.suggest_categorical("tp_mode", ["pct", "rr"]))
    tp_pct = 1.0
    tp_rr = 2.0
    if tp_mode == "pct":
        tp_pct = float(trial.suggest_float("tp_pct", 0.1, 10.0))
    else:
        tp_rr = float(trial.suggest_float("tp_rr", 0.2, 10.0))

    tp_mgmt = str(trial.suggest_categorical("tp_mgmt", ["full", "partial_trailing"]))
    tp1_close_frac = 0.5
    tp_trail_pct = 0.5
    if tp_mgmt == "partial_trailing":
        tp1_close_frac = float(trial.suggest_float("tp1_close_frac", 0.1, 0.9))
        tp_trail_pct = float(trial.suggest_float("tp_trail_pct", 0.05, 5.0))

    sl_type = str(trial.suggest_categorical("sl_type", ["pct", "atr"]))
    sl_trailing = bool(trial.suggest_categorical("sl_trailing", [False, True]))
    exit_on_flat = bool(trial.suggest_categorical("exit_on_flat", [False, True]))

    sl_pct = 1.0
    sl_atr_period = 14
    sl_atr_mult = 2.0

    if sl_type == "pct":
        sl_pct = float(trial.suggest_float("sl_pct", 0.1, 10.0))
    else:
        sl_atr_period = int(trial.suggest_int("sl_atr_period", 5, 60))
        sl_atr_mult = float(trial.suggest_float("sl_atr_mult", 0.5, 6.0))

    costs = ExecutionCosts(fee_bps=base.fee_bps, slippage_bps=base.slippage_bps)
    risk = RiskConfig(risk_pct=base.risk_pct, max_position_notional_pct_equity=base.max_position_notional_pct_equity)
    common = CommonTradeParams(
        tp_mode=tp_mode,
        tp_pct=tp_pct,
        tp_rr=tp_rr,
        tp_mgmt=tp_mgmt,
        tp1_close_frac=tp1_close_frac,
        tp_trail_pct=tp_trail_pct,
        sl_type=sl_type,
        sl_pct=sl_pct,
        sl_atr_period=sl_atr_period,
        sl_atr_mult=sl_atr_mult,
        sl_trailing=sl_trailing,
        exit_on_flat=exit_on_flat,
    )

    pm_mode = base.pm_mode
    if pm_mode == "auto":
        pm_mode = str(trial.suggest_categorical("pm_mode", ["none", "grid", "martingale"]))

    if pm_mode == "grid":
        max_adds = int(trial.suggest_int("grid_max_adds", 1, 8))
        spacing_pct = float(trial.suggest_float("grid_spacing_pct", 0.05, 3.0))
        size_mult = float(trial.suggest_float("grid_size_multiplier", 1.0, 3.0))
        pm = PositionManagerConfig(
            mode="grid",
            grid=GridConfig(max_adds=max_adds, spacing_pct=spacing_pct, size_multiplier=size_mult),
        )
    elif pm_mode == "martingale":
        mult = float(trial.suggest_float("martingale_multiplier", 1.05, 3.5))
        steps = int(trial.suggest_int("martingale_max_steps", 1, 10))
        pm = PositionManagerConfig(
            mode="martingale",
            martingale=MartingaleConfig(multiplier=mult, max_steps=steps),
        )
    else:
        pm = PositionManagerConfig(mode="none")

    return BacktestConfig(initial_equity=base.initial_equity, costs=costs, risk=risk, common=common, pm=pm)


def build_backtest_config_from_params(*, params: dict, base: OptimizationConfig) -> BacktestConfig:
    return _build_backtest_config_from_params(params=params, base=base)


def _build_backtest_config_from_params(*, params: dict, base: OptimizationConfig) -> BacktestConfig:
    tp_mode = str(params.get("tp_mode", "pct"))
    tp_pct = float(params.get("tp_pct", 1.0))
    tp_rr = float(params.get("tp_rr", 2.0))
    tp_mgmt = str(params.get("tp_mgmt", "full"))
    tp1_close_frac = float(params.get("tp1_close_frac", 0.5))
    tp_trail_pct = float(params.get("tp_trail_pct", 0.5))

    sl_type = str(params.get("sl_type", "pct"))
    sl_pct = float(params.get("sl_pct", 1.0))
    sl_atr_period = int(params.get("sl_atr_period", 14))
    sl_atr_mult = float(params.get("sl_atr_mult", 2.0))
    sl_trailing = bool(params.get("sl_trailing", False))
    exit_on_flat = bool(params.get("exit_on_flat", False))

    costs = ExecutionCosts(fee_bps=base.fee_bps, slippage_bps=base.slippage_bps)
    risk = RiskConfig(risk_pct=base.risk_pct, max_position_notional_pct_equity=base.max_position_notional_pct_equity)
    common = CommonTradeParams(
        tp_mode=tp_mode,
        tp_pct=tp_pct,
        tp_rr=tp_rr,
        tp_mgmt=tp_mgmt,
        tp1_close_frac=tp1_close_frac,
        tp_trail_pct=tp_trail_pct,
        sl_type=sl_type,
        sl_pct=sl_pct,
        sl_atr_period=sl_atr_period,
        sl_atr_mult=sl_atr_mult,
        sl_trailing=sl_trailing,
        exit_on_flat=exit_on_flat,
    )

    pm_mode = str(params.get("pm_mode", base.pm_mode))
    if pm_mode == "auto":
        pm_mode = "none"

    if pm_mode == "grid":
        pm = PositionManagerConfig(
            mode="grid",
            grid=GridConfig(
                max_adds=int(params.get("grid_max_adds", 1)),
                spacing_pct=float(params.get("grid_spacing_pct", 0.25)),
                size_multiplier=float(params.get("grid_size_multiplier", 1.0)),
            ),
        )
    elif pm_mode == "martingale":
        pm = PositionManagerConfig(
            mode="martingale",
            martingale=MartingaleConfig(
                multiplier=float(params.get("martingale_multiplier", 1.25)),
                max_steps=int(params.get("martingale_max_steps", 3)),
            ),
        )
    else:
        pm = PositionManagerConfig(mode="none")

    return BacktestConfig(initial_equity=base.initial_equity, costs=costs, risk=risk, common=common, pm=pm)


def run_optimization(
    *,
    df: pd.DataFrame,
    config: OptimizationConfig,
    progress_cb: Callable[[dict], None] | None = None,
    stop_event: threading.Event | None = None,
    stop_flag_path: str | None = None,
) -> OptimizationReport:
    df_train, df_test = _train_test_split(df, train_frac=float(getattr(config, "train_frac", 0.75)))

    min_trades_train = int(config.min_trades_train)
    min_trades_test = int(config.min_trades_test)

    strategies = builtin_strategies()
    if config.strategies is not None:
        allowed = set([s.strip() for s in config.strategies if s and s.strip()])
        if allowed:
            strategies = [s for s in strategies if s.name in allowed]

    candidates_rows: list[dict] = []
    strategies_skipped: dict[str, str] = {}

    global_t0 = time.time()

    stop_flag = Path(stop_flag_path) if stop_flag_path else None

    for strat in strategies:
        if stop_flag is not None and stop_flag.exists():
            break
        if stop_event is not None and stop_event.is_set():
            break

        ctx = StrategyContext(timeframe=config.timeframe)

        if progress_cb is not None:
            progress_cb(
                {
                    "event": "strategy_start",
                    "strategy": strat.name,
                    "global_elapsed": time.time() - global_t0,
                }
            )

        remaining = None
        if config.time_budget_seconds:
            remaining = float(config.time_budget_seconds) - float(time.time() - global_t0)
            if remaining <= 0:
                break

        sampler = _make_sampler(seed=42)
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=sampler,
            study_name=f"{strat.name}",
        )

        t0 = time.time()

        best_eval_cache: dict[int, dict] = {}
        last_printed_best: int | None = None

        def on_trial_complete(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
            if config.time_budget_seconds and (time.time() - global_t0) >= float(config.time_budget_seconds):
                st.stop()

            if stop_flag is not None and stop_flag.exists():
                st.stop()
            if stop_event is not None and stop_event.is_set():
                st.stop()

            best = None
            if st.best_trials:
                best = sorted(list(st.best_trials), key=lambda t: (t.values[1], -t.values[0]))[0]

            if progress_cb is not None:
                payload = {
                    "event": "trial_complete",
                    "strategy": strat.name,
                    "trial": tr.number,
                    "trials_completed": len(st.trials),
                    "global_elapsed": time.time() - global_t0,
                    "strategy_elapsed": time.time() - t0,
                }
                if best is not None and best.values is not None:
                    payload.update(
                        {
                            "best_trial": best.number,
                            "best_return_train": float(best.values[0]),
                            "best_dd_train": float(best.values[1]),
                        }
                    )

                    if int(config.n_jobs) == 1:
                        if best.number not in best_eval_cache:
                            bt_cfg = _build_backtest_config_from_params(params=best.params, base=config)
                            sig_all = strat.compute_signal(df, best.params, ctx)
                            sig_train = sig_all.iloc[: len(df_train)].reset_index(drop=True)
                            sig_test = sig_all.iloc[len(df_train) :].reset_index(drop=True)

                            res_train = run_backtest(df=df_train, signal=sig_train, config=bt_cfg)
                            res_test = run_backtest(df=df_test, signal=sig_test, config=bt_cfg)

                            sharpe_train = _sharpe_from_equity(np.asarray(res_train.equity_curve, dtype="float64"))
                            sharpe_test = _sharpe_from_equity(np.asarray(res_test.equity_curve, dtype="float64"))

                            _, ov_train = summarize_positions(res_train.trades)
                            _, ov_test = summarize_positions(res_test.trades)

                            def metric_value(metric: str) -> float:
                                m = str(metric)
                                if m == "return_test_pct":
                                    return float(res_test.net_return_pct)
                                if m == "return_train_pct":
                                    return float(res_train.net_return_pct)
                                if m == "dd_test_pct":
                                    return float(res_test.max_drawdown_pct)
                                if m == "dd_train_pct":
                                    return float(res_train.max_drawdown_pct)
                                if m == "sharpe_test":
                                    return float(sharpe_test)
                                if m == "sharpe_train":
                                    return float(sharpe_train)
                                if m == "median_pnl_per_position_test":
                                    return float(ov_test.get("median_pnl_per_position", 0.0))
                                if m == "median_pnl_per_position_train":
                                    return float(ov_train.get("median_pnl_per_position", 0.0))
                                if m == "avg_pnl_per_position_test":
                                    return float(ov_test.get("avg_pnl_per_position", 0.0))
                                if m == "avg_pnl_per_position_train":
                                    return float(ov_train.get("avg_pnl_per_position", 0.0))
                                if m == "sharpe_pnl_per_position_test":
                                    return float(ov_test.get("sharpe_pnl_per_position", 0.0))
                                if m == "sharpe_pnl_per_position_train":
                                    return float(ov_train.get("sharpe_pnl_per_position", 0.0))
                                if m == "median_pnl_per_position_test_pct":
                                    v = float(ov_test.get("median_pnl_per_position", 0.0))
                                    return (v / max(float(config.initial_equity), 1e-12)) * 100.0
                                if m == "median_pnl_per_position_train_pct":
                                    v = float(ov_train.get("median_pnl_per_position", 0.0))
                                    return (v / max(float(config.initial_equity), 1e-12)) * 100.0
                                if m == "avg_pnl_per_position_test_pct":
                                    v = float(ov_test.get("avg_pnl_per_position", 0.0))
                                    return (v / max(float(config.initial_equity), 1e-12)) * 100.0
                                if m == "avg_pnl_per_position_train_pct":
                                    v = float(ov_train.get("avg_pnl_per_position", 0.0))
                                    return (v / max(float(config.initial_equity), 1e-12)) * 100.0
                                return 0.0

                            best_eval_cache[best.number] = {
                                "best_return_test": float(res_test.net_return_pct),
                                "best_dd_test": float(res_test.max_drawdown_pct),
                                "best_trades_train": _count_positions(res_train.trades),
                                "best_trades_test": _count_positions(res_test.trades),
                                "best_ranking_metric_test": metric_value(str(getattr(config, "ranking_metric", "median_pnl_per_position_test"))),
                            }

                        payload.update(best_eval_cache[best.number])
                progress_cb(payload)

            nonlocal last_printed_best
            if best is not None and best.values is not None and best.number != last_printed_best:
                last_printed_best = int(best.number)
                msg = (
                    f"[{strat.name}] new best trial={int(best.number)} "
                    f"train_return={float(best.values[0]):.6f} train_dd={float(best.values[1]):.6f}"
                )
                if int(config.n_jobs) == 1 and best.number in best_eval_cache:
                    ev = best_eval_cache[best.number]
                    msg += (
                        f" test_return={float(ev.get('best_return_test', 0.0)):.6f}"
                        f" test_dd={float(ev.get('best_dd_test', 0.0)):.6f}"
                        f" {str(getattr(config, 'ranking_metric', 'median_pnl_per_position_test'))}={float(ev.get('best_ranking_metric_test', 0.0)):.6f}"
                    )
                print(msg, flush=True)

        def objective(trial: optuna.Trial):
            if config.time_budget_seconds and (time.time() - global_t0) >= float(config.time_budget_seconds):
                study.stop()
                raise optuna.TrialPruned()

            if stop_flag is not None and stop_flag.exists():
                study.stop()
                raise optuna.TrialPruned()
            if stop_event is not None and stop_event.is_set():
                study.stop()
                raise optuna.TrialPruned()

            _ = strat.sample_params(trial)
            bt_cfg = _build_backtest_config(trial=trial, base=config)

            sig_all = strat.compute_signal(df, trial.params, ctx)
            sig_train = sig_all.iloc[: len(df_train)].reset_index(drop=True)
            res = run_backtest(df=df_train, signal=sig_train, config=bt_cfg)

            return (res.net_return_pct, res.max_drawdown_pct)

        try:
            study.optimize(
                objective,
                n_trials=int(config.max_trials),
                timeout=float(remaining) if remaining is not None else None,
                n_jobs=int(config.n_jobs),
                gc_after_trial=True,
                callbacks=[on_trial_complete],
            )
        except KeyboardInterrupt as e:
            strategies_skipped[strat.name] = f"optimization_error:KeyboardInterrupt:{e}"
            if progress_cb is not None:
                progress_cb(
                    {
                        "event": "optimization_error",
                        "strategy": strat.name,
                        "error": f"KeyboardInterrupt: {e}",
                        "global_elapsed": time.time() - global_t0,
                        "strategy_elapsed": time.time() - t0,
                    }
                )
        except Exception as e:
            strategies_skipped[strat.name] = f"optimization_error:{type(e).__name__}:{e}"
            if progress_cb is not None:
                progress_cb(
                    {
                        "event": "optimization_error",
                        "strategy": strat.name,
                        "error": f"{type(e).__name__}: {e}",
                        "global_elapsed": time.time() - global_t0,
                        "strategy_elapsed": time.time() - t0,
                    }
                )

        if progress_cb is not None:
            progress_cb(
                {
                    "event": "strategy_end",
                    "strategy": strat.name,
                    "trials_completed": len(study.trials),
                    "global_elapsed": time.time() - global_t0,
                    "strategy_elapsed": time.time() - t0,
                }
            )

        trials = _trial_pool(study=study, config=config)
        if not trials:
            strategies_skipped[strat.name] = "no_complete_trials"
            continue

        sig_all_cache: dict[int, pd.Series] = {}
        for tr in trials:
            bt_cfg = _build_backtest_config_from_params(params=tr.params, base=config)

            if tr.number not in sig_all_cache:
                sig_all_cache[tr.number] = strat.compute_signal(df, tr.params, ctx)

            sig_all = sig_all_cache[tr.number]
            sig_train = sig_all.iloc[: len(df_train)].reset_index(drop=True)
            sig_test = sig_all.iloc[len(df_train) :].reset_index(drop=True)

            res_train = run_backtest(df=df_train, signal=sig_train, config=bt_cfg)
            res_test = run_backtest(df=df_test, signal=sig_test, config=bt_cfg)

            sharpe_train = _sharpe_from_equity(res_train.equity_curve)
            sharpe_test = _sharpe_from_equity(res_test.equity_curve)

            _, overview_train = summarize_positions(res_train.trades)
            _, overview_test = summarize_positions(res_test.trades)

            trades_train = _count_positions(res_train.trades)
            trades_test = _count_positions(res_test.trades)
            eligible = (trades_train >= min_trades_train) and (trades_test >= min_trades_test)

            median_pnl_train = float(overview_train.get("median_pnl_per_position", 0.0))
            median_pnl_test = float(overview_test.get("median_pnl_per_position", 0.0))
            avg_pnl_train = float(overview_train.get("avg_pnl_per_position", 0.0))
            avg_pnl_test = float(overview_test.get("avg_pnl_per_position", 0.0))
            sharpe_pnl_train = float(overview_train.get("sharpe_pnl_per_position", 0.0))
            sharpe_pnl_test = float(overview_test.get("sharpe_pnl_per_position", 0.0))

            row = {
                "strategy": strat.name,
                "return_train_pct": res_train.net_return_pct,
                "dd_train_pct": res_train.max_drawdown_pct,
                "return_test_pct": res_test.net_return_pct,
                "dd_test_pct": res_test.max_drawdown_pct,
                "sharpe_train": sharpe_train,
                "sharpe_test": sharpe_test,
                "median_pnl_per_position_train": median_pnl_train,
                "median_pnl_per_position_test": median_pnl_test,
                "avg_pnl_per_position_train": avg_pnl_train,
                "avg_pnl_per_position_test": avg_pnl_test,
                "sharpe_pnl_per_position_train": sharpe_pnl_train,
                "sharpe_pnl_per_position_test": sharpe_pnl_test,
                "median_pnl_per_position_train_pct": (median_pnl_train / max(float(config.initial_equity), 1e-12)) * 100.0,
                "median_pnl_per_position_test_pct": (median_pnl_test / max(float(config.initial_equity), 1e-12)) * 100.0,
                "trades_train": trades_train,
                "trades_test": trades_test,
                "eligible": eligible,
                "trial": tr.number,
                "seconds": time.time() - t0,
            }
            for k, v in tr.params.items():
                row[k] = v

            candidates_rows.append(row)

    candidates = pd.DataFrame(candidates_rows)
    if not candidates.empty:
        candidates = _add_rank_columns(df=candidates, config=config)

    global_lb, champion_global, champions_by_strategy = _build_global_outputs(candidates=candidates, config=config)
    leaderboard = pd.DataFrame(list(champions_by_strategy.values())) if champions_by_strategy else pd.DataFrame()

    return OptimizationReport(
        leaderboard=leaderboard,
        champion=champion_global,
        global_leaderboard=global_lb,
        candidates=candidates,
        champions_by_strategy=champions_by_strategy,
        champion_global=champion_global,
        strategies_skipped=strategies_skipped,
        meta={
            "ranking_metric": str(getattr(config, "ranking_metric", "median_pnl_per_position_test")),
            "candidate_pool": str(getattr(config, "candidate_pool", "pareto")),
            "global_top_k": int(getattr(config, "global_top_k", 50) or 50),
        },
    )
