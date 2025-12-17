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
    BrokerConfig,
    CommonTradeParams,
    ExecutionCosts,
    ExecutionConstraints,
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
    risk_mode: str = "risk"
    risk_pct: float = 0.01
    fixed_notional_pct_equity: float = 0.0
    max_position_notional_pct_equity: float = 100.0
    max_leverage: float | None = None
    min_qty: float = 0.0
    qty_step: float = 0.0
    min_notional: float = 0.0
    broker_profile: str = "perps"  # spot|perps|cfd
    perps_maintenance_margin_rate: float = 0.01
    cfd_initial_margin_rate: float = 0.01
    cfd_stopout_margin_level: float = 0.5
    pareto_candidates_max: int = 50
    candidate_pool: str = "pareto"  # pareto|complete
    global_top_k: int = 50
    ranking_metric: str = "median_pnl_per_position_test"
    train_frac: float = 0.75
    optuna_objective_metric: str = "return_train_pct"
    require_positive_train_metric_for_test: bool = True
    tp_mode_policy: str = "auto"
    tp_rr_fixed: float = 2.0


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


def _position_pnl_sequence(trades: pd.DataFrame) -> list[float]:
    if trades is None or trades.empty:
        return []

    if "position_id" in trades.columns:
        pos_key = "position_id"
    elif "entry_ts" in trades.columns:
        pos_key = "entry_ts"
    else:
        return []

    if "pnl" in trades.columns:
        pnl_col = "pnl"
    elif "pnl_total" in trades.columns:
        pnl_col = "pnl_total"
    else:
        return []

    df = trades[[pos_key, pnl_col]].copy()
    df = df.rename(columns={pos_key: "position_id", pnl_col: "pnl"})
    if "exit_ts" in trades.columns:
        df["_sort_ts"] = trades["exit_ts"]
    elif "entry_ts" in trades.columns:
        df["_sort_ts"] = trades["entry_ts"]
    else:
        df["_sort_ts"] = np.arange(len(df), dtype=np.int64)

    g = df.groupby("position_id", dropna=True)
    pos = g.agg({"pnl": "sum", "_sort_ts": "max"}).reset_index(drop=True)
    pos = pos.sort_values("_sort_ts", ascending=True)
    return [float(x) for x in pos["pnl"].to_list()]


def _martingale_usage_metrics(*, trades: pd.DataFrame, multiplier: float, max_steps: int) -> dict:
    pnls = _position_pnl_sequence(trades)
    if not pnls:
        return {
            "martingale_max_loss_streak": 0,
            "martingale_max_step_used": 0,
            "martingale_max_multiplier_used": 1.0,
        }

    step = 0
    max_step_used = 0
    loss_streak = 0
    max_loss_streak = 0

    for pnl in pnls:
        max_step_used = max(int(max_step_used), int(step))
        if float(pnl) < 0:
            loss_streak += 1
            max_loss_streak = max(int(max_loss_streak), int(loss_streak))
            step = min(int(step) + 1, int(max_steps))
        else:
            loss_streak = 0
            step = 0

    try:
        max_mult = float(multiplier) ** int(max_step_used)
    except Exception:
        max_mult = 1.0

    return {
        "martingale_max_loss_streak": int(max_loss_streak),
        "martingale_max_step_used": int(max_step_used),
        "martingale_max_multiplier_used": float(max_mult),
    }


def _grid_usage_metrics(*, trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"grid_max_adds_used": 0, "grid_max_multiplier_used": 1.0}
    if "position_id" not in trades.columns:
        return {"grid_max_adds_used": 0, "grid_max_multiplier_used": 1.0}

    max_adds = 0
    max_mult = 1.0

    if "grid_adds_done" in trades.columns:
        try:
            max_adds = int(pd.to_numeric(trades["grid_adds_done"], errors="coerce").fillna(0).max())
        except Exception:
            max_adds = 0

    if "peak_qty_mult" in trades.columns:
        try:
            max_mult = float(pd.to_numeric(trades["peak_qty_mult"], errors="coerce").fillna(1.0).max())
        except Exception:
            max_mult = 1.0

    return {"grid_max_adds_used": int(max_adds), "grid_max_multiplier_used": float(max_mult)}


def _connect_args_for_storage_url(storage_url: str) -> dict:
    u = str(storage_url or "").strip().lower()
    if u.startswith("sqlite"):
        return {"timeout": 60}
    if u.startswith("postgres"):
        return {"connect_timeout": 60}
    return {}


def build_report_from_storage(
    *,
    df: pd.DataFrame,
    config: OptimizationConfig,
    storage_url: str,
    study_name_prefix: str | None = None,
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

    pm_usage_cols = [
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
    ]

    connect_args = _connect_args_for_storage_url(str(storage_url))
    storage = optuna.storages.RDBStorage(
        str(storage_url),
        engine_kwargs={"connect_args": connect_args},
    )

    for strat in strategies:
        try:
            study_name = str(strat.name)
            if study_name_prefix:
                study_name = f"{str(study_name_prefix)}.{study_name}"
            study = optuna.load_study(study_name=study_name, storage=storage)
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
            sharpe_train = _sharpe_from_equity(res_train.equity_curve)
            _, overview_train = summarize_positions(res_train.trades)
            trades_train = _count_positions(res_train.trades)

            median_pnl_train = float(overview_train.get("median_pnl_per_position", 0.0))
            avg_pnl_train = float(overview_train.get("avg_pnl_per_position", 0.0))
            sharpe_pnl_train = float(overview_train.get("sharpe_pnl_per_position", 0.0))

            metric_test = str(getattr(config, "ranking_metric", "median_pnl_per_position_test"))
            metric_train_name = _metric_train_name(metric_test)

            train_metric_value = 0.0
            if metric_train_name == "return_train_pct":
                train_metric_value = float(res_train.net_return_pct)
            elif metric_train_name == "sharpe_train":
                train_metric_value = float(sharpe_train)
            elif metric_train_name == "median_pnl_per_position_train":
                train_metric_value = float(median_pnl_train)
            elif metric_train_name == "avg_pnl_per_position_train":
                train_metric_value = float(avg_pnl_train)
            elif metric_train_name == "sharpe_pnl_per_position_train":
                train_metric_value = float(sharpe_pnl_train)
            elif metric_train_name == "median_pnl_per_position_train_pct":
                train_metric_value = (float(median_pnl_train) / max(float(config.initial_equity), 1e-12)) * 100.0
            else:
                train_metric_value = float(res_train.net_return_pct)

            eligible_train = trades_train >= min_trades_train
            require_positive = bool(getattr(config, "require_positive_train_metric_for_test", True))
            do_test_eval = (not require_positive) or (float(train_metric_value) > 0.0)

            if do_test_eval:
                res_test = run_backtest(df=df_test, signal=sig_test, config=bt_cfg)
                sharpe_test = _sharpe_from_equity(res_test.equity_curve)
                _, overview_test = summarize_positions(res_test.trades)
                trades_test = _count_positions(res_test.trades)

                median_pnl_test = float(overview_test.get("median_pnl_per_position", 0.0))
                avg_pnl_test = float(overview_test.get("avg_pnl_per_position", 0.0))
                sharpe_pnl_test = float(overview_test.get("sharpe_pnl_per_position", 0.0))
                dd_test_pct = float(res_test.max_drawdown_pct)
                ret_test_pct = float(res_test.net_return_pct)

                dd_test_intrabar_pct = float(res_test.max_drawdown_intrabar_pct)
                liquidated_test = bool(res_test.liquidated)
                peak_notional_pct_equity_test = float(res_test.peak_notional_pct_equity)
                peak_qty_mult_test = float(res_test.peak_qty_mult)
                cap_hit_rate_test = float(res_test.cap_hit_rate)
                exec_reject_rate_test = float(getattr(res_test, "exec_reject_rate", 0.0))
                exec_round_rate_test = float(getattr(res_test, "exec_round_rate", 0.0))
            else:
                sharpe_test = float("nan")
                trades_test = 0
                median_pnl_test = float("nan")
                avg_pnl_test = float("nan")
                sharpe_pnl_test = float("nan")
                dd_test_pct = float("nan")
                ret_test_pct = float("nan")

                dd_test_intrabar_pct = float("nan")
                liquidated_test = False
                peak_notional_pct_equity_test = float("nan")
                peak_qty_mult_test = float("nan")
                cap_hit_rate_test = float("nan")
                exec_reject_rate_test = float("nan")
                exec_round_rate_test = float("nan")

            eligible = bool(eligible_train)

            pm_mode_used = str(getattr(bt_cfg.pm, "mode", "none"))
            if pm_mode_used == "martingale" and bt_cfg.pm.martingale is not None:
                mm_train = _martingale_usage_metrics(
                    trades=res_train.trades,
                    multiplier=float(bt_cfg.pm.martingale.multiplier),
                    max_steps=int(bt_cfg.pm.martingale.max_steps),
                )
                if do_test_eval:
                    mm_test = _martingale_usage_metrics(
                        trades=res_test.trades,
                        multiplier=float(bt_cfg.pm.martingale.multiplier),
                        max_steps=int(bt_cfg.pm.martingale.max_steps),
                    )
                else:
                    mm_test = {"martingale_max_loss_streak": 0, "martingale_max_step_used": 0, "martingale_max_multiplier_used": 1.0}
            else:
                mm_train = {"martingale_max_loss_streak": 0, "martingale_max_step_used": 0, "martingale_max_multiplier_used": 1.0}
                mm_test = {"martingale_max_loss_streak": 0, "martingale_max_step_used": 0, "martingale_max_multiplier_used": 1.0}

            if pm_mode_used == "grid":
                gm_train = _grid_usage_metrics(trades=res_train.trades)
                if do_test_eval:
                    gm_test = _grid_usage_metrics(trades=res_test.trades)
                else:
                    gm_test = {"grid_max_adds_used": 0, "grid_max_multiplier_used": 1.0}
            else:
                gm_train = {"grid_max_adds_used": 0, "grid_max_multiplier_used": 1.0}
                gm_test = {"grid_max_adds_used": 0, "grid_max_multiplier_used": 1.0}

            row = {
                "strategy": strat.name,
                "return_train_pct": res_train.net_return_pct,
                "dd_train_pct": res_train.max_drawdown_pct,
                "dd_train_intrabar_pct": float(res_train.max_drawdown_intrabar_pct),
                "exec_reject_rate_train": float(getattr(res_train, "exec_reject_rate", 0.0)),
                "exec_round_rate_train": float(getattr(res_train, "exec_round_rate", 0.0)),
                "return_test_pct": ret_test_pct,
                "dd_test_pct": dd_test_pct,
                "dd_test_intrabar_pct": float(dd_test_intrabar_pct),
                "exec_reject_rate_test": float(exec_reject_rate_test),
                "exec_round_rate_test": float(exec_round_rate_test),
                "tested": bool(do_test_eval) and (int(trades_test) >= int(min_trades_test)),
                "liquidated_train": bool(res_train.liquidated),
                "liquidated_test": bool(liquidated_test),
                "peak_notional_pct_equity_train": float(res_train.peak_notional_pct_equity),
                "peak_notional_pct_equity_test": float(peak_notional_pct_equity_test),
                "peak_qty_mult_train": float(res_train.peak_qty_mult),
                "peak_qty_mult_test": float(peak_qty_mult_test),
                "cap_hit_rate_train": float(res_train.cap_hit_rate),
                "cap_hit_rate_test": float(cap_hit_rate_test),
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
                "martingale_max_loss_streak_train": int(mm_train.get("martingale_max_loss_streak", 0)),
                "martingale_max_loss_streak_test": int(mm_test.get("martingale_max_loss_streak", 0)),
                "martingale_max_step_used_train": int(mm_train.get("martingale_max_step_used", 0)),
                "martingale_max_step_used_test": int(mm_test.get("martingale_max_step_used", 0)),
                "martingale_max_multiplier_used_train": float(mm_train.get("martingale_max_multiplier_used", 1.0)),
                "martingale_max_multiplier_used_test": float(mm_test.get("martingale_max_multiplier_used", 1.0)),
                "grid_max_adds_used_train": int(gm_train.get("grid_max_adds_used", 0)),
                "grid_max_adds_used_test": int(gm_test.get("grid_max_adds_used", 0)),
                "grid_max_multiplier_used_train": float(gm_train.get("grid_max_multiplier_used", 1.0)),
                "grid_max_multiplier_used_test": float(gm_test.get("grid_max_multiplier_used", 1.0)),
            }
            for k, v in tr.params.items():
                row[k] = v

            candidates_rows.append(row)

    candidates = pd.DataFrame(candidates_rows)
    if not candidates.empty:
        candidates = _add_rank_columns(df=candidates, config=config)

    candidates_for_global = candidates.drop(columns=pm_usage_cols, errors="ignore") if candidates is not None else candidates

    global_lb, champion_global, champions_by_strategy = _build_global_outputs(candidates=candidates_for_global, config=config)
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
    tp_mode_policy = str(getattr(base, "tp_mode_policy", "auto") or "auto").strip().lower()
    tp_pct = 1.0
    tp_rr = 2.0
    if tp_mode_policy in {"rr_fixed", "force_rr", "rr"}:
        rr_fixed = float(getattr(base, "tp_rr_fixed", 2.0) or 2.0)
        if (not np.isfinite(float(rr_fixed))) or float(rr_fixed) <= 0:
            rr_fixed = 2.0
        tp_mode = str(trial.suggest_categorical("tp_mode", ["rr"]))
        tp_rr = float(trial.suggest_categorical("tp_rr", [float(rr_fixed)]))
    else:
        tp_mode = str(trial.suggest_categorical("tp_mode", ["pct", "rr"]))
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
    max_notional_pct_eq = float(base.max_position_notional_pct_equity)
    if getattr(base, "max_leverage", None) is not None:
        try:
            lev = float(getattr(base, "max_leverage") or 0.0)
            if lev > 0:
                max_notional_pct_eq = lev * 100.0
        except Exception:
            pass
    risk = RiskConfig(
        mode=str(getattr(base, "risk_mode", "risk") or "risk"),
        risk_pct=float(getattr(base, "risk_pct", 0.01) or 0.0),
        fixed_notional_pct_equity=float(getattr(base, "fixed_notional_pct_equity", 0.0) or 0.0),
        max_position_notional_pct_equity=max_notional_pct_eq,
    )
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

    prof_raw = str(getattr(base, "broker_profile", "perps") or "perps").strip().lower()
    if prof_raw in ("spot",):
        prof = "spot"
    elif ("icmarkets" in prof_raw) or ("mt5" in prof_raw) or (prof_raw == "cfd"):
        prof = "cfd"
    else:
        prof = "perps"

    broker = BrokerConfig(
        profile=str(prof),
        perps_maintenance_margin_rate=float(getattr(base, "perps_maintenance_margin_rate", 0.01) or 0.01),
        cfd_initial_margin_rate=float(getattr(base, "cfd_initial_margin_rate", 0.01) or 0.01),
        cfd_stopout_margin_level=float(getattr(base, "cfd_stopout_margin_level", 0.5) or 0.5),
    )

    execution = ExecutionConstraints(
        min_qty=float(getattr(base, "min_qty", 0.0) or 0.0),
        qty_step=float(getattr(base, "qty_step", 0.0) or 0.0),
        min_notional=float(getattr(base, "min_notional", 0.0) or 0.0),
    )

    return BacktestConfig(
        initial_equity=base.initial_equity,
        costs=costs,
        risk=risk,
        common=common,
        pm=pm,
        broker=broker,
        execution=execution,
    )


def build_backtest_config_from_params(*, params: dict, base: OptimizationConfig) -> BacktestConfig:
    return _build_backtest_config_from_params(params=params, base=base)


def _build_backtest_config_from_params(*, params: dict, base: OptimizationConfig) -> BacktestConfig:
    tp_mode = str(params.get("tp_mode", "pct"))
    tp_pct = float(params.get("tp_pct", 1.0))
    tp_rr = float(params.get("tp_rr", 2.0))
    tp_mgmt = str(params.get("tp_mgmt", "full"))
    tp1_close_frac = float(params.get("tp1_close_frac", 0.5))
    tp_trail_pct = float(params.get("tp_trail_pct", 0.5))

    tp_mode_policy = str(getattr(base, "tp_mode_policy", "auto") or "auto").strip().lower()
    if tp_mode_policy in {"rr_fixed", "force_rr", "rr"}:
        rr_fixed = float(getattr(base, "tp_rr_fixed", 2.0) or 2.0)
        if (not np.isfinite(float(rr_fixed))) or float(rr_fixed) <= 0:
            rr_fixed = 2.0
        tp_mode = "rr"
        tp_rr = float(rr_fixed)

    sl_type = str(params.get("sl_type", "pct"))
    sl_pct = float(params.get("sl_pct", 1.0))
    sl_atr_period = int(params.get("sl_atr_period", 14))
    sl_atr_mult = float(params.get("sl_atr_mult", 2.0))
    sl_trailing = bool(params.get("sl_trailing", False))
    exit_on_flat = bool(params.get("exit_on_flat", False))

    costs = ExecutionCosts(fee_bps=base.fee_bps, slippage_bps=base.slippage_bps)
    max_notional_pct_eq = float(base.max_position_notional_pct_equity)
    if getattr(base, "max_leverage", None) is not None:
        try:
            lev = float(getattr(base, "max_leverage") or 0.0)
            if lev > 0:
                max_notional_pct_eq = lev * 100.0
        except Exception:
            pass
    risk = RiskConfig(
        mode=str(getattr(base, "risk_mode", "risk") or "risk"),
        risk_pct=float(getattr(base, "risk_pct", 0.01) or 0.0),
        fixed_notional_pct_equity=float(getattr(base, "fixed_notional_pct_equity", 0.0) or 0.0),
        max_position_notional_pct_equity=max_notional_pct_eq,
    )
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

    prof_raw = str(getattr(base, "broker_profile", "perps") or "perps").strip().lower()
    if prof_raw in ("spot",):
        prof = "spot"
    elif ("icmarkets" in prof_raw) or ("mt5" in prof_raw) or (prof_raw == "cfd"):
        prof = "cfd"
    else:
        prof = "perps"

    broker = BrokerConfig(
        profile=str(prof),
        perps_maintenance_margin_rate=float(getattr(base, "perps_maintenance_margin_rate", 0.01) or 0.01),
        cfd_initial_margin_rate=float(getattr(base, "cfd_initial_margin_rate", 0.01) or 0.01),
        cfd_stopout_margin_level=float(getattr(base, "cfd_stopout_margin_level", 0.5) or 0.5),
    )

    execution = ExecutionConstraints(
        min_qty=float(getattr(base, "min_qty", 0.0) or 0.0),
        qty_step=float(getattr(base, "qty_step", 0.0) or 0.0),
        min_notional=float(getattr(base, "min_notional", 0.0) or 0.0),
    )

    return BacktestConfig(
        initial_equity=base.initial_equity,
        costs=costs,
        risk=risk,
        common=common,
        pm=pm,
        broker=broker,
        execution=execution,
    )


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
                obj_name = str(getattr(config, "optuna_objective_metric", "return_train_pct"))
                msg = (
                    f"[{strat.name}] new best trial={int(best.number)} "
                    f"train_obj({obj_name})={float(best.values[0]):.6f} train_dd={float(best.values[1]):.6f}"
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

            metric = str(getattr(config, "optuna_objective_metric", "return_train_pct"))
            v = 0.0
            if metric == "return_train_pct":
                v = float(res.net_return_pct)
            elif metric == "sharpe_train":
                v = float(_sharpe_from_equity(np.asarray(res.equity_curve, dtype="float64")))
            else:
                _, ov = summarize_positions(res.trades)
                if metric == "median_pnl_per_position_train":
                    v = float(ov.get("median_pnl_per_position", 0.0))
                elif metric == "avg_pnl_per_position_train":
                    v = float(ov.get("avg_pnl_per_position", 0.0))
                elif metric == "sharpe_pnl_per_position_train":
                    v = float(ov.get("sharpe_pnl_per_position", 0.0))
                elif metric == "win_rate_train":
                    positions = int(ov.get("positions", 0) or 0)
                    min_trades_train = int(getattr(config, "min_trades_train", 0) or 0)
                    if min_trades_train > 0 and positions < min_trades_train:
                        raise optuna.TrialPruned()
                    v = float(ov.get("win_rate", 0.0))
                elif metric == "median_pnl_per_position_train_pct":
                    v0 = float(ov.get("median_pnl_per_position", 0.0))
                    v = (v0 / max(float(config.initial_equity), 1e-12)) * 100.0
                elif metric == "avg_pnl_per_position_train_pct":
                    v0 = float(ov.get("avg_pnl_per_position", 0.0))
                    v = (v0 / max(float(config.initial_equity), 1e-12)) * 100.0
                else:
                    v = float(res.net_return_pct)

            return (float(v), float(res.max_drawdown_pct))

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
                "exec_reject_rate_train": float(getattr(res_train, "exec_reject_rate", 0.0)),
                "exec_round_rate_train": float(getattr(res_train, "exec_round_rate", 0.0)),
                "return_test_pct": res_test.net_return_pct,
                "dd_test_pct": res_test.max_drawdown_pct,
                "exec_reject_rate_test": float(getattr(res_test, "exec_reject_rate", 0.0)),
                "exec_round_rate_test": float(getattr(res_test, "exec_round_rate", 0.0)),
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
