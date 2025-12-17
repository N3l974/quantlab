from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import optuna
import pandas as pd

from hyperliquid_ohlcv_optimizer.backtest.backtester import run_backtest
from hyperliquid_ohlcv_optimizer.backtest.trade_analysis import summarize_positions
from hyperliquid_ohlcv_optimizer.data.ohlcv_loader import load_ohlcv
from hyperliquid_ohlcv_optimizer.optimize.optuna_runner import (
    OptimizationConfig,
    _build_backtest_config,
    _make_sampler,
    _sharpe_from_equity,
    _train_test_split,
)
from hyperliquid_ohlcv_optimizer.optimize.run_optimize import _connect_args_for_storage_url
from hyperliquid_ohlcv_optimizer.strategies.base import StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.registry import builtin_strategies


def _load_filtered_df(*, data_root: str, source: str, symbol: str, timeframe: str, start_ms: int | None, end_ms: int | None) -> pd.DataFrame:
    df = load_ohlcv(data_root=Path(data_root), source=source, symbol=symbol, timeframe=timeframe)
    if start_ms is not None:
        df = df[df["timestamp_ms"] >= int(start_ms)]
    if end_ms is not None:
        df = df[df["timestamp_ms"] < int(end_ms)]
    return df.reset_index(drop=True)


def main() -> None:
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--storage-url", required=True)
    p.add_argument("--study-name", required=True)
    p.add_argument("--strategy", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--n-trials", type=int, required=True)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop-flag", default=None)
    args = p.parse_args()

    ctx = json.loads(Path(args.context).read_text(encoding="utf-8"))

    base = OptimizationConfig(
        initial_equity=float(ctx["config"]["initial_equity"]),
        fee_bps=float(ctx["config"]["fee_bps"]),
        slippage_bps=float(ctx["config"]["slippage_bps"]),
        dd_threshold_pct=float(ctx["config"]["dd_threshold_pct"]),
        max_trials=int(ctx["config"]["max_trials"]),
        time_budget_seconds=int(ctx["config"]["time_budget_seconds"]) if ctx["config"]["time_budget_seconds"] is not None else None,
        n_jobs=1,
        min_trades_train=int(ctx["config"].get("min_trades_train", 0)),
        min_trades_test=int(ctx["config"].get("min_trades_test", 0)),
        timeframe=str(ctx["config"].get("timeframe", "5m")),
        pm_mode=str(ctx["config"].get("pm_mode", "auto")),
        strategies=[str(args.strategy)],
        risk_pct=float(ctx["config"].get("risk_pct", 0.01)),
        max_position_notional_pct_equity=float(ctx["config"].get("max_position_notional_pct_equity", 100.0)),
        max_leverage=float(ctx["config"]["max_leverage"]) if ctx["config"].get("max_leverage") is not None else None,
        min_qty=float(ctx["config"].get("min_qty", 0.0)),
        qty_step=float(ctx["config"].get("qty_step", 0.0)),
        min_notional=float(ctx["config"].get("min_notional", 0.0)),
        broker_profile=str(ctx["config"].get("broker_profile", "perps")),
        perps_maintenance_margin_rate=float(ctx["config"].get("perps_maintenance_margin_rate", 0.01)),
        cfd_initial_margin_rate=float(ctx["config"].get("cfd_initial_margin_rate", 0.01)),
        cfd_stopout_margin_level=float(ctx["config"].get("cfd_stopout_margin_level", 0.5)),
        pareto_candidates_max=int(ctx["config"].get("pareto_candidates_max", 50)),
        candidate_pool=str(ctx["config"].get("candidate_pool", "pareto")),
        global_top_k=int(ctx["config"].get("global_top_k", 50)),
        ranking_metric=str(ctx["config"].get("ranking_metric", "median_pnl_per_position_test")),
        train_frac=float(ctx["config"].get("train_frac", 0.75)),
        optuna_objective_metric=str(ctx["config"].get("optuna_objective_metric", "return_train_pct")),
    )

    df = _load_filtered_df(
        data_root=str(ctx["data"]["data_root"]),
        source=str(ctx["data"]["source"]),
        symbol=str(ctx["data"]["symbol"]),
        timeframe=str(ctx["data"]["timeframe"]),
        start_ms=ctx["data"].get("start_ms"),
        end_ms=ctx["data"].get("end_ms"),
    )

    df_train, _ = _train_test_split(df, train_frac=float(getattr(base, "train_frac", 0.75)))

    strats = builtin_strategies()
    strat = next((s for s in strats if s.name == str(args.strategy)), None)
    if strat is None:
        raise SystemExit(f"Unknown strategy: {args.strategy}")

    sampler = _make_sampler(seed=int(args.seed))
    storage = optuna.storages.RDBStorage(str(args.storage_url), engine_kwargs={"connect_args": _connect_args_for_storage_url(str(args.storage_url))})
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=sampler,
        study_name=str(args.study_name),
        storage=storage,
        load_if_exists=True,
    )

    st_ctx = StrategyContext(timeframe=base.timeframe)
    stop_flag = Path(args.stop_flag) if args.stop_flag else None

    def _complete_trials_count(st: optuna.Study) -> int:
        try:
            return len(
                st.get_trials(
                    deepcopy=False,
                    states=(optuna.trial.TrialState.COMPLETE,),
                )
            )
        except Exception:
            return len([t for t in st.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def on_trial_complete(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        if stop_flag is not None and stop_flag.exists():
            st.stop()
        if int(base.max_trials) > 0 and _complete_trials_count(st) >= int(base.max_trials):
            st.stop()

    def objective(trial: optuna.Trial):
        if stop_flag is not None and stop_flag.exists():
            raise optuna.TrialPruned()

        if int(base.max_trials) > 0 and _complete_trials_count(study) >= int(base.max_trials):
            raise optuna.TrialPruned()

        _ = strat.sample_params(trial)
        bt_cfg = _build_backtest_config(trial=trial, base=base)

        sig_all = strat.compute_signal(df, trial.params, st_ctx)
        sig_train = sig_all.iloc[: len(df_train)].reset_index(drop=True)
        res = run_backtest(df=df_train, signal=sig_train, config=bt_cfg)

        metric = str(getattr(base, "optuna_objective_metric", "return_train_pct"))
        v = 0.0
        if metric == "return_train_pct":
            v = float(res.net_return_pct)
        elif metric == "sharpe_train":
            try:
                import numpy as np

                v = float(_sharpe_from_equity(np.asarray(res.equity_curve, dtype="float64")))
            except Exception:
                v = 0.0
        else:
            _, ov = summarize_positions(res.trades)
            if metric == "median_pnl_per_position_train":
                v = float(ov.get("median_pnl_per_position", 0.0))
            elif metric == "avg_pnl_per_position_train":
                v = float(ov.get("avg_pnl_per_position", 0.0))
            elif metric == "sharpe_pnl_per_position_train":
                v = float(ov.get("sharpe_pnl_per_position", 0.0))
            elif metric == "median_pnl_per_position_train_pct":
                v0 = float(ov.get("median_pnl_per_position", 0.0))
                v = (v0 / max(float(base.initial_equity), 1e-12)) * 100.0
            elif metric == "avg_pnl_per_position_train_pct":
                v0 = float(ov.get("avg_pnl_per_position", 0.0))
                v = (v0 / max(float(base.initial_equity), 1e-12)) * 100.0
            else:
                v = float(res.net_return_pct)

        return (float(v), float(res.max_drawdown_pct))

    study.optimize(
        objective,
        n_trials=int(args.n_trials),
        timeout=float(args.timeout) if args.timeout is not None else None,
        n_jobs=1,
        gc_after_trial=True,
        callbacks=[on_trial_complete],
    )


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        _ = time.time() - t0
