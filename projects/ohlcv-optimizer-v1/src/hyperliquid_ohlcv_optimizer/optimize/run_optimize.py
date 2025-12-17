from __future__ import annotations

import argparse
import json
import os
import subprocess
import sqlite3
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import optuna
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hyperliquid_ohlcv_optimizer.data.ohlcv_loader import load_ohlcv
from hyperliquid_ohlcv_optimizer.backtest.backtester import run_backtest
from hyperliquid_ohlcv_optimizer.backtest.trade_analysis import summarize_positions
from hyperliquid_ohlcv_optimizer.optimize.optuna_runner import (
    OptimizationConfig,
    build_backtest_config_from_params,
    build_report_from_storage,
)
from hyperliquid_ohlcv_optimizer.strategies.base import StrategyContext
from hyperliquid_ohlcv_optimizer.strategies.registry import builtin_strategies


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_run_dir(*, run: str | None, run_dir: str | None, latest: bool) -> Path:
    root = _project_root()
    runs_root = root / "runs"

    if run_dir:
        p = Path(run_dir)
        return p if p.is_absolute() else (runs_root / p)

    if latest:
        if not runs_root.exists():
            raise SystemExit(f"runs folder not found: {runs_root}")
        dirs = [p for p in runs_root.iterdir() if p.is_dir()]
        if not dirs:
            raise SystemExit(f"no runs found under: {runs_root}")
        dirs.sort(key=lambda p: p.name, reverse=True)
        return dirs[0]

    if run:
        return runs_root / run

    raise SystemExit("Provide --run-dir, --run, or --latest")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        tmp.write_text(data, encoding="utf-8")
    except Exception:
        try:
            path.write_text(data, encoding="utf-8")
        except Exception:
            pass
        return

    for i in range(25):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05 * float(i + 1))
        except Exception:
            break

    try:
        path.write_text(data, encoding="utf-8")
    except Exception:
        pass
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


def _load_filtered_df(*, data_root: str, source: str, symbol: str, timeframe: str, start_ms: int | None, end_ms: int | None) -> pd.DataFrame:
    df = load_ohlcv(data_root=Path(data_root), source=source, symbol=symbol, timeframe=timeframe)
    if start_ms is not None:
        df = df[df["timestamp_ms"] >= int(start_ms)]
    if end_ms is not None:
        df = df[df["timestamp_ms"] < int(end_ms)]
    return df.reset_index(drop=True)


def _cfg_from_context(cfg_payload: dict) -> OptimizationConfig:
    return OptimizationConfig(
        initial_equity=float(cfg_payload.get("initial_equity", 10_000.0)),
        fee_bps=float(cfg_payload.get("fee_bps", 4.5)),
        slippage_bps=float(cfg_payload.get("slippage_bps", 1.0)),
        dd_threshold_pct=float(cfg_payload.get("dd_threshold_pct", 40.0)),
        max_trials=int(cfg_payload.get("max_trials", 300)),
        time_budget_seconds=int(cfg_payload["time_budget_seconds"]) if cfg_payload.get("time_budget_seconds") is not None else None,
        n_jobs=1,
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
        broker_profile=str(cfg_payload.get("broker_profile", cfg_payload.get("source", "perps"))),
        perps_maintenance_margin_rate=float(cfg_payload.get("perps_maintenance_margin_rate", 0.01)),
        cfd_initial_margin_rate=float(cfg_payload.get("cfd_initial_margin_rate", 0.01)),
        cfd_stopout_margin_level=float(cfg_payload.get("cfd_stopout_margin_level", 0.5)),
        pareto_candidates_max=int(cfg_payload.get("pareto_candidates_max", 50)),
        candidate_pool=str(cfg_payload.get("candidate_pool", "pareto")),
        global_top_k=int(cfg_payload.get("global_top_k", 50)),
        ranking_metric=str(cfg_payload.get("ranking_metric", "median_pnl_per_position_test")),
        train_frac=float(cfg_payload.get("train_frac", 0.75)),
        optuna_objective_metric=str(cfg_payload.get("optuna_objective_metric", "return_train_pct")),
        require_positive_train_metric_for_test=bool(cfg_payload.get("require_positive_train_metric_for_test", True)),
        tp_mode_policy=str(cfg_payload.get("tp_mode_policy", "auto")),
        tp_rr_fixed=float(cfg_payload.get("tp_rr_fixed", 2.0)),
    )


def _configure_sqlite_pragmas(*, db_path: Path) -> None:
    try:
        conn = sqlite3.connect(str(db_path), timeout=60)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA busy_timeout=60000;")
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception:
        pass


def _connect_args_for_storage_url(storage_url: str) -> dict:
    u = str(storage_url or "").strip().lower()
    if u.startswith("sqlite"):
        return {"timeout": 60}
    if u.startswith("postgres"):
        return {"connect_timeout": 60}
    return {}


def _study_progress(study: optuna.Study) -> tuple[int, int, int, int]:
    complete = 0
    running = 0
    pruned = 0
    fail = 0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            complete += 1
        elif t.state == optuna.trial.TrialState.RUNNING:
            running += 1
        elif t.state == optuna.trial.TrialState.PRUNED:
            pruned += 1
        elif t.state == optuna.trial.TrialState.FAIL:
            fail += 1
    return complete, running, pruned, fail


def _best_pareto_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    if not study.best_trials:
        return None
    bt = [t for t in study.best_trials if t.values is not None]
    if not bt:
        return None
    return sorted(bt, key=lambda t: (t.values[1], -t.values[0]))[0]


def _bar(frac: float, *, width: int = 28) -> str:
    try:
        f = float(frac)
    except Exception:
        f = 0.0
    if f != f:
        f = 0.0
    f = max(0.0, min(1.0, f))
    n = int(round(f * width))
    return "[" + ("#" * n) + ("-" * (width - n)) + "]"


def _print_inline(msg: str, *, last_len: int) -> int:
    m = str(msg)
    pad = " " * max(0, int(last_len) - len(m))
    sys.stdout.write("\r" + m + pad)
    sys.stdout.flush()
    return len(m)


def _start_spinner(prefix: str) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def _run() -> None:
        frames = "|/-\\"
        i = 0
        t0 = time.time()
        last_len = 0
        while not stop.is_set():
            elapsed = int(time.time() - t0)
            msg = f"{prefix} {frames[i % len(frames)]} elapsed={elapsed}s"
            last_len = _print_inline(msg, last_len=last_len)
            i += 1
            stop.wait(0.2)

        _ = _print_inline(f"{prefix} done", last_len=last_len)
        sys.stdout.write("\n")
        sys.stdout.flush()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return stop, th


def _save_report_artifacts(
    *,
    run_dir: Path,
    report,
    ctx: dict,
    cfg: OptimizationConfig,
    df: pd.DataFrame,
    storage_url: str,
) -> None:
    run_id = str(ctx.get("run_id") or time.strftime("%Y%m%d_%H%M%S"))

    leaderboard_path = run_dir / "leaderboard.csv"
    global_leaderboard_path = run_dir / "global_leaderboard.csv"
    candidates_path = run_dir / "candidates.csv"
    champion_path = run_dir / "champion.json"
    report_path = run_dir / "report.json"

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
        champion_path.write_text(json.dumps(report.champion, indent=2, ensure_ascii=False), encoding="utf-8")

    report_payload = {
        "project": "ohlcv-optimizer-v1",
        "version": 1,
        "saved_at": run_id,
        "meta": getattr(report, "meta", None),
        "strategies_skipped": getattr(report, "strategies_skipped", None),
        "champion_global": getattr(report, "champion_global", None),
        "champions_by_strategy": getattr(report, "champions_by_strategy", None),
        "data": {
            "data_root": str((ctx.get("data") or {}).get("data_root")),
            "source": (ctx.get("data") or {}).get("source"),
            "symbol": (ctx.get("data") or {}).get("symbol"),
            "timeframe": (ctx.get("data") or {}).get("timeframe"),
            "start_ms": (ctx.get("data") or {}).get("start_ms"),
            "end_ms": (ctx.get("data") or {}).get("end_ms"),
            "candles_total": int(len(df)) if df is not None else None,
        },
        "config": {
            "initial_equity": cfg.initial_equity,
            "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps,
            "dd_threshold_pct": cfg.dd_threshold_pct,
            "storage_url": storage_url,
            "optuna_objective_metric": str((ctx.get("config") or {}).get("optuna_objective_metric") or "return_train_pct"),
            "max_trials": cfg.max_trials,
            "time_budget_seconds": cfg.time_budget_seconds,
            "n_jobs": int(getattr(cfg, "n_jobs", 1) or 1),
            "timeframe": cfg.timeframe,
            "pm_mode": cfg.pm_mode,
            "strategies": cfg.strategies,
            "risk_mode": str(getattr(cfg, "risk_mode", "risk")),
            "risk_pct": cfg.risk_pct,
            "fixed_notional_pct_equity": float(getattr(cfg, "fixed_notional_pct_equity", 0.0) or 0.0),
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
            "require_positive_train_metric_for_test": bool(getattr(cfg, "require_positive_train_metric_for_test", True)),
            "tp_mode_policy": str(getattr(cfg, "tp_mode_policy", "auto") or "auto"),
            "tp_rr_fixed": float(getattr(cfg, "tp_rr_fixed", 2.0) or 2.0),
            "multiprocess": True,
            "workers": int((ctx.get("config") or {}).get("workers", 1) or 1),
        },
        "leaderboard": report.leaderboard.to_dict(orient="records"),
        "global_leaderboard": (
            report.global_leaderboard.to_dict(orient="records") if getattr(report, "global_leaderboard", None) is not None else None
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


def _safe_name(value: str) -> str:
    s = str(value or "").strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "unknown"


def _json_default(v):
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    try:
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
    except Exception:
        pass
    return str(v)


def _candidate_params_from_row(row: dict) -> dict:
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
        "liquidated_train",
        "liquidated_test",
        "peak_notional_pct_equity_train",
        "peak_notional_pct_equity_test",
        "peak_qty_mult_train",
        "peak_qty_mult_test",
        "cap_hit_rate_train",
        "cap_hit_rate_test",
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
        "eligible",
        "trial",
        "seconds",
        "rank_pre_wf",
        "rank_post_wf",
        "rank_delta",
    }
    params: dict = {}
    for k, v in (row or {}).items():
        if k in drop_keys:
            continue
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        params[str(k)] = v
    return params


def _profit_factor(pos_df: pd.DataFrame) -> float:
    if pos_df is None or pos_df.empty or "pnl_total" not in pos_df.columns:
        return 0.0
    pnl = pd.to_numeric(pos_df["pnl_total"], errors="coerce").fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float((-pnl[pnl < 0]).sum())
    if gross_loss <= 1e-12:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _period_days_from_df(df_slice: pd.DataFrame) -> float:
    if df_slice is None or df_slice.empty or "timestamp_ms" not in df_slice.columns:
        return 1e-9
    start_ms = int(df_slice["timestamp_ms"].iloc[0])
    end_ms = int(df_slice["timestamp_ms"].iloc[-1])
    return max(1e-9, float(end_ms - start_ms) / 86_400_000.0)


def _ann_return_from_equity(eq: list[float], *, period_days: float) -> float:
    if not eq:
        return 0.0
    start_e = float(eq[0])
    end_e = float(eq[-1])
    if start_e <= 0:
        return 0.0
    total = end_e / max(start_e, 1e-12)
    return float(total ** (365.0 / max(float(period_days), 1e-9)) - 1.0)


def _sharpe_from_equity(eq: list[float]) -> float:
    try:
        s = pd.Series(eq).astype("float64")
    except Exception:
        return 0.0
    if len(s) < 2:
        return 0.0
    prev = s.shift(1)
    rets = (s - prev) / prev.replace(0, 1e-12)
    rets = rets.dropna()
    if rets.empty:
        return 0.0
    mu = float(rets.mean())
    sigma = float(rets.std(ddof=0))
    if sigma <= 1e-12:
        return 0.0
    return float((mu / sigma) * (float(len(rets)) ** 0.5))


def _downsample_ts_eq(*, ts_ms: list[int], eq: list[float], max_points: int = 3000) -> tuple[list[int], list[float]]:
    n = min(int(len(ts_ms)), int(len(eq)))
    if n <= 0:
        return ([], [])
    if n <= int(max_points):
        return (ts_ms[:n], eq[:n])
    step = max(1, int((n + int(max_points) - 1) // int(max_points)))
    idx = list(range(0, n, step))
    return ([int(ts_ms[i]) for i in idx], [float(eq[i]) for i in idx])


def _build_candidate_analytics_artifacts(*, run_dir: Path, report, cfg: OptimizationConfig, df: pd.DataFrame) -> None:
    if getattr(report, "global_leaderboard", None) is None:
        return
    glb = report.global_leaderboard
    if glb is None or glb.empty:
        return

    out_dir = run_dir / "candidate_analytics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_strats = builtin_strategies()
    by_name = {s.name: s for s in all_strats}

    train_frac = float(getattr(cfg, "train_frac", 0.75) or 0.75)
    n = len(df)
    if n < 10:
        cut = n
    else:
        cut = max(1, min(n - 1, int(n * train_frac)))

    df_train = df.iloc[:cut].reset_index(drop=True)
    df_test = df.iloc[cut:].reset_index(drop=True) if cut < n else df.iloc[:0].copy()

    ts_full = [int(x) for x in pd.to_numeric(df["timestamp_ms"], errors="coerce").fillna(0).astype("int64").to_list()]
    ts_train = [int(x) for x in pd.to_numeric(df_train["timestamp_ms"], errors="coerce").fillna(0).astype("int64").to_list()]
    ts_test = [int(x) for x in pd.to_numeric(df_test["timestamp_ms"], errors="coerce").fillna(0).astype("int64").to_list()]

    ctx = StrategyContext(timeframe=str(getattr(cfg, "timeframe", "5m")))

    rows = glb.to_dict(orient="records")
    for row in rows:
        try:
            strat_name = str(row.get("strategy") or "")
            if not strat_name:
                continue
            tr_num = int(row.get("trial"))
        except Exception:
            continue

        strat = by_name.get(strat_name)
        if strat is None:
            continue

        out_path = out_dir / f"{_safe_name(strat_name)}__trial_{int(tr_num)}.json"
        if out_path.exists():
            continue

        params = _candidate_params_from_row(row)
        params["strategy"] = strat_name

        try:
            bt_cfg = build_backtest_config_from_params(params=params, base=cfg)
            sig_all = strat.compute_signal(df, params, ctx)
        except Exception:
            continue

        sig_train = sig_all.iloc[: len(df_train)].reset_index(drop=True)
        sig_test = sig_all.iloc[len(df_train) : len(df_train) + len(df_test)].reset_index(drop=True)

        payload: dict = {
            "version": 1,
            "strategy": strat_name,
            "trial": int(tr_num),
            "analytics": {
                "train_frac": float(train_frac),
                "cut": int(cut),
                "candles_total": int(n),
            },
            "segments": {},
        }

        for seg_name, df_slice, sig_slice, ts_ms in [
            ("train", df_train, sig_train, ts_train),
            ("test", df_test, sig_test, ts_test),
            ("full", df.reset_index(drop=True), sig_all.reset_index(drop=True), ts_full),
        ]:
            if df_slice is None or df_slice.empty:
                payload["segments"][seg_name] = {"empty": True}
                continue

            try:
                period_start_ms = int(pd.to_numeric(df_slice["timestamp_ms"].iloc[0], errors="coerce"))
                period_end_ms = int(pd.to_numeric(df_slice["timestamp_ms"].iloc[-1], errors="coerce"))
            except Exception:
                period_start_ms = None
                period_end_ms = None
            try:
                res = run_backtest(df=df_slice, signal=sig_slice, config=bt_cfg)
                pos_df, overview = summarize_positions(res.trades)
            except Exception:
                payload["segments"][seg_name] = {"error": True}
                continue

            eq = [float(x) for x in pd.Series(res.equity_curve).astype("float64").to_list()]
            period_days = float(_period_days_from_df(df_slice))
            ann = float(_ann_return_from_equity(eq, period_days=period_days))
            pf = float(_profit_factor(pos_df))
            sharpe_eq = float(_sharpe_from_equity(eq))

            ts_ds, eq_ds = _downsample_ts_eq(ts_ms=ts_ms, eq=eq, max_points=3000)

            payload["segments"][seg_name] = {
                "period_start_ms": int(period_start_ms) if period_start_ms is not None else None,
                "period_end_ms": int(period_end_ms) if period_end_ms is not None else None,
                "ts_ms": ts_ds,
                "equity": eq_ds,
                "kpis": {
                    "return_period_pct": float((eq[-1] / max(eq[0], 1e-12) - 1.0) * 100.0) if eq else 0.0,
                    "return_annualized_pct": float(ann * 100.0),
                    "max_drawdown_pct": float(res.max_drawdown_pct),
                    "max_drawdown_intrabar_pct": float(getattr(res, "max_drawdown_intrabar_pct", 0.0)),
                    "exec_reject_rate": float(getattr(res, "exec_reject_rate", 0.0)),
                    "exec_round_rate": float(getattr(res, "exec_round_rate", 0.0)),
                    "liquidated": bool(getattr(res, "liquidated", False)),
                    "peak_notional_pct_equity": float(getattr(res, "peak_notional_pct_equity", 0.0)),
                    "peak_qty_mult": float(getattr(res, "peak_qty_mult", 0.0)),
                    "cap_hit_rate": float(getattr(res, "cap_hit_rate", 0.0)),
                    "profit_factor": float(pf),
                    "sharpe_equity": float(sharpe_eq),
                    "period_days": float(period_days),
                },
                "overview": overview,
                "positions": pos_df.to_dict(orient="records") if pos_df is not None and not pos_df.empty else [],
            }

        try:
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    p = argparse.ArgumentParser(prog="ohlcv-optimizer-v1")
    p.add_argument("--run", default=None)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--latest", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--no-report", action="store_true")
    args = p.parse_args()

    run_dir = _resolve_run_dir(run=str(args.run) if args.run else None, run_dir=str(args.run_dir) if args.run_dir else None, latest=bool(args.latest))
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx_path = run_dir / "context.json"
    if not ctx_path.exists():
        raise SystemExit(f"Missing context.json in run dir: {run_dir}")

    ctx = _read_json(ctx_path)
    cfg = _cfg_from_context(ctx.get("config") or {})

    data = ctx.get("data") or {}
    data_root = str(data.get("data_root"))
    source = str(data.get("source"))
    symbol = str(data.get("symbol"))
    timeframe = str(data.get("timeframe"))
    start_ms = data.get("start_ms")
    end_ms = data.get("end_ms")

    workers = int(args.workers) if args.workers is not None else int((ctx.get("config") or {}).get("workers", 1) or 1)
    workers = max(1, workers)

    stop_flag = run_dir / "stop.flag"
    if stop_flag.exists():
        try:
            stop_flag.unlink()
        except Exception:
            pass

    db_path = run_dir / "optuna.db"
    default_storage_url = f"sqlite:///{db_path.as_posix()}"

    cfg_storage_url = str((ctx.get("config") or {}).get("storage_url") or "").strip()
    if not cfg_storage_url:
        storage_url = default_storage_url
    else:
        u_cfg = cfg_storage_url.strip().lower()
        if u_cfg.startswith("sqlite"):
            try:
                parsed = urlparse(str(cfg_storage_url))
                p = unquote(str(parsed.path or ""))
                if p.startswith("/") and len(p) >= 3 and p[2] == ":":
                    p = p[1:]
                cfg_db_path = Path(p)
                if cfg_db_path.resolve() != db_path.resolve():
                    storage_url = default_storage_url
                else:
                    storage_url = cfg_storage_url
            except Exception:
                storage_url = default_storage_url
        else:
            storage_url = cfg_storage_url

    try:
        u = str(storage_url or "").strip().lower()
        if u.startswith("sqlite"):
            parsed = urlparse(str(storage_url))
            p = unquote(str(parsed.path or ""))
            if p.startswith("/") and len(p) >= 3 and p[2] == ":":
                p = p[1:]
            if p and p != ":memory:":
                _configure_sqlite_pragmas(db_path=Path(p))
    except Exception:
        pass

    connect_args = _connect_args_for_storage_url(storage_url)
    storage = optuna.storages.RDBStorage(
        str(storage_url),
        engine_kwargs={"connect_args": connect_args},
    )

    status_path = run_dir / "status.json"
    progress_path = run_dir / "progress.jsonl"

    strategies = cfg.strategies or []
    if not strategies:
        from hyperliquid_ohlcv_optimizer.strategies.registry import builtin_strategies

        strategies = [s.name for s in builtin_strategies()]

    stop_mode = str((ctx.get("config") or {}).get("stop_mode") or "").strip().lower()
    is_time_budget = (cfg.time_budget_seconds is not None) and (stop_mode == "time")

    optuna_objective_metric = str((ctx.get("config") or {}).get("optuna_objective_metric") or "return_train_pct")

    run_started_at = time.time()

    _write_json_atomic(
        status_path,
        {
            "state": "running",
            "run_dir": str(run_dir),
            "started_at": run_started_at,
            "updated_at": run_started_at,
            "storage_url": storage_url,
            "workers": int(workers),
            "strategies": strategies,
            "current_strategy": None,
            "current_strategy_index": None,
            "strategies_total": int(len(strategies)),
            "max_trials_per_strategy": int(cfg.max_trials),
            "time_budget_seconds": cfg.time_budget_seconds,
        },
    )

    env = os.environ.copy()
    src_dir = str(_project_root() / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + str(env.get("PYTHONPATH", ""))

    had_keyboard_interrupt = False
    try:
        for si, strat_name in enumerate(strategies):
            if stop_flag.exists():
                break

            study_name = f"{run_dir.name}.{str(strat_name)}"

            remaining = None
            if is_time_budget:
                remaining = float(cfg.time_budget_seconds)

            _write_json_atomic(
                status_path,
                {
                    "state": "running",
                    "run_dir": str(run_dir),
                    "started_at": run_started_at,
                    "updated_at": time.time(),
                    "storage_url": storage_url,
                    "workers": int(workers),
                    "strategies": strategies,
                    "current_strategy": str(strat_name),
                    "current_strategy_index": int(si) + 1,
                    "strategies_total": int(len(strategies)),
                    "max_trials_per_strategy": int(cfg.max_trials),
                    "time_budget_seconds": cfg.time_budget_seconds,
                },
            )

            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "event": "strategy_start", "strategy": strat_name}, ensure_ascii=False) + "\n")

            print(f"\n[{si+1}/{len(strategies)}] strategy_start: {strat_name} (workers={workers})", flush=True)

            ctx_payload = ctx.copy()
            ctx_payload["config"] = dict(ctx_payload.get("config") or {})
            ctx_payload["config"]["strategies"] = [str(strat_name)]
            ctx_payload["config"]["workers"] = int(workers)
            tmp_ctx_path = run_dir / f"context.{strat_name}.json"
            tmp_ctx_path.write_text(json.dumps(ctx_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            procs: list[tuple[subprocess.Popen, object, object]] = []
            for i in range(workers):
                cmd = [
                    sys.executable,
                    "-m",
                    "hyperliquid_ohlcv_optimizer.optimize.optuna_worker",
                    "--storage-url",
                    storage_url,
                    "--study-name",
                    str(study_name),
                    "--strategy",
                    str(strat_name),
                    "--context",
                    str(tmp_ctx_path),
                    "--n-trials",
                    str(int(cfg.max_trials)),
                    "--seed",
                    str(42 + i),
                    "--stop-flag",
                    str(stop_flag),
                ]
                if remaining is not None:
                    cmd += ["--timeout", str(float(remaining))]
                out_path = logs_dir / f"worker.{strat_name}.{i}.out.log"
                err_path = logs_dir / f"worker.{strat_name}.{i}.err.log"
                out_f = out_path.open("a", encoding="utf-8")
                err_f = err_path.open("a", encoding="utf-8")
                procs.append((subprocess.Popen(cmd, env=env, stdout=out_f, stderr=err_f), out_f, err_f))

            last_print_t = 0.0
            last_best_num = None
            last_inline_len = 0
            while any(p0.poll() is None for p0, _, _ in procs):
                if stop_flag.exists():
                    pass

                if args.poll > 0:
                    now = time.time()
                    if (now - last_print_t) >= float(args.poll):
                        last_print_t = now
                        try:
                            study = optuna.load_study(study_name=str(study_name), storage=storage)
                            c, r, pr, fl = _study_progress(study)
                            best = _best_pareto_trial(study)

                            pct = (float(c) / float(cfg.max_trials)) if (not is_time_budget and int(cfg.max_trials) > 0) else 0.0

                            if is_time_budget:
                                msg = (
                                    f"[{si+1}/{len(strategies)}] {strat_name} "
                                    f"{_bar(pct)} complete={c} running={r} pruned={pr} fail={fl}"
                                )
                            else:
                                total_planned = int(cfg.max_trials) * int(len(strategies))
                                total_done = int(si) * int(cfg.max_trials) + int(c)
                                total_pct = float(total_done) / float(total_planned) if total_planned > 0 else 0.0
                                msg = (
                                    f"[{si+1}/{len(strategies)}] {strat_name} "
                                    f"{_bar(pct)} {c}/{int(cfg.max_trials)} "
                                    f"(running={r}) | total { _bar(total_pct) } {total_done}/{total_planned}"
                                )
                            if best is not None and best.values is not None:
                                msg += (
                                    f" | best_trial={int(best.number)} "
                                    f"train_obj({optuna_objective_metric})={float(best.values[0]):.6f} "
                                    f"train_dd={float(best.values[1]):.6f}"
                                )

                                if last_best_num != int(best.number):
                                    last_best_num = int(best.number)
                                    with progress_path.open("a", encoding="utf-8") as f:
                                        f.write(
                                            json.dumps(
                                                {
                                                    "t": time.time(),
                                                    "event": "best_update",
                                                    "strategy": str(strat_name),
                                                    "trial": int(best.number),
                                                    "train_objective_metric": str(optuna_objective_metric),
                                                    "train_objective": float(best.values[0]),
                                                    "train_dd": float(best.values[1]),
                                                },
                                                ensure_ascii=False,
                                            )
                                            + "\n"
                                        )

                            last_inline_len = _print_inline(msg, last_len=last_inline_len)

                            _write_json_atomic(
                                status_path,
                                {
                                    "state": "running",
                                    "run_dir": str(run_dir),
                                    "started_at": run_started_at,
                                    "updated_at": time.time(),
                                    "storage_url": storage_url,
                                    "workers": int(workers),
                                    "strategies": strategies,
                                    "current_strategy": str(strat_name),
                                    "current_strategy_index": int(si) + 1,
                                    "strategies_total": int(len(strategies)),
                                    "max_trials_per_strategy": int(cfg.max_trials),
                                    "time_budget_seconds": cfg.time_budget_seconds,
                                    "progress": {
                                        "complete": int(c),
                                        "running": int(r),
                                        "pruned": int(pr),
                                        "fail": int(fl),
                                        "best_trial": int(best.number) if best is not None else None,
                                    },
                                },
                            )
                        except Exception:
                            pass

                time.sleep(0.2)

            sys.stdout.write("\n")
            sys.stdout.flush()

            for p0, out_f, err_f in procs:
                try:
                    p0.wait(timeout=2.0)
                except Exception:
                    try:
                        p0.terminate()
                    except Exception:
                        pass
                try:
                    out_f.close()
                except Exception:
                    pass
                try:
                    err_f.close()
                except Exception:
                    pass

            with progress_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "event": "strategy_end", "strategy": strat_name}, ensure_ascii=False) + "\n")

            print(f"[{si+1}/{len(strategies)}] strategy_end: {strat_name}", flush=True)

    except KeyboardInterrupt:
        try:
            stop_flag.write_text("keyboard_interrupt", encoding="utf-8")
        except Exception:
            pass
        had_keyboard_interrupt = True
        print("\nStop requested (KeyboardInterrupt).", flush=True)

    final_state = "stopped" if (stop_flag.exists() or had_keyboard_interrupt) else "done"

    if args.no_report:
        _write_json_atomic(
            status_path,
            {
                "state": final_state,
                "run_dir": str(run_dir),
                "started_at": run_started_at,
                "updated_at": time.time(),
                "storage_url": storage_url,
                "workers": int(workers),
                "strategies": strategies,
                "current_strategy": None,
                "current_strategy_index": None,
                "strategies_total": int(len(strategies)),
                "max_trials_per_strategy": int(cfg.max_trials),
                "time_budget_seconds": cfg.time_budget_seconds,
            },
        )
        return

    _write_json_atomic(
        status_path,
        {
            "state": "building_report",
            "run_dir": str(run_dir),
            "started_at": run_started_at,
            "updated_at": time.time(),
            "storage_url": storage_url,
            "workers": int(workers),
            "strategies": strategies,
            "current_strategy": None,
            "current_strategy_index": None,
            "strategies_total": int(len(strategies)),
            "max_trials_per_strategy": int(cfg.max_trials),
            "time_budget_seconds": cfg.time_budget_seconds,
        },
    )

    spinner_stop, spinner_th = _start_spinner("Building report (train+test)")
    try:
        df = _load_filtered_df(
            data_root=data_root,
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if df.empty:
            raise SystemExit("No candles loaded. Cannot build report.")

        study_name_prefix = run_dir.name
        report = build_report_from_storage(
            df=df,
            config=cfg,
            storage_url=storage_url,
            study_name_prefix=study_name_prefix,
        )
        _save_report_artifacts(run_dir=run_dir, report=report, ctx=ctx, cfg=cfg, df=df, storage_url=storage_url)
        try:
            _build_candidate_analytics_artifacts(run_dir=run_dir, report=report, cfg=cfg, df=df)
        except Exception:
            pass
    finally:
        try:
            spinner_stop.set()
        except Exception:
            pass
        try:
            spinner_th.join(timeout=2.0)
        except Exception:
            pass

    _write_json_atomic(
        status_path,
        {
            "state": final_state,
            "run_dir": str(run_dir),
            "started_at": run_started_at,
            "updated_at": time.time(),
            "storage_url": storage_url,
            "workers": int(workers),
            "strategies": strategies,
            "current_strategy": None,
            "current_strategy_index": None,
            "strategies_total": int(len(strategies)),
            "max_trials_per_strategy": int(cfg.max_trials),
            "time_budget_seconds": cfg.time_budget_seconds,
        },
    )

    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
