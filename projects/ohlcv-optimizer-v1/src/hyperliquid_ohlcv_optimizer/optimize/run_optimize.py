from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hyperliquid_ohlcv_optimizer.data.ohlcv_loader import load_ohlcv
from hyperliquid_ohlcv_optimizer.optimize.optuna_runner import OptimizationConfig, build_report_from_storage


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
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


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
        risk_pct=float(cfg_payload.get("risk_pct", 0.01)),
        max_position_notional_pct_equity=float(cfg_payload.get("max_position_notional_pct_equity", 100.0)),
        pareto_candidates_max=int(cfg_payload.get("pareto_candidates_max", 50)),
        candidate_pool=str(cfg_payload.get("candidate_pool", "pareto")),
        global_top_k=int(cfg_payload.get("global_top_k", 50)),
        ranking_metric=str(cfg_payload.get("ranking_metric", "median_pnl_per_position_test")),
        train_frac=float(cfg_payload.get("train_frac", 0.75)),
    )


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


def _save_report_artifacts(*, run_dir: Path, report, ctx: dict, cfg: OptimizationConfig, df: pd.DataFrame) -> None:
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
            "min_trades_train": cfg.min_trades_train,
            "min_trades_test": cfg.min_trades_test,
            "max_trials": cfg.max_trials,
            "time_budget_seconds": cfg.time_budget_seconds,
            "timeframe": cfg.timeframe,
            "pm_mode": cfg.pm_mode,
            "strategies": cfg.strategies,
            "pareto_candidates_max": cfg.pareto_candidates_max,
            "candidate_pool": cfg.candidate_pool,
            "global_top_k": cfg.global_top_k,
            "ranking_metric": cfg.ranking_metric,
            "train_frac": cfg.train_frac,
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
    storage_url = f"sqlite:///{db_path.as_posix()}"

    storage = optuna.storages.RDBStorage(
        str(storage_url),
        engine_kwargs={"connect_args": {"timeout": 60}},
    )

    status_path = run_dir / "status.json"
    progress_path = run_dir / "progress.jsonl"

    strategies = cfg.strategies or []
    if not strategies:
        from hyperliquid_ohlcv_optimizer.strategies.registry import builtin_strategies

        strategies = [s.name for s in builtin_strategies()]

    stop_mode = str((ctx.get("config") or {}).get("stop_mode") or "").strip().lower()
    is_time_budget = (cfg.time_budget_seconds is not None) and (stop_mode == "time")

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
        global_t0 = time.time()

        for si, strat_name in enumerate(strategies):
            if stop_flag.exists():
                break

            remaining = None
            if cfg.time_budget_seconds is not None:
                remaining = float(cfg.time_budget_seconds) - float(time.time() - global_t0)
                if remaining <= 0:
                    try:
                        stop_flag.write_text("time_budget", encoding="utf-8")
                    except Exception:
                        pass
                    break

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
                    str(strat_name),
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
                            study = optuna.load_study(study_name=str(strat_name), storage=storage)
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
                                msg += f" | best_trial={int(best.number)} train_return={float(best.values[0]):.6f} train_dd={float(best.values[1]):.6f}"

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
                                                    "train_return": float(best.values[0]),
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

        report = build_report_from_storage(df=df, config=cfg, storage_url=storage_url)
        _save_report_artifacts(run_dir=run_dir, report=report, ctx=ctx, cfg=cfg, df=df)
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
