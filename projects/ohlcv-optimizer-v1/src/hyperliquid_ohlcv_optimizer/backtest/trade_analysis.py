from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_positions(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if trades is None or trades.empty:
        return (pd.DataFrame(), {"positions": 0})

    df = trades.copy()

    if "position_id" in df.columns:
        pos_key = "position_id"
    elif "entry_ts" in df.columns:
        pos_key = "entry_ts"
    else:
        return (pd.DataFrame(), {"positions": int(len(df))})

    if "exit_reason" not in df.columns:
        df["exit_reason"] = "unknown"

    g = df.groupby(pos_key, dropna=True)

    pnl_tp1 = (
        df.loc[df["exit_reason"] == "tp1_partial"].groupby(pos_key, dropna=True)["pnl"].sum()
        if "pnl" in df.columns
        else pd.Series(dtype="float64")
    )

    pos = pd.DataFrame(
        {
            "entry_ts": g["entry_ts"].min(),
            "exit_ts": g["exit_ts"].max(),
            "direction": g["direction"].first(),
            "pnl_total": g["pnl"].sum(),
            "fills": g.size(),
            "tp1_hit": g["exit_reason"].apply(lambda s: (s == "tp1_partial").any()),
            "final_exit_reason": g["exit_reason"].last(),
        }
    )

    pos["pnl_tp1"] = pnl_tp1.reindex(pos.index).fillna(0.0).astype("float64")

    pos["pnl_non_tp1"] = pos["pnl_total"] - pos["pnl_tp1"]

    total_pnl = float(pos["pnl_total"].sum()) if len(pos) else 0.0
    total_tp1_pnl = float(pos["pnl_tp1"].sum()) if len(pos) else 0.0
    tp1_pnl_share = (total_tp1_pnl / total_pnl) if abs(total_pnl) > 1e-12 else 0.0

    pnl_std = float(pos["pnl_total"].std(ddof=0)) if len(pos) else 0.0
    pnl_mean = float(pos["pnl_total"].mean()) if len(pos) else 0.0
    sharpe_pnl = (pnl_mean / pnl_std) * float(np.sqrt(len(pos))) if pnl_std > 1e-12 else 0.0

    overview = {
        "positions": int(len(pos)),
        "tp1_hit_rate": float(pos["tp1_hit"].mean()) if len(pos) else 0.0,
        "win_rate": float((pos["pnl_total"] > 0).mean()) if len(pos) else 0.0,
        "median_pnl_per_position": float(pos["pnl_total"].median()) if len(pos) else 0.0,
        "avg_pnl_per_position": float(pos["pnl_total"].mean()) if len(pos) else 0.0,
        "avg_fills_per_position": float(pos["fills"].mean()) if len(pos) else 0.0,
        "tp1_pnl_share": float(tp1_pnl_share),
        "sharpe_pnl_per_position": float(sharpe_pnl),
    }

    final_exit_dist = pos["final_exit_reason"].value_counts(dropna=False).to_dict()
    overview["final_exit_reason_dist"] = {str(k): int(v) for k, v in final_exit_dist.items()}

    return (pos.reset_index(drop=False).rename(columns={pos_key: "position_id"}), overview)
