"""
Paper Trading Journal
=====================
Track H5 and Momentum signals manually.
Records every paper trade with full context.
Stored as CSV — no database needed.
"""

import pandas as pd
import os
from datetime import datetime

PAPER_TRADES_FILE = "data/paper_trades.csv"

COLUMNS = [
    "trade_id",
    "date_entered",
    "strategy",        # H5 / MOMENTUM
    "version",         # strategy version
    "ticker",
    "direction",       # BUY / SELL
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "qty",
    "gross_ret_pct",
    "net_ret_pct",
    "hold_days",
    "exit_reason",     # SIGNAL / STOP / MANUAL / REGIME_EXIT
    "regime_at_entry", # BULL / BEAR etc
    "nifty_at_entry",
    "notes",
    "status",          # OPEN / CLOSED
]


def load_trades() -> pd.DataFrame:
    if not os.path.exists(PAPER_TRADES_FILE):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(PAPER_TRADES_FILE)
    # ensure all columns exist
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df


def save_trades(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_csv(PAPER_TRADES_FILE, index=False)


def add_trade(
    strategy:         str,
    ticker:           str,
    entry_date:       str,
    entry_price:      float,
    qty:              int   = 1,
    version:          str   = "1.0.0",
    regime_at_entry:  str   = "",
    nifty_at_entry:   float = 0.0,
    notes:            str   = "",
) -> str:
    df = load_trades()
    trade_id = f"PT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    new_row = {col: None for col in COLUMNS}
    new_row.update({
        "trade_id":        trade_id,
        "date_entered":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "strategy":        strategy,
        "version":         version,
        "ticker":          ticker,
        "direction":       "BUY",
        "entry_date":      entry_date,
        "entry_price":     entry_price,
        "qty":             qty,
        "regime_at_entry": regime_at_entry,
        "nifty_at_entry":  nifty_at_entry,
        "notes":           notes,
        "status":          "OPEN",
    })
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_trades(df)
    return trade_id


def close_trade(
    trade_id:    str,
    exit_date:   str,
    exit_price:  float,
    exit_reason: str = "SIGNAL",
    notes:       str = "",
) -> dict | None:
    df = load_trades()
    mask = df["trade_id"] == trade_id
    if not mask.any():
        return None

    idx          = df[mask].index[0]
    entry_price  = float(df.loc[idx, "entry_price"])
    entry_date   = pd.to_datetime(df.loc[idx, "entry_date"])
    exit_date_dt = pd.to_datetime(exit_date)
    hold_days    = (exit_date_dt - entry_date).days

    gross_ret = (exit_price - entry_price) / entry_price * 100
    net_ret   = gross_ret - 0.394   # realistic round-trip cost

    df.loc[idx, "exit_date"]   = exit_date
    df.loc[idx, "exit_price"]  = exit_price
    df.loc[idx, "gross_ret_pct"] = round(gross_ret, 2)
    df.loc[idx, "net_ret_pct"] = round(net_ret, 2)
    df.loc[idx, "hold_days"]   = hold_days
    df.loc[idx, "exit_reason"] = exit_reason
    df.loc[idx, "status"]      = "CLOSED"
    if notes:
        df.loc[idx, "notes"] = str(df.loc[idx, "notes"]) + " | " + notes

    save_trades(df)
    return {
        "trade_id":    trade_id,
        "ticker":      df.loc[idx, "ticker"],
        "gross_ret":   round(gross_ret, 2),
        "net_ret":     round(net_ret, 2),
        "hold_days":   hold_days,
        "exit_reason": exit_reason,
    }


def get_summary() -> dict:
    df = load_trades()
    if df.empty:
        return {"total": 0, "open": 0, "closed": 0}

    closed = df[df["status"] == "CLOSED"].copy()
    open_  = df[df["status"] == "OPEN"].copy()

    summary = {
        "total":  len(df),
        "open":   len(open_),
        "closed": len(closed),
    }

    if not closed.empty:
        rets = pd.to_numeric(closed["net_ret_pct"], errors="coerce").dropna()
        if len(rets) > 0:
            wins = (rets > 0).sum()
            summary.update({
                "win_rate":     round(wins / len(rets) * 100, 1),
                "avg_ret":      round(rets.mean(), 2),
                "total_ret":    round(rets.sum(), 2),
                "best_trade":   round(rets.max(), 2),
                "worst_trade":  round(rets.min(), 2),
                "profit_factor": round(
                    abs(rets[rets > 0].sum()) / abs(rets[rets < 0].sum()), 2
                ) if rets[rets < 0].sum() != 0 else 999,
            })

    return summary