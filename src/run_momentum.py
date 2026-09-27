"""
Cross-Sectional Momentum Strategy — Full Research Runner
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import yfinance as yf
from universe import get_yf_symbols
from strategies.investment.momentum import MomentumConfig, MomentumBacktester

PERIOD     = "5y"
MAX_STOCKS = 50

def load_data(max_stocks=50, period="5y"):
    symbols = get_yf_symbols()[:max_stocks]
    data    = {}
    print(f"Loading {len(symbols)} stocks...")
    for ticker in symbols:
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 252:
                data[ticker] = df
        except:
            pass
    print(f"Loaded: {len(data)} stocks")
    return data

def load_nifty(period="5y"):
    df = yf.download("^NSEI", period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def main():
    print("\n" + "="*55)
    print("  PulseAlgo — Cross-Sectional Momentum Research")
    print("="*55)

    data_dict = load_data(MAX_STOCKS, PERIOD)
    nifty_df  = load_nifty(PERIOD)

    # ── Run 1: Default (top 10, all lookbacks, regime ON) ────────
    print("\n[1/3] Default: Top 10, 3M+6M+12M, Regime ON")
    cfg1 = MomentumConfig(
        top_n=10,
        use_3m=True, use_6m=True, use_12m=True,
        filter_d=True,
    )
    bt1 = MomentumBacktester(data_dict, nifty_df, cfg1)
    trades1, hist1 = bt1.run(mode="train")
    bt1.print_results(trades1, hist1, "train")

    trades1_oos, hist1_oos = bt1.run(mode="oos")
    bt1.print_results(trades1_oos, hist1_oos, "oos")

    # ── Run 2: 12M only (strongest academic signal) ──────────────
    print("\n[2/3] 12M momentum only (strongest academic signal)")
    cfg2 = MomentumConfig(
        top_n=10,
        use_3m=False, use_6m=False, use_12m=True,
        weight_12m=1.0,
        filter_d=True,
    )
    bt2 = MomentumBacktester(data_dict, nifty_df, cfg2)
    trades2, hist2 = bt2.run(mode="train")
    bt2.print_results(trades2, hist2, "train")

    # ── Run 3: Top 5 vs Top 20 comparison ────────────────────────
    print("\n[3/3] Portfolio size sensitivity: Top 5 vs Top 20")
    for top_n in [5, 10, 20]:
        cfg = MomentumConfig(top_n=top_n, filter_d=True)
        bt  = MomentumBacktester(data_dict, nifty_df, cfg)
        t, h = bt.run(mode="train")
        if not t.empty:
            wr  = (t["net_ret"] > 0).sum() / len(t) * 100
            exp = t["net_ret"].mean()
            print(f"  Top {top_n:2d}: Trades={len(t):3d} | "
                  f"WR={wr:.1f}% | Expectancy={exp:.2f}%")

    # Save results
    os.makedirs("data/backtest", exist_ok=True)
    if not trades1.empty:
        trades1.to_csv("data/backtest/momentum_trades_train.csv", index=False)
    if not trades1_oos.empty:
        trades1_oos.to_csv("data/backtest/momentum_trades_oos.csv", index=False)
    print("\nSaved: data/backtest/momentum_trades_*.csv")
    print("\n" + "="*55)
    print("  Research Complete")
    print("="*55)

if __name__ == "__main__":
    main()