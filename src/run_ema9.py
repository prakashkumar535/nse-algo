"""
EMA9 Trend Strategy — Full Backtest Runner
==========================================
Runs complete pipeline:
  Data → Features → Signals → Trades → Costs → Metrics → OOS → Walk-Forward
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import yfinance as yf
from universe import get_yf_symbols
from fetch_data import fetch_nifty_regime
from strategies.investment.ema9_trend import (
    EMA9TrendStrategy, EMA9Config, parameter_sensitivity
)
from backtest.engine import BacktestEngine

# ─── Config ─────────────────────────────────────────────────────

PERIOD      = "5y"
MAX_STOCKS  = 30      # increase to None for full universe
COST_PRESET = "realistic"

# ─── Load Data ──────────────────────────────────────────────────

def load_universe(max_stocks=30, period="5y"):
    symbols = get_yf_symbols()
    if max_stocks:
        symbols = symbols[:max_stocks]

    data = {}
    failed = []
    print(f"Loading {len(symbols)} stocks ({period})...")

    for ticker in symbols:
        try:
            df = yf.download(ticker, period=period, interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 100:
                failed.append(ticker)
                continue
            data[ticker] = df
        except Exception as e:
            failed.append(ticker)

    print(f"Loaded: {len(data)} stocks | Failed: {len(failed)}")
    if failed:
        print(f"Failed: {failed}")
    return data

def load_nifty(period="5y"):
    print("Loading NIFTY 50...")
    df = yf.download("^NSEI", period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# ─── Main ───────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  PulseAlgo — EMA9 Trend Strategy Research")
    print("="*55)

    # Load data
    data_dict = load_universe(max_stocks=MAX_STOCKS, period=PERIOD)
    nifty_df  = load_nifty(period=PERIOD)

    if not data_dict:
        print("No data loaded. Exiting.")
        return

    # ── Run 1: Default config (Filter A + D ON) ──────────────────
    print("\n[1/3] Default config: Filter A + D ON")
    cfg      = EMA9Config(
        universe=f"NIFTY100-{MAX_STOCKS}",
        cost_preset=COST_PRESET,
        filter_a=True,
        filter_b=False,
        filter_c=False,
        filter_d=True,
    )
    strategy = EMA9TrendStrategy(config=cfg)
    engine   = BacktestEngine(
        strategy=strategy,
        data=data_dict,
        nifty_df=nifty_df,
        config=cfg,
    )
    results = engine.run(
        modes=["train", "oos"],
        notes="Default config — FilterA + FilterD"
    )

    # ── Run 2: No filters (raw EMA9 crossover) ───────────────────
    print("\n[2/3] No filters — raw EMA9 crossover only")
    cfg2      = EMA9Config(
        name="EMA9 Trend",
        version="1.0.1",
        universe=f"NIFTY100-{MAX_STOCKS}",
        cost_preset=COST_PRESET,
        filter_a=False,
        filter_b=False,
        filter_c=False,
        filter_d=False,
    )
    strategy2 = EMA9TrendStrategy(config=cfg2)
    engine2   = BacktestEngine(
        strategy=strategy2,
        data=data_dict,
        nifty_df=nifty_df,
        config=cfg2,
    )
    results2 = engine2.run(
        modes=["train", "oos"],
        notes="No filters — raw crossover"
    )

    # ── Run 3: Parameter Sensitivity ─────────────────────────────
    print("\n[3/3] Parameter sensitivity matrix (EMA range)...")
    param_df = parameter_sensitivity(
        data_dict=data_dict,
        nifty_df=nifty_df,
        ema_range=[5, 9, 10, 20, 21],
        mode="train",
    )

    # Save sensitivity results
    os.makedirs("data/backtest", exist_ok=True)
    param_df.to_csv("data/backtest/ema9_sensitivity.csv", index=False)
    print("\nSaved: data/backtest/ema9_sensitivity.csv")

    # ── Walk Forward ─────────────────────────────────────────────
    print("\n[4/4] Walk-forward validation...")
    wf_results = engine.walk_forward(
        train_days=756,   # 3 years
        test_days=126,    # 6 months
    )

    print("\n" + "="*55)
    print("  Research Complete")
    print("  Check /runs/ folder for full backtest records")
    print("="*55)

if __name__ == "__main__":
    main()