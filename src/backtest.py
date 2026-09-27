import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from universe import get_yf_symbols
from fetch_data import calculate_ema, calculate_rsi, calculate_atr, fetch_nifty_regime

ROUND_TRIP_COST = 0.004  # 0.4% round-trip

# ─── Data ───────────────────────────────────────────────────────

def fetch_history(ticker, period="2y"):
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 100:
            return None
        return df
    except Exception as e:
        print(f"  Failed {ticker}: {e}")
        return None

def add_features(df, nifty_df=None):
    df = df.copy()
    df["EMA20"]    = calculate_ema(df["Close"], 20)
    df["EMA50"]    = calculate_ema(df["Close"], 50)
    df["EMA200"]   = calculate_ema(df["Close"], 200)
    df["RSI14"]    = calculate_rsi(df["Close"], 14)
    df["ATR14"]    = calculate_atr(df, 14)
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolRatio"] = df["Volume"] / df["VolAvg20"]
    df["VWAP"]     = (df["High"] + df["Low"] + df["Close"]) / 3
    df["EMA200_prev"] = df["EMA200"].shift(5)
    df["Uptrend"]  = (df["EMA200"] > df["EMA200_prev"]).astype(int)

    if nifty_df is not None:
        nifty_clean = nifty_df[["Close", "EMA50", "EMA200"]].copy()
        nifty_clean.columns = ["N_Close", "N_EMA50", "N_EMA200"]
        df = df.join(nifty_clean, how="left")
        df["N_Close"]  = df["N_Close"].ffill()
        df["N_EMA50"]  = df["N_EMA50"].ffill()
        df["N_EMA200"] = df["N_EMA200"].ffill()
        df["BullRegime"] = (
            (df["N_Close"] > df["N_EMA50"]) &
            (df["N_EMA50"] > df["N_EMA200"])
        ).astype(int)
    else:
        df["BullRegime"] = 1

    return df.dropna()

def merge_delivery(df, ticker):
    """
    Merge NSE delivery data into OHLCV dataframe.
    Matches by date. Adds DelivPct column.
    """
    symbol = ticker.replace(".NS", "")
    delivery_dir = "data/delivery"

    if not os.path.exists(delivery_dir):
        return df

    files = sorted([f for f in os.listdir(delivery_dir)
                    if f.endswith(".csv")])
    if not files:
        return df

    records = []
    for fname in files:
        date_str = fname.replace(".csv", "")
        try:
            ddf = pd.read_csv(f"{delivery_dir}/{fname}")
            ddf.columns = ddf.columns.str.strip()
            if "Symbol" not in ddf.columns:
                continue
            row = ddf[ddf["Symbol"].str.strip() == symbol]
            if not row.empty and "DelivPct" in row.columns:
                records.append({
                    "Date":     pd.to_datetime(date_str),
                    "DelivPct": float(row["DelivPct"].values[0])
                })
        except Exception:
            continue

    if not records:
        return df

    deliv_df = pd.DataFrame(records).set_index("Date")
    df = df.join(deliv_df, how="left")
    df["DelivPct"] = df["DelivPct"].ffill()
    return df

# ─── Signal Hypotheses ──────────────────────────────────────────

def signal_h1(df):
    """H1: Volume Surge in Bull Regime"""
    return (
        (df["Close"] > df["EMA20"]) &
        (df["VolRatio"] >= 1.5) &
        (df["Close"] > df["VWAP"]) &
        (df["Uptrend"] == 1) &
        (df["BullRegime"] == 1)
    ).astype(int)

def signal_h2(df):
    """H2: RSI Momentum with Trend in Bull Regime"""
    return (
        (df["RSI14"] >= 40) &
        (df["RSI14"] <= 60) &
        (df["Close"] > df["EMA50"]) &
        (df["VolRatio"] >= 1.2) &
        (df["EMA20"] > df["EMA50"]) &
        (df["BullRegime"] == 1)
    ).astype(int)

def signal_h3(df):
    """H3: Full EMA Alignment in Bull Regime"""
    return (
        (df["Close"] > df["EMA20"]) &
        (df["EMA20"] > df["EMA50"]) &
        (df["EMA50"] > df["EMA200"]) &
        (df["RSI14"] < 70) &
        (df["BullRegime"] == 1)
    ).astype(int)

def signal_h4(df):
    """
    H4: Mean Reversion
    - 3 consecutive down days
    - Total 3-day drop > 5%
    - RSI < 35 (oversold)
    - Volume declining on day 3 (exhaustion)
    - Works in ANY regime (mean reversion doesn't need bull market)
    """
    close = df["Close"]
    volume = df["Volume"]

    day1_down = close < close.shift(1)
    day2_down = close.shift(1) < close.shift(2)
    day3_down = close.shift(2) < close.shift(3)

    three_day_drop = (close - close.shift(3)) / close.shift(3) * 100
    vol_declining  = volume < volume.shift(1)

    return (
        day1_down &
        day2_down &
        day3_down &
        (three_day_drop < -5) &
        (df["RSI14"] < 35) &
        vol_declining
    ).astype(int)

# ─── HYPOTHESIS H5: 52-Week High Momentum ───────────────────────
def signal_h5(df):
    """
    H5 v2: 52-week high breakout momentum — tightened
    - Close ABOVE previous 52w high (true breakout)
    - Volume > 2x avg (strong conviction)
    - RSI 55-75 (momentum but not extreme)
    - EMA20 > EMA50 (trend confirmation)
    - Bull regime
    """
    # 52w high of PREVIOUS day (avoid look-ahead)
    high_52w_prev = df["High"].shift(1).rolling(252).max()

    true_breakout = df["Close"] > high_52w_prev

    return (
        true_breakout &
        (df["VolRatio"] >= 2.0) &
        (df["RSI14"] >= 55) &
        (df["RSI14"] <= 75) &
        (df["EMA20"] > df["EMA50"]) &
        (df["BullRegime"] == 1)
    ).astype(int)

# ─── HYPOTHESIS H6: Delivery + Price Momentum ───────────────────
def signal_h6(df):
    """
    H6: High delivery % + price uptrend
    Requires DelivPct column (from NSE delivery data)
    - Delivery % > 60 (institutional accumulation proxy)
    - Close > EMA20 (uptrend)
    - VolRatio > 1.2
    - RSI 45-65
    - Bull regime
    Only fires when delivery data available
    """
    if "DelivPct" not in df.columns:
        return pd.Series(0, index=df.index)

    return (
        (df["DelivPct"] > 60) &
        (df["Close"] > df["EMA20"]) &
        (df["VolRatio"] >= 1.2) &
        (df["RSI14"] >= 45) &
        (df["RSI14"] <= 65) &
        (df["BullRegime"] == 1)
    ).astype(int)

# ─── Backtest Engine ────────────────────────────────────────────

def backtest_signal(df, signal_series, hold_days=5, label="H?",
                    stop_loss_atr=2.0):
    """
    stop_loss_atr: exit early if price drops > N x ATR14 from entry
    """
    trades  = []
    signals = signal_series.values
    opens   = df["Open"].values
    closes  = df["Close"].values
    lows    = df["Low"].values
    atrs    = df["ATR14"].values
    dates   = df.index

    for i in range(len(df) - hold_days - 1):
        if signals[i] != 1:
            continue

        entry_price = opens[i + 1]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        atr        = atrs[i]
        stop_price = entry_price - (stop_loss_atr * atr)

        exit_price = None
        exit_date  = None
        stopped    = False

        for j in range(i + 1, min(i + hold_days + 1, len(df))):
            if lows[j] <= stop_price:
                exit_price = stop_price   # assume stop hit at stop price
                exit_date  = dates[j]
                stopped    = True
                break

        if not stopped:
            exit_price = closes[i + hold_days]
            exit_date  = dates[i + hold_days]

        if pd.isna(exit_price):
            continue

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret   = gross_ret - ROUND_TRIP_COST

        trades.append({
            "entry_date": dates[i + 1],
            "exit_date":  exit_date,
            "entry":      round(entry_price, 2),
            "exit":       round(exit_price, 2),
            "gross_ret":  round(gross_ret * 100, 2),
            "net_ret":    round(net_ret * 100, 2),
            "stopped":    stopped,
            "hypothesis": label,
        })

    return pd.DataFrame(trades)

def calc_metrics(trades_df, label="", hold_days=5):
    if trades_df.empty or len(trades_df) < 5:
        return None

    rets  = trades_df["net_ret"].values / 100
    n     = len(rets)
    wins  = (rets > 0).sum()
    losses = (rets < 0).sum()

    win_rate      = wins / n
    avg_ret       = rets.mean()
    avg_win       = rets[rets > 0].mean() if wins > 0 else 0
    avg_loss      = rets[rets < 0].mean() if losses > 0 else 0
    gross_profit  = rets[rets > 0].sum()
    gross_loss    = rets[rets < 0].sum()
    profit_factor = abs(gross_profit) / abs(gross_loss) if gross_loss != 0 else np.inf

    sharpe = (avg_ret / rets.std(ddof=1)) * np.sqrt(252 / hold_days) if rets.std(ddof=1) > 0 else 0

    cum      = (1 + pd.Series(rets)).cumprod()
    roll_max = cum.cummax()
    max_dd   = ((cum - roll_max) / roll_max).min()

    return {
        "Hypothesis":     label,
        "Trades":         n,
        "Win Rate %":     round(win_rate * 100, 1),
        "Avg Net Ret %":  round(avg_ret * 100, 2),
        "Avg Win %":      round(avg_win * 100, 2),
        "Avg Loss %":     round(avg_loss * 100, 2),
        "Profit Factor":  round(profit_factor, 2),
        "Sharpe":         round(sharpe, 2),
        "Max Drawdown %": round(max_dd * 100, 2),
        "Expectancy %":   round(avg_ret * 100, 2),
    }

# ─── Main Runner ────────────────────────────────────────────────

def run_backtest(max_stocks=30, hold_days=15, period="2y", oos=False):
    dataset_name = "OUT-OF-SAMPLE (TEST SET)" if oos else "IN-SAMPLE (TRAIN SET)"
    suffix       = "oos" if oos else "train"

    print("\n" + "=" * 60)
    print("=== PulseAlgo Backtester ===")
    print("=" * 60)
    print(f"Dataset : {dataset_name}")
    print(f"Period  : {period} | Hold: {hold_days}d | Cost: {ROUND_TRIP_COST*100:.2f}%")
    print(f"Universe: up to {max_stocks} stocks\n")

    symbols = get_yf_symbols()
    if max_stocks:
        symbols = symbols[:max_stocks]

    print("  Fetching NIFTY 50 regime data...")
    nifty_df = fetch_nifty_regime(period=period)

    all_trades = {"H5_52wHigh": [], "H6_DelivMomentum": []}
    processed  = 0

    for i, ticker in enumerate(symbols):
        df = fetch_history(ticker, period=period)
        if df is None:
            continue

        df = add_features(df, nifty_df=nifty_df)
        df = merge_delivery(df, ticker)
        if len(df) < 60:
            continue

        split    = int(len(df) * 0.70)
        df_part  = df.iloc[split:] if oos else df.iloc[:split]

        if len(df_part) <= hold_days + 1:
            continue

        for label, sig_fn in [
            ("H5_52wHigh",       signal_h5),
            ("H6_DelivMomentum", signal_h6),
        ]:
        
            sig = sig_fn(df_part)
            trd = backtest_signal(df_part, sig, hold_days=hold_days, label=label)
            if not trd.empty:
                trd["ticker"]  = ticker.replace(".NS", "")
                trd["dataset"] = "OOS" if oos else "TRAIN"
                all_trades[label].append(trd)

        processed += 1
        if (i + 1) % 5 == 0:
            print(f"  Processed {i+1}/{len(symbols)} stocks...")

    print(f"\n=== RESULTS ({dataset_name}) ===\n")
    metrics_list = []

    for label, trade_list in all_trades.items():
        if not trade_list:
            print(f"{label}: No trades generated")
            continue
        combined = pd.concat(trade_list, ignore_index=True)
        m = calc_metrics(combined, label=label, hold_days=hold_days)
        if m is None:
            print(f"{label}: Insufficient trades")
            continue
        metrics_list.append(m)
        print(f"--- {label} ---")
        for k, v in m.items():
            if k != "Hypothesis":
                print(f"  {k:<20}: {v}")
        print()

    os.makedirs("data/backtest", exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    if metrics_list:
        mdf = pd.DataFrame(metrics_list)
        mdf.to_csv(f"data/backtest/latest_metrics_{suffix}.csv", index=False)
        mdf.to_csv(f"data/backtest/metrics_{suffix}_{today}.csv", index=False)
        print(f"Saved: data/backtest/latest_metrics_{suffix}.csv")

    for label, trade_list in all_trades.items():
        if trade_list:
            pd.concat(trade_list, ignore_index=True).to_csv(
                f"data/backtest/trades_{label}_{suffix}_{today}.csv", index=False)

    print(f"\nStocks tested: {processed} | Hold: {hold_days}d | Dataset: {dataset_name}")
    print("Acceptance: Win Rate > 55%, Sharpe > 0.5, Trades > 50\n")


if __name__ == "__main__":
    run_backtest(max_stocks=None, hold_days=20, period="5y", oos=False)