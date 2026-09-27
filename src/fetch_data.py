import yfinance as yf
import pandas as pd
import numpy as np
from universe import get_yf_symbols

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_stock(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 50:
            return None

        df["EMA20"]    = calculate_ema(df["Close"], 20)
        df["EMA50"]    = calculate_ema(df["Close"], 50)
        df["EMA200"]   = calculate_ema(df["Close"], 200)
        df["RSI14"]    = calculate_rsi(df["Close"], 14)
        df["ATR14"]    = calculate_atr(df, 14)
        df["VolAvg20"] = df["Volume"].rolling(20).mean()
        df["VolRatio"] = df["Volume"] / df["VolAvg20"]
        df["VWAP"]     = (df["High"] + df["Low"] + df["Close"]) / 3
        df["Ticker"]   = ticker.replace(".NS", "")
        return df
    except Exception as e:
        print(f"ERROR {ticker}: {e}")
        return None

def fetch_all(period="1y", verbose=True):
    symbols = get_yf_symbols()
    results = []
    failed = []

    for ticker in symbols:
        df = fetch_stock(ticker, period=period)
        if df is None:
            failed.append(ticker)
            continue
        latest = df.iloc[-1]
        try:
            results.append({
                "Stock":    ticker.replace(".NS", ""),
                "Close":    round(float(latest["Close"]), 2),
                "EMA20":    round(float(latest["EMA20"]), 2),
                "EMA50":    round(float(latest["EMA50"]), 2),
                "EMA200":   round(float(latest["EMA200"]), 2),
                "RSI14":    round(float(latest["RSI14"]), 2),
                "ATR14":    round(float(latest["ATR14"]), 2),
                "VolRatio": round(float(latest["VolRatio"]), 2),
                "VWAP":     round(float(latest["VWAP"]), 2),
                "Date":     df.index[-1].strftime("%Y-%m-%d"),
            })
            if verbose:
                print(f"OK: {ticker}")
        except Exception as e:
            failed.append(ticker)
            print(f"PARSE ERROR {ticker}: {e}")

    if failed:
        print(f"\nFailed ({len(failed)}): {failed}")

    return pd.DataFrame(results)

def fetch_nifty_regime(period="1y"):
    """Fetch NIFTY 50 for regime detection"""
    try:
        df = yf.download("^NSEI", period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["EMA50"]  = calculate_ema(df["Close"], 50)
        df["EMA200"] = calculate_ema(df["Close"], 200)
        return df
    except Exception as e:
        print(f"NIFTY fetch error: {e}")
        return None

if __name__ == "__main__":
    print("Fetching NIFTY regime...")
    nifty = fetch_nifty_regime()
    if nifty is not None:
        latest = nifty.iloc[-1]
        print(f"NIFTY Close: {float(latest['Close']):.0f}")
        print(f"EMA50: {float(latest['EMA50']):.0f}")
        print(f"EMA200: {float(latest['EMA200']):.0f}")

    print("\nFetching universe...")
    df = fetch_all()
    print(f"\nTotal fetched: {len(df)} stocks")
    print(df.head())