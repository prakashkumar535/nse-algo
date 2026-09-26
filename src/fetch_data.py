import yfinance as yf
import pandas as pd
import numpy as np

WATCHLIST = [
    "MOTISONS.NS", "IFCI.NS", "BANDHANBNK.NS", "TATASTEEL.NS",
    "IDFCFIRSTB.NS", "AHCL.NS", "IDEA.NS", "IRB.NS",
    "YESBANK.NS", "JPPOWER.NS"
]

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_all():
    results = []
    for ticker in WATCHLIST:
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)

            # Fix multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 21:
                print(f"Skipping {ticker} — not enough data")
                continue

            df["EMA20"] = calculate_ema(df["Close"], 20)
            df["RSI14"] = calculate_rsi(df["Close"], 14)
            df["VolAvg20"] = df["Volume"].rolling(20).mean()
            df["VolRatio"] = df["Volume"] / df["VolAvg20"]

            latest = df.iloc[-1]
            results.append({
                "Stock": ticker.replace(".NS", ""),
                "Close": round(float(latest["Close"]), 2),
                "EMA20": round(float(latest["EMA20"]), 2),
                "RSI14": round(float(latest["RSI14"]), 2),
                "VolRatio": round(float(latest["VolRatio"]), 2),
                "Date": df.index[-1].strftime("%Y-%m-%d")
            })
            print(f"OK: {ticker}")
        except Exception as e:
            print(f"ERROR {ticker}: {e}")
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = fetch_all()
    print(df)