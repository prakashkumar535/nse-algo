import pandas as pd
from fetch_data import fetch_nifty_regime

def detect_regime(nifty_df=None):
    """
    Returns regime string based on NIFTY 50 data.
    Regimes: STRONG_BULL, BULL, SIDEWAYS, BEAR, UNKNOWN
    """
    if nifty_df is None:
        nifty_df = fetch_nifty_regime()

    if nifty_df is None or nifty_df.empty:
        return "UNKNOWN"

    latest = nifty_df.iloc[-1]

    try:
        close  = float(latest["Close"])
        ema50  = float(latest["EMA50"])
        ema200 = float(latest["EMA200"])
    except Exception:
        return "UNKNOWN"

    # Distance from 200 EMA as %
    pct_from_200 = (close - ema200) / ema200 * 100

    if close > ema50 and ema50 > ema200 and pct_from_200 > 5:
        regime = "STRONG_BULL"
    elif close > ema200 and pct_from_200 > 0:
        regime = "BULL"
    elif abs(pct_from_200) <= 3:
        regime = "SIDEWAYS"
    elif close < ema200:
        regime = "BEAR"
    else:
        regime = "SIDEWAYS"

    return regime

def regime_allows_buy(regime):
    return regime in ("STRONG_BULL", "BULL")

def regime_summary(nifty_df=None):
    if nifty_df is None:
        nifty_df = fetch_nifty_regime()

    regime = detect_regime(nifty_df)
    latest = nifty_df.iloc[-1]

    return {
        "regime":  regime,
        "close":   round(float(latest["Close"]), 0),
        "ema50":   round(float(latest["EMA50"]), 0),
        "ema200":  round(float(latest["EMA200"]), 0),
        "buy_ok":  regime_allows_buy(regime),
        "date":    nifty_df.index[-1].strftime("%Y-%m-%d"),
    }

if __name__ == "__main__":
    summary = regime_summary()
    print(f"Regime    : {summary['regime']}")
    print(f"NIFTY     : {summary['close']}")
    print(f"EMA50     : {summary['ema50']}")
    print(f"EMA200    : {summary['ema200']}")
    print(f"BUY OK?   : {summary['buy_ok']}")
    print(f"Date      : {summary['date']}")