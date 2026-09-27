import pandas as pd
import os
from datetime import datetime
from fetch_data import fetch_all, fetch_nifty_regime
from regime import detect_regime, regime_allows_buy, regime_summary

def generate_signal(row, regime):
    buy_conditions = [
        row["Close"] > row["EMA20"],
        40 <= row["RSI14"] <= 60,
        row["VolRatio"] >= 1.5,
        row["Close"] > row["VWAP"],
        row["EMA20"] > row["EMA50"],  # short trend aligned
    ]
    sell_conditions = [
        row["Close"] < row["EMA20"],
        row["RSI14"] > 70,
        row["Close"] < row["VWAP"],
    ]

    buy_score  = sum(buy_conditions)
    sell_score = sum(sell_conditions)

    # Regime gate — no BUY in BEAR/SIDEWAYS
    if buy_score >= 3 and regime_allows_buy(regime):
        signal = "BUY"
        confidence = round((buy_score / len(buy_conditions)) * 10, 1)
    elif sell_score >= 2:
        signal = "SELL"
        confidence = round((sell_score / len(sell_conditions)) * 10, 1)
    else:
        signal = "HOLD"
        confidence = 5.0

    return signal, confidence

def run():
    print("=== PulseAlgo Signal Engine ===")

    # Step 1: Regime
    print("\n[1/3] Detecting market regime...")
    nifty_df = fetch_nifty_regime()
    r = regime_summary(nifty_df)
    regime = r["regime"]
    print(f"  Regime : {regime}")
    print(f"  NIFTY  : {r['close']} | EMA50: {r['ema50']} | EMA200: {r['ema200']}")
    print(f"  BUY OK : {r['buy_ok']}")

    # Step 2: Fetch universe
    print(f"\n[2/3] Fetching NIFTY 100 universe...")
    df = fetch_all(verbose=False)
    print(f"  Fetched: {len(df)} stocks")

    if df.empty:
        print("No data. Exiting.")
        return

    # Step 3: Generate signals
    print("\n[3/3] Generating signals...")
    signals = []
    for _, row in df.iterrows():
        signal, confidence = generate_signal(row, regime)
        signals.append({
            "Stock":      row["Stock"],
            "Signal":     signal,
            "Confidence": confidence,
            "Close":      row["Close"],
            "EMA20":      row["EMA20"],
            "EMA50":      row["EMA50"],
            "RSI14":      row["RSI14"],
            "VolRatio":   row["VolRatio"],
            "ATR14":      row["ATR14"],
            "Regime":     regime,
            "Date":       row["Date"],
        })

    result_df = pd.DataFrame(signals)

    # Summary
    buy_count  = len(result_df[result_df["Signal"] == "BUY"])
    sell_count = len(result_df[result_df["Signal"] == "SELL"])
    hold_count = len(result_df[result_df["Signal"] == "HOLD"])
    print(f"\n  BUY: {buy_count} | SELL: {sell_count} | HOLD: {hold_count}")

    # Top BUYs
    buys = result_df[result_df["Signal"] == "BUY"].sort_values(
        "Confidence", ascending=False)
    if not buys.empty:
        print("\n  Top BUY signals:")
        print(buys[["Stock","Confidence","Close","RSI14","VolRatio"]].head(10).to_string(index=False))

    # Save
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data/signals", exist_ok=True)
    dated_path = f"data/signals/{today}.csv"
    result_df.to_csv(dated_path, index=False)
    result_df.to_csv("data/latest_signal.csv", index=False)

    # Save regime
    os.makedirs("data/regime", exist_ok=True)
    pd.DataFrame([r]).to_csv(f"data/regime/{today}.csv", index=False)
    pd.DataFrame([r]).to_csv("data/latest_regime.csv", index=False)

    print(f"\n  Saved: {dated_path}")
    print(f"  Saved: data/latest_signal.csv")
    print("=== Done ===")

if __name__ == "__main__":
    run()