import pandas as pd
import os
from datetime import datetime
from fetch_data import fetch_all

def generate_signal(row):
    score = 0
    signal = "HOLD"

    buy_conditions = [
        row["Close"] > row["EMA20"],
        40 <= row["RSI14"] <= 60,
        row["VolRatio"] >= 1.5
    ]
    sell_conditions = [
        row["Close"] < row["EMA20"],
        row["RSI14"] > 70
    ]

    buy_score = sum(buy_conditions)
    sell_score = sum(sell_conditions)

    if buy_score >= 2:
        signal = "BUY"
        score = round((buy_score / 3) * 10)
    elif sell_score >= 2:
        signal = "SELL"
        score = round((sell_score / 2) * 10)
    else:
        signal = "HOLD"
        score = 5

    return signal, score

def run():
    print("Fetching data...")
    df = fetch_all()

    if df.empty:
        print("No data fetched. Exiting.")
        return

    signals = []
    for _, row in df.iterrows():
        signal, confidence = generate_signal(row)
        signals.append({
            "Stock": row["Stock"],
            "Signal": signal,
            "Confidence": confidence,
            "Close": row["Close"],
            "EMA20": row["EMA20"],
            "RSI14": row["RSI14"],
            "VolRatio": row["VolRatio"],
            "Date": row["Date"]
        })

    result_df = pd.DataFrame(signals)
    print(result_df)

    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("data/signals", exist_ok=True)
    dated_path = f"data/signals/{today}.csv"
    result_df.to_csv(dated_path, index=False)
    print(f"Saved: {dated_path}")

    result_df.to_csv("data/latest_signal.csv", index=False)
    print("Updated: data/latest_signal.csv")

if __name__ == "__main__":
    run()