import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

DELIVERY_DIR = "data/delivery"

def get_nse_delivery(date: datetime) -> pd.DataFrame | None:
    """
    Download NSE equity delivery data for a given date.
    NSE publishes this as a CSV at their archive.
    Format: ddmmyyyy
    """
    date_str = date.strftime("%d%m%Y")
    url = (
        f"https://archives.nseindia.com/products/content/"
        f"sec_bhavdata_full_{date_str}.csv"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"  No delivery data for {date.strftime('%Y-%m-%d')} "
                  f"(HTTP {resp.status_code})")
            return None

        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()

        # Keep only equity series
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"]

        # Standardize columns
        col_map = {
            "SYMBOL":           "Symbol",
            "DELIV_QTY":        "DelivQty",
            "DELIV_PER":        "DelivPct",
            "TTL_TRD_QNTY":     "TotalQty",
            "NO_OF_TRADES":     "Trades",
            "CLOSE_PRICE":      "Close",
        }
        df = df.rename(columns={
            k: v for k, v in col_map.items() if k in df.columns
        })

        df["Date"] = date.strftime("%Y-%m-%d")

        keep = [c for c in ["Symbol","Date","DelivQty","DelivPct",
                             "TotalQty","Trades","Close"] if c in df.columns]
        df = df[keep]
        df["DelivPct"] = pd.to_numeric(df["DelivPct"], errors="coerce")
        df["DelivQty"] = pd.to_numeric(df["DelivQty"], errors="coerce")

        return df

    except Exception as e:
        print(f"  Error fetching delivery data: {e}")
        return None

def download_recent_delivery(days_back=10):
    """Download last N trading days of delivery data."""
    os.makedirs(DELIVERY_DIR, exist_ok=True)

    saved = 0
    date  = datetime.now()

    for _ in range(days_back * 2):  # extra iterations to skip weekends
        date -= timedelta(days=1)

        # Skip weekends
        if date.weekday() >= 5:
            continue

        date_str  = date.strftime("%Y-%m-%d")
        file_path = f"{DELIVERY_DIR}/{date_str}.csv"

        if os.path.exists(file_path):
            print(f"  Already exists: {date_str}")
            saved += 1
        else:
            print(f"  Downloading: {date_str}...")
            df = get_nse_delivery(date)
            if df is not None and not df.empty:
                df.to_csv(file_path, index=False)
                print(f"  Saved: {file_path} ({len(df)} stocks)")
                saved += 1

        if saved >= days_back:
            break

    print(f"\nDownloaded {saved} days of delivery data.")

def load_delivery(date_str: str) -> pd.DataFrame | None:
    """Load delivery data for a specific date."""
    path = f"{DELIVERY_DIR}/{date_str}.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def get_latest_delivery() -> pd.DataFrame | None:
    """Get most recent available delivery data."""
    if not os.path.exists(DELIVERY_DIR):
        return None
    files = sorted([
        f for f in os.listdir(DELIVERY_DIR)
        if f.endswith(".csv")
    ])
    if not files:
        return None
    return pd.read_csv(f"{DELIVERY_DIR}/{files[-1]}")

def delivery_signal(symbol: str, days_back=5) -> dict:
    """
    Compute delivery-based signal for a stock.
    Returns avg delivery %, trend, and signal.
    """
    records = []
    date = datetime.now()

    for _ in range(days_back * 3):
        date -= timedelta(days=1)
        if date.weekday() >= 5:
            continue
        df = load_delivery(date.strftime("%Y-%m-%d"))
        if df is None:
            continue
        row = df[df["Symbol"].str.strip() == symbol]
        if not row.empty and "DelivPct" in row.columns:
            records.append(float(row["DelivPct"].values[0]))
        if len(records) >= days_back:
            break

    if not records:
        return {"symbol": symbol, "avg_deliv_pct": None, "signal": "NO_DATA"}

    avg = sum(records) / len(records)
    trend = "RISING" if len(records) >= 2 and records[0] > records[-1] else "FALLING"

    if avg > 60 and trend == "RISING":
        sig = "ACCUMULATION"
    elif avg < 30:
        sig = "DISTRIBUTION"
    else:
        sig = "NEUTRAL"

    return {
        "symbol":        symbol,
        "avg_deliv_pct": round(avg, 2),
        "trend":         trend,
        "signal":        sig,
        "days":          len(records),
    }

if __name__ == "__main__":
    print("Downloading recent NSE delivery data...")
    download_recent_delivery(days_back=5)

    print("\nLatest delivery data sample:")
    df = get_latest_delivery()
    if df is not None:
        print(df.head(10))
        print(f"\nTotal stocks: {len(df)}")

        # High delivery stocks
        if "DelivPct" in df.columns:
            high_del = df[df["DelivPct"] > 70].sort_values(
                "DelivPct", ascending=False)
            print(f"\nHigh delivery stocks (>70%): {len(high_del)}")
            print(high_del[["Symbol","DelivPct","TotalQty"]].head(10))