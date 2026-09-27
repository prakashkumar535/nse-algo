"""
Telegram Alert System
=====================
Sends alerts for:
- Regime change (BEAR → BULL)
- H5 signal fired
- Daily summary
- Paper trade reminders
"""

import os
import requests
import pandas as pd
from datetime import datetime

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

def send_message(text: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("  Telegram not configured.")
        return False
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, data=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  Telegram error: {e}")
        return False

def alert_regime_change(old_regime: str, new_regime: str, nifty: float):
    emoji = "🟢" if "BULL" in new_regime else "🔴"
    msg = (
        f"{emoji} <b>PulseAlgo — REGIME CHANGE</b>\n\n"
        f"<b>{old_regime}</b> → <b>{new_regime}</b>\n"
        f"NIFTY: {nifty:,.0f}\n\n"
    )
    if "BULL" in new_regime:
        msg += (
            "✅ BUY signals now ACTIVE\n"
            "👀 Monitor H5 watchlist\n"
            "📊 Check momentum rankings\n\n"
            "<i>Paper trade only — not live yet</i>"
        )
    else:
        msg += (
            "❌ BUY signals BLOCKED\n"
            "💰 Cash is a position\n"
            "📉 No new entries"
        )
    return send_message(msg)

def alert_h5_signal(stocks: list[dict]):
    if not stocks:
        return False
    msg = "🚀 <b>PulseAlgo — H5 SIGNAL FIRED</b>\n\n"
    msg += "52-Week High Breakout candidates:\n\n"
    for s in stocks[:5]:
        msg += (
            f"📌 <b>{s.get('Stock', '—')}</b>\n"
            f"   Close: ₹{s.get('Close', 0):,.2f}\n"
            f"   RSI: {s.get('RSI14', 0):.1f} | "
            f"Vol Ratio: {s.get('VolRatio', 0):.1f}x\n\n"
        )
    msg += (
        "⚠️ <i>Verify H5 checklist before paper trading:\n"
        "- Close above 52w high\n"
        "- Volume > 2x avg\n"
        "- RSI 55-75\n"
        "- Regime = BULL</i>"
    )
    return send_message(msg)

def alert_daily_summary(
    regime:     str,
    buy_count:  int,
    sell_count: int,
    hold_count: int,
    nifty:      float,
    top_buys:   list[dict] = [],
):
    emoji = "🟢" if "BULL" in regime else "🔴"
    msg = (
        f"📈 <b>PulseAlgo Daily Summary</b>\n"
        f"{datetime.now().strftime('%d %b %Y')}\n\n"
        f"{emoji} Regime: <b>{regime}</b>\n"
        f"NIFTY: {nifty:,.0f}\n\n"
        f"Signals (NIFTY 100):\n"
        f"  🟢 BUY  : {buy_count}\n"
        f"  🔴 SELL : {sell_count}\n"
        f"  🟡 HOLD : {hold_count}\n"
    )
    if top_buys:
        msg += "\n<b>Top BUY signals:</b>\n"
        for s in top_buys[:3]:
            msg += (
                f"  • {s.get('Stock','—')} "
                f"@ ₹{s.get('Close',0):,.2f} "
                f"(conf: {s.get('Confidence',0)})\n"
            )
    if buy_count == 0:
        msg += "\n<i>No BUY signals — regime blocking entries</i>"
    return send_message(msg)

def alert_paper_trade_reminder(open_trades: int):
    if open_trades == 0:
        return False
    msg = (
        f"⏰ <b>PulseAlgo — Open Positions Reminder</b>\n\n"
        f"You have <b>{open_trades}</b> open paper trade(s).\n\n"
        f"Check your positions:\n"
        f"- Any exit signals?\n"
        f"- Regime still BULL?\n"
        f"- Stop levels breached?\n\n"
        f"<i>Review dashboard for details.</i>"
    )
    return send_message(msg)

def load_regime_cache() -> str:
    path = "data/latest_regime.csv"
    if not os.path.exists(path):
        return "UNKNOWN"
    try:
        df = pd.read_csv(path)
        return str(df.iloc[0]["regime"])
    except:
        return "UNKNOWN"

def save_regime_cache(regime: str):
    os.makedirs("data", exist_ok=True)
    pd.DataFrame([{"regime": regime, "date": datetime.now().strftime("%Y-%m-%d")}])\
      .to_csv("data/regime_cache.csv", index=False)

def load_previous_regime() -> str:
    path = "data/regime_cache.csv"
    if not os.path.exists(path):
        return "UNKNOWN"
    try:
        df = pd.read_csv(path)
        return str(df.iloc[0]["regime"])
    except:
        return "UNKNOWN"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("Testing Telegram connection...")
    ok = send_message(
        "🤖 <b>PulseAlgo Bot Connected!</b>\n\n"
        "Alert system is live.\n"
        "You will receive:\n"
        "- 📊 Daily market summary\n"
        "- 🚨 Regime change alerts\n"
        "- 🚀 H5 signal alerts\n"
        "- ⏰ Open position reminders\n\n"
        "<i>Market is currently BEAR — monitoring only.</i>"
    )
    print("Sent!" if ok else "Failed — check token in environment.")