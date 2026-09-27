import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(
    page_title="PulseAlgo",
    page_icon="📈",
    layout="wide"
)

# ─── Header ─────────────────────────────────────────────────────
st.title("📈 PulseAlgo — NSE Research Dashboard")
st.caption("Quantitative research platform · NIFTY 100 universe · Evidence-based signals")

# ─── Helpers ────────────────────────────────────────────────────
def color_signal(val):
    if val == "BUY":
        return "background-color:#1a3a1a; color:#3fb950; font-weight:bold"
    elif val == "SELL":
        return "background-color:#3a1a1a; color:#f85149; font-weight:bold"
    return "background-color:#2a2200; color:#d29922; font-weight:bold"

def color_confidence(val):
    if val >= 8:  return "color:#3fb950; font-weight:bold"
    elif val >= 5: return "color:#d29922"
    return "color:#f85149"

def regime_color(regime):
    colors = {
        "STRONG_BULL": ("#1a3a1a", "#3fb950"),
        "BULL":        ("#0d2a0d", "#39d353"),
        "SIDEWAYS":    ("#2a2200", "#d29922"),
        "BEAR":        ("#3a1a1a", "#f85149"),
        "UNKNOWN":     ("#1c2128", "#8b949e"),
    }
    return colors.get(regime, colors["UNKNOWN"])

# ─── Load Data ──────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_signals():
    path = "data/latest_signal.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None

@st.cache_data(ttl=300)
def load_regime():
    path = "data/latest_regime.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.iloc[0].to_dict() if not df.empty else None

@st.cache_data(ttl=300)
def load_backtest():
    path = "data/backtest/latest_metrics_train.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

signals_df = load_signals()
regime     = load_regime()
bt_df      = load_backtest()

# ─── TAB LAYOUT ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Market Overview",
    "🔍 H5 Watchlist",
    "📊 Backtest Results",
    "📋 All Signals",
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Market Overview
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Market Regime")

    if regime:
        r         = regime.get("regime", "UNKNOWN")
        bg, fg    = regime_color(r)
        buy_ok    = regime.get("buy_ok", False)
        nifty_val = regime.get("close", "—")
        ema50     = regime.get("ema50", "—")
        ema200    = regime.get("ema200", "—")
        r_date    = regime.get("date", "—")

        st.markdown(
            f"""
            <div style="background:{bg}; border:1px solid {fg};
                        border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:1rem;">
                <div style="font-size:1.6rem; font-weight:700; color:{fg}">
                    {r}
                </div>
                <div style="color:#8b949e; font-size:.85rem; margin-top:.3rem">
                    BUY signals allowed: <b style="color:{fg}">{'YES ✅' if buy_ok else 'NO ❌'}</b>
                    &nbsp;·&nbsp; As of {r_date}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("NIFTY 50", f"{nifty_val:,.0f}" if isinstance(nifty_val, float) else nifty_val)
        col2.metric("EMA 50",   f"{ema50:,.0f}"  if isinstance(ema50,  float) else ema50)
        col3.metric("EMA 200",  f"{ema200:,.0f}" if isinstance(ema200, float) else ema200)

        st.divider()
        st.subheader("Regime Guide")
        guide = {
            "STRONG_BULL": "NIFTY above EMA50 > EMA200, trending up strongly. All signals active.",
            "BULL":        "NIFTY above EMA200. Momentum signals active. Stay selective.",
            "SIDEWAYS":    "NIFTY near EMA200. No BUY signals. Reduce exposure.",
            "BEAR":        "NIFTY below EMA200. NO BUY signals. Cash is a position.",
        }
        for reg, desc in guide.items():
            bg2, fg2 = regime_color(reg)
            active = "◀ CURRENT" if reg == r else ""
            st.markdown(
                f"""<div style="background:{bg2}; border:1px solid {fg2};
                    border-radius:6px; padding:.6rem 1rem; margin:.3rem 0;
                    font-size:.88rem;">
                    <b style="color:{fg2}">{reg}</b> {active} — {desc}
                </div>""",
                unsafe_allow_html=True
            )
    else:
        st.warning("No regime data found. Run `python src/signal_engine.py` first.")

    # Signal summary
    if signals_df is not None:
        st.divider()
        st.subheader("Today's Signal Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Universe",     len(signals_df))
        c2.metric("BUY signals",  len(signals_df[signals_df["Signal"] == "BUY"]))
        c3.metric("SELL signals", len(signals_df[signals_df["Signal"] == "SELL"]))
        c4.metric("HOLD signals", len(signals_df[signals_df["Signal"] == "HOLD"]))

# ════════════════════════════════════════════════════════════════
# TAB 2 — H5 Watchlist (52w High Breakout Candidates)
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("H5 — 52-Week High Breakout Watchlist")
    st.caption(
        "Stocks approaching or breaking 52-week highs with strong volume. "
        "H5 is our best validated hypothesis (in-sample Sharpe 0.88, Win Rate 56.8%). "
        "Only act when Regime = BULL or STRONG_BULL."
    )

    if signals_df is not None:
        df = signals_df.copy()

        # Need 52w high — compute from Close as proxy using available data
        # Show stocks where Close > EMA50 > EMA20 as momentum candidates
        if "EMA20" in df.columns and "EMA50" in df.columns:
            # H5 proxy: uptrend + high vol
            h5_candidates = df[
                (df["Close"] > df["EMA20"]) &
                (df["EMA20"] > df["EMA50"]) &
                (df["VolRatio"] >= 1.2)
            ].copy()

            h5_candidates["Trend"] = "↑ Uptrend"
            h5_candidates["H5_Ready"] = h5_candidates["VolRatio"] >= 1.5

            if not h5_candidates.empty:
                h5_candidates = h5_candidates.sort_values("VolRatio", ascending=False)

                if regime:
                    buy_ok = regime.get("buy_ok", False)
                    if buy_ok:
                        st.success("✅ Regime allows BUY — H5 signals are active")
                    else:
                        st.error(f"❌ Regime is {regime.get('regime')} — H5 signals BLOCKED. Monitor only, do not trade.")

                st.markdown(f"**{len(h5_candidates)} candidates found:**")

                display_cols = [c for c in [
                    "Stock", "Close", "EMA20", "EMA50",
                    "RSI14", "VolRatio", "ATR14", "Date"
                ] if c in h5_candidates.columns]

                st.dataframe(
                    h5_candidates[display_cols].reset_index(drop=True),
                    use_container_width=True
                )

                st.divider()
                st.markdown("**H5 Entry Checklist** (verify before any paper trade):")
                st.markdown("""
                - [ ] Close above previous 52-week high
                - [ ] Volume > 2x 20-day average
                - [ ] RSI between 55–75
                - [ ] EMA20 > EMA50
                - [ ] NIFTY regime = BULL or STRONG_BULL
                - [ ] No major news/results driving the move (avoid event-driven)
                - [ ] Paper trade only — not live trading yet
                """)
            else:
                st.info("No H5 candidates today. Market may be in downtrend.")
        else:
            st.warning("Signal data incomplete. Re-run signal engine.")
    else:
        st.warning("No signal data. Run `python src/signal_engine.py` first.")

# ════════════════════════════════════════════════════════════════
# TAB 3 — Backtest Results
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Backtest Research Results")
    st.caption("In-sample results only. OOS validation pending next bull market cycle.")

    if bt_df is not None:
        for _, row in bt_df.iterrows():
            hyp    = row.get("Hypothesis", "Unknown")
            sharpe = row.get("Sharpe", 0)
            wr     = row.get("Win Rate %", 0)
            trades = row.get("Trades", 0)

            passed = sharpe >= 0.5 and wr >= 55 and trades >= 50
            status = "✅ PASSES criteria" if passed else "❌ FAILS criteria"
            color  = "#1a3a1a" if passed else "#3a1a1a"
            fg     = "#3fb950" if passed else "#f85149"

            st.markdown(
                f"""<div style="background:{color}; border:1px solid {fg};
                    border-radius:8px; padding:1rem; margin:.5rem 0;">
                    <b style="color:{fg}; font-size:1rem">{hyp}</b>
                    <span style="color:#8b949e; float:right">{status}</span>
                </div>""",
                unsafe_allow_html=True
            )

            cols = st.columns(5)
            cols[0].metric("Trades",        int(trades))
            cols[1].metric("Win Rate",       f"{wr}%")
            cols[2].metric("Sharpe",         f"{sharpe}")
            cols[3].metric("Profit Factor",  f"{row.get('Profit Factor', '—')}")
            cols[4].metric("Avg Net Ret",    f"{row.get('Avg Net Ret %', '—')}%")
            st.divider()

        st.markdown("""
        **Research Summary:**
        - H1 (Volume Surge): ❌ Rejected — negative expectancy
        - H2 (RSI Momentum): ❌ Rejected — OOS collapse
        - H3 (EMA Align): ❌ Rejected — catastrophic drawdown
        - H4 (Mean Revert): ❌ Rejected — low win rate
        - **H5 (52w Breakout): ⚠️ Promising — in-sample validated, OOS pending**

        **Next step:** Paper trade H5 during next BULL regime.
        Minimum 20 paper trades before considering live capital.
        """)
    else:
        st.info("No backtest results yet. Run `python src/backtest.py` first.")

# ════════════════════════════════════════════════════════════════
# TAB 4 — All Signals
# ════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("All Signals — NIFTY 100 Universe")

    if signals_df is not None:
        date_val = signals_df["Date"].iloc[0] if "Date" in signals_df.columns else "—"
        st.caption(f"Last updated: {date_val} · {len(signals_df)} stocks")

        # Filter
        col1, col2 = st.columns(2)
        sig_filter = col1.multiselect(
            "Filter by Signal",
            ["BUY", "SELL", "HOLD"],
            default=["BUY", "SELL", "HOLD"]
        )
        min_vol = col2.slider("Min Volume Ratio", 0.0, 3.0, 0.0, 0.1)

        filtered = signals_df[
            (signals_df["Signal"].isin(sig_filter)) &
            (signals_df["VolRatio"] >= min_vol)
        ].copy()

        display_cols = [c for c in [
            "Stock", "Signal", "Confidence", "Close",
            "EMA20", "EMA50", "RSI14", "VolRatio", "ATR14", "Regime", "Date"
        ] if c in filtered.columns]

        styled = filtered[display_cols].style.map(
            color_signal, subset=["Signal"]
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.warning("No signal data. Run `python src/signal_engine.py` first.")