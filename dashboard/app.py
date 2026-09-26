import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="PulseAlgo", page_icon="📈", layout="wide")

st.title("📈 PulseAlgo — NSE Signal Dashboard")

DATA_PATH = "data/latest_signal.csv"

def color_signal(val):
    if val == "BUY":
        return "background-color: #d4edda; color: #155724; font-weight: bold"
    elif val == "SELL":
        return "background-color: #f8d7da; color: #721c24; font-weight: bold"
    else:
        return "background-color: #fff3cd; color: #856404; font-weight: bold"

def color_confidence(val):
    if val >= 8:
        return "color: green; font-weight: bold"
    elif val >= 5:
        return "color: orange"
    else:
        return "color: red"

if not os.path.exists(DATA_PATH):
    st.warning("No signal data found. Run signal_engine.py first.")
else:
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        st.warning("Signal file is empty.")
    else:
        date = df["Date"].iloc[0]
        st.caption(f"Last updated: {date}")

        col1, col2, col3 = st.columns(3)
        col1.metric("BUY signals", len(df[df["Signal"] == "BUY"]))
        col2.metric("SELL signals", len(df[df["Signal"] == "SELL"]))
        col3.metric("HOLD signals", len(df[df["Signal"] == "HOLD"]))

        st.divider()

        styled = df.style.map(color_signal, subset=["Signal"]) \
                 .map(color_confidence, subset=["Confidence"])

        st.dataframe(styled, use_container_width=True)