"""
EMA9 Trend Following Strategy
==============================
Category  : Investment / Swing
Timeframe : Daily
Direction : Long only

Hypothesis:
    When a stock crosses above its 9-period EMA after being below it,
    and optional trend filters confirm the broader uptrend,
    the probability of a positive return over the next N days
    may exceed a random baseline.

Entry Signal (T close):
    Previous Close <= Previous EMA9
    AND Current Close > Current EMA9

Execution: T+1 Open (no look-ahead bias)

Exit Signal (T close):
    Current Close < Current EMA9

Execution: T+1 Open

Filters (each independently toggleable):
    A: Close > EMA50 AND EMA50 > EMA200
    B: Close > EMA50 AND EMA50 > EMA50[20]
    C: Close > EMA200 AND EMA200 rising
    D: NIFTY50 Close > NIFTY50 EMA200

Status: Research → Backtesting
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from strategies.base import BaseStrategy, StrategyConfig, StrategyStatus
from dataclasses import dataclass


# ─── EMA9 Config ────────────────────────────────────────────────

@dataclass
class EMA9Config(StrategyConfig):
    """EMA9-specific configuration — all parameters toggleable."""

    # Strategy identity
    name:        str = "EMA9 Trend"
    version:     str = "1.0.0"
    status:      str = StrategyStatus.BACKTESTING
    category:    str = "investment"
    description: str = (
        "Long-only EMA9 crossover with configurable trend filters. "
        "Entry on close crossing above EMA9. Exit on close below EMA9."
    )

    # EMA periods (configurable)
    ema_entry:   int = 9
    ema_exit:    int = 9
    ema_fast:    int = 50
    ema_slow:    int = 200

    # Filter toggles
    filter_a:    bool = True   # Close > EMA50 AND EMA50 > EMA200
    filter_b:    bool = False  # Close > EMA50 AND EMA50 > EMA50[20]
    filter_c:    bool = False  # Close > EMA200 AND EMA200 rising
    filter_d:    bool = True   # NIFTY regime filter

    # Filter B lookback
    filter_b_lookback: int = 20

    # Min holding days (avoid whipsaws)
    min_hold_days: int = 5


# ─── Feature Calculation ────────────────────────────────────────

def add_ema9_features(df: pd.DataFrame, config: EMA9Config) -> pd.DataFrame:
    """Add all required features for EMA9 strategy."""
    df = df.copy()

    close = df["Close"]

    df[f"EMA{config.ema_entry}"]  = close.ewm(
        span=config.ema_entry, adjust=False).mean()
    df[f"EMA{config.ema_fast}"]   = close.ewm(
        span=config.ema_fast,  adjust=False).mean()
    df[f"EMA{config.ema_slow}"]   = close.ewm(
        span=config.ema_slow,  adjust=False).mean()

    # For filter B — EMA50 N days ago
    df[f"EMA{config.ema_fast}_lag"] = df[
        f"EMA{config.ema_fast}"
    ].shift(config.filter_b_lookback)

    # EMA200 slope for filter C
    df["EMA200_prev5"] = df[f"EMA{config.ema_slow}"].shift(5)

    # Previous values for crossover detection
    df["Close_prev"]       = close.shift(1)
    df[f"EMA{config.ema_entry}_prev"] = df[f"EMA{config.ema_entry}"].shift(1)

    return df.dropna()


def add_nifty_features(
    nifty_df: pd.DataFrame,
    ema_slow:  int = 200,
) -> pd.DataFrame:
    """Add regime features to NIFTY dataframe."""
    df = nifty_df.copy()
    df["NIFTY_EMA200"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
    df["NIFTY_Bull"]   = (df["Close"] > df["NIFTY_EMA200"]).astype(int)
    return df


# ─── EMA9 Strategy ──────────────────────────────────────────────

class EMA9TrendStrategy(BaseStrategy):
    """
    EMA9 crossover trend-following strategy.
    Long only. Daily timeframe. NSE equities.
    """

    def get_default_config(self) -> EMA9Config:
        return EMA9Config()

    def generate_signals(
        self,
        data:     pd.DataFrame,
        config:   EMA9Config = None,
        nifty_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Generate entry/exit signals for one stock.

        Returns DataFrame with columns:
            date, signal (1=buy, -1=sell, 0=hold), notes
        """
        cfg = config or self.config
        if not isinstance(cfg, EMA9Config):
            cfg = EMA9Config(**{
                k: v for k, v in cfg.to_dict().items()
                if hasattr(EMA9Config(), k)
            })

        df = add_ema9_features(data, cfg)
        if len(df) < 30:
            return pd.DataFrame()

        # ── Nifty regime ────────────────────────────────────────
        nifty_bull = pd.Series(True, index=df.index)
        if cfg.filter_d and nifty_df is not None:
            nifty_feat = add_nifty_features(nifty_df, cfg.ema_slow)
            nifty_aligned = nifty_feat["NIFTY_Bull"].reindex(
                df.index, method="ffill"
            ).fillna(0)
            nifty_bull = nifty_aligned == 1

        ema_entry_col = f"EMA{cfg.ema_entry}"
        ema_fast_col  = f"EMA{cfg.ema_fast}"
        ema_slow_col  = f"EMA{cfg.ema_slow}"

        # ── Entry conditions ────────────────────────────────────
        cross_above = (
            (df["Close_prev"] <= df[f"{ema_entry_col}_prev"]) &
            (df["Close"]      >  df[ema_entry_col])
        )

        # Filter A: Close > EMA50 AND EMA50 > EMA200
        filter_a = pd.Series(True, index=df.index)
        if cfg.filter_a:
            filter_a = (
                (df["Close"]       > df[ema_fast_col]) &
                (df[ema_fast_col]  > df[ema_slow_col])
            )

        # Filter B: EMA50 > EMA50[20]
        filter_b = pd.Series(True, index=df.index)
        if cfg.filter_b:
            filter_b = df[ema_fast_col] > df[f"{ema_fast_col}_lag"]

        # Filter C: Close > EMA200 AND EMA200 rising
        filter_c = pd.Series(True, index=df.index)
        if cfg.filter_c:
            filter_c = (
                (df["Close"]      > df[ema_slow_col]) &
                (df[ema_slow_col] > df["EMA200_prev5"])
            )

        # ── Entry signal ────────────────────────────────────────
        buy_signal = (
            cross_above &
            filter_a &
            filter_b &
            filter_c &
            nifty_bull
        )

        # ── Exit condition ──────────────────────────────────────
        exit_signal = df["Close"] < df[ema_entry_col]

        # ── Build signal series ─────────────────────────────────
        signals = []
        in_trade    = False
        entry_date  = None
        hold_count  = 0

        for date, row in df.iterrows():
            sig   = 0
            notes = ""

            if not in_trade:
                if buy_signal.loc[date]:
                    sig       = 1
                    in_trade  = True
                    entry_date = date
                    hold_count = 0
                    notes     = (
                        f"EMA{cfg.ema_entry} crossover"
                        f"{' | FilterA' if cfg.filter_a else ''}"
                        f"{' | FilterD' if cfg.filter_d else ''}"
                    )
            else:
                hold_count += 1
                if exit_signal.loc[date] and hold_count >= cfg.min_hold_days:
                    sig      = -1
                    in_trade = False
                    notes    = f"Close < EMA{cfg.ema_exit} | held {hold_count}d"

            signals.append({
                "date":   date,
                "signal": sig,
                "close":  row["Close"],
                f"ema{cfg.ema_entry}": row[ema_entry_col],
                "notes":  notes,
            })

        result = pd.DataFrame(signals)
        result["date"] = pd.to_datetime(result["date"])
        return result


# ─── Parameter Sensitivity ──────────────────────────────────────

def parameter_sensitivity(
    data_dict:  dict,
    nifty_df:   pd.DataFrame,
    ema_range:  list = [5, 9, 10, 20, 21],
    hold_days:  list = [5, 9, 20],
    mode:       str  = "train",
) -> pd.DataFrame:
    """
    Test EMA9 strategy across parameter ranges.
    Identifies stable regions, not just peak performance.
    DO NOT use to select the single best parameter.
    """
    from backtest.engine import BacktestEngine

    results = []

    for ema in ema_range:
        cfg = EMA9Config(
            ema_entry=ema,
            ema_exit=ema,
        )
        strategy = EMA9TrendStrategy(config=cfg)
        engine   = BacktestEngine(
            strategy=strategy,
            data=data_dict,
            nifty_df=nifty_df,
            config=cfg,
        )
        metrics_dict = engine.run(modes=[mode])
        if mode in metrics_dict:
            m = metrics_dict[mode]
            results.append({
                "EMA":           ema,
                "Trades":        m.total_trades,
                "Win Rate %":    m.win_rate_pct,
                "CAGR %":        m.cagr_pct,
                "Max DD %":      m.max_drawdown_pct,
                "Sharpe":        m.sharpe,
                "Profit Factor": m.profit_factor,
            })

    df = pd.DataFrame(results)
    print("\n=== PARAMETER SENSITIVITY MATRIX ===")
    print(df.to_string(index=False))
    print("\nObjective: identify STABLE regions, not peak values.")
    print("If one EMA stands out as best — treat with suspicion.")
    return df


if __name__ == "__main__":
    # Quick smoke test
    import yfinance as yf

    print("Loading test data...")
    ticker   = "RELIANCE.NS"
    df       = yf.download(ticker, period="5y", interval="1d",
                           progress=False, auto_adjust=True)
    nifty_df = yf.download("^NSEI", period="5y", interval="1d",
                           progress=False, auto_adjust=True)

    for d in [df, nifty_df]:
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

    strategy = EMA9TrendStrategy()
    print(strategy.describe())

    signals = strategy.generate_signals(
        data=df,
        nifty_df=nifty_df,
    )
    buy_count  = (signals["signal"] ==  1).sum()
    sell_count = (signals["signal"] == -1).sum()
    print(f"\nSignals on {ticker}:")
    print(f"  BUY  signals: {buy_count}")
    print(f"  SELL signals: {sell_count}")
    print(signals[signals["signal"] != 0].head(10))