"""
Cross-Sectional Momentum Strategy
===================================
Category  : Investment
Timeframe : Monthly rebalance
Direction : Long only

Hypothesis:
    Stocks that have outperformed their peers over the past
    3, 6, and 12 months tend to continue outperforming
    over the next 1-3 months (momentum persistence).

    This is NOT a price indicator strategy.
    This is cross-sectional RANKING — we compare stocks
    against each other, not against fixed thresholds.

Methodology:
    1. At end of each month, calculate momentum scores
       for all stocks in universe
    2. Rank stocks by composite momentum score
    3. Buy top N stocks (equal weight)
    4. Hold for one month
    5. Rebalance — sell laggards, buy new leaders

Momentum Score:
    score = w3 * ret_3m + w6 * ret_6m + w12 * ret_12m

    Default weights: 25% / 25% / 50%
    (12M return weighted more — strongest academic evidence)

    Skip last 1 month (ret_1m) to avoid short-term reversal.

Academic basis:
    Jegadeesh & Titman (1993) — original momentum paper
    Adapted for NSE India by multiple SEBI research papers
    Effect documented across emerging markets

Status: Research → Backtesting
"""

import pandas as pd
import numpy as np
import sys
import os
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from strategies.base import BaseStrategy, StrategyConfig, StrategyStatus


# ─── Momentum Config ────────────────────────────────────────────

@dataclass
class MomentumConfig(StrategyConfig):
    """Cross-sectional momentum configuration."""

    # Identity
    name:        str = "Cross-Sectional Momentum"
    version:     str = "1.0.0"
    status:      str = StrategyStatus.BACKTESTING
    category:    str = "investment"
    description: str = (
        "Monthly rebalanced cross-sectional momentum. "
        "Ranks NIFTY 100 stocks by 3M/6M/12M returns. "
        "Holds top N stocks equal-weighted."
    )

    # Momentum lookback periods (in trading days)
    lookback_3m:  int  = 63    # ~3 months
    lookback_6m:  int  = 126   # ~6 months
    lookback_12m: int  = 252   # ~12 months
    skip_days:    int  = 21    # skip last 1 month (reversal avoidance)

    # Weights for composite score
    weight_3m:   float = 0.25
    weight_6m:   float = 0.25
    weight_12m:  float = 0.50

    # Which lookbacks to use (toggleable)
    use_3m:  bool = True
    use_6m:  bool = True
    use_12m: bool = True

    # Portfolio
    top_n:          int  = 10     # hold top N stocks
    rebalance_days: int  = 21     # rebalance every N trading days

    # Filters
    min_price:      float = 50.0  # exclude penny stocks
    min_volume:     int   = 100000 # minimum avg daily volume

    # Regime filter
    filter_d:       bool  = True   # NIFTY above EMA200


# ─── Momentum Engine ────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    """
    Cross-sectional momentum strategy.
    Requires multi-stock data — cannot run on single ticker.
    Run via MomentumBacktester, not BacktestEngine directly.
    """

    def get_default_config(self) -> MomentumConfig:
        return MomentumConfig()

    def generate_signals(
        self,
        data:     pd.DataFrame,
        config:   MomentumConfig = None,
        nifty_df: pd.DataFrame   = None,
    ) -> pd.DataFrame:
        """Single-stock signals — not primary interface for momentum."""
        return pd.DataFrame()

    def score_stock(
        self,
        df:     pd.DataFrame,
        date:   pd.Timestamp,
        config: MomentumConfig,
    ) -> dict | None:
        """
        Calculate momentum score for one stock at one date.
        Returns None if insufficient data.
        """
        try:
            hist = df[df.index <= date]
            if len(hist) < config.lookback_12m + config.skip_days + 5:
                return None

            close_now  = float(hist["Close"].iloc[-1])
            close_skip = float(hist["Close"].iloc[-(config.skip_days + 1)])

            if close_now < config.min_price:
                return None

            # Returns skip last 1M to avoid reversal
            def ret(lookback):
                idx = -(lookback + config.skip_days)
                if abs(idx) > len(hist):
                    return None
                past = float(hist["Close"].iloc[idx])
                if past <= 0:
                    return None
                return (close_skip - past) / past

            r3  = ret(config.lookback_3m)  if config.use_3m  else None
            r6  = ret(config.lookback_6m)  if config.use_6m  else None
            r12 = ret(config.lookback_12m) if config.use_12m else None

            # Composite score
            score  = 0.0
            w_sum  = 0.0
            if r3  is not None: score += config.weight_3m  * r3;  w_sum += config.weight_3m
            if r6  is not None: score += config.weight_6m  * r6;  w_sum += config.weight_6m
            if r12 is not None: score += config.weight_12m * r12; w_sum += config.weight_12m

            if w_sum == 0:
                return None

            score /= w_sum  # normalize

            return {
                "close":  close_now,
                "ret_3m":  round(r3  * 100, 2) if r3  is not None else None,
                "ret_6m":  round(r6  * 100, 2) if r6  is not None else None,
                "ret_12m": round(r12 * 100, 2) if r12 is not None else None,
                "score":   round(score * 100, 4),
            }
        except Exception as e:
            return None


# ─── Momentum Backtester ────────────────────────────────────────

class MomentumBacktester:
    """
    Portfolio-level momentum backtester.
    Handles rebalancing, portfolio construction, and metrics.
    """

    def __init__(
        self,
        data_dict:  dict,           # {ticker: OHLCV DataFrame}
        nifty_df:   pd.DataFrame,
        config:     MomentumConfig = None,
    ):
        self.data     = data_dict
        self.nifty_df = nifty_df
        self.config   = config or MomentumConfig()
        self.strategy = MomentumStrategy(config=self.config)

    def get_regime(self, date: pd.Timestamp) -> bool:
        """Check if NIFTY is in bull regime at given date."""
        if not self.config.filter_d or self.nifty_df is None:
            return True
        hist = self.nifty_df[self.nifty_df.index <= date]
        if len(hist) < 200:
            return True
        ema200 = hist["Close"].ewm(span=200, adjust=False).mean().iloc[-1]
        return float(hist["Close"].iloc[-1]) > float(ema200)

    def get_rebalance_dates(
        self,
        start: pd.Timestamp,
        end:   pd.Timestamp,
    ) -> list:
        """Generate monthly rebalance dates."""
        all_dates = set()
        for df in self.data.values():
            if df is not None:
                all_dates.update(df.index.tolist())
        dates = sorted([d for d in all_dates if start <= d <= end])

        rebalance = []
        for i in range(0, len(dates), self.config.rebalance_days):
            rebalance.append(dates[i])
        return rebalance

    def rank_universe(
        self,
        date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Rank all stocks by momentum score at given date."""
        scores = []
        for ticker, df in self.data.items():
            if df is None:
                continue
            result = self.strategy.score_stock(df, date, self.config)
            if result is not None:
                result["ticker"] = ticker
                result["date"]   = date
                scores.append(result)

        if not scores:
            return pd.DataFrame()

        ranked = pd.DataFrame(scores).sort_values(
            "score", ascending=False
        ).reset_index(drop=True)
        ranked["rank"] = ranked.index + 1
        return ranked

    def run(
        self,
        mode: str = "train",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full momentum backtest.
        Returns (trades_df, portfolio_history_df)
        """
        cfg = self.config

        # Get date range
        all_dates = set()
        for df in self.data.values():
            if df is not None:
                all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if len(dates) < cfg.lookback_12m + cfg.skip_days + cfg.rebalance_days:
            print("  Insufficient data for momentum backtest.")
            return pd.DataFrame(), pd.DataFrame()

        # Split
        n     = len(dates)
        t_end = int(n * cfg.train_pct)
        v_end = int(n * (cfg.train_pct + cfg.validation_pct))

        if mode == "train":
            date_range = dates[cfg.lookback_12m:t_end]
        elif mode == "validation":
            date_range = dates[t_end:v_end]
        elif mode == "oos":
            date_range = dates[v_end:]
        else:
            date_range = dates[cfg.lookback_12m:]

        if len(date_range) < cfg.rebalance_days * 2:
            print(f"  Insufficient dates for mode={mode}")
            return pd.DataFrame(), pd.DataFrame()

        start = date_range[0]
        end   = date_range[-1]

        rebalance_dates = self.get_rebalance_dates(start, end)
        print(f"  Mode: {mode.upper()} | "
              f"{str(start)[:10]} → {str(end)[:10]} | "
              f"Rebalances: {len(rebalance_dates)}")

        trades        = []
        portfolio     = {}   # {ticker: entry_price}
        portfolio_history = []

        for i, rb_date in enumerate(rebalance_dates):
            # Check regime
            regime_ok = self.get_regime(rb_date)
            if not regime_ok:
                # Exit all positions
                for ticker, entry in portfolio.items():
                    df = self.data.get(ticker)
                    if df is not None:
                        future = df[df.index >= rb_date]
                        exit_p = float(future.iloc[0]["Close"]) \
                            if not future.empty else entry
                        trades.append({
                            "ticker":      ticker,
                            "entry_date":  portfolio_entries.get(ticker, rb_date),
                            "exit_date":   rb_date,
                            "entry_price": entry,
                            "exit_price":  exit_p,
                            "gross_ret":   (exit_p - entry) / entry * 100,
                            "net_ret":     ((exit_p - entry) / entry - 0.004) * 100,
                            "exit_reason": "REGIME_EXIT",
                        })
                portfolio = {}
                portfolio_entries = {}

                portfolio_history.append({
                    "date":     rb_date,
                    "holdings": 0,
                    "regime":   "BEAR",
                    "top_stock": None,
                })
                continue

            # Rank universe
            ranked = self.rank_universe(rb_date)
            if ranked.empty:
                continue

            top_n    = ranked.head(cfg.top_n)
            top_set  = set(top_n["ticker"].tolist())
            curr_set = set(portfolio.keys())

            # Sell stocks no longer in top N
            exit_set = curr_set - top_set
            for ticker in exit_set:
                df = self.data.get(ticker)
                entry = portfolio[ticker]
                if df is not None:
                    future = df[df.index > rb_date]
                    exit_p = float(future.iloc[0]["Open"]) \
                        if not future.empty else entry
                else:
                    exit_p = entry

                gross = (exit_p - entry) / entry
                net   = gross - 0.004  # round-trip cost

                trades.append({
                    "ticker":      ticker,
                    "entry_date":  portfolio_entries.get(ticker, rb_date),
                    "exit_date":   rb_date,
                    "entry_price": round(entry, 2),
                    "exit_price":  round(exit_p, 2),
                    "gross_ret":   round(gross * 100, 2),
                    "net_ret":     round(net * 100, 2),
                    "hold_days":   (rb_date - portfolio_entries.get(
                        ticker, rb_date)).days,
                    "exit_reason": "REBALANCE",
                    "rank_at_exit": int(
                        ranked[ranked["ticker"] == ticker]["rank"].values[0]
                    ) if ticker in ranked["ticker"].values else 999,
                })
                del portfolio[ticker]
                if ticker in portfolio_entries:
                    del portfolio_entries[ticker]

            # Buy stocks newly in top N
            buy_set = top_set - curr_set
            if not hasattr(self, '_portfolio_entries'):
                portfolio_entries = {}

            for ticker in buy_set:
                df = self.data.get(ticker)
                if df is None:
                    continue
                future = df[df.index > rb_date]
                if future.empty:
                    continue
                entry_p = float(future.iloc[0]["Open"])
                portfolio[ticker]         = entry_p
                portfolio_entries[ticker] = rb_date

            portfolio_history.append({
                "date":      rb_date,
                "holdings":  len(portfolio),
                "regime":    "BULL",
                "top_stock": ranked.iloc[0]["ticker"] if not ranked.empty else None,
                "top_score": ranked.iloc[0]["score"]  if not ranked.empty else None,
            })

            if (i + 1) % 5 == 0:
                print(f"    Rebalance {i+1}/{len(rebalance_dates)}: "
                      f"{str(rb_date)[:10]} | "
                      f"Holdings: {len(portfolio)} | "
                      f"Regime: {'BULL' if regime_ok else 'BEAR'}")

        # Close remaining positions at end
        if portfolio and date_range:
            last_date = date_range[-1]
            for ticker, entry in portfolio.items():
                df = self.data.get(ticker)
                if df is not None:
                    hist = df[df.index <= last_date]
                    exit_p = float(hist.iloc[-1]["Close"]) \
                        if not hist.empty else entry
                else:
                    exit_p = entry
                gross = (exit_p - entry) / entry
                net   = gross - 0.004
                trades.append({
                    "ticker":      ticker,
                    "entry_date":  portfolio_entries.get(ticker, last_date),
                    "exit_date":   last_date,
                    "entry_price": round(entry, 2),
                    "exit_price":  round(exit_p, 2),
                    "gross_ret":   round(gross * 100, 2),
                    "net_ret":     round(net * 100, 2),
                    "hold_days":   (last_date - portfolio_entries.get(
                        ticker, last_date)).days,
                    "exit_reason": "END_OF_PERIOD",
                })

        trades_df  = pd.DataFrame(trades)
        history_df = pd.DataFrame(portfolio_history)
        return trades_df, history_df

    def print_results(
        self,
        trades_df:  pd.DataFrame,
        history_df: pd.DataFrame,
        mode:       str = "train",
    ):
        """Print standardized results."""
        cfg = self.config
        print(f"\n{'='*55}")
        print(f"  {cfg.name} v{cfg.version} — {mode.upper()}")
        print(f"  Top {cfg.top_n} stocks | "
              f"Rebalance: {cfg.rebalance_days}d | "
              f"Regime filter: {cfg.filter_d}")
        print(f"{'='*55}")

        if trades_df.empty:
            print("  No trades.")
            return

        rets  = trades_df["net_ret"].values
        n     = len(rets)
        wins  = (rets > 0).sum()
        loss  = (rets < 0).sum()

        print(f"\n  TRADING")
        print(f"  Trades      : {n}")
        print(f"  Win Rate    : {wins/n*100:.1f}%")
        print(f"  Avg Net Ret : {rets.mean():.2f}%")
        print(f"  Avg Win     : {rets[rets>0].mean():.2f}%" if wins > 0 else "  Avg Win     : —")
        print(f"  Avg Loss    : {rets[rets<0].mean():.2f}%" if loss > 0 else "  Avg Loss    : —")

        gp = rets[rets > 0].sum()
        gl = rets[rets < 0].sum()
        pf = abs(gp/gl) if gl != 0 else 999
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Expectancy  : {rets.mean():.2f}%")

        if "hold_days" in trades_df.columns:
            print(f"  Avg Hold    : {trades_df['hold_days'].mean():.0f} days")

        # Top performers
        if not trades_df.empty:
            top = trades_df.nlargest(5, "net_ret")[
                ["ticker","entry_date","exit_date","net_ret","exit_reason"]
            ]
            print(f"\n  TOP 5 TRADES:")
            print(top.to_string(index=False))

            worst = trades_df.nsmallest(5, "net_ret")[
                ["ticker","entry_date","exit_date","net_ret","exit_reason"]
            ]
            print(f"\n  WORST 5 TRADES:")
            print(worst.to_string(index=False))

        # Regime distribution
        if not history_df.empty and "regime" in history_df.columns:
            regime_counts = history_df["regime"].value_counts()
            print(f"\n  REGIME DISTRIBUTION:")
            for r, c in regime_counts.items():
                print(f"    {r}: {c} rebalances")