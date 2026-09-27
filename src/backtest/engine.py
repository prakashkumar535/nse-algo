"""
Backtest Engine
===============
Clean, reusable engine that works with any BaseStrategy.
Supports:
- Train / Validation / OOS splits
- Walk-forward validation
- Realistic transaction costs
- Benchmark comparison
- Full reproducibility via BacktestRun records
"""

import os
import json
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, StrategyConfig, BacktestRun, StrategyStatus
from backtest.costs import PRESETS as COST_PRESETS, CostModel
from backtest.metrics import calculate_metrics, build_equity_curve, BacktestMetrics


# ─── Engine ─────────────────────────────────────────────────────

class BacktestEngine:
    """
    Runs a strategy against historical data.
    Handles splits, costs, metrics, and run storage.
    """

    RUNS_DIR = "runs"

    def __init__(
        self,
        strategy:   BaseStrategy,
        data:       dict[str, pd.DataFrame],   # {ticker: OHLCV DataFrame}
        nifty_df:   Optional[pd.DataFrame] = None,
        config:     Optional[StrategyConfig] = None,
    ):
        self.strategy  = strategy
        self.data      = data
        self.nifty_df  = nifty_df
        self.config    = config or strategy.config
        self.cost_model: CostModel = COST_PRESETS.get(
            self.config.cost_preset, COST_PRESETS["realistic"]
        )
        os.makedirs(self.RUNS_DIR, exist_ok=True)

    # ── Data Splitting ───────────────────────────────────────────

    def split_data(
        self,
        df: pd.DataFrame,
        mode: str = "train",  # train / validation / oos
    ) -> pd.DataFrame:
        """
        Split dataframe by time into train / validation / oos.
        No data leakage — each split is strictly chronological.
        """
        n     = len(df)
        t_end = int(n * self.config.train_pct)
        v_end = int(n * (self.config.train_pct + self.config.validation_pct))

        if mode == "train":
            return df.iloc[:t_end].copy()
        elif mode == "validation":
            return df.iloc[t_end:v_end].copy()
        elif mode == "oos":
            return df.iloc[v_end:].copy()
        else:
            return df.copy()

    # ── Signal Generation ────────────────────────────────────────

    def generate_all_signals(
        self,
        mode: str = "train",
    ) -> pd.DataFrame:
        """
        Run strategy signal generation across all tickers.
        Returns combined signals DataFrame.
        """
        all_signals = []

        for ticker, df in self.data.items():
            if df is None or len(df) < 60:
                continue
            df_part = self.split_data(df, mode=mode)
            if len(df_part) < 30:
                continue
            try:
                signals = self.strategy.generate_signals(
                    data=df_part,
                    config=self.config,
                    nifty_df=self.nifty_df,
                )
                if signals is not None and not signals.empty:
                    signals["ticker"] = ticker
                    all_signals.append(signals)
            except Exception as e:
                print(f"  Signal error {ticker}: {e}")

        if not all_signals:
            return pd.DataFrame()
        return pd.concat(all_signals, ignore_index=True)

    # ── Trade Simulation ─────────────────────────────────────────

    def simulate_trades(
        self,
        signals_df: pd.DataFrame,
        mode:       str = "train",
    ) -> pd.DataFrame:
        """
        Convert signals into trades with realistic execution.
        Entry: T+1 open after signal
        Exit: T+1 open after exit signal
        Applies transaction costs.
        """
        if signals_df.empty:
            return pd.DataFrame()

        trades = []

        for ticker in signals_df["ticker"].unique():
            ticker_sigs = signals_df[
                signals_df["ticker"] == ticker
            ].sort_values("date").reset_index(drop=True)

            df = self.data.get(ticker)
            if df is None:
                continue
            df_part = self.split_data(df, mode=mode)

            in_trade    = False
            entry_price = None
            entry_date  = None
            entry_idx   = None

            for i, row in ticker_sigs.iterrows():
                sig = row.get("signal", 0)

                if not in_trade and sig == 1:
                    # Find T+1 open
                    sig_date = pd.to_datetime(row["date"])
                    future   = df_part[df_part.index > sig_date]
                    if future.empty:
                        continue
                    entry_price = float(future.iloc[0]["Open"])
                    entry_date  = future.index[0]
                    entry_idx   = i
                    in_trade    = True

                elif in_trade and sig == -1:
                    # Find T+1 open for exit
                    sig_date = pd.to_datetime(row["date"])
                    future   = df_part[df_part.index > sig_date]
                    if future.empty:
                        # Exit at last available close
                        exit_price = float(df_part.iloc[-1]["Close"])
                        exit_date  = df_part.index[-1]
                    else:
                        exit_price = float(future.iloc[0]["Open"])
                        exit_date  = future.index[0]

                    gross_ret = (exit_price - entry_price) / entry_price
                    net_ret   = gross_ret - self.cost_model.round_trip_cost()
                    hold_days = (exit_date - entry_date).days

                    trades.append({
                        "ticker":      ticker,
                        "entry_date":  entry_date,
                        "exit_date":   exit_date,
                        "entry_price": round(entry_price, 2),
                        "exit_price":  round(exit_price, 2),
                        "gross_ret":   round(gross_ret * 100, 2),
                        "net_ret":     round(net_ret * 100, 2),
                        "hold_days":   hold_days,
                        "cost_pct":    round(
                            self.cost_model.round_trip_cost() * 100, 4
                        ),
                        "dataset":     mode.upper(),
                    })
                    in_trade = False

        return pd.DataFrame(trades)

    # ── Full Run ─────────────────────────────────────────────────

    def run(
        self,
        modes:  list[str] = ["train", "oos"],
        notes:  str = "",
    ) -> dict[str, BacktestMetrics]:
        """
        Full backtest run across specified modes.
        Returns metrics for each mode.
        Saves run record to /runs/.
        """
        run = BacktestRun(
            strategy=self.config.name,
            version=self.config.version,
            config=self.config.to_dict(),
            universe=self.config.universe,
            cost_model=self.config.cost_preset,
            notes=notes,
        )

        print(f"\n{'='*55}")
        print(f"  PulseAlgo Backtest Engine")
        print(f"  Run ID  : {run.run_id}")
        print(f"  Strategy: {self.config.name} v{self.config.version}")
        print(f"  Universe: {self.config.universe}")
        print(f"  Cost    : {self.config.cost_preset} "
              f"({self.cost_model.round_trip_cost()*100:.3f}% RT)")
        print(f"{'='*55}\n")

        # Validate config
        warnings = self.strategy.validate_config(self.config)
        for w in warnings:
            print(f"  ⚠️  CONFIG WARNING: {w}")

        results     = {}
        all_trades  = {}

        # Get benchmark returns
        bench_rets = None
        if self.nifty_df is not None and "Close" in self.nifty_df.columns:
            bench_rets = self.nifty_df["Close"].pct_change().dropna()

        for mode in modes:
            print(f"\n  ── {mode.upper()} ──")
            print(f"  Generating signals...")
            signals = self.generate_all_signals(mode=mode)

            if signals.empty:
                print(f"  No signals generated for {mode}")
                continue

            buy_count = (signals["signal"] == 1).sum()
            print(f"  Signals: {len(signals)} ({buy_count} BUY)")

            print(f"  Simulating trades...")
            trades = self.simulate_trades(signals, mode=mode)

            if trades.empty:
                print(f"  No trades for {mode}")
                continue

            print(f"  Trades: {len(trades)}")
            all_trades[mode] = trades

            # Build equity curve
            all_dates = []
            for df in self.data.values():
                if df is not None:
                    part = self.split_data(df, mode=mode)
                    all_dates.extend(part.index.tolist())
            date_idx  = pd.DatetimeIndex(sorted(set(all_dates)))
            equity    = build_equity_curve(trades, date_idx)

            # Benchmark slice
            bench_slice = None
            if bench_rets is not None and len(date_idx) > 0:
                bench_slice = bench_rets.reindex(date_idx).dropna()

            hold_series = trades["hold_days"] if "hold_days" in trades.columns else None

            metrics = calculate_metrics(
                trades_df=trades,
                equity_curve=equity,
                strategy=self.config.name,
                version=self.config.version,
                universe=self.config.universe,
                dataset=mode.upper(),
                benchmark_rets=bench_slice,
                hold_days=hold_series,
            )

            results[mode] = metrics
            print(metrics.summary())

        # Save run record
        run_record = {
            "run":     run.to_dict(),
            "results": {
                k: v.to_dict() for k, v in results.items()
            },
            "trades": {
                k: v.to_dict(orient="records")
                for k, v in all_trades.items()
            },
        }

        run_path = os.path.join(
            self.RUNS_DIR,
            f"{run.run_id}.json"
        )
        with open(run_path, "w") as f:
            json.dump(run_record, f, indent=2, default=str)
        print(f"\n  Run saved: {run_path}")

        return results

    # ── Walk Forward ─────────────────────────────────────────────

    def walk_forward(
        self,
        train_days: int = 756,   # 3 years
        test_days:  int = 126,   # 6 months
    ) -> list[BacktestMetrics]:
        """
        Rolling walk-forward validation.
        Train on N days, test on M days, roll forward.
        """
        print(f"\n  Walk-Forward: {train_days}d train / {test_days}d test")

        # Get common date range
        all_dates = set()
        for df in self.data.values():
            if df is not None:
                all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if len(dates) < train_days + test_days:
            print("  Not enough data for walk-forward.")
            return []

        windows     = []
        start_idx   = 0
        window_num  = 0

        while start_idx + train_days + test_days <= len(dates):
            train_end = start_idx + train_days
            test_end  = train_end + test_days

            train_dates = set(dates[start_idx:train_end])
            test_dates  = set(dates[train_end:test_end])

            # Slice data for this window
            window_data = {}
            for ticker, df in self.data.items():
                if df is None:
                    continue
                train_df = df[df.index.isin(train_dates)]
                test_df  = df[df.index.isin(test_dates)]
                window_data[ticker] = {
                    "train": train_df,
                    "test":  test_df
                }

            window_num += 1
            print(f"\n  Window {window_num}: "
                  f"{str(dates[start_idx])[:10]} → "
                  f"{str(dates[test_end-1])[:10]}")

            # Run signals on test window
            test_signals = []
            for ticker, dfs in window_data.items():
                test_df = dfs["test"]
                if len(test_df) < 10:
                    continue
                try:
                    sigs = self.strategy.generate_signals(
                        data=test_df,
                        config=self.config,
                        nifty_df=self.nifty_df,
                    )
                    if sigs is not None and not sigs.empty:
                        sigs["ticker"] = ticker
                        test_signals.append(sigs)
                except Exception as e:
                    print(f"    Signal error {ticker}: {e}")

            if not test_signals:
                start_idx += test_days
                continue

            combined_sigs = pd.concat(test_signals, ignore_index=True)

            # Temporarily replace data with window test data
            orig_data  = self.data
            self.data  = {
                t: dfs["test"]
                for t, dfs in window_data.items()
            }
            trades = self.simulate_trades(combined_sigs, mode="full")
            self.data = orig_data

            if trades.empty:
                start_idx += test_days
                continue

            date_idx = pd.DatetimeIndex(sorted(test_dates))
            equity   = build_equity_curve(trades, date_idx)
            metrics  = calculate_metrics(
                trades_df=trades,
                equity_curve=equity,
                strategy=self.config.name,
                version=self.config.version,
                universe=self.config.universe,
                dataset=f"WF-Window-{window_num}",
            )
            windows.append(metrics)
            print(f"    Trades: {metrics.total_trades} | "
                  f"WR: {metrics.win_rate_pct}% | "
                  f"Sharpe: {metrics.sharpe}")

            start_idx += test_days

        # Summary
        if windows:
            wf_sharpes = [w.sharpe for w in windows]
            wf_wr      = [w.win_rate_pct for w in windows]
            print(f"\n  Walk-Forward Summary ({len(windows)} windows):")
            print(f"    Avg Sharpe  : {np.mean(wf_sharpes):.2f}")
            print(f"    Avg Win Rate: {np.mean(wf_wr):.1f}%")
            print(f"    WF Win Rate : "
                  f"{sum(s > 0 for s in wf_sharpes)}/{len(wf_sharpes)} "
                  f"windows positive Sharpe")

        return windows