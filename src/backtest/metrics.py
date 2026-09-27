"""
Standardized Performance Metrics
=================================
Every strategy produces identical metrics for fair comparison.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BacktestMetrics:
    """Complete standardized metrics for any strategy."""

    # Identity
    strategy:        str = ""
    version:         str = "1.0.0"
    universe:        str = ""
    period_start:    str = ""
    period_end:      str = ""
    dataset:         str = ""   # TRAIN / VALIDATION / OOS

    # Performance
    total_return_pct:   float = 0.0
    cagr_pct:           float = 0.0
    benchmark_cagr_pct: float = 0.0
    alpha_pct:          float = 0.0

    # Risk
    max_drawdown_pct:   float = 0.0
    avg_drawdown_pct:   float = 0.0
    volatility_pct:     float = 0.0
    downside_dev_pct:   float = 0.0

    # Risk-adjusted
    sharpe:             float = 0.0
    sortino:            float = 0.0
    calmar:             float = 0.0

    # Trading
    total_trades:       int   = 0
    win_rate_pct:       float = 0.0
    avg_win_pct:        float = 0.0
    avg_loss_pct:       float = 0.0
    profit_factor:      float = 0.0
    expectancy_pct:     float = 0.0
    max_consec_losses:  int   = 0
    avg_hold_days:      float = 0.0
    median_hold_days:   float = 0.0

    # Exposure
    time_in_market_pct: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  {self.strategy} v{self.version} — {self.dataset}\n"
            f"{'='*55}\n"
            f"  Period      : {self.period_start} → {self.period_end}\n"
            f"  Universe    : {self.universe}\n"
            f"\n  PERFORMANCE\n"
            f"  Total Return: {self.total_return_pct:.1f}%\n"
            f"  CAGR        : {self.cagr_pct:.1f}%\n"
            f"  Benchmark   : {self.benchmark_cagr_pct:.1f}%\n"
            f"  Alpha       : {self.alpha_pct:.1f}%\n"
            f"\n  RISK\n"
            f"  Max DD      : {self.max_drawdown_pct:.1f}%\n"
            f"  Volatility  : {self.volatility_pct:.1f}%\n"
            f"\n  RISK-ADJUSTED\n"
            f"  Sharpe      : {self.sharpe:.2f}\n"
            f"  Sortino     : {self.sortino:.2f}\n"
            f"  Calmar      : {self.calmar:.2f}\n"
            f"\n  TRADING\n"
            f"  Trades      : {self.total_trades}\n"
            f"  Win Rate    : {self.win_rate_pct:.1f}%\n"
            f"  Avg Win     : {self.avg_win_pct:.2f}%\n"
            f"  Avg Loss    : {self.avg_loss_pct:.2f}%\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  Expectancy  : {self.expectancy_pct:.2f}%\n"
            f"  Avg Hold    : {self.avg_hold_days:.1f} days\n"
            f"  Time in Mkt : {self.time_in_market_pct:.1f}%\n"
            f"{'='*55}"
        )


def calculate_metrics(
    trades_df:      pd.DataFrame,
    equity_curve:   pd.Series,
    strategy:       str = "",
    version:        str = "1.0.0",
    universe:       str = "",
    dataset:        str = "TRAIN",
    benchmark_rets: Optional[pd.Series] = None,
    hold_days:      Optional[pd.Series] = None,
    trading_days:   int = 252,
) -> BacktestMetrics:
    """
    Calculate all standardized metrics from trades + equity curve.

    Parameters
    ----------
    trades_df     : DataFrame with columns [net_ret, entry_date, exit_date]
    equity_curve  : Series of cumulative portfolio value (starts at 1.0)
    benchmark_rets: Series of benchmark daily returns (NIFTY)
    hold_days     : Series of holding periods per trade
    """
    m = BacktestMetrics(
        strategy=strategy,
        version=version,
        universe=universe,
        dataset=dataset,
    )

    if trades_df.empty or equity_curve.empty:
        return m

    # ── Period ──────────────────────────────────────────────────
    if "entry_date" in trades_df.columns:
        m.period_start = str(trades_df["entry_date"].min())[:10]
        m.period_end   = str(trades_df["exit_date"].max())[:10]

    # ── Returns ─────────────────────────────────────────────────
    rets = trades_df["net_ret"].values / 100
    n    = len(rets)
    wins = (rets > 0).sum()
    loss = (rets < 0).sum()

    m.total_trades  = n
    m.win_rate_pct  = round(wins / n * 100, 1) if n > 0 else 0
    m.avg_win_pct   = round(rets[rets > 0].mean() * 100, 2) if wins > 0 else 0
    m.avg_loss_pct  = round(rets[rets < 0].mean() * 100, 2) if loss > 0 else 0
    m.expectancy_pct = round(rets.mean() * 100, 2)

    gp = rets[rets > 0].sum()
    gl = rets[rets < 0].sum()
    m.profit_factor = round(abs(gp) / abs(gl), 2) if gl != 0 else 999.0

    # Max consecutive losses
    consec = max_consecutive_losses(rets)
    m.max_consec_losses = consec

    # ── Equity Curve ────────────────────────────────────────────
    ec = equity_curve.dropna()
    if len(ec) < 2:
        return m

    total_ret = (ec.iloc[-1] / ec.iloc[0]) - 1
    m.total_return_pct = round(total_ret * 100, 1)

    # CAGR
    n_years = len(ec) / trading_days
    if n_years > 0 and ec.iloc[0] > 0:
        m.cagr_pct = round(
            ((ec.iloc[-1] / ec.iloc[0]) ** (1 / n_years) - 1) * 100, 1
        )

    # Daily returns from equity curve
    daily_rets = ec.pct_change().dropna()

    m.volatility_pct = round(
        daily_rets.std() * np.sqrt(trading_days) * 100, 1
    )

    # Downside deviation
    neg_rets = daily_rets[daily_rets < 0]
    m.downside_dev_pct = round(
        neg_rets.std() * np.sqrt(trading_days) * 100, 1
    ) if len(neg_rets) > 0 else 0

    # Sharpe (rf = 6% India)
    rf_daily = 0.06 / trading_days
    excess   = daily_rets - rf_daily
    m.sharpe = round(
        excess.mean() / daily_rets.std() * np.sqrt(trading_days), 2
    ) if daily_rets.std() > 0 else 0

    # Sortino
    m.sortino = round(
        excess.mean() / neg_rets.std() * np.sqrt(trading_days), 2
    ) if len(neg_rets) > 0 and neg_rets.std() > 0 else 0

    # Max drawdown
    roll_max = ec.cummax()
    dd       = (ec - roll_max) / roll_max
    m.max_drawdown_pct = round(dd.min() * 100, 1)
    m.avg_drawdown_pct = round(dd[dd < 0].mean() * 100, 1) if (dd < 0).any() else 0

    # Calmar
    if m.max_drawdown_pct != 0:
        m.calmar = round(m.cagr_pct / abs(m.max_drawdown_pct), 2)

    # ── Benchmark ───────────────────────────────────────────────
    if benchmark_rets is not None and len(benchmark_rets) > 1:
        bench_ec    = (1 + benchmark_rets).cumprod()
        bench_years = len(bench_ec) / trading_days
        if bench_years > 0:
            m.benchmark_cagr_pct = round(
                ((bench_ec.iloc[-1]) ** (1 / bench_years) - 1) * 100, 1
            )
        m.alpha_pct = round(m.cagr_pct - m.benchmark_cagr_pct, 1)

    # ── Hold Period ─────────────────────────────────────────────
    if hold_days is not None and len(hold_days) > 0:
        m.avg_hold_days    = round(float(hold_days.mean()), 1)
        m.median_hold_days = round(float(hold_days.median()), 1)

    return m


def max_consecutive_losses(rets: np.ndarray) -> int:
    max_cl = 0
    current = 0
    for r in rets:
        if r < 0:
            current += 1
            max_cl = max(max_cl, current)
        else:
            current = 0
    return max_cl


def build_equity_curve(
    trades_df:  pd.DataFrame,
    date_index: pd.DatetimeIndex,
    initial:    float = 100.0,
) -> pd.Series:
    """
    Build equity curve by applying trade returns chronologically.
    Each trade return is applied at exit date.
    Equity compounds across trades.
    """
    if trades_df.empty or len(date_index) == 0:
        return pd.Series(initial, index=date_index, dtype=float)

    # Start with flat equity
    equity = pd.Series(initial, index=date_index, dtype=float)

    # Sort trades by exit date
    trades_sorted = trades_df.copy()
    trades_sorted["exit_date"] = pd.to_datetime(trades_sorted["exit_date"])
    trades_sorted = trades_sorted.sort_values("exit_date").reset_index(drop=True)

    running_value = initial

    for _, trade in trades_sorted.iterrows():
        exit_date = trade["exit_date"]
        ret       = trade["net_ret"] / 100
        running_value *= (1 + ret)

        # Find nearest date in index
        idx_after = date_index[date_index >= exit_date]
        if len(idx_after) == 0:
            continue
        apply_date = idx_after[0]

        # Apply from this date forward
        equity.loc[apply_date:] = running_value

    return equity


if __name__ == "__main__":
    # Quick test
    trades = pd.DataFrame([
        {"net_ret": 2.1,  "entry_date": "2023-01-02", "exit_date": "2023-01-20"},
        {"net_ret": -1.5, "entry_date": "2023-02-01", "exit_date": "2023-02-19"},
        {"net_ret": 3.2,  "entry_date": "2023-03-01", "exit_date": "2023-03-19"},
        {"net_ret": -0.8, "entry_date": "2023-04-01", "exit_date": "2023-04-19"},
        {"net_ret": 1.9,  "entry_date": "2023-05-01", "exit_date": "2023-05-19"},
    ])
    idx = pd.date_range("2023-01-01", "2023-06-01", freq="B")
    ec  = build_equity_curve(trades, idx)
    m   = calculate_metrics(trades, ec, strategy="Test", dataset="TRAIN")
    print(m.summary())