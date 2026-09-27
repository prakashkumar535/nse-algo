"""
Abstract Base Strategy
======================
Every strategy inherits from this class.
Enforces consistent interface across all strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import json
import uuid
from datetime import datetime


# ─── Strategy Status ────────────────────────────────────────────

class StrategyStatus:
    RESEARCH     = "Research"
    BACKTESTING  = "Backtesting"
    OOS_TESTING  = "OOS Testing"
    WALK_FORWARD = "Walk Forward"
    PAPER_TRADING = "Paper Trading"
    LIVE         = "Live"
    RETIRED      = "Retired"


# ─── Strategy Config ────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """
    Base configuration for every strategy.
    All parameters must be configurable — nothing hard-coded.
    """
    # Identity
    name:        str = ""
    version:     str = "1.0.0"
    status:      str = StrategyStatus.RESEARCH
    description: str = ""
    category:    str = ""   # investment / swing / intraday

    # Universe
    universe:    str = "NIFTY100"

    # Execution
    entry_shift: int = 1    # T+N open (1 = next day open, default)
    exit_shift:  int = 1    # T+N open on exit signal

    # Cost model
    cost_preset: str = "realistic"   # zero / realistic / conservative
    slippage_pct: float = 0.001

    # Risk
    max_positions:      int   = 10
    risk_per_trade_pct: float = 1.0
    max_portfolio_dd_pct: float = 20.0

    # Validation
    train_pct:      float = 0.70
    validation_pct: float = 0.15
    oos_pct:        float = 0.15

    # Filters (each independently toggleable)
    use_regime_filter: bool = True   # NIFTY above EMA200

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        obj = cls()
        for k, v in d.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj


# ─── Run Record ─────────────────────────────────────────────────

@dataclass
class BacktestRun:
    """
    Every backtest run is recorded with full reproducibility info.
    """
    run_id:          str = field(default_factory=lambda: f"BT-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}")
    strategy:        str = ""
    version:         str = "1.0.0"
    config:          dict = field(default_factory=dict)
    universe:        str = ""
    period_start:    str = ""
    period_end:      str = ""
    data_version:    str = "yfinance"
    cost_model:      str = "realistic"
    validation_method: str = "train_oos"
    timestamp:       str = field(default_factory=lambda: datetime.now().isoformat())
    notes:           str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ─── Signal ─────────────────────────────────────────────────────

@dataclass
class Signal:
    """A single trading signal."""
    date:       str   = ""
    ticker:     str   = ""
    direction:  str   = ""   # BUY / SELL / HOLD
    strategy:   str   = ""
    version:    str   = "1.0.0"
    confidence: float = 0.0
    entry_price: Optional[float] = None
    stop_price:  Optional[float] = None
    notes:       str  = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─── Abstract Strategy ──────────────────────────────────────────

class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.

    Every strategy must implement:
    - generate_signals(data, config) → DataFrame of signals
    - get_default_config() → StrategyConfig

    Optional overrides:
    - validate_config(config)
    - describe()
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config  = config or self.get_default_config()
        self.version = self.config.version

    @abstractmethod
    def get_default_config(self) -> StrategyConfig:
        """Return default configuration for this strategy."""
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self,
        data:       pd.DataFrame,
        config:     Optional[StrategyConfig] = None,
        nifty_df:   Optional[pd.DataFrame]  = None,
    ) -> pd.DataFrame:
        """
        Generate buy/sell signals from OHLCV data.

        Parameters
        ----------
        data     : OHLCV DataFrame for one stock, with features pre-calculated
        config   : strategy config (uses self.config if None)
        nifty_df : NIFTY 50 OHLCV for regime filter

        Returns
        -------
        DataFrame with columns:
            date, ticker, signal (1=buy, -1=sell, 0=hold),
            entry_price, stop_price, notes
        """
        raise NotImplementedError

    def validate_config(self, config: StrategyConfig) -> list[str]:
        """
        Validate configuration. Returns list of warnings.
        Empty list = config is valid.
        """
        warnings = []
        if config.train_pct + config.validation_pct + config.oos_pct != 1.0:
            warnings.append("train + validation + oos must sum to 1.0")
        if config.max_positions < 1:
            warnings.append("max_positions must be >= 1")
        return warnings

    def describe(self) -> str:
        """Human-readable strategy description."""
        c = self.config
        return (
            f"Strategy : {c.name} v{c.version}\n"
            f"Status   : {c.status}\n"
            f"Category : {c.category}\n"
            f"Universe : {c.universe}\n"
            f"Desc     : {c.description}\n"
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} v{self.version} [{self.config.status}]>"