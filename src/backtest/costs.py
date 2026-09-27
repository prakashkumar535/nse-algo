"""
Transaction Cost Engine — NSE Realistic Model
=============================================
All costs are configurable. Default = realistic NSE delivery trade.
"""

from dataclasses import dataclass

@dataclass
class CostModel:
    """
    Realistic NSE transaction cost model.
    All rates are configurable per backtest run.
    """
    brokerage_pct:    float = 0.0003   # 0.03% or ₹20 flat (modeled as %)
    stt_buy_pct:      float = 0.0      # STT on buy side (delivery = 0)
    stt_sell_pct:     float = 0.001    # STT on sell side (delivery = 0.1%)
    exchange_pct:     float = 0.0000335 # NSE transaction charge
    gst_rate:         float = 0.18     # 18% on brokerage + exchange
    sebi_pct:         float = 0.000001 # ₹10/crore = negligible
    stamp_duty_pct:   float = 0.00015  # 0.015% on buy side
    slippage_pct:     float = 0.001    # 0.1% per side conservative

    def total_buy_cost(self) -> float:
        """Total cost as % of trade value on BUY side."""
        base = self.brokerage_pct + self.exchange_pct + self.stamp_duty_pct
        gst  = (self.brokerage_pct + self.exchange_pct) * self.gst_rate
        return base + gst + self.stt_buy_pct + self.slippage_pct + self.sebi_pct

    def total_sell_cost(self) -> float:
        """Total cost as % of trade value on SELL side."""
        base = self.brokerage_pct + self.exchange_pct
        gst  = (self.brokerage_pct + self.exchange_pct) * self.gst_rate
        return base + gst + self.stt_sell_pct + self.slippage_pct + self.sebi_pct

    def round_trip_cost(self) -> float:
        """Total cost for one complete trade (buy + sell)."""
        return self.total_buy_cost() + self.total_sell_cost()

    def summary(self) -> dict:
        return {
            "buy_cost_pct":        round(self.total_buy_cost() * 100, 4),
            "sell_cost_pct":       round(self.total_sell_cost() * 100, 4),
            "round_trip_cost_pct": round(self.round_trip_cost() * 100, 4),
        }


# ─── Preset Cost Models ─────────────────────────────────────────

ZERO_COST = CostModel(
    brokerage_pct=0, stt_buy_pct=0, stt_sell_pct=0,
    exchange_pct=0, gst_rate=0, sebi_pct=0,
    stamp_duty_pct=0, slippage_pct=0
)

REALISTIC_DELIVERY = CostModel()  # defaults = realistic delivery

CONSERVATIVE = CostModel(
    slippage_pct=0.002,   # 0.2% slippage (illiquid stocks)
    brokerage_pct=0.0005,
)

PRESETS = {
    "zero":         ZERO_COST,
    "realistic":    REALISTIC_DELIVERY,
    "conservative": CONSERVATIVE,
}


if __name__ == "__main__":
    for name, model in PRESETS.items():
        s = model.summary()
        print(f"\n{name.upper()}")
        print(f"  Buy cost  : {s['buy_cost_pct']}%")
        print(f"  Sell cost : {s['sell_cost_pct']}%")
        print(f"  Round trip: {s['round_trip_cost_pct']}%")