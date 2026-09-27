# NIFTY 100 universe — yfinance symbols (append .NS)
NIFTY100 = [
    "RELIANCE", "TCS", "HDFCBANK", "BHARTIARTL", "ICICIBANK",
    "INFY", "SBIN", "HINDUNILVR", "ITC", "LT",
    "KOTAKBANK", "HCLTECH", "BAJFINANCE", "MARUTI", "AXISBANK",
    "ASIANPAINT", "TITAN", "WIPRO", "ULTRACEMCO", "NTPC",
    "POWERGRID", "NESTLEIND", "SUNPHARMA", "TECHM", "ADANIENT",
    "ADANIPORTS", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "BAJAJFINSV", "ONGC", "COALINDIA", "GRASIM", "DRREDDY",
    "CIPLA", "DIVISLAB", "APOLLOHOSP", "TRENT", "BAJAJ_AUTO",
    "EICHERMOT", "HEROMOTOCO", "M_M", "TVSMOTOR", "BOSCHLTD",
    "BRITANNIA", "DABUR", "GODREJCP", "MARICO", "COLPAL",
    "PIDILITIND", "BERGEPAINT", "HAVELLS", "VOLTAS", "WHIRLPOOL",
    "DMART", "NYKAA", "ZOMATO", "PAYTM", "POLICYBZR",
    "HDFCLIFE", "SBILIFE", "ICICIPRULI", "GICRE", "ICICIGI",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
    "BANKBARODA", "CANBK", "UNIONBANK", "RBLBANK", "AUBANK",
    "CHOLAFIN", "BAJAJHLDNG", "SHRIRAMFIN", "MUTHOOTFIN", "MANAPPURAM",
    "ATUL", "PIDILITIND", "SRF", "AARTIIND", "DEEPAKNTR",
    "ADANIGREEN", "ADANIPOWER", "TATAPOWER", "CESC", "TORNTPOWER",
    "BPCL", "IOC", "HINDPETRO", "GAIL", "PETRONET",
    "VEDL", "HINDZINC", "NMDC", "SAIL", "NATIONALUM",
    "DRREDDY", "LUPIN", "BIOCON", "TORNTPHARM", "AUROPHARMA",
]

# Remove duplicates
NIFTY100 = list(dict.fromkeys(NIFTY100))

# Add .NS suffix for yfinance
# Special cases
SYMBOL_MAP = {
    "BAJAJ_AUTO":  "BAJAJ-AUTO",
    "M_M":         "M&M",
    "TATAMOTORS":  "TMCV",      # renamed Oct 2025
    "ZOMATO":      "ETERNAL",   # renamed Apr 2025
}

def get_yf_symbols():
    symbols = []
    for s in NIFTY100:
        yf_sym = SYMBOL_MAP.get(s, s)
        symbols.append(f"{yf_sym}.NS")
    return symbols

def get_nse_symbols():
    return [SYMBOL_MAP.get(s, s) for s in NIFTY100]

if __name__ == "__main__":
    syms = get_yf_symbols()
    print(f"Universe size: {len(syms)}")
    print(syms[:10])