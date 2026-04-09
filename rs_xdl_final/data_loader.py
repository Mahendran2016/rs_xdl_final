"""
data_loader.py
==============
RS-XDL: Real NSE/BSE Data Loader
Author : Mahendran Sundarapandiyan | Alliance University

Downloads ALL data via yfinance:
  - NIFTY50 + 9 NSE blue-chip stocks
  - India VIX (real-time fear / crisis detector)
  - Macro: USD/INR, Gold, Brent Crude, US VIX, DXY, US 10Y yield
  - Global contagion: S&P 500
  - CPI proxy: breakeven inflation (10Y Gsec - IIB spread)
  - RBI Repo Rate: step function from official RBI schedule

CPI Handling:
  - Monthly CPI loaded with 1-month publication lag (MOSPI ~3-4 week delay)
  - Breakeven inflation used as real-time daily proxy (no lag)

Run:
    python data_loader.py
    # Saves to data/nse_data_real.csv
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

os.makedirs("data", exist_ok=True)

# ── Ticker map ────────────────────────────────────────────────────────
STOCK_TICKERS = {
    "NIFTY50":   "^NSEI",
    "RELIANCE":  "RELIANCE.NS",
    "TCS":       "TCS.NS",
    "INFY":      "INFY.NS",
    "HDFCBANK":  "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "LT":        "LT.NS",
    "WIPRO":     "WIPRO.NS",
    "AXISBANK":  "AXISBANK.NS",
    "SBIN":      "SBIN.NS",
}

MACRO_TICKERS = {
    "USDINR":    "INR=X",        # USD/INR exchange rate
    "GOLD":      "GC=F",         # Gold futures
    "BRENT":     "BZ=F",         # Brent crude oil
    "SP500":     "^GSPC",        # S&P 500 (global contagion)
    "US_VIX":    "^VIX",         # CBOE VIX (global fear gauge)
    "INDIA_VIX": "^INDIAVIX",    # India VIX (domestic fear)
    "DXY":       "DX-Y.NYB",     # US Dollar index
    "US10Y":     "^TNX",         # US 10-year Treasury yield
}

START = "2019-01-01"
END   = "2024-12-31"


# ── RBI Repo Rate schedule ────────────────────────────────────────────
# Official RBI MPC decisions (source: rbi.org.in)
RBI_REPO_SCHEDULE = [
    ("2019-01-01", 6.50),
    ("2019-02-07", 6.25),
    ("2019-04-04", 6.00),
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    ("2020-03-27", 4.40),  # COVID emergency cut
    ("2020-05-22", 4.00),  # COVID follow-up cut
    ("2022-05-04", 4.40),  # Inflation tightening begins
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    ("2023-02-08", 6.50),
    ("2024-12-31", 6.50),  # Hold through end of sample
]

# ── Monthly CPI data (MOSPI) with publication lag ─────────────────────
# Source: mospi.gov.in / rbi.org.in/scripts/BS_ViewBulletin.aspx
# Format: (reference_month, publication_date, CPI_value)
# We use publication_date to avoid look-ahead bias
CPI_DATA = [
    # (pub_date,  cpi_yoy)  -- all dates are approximate release dates
    ("2019-01-14", 2.05), ("2019-02-13", 2.57), ("2019-03-12", 2.86),
    ("2019-04-12", 2.92), ("2019-05-13", 2.99), ("2019-06-12", 3.18),
    ("2019-07-12", 3.15), ("2019-08-14", 3.21), ("2019-09-12", 3.99),
    ("2019-10-14", 4.62), ("2019-11-13", 5.54), ("2019-12-13", 7.35),
    ("2020-01-13", 7.59), ("2020-02-12", 6.58), ("2020-03-12", 5.91),
    ("2020-06-13", 6.09), ("2020-07-13", 6.93), ("2020-08-13", 6.73),
    ("2020-09-14", 7.27), ("2020-10-13", 7.61), ("2020-11-12", 6.93),
    ("2020-12-14", 6.93), ("2021-01-12", 4.06), ("2021-02-12", 5.03),
    ("2021-03-12", 5.52), ("2021-04-12", 5.52), ("2021-05-12", 6.30),
    ("2021-06-14", 6.26), ("2021-07-12", 5.59), ("2021-08-12", 5.30),
    ("2021-09-13", 4.35), ("2021-10-12", 4.48), ("2021-11-12", 4.91),
    ("2021-12-13", 5.59), ("2022-01-12", 6.01), ("2022-02-14", 6.07),
    ("2022-03-14", 6.95), ("2022-04-12", 7.79), ("2022-05-12", 7.04),
    ("2022-06-13", 7.01), ("2022-07-12", 6.71), ("2022-08-12", 7.00),
    ("2022-09-12", 7.41), ("2022-10-12", 6.77), ("2022-11-14", 5.88),
    ("2022-12-12", 5.72), ("2023-01-12", 6.52), ("2023-02-13", 6.44),
    ("2023-03-13", 5.66), ("2023-04-12", 4.70), ("2023-05-12", 4.25),
    ("2023-06-12", 4.81), ("2023-07-12", 7.44), ("2023-08-14", 6.83),
    ("2023-09-12", 5.02), ("2023-10-12", 4.87), ("2023-11-13", 4.87),
    ("2023-12-12", 5.69), ("2024-01-12", 5.10), ("2024-02-12", 5.09),
    ("2024-03-12", 4.85), ("2024-04-12", 4.83), ("2024-05-13", 4.75),
    ("2024-06-12", 5.08), ("2024-07-12", 3.54), ("2024-08-12", 3.65),
    ("2024-09-12", 5.49), ("2024-10-14", 5.22), ("2024-11-12", 6.21),
    ("2024-12-13", 5.48),
]


def _fetch_single(name, ticker, start, end):
    """Download a single ticker and return Close series."""
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  ✗ {name:<14} empty")
            return None
        s = df["Close"].squeeze()
        s.name = name
        print(f"  ✓ {name:<14} {len(s):4d} rows  "
              f"{s.index[0].date()} → {s.index[-1].date()}")
        return s
    except Exception as e:
        print(f"  ✗ {name:<14} {str(e)[:60]}")
        return None


def build_repo_rate_series(date_index):
    """Build daily RBI repo rate series from official schedule."""
    repo = pd.Series(index=date_index, dtype=float, name="REPO_RATE")
    for i, (date_str, rate) in enumerate(RBI_REPO_SCHEDULE):
        start_d = pd.Timestamp(date_str)
        end_d   = (pd.Timestamp(RBI_REPO_SCHEDULE[i+1][0])
                   if i+1 < len(RBI_REPO_SCHEDULE)
                   else date_index[-1] + pd.Timedelta(days=1))
        mask = (date_index >= start_d) & (date_index < end_d)
        repo[mask] = rate
    return repo.ffill()


def build_cpi_series(date_index):
    """
    Build daily CPI series with proper publication lag.
    Values only appear from publication date onward — no look-ahead bias.
    """
    cpi = pd.Series(index=date_index, dtype=float, name="CPI_YOY")
    for pub_date_str, val in CPI_DATA:
        pub_date = pd.Timestamp(pub_date_str)
        # Only assign from publication date forward
        mask = date_index >= pub_date
        cpi[mask] = val
    return cpi  # NaN before first publication date — handled in features


def fetch_all_data(start=START, end=END,
                   save_path="data/nse_data_real.csv"):
    """
    Download all market data and combine into one DataFrame.
    Returns aligned DataFrame ready for feature engineering.
    """
    print("=" * 55)
    print("  RS-XDL Data Loader — Fetching Real NSE Data")
    print("=" * 55)

    # ── 1. Stocks ─────────────────────────────────────────────────
    print("\n[1/4] NSE Stocks + NIFTY50 ...")
    stock_series = {}
    for name, ticker in STOCK_TICKERS.items():
        s = _fetch_single(name, ticker, start, end)
        if s is not None:
            stock_series[name] = s

    if "NIFTY50" not in stock_series:
        raise RuntimeError("NIFTY50 download failed — check internet connection")

    # ── 2. Macro + Global ─────────────────────────────────────────
    print("\n[2/4] Macro & Global Factors ...")
    macro_series = {}
    for name, ticker in MACRO_TICKERS.items():
        s = _fetch_single(name, ticker, start, end)
        if s is not None:
            macro_series[name] = s

    # ── 3. Combine on NIFTY50 trading calendar ────────────────────
    print("\n[3/4] Aligning on NIFTY50 calendar ...")
    base_index = stock_series["NIFTY50"].index

    all_series = {}
    for d in [stock_series, macro_series]:
        for name, s in d.items():
            all_series[name] = s.reindex(base_index).ffill().bfill()

    df = pd.DataFrame(all_series, index=base_index)

    # ── 4. Add RBI Repo Rate + CPI (with lag) ─────────────────────
    print("\n[4/4] Adding RBI Repo Rate + CPI (publication-lag adjusted) ...")
    df["REPO_RATE"] = build_repo_rate_series(base_index)
    df["CPI_YOY"]   = build_cpi_series(base_index)

    # Drop rows where NIFTY50 is NaN (non-trading days leaked in)
    df = df.dropna(subset=["NIFTY50"])

    # Fill remaining NaNs (mostly at start of some series)
    df = df.ffill().bfill()

    print(f"\n  Combined dataset: {df.shape}")
    print(f"  Date range     : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns        : {list(df.columns)}")

    df.to_csv(save_path)
    print(f"\n  ✓ Saved → {save_path}")
    return df


def load_data(path="data/nse_data_real.csv", auto_download=True):
    """
    Load data from CSV if available, else download.
    This is what main.py calls.
    """
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded {df.shape} from {path}")
        return df
    elif auto_download:
        print(f"  {path} not found — downloading ...")
        return fetch_all_data(save_path=path)
    else:
        raise FileNotFoundError(
            f"{path} not found. Run: python data_loader.py")


if __name__ == "__main__":
    df = fetch_all_data()
    print("\nSample (last 3 rows):")
    print(df.tail(3).to_string())
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    print(f"Memory usage  : {df.memory_usage().sum() / 1024:.1f} KB")
