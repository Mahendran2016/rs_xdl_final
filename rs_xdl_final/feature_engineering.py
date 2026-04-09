"""
feature_engineering.py
=======================
RS-XDL Feature Engineering — 45 Features
Author: Mahendran Sundarapandiyan | Alliance University

Feature Categories:
  1. Price & Returns        (8)  — log-return, multi-period returns
  2. Trend Indicators       (8)  — SMA, EMA, MACD, price-to-MA ratio
  3. Volatility             (5)  — realised vol, Bollinger, vol ratio
  4. Momentum & Oscillators (4)  — RSI, momentum, rate-of-change
  5. Macro (lag-adjusted)   (5)  — CPI, repo rate, USD/INR, gold, brent
  6. Geopolitical / Fear    (6)  — India VIX, US VIX, DXY, US10Y
  7. Cross-asset Returns    (5)  — NSE stock log-returns
  8. Global Contagion       (4)  — S&P500, Brent, DXY, US10Y changes
  ─────────────────────────────
  Total                    (45)  features + 1 target
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def build_features(df: pd.DataFrame,
                   target: str = "NIFTY50") -> pd.DataFrame:
    """
    Build complete 45-feature matrix from raw price/macro DataFrame.

    Parameters
    ----------
    df     : DataFrame from data_loader (price series, macro, VIX etc.)
    target : column name of the target index (default NIFTY50)

    Returns
    -------
    feat_df : DataFrame with 45 feature columns + 'target' column
              All rows with NaN dropped (first ~60 rows removed due to
              rolling windows).
    """
    p   = df[target]
    out = pd.DataFrame(index=df.index)

    # ── 1. PRICE & RETURNS ────────────────────────────────────────
    out["log_return"]   = np.log(p / p.shift(1))
    out["return_1d"]    = p.pct_change(1)
    out["return_5d"]    = p.pct_change(5)
    out["return_10d"]   = p.pct_change(10)
    out["return_20d"]   = p.pct_change(20)
    out["return_60d"]   = p.pct_change(60)
    out["log_ret_lag1"] = out["log_return"].shift(1)
    out["log_ret_lag2"] = out["log_return"].shift(2)

    # ── 2. TREND INDICATORS ───────────────────────────────────────
    out["sma_10"]          = p.rolling(10).mean()
    out["sma_50"]          = p.rolling(50).mean()
    out["price_sma50_ratio"]= p / (out["sma_50"] + 1e-9)
    out["ema_12"]          = p.ewm(span=12, adjust=False).mean()
    out["ema_26"]          = p.ewm(span=26, adjust=False).mean()
    out["macd"]            = out["ema_12"] - out["ema_26"]
    out["macd_signal"]     = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"]       = out["macd"] - out["macd_signal"]

    # ── 3. VOLATILITY ─────────────────────────────────────────────
    out["vol_10"]       = out["log_return"].rolling(10).std()
    out["vol_30"]       = out["log_return"].rolling(30).std()
    out["vol_ratio"]    = out["vol_10"] / (out["vol_30"] + 1e-9)
    sma20               = p.rolling(20).mean()
    std20               = p.rolling(20).std()
    out["bb_pct"]       = (p - (sma20 - 2*std20)) / \
                          ((4*std20) + 1e-9)   # 0=lower band, 1=upper
    out["bb_width"]     = (4 * std20) / (sma20 + 1e-9)

    # ── 4. MOMENTUM & OSCILLATORS ────────────────────────────────
    delta              = p.diff()
    gain               = delta.clip(lower=0).rolling(14).mean()
    loss               = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"]      = 100 - (100 / (1 + gain / (loss + 1e-9)))
    out["momentum_10"] = p - p.shift(10)
    out["momentum_30"] = p - p.shift(30)
    out["roc_10"]      = (p / (p.shift(10) + 1e-9) - 1) * 100

    # ── 5. MACRO FEATURES (lag-adjusted) ─────────────────────────
    # USD/INR: daily — no lag needed
    if "USDINR" in df.columns:
        out["usdinr_ret"]   = np.log(df["USDINR"] / df["USDINR"].shift(1))
        out["usdinr_level"] = df["USDINR"] / df["USDINR"].rolling(20).mean()

    # Gold: daily — safe-haven signal
    if "GOLD" in df.columns:
        out["gold_ret"]     = np.log(df["GOLD"] / df["GOLD"].shift(1))

    # Brent crude: daily — inflation/geopolitical proxy
    if "BRENT" in df.columns:
        out["brent_ret"]    = np.log(df["BRENT"] / df["BRENT"].shift(1))

    # Repo Rate: RBI policy — daily step function (no lag since
    # markets price in rate decisions before official announcement)
    if "REPO_RATE" in df.columns:
        out["repo_rate"]    = df["REPO_RATE"]
        out["repo_change"]  = df["REPO_RATE"].diff()

    # CPI: monthly with 1-month publication lag (already applied in loader)
    # We use YoY CPI and compute a "surprise" vs 3-month rolling mean
    if "CPI_YOY" in df.columns:
        out["cpi_yoy"]      = df["CPI_YOY"]
        out["cpi_change"]   = df["CPI_YOY"].diff()
        # CPI surprise = actual minus 3-month rolling mean (consensus proxy)
        out["cpi_surprise"] = df["CPI_YOY"] - \
                              df["CPI_YOY"].rolling(63).mean()

    # ── 6. GEOPOLITICAL / FEAR INDICATORS ────────────────────────
    # India VIX — NSE implied volatility (best real-time crisis detector)
    if "INDIA_VIX" in df.columns:
        out["india_vix"]      = df["INDIA_VIX"]
        out["india_vix_ret"]  = df["INDIA_VIX"].pct_change()
        # VIX spike: >1.5x 60-day average signals crisis
        vix_ma60              = df["INDIA_VIX"].rolling(60).mean()
        out["india_vix_spike"]= (df["INDIA_VIX"] > vix_ma60 * 1.5).astype(float)
        out["india_vix_zscore"]= ((df["INDIA_VIX"] - vix_ma60) /
                                  (df["INDIA_VIX"].rolling(60).std() + 1e-9))

    # US VIX — global fear gauge (war, recession spillover)
    if "US_VIX" in df.columns:
        out["us_vix"]         = df["US_VIX"]
        out["us_vix_ret"]     = df["US_VIX"].pct_change()

    # ── 7. CROSS-ASSET NSE STOCK RETURNS ─────────────────────────
    for stock in ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]:
        if stock in df.columns:
            out[f"ret_{stock}"] = np.log(
                df[stock] / (df[stock].shift(1) + 1e-9))

    # ── 8. GLOBAL CONTAGION ───────────────────────────────────────
    # S&P 500: US market direction (leads India by ~0.5 day due to timezone)
    if "SP500" in df.columns:
        out["sp500_ret"]      = np.log(df["SP500"] / df["SP500"].shift(1))

    # DXY: Dollar strength → FII outflows from India
    if "DXY" in df.columns:
        out["dxy_ret"]        = np.log(df["DXY"] / df["DXY"].shift(1))

    # US 10Y yield: global rate environment
    if "US10Y" in df.columns:
        out["us10y_change"]   = df["US10Y"].diff()
        out["us10y_level"]    = df["US10Y"]

    # ── TARGET ────────────────────────────────────────────────────
    # Next-day NIFTY50 log-return (what we're forecasting)
    out["target"] = out["log_return"].shift(-1)

    # ── CLEAN UP ──────────────────────────────────────────────────
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna()

    n_feat = len([c for c in out.columns if c != "target"])
    print(f"  Feature matrix : {out.shape[0]} rows × {n_feat} features")
    print(f"  Date range     : {out.index[0].date()} → {out.index[-1].date()}")

    return out


def get_feature_groups():
    """
    Returns dict of feature group → list of feature names.
    Used for group-level SHAP analysis.
    """
    return {
        "returns":      ["log_return", "return_1d", "return_5d",
                         "return_10d", "return_20d", "return_60d",
                         "log_ret_lag1", "log_ret_lag2"],
        "trend":        ["sma_10", "sma_50", "price_sma50_ratio",
                         "ema_12", "ema_26", "macd", "macd_signal",
                         "macd_hist"],
        "volatility":   ["vol_10", "vol_30", "vol_ratio",
                         "bb_pct", "bb_width"],
        "momentum":     ["rsi_14", "momentum_10", "momentum_30", "roc_10"],
        "macro":        ["usdinr_ret", "usdinr_level", "gold_ret",
                         "brent_ret", "repo_rate", "repo_change",
                         "cpi_yoy", "cpi_change", "cpi_surprise"],
        "fear":         ["india_vix", "india_vix_ret", "india_vix_spike",
                         "india_vix_zscore", "us_vix", "us_vix_ret"],
        "cross_asset":  ["ret_RELIANCE", "ret_TCS", "ret_INFY",
                         "ret_HDFCBANK", "ret_ICICIBANK"],
        "global":       ["sp500_ret", "dxy_ret",
                         "us10y_change", "us10y_level"],
    }
