# RS-XDL: Regime-Switching Explainable Deep Learning
### Indian Equity Markets — PhD Research Codebase

**Paper:** Regime-Switching Explainable Deep Learning for Indian Equity Markets:
An HMM–SHAP Portfolio Optimization Framework

**Target Journal:** Expert Systems with Applications (Elsevier, Q1, IF ~8.5)

**Author:** Mahendran Sundarapandiyan | Alliance University, Bangalore

---

## Quick Start (2 steps)

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Download real NSE data (run once, needs internet)
python data_loader.py

# Step 3 — Run full research pipeline
python main.py
```

All 8 paper figures saved to `outputs/figures/`
All 6 paper tables saved to `outputs/tables/`

---

## Project Structure

```
rs_xdl_final/
├── main.py                  # Full 9-stage pipeline (entry point)
├── data_loader.py           # Real NSE data downloader (yfinance)
├── feature_engineering.py   # 45-feature builder (no look-ahead bias)
├── hmm_regime.py            # HMM regime detector (Bull/Bear/Crisis)
├── rc_shap.py               # RC-SHAP novel contribution
├── evaluation.py            # DM tests, metrics, portfolio stats
├── requirements.txt
├── README.md
├── data/                    # NSE CSV saved here after data_loader.py
├── outputs/
│   ├── figures/             # 8 publication-quality PNG figures
│   └── tables/              # 6 CSV tables for paper
└── notebooks/
    └── 01_kaggle_pipeline.ipynb
```

---

## Data Downloaded by data_loader.py

| Series | Source | Ticker | Notes |
|---|---|---|---|
| NIFTY50 | NSE via Yahoo | ^NSEI | Target variable |
| RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, LT, WIPRO, AXISBANK, SBIN | NSE | .NS suffix | Cross-asset features |
| USD/INR | Forex | INR=X | Daily, no lag |
| Gold futures | COMEX | GC=F | Safe-haven proxy |
| Brent Crude | ICE | BZ=F | Oil/geopolitical proxy |
| S&P 500 | NYSE | ^GSPC | US contagion |
| US VIX | CBOE | ^VIX | Global fear gauge |
| India VIX | NSE | ^INDIAVIX | Domestic crisis detector |
| DXY Dollar Index | ICE | DX-Y.NYB | FII flow proxy |
| US 10Y Treasury | CBOT | ^TNX | Global rate environment |
| RBI Repo Rate | Hardcoded | — | Official MPC schedule |
| CPI YoY | Hardcoded | — | MOSPI + 1-month pub lag |

---

## Feature Categories (45 total)

1. **Price & Returns (8)** — log-return, multi-period returns, lags
2. **Trend (8)** — SMA, EMA, MACD, price-to-MA ratio
3. **Volatility (5)** — realised vol, Bollinger Band pct/width
4. **Momentum (4)** — RSI, momentum, rate-of-change
5. **Macro lag-adjusted (9)** — CPI (1-month pub lag), repo rate, USD/INR, gold, brent
6. **Geopolitical/Fear (6)** — India VIX, US VIX, VIX spike flag, VIX z-score
7. **Cross-asset (5)** — NSE stock log-returns
8. **Global Contagion (4)** — S&P500, DXY, US 10Y yield

**CPI handling:** Monthly MOSPI CPI data is only made available to the model
from the publication date (approx 12th of following month), not the reference
month — eliminating look-ahead bias.

---

## Novel Contributions

### RC-SHAP (Regime-Conditioned SHAP)

Standard SHAP:
```
φ_j(x) = Σ_{S⊆F\{j}} [|S|!(|F|-|S|-1)!/|F|!] · [f(x_{S∪{j}}) − f(x_S)]
```

RC-SHAP (our definition):
```
φ_j^(r)(x) = E[φ_j(x) | R = r]
           ≈ (1/|D_r|) Σ_{x ∈ D_r} φ_j(x)
```

Regime-weighted global importance:
```
Φ_j = Σ_r P̂(R=r) · E[|φ_j^(r)(x)|]
```

Statistical test: Kruskal-Wallis H-test per feature
  H_0: φ_j^(bull) = φ_j^(bear) = φ_j^(crisis)
  Rejection → feature j is regime-dependent

---

## Expected Results on Real NSE Data

| Model | RMSE | R² | DirAcc | Sharpe |
|---|---|---|---|---|
| XGBoost | ~0.010 | ~0.15–0.20 | ~56–58% | ~1.0–1.5 |
| ARIMA | ~0.015 | ~0.02–0.05 | ~51–53% | ~0.3–0.5 |
| Naive RW | ~0.016 | <0 | ~50% | ~0.1 |
| Ridge | ~0.013 | ~0.05–0.10 | ~52–55% | ~0.5–0.8 |

Portfolio: SHAP-guided Sharpe typically 15–30% above equal-weight on real data.

---

## Citation

```bibtex
@article{sundarapandiyan2025rsxdl,
  title   = {Regime-Switching Explainable Deep Learning for Indian Equity
             Markets: An HMM-SHAP Portfolio Optimization Framework},
  author  = {Sundarapandiyan, Mahendran},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Under review}
}
```

---

## License
MIT — free for academic research with attribution.
