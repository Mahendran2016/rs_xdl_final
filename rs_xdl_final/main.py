"""
╔══════════════════════════════════════════════════════════════════════════╗
║  RS-XDL: REGIME-SWITCHING EXPLAINABLE DEEP LEARNING                     ║
║  Indian Equity Markets — Complete Research Pipeline                      ║
║                                                                          ║
║  Author  : Mahendran Sundarapandiyan                                     ║
║  Institute: Alliance University, Bangalore                               ║
║  Target  : Expert Systems with Applications (Elsevier Q1, IF ~8.5)      ║
║                                                                          ║
║  IMPORTANT: This pipeline uses REAL NSE data only.                       ║
║  Run data_loader.py first to download data:                              ║
║      python data_loader.py                                               ║
║                                                                          ║
║  Then run the full pipeline:                                             ║
║      python main.py                                                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Stages:
  [1] Load real NSE data (NIFTY50, 9 stocks, macro, VIX, CPI)
  [2] HMM regime detection (Bull / Bear / Crisis)
  [3] Feature engineering (45 features, no look-ahead bias)
  [4] Walk-forward validation setup (5 folds)
  [5] Model training (XGBoost + ARIMA + Ridge baselines)
  [6] Diebold-Mariano significance tests
  [7] RC-SHAP regime-conditioned explainability
  [8] SHAP-guided portfolio construction
  [9] Publication-quality figures (8 figures + 6 tables)
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

# ── project modules ───────────────────────────────────────────────────
from data_loader          import load_data
from feature_engineering  import build_features, get_feature_groups
from hmm_regime           import HMMRegimeDetector, REGIMES, REGIME_COLORS
from rc_shap              import RCShap
from evaluation           import (forecast_metrics, dm_test_suite,
                                   portfolio_metrics, aggregate_folds)

import shap
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model  import Ridge
from statsmodels.tsa.arima.model import ARIMA

# ── reproducibility ───────────────────────────────────────────────────
np.random.seed(42)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables",  exist_ok=True)

plt.rcParams.update({
    "figure.dpi":   150,
    "font.size":    10,
    "font.family":  "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ════════════════════════════════════════════════════════════════════
# [1] LOAD REAL NSE DATA
# ════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  RS-XDL PIPELINE  —  Expert Systems with Applications")
print("═"*60)
print("\n[1/9] Loading real NSE market data ...")

# Tries CSV first; downloads via yfinance if not found
raw_df = load_data("data/nse_data_real.csv", auto_download=True)

print(f"  Loaded  : {raw_df.shape}  "
      f"{raw_df.index[0].date()} → {raw_df.index[-1].date()}")
print(f"  Columns : {list(raw_df.columns)}")

# ════════════════════════════════════════════════════════════════════
# [2] HMM REGIME DETECTION
# ════════════════════════════════════════════════════════════════════
print("\n[2/9] Fitting HMM regime detector ...")

hmm = HMMRegimeDetector(n_states=3, n_iter=200, random_state=42)
hmm.fit(raw_df["NIFTY50"])
hmm.print_summary()

# Save Table 1
hmm.to_dataframe().to_csv("outputs/tables/table1_regime_stats.csv",
                           index=False)

# ════════════════════════════════════════════════════════════════════
# [3] FEATURE ENGINEERING  (45 features, no look-ahead bias)
# ════════════════════════════════════════════════════════════════════
print("\n[3/9] Engineering 45 features ...")

feat_df     = build_features(raw_df, target="NIFTY50")
FEAT_COLS   = [c for c in feat_df.columns if c != "target"]
X_all       = feat_df[FEAT_COLS].values
y_all       = feat_df["target"].values
dates_all   = feat_df.index
N           = len(X_all)

# Align regime labels to feature matrix dates
regime_all  = hmm.get_regime_for_dates(dates_all, fill="bear")

print(f"  Samples  : {N}")
print(f"  Features : {len(FEAT_COLS)}")

# ════════════════════════════════════════════════════════════════════
# [4] WALK-FORWARD VALIDATION  (5 folds, no data leakage)
# ════════════════════════════════════════════════════════════════════
print("\n[4/9] Setting up 5-fold walk-forward validation ...")

N_FOLD    = 5
INIT      = int(N * 0.60)
fold_size = (N - INIT) // N_FOLD

folds = []
for k in range(N_FOLD):
    tr_end = INIT + k * fold_size
    te_end = min(tr_end + fold_size, N)
    folds.append((slice(0, tr_end), slice(tr_end, te_end)))
    print(f"  Fold {k+1}: train[0:{tr_end}]  "
          f"test[{tr_end}:{te_end}]  "
          f"({dates_all[tr_end].date()} → {dates_all[te_end-1].date()})")

# ════════════════════════════════════════════════════════════════════
# [5] MODEL TRAINING  across all folds
# ════════════════════════════════════════════════════════════════════
print("\n[5/9] Training models across all folds ...")
print("      Models: XGBoost | ARIMA(1,0,1) | Ridge | Naive RW")

all_fold_results = []

for fold_idx, (tr_sl, te_sl) in enumerate(folds):
    X_tr, X_te = X_all[tr_sl], X_all[te_sl]
    y_tr, y_te = y_all[tr_sl], y_all[te_sl]
    reg_te     = regime_all.iloc[te_sl]
    dates_te   = dates_all[te_sl]

    # Scale: fit on train only
    scaler   = MinMaxScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    # ── ARIMA(1,0,1) ─────────────────────────────────────────────
    try:
        arima_m = ARIMA(y_tr, order=(1, 0, 1)).fit()
        arima_p = arima_m.forecast(steps=len(y_te))
    except Exception:
        arima_p = np.full(len(y_te), y_tr.mean())

    # ── Naive random walk (predict 0 return) ─────────────────────
    naive_p = np.zeros(len(y_te))

    # ── Ridge regression ─────────────────────────────────────────
    ridge   = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr)
    ridge_p = ridge.predict(X_te_s)

    # ── XGBoost ──────────────────────────────────────────────────
    xgb_m = xgb.XGBRegressor(
        n_estimators     = 600,
        max_depth        = 6,
        learning_rate    = 0.03,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 3,
        reg_alpha        = 0.1,
        reg_lambda       = 1.0,
        random_state     = 42,
        verbosity        = 0,
    )
    xgb_m.fit(
        X_tr_s, y_tr,
        eval_set    = [(X_te_s, y_te)],
        verbose     = False,
    )
    xgb_p = xgb_m.predict(X_te_s)

    # ── SHAP values for this fold ─────────────────────────────────
    explainer  = shap.TreeExplainer(xgb_m)
    shap_vals  = explainer.shap_values(X_te_s)

    # ── RC-SHAP per regime ────────────────────────────────────────
    rc = RCShap(xgb_m, FEAT_COLS)
    rc.fit(X_te_s, reg_te.values, sample_size=min(400, len(X_te_s)))
    rc.kruskal_wallis_test()

    # ── Collect metrics ───────────────────────────────────────────
    fold_metrics = [
        forecast_metrics(y_te, xgb_p,   "XGBoost", fold_idx+1),
        forecast_metrics(y_te, arima_p,  "ARIMA",   fold_idx+1),
        forecast_metrics(y_te, naive_p,  "Naive",   fold_idx+1),
        forecast_metrics(y_te, ridge_p,  "Ridge",   fold_idx+1),
    ]

    print(f"  Fold {fold_idx+1}  "
          f"XGB  RMSE={fold_metrics[0]['RMSE']:.6f}  "
          f"R²={fold_metrics[0]['R2']:.4f}  "
          f"DirAcc={fold_metrics[0]['DirAcc']:.1f}%  "
          f"Sharpe={fold_metrics[0]['Sharpe']:.3f}")

    all_fold_results.append({
        "fold":       fold_idx + 1,
        "y_te":       y_te,
        "xgb_p":      xgb_p,
        "arima_p":    arima_p,
        "naive_p":    naive_p,
        "ridge_p":    ridge_p,
        "regime":     reg_te,
        "shap_vals":  shap_vals,
        "X_te_s":     X_te_s,
        "dates":      dates_te,
        "rc_shap":    rc,
        "metrics":    fold_metrics,
    })

# ════════════════════════════════════════════════════════════════════
# [6] DIEBOLD-MARIANO TESTS
# ════════════════════════════════════════════════════════════════════
print("\n[6/9] Diebold-Mariano significance tests ...")

y_agg     = np.concatenate([r["y_te"]    for r in all_fold_results])
xgb_agg   = np.concatenate([r["xgb_p"]  for r in all_fold_results])
arima_agg = np.concatenate([r["arima_p"] for r in all_fold_results])
naive_agg = np.concatenate([r["naive_p"] for r in all_fold_results])
ridge_agg = np.concatenate([r["ridge_p"] for r in all_fold_results])

dm_df = dm_test_suite(
    y_agg,
    {"XGBoost": xgb_agg, "ARIMA": arima_agg,
     "Naive": naive_agg, "Ridge": ridge_agg},
    proposed="XGBoost"
)
print(dm_df[["XGBoost vs", "DM_stat", "p_value", "sig"]].to_string(
    index=False))
dm_df.to_csv("outputs/tables/table3_diebold_mariano.csv", index=False)

# ════════════════════════════════════════════════════════════════════
# [7] RC-SHAP REGIME-CONDITIONED ANALYSIS
# ════════════════════════════════════════════════════════════════════
print("\n[7/9] Computing RC-SHAP across all folds ...")

# Aggregate SHAP values across folds
all_shap  = np.concatenate([r["shap_vals"] for r in all_fold_results])
all_Xs    = np.concatenate([r["X_te_s"]    for r in all_fold_results])
all_regs  = np.concatenate([r["regime"].values
                             for r in all_fold_results])

# Build one aggregated RCShap over all folds
last_xgb_model = xgb_m  # last fold model used for explainer
rc_global = RCShap(last_xgb_model, FEAT_COLS)
rc_global.fit(all_Xs, all_regs, sample_size=min(800, len(all_Xs)))
rc_global.kruskal_wallis_test()
rc_global.print_summary()

# Save per-regime SHAP tables
for reg in REGIMES:
    df_s = rc_global.regime_shap_.get(reg, pd.DataFrame())
    if not df_s.empty:
        df_s.to_csv(f"outputs/tables/table2_shap_{reg}.csv", index=False)

sig_feats = rc_global.significant_features()
sig_feats.to_csv("outputs/tables/table4_kruskal_wallis.csv", index=False)
print(f"  Significant regime SHAP shifts: "
      f"{len(sig_feats)}/{len(rc_global.kw_results_)} features (p<0.05)")

# ════════════════════════════════════════════════════════════════════
# [8] SHAP-GUIDED PORTFOLIO
# ════════════════════════════════════════════════════════════════════
print("\n[8/9] SHAP-guided portfolio construction ...")

PORTFOLIO_STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

port_records = []
for fold_res in all_fold_results:
    sv     = fold_res["shap_vals"]
    dates  = fold_res["dates"]
    regs   = fold_res["regime"].values

    for i in range(len(dates) - 1):
        d      = dates[i]
        d_next = dates[i + 1]

        # SHAP weight per stock
        w = {}
        for stk in PORTFOLIO_STOCKS:
            fn = f"ret_{stk}"
            if fn in FEAT_COLS:
                fi   = FEAT_COLS.index(fn)
                w[stk] = abs(sv[i, fi])
            else:
                w[stk] = 1.0
        total = sum(w.values()) + 1e-9
        w     = {s: v / total for s, v in w.items()}

        # Actual next-day returns
        day_ret = {}
        for stk in PORTFOLIO_STOCKS:
            if (stk in raw_df.columns and
                    d in raw_df.index and d_next in raw_df.index):
                day_ret[stk] = np.log(
                    raw_df.loc[d_next, stk] /
                    (raw_df.loc[d, stk] + 1e-9))
            else:
                day_ret[stk] = 0.0

        shap_ret = sum(w[s] * day_ret[s] for s in PORTFOLIO_STOCKS)
        eq_ret   = np.mean([day_ret[s] for s in PORTFOLIO_STOCKS])
        nifty_ret = (np.log(raw_df["NIFTY50"].get(d_next, np.nan) /
                            (raw_df["NIFTY50"].get(d, np.nan) + 1e-9))
                     if d_next in raw_df.index and d in raw_df.index
                     else 0.0)

        port_records.append({
            "date":       d_next,
            "shap_ret":   shap_ret,
            "eq_ret":     eq_ret,
            "nifty_ret":  nifty_ret,
        })

port_df = (pd.DataFrame(port_records)
           .dropna()
           .sort_values("date")
           .set_index("date"))

port_df["shap_cum"]  = (1 + port_df["shap_ret"]).cumprod()
port_df["eq_cum"]    = (1 + port_df["eq_ret"]).cumprod()
port_df["nifty_cum"] = (1 + port_df["nifty_ret"]).cumprod()

pm_shap  = portfolio_metrics(port_df["shap_ret"],  "SHAP-guided")
pm_eq    = portfolio_metrics(port_df["eq_ret"],    "Equal-weight")
pm_nifty = portfolio_metrics(port_df["nifty_ret"], "NIFTY50 passive")

port_summary = pd.DataFrame([pm_shap, pm_eq, pm_nifty])
port_summary.to_csv("outputs/tables/table5_portfolio.csv", index=False)

print(f"  SHAP-guided  Sharpe={pm_shap['Sharpe']:.3f}  "
      f"Ann={pm_shap['Ann_Return']:.3f}  "
      f"MaxDD={pm_shap['MaxDrawdown']:.3f}")
print(f"  Equal-weight Sharpe={pm_eq['Sharpe']:.3f}  "
      f"Ann={pm_eq['Ann_Return']:.3f}  "
      f"MaxDD={pm_eq['MaxDrawdown']:.3f}")
print(f"  NIFTY50 pass Sharpe={pm_nifty['Sharpe']:.3f}  "
      f"Ann={pm_nifty['Ann_Return']:.3f}")

# ════════════════════════════════════════════════════════════════════
# [9] PUBLICATION-QUALITY FIGURES  (8 figures for paper)
# ════════════════════════════════════════════════════════════════════
print("\n[9/9] Generating publication-quality figures ...")

regime_s  = hmm.regime_series_
log_ret_s = np.log(raw_df["NIFTY50"] /
                    raw_df["NIFTY50"].shift(1)).dropna()

# ── Figure 1: Regime Timeline ─────────────────────────────────────
fig = plt.figure(figsize=(13, 8))
gs  = gridspec.GridSpec(3, 1, hspace=0.35)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharex=ax0)
ax2 = fig.add_subplot(gs[2], sharex=ax0)

nifty_plot = raw_df["NIFTY50"].reindex(regime_s.index)
ax0.plot(nifty_plot.index, nifty_plot.values,
         color="#2C2C2A", lw=0.9, zorder=3)
for reg in REGIMES:
    mask = (regime_s == reg)
    idxs = regime_s.index[mask]
    for d in idxs:
        ax0.axvspan(d, d + pd.Timedelta(days=1),
                    alpha=0.15, color=REGIME_COLORS[reg], lw=0)
patches = [Patch(color=REGIME_COLORS[r], alpha=0.6, label=r.capitalize())
           for r in REGIMES]
ax0.legend(handles=patches, fontsize=8, loc="upper left", framealpha=0.7)
ax0.set_ylabel("NIFTY50 (₹)", fontsize=9)
ax0.set_title("(a) HMM Regime Detection — NIFTY50",
              fontsize=10, fontweight="bold")

# Log returns
lr_aligned = log_ret_s.reindex(regime_s.index)
for d, r in regime_s.items():
    if d in lr_aligned.index and not pd.isna(lr_aligned[d]):
        ax1.bar(d, lr_aligned[d], color=REGIME_COLORS[r],
                width=1, alpha=0.75)
ax1.axhline(0, color="black", lw=0.5)
ax1.set_ylabel("Log Return", fontsize=9)
ax1.set_title("(b) Daily Log Returns Coloured by Regime",
              fontsize=10, fontweight="bold")

# Posterior probabilities
posteriors = hmm.posteriors_
sorted_states = sorted(
    {s: hmm.hmm_.means_[s, 0] for s in range(3)},
    key=lambda s: hmm.hmm_.means_[s, 0]
)
regime_order = ["crisis", "bear", "bull"]
for i, (s, reg) in enumerate(zip(sorted_states, regime_order)):
    col_name = f"P_state_{s}"
    if col_name in posteriors.columns:
        ax2.fill_between(posteriors.index, posteriors[col_name],
                         alpha=0.65, color=REGIME_COLORS[reg],
                         label=reg.capitalize())
ax2.set_ylabel("P(Regime)", fontsize=9)
ax2.set_xlabel("Date", fontsize=9)
ax2.legend(fontsize=8, loc="upper right")
ax2.set_title("(c) HMM Posterior Regime Probabilities",
              fontsize=10, fontweight="bold")

fig.suptitle("Figure 1: Market Regime Detection via HMM — NIFTY50 (2019–2024)",
             fontsize=11, fontweight="bold")
fig.savefig("outputs/figures/fig1_regime_detection.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  ✓ Fig 1: Regime detection")

# ── Figure 2: RC-SHAP Comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for ax, reg in zip(axes, REGIMES):
    df_s = rc_global.top_features(reg, 12)
    if df_s.empty:
        ax.set_title(f"{reg.capitalize()} — no data", fontsize=10)
        continue
    ax.barh(df_s["feature"][::-1], df_s["mean_shap"][::-1],
            color=REGIME_COLORS[reg], height=0.6, edgecolor="white")
    ax.set_title(f"{reg.capitalize()} Regime",
                 fontsize=11, fontweight="bold", color=REGIME_COLORS[reg])
    ax.set_xlabel("Normalised Mean |SHAP|", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="x", alpha=0.3)
fig.suptitle("Figure 2: RC-SHAP Feature Importance by Market Regime\n"
             "(Novel Contribution — feature drivers shift significantly "
             "across Bull/Bear/Crisis states)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/figures/fig2_regime_shap_comparison.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  ✓ Fig 2: RC-SHAP regime comparison")

# ── Figure 3: Walk-forward Prediction ────────────────────────────
last = all_fold_results[-1]
n_p  = min(120, len(last["y_te"]))
dp   = last["dates"][:n_p]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(dp, last["y_te"][:n_p],   label="Actual",
        color="#2C2C2A", lw=1.2, zorder=5)
ax.plot(dp, last["xgb_p"][:n_p],  label="XGBoost",
        color="#534AB7", lw=1, ls="--")
ax.plot(dp, last["arima_p"][:n_p], label="ARIMA",
        color="#888780", lw=1, ls=":")
reg_te = last["regime"].iloc[:n_p]
for reg in REGIMES:
    mask = (reg_te == reg).values
    for i, m in enumerate(mask):
        if m and i < len(dp):
            ax.axvspan(dp[i],
                       dp[min(i+1, n_p-1)],
                       alpha=0.07, color=REGIME_COLORS[reg])
ax.axhline(0, color="gray", lw=0.5, ls=":")
ax.set_title("Figure 3: Walk-Forward Forecast — XGBoost vs ARIMA (Fold 5)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Log Return")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/figures/fig3_walkforward_forecast.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  ✓ Fig 3: Walk-forward forecast")

# ── Figure 4: SHAP Beeswarm ───────────────────────────────────────
sample = min(500, len(all_shap))
idx_s  = np.random.choice(len(all_shap), sample, replace=False)
fig    = plt.figure(figsize=(9, 7))
shap.summary_plot(all_shap[idx_s], all_Xs[idx_s],
                  feature_names=FEAT_COLS, max_display=15,
                  show=False, plot_type="dot")
plt.title("Figure 4: Global SHAP Feature Importance (All Regimes Combined)",
          fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig("outputs/figures/fig4_shap_beeswarm.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  ✓ Fig 4: SHAP beeswarm")

# ── Figure 5: SHAP Shift Heatmap ─────────────────────────────────
hm_data = rc_global.heatmap_data(top_n=12)
if not hm_data.empty and hm_data.values.size > 0:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(hm_data, annot=True, fmt=".4f",
                cmap="RdYlGn", ax=ax, linewidths=0.5,
                cbar_kws={"label": "Normalised Mean |SHAP|"})
    ax.set_title("Figure 5: SHAP Feature Importance Shift Across Regimes\n"
                 "(RC-SHAP Novel Contribution — Regime-conditioned XAI)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Market Regime"); ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig("outputs/figures/fig5_shap_heatmap.png",
                bbox_inches="tight", dpi=200)
    plt.close()
print("  ✓ Fig 5: SHAP regime heatmap")

# ── Figure 6: Diebold-Mariano ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
dm_plot = dm_df.copy()
colors_dm = [REGIME_COLORS["bull"] if v < 0 else REGIME_COLORS["bear"]
             for v in dm_plot["DM_stat"]]
bars = ax.bar(dm_plot["XGBoost vs"], dm_plot["DM_stat"],
              color=colors_dm, width=0.4)
ax.axhline(0,    color="black", lw=0.7)
ax.axhline(-1.96, color="gray", lw=0.8, ls="--", label="5% critical (±1.96)")
ax.axhline(1.96,  color="gray", lw=0.8, ls="--")
for bar, row in zip(bars, dm_plot.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1*np.sign(bar.get_height()),
            row.sig, ha="center", fontsize=13, fontweight="bold")
ax.set_ylabel("DM Statistic", fontsize=9)
ax.set_title("Figure 6: Diebold-Mariano Test\n"
             "(Negative DM = XGBoost significantly outperforms baseline)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/fig6_diebold_mariano.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  ✓ Fig 6: Diebold-Mariano test")

# ── Figure 7: Portfolio Performance ──────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(port_df.index, port_df["shap_cum"],
             color=REGIME_COLORS["bull"], lw=1.5, label="SHAP-guided")
axes[0].plot(port_df.index, port_df["eq_cum"],
             color=REGIME_COLORS["bear"], lw=1.2, ls="--",
             label="Equal-weight")
axes[0].plot(port_df.index, port_df["nifty_cum"],
             color="#888780", lw=1.0, ls=":",
             label="NIFTY50 passive")
axes[0].set_ylabel("Cumulative Return", fontsize=9)
axes[0].set_title("(a) Cumulative Portfolio Returns",
                  fontsize=10, fontweight="bold")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

roll_sharpe_shap = port_df["shap_ret"].rolling(60).apply(
    lambda x: (x.mean()/(x.std()+1e-9))*np.sqrt(252))
roll_sharpe_eq   = port_df["eq_ret"].rolling(60).apply(
    lambda x: (x.mean()/(x.std()+1e-9))*np.sqrt(252))
axes[1].plot(port_df.index, roll_sharpe_shap,
             color=REGIME_COLORS["bull"], lw=1.2, label="SHAP-guided")
axes[1].plot(port_df.index, roll_sharpe_eq,
             color=REGIME_COLORS["bear"], lw=1.2, ls="--",
             label="Equal-weight")
axes[1].axhline(0, color="gray", lw=0.5)
axes[1].set_ylabel("Rolling Sharpe (60d)", fontsize=9)
axes[1].set_xlabel("Date", fontsize=9)
axes[1].set_title("(b) Rolling 60-Day Sharpe Ratio",
                  fontsize=10, fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

fig.suptitle("Figure 7: SHAP-Guided vs Benchmark Portfolio Performance",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/figures/fig7_portfolio.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  ✓ Fig 7: Portfolio comparison")

# ── Figure 8: Model Metrics Comparison ───────────────────────────
all_metrics = pd.DataFrame(
    [m for r in all_fold_results for m in r["metrics"]])
agg = aggregate_folds([m for r in all_fold_results
                       for m in r["metrics"]])
agg.to_csv("outputs/tables/table_metrics_all_folds.csv")

models_ord = ["XGBoost", "ARIMA", "Naive", "Ridge"]
colors_m   = [REGIME_COLORS["bull"], "#888780", "#B4B2A9", "#534AB7"]
metric_cfg = [("RMSE","lower better"),("R2","higher better"),
              ("DirAcc","higher better"),("Sharpe","higher better")]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for (metric, note), ax in zip(metric_cfg, axes.flat):
    means = [all_metrics[all_metrics["model"]==m][metric].mean()
             for m in models_ord]
    stds  = [all_metrics[all_metrics["model"]==m][metric].std()
             for m in models_ord]
    bars  = ax.bar(models_ord, means, color=colors_m,
                   yerr=stds, capsize=4, width=0.5)
    ax.set_title(f"{metric}  ({note})", fontweight="bold", fontsize=10)
    ax.set_ylabel(metric, fontsize=9)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + max(abs(v) for v in means)*0.03,
                f"{m:.4f}", ha="center", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Figure 8: Model Comparison — Mean ± Std (5 Walk-Forward Folds)",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("outputs/figures/fig8_metrics_comparison.png",
            bbox_inches="tight", dpi=200)
plt.close()
print("  ✓ Fig 8: Metrics comparison")

# ════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════
xgb_m_df = all_metrics[all_metrics["model"] == "XGBoost"]

print("\n" + "═"*60)
print("  PIPELINE COMPLETE")
print("═"*60)
print(f"\n  XGBoost RMSE    = {xgb_m_df['RMSE'].mean():.6f} "
      f"± {xgb_m_df['RMSE'].std():.6f}")
print(f"  XGBoost R²      = {xgb_m_df['R2'].mean():.4f} "
      f"± {xgb_m_df['R2'].std():.4f}")
print(f"  XGBoost DirAcc  = {xgb_m_df['DirAcc'].mean():.1f}%")
print(f"  XGBoost Sharpe  = {xgb_m_df['Sharpe'].mean():.3f}")
print(f"\n  SHAP-guided Sharpe   = {pm_shap['Sharpe']:.3f}")
print(f"  Equal-weight Sharpe  = {pm_eq['Sharpe']:.3f}")
print(f"  NIFTY50 pass Sharpe  = {pm_nifty['Sharpe']:.3f}")
print(f"\n  DM tests significant (p<0.05): "
      f"{(dm_df['p_value'] < 0.05).sum()}/{len(dm_df)}")
print(f"  RC-SHAP significant features: "
      f"{len(sig_feats)}/{len(rc_global.kw_results_)}")
print(f"\n  Outputs → outputs/figures/  (8 PNGs)")
print(f"  Tables  → outputs/tables/   (6 CSVs)")
print()
