"""
evaluation.py
=============
RS-XDL: Statistical Evaluation Module
Author: Mahendran Sundarapandiyan | Alliance University

Functions:
  forecast_metrics()   — RMSE, MAE, R², DirAcc, Sharpe
  diebold_mariano()    — Harvey-Leybourne-Newbold (1997) DM test
  dm_test_suite()      — DM tests: proposed vs all baselines
  portfolio_metrics()  — Sharpe, MaxDD, Ann Return, Calmar
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (mean_squared_error,
                              mean_absolute_error, r2_score)


# ── Forecast metrics ──────────────────────────────────────────────
def forecast_metrics(y_true, y_pred,
                     model_name="Model", fold=None) -> dict:
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    mae     = mean_absolute_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)
    dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100
    rs      = np.where(y_pred > 0, y_true, -y_true)
    sharpe  = (rs.mean() / (rs.std() + 1e-9)) * np.sqrt(252)
    res     = {"model": model_name, "RMSE": round(rmse, 6),
               "MAE": round(mae, 6),   "R2": round(r2, 4),
               "DirAcc": round(dir_acc, 2), "Sharpe": round(sharpe, 3)}
    if fold is not None:
        res["fold"] = fold
    return res


# ── Diebold-Mariano test ──────────────────────────────────────────
def diebold_mariano(y_true, forecast1, forecast2,
                    h: int = 1, criterion: str = "MSE"):
    """
    Harvey-Leybourne-Newbold (1997) modified DM test.
    Negative DM → forecast1 is significantly better.
    """
    e1 = y_true - forecast1
    e2 = y_true - forecast2
    d  = e1**2 - e2**2 if criterion == "MSE" else np.abs(e1) - np.abs(e2)
    n  = len(d)

    d_bar   = d.mean()
    gamma_0 = np.var(d, ddof=1)
    gammas  = [np.cov(d[k:], d[:-k])[0, 1]
               for k in range(1, h) if k < n]
    lrv     = max(gamma_0 + 2 * sum(gammas), 1e-12)
    hlncf   = np.sqrt(max((n + 1 - 2*h + h*(h-1)/n) / n, 1e-12))
    dm      = d_bar / np.sqrt(lrv / n) * hlncf
    p_val   = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return round(dm, 4), round(p_val, 5)


def dm_test_suite(y_true, preds_dict: dict,
                  proposed: str) -> pd.DataFrame:
    """Run DM tests: proposed model vs all others."""
    if proposed not in preds_dict:
        raise ValueError(f"{proposed} not in preds_dict")
    rows = []
    for name, preds in preds_dict.items():
        if name == proposed:
            continue
        dm, p = diebold_mariano(y_true, preds_dict[proposed], preds)
        sig   = ("***" if p < 0.001 else "**" if p < 0.01
                 else "*" if p < 0.05 else "ns")
        rows.append({f"{proposed} vs": name,
                     "DM_stat": dm, "p_value": p, "sig": sig,
                     "result": f"{proposed} better" if dm < 0
                               else "No significant difference"})
    return pd.DataFrame(rows)


# ── Portfolio metrics ─────────────────────────────────────────────
def portfolio_metrics(returns, name="Strategy") -> dict:
    r      = np.asarray(returns)
    ann_r  = r.mean() * 252
    ann_v  = r.std() * np.sqrt(252)
    sharpe = ann_r / (ann_v + 1e-9)
    cum    = (1 + r).cumprod()
    dd     = (cum - np.maximum.accumulate(cum)) / \
             (np.maximum.accumulate(cum) + 1e-9)
    max_dd = dd.min()
    return {"strategy":    name,
            "Ann_Return":  round(ann_r, 4),
            "Ann_Vol":     round(ann_v, 4),
            "Sharpe":      round(sharpe, 3),
            "MaxDrawdown": round(max_dd, 4),
            "Calmar":      round(ann_r / (abs(max_dd) + 1e-9), 3)}


# ── Fold aggregation ──────────────────────────────────────────────
def aggregate_folds(fold_metrics: list) -> pd.DataFrame:
    """Mean ± std across walk-forward folds."""
    df = pd.DataFrame(fold_metrics)
    return df.groupby("model")[
        ["RMSE", "MAE", "R2", "DirAcc", "Sharpe"]
    ].agg(["mean", "std"]).round(4)
