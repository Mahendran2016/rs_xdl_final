"""
hmm_regime.py
=============
RS-XDL: HMM Regime Detection Module
Author: Mahendran Sundarapandiyan | Alliance University

Three-state Gaussian HMM calibrated on:
  - NIFTY50 daily log-returns
  - 20-day realised volatility

States labelled semantically:
  Bull   — highest mean return
  Bear   — middle state
  Crisis — lowest mean return, highest volatility
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore")

REGIMES   = ["bull", "bear", "crisis"]
REGIME_COLORS = {"bull": "#1D9E75", "bear": "#E24B4A", "crisis": "#BA7517"}


class HMMRegimeDetector:
    """
    Three-state Hidden Markov Model for market regime detection.

    Parameters
    ----------
    n_states         : number of HMM states (default 3)
    covariance_type  : 'full' or 'diag' (default 'full')
    n_iter           : EM iterations (default 200)
    random_state     : for reproducibility
    """

    def __init__(self, n_states=3, covariance_type="full",
                 n_iter=200, random_state=42):
        self.n_states  = n_states
        self.hmm_      = GaussianHMM(
            n_components    = n_states,
            covariance_type = covariance_type,
            n_iter          = n_iter,
            random_state    = random_state,
            init_params     = 'stmc',
            params          = 'stmc',
        )
        self.label_map_     = {}   # state_int → regime_name
        self.regime_series_ = None
        self.posteriors_    = None
        self.state_stats_   = {}
        self.is_fitted_     = False

    # ── Fit ──────────────────────────────────────────────────────────
    def fit(self, price_series: pd.Series):
        """
        Fit HMM on log-returns + realised volatility of price series.

        Parameters
        ----------
        price_series : daily price series (e.g. NIFTY50 Close)

        Returns
        -------
        self
        """
        log_ret  = np.log(price_series / price_series.shift(1)).dropna()
        vol_20   = log_ret.rolling(20).std().dropna()
        common   = log_ret.index.intersection(vol_20.index)

        self._log_ret  = log_ret.loc[common]
        self._vol_20   = vol_20.loc[common]
        self._obs_idx  = common

        X = np.column_stack([self._log_ret.values, self._vol_20.values])

        # Standardise observations for numerical stability
        from sklearn.preprocessing import StandardScaler
        self._obs_scaler = StandardScaler()
        X = self._obs_scaler.fit_transform(X)
        self.hmm_.fit(X)
        states_raw = self.hmm_.predict(X)
        self.posteriors_ = pd.DataFrame(
            self.hmm_.predict_proba(X),
            index   = common,
            columns = [f"P_state_{i}" for i in range(self.n_states)]
        )

        # ── Label states by mean return ───────────────────────────
        state_means = {
            s: X[states_raw == s, 0].mean()
            for s in range(self.n_states)
        }
        sorted_states = sorted(state_means, key=lambda s: state_means[s])
        labels        = ["crisis", "bear", "bull"]
        self.label_map_ = {s: labels[i]
                           for i, s in enumerate(sorted_states)}

        self.regime_series_ = pd.Series(
            [self.label_map_[s] for s in states_raw],
            index = common,
            name  = "regime"
        )

        # ── State statistics ──────────────────────────────────────
        for reg in REGIMES:
            mask = self.regime_series_ == reg
            r    = self._log_ret[mask]
            if len(r) == 0:
                self.state_stats_[reg] = {
                    "count":0, "pct":0,
                    "mean_ret":np.nan, "volatility":np.nan, "sharpe":np.nan
                }
                continue
            self.state_stats_[reg] = {
                "count":      int(mask.sum()),
                "pct":        round(mask.mean() * 100, 1),
                "mean_ret":   round(r.mean() * 252, 4),
                "volatility": round(r.std() * np.sqrt(252), 4),
                "sharpe":     round(
                    (r.mean() / (r.std() + 1e-9)) * np.sqrt(252), 3),
            }

        self.is_fitted_ = True
        return self

    # ── Predict ──────────────────────────────────────────────────────
    def predict(self, price_series: pd.Series = None) -> pd.Series:
        """
        Return regime labels aligned to price_series index.
        If price_series is None, returns training regime series.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")

        if price_series is None:
            return self.regime_series_

        # Predict on new data
        log_ret = np.log(price_series / price_series.shift(1)).dropna()
        vol_20  = log_ret.rolling(20).std().dropna()
        common  = log_ret.index.intersection(vol_20.index)
        X       = np.column_stack([log_ret.loc[common].values,
                                   vol_20.loc[common].values])
        states_raw = self.hmm_.predict(X)
        return pd.Series(
            [self.label_map_[s] for s in states_raw],
            index = common,
            name  = "regime"
        )

    def get_regime_for_dates(self, date_index: pd.DatetimeIndex,
                              fill: str = "bear") -> pd.Series:
        """
        Return regime labels aligned to any DatetimeIndex.
        Fills missing dates with `fill` (default 'bear').
        """
        return self.regime_series_.reindex(date_index).fillna(fill)

    # ── Summary ──────────────────────────────────────────────────────
    def print_summary(self):
        print("\n" + "─" * 55)
        print("  HMM REGIME DETECTION SUMMARY")
        print("─" * 55)
        for reg in REGIMES:
            s = self.state_stats_.get(reg, {})
            print(f"  {reg.upper():8s}  "
                  f"days={s.get('count',0):4d}  "
                  f"({s.get('pct',0):4.1f}%)  "
                  f"ann_ret={s.get('mean_ret',np.nan):+.3f}  "
                  f"vol={s.get('volatility',np.nan):.3f}  "
                  f"Sharpe={s.get('sharpe',np.nan):+.3f}")
        print("─" * 55)

    def to_dataframe(self) -> pd.DataFrame:
        """Return regime statistics as DataFrame (Table 3 in paper)."""
        rows = []
        for reg in REGIMES:
            s = self.state_stats_.get(reg, {})
            rows.append({
                "Regime":      reg.capitalize(),
                "Days":        s.get("count", 0),
                "Pct_Total":   s.get("pct", 0),
                "Ann_Return":  s.get("mean_ret", np.nan),
                "Ann_Vol":     s.get("volatility", np.nan),
                "Sharpe":      s.get("sharpe", np.nan),
            })
        return pd.DataFrame(rows)
