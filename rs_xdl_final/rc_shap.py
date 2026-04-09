"""
rc_shap.py
==========
RS-XDL: Regime-Conditioned SHAP (RC-SHAP) — Novel Contribution
Author: Mahendran Sundarapandiyan | Alliance University

Formal Definition:
    φ_j^(r)(x) = E[φ_j(x) | R = r]
               ≈ (1/|D_r|) Σ_{x ∈ D_r} φ_j(x)

Regime-weighted global importance:
    Φ_j = Σ_r P̂(R=r) · E[|φ_j^(r)(x)|]

Statistical Test (Kruskal-Wallis):
    H_0: φ_j^(bull) = φ_j^(bear) = φ_j^(crisis)
    Rejection → feature j is regime-dependent
"""

import numpy as np
import pandas as pd
from scipy import stats
import shap
import warnings
warnings.filterwarnings("ignore")

REGIMES = ["bull", "bear", "crisis"]


class RCShap:
    """
    Regime-Conditioned SHAP analyser.

    Parameters
    ----------
    model         : fitted XGBoost/tree model
    feature_names : list of feature column names
    """

    def __init__(self, model, feature_names: list):
        self.model          = model
        self.feature_names  = feature_names
        self.explainer_     = shap.TreeExplainer(model)
        self.shap_values_   = None
        self.regime_shap_   = {}
        self.global_phi_    = None
        self.kw_results_    = None

    # ── Step 1: Compute SHAP + condition on regime ────────────────
    def fit(self, X: np.ndarray, regime_labels,
            sample_size: int = 600):
        """
        Compute SHAP values and condition on regime labels.

        Parameters
        ----------
        X             : scaled feature matrix (n_samples, n_features)
        regime_labels : array-like of regime strings per sample
        sample_size   : SHAP background sample size
        """
        n   = min(sample_size, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        Xs  = X[idx]
        rl  = np.array(regime_labels)[idx]

        self.shap_values_    = self.explainer_.shap_values(Xs)
        self._rl             = rl
        self._Xs             = Xs

        # ── RC-SHAP per regime ─────────────────────────────────
        for reg in REGIMES:
            mask = rl == reg
            n_r  = mask.sum()
            if n_r < 5:
                self.regime_shap_[reg] = pd.DataFrame({
                    "feature":   self.feature_names,
                    "mean_shap": np.zeros(len(self.feature_names)),
                    "n_samples": 0,
                })
                continue
            sv_r     = self.shap_values_[mask]
            mean_abs = np.abs(sv_r).mean(axis=0)
            total    = mean_abs.sum() + 1e-12
            self.regime_shap_[reg] = pd.DataFrame({
                "feature":       self.feature_names,
                "mean_shap":     mean_abs / total,
                "mean_shap_raw": mean_abs,
                "n_samples":     n_r,
            }).sort_values("mean_shap", ascending=False
             ).reset_index(drop=True)

        # ── Regime-weighted global Φ_j ─────────────────────────
        counts  = {r: (rl == r).sum() for r in REGIMES}
        tot     = sum(counts.values()) + 1e-12
        probs   = {r: counts[r] / tot for r in REGIMES}
        phi     = np.zeros(len(self.feature_names))
        for reg in REGIMES:
            df_r = self.regime_shap_.get(reg, pd.DataFrame())
            if not df_r.empty and "mean_shap_raw" in df_r.columns:
                raw = (df_r.set_index("feature")["mean_shap_raw"]
                          .reindex(self.feature_names).fillna(0).values)
                phi += probs[reg] * raw
        self.global_phi_ = pd.DataFrame({
            "feature": self.feature_names, "Phi_j": phi
        }).sort_values("Phi_j", ascending=False).reset_index(drop=True)
        return self
    
    def kruskal_wallis_test(self) -> pd.DataFrame:
        """
        Test H0: SHAP distributions equal across regimes.
        Returns DataFrame with H, p-value, significance per feature.
        """

        import pandas as pd
        from scipy import stats

        results = []

        # 🔍 Debug info (keep for now)
        print("SHAP shape:", self.shap_values_.shape)
        print("Regime counts:\n", pd.Series(self._rl).value_counts())

        for feat in self.feature_names:
            fi = self.feature_names.index(feat)
            groups = []

            for reg in REGIMES:
                mask = self._rl == reg
                vals = self.shap_values_[mask, fi]

                # ✅ Ensure enough samples + no NaN
                vals = vals[~pd.isna(vals)]

                if len(vals) >= 5:
                    groups.append(vals)

            # ✅ Need at least 2 valid groups
            if len(groups) >= 2:
                try:
                    H, p = stats.kruskal(*groups)

                    sig = (
                        "***" if p < 0.001 else
                        "**"  if p < 0.01  else
                        "*"   if p < 0.05  else "ns"
                    )

                    results.append({
                        "feature": feat,
                        "H_stat": round(H, 3),
                        "p_value": round(p, 5),
                        "sig": sig,
                    })

                except Exception as e:
                    print(f"Skipping {feat}: {e}")

        # ✅ CRITICAL FIX
        if len(results) == 0:
            print("⚠️ No valid Kruskal-Wallis results generated")
            self.kw_results_ = pd.DataFrame(
                columns=["feature", "H_stat", "p_value", "sig"]
            )
            return self.kw_results_

        df = pd.DataFrame(results)

        self.kw_results_ = (
            df.sort_values("H_stat", ascending=False)
            .reset_index(drop=True)
        )

        return self.kw_results_
    
    # ── Helpers ───────────────────────────────────────────────────
    def top_features(self, regime: str, n: int = 10) -> pd.DataFrame:
        return self.regime_shap_.get(regime, pd.DataFrame()).head(n)

    def heatmap_data(self, top_n: int = 12) -> pd.DataFrame:
        """Wide DataFrame for SHAP shift heatmap figure."""
        ref = next((r for r in REGIMES
                    if not self.regime_shap_.get(r, pd.DataFrame()).empty), None)
        if ref is None:
            return pd.DataFrame()
        top_feats = self.regime_shap_[ref]["feature"].head(top_n).tolist()
        data = {}
        for reg in REGIMES:
            df_r = self.regime_shap_.get(reg, pd.DataFrame())
            if not df_r.empty:
                lkp = df_r.set_index("feature")["mean_shap"]
                data[reg] = [float(lkp.get(f, 0.0)) for f in top_feats]
            else:
                data[reg] = [0.0] * len(top_feats)
        return pd.DataFrame(data, index=top_feats)

    def significant_features(self, alpha: float = 0.05) -> pd.DataFrame:
        if self.kw_results_ is None:
            self.kruskal_wallis_test()
        return self.kw_results_[self.kw_results_["p_value"] < alpha]

    def print_summary(self):
        print("\n" + "─" * 55)
        print("  RC-SHAP SUMMARY")
        print("─" * 55)
        for reg in REGIMES:
            df = self.top_features(reg, 5)
            n  = int(self.regime_shap_[reg]["n_samples"].iloc[0]) \
                 if "n_samples" in self.regime_shap_[reg].columns else 0
            print(f"\n  {reg.upper()} (n={n}):")
            for _, row in df.iterrows():
                print(f"    {row['feature']:<28}  {row['mean_shap']:.5f}")
        if self.kw_results_ is not None:
            sig = self.significant_features()
            print(f"\n  Significant regime shifts: "
                  f"{len(sig)}/{len(self.kw_results_)} features (p<0.05)")
        print("─" * 55)
