from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class MeanReversionConfig:
    ar_window: int = 252
    horizon_days: int = 5
    min_points: int = 80


def _fit_ar1(z: pd.Series) -> Optional[Tuple[float, float]]:
    """Fit z_t = a + b z_{t-1} + e. Return (b, sigma_e)."""
    z = z.dropna()
    if len(z) < 30:
        return None
    z_lag = z.shift(1).dropna()
    z_now = z.loc[z_lag.index]
    X = np.vstack([np.ones(len(z_lag)), z_lag.values]).T
    y = z_now.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    b = float(beta[1])
    se = float(np.std(resid, ddof=1))
    return b, se


def mean_reversion_targets(
    z: pd.DataFrame,
    cfg: MeanReversionConfig
) -> Dict[str, pd.DataFrame]:
    """Compute per-asset targets:
    - expected_z_H: E[z_{t+H} | z_t] assuming AR(1)
    - p_sign_flip_H: P(sign flips by H) assuming normal innovations
    Returns dict of wide dataframes.
    """
    idx = z.index
    cols = z.columns
    expected = pd.DataFrame(index=idx, columns=cols, dtype=float)
    pflip = pd.DataFrame(index=idx, columns=cols, dtype=float)
    b_est = pd.DataFrame(index=idx, columns=cols, dtype=float)

    H = cfg.horizon_days
    for t_idx in range(len(idx)):
        start = max(0, t_idx - cfg.ar_window)
        win = z.iloc[start:t_idx+1]
        if len(win) < cfg.min_points:
            continue
        for c in cols:
            s = win[c].dropna()
            if len(s) < cfg.min_points:
                continue
            fit = _fit_ar1(s)
            if fit is None:
                continue
            b, se = fit
            zt = z.iloc[t_idx][c]
            if pd.isna(zt):
                continue
            mu = (b ** H) * float(zt)
            # approximate variance after H steps: se^2 * sum_{k=0}^{H-1} b^{2k}
            var = (se ** 2) * sum((b ** (2*k)) for k in range(H))
            sd = float(np.sqrt(max(var, 1e-12)))
            expected.iloc[t_idx][c] = mu
            b_est.iloc[t_idx][c] = b
            # probability sign flips: P(Z_H * zt < 0)
            # if zt>0 -> P(Z_H<0) = Φ((0-mu)/sd)
            # if zt<0 -> P(Z_H>0) = 1-Φ((0-mu)/sd)
            from math import erf, sqrt
            def phi(x):  # standard normal CDF
                return 0.5 * (1.0 + erf(x / sqrt(2.0)))
            if float(zt) > 0:
                p = phi((0.0 - mu) / sd)
            elif float(zt) < 0:
                p = 1.0 - phi((0.0 - mu) / sd)
            else:
                p = 0.5
            pflip.iloc[t_idx][c] = float(p)

    return {
        "expected_z_H": expected,
        "p_sign_flip_H": pflip,
        "ar1_b": b_est,
    }
