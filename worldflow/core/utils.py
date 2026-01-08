from __future__ import annotations
import math
from typing import Optional, Tuple
import numpy as np
import pandas as pd


def log_return(px: pd.Series) -> pd.Series:
    return np.log(px).diff()


def diff_change(x: pd.Series) -> pd.Series:
    return x.diff()


def ewma_std(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).std()


def clip_series(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.clip(lower=lo, upper=hi)


def safe_zscore(x: pd.Series, sigma: pd.Series, eps: float = 1e-8) -> pd.Series:
    return x / (sigma.replace(0, np.nan) + eps)


def estimate_half_life(z: pd.Series, min_points: int = 60) -> Optional[float]:
    """AR(1) half-life estimate. Returns days or None."""
    z = z.dropna()
    if len(z) < min_points:
        return None
    z_lag = z.shift(1).dropna()
    z_now = z.loc[z_lag.index]
    X = np.vstack([np.ones(len(z_lag)), z_lag.values]).T
    y = z_now.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    b = float(beta[1])
    if not (0 < b < 1):
        return None
    return float(-math.log(2) / math.log(b))


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of X."""
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / denom
    return Xn @ Xn.T
