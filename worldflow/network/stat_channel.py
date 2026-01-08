from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class StatChannelConfig:
    window: int = 60
    shrink: float = 0.15
    top_k: int = 12
    remove_market_mode: bool = True
    method: str = "partial"  # corr | partial


def _sparsify(W: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if not top_k or top_k <= 0:
        return W
    out = W.copy()
    for i in out.index:
        row = out.loc[i]
        keep = row.abs().nlargest(top_k).index
        drop = row.index.difference(keep)
        out.loc[i, drop] = 0.0
    return out


def _corr_adj(X: pd.DataFrame, cfg: StatChannelConfig) -> pd.DataFrame:
    C = X.corr().fillna(0.0)
    n = C.shape[0]
    I = pd.DataFrame(np.eye(n), index=C.index, columns=C.columns)
    C = (1 - cfg.shrink) * C + cfg.shrink * I
    np.fill_diagonal(C.values, 0.0)
    return _sparsify(C, cfg.top_k)


def _partial_corr_adj(X: pd.DataFrame, cfg: StatChannelConfig) -> pd.DataFrame:
    # LedoitWolf covariance shrinkage then precision matrix inversion -> partial correlations
    Xv = X.values.astype(float)
    lw = LedoitWolf().fit(Xv)
    cov = lw.covariance_
    # additional diagonal shrink towards identity (optional)
    n = cov.shape[0]
    cov = (1 - cfg.shrink) * cov + cfg.shrink * np.eye(n)

    prec = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(prec) + 1e-12)
    pcorr = -prec / (d[:, None] * d[None, :])
    np.fill_diagonal(pcorr, 0.0)

    W = pd.DataFrame(pcorr, index=X.columns, columns=X.columns)
    return _sparsify(W, cfg.top_k)


def build_stat_adjacency(std_change_window: pd.DataFrame, cfg: StatChannelConfig) -> pd.DataFrame:
    """Rolling adjacency from standardized changes."""
    X = std_change_window.copy()
    X = X.dropna(axis=1, how="any")
    if X.shape[1] < 3 or len(X) < 20:
        return pd.DataFrame()

    if cfg.remove_market_mode:
        m = X.mean(axis=1)
        X = X.sub(m, axis=0)

    if str(cfg.method).lower() == "corr":
        return _corr_adj(X, cfg)
    return _partial_corr_adj(X, cfg)
