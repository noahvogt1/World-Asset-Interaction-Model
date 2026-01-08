from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd


@dataclass
class CostConfig:
    asset_class_bps: Dict[str, float]
    impact_enabled: bool = False
    impact_k: float = 0.0


def compute_strategy_pnl(change: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Compute daily strategy return using weights lagged by 1 day."""
    # align
    idx = change.index.intersection(weights.index)
    X = change.reindex(idx)
    W = weights.reindex(idx).fillna(0.0)
    # lag weights
    W_lag = W.shift(1).fillna(0.0)
    common_cols = [c for c in W_lag.columns if c in X.columns]
    pnl = (W_lag[common_cols] * X[common_cols]).sum(axis=1)
    return pnl


def compute_costs(weights: pd.DataFrame, asset_class_map: Dict[str, str], cfg: CostConfig) -> pd.Series:
    idx = weights.index
    W = weights.fillna(0.0)
    turnover = W.diff().abs().fillna(0.0)
    # cost per day = sum_i turnover_i * bps_i/10000
    costs = pd.Series(0.0, index=idx)
    for c in turnover.columns:
        bps = float(cfg.asset_class_bps.get(asset_class_map.get(c, "equities"), 5.0))
        costs += turnover[c] * (bps / 10000.0)
    if cfg.impact_enabled and cfg.impact_k > 0:
        costs += cfg.impact_k * np.sqrt(turnover.sum(axis=1))
    return costs


def perf_stats(ret: pd.Series) -> Dict[str, float]:
    r = ret.dropna()
    if len(r) == 0:
        return {}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    sharpe = (mu / sd) * np.sqrt(252) if sd > 0 else np.nan
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    ann = float((cum.iloc[-1] ** (252/len(r)) - 1)) if len(r) > 0 else np.nan
    return {
        "ann_return": ann,
        "ann_vol": sd * np.sqrt(252),
        "sharpe": sharpe,
        "max_drawdown": float(dd),
        "avg_daily": mu,
        "days": float(len(r)),
    }
