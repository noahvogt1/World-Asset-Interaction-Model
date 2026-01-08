from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from worldflow.core.utils import ewma_std, safe_zscore, estimate_half_life


@dataclass
class ResidualConfig:
    resid_sigma_span: int = 60
    half_life_min_points: int = 80
    crisis_sigma_mult: float = 0.0  # sigma *= (1 + crisis_sigma_mult * p_off)


def compute_z(
    change: pd.DataFrame,
    yhat: pd.DataFrame,
    cfg: ResidualConfig,
    risk_off_prob: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    resid = change - yhat
    sigma = resid.apply(lambda s: ewma_std(s, span=cfg.resid_sigma_span))

    if risk_off_prob is not None and cfg.crisis_sigma_mult and cfg.crisis_sigma_mult != 0.0:
        p = risk_off_prob.reindex(resid.index).fillna(method="ffill").fillna(0.0).clip(0.0, 1.0)
        mult = (1.0 + cfg.crisis_sigma_mult * p).values.reshape(-1, 1)
        sigma = sigma * mult

    z = pd.DataFrame(index=change.index, columns=change.columns, dtype=float)
    for c in change.columns:
        z[c] = safe_zscore(resid[c], sigma[c])

    half_life = {}
    for c in z.columns:
        hl = estimate_half_life(z[c], min_points=cfg.half_life_min_points)
        if hl is not None:
            half_life[c] = hl
    hl_s = pd.Series(half_life).sort_index()
    return z, resid, hl_s
