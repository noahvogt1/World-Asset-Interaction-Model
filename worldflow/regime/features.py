from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class RegimeFeatureConfig:
    # which instruments are used as macro proxies
    vol_id: str = "VIX"
    credit_hy_id: str = "HYG"
    credit_ig_id: str = "LQD"


def build_regime_features(change: pd.DataFrame, std_change: pd.DataFrame,
                          stat_avg_corr: pd.Series,
                          cfg: RegimeFeatureConfig) -> pd.DataFrame:
    """Returns dataframe indexed by date with columns: vol_z, credit_z, corr_spike."""
    idx = change.index
    out = pd.DataFrame(index=idx)

    # vol proxy: standardized change of VIX (or VOL)
    vol_series = std_change[cfg.vol_id] if cfg.vol_id in std_change.columns else std_change.mean(axis=1)
    out["vol_z"] = vol_series

    # credit stress proxy: HYG - LQD spread proxy (price return difference, crude)
    if cfg.credit_hy_id in change.columns and cfg.credit_ig_id in change.columns:
        credit_raw = (change[cfg.credit_hy_id] - change[cfg.credit_ig_id])
        credit_z = (credit_raw - credit_raw.rolling(60).mean()) / (credit_raw.rolling(60).std() + 1e-8)
        out["credit_z"] = credit_z
    else:
        out["credit_z"] = 0.0

    # correlation spike proxy (average off-diagonal corr) already computed per day
    out["corr_spike"] = stat_avg_corr.reindex(idx).fillna(method="ffill")

    return out
