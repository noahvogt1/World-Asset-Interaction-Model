from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

from worldflow.core.utils import log_return, diff_change, ewma_std, safe_zscore, clip_series
from worldflow.data.registry import Registry


@dataclass
class FeatureConfig:
    ret_clip: float = 0.25
    vol_span: int = 60
    yield_to_total_return: bool = True  # if duration present, use TR proxy for yields


def compute_change_series(
    registry: Registry,
    wide_px: pd.DataFrame,
    wide_yield: pd.DataFrame,
    wide_spread: pd.DataFrame,
    cfg: FeatureConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (change, std_change, vol_change) wide dataframes (date x instrument_id).
    change is the modeling change series:
      - prices: log returns
      - yields: if duration present and yield_to_total_return, use TR proxy ~ -duration*dy else dy
      - spreads: dy
    """
    all_ids = registry.list_instruments(include_factors=True)
    dates = sorted(set(wide_px.index).union(wide_yield.index).union(wide_spread.index))
    idx = pd.DatetimeIndex(dates)

    change = pd.DataFrame(index=idx, columns=all_ids, dtype=float)

    # Price-like
    for iid in wide_px.columns:
        if iid not in change.columns:
            continue
        s = wide_px[iid].reindex(idx)
        c = log_return(s)
        c = clip_series(c, -cfg.ret_clip, cfg.ret_clip)
        change[iid] = c

    # Yield-like
    for iid in wide_yield.columns:
        if iid not in change.columns:
            continue
        s = wide_yield[iid].reindex(idx)
        dy = diff_change(s)

        inst = registry.instruments.get(iid, None)
        dur = None
        if inst is not None and inst.meta and isinstance(inst.meta, dict):
            dur = inst.meta.get("duration", None)

        if cfg.yield_to_total_return and dur is not None:
            # TR proxy ~ -D * dy
            change[iid] = -float(dur) * dy
        else:
            change[iid] = dy

    # Spread-like (changes)
    for iid in wide_spread.columns:
        if iid not in change.columns:
            continue
        s = wide_spread[iid].reindex(idx)
        change[iid] = diff_change(s)

    # Vol estimates and standardized changes
    vol_change = pd.DataFrame(index=idx, columns=all_ids, dtype=float)
    std_change = pd.DataFrame(index=idx, columns=all_ids, dtype=float)
    for iid in all_ids:
        c = change[iid]
        sig = ewma_std(c, span=cfg.vol_span)
        vol_change[iid] = sig
        std_change[iid] = safe_zscore(c, sig)

    return change.sort_index(), std_change.sort_index(), vol_change.sort_index()
