from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from worldflow.data.registry import Registry


@dataclass
class CurrencyConfig:
    base_currency: str = "USD"


def _find_fx_series(registry: Registry, fx_px: pd.DataFrame, base: str, ccy: str) -> Tuple[Optional[str], int]:
    """Find an FX instrument that allows conversion from ccy to base.
    Returns (instrument_id, sign) where sign=+1 means add FX return, sign=-1 means subtract FX return.
    Convention: FX px is quote_ccy per base_ccy (quote/base).
    If we need base per ccy (base/ccy) returns, that's +1.
    If we have ccy per base (ccy/base), then base/ccy is inverse => subtract return.
    """
    # prefer direct: quote=base, base=ccy (base per ccy)
    for iid, inst in registry.instruments.items():
        if inst.asset_class != "fx":
            continue
        if inst.quote_ccy == base and inst.base_ccy == ccy and iid in fx_px.columns:
            return iid, +1
    # fallback inverse: quote=ccy, base=base (ccy per base)
    for iid, inst in registry.instruments.items():
        if inst.asset_class != "fx":
            continue
        if inst.quote_ccy == ccy and inst.base_ccy == base and iid in fx_px.columns:
            return iid, -1
    return None, 0


def apply_currency_normalization(
    registry: Registry,
    price_change: pd.DataFrame,
    fx_price_change: pd.DataFrame,
    cfg: CurrencyConfig
) -> pd.DataFrame:
    """Return USD-normalized changes for price-like instruments.
    For an asset priced in ccy != base, USD return = local_return + return(base/ccy).
    If only (ccy/base) is available, use -return(ccy/base).
    Yield/spread instruments are assumed already in base currency units in v1+.
    """
    base = cfg.base_currency
    out = price_change.copy()

    # Map currencies needed
    for iid in out.columns:
        inst = registry.instruments.get(iid, None)
        if inst is None:
            continue
        # only price-type instruments (returns) should be currency converted
        if inst.data_type != "price":
            continue
        ccy = inst.quote_ccy
        if ccy == base:
            continue
        fx_id, sign = _find_fx_series(registry, fx_price_change, base, ccy)
        if fx_id is None:
            raise ValueError(f"No FX series found to convert {iid} priced in {ccy} to {base}. Add an FX instrument in registry.")
        out[iid] = out[iid] + sign * fx_price_change[fx_id]

    return out
