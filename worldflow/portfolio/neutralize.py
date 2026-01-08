from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from worldflow.data.registry import Registry


@dataclass
class NeutralizeConfig:
    # which exposures to neutralize
    equity_beta_to: Optional[str] = "F_MKT_US"   # factor column name to neutralize against (in beta matrix)
    duration_neutral: bool = True
    fx_usd_neutral: bool = True


def _project_out(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project weights w onto the nullspace of exposures X (minimally adjusted)."""
    if X.size == 0:
        return w
    # Remove columns with near-zero variance
    keep = np.where(np.std(X, axis=0) > 1e-12)[0]
    X = X[:, keep] if len(keep) > 0 else np.empty((X.shape[0], 0))
    if X.shape[1] == 0:
        return w
    XtX = X.T @ X
    pinv = np.linalg.pinv(XtX)
    adj = X @ (pinv @ (X.T @ w))
    return w - adj


def neutralize_weights(
    registry: Registry,
    weights: pd.Series,
    betas: Optional[pd.DataFrame] = None,   # rows instrument_id, cols factor names
    cfg: NeutralizeConfig = NeutralizeConfig()
) -> pd.Series:
    insts = weights.index.tolist()
    w = weights.values.astype(float)

    exposure_cols = []
    X_list = []

    # Equity beta neutralization
    if betas is not None and cfg.equity_beta_to is not None and cfg.equity_beta_to in betas.columns:
        # use beta only for equities; others zero
        x = np.zeros(len(insts), dtype=float)
        for k, iid in enumerate(insts):
            inst = registry.instruments.get(iid)
            if inst and inst.asset_class == "equities":
                if iid in betas.index:
                    x[k] = float(betas.loc[iid, cfg.equity_beta_to])
        X_list.append(x.reshape(-1, 1))
        exposure_cols.append("equity_beta")

    # Duration neutralization (rates/cash instruments with duration meta)
    if cfg.duration_neutral:
        x = np.zeros(len(insts), dtype=float)
        for k, iid in enumerate(insts):
            inst = registry.instruments.get(iid)
            if not inst:
                continue
            if inst.asset_class in ("rates", "cash") and inst.meta and "duration" in inst.meta:
                x[k] = float(inst.meta["duration"])
        if np.any(np.abs(x) > 0):
            X_list.append(x.reshape(-1, 1))
            exposure_cols.append("duration")

    # FX USD neutrality (net USD exposure proxy)
    # Approximate: for FX instruments, treat base_ccy exposure +1, quote_ccy exposure -1 in USD terms.
    if cfg.fx_usd_neutral:
        x = np.zeros(len(insts), dtype=float)
        for k, iid in enumerate(insts):
            inst = registry.instruments.get(iid)
            if not inst or inst.asset_class != "fx":
                continue
            # If USD is base_ccy, holding the pair increases USD exposure (approx) => +1
            # If USD is quote_ccy, holding the pair decreases USD exposure => -1
            if inst.base_ccy == "USD":
                x[k] = +1.0
            elif inst.quote_ccy == "USD":
                x[k] = -1.0
        if np.any(np.abs(x) > 0):
            X_list.append(x.reshape(-1, 1))
            exposure_cols.append("usd_fx")

    X = np.concatenate(X_list, axis=1) if X_list else np.empty((len(insts), 0))
    w2 = _project_out(w, X)
    return pd.Series(w2, index=insts)
