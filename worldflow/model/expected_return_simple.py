from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class SimpleExpectedReturnConfig:
    neighbor_lambda: float = 1.0


def predict_from_betas_and_network(
    rt: pd.Series,                     # realized changes at t for instruments
    f_t: pd.Series,                    # factor changes at t (factor columns)
    betas: pd.DataFrame,               # rows instruments, cols factor columns
    W: pd.DataFrame,
    cfg: SimpleExpectedReturnConfig
) -> Tuple[pd.Series, Dict[str, Dict[str, float]]]:
    """Interpretable baseline:
    rhat_i = sum_k beta_{i,k} f_k(t) + lambda * sum_j W_ij r_j(t)
    Returns predictions and explain contributions.
    """
    insts = [i for i in rt.index if i in W.index and i in betas.index]
    if len(insts) == 0:
        return pd.Series(index=rt.index, dtype=float), {}

    # neighbor term uses realized rt (same-time, like your spec)
    neigh = W.loc[insts, insts].dot(rt.loc[insts]) * cfg.neighbor_lambda

    preds = pd.Series(index=rt.index, dtype=float)
    explain: Dict[str, Dict[str, float]] = {}

    # Align factors
    f_t = f_t.reindex(betas.columns).fillna(0.0)

    for i in insts:
        b = betas.loc[i].reindex(betas.columns).fillna(0.0)
        fac_contrib = (b * f_t).to_dict()
        neighbor = float(neigh.loc[i]) if i in neigh.index else 0.0
        yhat = float(b.dot(f_t) + neighbor)
        preds.loc[i] = yhat
        fac_contrib["neighbor"] = neighbor
        fac_contrib["pred"] = yhat
        explain[i] = {k: float(v) for k, v in fac_contrib.items()}

    return preds, explain
