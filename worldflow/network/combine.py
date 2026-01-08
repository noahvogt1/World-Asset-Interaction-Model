from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def align_adjacencies(*Ws: pd.DataFrame) -> Tuple[pd.Index, list]:
    nodes = None
    for W in Ws:
        if W is None or W.empty:
            continue
        idx = W.index
        nodes = idx if nodes is None else nodes.union(idx)
    if nodes is None:
        return pd.Index([]), []
    out = []
    for W in Ws:
        if W is None or W.empty:
            out.append(pd.DataFrame(0.0, index=nodes, columns=nodes))
        else:
            out.append(W.reindex(index=nodes, columns=nodes).fillna(0.0))
    return nodes, out


def combine_channels(A_stat: pd.DataFrame, A_fund: pd.DataFrame, A_comp: pd.DataFrame,
                     w_stat: float, w_fund: float, w_comp: float) -> pd.DataFrame:
    nodes, mats = align_adjacencies(A_stat, A_fund, A_comp)
    if len(nodes) == 0:
        return pd.DataFrame()
    S, F, C = mats
    W = w_stat * S + w_fund * F + w_comp * C
    np.fill_diagonal(W.values, 0.0)
    return W
