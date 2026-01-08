from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class ExplainConfig:
    top_neighbors: int = 5


def top_neighbor_contributions(change_t: pd.Series, W: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """For each i, compute top-k neighbors by |W_ij * r_j| on day t."""
    insts = [i for i in change_t.index if i in W.index]
    rows = []
    for i in insts:
        wrow = W.loc[i, insts]
        contrib = wrow * change_t.loc[insts]
        contrib = contrib.drop(labels=[i], errors="ignore")
        top = contrib.abs().sort_values(ascending=False).head(top_k).index.tolist()
        for j in top:
            rows.append({
                "asset": i,
                "neighbor": j,
                "edge_weight": float(wrow[j]),
                "neighbor_change": float(change_t[j]),
                "contribution": float(contrib[j]),
                "abs_contribution": float(abs(contrib[j])),
            })
    return pd.DataFrame(rows)
