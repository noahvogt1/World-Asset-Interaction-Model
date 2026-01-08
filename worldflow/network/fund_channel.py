from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from worldflow.core.utils import cosine_similarity_matrix


@dataclass
class FundChannelConfig:
    beta_window: int = 120
    top_k: int = 12
    include_static_edges: bool = True
    static_edge_scale: float = 1.0


def rolling_betas(
    change: pd.DataFrame,
    factor_change: pd.DataFrame,
    end_idx: int,
    window: int
) -> pd.DataFrame:
    """Compute betas for each instrument to each factor over [end-window, end-1]."""
    start = max(0, end_idx - window)
    y = change.iloc[start:end_idx]
    f = factor_change.iloc[start:end_idx]
    y = y.dropna(axis=1, how="any")
    f = f.dropna(axis=1, how="any")
    common_idx = y.index.intersection(f.index)
    y = y.reindex(common_idx)
    f = f.reindex(common_idx)
    if y.shape[1] < 3 or f.shape[1] < 1 or len(common_idx) < 30:
        return pd.DataFrame()

    # OLS beta = cov(y,f)/var(f) for each factor independently (simple + stable)
    betas = {}
    for inst in y.columns:
        yi = y[inst].values
        betas_row = []
        for fac in f.columns:
            fi = f[fac].values
            v = np.var(fi)
            if v < 1e-12:
                betas_row.append(0.0)
            else:
                betas_row.append(float(np.cov(yi, fi, ddof=0)[0,1] / v))
        betas[inst] = betas_row

    B = pd.DataFrame.from_dict(betas, orient="index", columns=list(f.columns))
    return B


def build_beta_similarity_adjacency(B: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Cosine similarity on beta vectors -> adjacency."""
    if B is None or B.empty or B.shape[0] < 3:
        return pd.DataFrame()
    X = B.values.astype(float)
    S = cosine_similarity_matrix(X)
    W = pd.DataFrame(S, index=B.index, columns=B.index)
    np.fill_diagonal(W.values, 0.0)

    if top_k and top_k > 0:
        for i in W.index:
            row = W.loc[i]
            keep = row.abs().nlargest(top_k).index
            drop = row.index.difference(keep)
            W.loc[i, drop] = 0.0
    return W


def load_static_edges(edges_fund_csv: str, scale: float = 1.0) -> pd.DataFrame:
    """Edge-list to adjacency (sparse)."""
    df = pd.read_csv(edges_fund_csv)
    if df.empty:
        return pd.DataFrame()
    # We'll build adjacency only on nodes present in the edge list
    nodes = sorted(set(df["src_instrument_id"]).union(set(df["dst_instrument_id"])))
    W = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for _, r in df.iterrows():
        s = str(r["src_instrument_id"])
        d = str(r["dst_instrument_id"])
        w = float(r["weight"]) * scale
        if s in W.index and d in W.columns:
            W.loc[s, d] += w
            W.loc[d, s] += w  # assume undirected in MVP
    np.fill_diagonal(W.values, 0.0)
    return W
