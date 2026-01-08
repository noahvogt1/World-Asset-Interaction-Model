from __future__ import annotations
import numpy as np
import pandas as pd


def load_comp_edges(edges_comp_csv: str, scale: float = 1.0) -> pd.DataFrame:
    df = pd.read_csv(edges_comp_csv)
    if df.empty:
        return pd.DataFrame()
    nodes = sorted(set(df["src_instrument_id"]).union(set(df["dst_instrument_id"])))
    W = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for _, r in df.iterrows():
        s = str(r["src_instrument_id"])
        d = str(r["dst_instrument_id"])
        w = float(r["weight"]) * scale
        W.loc[s, d] += w
        W.loc[d, s] += w  # assume symmetric
    np.fill_diagonal(W.values, 0.0)
    return W
