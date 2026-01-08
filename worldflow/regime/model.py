from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass
class RegimeModelConfig:
    vol_z_weight: float = 0.9
    credit_z_weight: float = 0.6
    corr_spike_weight: float = 0.6
    bias: float = -0.5

    # channel weights for risk_on vs risk_off
    w_stat_on: float = 0.65
    w_fund_on: float = 0.30
    w_comp_on: float = 0.05

    w_stat_off: float = 0.25
    w_fund_off: float = 0.65
    w_comp_off: float = 0.10

    gross_on: float = 1.0
    gross_off: float = 0.6


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def infer_regime_and_weights(features: pd.DataFrame, cfg: RegimeModelConfig) -> pd.DataFrame:
    """Returns df with risk_off_prob, w_stat,w_fund,w_comp,gross_mult"""
    f = features.copy().fillna(0.0)
    score = (
        cfg.vol_z_weight * f["vol_z"].values +
        cfg.credit_z_weight * f["credit_z"].values +
        cfg.corr_spike_weight * f["corr_spike"].values +
        cfg.bias
    )
    p_off = sigmoid(score)

    w_stat = (1 - p_off) * cfg.w_stat_on + p_off * cfg.w_stat_off
    w_fund = (1 - p_off) * cfg.w_fund_on + p_off * cfg.w_fund_off
    w_comp = (1 - p_off) * cfg.w_comp_on + p_off * cfg.w_comp_off
    gross = (1 - p_off) * cfg.gross_on + p_off * cfg.gross_off

    out = pd.DataFrame(index=features.index)
    out["risk_off_prob"] = p_off
    out["w_stat"] = w_stat
    out["w_fund"] = w_fund
    out["w_comp"] = w_comp
    out["gross_mult"] = gross
    return out
