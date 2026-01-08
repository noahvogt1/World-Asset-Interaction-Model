from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

from worldflow.data.registry import Registry
from worldflow.portfolio.neutralize import neutralize_weights, NeutralizeConfig


@dataclass
class PortfolioConfig:
    group_type: str = "micro"
    per_group_long: int = 1
    per_group_short: int = 1
    max_abs_weight: float = 0.03
    gross_target: float = 1.0
    net_target: float = 0.0
    z_cap: float = 3.0
    use_vol_scaling: bool = True


def build_within_group_ls(
    registry: Registry,
    z: pd.DataFrame,
    vol: Optional[pd.DataFrame],
    betas_for_neutral: Optional[pd.DataFrame],
    cfg: PortfolioConfig,
    gross_mult: Optional[pd.Series] = None,
    neutral_cfg: NeutralizeConfig = NeutralizeConfig()
) -> pd.DataFrame:
    insts = [c for c in z.columns if c in registry.instruments and not registry.instruments[c].is_factor]
    groups = registry.group_ids(cfg.group_type)
    w = pd.DataFrame(0.0, index=z.index, columns=insts)

    for date in z.index:
        zt = z.loc[date, insts].dropna()
        if zt.empty:
            continue

        day_w = pd.Series(0.0, index=insts)
        for gid in groups:
            members = [i for i in registry.instruments_in_group(gid) if i in insts and i in zt.index]
            if len(members) < (cfg.per_group_long + cfg.per_group_short + 1):
                continue
            zg = zt[members].sort_values()
            longs = list(zg.head(cfg.per_group_long).index)
            shorts = list(zg.tail(cfg.per_group_short).index)

            # confidence weights proportional to capped |z| within leg
            if longs:
                conf = np.clip(np.abs(zt.loc[longs].values), 0.0, cfg.z_cap)
                conf = conf / (conf.sum() + 1e-12)
                day_w.loc[longs] += conf
            if shorts:
                conf = np.clip(np.abs(zt.loc[shorts].values), 0.0, cfg.z_cap)
                conf = conf / (conf.sum() + 1e-12)
                day_w.loc[shorts] -= conf

        if day_w.abs().sum() == 0:
            continue

        # vol scaling (risk parity-ish): divide by forecast vol
        if cfg.use_vol_scaling and vol is not None and date in vol.index:
            v = vol.loc[date, insts].replace(0, np.nan)
            day_w = day_w / (v.fillna(v.median()) + 1e-8)

        # net adjust
        day_w = day_w - day_w.mean() + (cfg.net_target / len(day_w))

        # Neutralize exposures
        try:
            day_w = neutralize_weights(registry, day_w, betas=betas_for_neutral, cfg=neutral_cfg)
        except Exception:
            # if betas missing, still proceed
            pass

        # cap
        day_w = day_w.clip(-cfg.max_abs_weight, cfg.max_abs_weight)

        # scale to gross target
        if day_w.abs().sum() > 0:
            day_w *= (cfg.gross_target / day_w.abs().sum())

        # regime gross multiplier
        if gross_mult is not None and date in gross_mult.index:
            day_w *= float(gross_mult.loc[date])

        w.loc[date] = day_w

    return w
