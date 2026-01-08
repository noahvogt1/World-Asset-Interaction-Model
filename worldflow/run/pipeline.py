from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yaml

from worldflow.data.registry import load_registry, Registry
from worldflow.data.store import MarketDataStore
from worldflow.data.transforms import compute_change_series, FeatureConfig
from worldflow.data.normalization import apply_currency_normalization, CurrencyConfig
from worldflow.data.curve_factors import compute_level_slope_curvature, compute_curve_factor_changes, CurveSpec

from worldflow.network.stat_channel import build_stat_adjacency, StatChannelConfig
from worldflow.network.fund_channel import (
    rolling_betas, build_beta_similarity_adjacency, FundChannelConfig, load_static_edges
)
from worldflow.network.comp_channel import load_comp_edges
from worldflow.network.combine import combine_channels

from worldflow.regime.features import build_regime_features, RegimeFeatureConfig
from worldflow.regime.model import infer_regime_and_weights, RegimeModelConfig

from worldflow.model.expected_return_simple import predict_from_betas_and_network, SimpleExpectedReturnConfig
from worldflow.model.residuals import compute_z, ResidualConfig
from worldflow.model.mean_reversion import mean_reversion_targets, MeanReversionConfig
from worldflow.model.explain import top_neighbor_contributions

from worldflow.portfolio.builder import build_within_group_ls, PortfolioConfig
from worldflow.portfolio.neutralize import NeutralizeConfig


@dataclass
class Paths:
    instrument_master: str
    group_master: str
    membership: str
    bars: str
    edges_fund: str
    edges_comp: str
    universe_yaml: str
    channels_yaml: str
    regimes_yaml: str
    model_yaml: str
    curves_yaml: str
    out_dir: str


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_end_to_end(paths: Paths) -> None:
    os.makedirs(paths.out_dir, exist_ok=True)

    reg = load_registry(paths.instrument_master, paths.group_master, paths.membership)
    store = MarketDataStore(paths.bars)

    univ = load_yaml(paths.universe_yaml)
    chcfg = load_yaml(paths.channels_yaml)
    rcfg = load_yaml(paths.regimes_yaml)
    mcfg = load_yaml(paths.model_yaml)
    curves = load_yaml(paths.curves_yaml)

    base_ccy = str(univ.get("base_currency", "USD"))
    factor_ids = list(univ.get("factor_instruments", []))
    score_universe = list(univ.get("score_universe", []) or [])

    bars = store.load_bars()
    wide_px = store.to_wide(bars, field="px")
    wide_yield = store.to_wide(bars, field="yield")
    wide_spread = store.to_wide(bars, field="spread")

    feat_cfg = FeatureConfig(
        ret_clip=float(mcfg["features"]["ret_clip"]),
        vol_span=int(mcfg["features"]["vol_span"]),
        yield_to_total_return=bool(mcfg["features"]["yield_to_total_return"]),
    )
    change, std_change, vol = compute_change_series(reg, wide_px, wide_yield, wide_spread, feat_cfg)

    # Currency normalization framework (applies to price instruments if any are non-USD quoted)
    # For v1+: change already contains returns; we adjust those returns by FX returns.
    fx_ids = [iid for iid, inst in reg.instruments.items() if inst.asset_class == "fx" and iid in change.columns]
    price_ids = [iid for iid, inst in reg.instruments.items() if inst.data_type == "price" and iid in change.columns]
    change.loc[:, price_ids] = apply_currency_normalization(reg, change.loc[:, price_ids], change.loc[:, fx_ids], CurrencyConfig(base_currency=base_ccy))

    # Curve factors (level/slope/curvature) from yields; add as additional factor columns
    curve_factor_changes = pd.DataFrame(index=change.index)
    for curve_name, spec in curves.get("curves", {}).items():
        pts = spec["points"]
        needed = [pts["short"], pts["mid"], pts["long"]]
        if all(x in wide_yield.columns for x in needed):
            levels = compute_level_slope_curvature(wide_yield, CurveSpec(short=pts["short"], mid=pts["mid"], long=pts["long"]))
            d = compute_curve_factor_changes(levels)
            d.columns = [f"CURVE_{curve_name}_{c}" for c in d.columns]
            curve_factor_changes = curve_factor_changes.join(d, how="outer")

    # Factor matrix: changes of factor instruments + curve factors + optional credit proxy
    factors = change.reindex(columns=[c for c in factor_ids if c in change.columns]).copy()
    factors.columns = [f"F_{c}" for c in factors.columns]
    factors = factors.join(curve_factor_changes, how="outer").sort_index()

    # Channel configs
    stat_cfg = StatChannelConfig(**chcfg["stat"])
    fund_cfg = FundChannelConfig(**chcfg["fund"])
    include_comp = bool(chcfg.get("comp", {}).get("include_negative_edges", True))
    comp_scale = float(chcfg.get("comp", {}).get("negative_edge_scale", 1.0))

    A_static_fund = load_static_edges(paths.edges_fund, scale=fund_cfg.static_edge_scale) if fund_cfg.include_static_edges else pd.DataFrame()
    A_comp = load_comp_edges(paths.edges_comp, scale=comp_scale) if include_comp else pd.DataFrame()

    # Regime config
    ro = rcfg["risk_off"]
    rm = rcfg["channel_weights"]
    pg = rcfg["portfolio_gross"]
    regime_cfg = RegimeModelConfig(
        vol_z_weight=float(ro["vol_z_weight"]),
        credit_z_weight=float(ro["credit_z_weight"]),
        corr_spike_weight=float(ro["corr_spike_weight"]),
        bias=float(ro["bias"]),
        w_stat_on=float(rm["risk_on"]["stat"]),
        w_fund_on=float(rm["risk_on"]["fund"]),
        w_comp_on=float(rm["risk_on"]["comp"]),
        w_stat_off=float(rm["risk_off"]["stat"]),
        w_fund_off=float(rm["risk_off"]["fund"]),
        w_comp_off=float(rm["risk_off"]["comp"]),
        gross_on=float(pg["risk_on"]),
        gross_off=float(pg["risk_off"]),
    )

    exp_cfg = SimpleExpectedReturnConfig(neighbor_lambda=float(mcfg["expected_return"]["neighbor_lambda"]))
    resid_cfg = ResidualConfig(
        resid_sigma_span=int(mcfg["residuals"]["resid_sigma_span"]),
        half_life_min_points=int(mcfg["residuals"]["half_life_min_points"]),
        crisis_sigma_mult=float(mcfg["residuals"].get("crisis_sigma_mult", 0.0)),
    )
    mr_cfg = MeanReversionConfig(
        ar_window=int(mcfg["mean_reversion"]["ar_window"]),
        horizon_days=int(mcfg["mean_reversion"]["horizon_days"]),
        min_points=80,
    )

    # Score set
    if score_universe:
        score_ids = [i for i in score_universe if i in change.columns]
    else:
        score_ids = [i for i in reg.list_instruments(include_factors=False) if i in change.columns]

    # Predictors storage
    yhat = pd.DataFrame(index=change.index, columns=score_ids, dtype=float)

    # Regime proxy series
    avg_corr_series = pd.Series(index=change.index, dtype=float)
    regime_daily = []
    explain_neighbors_rows = []

    # risk_off series for mixture training
    risk_off_series = pd.Series(index=change.index, dtype=float)

    # Warmup
    min_warmup = max(stat_cfg.window, fund_cfg.beta_window, int(mcfg['expected_return']['train_window']), 260)

    for t_idx in range(min_warmup, len(change.index)):
        date = change.index[t_idx]

        # A_stat(t) using standardized changes window up to t-1
        wstart = max(0, t_idx - stat_cfg.window)
        std_win = std_change.iloc[wstart:t_idx]
        A_stat = build_stat_adjacency(std_win[score_ids], stat_cfg) if score_ids else pd.DataFrame()

        # avg correlation proxy
        if not A_stat.empty:
            C = std_win[score_ids].corr().fillna(0.0).values
            n = C.shape[0]
            if n > 1:
                avg = (C.sum() - np.trace(C)) / (n * (n - 1))
                avg_corr_series.loc[date] = float(avg)

        # A_fund(t) beta similarity + static edges
        B = rolling_betas(change[score_ids], factors, end_idx=t_idx, window=fund_cfg.beta_window)
        A_beta = build_beta_similarity_adjacency(B, top_k=fund_cfg.top_k)
        A_fund = A_beta.copy()
        if A_static_fund is not None and not A_static_fund.empty:
            A_fund = A_fund.add(A_static_fund.reindex_like(A_fund).fillna(0.0), fill_value=0.0)

        # regime features at date
        reg_feats = build_regime_features(
            change=change,
            std_change=std_change,
            stat_avg_corr=avg_corr_series,
            cfg=RegimeFeatureConfig(vol_id="VIX", credit_hy_id="HYG", credit_ig_id="LQD"),
        ).loc[[date]]

        rw = infer_regime_and_weights(reg_feats, regime_cfg).iloc[0].to_dict()
        regime_daily.append({"date": date, **rw})
        risk_off_series.loc[date] = float(rw["risk_off_prob"])

        # W(t)
        W = combine_channels(
            A_stat=A_stat, A_fund=A_fund, A_comp=A_comp,
            w_stat=float(rw["w_stat"]), w_fund=float(rw["w_fund"]), w_comp=float(rw["w_comp"])
        )
        if W.empty:
            continue

        # Predict expected change (fast interpretable baseline)
        rt = change.loc[date, score_ids]
        f_t = factors.loc[date] if date in factors.index else pd.Series(index=factors.columns, dtype=float)
        preds, _ = predict_from_betas_and_network(rt=rt, f_t=f_t, betas=B, W=W, cfg=exp_cfg)
        for i in score_ids:
            if i in preds.index and pd.notna(preds[i]):
                yhat.loc[date, i] = float(preds[i])

        # Neighbor explainability (top neighbors)
        ch_t = change.loc[date, score_ids].dropna()
        if len(ch_t) > 0:
            nb = top_neighbor_contributions(ch_t, W, top_k=5)
            if not nb.empty:
                nb["date"] = date
                explain_neighbors_rows.append(nb)

    # Compute z, residuals, half-life
    z, resid, half_life = compute_z(change[score_ids], yhat, resid_cfg, risk_off_prob=risk_off_series)

    # Mean reversion targets (expected decay + sign flip prob)
    mr = mean_reversion_targets(z, mr_cfg)
    mr["expected_z_H"].to_csv(os.path.join(paths.out_dir, "expected_z_H.csv"), index=True)
    mr["p_sign_flip_H"].to_csv(os.path.join(paths.out_dir, "p_sign_flip_H.csv"), index=True)
    mr["ar1_b"].to_csv(os.path.join(paths.out_dir, "ar1_b.csv"), index=True)

    # Build portfolio with neutralization + vol scaling + confidence sizing
    gross_mult = pd.Series({pd.to_datetime(r["date"]): float(r["gross_mult"]) for r in regime_daily}).sort_index()

    # Use most recent beta matrix B for neutralization if available
    betas_for_neutral = B if ('B' in locals() and B is not None and not B.empty) else None

    port_cfg = PortfolioConfig(
        group_type=str(univ.get("portfolio_group_type", "micro")),
        per_group_long=1,
        per_group_short=1,
        max_abs_weight=0.10,
        gross_target=1.0,
        net_target=0.0,
        z_cap=3.0,
        use_vol_scaling=True,
    )
    neutral_cfg = NeutralizeConfig(
        equity_beta_to="F_MKT_US" if (betas_for_neutral is not None and "F_MKT_US" in betas_for_neutral.columns) else None,
        duration_neutral=True,
        fx_usd_neutral=True,
    )
    weights = build_within_group_ls(reg, z, vol, betas_for_neutral, port_cfg, gross_mult=gross_mult, neutral_cfg=neutral_cfg)

    # Outputs
    z.to_csv(os.path.join(paths.out_dir, "signals_z_wide.csv"), index=True, index_label="date")
    weights.to_csv(os.path.join(paths.out_dir, "weights_wide.csv"), index=True, index_label="date")
    pd.DataFrame(regime_daily).to_csv(os.path.join(paths.out_dir, "regime_daily.csv"), index=False)
    half_life.to_csv(os.path.join(paths.out_dir, "half_life.csv"), header=["half_life_days"])

    if explain_neighbors_rows:
        ex = pd.concat(explain_neighbors_rows, ignore_index=True)
        ex.to_csv(os.path.join(paths.out_dir, "explain_top_neighbors.csv"), index=False)
