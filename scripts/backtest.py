from __future__ import annotations
import os, sys
import yaml
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from worldflow.data.registry import load_registry  # noqa: E402
from worldflow.data.store import MarketDataStore  # noqa: E402
from worldflow.data.transforms import compute_change_series, FeatureConfig  # noqa: E402
from worldflow.run.backtest import compute_strategy_pnl, compute_costs, perf_stats, CostConfig  # noqa: E402

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    reg = load_registry(
        os.path.join(BASE_DIR, "config", "instrument_master.csv"),
        os.path.join(BASE_DIR, "config", "group_master.csv"),
        os.path.join(BASE_DIR, "config", "instrument_group_membership.csv"),
    )
    store = MarketDataStore(os.path.join(BASE_DIR, "data", "processed", "bars.csv"))
    bars = store.load_bars()
    wide_px = store.to_wide(bars, "px")
    wide_yield = store.to_wide(bars, "yield")
    wide_spread = store.to_wide(bars, "spread")

    model_cfg = load_yaml(os.path.join(BASE_DIR, "config", "model.yaml"))
    feat_cfg = FeatureConfig(
        ret_clip=float(model_cfg["features"]["ret_clip"]),
        vol_span=int(model_cfg["features"]["vol_span"]),
        yield_to_total_return=bool(model_cfg["features"]["yield_to_total_return"]),
    )
    change, std_change, vol = compute_change_series(reg, wide_px, wide_yield, wide_spread, feat_cfg)

    weights = pd.read_csv(os.path.join(BASE_DIR, "data", "outputs", "weights_wide.csv"), index_col=0, parse_dates=True)
    weights.index.name = "date"
    pnl = compute_strategy_pnl(change, weights)

    costs_cfg = load_yaml(os.path.join(BASE_DIR, "config", "costs.yaml"))
    asset_class_map = {iid: reg.instruments[iid].asset_class for iid in weights.columns if iid in reg.instruments}
    cc = CostConfig(
        asset_class_bps=costs_cfg["asset_class_bps"],
        impact_enabled=bool(costs_cfg.get("impact", {}).get("enabled", False)),
        impact_k=float(costs_cfg.get("impact", {}).get("k", 0.0)),
    )
    costs = compute_costs(weights, asset_class_map, cc)
    net = pnl - costs

    out = pd.DataFrame({"gross_pnl": pnl, "costs": costs, "net_pnl": net})
    out.to_csv(os.path.join(BASE_DIR, "data", "outputs", "backtest_daily.csv"), index=True)

    print("Gross stats:", perf_stats(pnl))
    print("Net stats:", perf_stats(net))

if __name__ == "__main__":
    main()
