# worldflow

A multi-asset, regime-aware network residual (mispricing) engine.

This repo is designed to scale from a small global macro universe to thousands of instruments
(e.g., the full S&P 500 + small-cap reps + FX/rates/commodities) without changing core code.
You expand by adding rows to the instrument registry and appending market data in long format.

## What it does (v1)
- Ingest long-format market data (price/yield/spread/etc.) using an **instrument registry**.
- Compute comparable "changes" (log returns for prices, diffs for yields/spreads).
- Build a 3-channel network:
  - `A_stat(t)`: rolling correlation on standardized changes (shrunk + sparsified)
  - `A_fund`: factor-beta similarity graph + optional static edge list
  - `A_comp`: optional negative edges (explicit mapping)
- Infer soft regimes (risk-off vs risk-on) and set channel weights α,β,γ.
- Fit per-instrument expected-change model:
  r_i(t) = b_i^T f(t) + λ Σ_j W_ij(t) r_j(t) + ε_i(t)
- Output standardized residuals z_i(t), half-life estimates, and explainability.

## Quickstart (runs with synthetic data)
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 1) generate synthetic multi-asset dataset
python scripts/make_synth_data.py

# 2) run the full pipeline
python scripts/run_daily.py
```

Outputs are written to `data/outputs/`:
- `signals_z_wide.csv`  (Date x Instrument z-scores)
- `weights_wide.csv`    (basic within-group L/S portfolio weights)
- `regime_daily.csv`    (risk_off probability + channel weights)
- `explain_top_neighbors.csv` (top neighbor attributions per instrument/day)

## Data model (expandable)
### Instrument registry (config/instrument_master.csv)
Defines instruments and how to transform them.
Adding 500 new equities later is just adding 500 rows.

### Market data fact table (data/processed/bars.csv)
Long format:
- date, instrument_id, field, value

Supported v1 fields:
- `px` for price-like instruments
- `yield` for yield instruments
- `spread` for spread instruments

## Replacing synthetic data with real data
When your Wharton feed arrives:
- Append real observations into `data/processed/bars.csv` (same schema)
- Expand `config/instrument_master.csv` and group membership
- Re-run `scripts/run_daily.py`

## Notes for scaling to thousands of equities
- Keep networks sparse (top-k neighbors, within-group edges, macro anchors).
- Use group memberships to compute stat edges within micro-groups + small cross-group bridges.
- Persist features and edges incrementally to avoid recomputation.
