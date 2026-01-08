from __future__ import annotations
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_BARS = os.path.join(BASE_DIR, "data", "processed", "bars.csv")

def main():
    os.makedirs(os.path.dirname(OUT_BARS), exist_ok=True)

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", "2025-12-31", freq="B")
    n = len(dates)

    # Latent factors: equity, usd, rates, oil, gold, vol, credit
    F = pd.DataFrame(index=dates)
    F["EQ"] = rng.normal(0, 0.008, n)
    F["USD"] = rng.normal(0, 0.004, n)
    F["RATES"] = rng.normal(0, 0.0007, n)  # yield changes
    F["OIL"] = rng.normal(0, 0.012, n)
    F["GOLD"] = rng.normal(0, 0.007, n)
    spikes = (rng.random(n) < 0.03).astype(float)
    F["VOL"] = rng.normal(0, 0.02, n) + spikes * rng.normal(0.10, 0.05, n)
    F["CREDIT"] = rng.normal(0, 0.004, n) + spikes * rng.normal(0.02, 0.01, n)

    instruments = {
        "SPY": ("px", 320.0, {"EQ": 1.0, "USD": -0.1, "OIL": 0.05, "GOLD": -0.02, "VOL": -0.2}),
        "QQQ": ("px", 220.0, {"EQ": 1.2, "USD": -0.1, "OIL": 0.03, "GOLD": -0.02, "VOL": -0.25}),
        "EURUSD": ("px", 1.10, {"USD": -0.9, "EQ": 0.1, "VOL": 0.05}),
        "USDJPY": ("px", 110.0, {"USD": 0.7, "EQ": -0.05, "VOL": 0.03}),
        "GOLD": ("px", 1500.0, {"GOLD": 1.0, "USD": -0.2, "VOL": 0.05}),
        "OIL": ("px", 60.0, {"OIL": 1.0, "EQ": 0.1, "USD": -0.05, "VOL": 0.05}),
        "HYG": ("px", 85.0, {"EQ": 0.4, "VOL": -0.25, "RATES": -0.1, "CREDIT": 0.6}),
        "LQD": ("px", 120.0, {"RATES": -0.4, "VOL": -0.05, "CREDIT": 0.2}),
        "VIX": ("px", 18.0, {"VOL": 1.0, "EQ": -0.3}),

        # Yield curve points
        "US2Y_YIELD": ("yield", 0.015, {"RATES": 1.2, "EQ": 0.02, "VOL": 0.08}),
        "US10Y_YIELD": ("yield", 0.018, {"RATES": 1.0, "EQ": 0.05, "VOL": 0.10}),
        "US30Y_YIELD": ("yield", 0.022, {"RATES": 0.8, "EQ": 0.07, "VOL": 0.12}),
        "TBILL3M_YIELD": ("yield", 0.012, {"RATES": 1.5, "VOL": 0.03}),

        # Factor instruments
        "MKT_US": ("px", 320.0, {"EQ": 1.0, "VOL": -0.2}),
        "USD": ("px", 100.0, {"USD": 1.0}),
        "RATES_US10Y": ("yield", 0.018, {"RATES": 1.0}),
        "VOL": ("px", 18.0, {"VOL": 1.0}),
    }

    bars_rows = []
    for iid, (field, start, loads) in instruments.items():
        if field == "px":
            r = np.zeros(n)
            for k, b in loads.items():
                if k == "RATES":
                    r += b * (F[k].values * 5.0)
                else:
                    r += b * F[k].values
            r += rng.normal(0, 0.003, n)
            px = start * np.exp(np.cumsum(r))
            for d, v in zip(dates, px):
                bars_rows.append({"date": d, "instrument_id": iid, "field": "px", "value": float(v)})
        elif field == "yield":
            dy = np.zeros(n)
            for k, b in loads.items():
                dy += b * F[k].values
            dy += rng.normal(0, 0.0002, n)
            y = start + np.cumsum(dy)
            y = np.clip(y, 0.0001, 0.08)
            for d, v in zip(dates, y):
                bars_rows.append({"date": d, "instrument_id": iid, "field": "yield", "value": float(v)})

    bars = pd.DataFrame(bars_rows).sort_values(["date", "instrument_id", "field"])
    bars.to_csv(OUT_BARS, index=False)
    print("Wrote:", OUT_BARS, "rows:", len(bars))

if __name__ == "__main__":
    main()
