from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence
import pandas as pd


@dataclass
class MarketDataStore:
    bars_path: str  # long format: date,instrument_id,field,value

    def load_bars(
        self,
        instrument_ids: Optional[Sequence[str]] = None,
        fields: Optional[Sequence[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        df = pd.read_csv(self.bars_path, parse_dates=["date"])
        if instrument_ids is not None:
            df = df[df["instrument_id"].isin(list(instrument_ids))]
        if fields is not None:
            df = df[df["field"].isin(list(fields))]
        if start is not None:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["date"] <= pd.to_datetime(end)]
        df = df.sort_values(["date", "instrument_id", "field"]).reset_index(drop=True)
        return df

    def to_wide(self, bars: pd.DataFrame, field: str) -> pd.DataFrame:
        df = bars[bars["field"] == field].pivot(index="date", columns="instrument_id", values="value").sort_index()
        df.index = pd.to_datetime(df.index)
        return df
