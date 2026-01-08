from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    vendor_symbol: str
    asset_class: str
    data_type: str           # price, yield, spread, etc.
    field: str               # px, yield, spread
    quote_ccy: str
    base_ccy: str
    is_factor: bool = False
    active_from: Optional[str] = None
    active_to: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    channel: str             # stat, fund, comp
    weight: float
    asof_date: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
