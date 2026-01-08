from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class CurveSpec:
    short: str
    mid: str
    long: str


def compute_level_slope_curvature(yields_wide: pd.DataFrame, spec: CurveSpec) -> pd.DataFrame:
    """Compute simple L/S/C factors (levels in yield space).
    level = average(short, mid, long)
    slope = long - short
    curvature = 2*mid - (short + long)
    """
    y = yields_wide[[spec.short, spec.mid, spec.long]].copy()
    out = pd.DataFrame(index=y.index)
    out["LEVEL"] = y.mean(axis=1)
    out["SLOPE"] = y[spec.long] - y[spec.short]
    out["CURVATURE"] = 2*y[spec.mid] - (y[spec.short] + y[spec.long])
    return out


def compute_curve_factor_changes(curve_levels: pd.DataFrame) -> pd.DataFrame:
    """Return daily changes of curve factors."""
    return curve_levels.diff()
