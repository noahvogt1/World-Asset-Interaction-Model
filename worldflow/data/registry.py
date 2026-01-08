from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from worldflow.core.types import Instrument


@dataclass
class Registry:
    instruments: Dict[str, Instrument]
    group_master: pd.DataFrame
    membership: pd.DataFrame

    def list_instruments(self, *, include_factors: bool = False) -> List[str]:
        ids = []
        for iid, inst in self.instruments.items():
            if include_factors or not inst.is_factor:
                ids.append(iid)
        return sorted(ids)

    def list_factors(self) -> List[str]:
        return sorted([iid for iid, inst in self.instruments.items() if inst.is_factor])

    def instrument(self, instrument_id: str) -> Instrument:
        return self.instruments[instrument_id]

    def groups_for_instrument(self, instrument_id: str, group_type: Optional[str] = None) -> pd.DataFrame:
        df = self.membership[self.membership["instrument_id"] == instrument_id].copy()
        if group_type is not None:
            df = df.merge(self.group_master, on="group_id", how="left")
            df = df[df["group_type"] == group_type]
        return df

    def instruments_in_group(self, group_id: str) -> List[str]:
        df = self.membership[self.membership["group_id"] == group_id]
        return sorted(df["instrument_id"].unique().tolist())

    def group_ids(self, group_type: str) -> List[str]:
        return sorted(self.group_master[self.group_master["group_type"] == group_type]["group_id"].tolist())


def load_registry(
    instrument_master_csv: str,
    group_master_csv: str,
    membership_csv: str
) -> Registry:
    im = pd.read_csv(instrument_master_csv)
    gm = pd.read_csv(group_master_csv)
    mem = pd.read_csv(membership_csv)

    instruments: Dict[str, Instrument] = {}
    for _, r in im.iterrows():
        meta = None
        if isinstance(r.get("meta_json", None), str) and r["meta_json"].strip():
            try:
                meta = json.loads(r["meta_json"])
            except Exception:
                meta = {"raw": r["meta_json"]}

        inst = Instrument(
            instrument_id=str(r["instrument_id"]),
            vendor_symbol=str(r["vendor_symbol"]),
            asset_class=str(r["asset_class"]),
            data_type=str(r["data_type"]),
            field=str(r["field"]),
            quote_ccy=str(r["quote_ccy"]),
            base_ccy=str(r["base_ccy"]),
            is_factor=bool(str(r.get("is_factor", "false")).lower() == "true"),
            active_from=str(r.get("active_from", "")) if pd.notna(r.get("active_from", None)) else None,
            active_to=str(r.get("active_to", "")) if pd.notna(r.get("active_to", None)) else None,
            meta=meta,
        )
        instruments[inst.instrument_id] = inst

    # basic validation
    missing = set(mem["instrument_id"].unique()) - set(instruments.keys())
    if missing:
        raise ValueError(f"Membership references missing instruments: {sorted(missing)[:10]}")

    return Registry(instruments=instruments, group_master=gm, membership=mem)
