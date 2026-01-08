from __future__ import annotations
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from worldflow.run.pipeline import run_end_to_end, Paths  # noqa: E402

def main():
    p = Paths(
        instrument_master=os.path.join(BASE_DIR, "config", "instrument_master.csv"),
        group_master=os.path.join(BASE_DIR, "config", "group_master.csv"),
        membership=os.path.join(BASE_DIR, "config", "instrument_group_membership.csv"),
        bars=os.path.join(BASE_DIR, "data", "processed", "bars.csv"),
        edges_fund=os.path.join(BASE_DIR, "config", "edges_fund.csv"),
        edges_comp=os.path.join(BASE_DIR, "config", "edges_comp.csv"),
        universe_yaml=os.path.join(BASE_DIR, "config", "universe.yaml"),
        channels_yaml=os.path.join(BASE_DIR, "config", "channels.yaml"),
        regimes_yaml=os.path.join(BASE_DIR, "config", "regimes.yaml"),
        model_yaml=os.path.join(BASE_DIR, "config", "model.yaml"),
        curves_yaml=os.path.join(BASE_DIR, "config", "curves.yaml"),
        out_dir=os.path.join(BASE_DIR, "data", "outputs"),
    )
    run_end_to_end(p)
    print("Done. Outputs in data/outputs/")

if __name__ == "__main__":
    main()
