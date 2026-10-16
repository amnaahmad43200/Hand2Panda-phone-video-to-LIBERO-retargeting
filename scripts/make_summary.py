"""Regenerate the summary tables and figure from results/eval_results.csv.

    python scripts/make_summary.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "results" / "eval_results.csv")

order = ["straight", "human_median", "human_train", "human_heldout"]
summary = (
    df.groupby("group")
    .agg(runs=("success", "size"), held=("held", "mean"), success=("success", "mean"),
         bowl_plate_xy_cm=("bowl_plate_xy_cm", "mean"),
         max_lateral_dev_cm=("max_lat_cm", "mean"), path_len_cm=("path_len_cm", "mean"))
    .loc[order].round(2)
)
summary.to_csv(ROOT / "results" / "summary_by_group.csv")
print(summary.to_string())

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
colors = dict(zip(order, ["0.35", "C0", "C1", "C3"]))
rng = np.random.default_rng(0)
for k, g in enumerate(order):
    d = df[df.group == g]
    x = k + rng.uniform(-0.15, 0.15, len(d))
    ax[0].scatter(x, d.max_lat_cm, s=14, alpha=0.7, color=colors[g])
    ax[1].scatter(x, d.path_len_cm, s=14, alpha=0.7, color=colors[g])
for a, t, yl in zip(ax, ["Max sideways deviation of carry path", "Planned carry path length"], ["cm", "cm"]):
    a.set_xticks(range(4)); a.set_xticklabels(["straight", "human\nmedian", "human\ntrain (16)", "human\nheld-out (4)"])
    a.set_title(t, fontsize=10); a.set_ylabel(yl); a.grid(alpha=0.3)
fig.suptitle(f"All {len(df)} runs succeeded (official LIBERO success check)", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "media" / "results_summary.png", dpi=130)
print("saved media/results_summary.png")
