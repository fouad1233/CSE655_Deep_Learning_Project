"""Test MAE vs CO2 (log scale), one point per configuration, Pareto front marked, two panels:
  (a) CO2 of the final training run        (column train_kg_co2 of results/results.csv)
  (b) CO2 of the whole project per family   (column project_kg_co2; run update_results_footprint.py first)
Plotting only, reads results/results.csv.  ->  results/figures/error_vs_co2_final_and_project.png

Usage:  python scripts/plot_error_vs_co2.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

REPO = Path(__file__).resolve().parents[1]
res = pd.read_csv(REPO / "results" / "results.csv")
res = res[res["run"].notna()].copy()
LIMIT = 1.15 * res.loc[~res["run"].str.contains("own_cnn"), "test_mae"].min()      # 1.15 x MAE_best

SHORT = {"resnet18_A_run02": "ResNet18 A", "resnet18_full_run01": "ResNet18 full",
         "mobilenet_v3_large_A_run02": "MobileNetV3-L A", "mobilenet_v3_large_full_run01": "MobileNetV3-L full",
         "own_cnn_run01": "own CNN v1 (1 block/stage)", "own_cnn_bottleneck_run01": "own CNN bottleneck (ablation)",
         "own_cnn_d2_run02": "own CNN final (2 blocks/stage)"}
# label offsets (points) where the default would overlap another label or the Pareto line
OFFSET = {("train_kg_co2", "resnet18_A_run02"): (6, 6), ("train_kg_co2", "own_cnn_d2_run02"): (8, -16),
          ("train_kg_co2", "mobilenet_v3_large_full_run01"): (-10, 9),
          ("project_kg_co2", "mobilenet_v3_large_full_run01"): (8, 4)}
STYLE = {"resnet18": ("tab:blue", "o"), "mobilenet": ("tab:green", "s"), "own_cnn": ("tab:red", "*")}


def style(run):
    return next(v for k, v in STYLE.items() if k in run)


def pareto(df, x):
    """points that no other point beats on both x (lower) and test MAE (lower)"""
    pts = df.sort_values([x, "test_mae"])
    front, best = [], float("inf")
    for _, r in pts.iterrows():
        if r["test_mae"] < best:
            front.append((r[x] * 1e3, r["test_mae"]))
            best = r["test_mae"]
    return front


fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, x, title in ((axes[0], "train_kg_co2", "(a) CO$_2$ of the final training run"),
                     (axes[1], "project_kg_co2", "(b) CO$_2$ of the whole project (all runs of the family)")):
    for _, r in res.iterrows():
        c, m = style(r["run"])
        ax.scatter(r[x] * 1e3, r["test_mae"], color=c, marker=m, s=170 if m == "*" else 70, zorder=3)
    # labels: stack the points that share an x value (panel b) so they do not overlap
    for _, r in res.sort_values("test_mae").iterrows():
        dx, dy = OFFSET.get((x, r["run"]), (8, -3))
        ax.annotate(SHORT[r["run"]], (r[x] * 1e3, r["test_mae"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5)
    fx, fy = zip(*pareto(res, x))
    ax.step(fx, fy, where="post", color="k", ls="--", lw=1, label="Pareto front", zorder=2)
    ax.axhline(LIMIT, color="0.4", ls=":", label=f"acceptability limit {LIMIT:.3f}")
    ax.set_xscale("log")
    ticks = [1, 1.5, 2, 3, 4, 5, 6] if x == "train_kg_co2" else [6, 8, 10, 12, 15, 20]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    lo, hi = res[x].min() * 1e3, res[x].max() * 1e3
    ax.set_xlim(lo / 1.25, hi * 2.1)                  # room on the right for the labels
    ax.set_xlabel("g CO$_2$eq (log scale)"); ax.set_ylabel("test MAE (years)")
    ax.set_title(title, fontsize=10.5); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.5, loc="center left")
plt.tight_layout()
out = REPO / "results" / "figures" / "error_vs_co2_final_and_project.png"
plt.savefig(out, dpi=150)
print("saved", out.relative_to(REPO), "limit", round(LIMIT, 3))
