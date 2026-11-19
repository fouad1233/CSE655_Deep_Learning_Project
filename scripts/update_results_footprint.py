"""Recompute the footprint columns of results/results.csv from logs/runs.csv (CSV arithmetic only, no model is run).

Columns written for every model row:
  train_s          wall-clock time of the final training run (runs.csv `duration_s`, kind == train, status == ok)
  project_kwh      energy of ALL tracked runs of that architecture family
  project_kg_co2   emissions of ALL tracked runs of that family
Family = every run whose name contains the family key: "own_cnn", "resnet18" or "mobilenet_v3".
This includes searches, failed runs, smoke tests, test passes, inference benchmarks and the balanced-loss
ResNet18 experiment. Shared runs (cache, code tests, baselines, the bias evaluation) belong to no family.

Usage:  python scripts/update_results_footprint.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
runs = pd.read_csv(REPO / "logs" / "runs.csv")
res = pd.read_csv(REPO / "results" / "results.csv")
FAMILIES = ("own_cnn", "resnet18", "mobilenet_v3")


def family(run):
    return next((f for f in FAMILIES if isinstance(run, str) and f in run), None)


runs["family"] = runs["run"].map(family)
fam = runs.groupby("family")[["energy_kwh", "kg_co2"]].sum()
train = runs[(runs["kind"] == "train") & (runs["status"] == "ok")].set_index("run")["duration_s"]

for i, row in res.iterrows():
    f = family(row.get("run"))
    if f is None:                      # baselines: no training, no model to time -> empty, not 0
        for col in ("train_kwh", "train_kg_co2", "ms_per_img"):
            if col in res.columns:
                res.loc[i, col] = float("nan")
        continue
    res.loc[i, "train_s"] = float(train.get(row["run"], float("nan")))
    res.loc[i, "project_kwh"] = fam.loc[f, "energy_kwh"]
    res.loc[i, "project_kg_co2"] = fam.loc[f, "kg_co2"]

res.to_csv(REPO / "results" / "results.csv", index=False)
print(fam.assign(Wh=fam.energy_kwh * 1e3, g=fam.kg_co2 * 1e3)[["Wh", "g"]].round(2))
print(res[["model", "train_s", "train_kwh", "project_kwh", "project_kg_co2"]].to_string(index=False))
total = runs[["energy_kwh", "kg_co2"]].sum()
print(f"all {len(runs)} runs: {total.energy_kwh * 1e3:.2f} Wh, {total.kg_co2 * 1e3:.2f} g CO2eq")
