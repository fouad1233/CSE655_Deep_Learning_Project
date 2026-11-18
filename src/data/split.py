"""Fixed, seeded 70/15/15 train/val/test split of the clean UTKFace list, stratified by age band x race.

Every model (mean baseline, standard models, own CNN) reads this one file, so all are compared on
exactly the same images. The test set is used once per final model, never for tuning.

How the stratified split works (per stratum = one age band x race cell, 50 cells):
    1. shuffle the stratum's rows with a generator seeded by SEED
    2. n_test = round(0.15 * n), n_val = round(0.15 * n), n_train = n - n_test - n_val
    3. first n_test rows -> test, next n_val -> val, rest -> train
  Example: stratum "20-29, White", n = 1935 -> test round(290.25) = 290, val 290, train 1355.
  Tiny strata stay valid: n = 2 (70-79, Others) -> 0 / 0 / 2.
  sklearn's train_test_split(stratify=...) cannot do this: it fails on strata with fewer than 2 rows per split.

Input : data/processed/utkface_clean.csv   (23,053 rows, from notebook section 9)
Output: splits/utkface_split_seed655.csv   (file, age, gender, race, age_band, split); committed to git

Usage:  python src/data/split.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 655                      # the project seed (same as the notebook)
FRAC_VAL = FRAC_TEST = 0.15     # train gets the rest (~0.70)
BANDS = [0, 3, 13, 20, 30, 40, 50, 60, 70, 80, 117]   # same bands as the notebook, lower bound inclusive
LABELS = ["0-2", "3-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

REPO = Path(__file__).resolve().parents[2]
CLEAN_CSV = REPO / "data" / "processed" / "utkface_clean.csv"
OUT_CSV = REPO / "splits" / f"utkface_split_seed{SEED}.csv"


def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = pd.cut(df.age, bins=BANDS, labels=LABELS, right=False).astype(str)
    df["split"] = ""
    rng = np.random.default_rng(seed)
    # sorted iteration order + sorted rows inside each stratum -> result depends only on the data and the seed
    for (_, _), idx in sorted(df.groupby(["age_band", "race"]).groups.items()):
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_test, n_val = round(FRAC_TEST * n), round(FRAC_VAL * n)
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"
        df.loc[idx[n_test + n_val:], "split"] = "train"
    return df


def main() -> None:
    clean = pd.read_csv(CLEAN_CSV).sort_values("file").reset_index(drop=True)
    df = make_split(clean)

    # sanity checks
    assert (df.split != "").all(), "every image must get a split"
    assert df.file.is_unique and len(df) == len(clean)
    assert df.md5.is_unique, "a duplicate image would leak between splits"

    OUT_CSV.parent.mkdir(exist_ok=True)
    df[["file", "age", "gender", "race", "age_band", "split"]].to_csv(OUT_CSV, index=False)

    n = df.split.value_counts().reindex(["train", "val", "test"])
    print(f"wrote {OUT_CSV.relative_to(REPO)}  ({len(df)} images)")
    print((n.astype(str) + "  (" + (100 * n / len(df)).round(2).astype(str) + " %)").to_string())
    print("\nage per split (mean / median / std):")
    print(df.groupby("split").age.agg(["mean", "median", "std"]).round(2).reindex(["train", "val", "test"]))
    print("\nrace share per split (%):")
    print((100 * pd.crosstab(df.split, df.race, normalize="index")).round(2).reindex(["train", "val", "test"]))


if __name__ == "__main__":
    main()
