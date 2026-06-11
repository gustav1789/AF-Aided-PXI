"""
Compute mean TM-score and lDDT from the pairwise scores HDF5 file.
The output CSV compares ProteinA and ProteinB with means and std as well as summing all legit computations

Output: a CSV with one row per protein pair containing:
    - protein_A, protein_B
    - mean_tm_score, std_tm_score
    - mean_lddt,     std_lddt
    - n_valid_tm,    n_valid_lddt   (non-NaN pairs counted)
"""

# ──────────────────────────────────────────────
# CONFIGUR Change Names For Desired Proteins
# ──────────────────────────────────────────────
INPUT_H5   = "scores_rfah.h5"
OUTPUT_CSV = "mean_scores_rfah.csv"
# ──────────────────────────────────────────────

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def summarise_h5(h5_path: str, csv_path: str) -> pd.DataFrame:
    rows = []

    with h5py.File(h5_path, "r") as h5:
        for key in sorted(h5.keys()):
            grp  = h5[key]
            tm   = grp["tm_score"][:]   # (n_sims_A, n_sims_B) float32
            lddt = grp["lddt"][:]

            # pull protein names from attributes if present, else parse key
            pA = grp.attrs.get("protein_A", key.split("_vs_")[0])
            pB = grp.attrs.get("protein_B", key.split("_vs_")[1])

            rows.append({
                "protein_A":     pA,
                "protein_B":     pB,
                "mean_tm_score": float(np.nanmean(tm)),
                "std_tm_score":  float(np.nanstd(tm)),
                "mean_lddt":     float(np.nanmean(lddt)),
                "std_lddt":      float(np.nanstd(lddt)),
                "n_valid_tm":    int(np.sum(~np.isnan(tm))),
                "n_valid_lddt":  int(np.sum(~np.isnan(lddt))),
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, float_format="%.4f")
    return df


def pretty_print(df: pd.DataFrame) -> None:
    col_w = max(df["protein_A"].str.len().max(), df["protein_B"].str.len().max())
    header = (f"{'Protein A':<{col_w}}  {'Protein B':<{col_w}}"
              f"  {'TM mean':>8}  {'TM std':>7}"
              f"  {'lDDT mean':>9}  {'lDDT std':>8}"
              f"  {'N (TM)':>7}  {'N (lDDT)':>8}")
    print(header)
    print("─" * len(header))
    for _, r in df.iterrows():
        print(
            f"{r.protein_A:<{col_w}}  {r.protein_B:<{col_w}}"
            f"  {r.mean_tm_score:>8.4f}  {r.std_tm_score:>7.4f}"
            f"  {r.mean_lddt:>9.4f}  {r.std_lddt:>8.4f}"
            f"  {int(r.n_valid_tm):>7}  {int(r.n_valid_lddt):>8}"
        )


if __name__ == "__main__":
    h5_path = Path(INPUT_H5).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    print(f"Reading: {h5_path}\n")
    df = summarise_h5(str(h5_path), OUTPUT_CSV)

    pretty_print(df)
    print(f"\nSaved to: {Path(OUTPUT_CSV).resolve()}")