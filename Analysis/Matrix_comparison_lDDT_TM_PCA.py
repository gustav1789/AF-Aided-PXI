import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

"""
Changes the order of the CSV from multi_score_mean into the matrix format desired
and plots lddt and tm against PCA euclidean CSV that must be added as well. 
For RfaH the names where correct directely and no name-swap dictionary was needed (remove the part where it is used)
"""


# Mapping from CSV names to correct labels, in desired order
order_kaib = ["KaiB-ground", "KaiB-α1", "KaiB-α2", "KaiB-α3", "KaiB-FS", "KaiB-α4", "KaiB-α5", "KaiB-α6"]
order_mad2 = ["Mad2-closed", "Mad2-α1", "Mad2-α2", "Mad2-open", "Mad2-α3", "Mad2-α4", "Mad2-α5"]
order_rfah = ["RfaH-AIn","RfaH-α1","RfaH-active","RfaH-α2","RfaH-α3","RfaH-α4","RfaH-α5","RfaH-α6"]

name_map_kaib = {
    "4KSOchainA":       "KaiB-ground",
    "KaiB_033":         "KaiB-α1",
    "KaiB_102":         "KaiB-α2",
    "KaiB_114":         "KaiB-α3",
    "fixed_5N8YchainG": "KaiB-FS",
    "KaiB_048":         "KaiB-α4",
    "KaiB_072":         "KaiB-α5",
    "KaiB_094":         "KaiB-α6",
}

name_map_mad2 = {
    "1S2H":         "Mad2-closed",
    "1DUJ":         "Mad2-open",
    "cluster_055":  "Mad2-α1",
    "cluster_081":  "Mad2-α2",
    "cluster_019":  "Mad2-α3",
    "cluster_027":  "Mad2-α4",
    "cluster_087":  "Mad2-α5",
    
}
#---------------------------------------
#CHANGE NAMES
order = order_mad2  #<-----
name_map = name_map_mad2  #<------
protein = "mad2"    #<------
title_p = "Mad2"    #<-------
df = pd.read_csv(f"mean_scores_{protein}.csv")
PCA_score = ["ed", "2D"]        #<------- controlls post exlosion metric
pca_mat = pd.read_csv(f"{protein}_{PCA_score[0]}_matrix_{PCA_score[1]}.csv", index_col=0)
#---------------------------------------

n = len(order)
tm_mat   = pd.DataFrame(np.nan, index=order, columns=order)
lddt_mat = pd.DataFrame(np.nan, index=order, columns=order)


for _, row in df.iterrows():
    a = name_map[row["protein_A"]]  #remove name_map if not needed (RfaH)
    b = name_map[row["protein_B"]]
    tm   = row["mean_tm_score"]
    lddt = row["mean_lddt"]
    # fill both directions (symmetric)
    tm_mat.loc[a, b]   = tm
    tm_mat.loc[b, a]   = tm
    lddt_mat.loc[a, b] = lddt
    lddt_mat.loc[b, a] = lddt

print("=== TM-score matrix ===")
print(tm_mat.to_string(float_format="%.4f"))
print()
print("=== lDDT matrix ===")
print(lddt_mat.to_string(float_format="%.4f"))

tm_mat.to_csv(f"tm_score_matrix_{protein}.csv")
lddt_mat.to_csv(f"lddt_matrix_{protein}.csv")
print("\nSaved: tm_score_matrix.csv, lddt_matrix.csv")

# ── Plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3,
    figsize=(24, 6),
    gridspec_kw={"wspace": 0.1},
)
 
datasets = [
    (tm_mat,   "TM-score",       "viridis", None,  None),
    (lddt_mat, "lDDT",           "viridis",    None,  None),
    (pca_mat,  "PCA Euclidean Distance",   "viridis_r",  None,  None),
]
 
for ax, (mat, title, cmap, vmin, vmax) in zip(axes, datasets):
    data = mat.astype(float)
    # auto range if not specified
    lo = vmin if vmin is not None else np.nanmin(data.values)
    hi = vmax if vmax is not None else np.nanmax(data.values)
 
    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        linewidths=0,        
        square=True,
        cbar=True,
        cbar_kws={"shrink": 0.8, "pad": 0.02},
        annot_kws={"size": 8},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
 
plt.suptitle(
    f"{title_p}",
    fontsize=25, y=1,
)
plt.savefig(f"{protein}_heatmaps.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {protein}_heatmaps.png")
