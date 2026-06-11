# The following script is used to generate UMAP plots of explosion maps 
# It takes in explosion maps of putative and examined structures as pickled Bunch objects
# Examined maps are transformed into the embedding of AF-generated structure maps
# For visualization, the UMAP embedding is run for one seed.
# Classification is carried out using kNN in a separate script, where UMAP is run for 100 seeds
# Make sure you are working in the Analysis_directory

import os
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
from Analysis.analysis_functions import load_data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# ── 1. Load putative and examined dataset and fit PCA ──────────────────────────────────
original_bunch = load_data("pickled_data/kaib_af.pkl") 
X_original = scaler.fit_transform(original_bunch.data)
y_original = original_bunch.target
target_names_original = original_bunch.target_names

new_bunch = load_data("pickled_data/kaib_og.pkl")
X_new = scaler.transform(new_bunch.data)
y_new = new_bunch.target
target_names_new = new_bunch.target_names

# Optional, used for prpojection of additional AF-structures 
extra_bunch = load_data("pickled_data/kaib_extra_af.pkl")
X_extra = scaler.transform(extra_bunch.data)
y_extra = extra_bunch.target
target_names_extra = extra_bunch.target_names

# PCA & UMAP
pca = PCA(n_components=90, svd_solver='full')
umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=0)
X_original_pca = pca.fit_transform(X_original)
X_original_umap = umap.fit_transform(X_original_pca)   # fit AND transform on original

# ── 2. Project new dataset onto the EXISTING UMAP space ────────────────────────

# IMPORTANT: use transform() not fit_transform() — reuses the original UMAP axes
# Both datasets must have the same number of features for this to work directly.
# If feature counts differ, slice or preprocess X_new first (see note below).
X_new_pca = pca.transform(X_new)
X_new_umap = umap.transform(X_new_pca)

X_extra_pca = pca.transform(X_extra)
X_extra_umap = umap.transform(X_extra_pca)

# ── 3. Project a NULL DATASET onto the EXISTING UMAP space for reference ────────────────────────

# Null: random data with same shape and per-feature statistics as X_null
X_null = np.random.multivariate_normal(X_original.mean(axis=0),
                                        np.cov(X_original.T),
                                        size=3*len(X_new))

y_null = np.zeros(3*len(y_new))
target_names_null = ['Null']
X_null_pca = pca.transform(X_null)
X_null_umap = umap.transform(X_null_pca)
# Compare cluster assignments of X_new_umap vs X_null_umap


# ── 4. Define map colours ────────────────────────

#KaiB
original_colors = [[0.082, 0.851, 0.067], [0.518, 0.941, 0.686], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878]]
new_colors = [[0.929, 0.275, 0.082], [0.929, 0.804, 0.082]]
extra_colors = [[0.596, 0.451, 1.000], [0.698, 0.922, 0.302]]

#Mad2
'''original_colors = [[0.082, 0.851, 0.067], [0.518, 0.941, 0.686], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878]]
new_colors = [[0.929, 0.275, 0.082], [0.929, 0.804, 0.082]]
extra_colors = [[0.596, 0.451, 1.000]]'''

#RfaH
'''original_colors = [[0.082, 0.851, 0.067], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878], 
[226/255, 120/255, 211/255], [210/255, 45/255, 186/255], [135/255, 29/255, 119/255]]
#original_colors = [[0.082, 0.851, 0.067], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878]]
'''

#Null
null_colors = '0.8'

# ── 4. Plot embedding results in 4 panels (3 for RfaH) ────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Change shape to [1,3] for RfaH

# --- Panel 1: Putative dataset only ---
ax0 = axes[0,0]
for i, name in enumerate(target_names_original):
    mask = y_original == i
    ax0.scatter(X_original_umap[mask, 0], X_original_umap[mask, 1],
               label=name, alpha=0.9, edgecolors='k', linewidths=0.4, s=60, color=original_colors[i])
ax0.set_title("(a) AF-predicted putative structures")
ax0.set_xlabel("UMAP dim. 1")
ax0.set_ylabel("UMAP dim. 2")
ax0.legend(title="Class", fontsize=8)

# --- Panel 2: Examined dataset (projected) ---
ax = axes[0,1]
for i, name in enumerate(target_names_original):
    mask = y_original == i
    ax.scatter(X_original_umap[mask, 0], X_original_umap[mask, 1],
               label=f"[Put. (AF)] {name}", alpha=0.1, edgecolors='k',
               linewidths=0.4, s=60, marker='o', color=original_colors[i])

for i, name in enumerate(target_names_new):
    mask = y_new == i
    ax.scatter(X_new_umap[mask, 0], X_new_umap[mask, 1],
               label=f"[Ex. (PDB)] {name}", alpha=0.9, edgecolors='k',
               linewidths=0.4, s=60, marker='s', color=new_colors[i])

ax.set_title("(b) Projection of PDB-sourced examined structures")
ax.set_xlabel("UMAP dim. 1")
ax.set_ylabel("UMAP dim. 2")
ax.legend(fontsize=7, ncol=2, title="● Put. (AF)  ■ Ex. (PDB)")

# --- Panel 3 (Optional, skip for RfaH): Extra dataset (projected) ---
ax = axes[1,0]
for i, name in enumerate(target_names_original):
    mask = y_original == i
    ax.scatter(X_original_umap[mask, 0], X_original_umap[mask, 1],
               label=f"[Put. (AF)] {name}", alpha=0.1, edgecolors='k',
               linewidths=0.4, s=60, marker='o', color=original_colors[i])

for i, name in enumerate(target_names_extra):
    mask = y_extra == i
    ax.scatter(X_extra_umap[mask, 0], X_extra_umap[mask, 1],
               label=f"[Ex. (AF)] {name}", alpha=0.9, edgecolors='k',
               linewidths=0.4, s=60, marker='s', color=extra_colors[i])

ax.set_title("(c) Projection of AF-generated examined structures")
ax.set_xlabel("UMAP dim. 1")
ax.set_ylabel("UMAP dim. 2")
ax.legend(fontsize=7, ncol=2, title="● Put. (AF)  ■ Ex. (AF)")

# --- Panel 4: Null dataset (projected) ---
ax = axes[1,1]

for i, name in enumerate(target_names_original):
    mask = y_original == i
    ax.scatter(X_original_umap[mask, 0], X_original_umap[mask, 1],
               label=f"[Put. (AF)] {name}", alpha=0.8, edgecolors='k',
               linewidths=0.4, s=60, marker='o', color=original_colors[i])
    
ax.scatter(X_null_umap[:,0], X_null_umap[:,1], label='Null', alpha=0.3, edgecolors='k', linewidths=0.4,
               s=60, marker='s', color=null_colors)
ax.set_title("(d) Projection of null data")
ax.set_xlabel("UMAP dim. 1")
ax.set_ylabel("UMAP dim. 2")
ax.legend(fontsize=7, ncol=2, title="● Put. (AF)  ■ Null")

plt.suptitle("Explosion maps in UMAP embedding space", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("KaiB_UMAP_class.png", dpi=150, bbox_inches='tight')
plt.show()
