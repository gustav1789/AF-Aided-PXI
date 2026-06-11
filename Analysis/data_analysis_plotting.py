import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

from Analysis.analysis_functions import load_data
from Analysis.analysis_functions import PCA_matrix

dataset = load_data("pickled_data/rfah_tutti.pkl")  # Expects pickled sklearn Bunch
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data = scaler.fit_transform(dataset.data) # Scale the data

# --- PCA Explained Variance Analysis ---
pca_full = PCA(svd_solver = 'full').fit(data)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar chart of individual explained variance
ax1.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_ * 100,
        alpha=0.6, color='steelblue', label='Individual')
ax1.set_xlabel('Number of Components')
ax1.set_ylabel('Explained Variance (%)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Overlay cumulative variance line on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(range(1, len(cumulative_variance) + 1),
         cumulative_variance * 100,
         color='tomato', marker='o', markersize=3, linewidth=2, label='Cumulative')
ax2.set_ylabel('Cumulative Explained Variance (%)', color='tomato')
ax2.tick_params(axis='y', labelcolor='tomato')
ax2.set_ylim(0, 100)

# Mark common thresholds
for threshold in [80, 90, 95]:
    n_components = np.argmax(cumulative_variance >= threshold / 100) + 1
    ax2.axhline(y=threshold, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
    ax2.text(len(cumulative_variance) * 0.65, threshold + 0.5,
             f'{threshold}% → {n_components} components', fontsize=8, color='grey')

ax1.set_title('PCA Explained Variance per Component')
fig.legend(loc='center right', bbox_to_anchor=(0.88, 0.5))
plt.tight_layout()
plt.savefig("rfah_pca_explained_variance.png", dpi=150, bbox_inches="tight")
plt.clf()

# --- End Explained Variance Analysis ---

#PCA to 90D
X_reduced_extra = PCA_matrix(dataset, 90, 'RfaH_ed_matrix_90D',  metric = 'distance', scaler = 'standard') # Using 90 components for classifier. The elbow seems to be here

# Definition of protein colours

# It assumes data with proteins ordered according to the following scheme:
# Mad2: ['Mad2-closed', 'Mad2-α1', 'Mad2-α2', 'Mad2-open', 'Mad2-α3', 'Mad2-α4', 'Mad2-α5']
# KaiB: ['KaiB-ground', 'KaiB-α1', 'KaiB-α2', 'KaiB-α3', 'KaiB-FS', 'KaiB-α4', 'KaiB-α5', 'KaiB-α6']
# RfaH: ['RfaH-AIn', 'RfaH-α1', 'RfaH-active', 'RfaH-α2', 'RfaH-α3', 'RfaH-α4', 'RfaH-α5', 'RfaH-α6']

# KaiB:
#colors = [[0.929, 0.275, 0.082], [0.082, 0.851, 0.067], [0.518, 0.941, 0.686], [0.698, 0.922, 0.302], [0.929, 0.804, 0.082], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878], [0.596, 0.451, 1.000]]

# Mad2
#colors = [[0.929, 0.275, 0.082], [0.082, 0.851, 0.067], [0.518, 0.941, 0.686], [0.929, 0.804, 0.082], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878], [0.596, 0.451, 1.000]]

#RfaH
colors = [[0.929, 0.275, 0.082], [0.082, 0.851, 0.067], [0.929, 0.804, 0.082], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878], 
[226/255, 120/255, 211/255], [210/255, 45/255, 186/255], [135/255, 29/255, 119/255]]
# With the outlier RfaH-active removed:
#colors = [[0.929, 0.275, 0.082], [0.082, 0.851, 0.067], [0.275 , 0.533, 0.831], [0.263 , 0.255, 0.878], 
#[226/255, 120/255, 211/255], [210/255, 45/255, 186/255], [135/255, 29/255, 119/255]]


color_array = [colors[t] for t in dataset.target]
#og_color_array = [og_colors[t] for t in X_clustered]


#3D PCA

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401

X_reduced = PCA_matrix(dataset, 3, 'RfaH_ed_matrix_3D', metric = 'distance', scaler='standard')

fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=color_array,
    s=40
)

ax.set(
    title="First three principal components",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
    zlabel="3rd Principal Component",
)

# Add a legend
import matplotlib.patches as mpatches

handles = [mpatches.Patch(color=colors[i], label=dataset.target_names[i])
           for i in range(len(colors))]

legend1 = ax.legend(
    handles=handles,
    loc="upper right",
    title="Proteins",
)
ax.add_artist(legend1)
plt.savefig("RfaH_pca_3D.png", dpi=150, bbox_inches="tight")
plt.clf()


#2D PCA

X_reduced = PCA_matrix(dataset, 2, 'RfaH_ed_matrix_2D', csv_name='rfah_hacked_ed_matrix_2D', metric = 'distance', scaler='standard')

fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=color_array,
    s=40
)

ax.set(
    title="First two principal components",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
)

import matplotlib.patches as mpatches

handles = [mpatches.Patch(color=colors[i], label=dataset.target_names[i])
           for i in range(len(colors))]

legend1 = ax.legend(
    handles=handles,
    loc="lower right",
    title="Proteins",
)
ax.add_artist(legend1)

plt.savefig("RfaH_pca_2D.png", dpi=150, bbox_inches="tight")


#2D TSNE

fig, ax = plt.subplots(figsize=(8, 6))

X_reduced = TSNE(n_components=2, random_state=42).fit_transform(X_reduced_extra)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=color_array,
    s=40
)

ax.set(
    title="t-SNE Components",
    xlabel="t-SNE dim. 1",
    ylabel="t-SNE dim. 2",
)
ax.set_xticklabels([])
ax.set_yticklabels([])

import matplotlib.patches as mpatches

handles = [mpatches.Patch(color=colors[i], label=dataset.target_names[i])
           for i in range(len(colors))]

legend1 = ax.legend(
    handles=handles,
    loc="upper right",
    title="Proteins",
)
ax.add_artist(legend1)

plt.savefig("RfaH_tsne.png", dpi=150, bbox_inches="tight")


#2D UMAP
reducer = umap.UMAP(n_neighbors=200, n_components=2, random_state=27)
fig, ax = plt.subplots(figsize=(8, 6))

X_reduced = reducer.fit_transform(X_reduced_extra)
scatter = ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=color_array,
    s=40
)

ax.set(
    title="UMAP Components",
    xlabel="UMAP dim. 1",
    ylabel="UMAP dim. 2",
)
ax.set_xticklabels([])
ax.set_yticklabels([])

import matplotlib.patches as mpatches

handles = [mpatches.Patch(color=colors[i], label=dataset.target_names[i])
           for i in range(len(colors))]

legend1 = ax.legend(
    handles=handles,
    loc="upper right",
    title="Proteins",
)
ax.add_artist(legend1)

plt.savefig("RfaH_umap.png", dpi=150, bbox_inches="tight")