import h5py
import os
import shutil
import numpy as np
import MDAnalysis as mda
import healpy as hp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import Bunch

from detector_functions import build_healpix_full


def clear_directory(path):
    """
    Function to clear directory
    """

    for name in os.listdir(path):
        full = os.path.join(path, name)

        if os.path.isfile(full) or os.path.islink(full):
            os.remove(full)
        elif os.path.isdir(full):
            shutil.rmtree(full)


def maps(protein_h5):
    """
    Function to plot explosion maps
    """

    ROOT = os.path.abspath(".")      #This should be the analysis folder
    output_path = os.path.join(ROOT,"output")
    input_path = os.path.join(ROOT, "simulation_inputs")
    h5_path = os.path.join(input_path,protein_h5)
    protein_h5 = protein_h5.replace('.h5','') #Remove hf sufix

    #Create a path for maps in the output directory and clear it if there is one with the same name
    map_path = os.path.join(output_path,f"Maps_{protein_h5}")
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    clear_directory(map_path)

    #Unlock the file and see what proteins are inside, create dictionary for maps of every protein 
    f = h5py.File(h5_path, "r")
    print(f"All conformations: {f.keys()}")
    maps ={}
    smoothed_maps_raw = {}
    
    #Loop over all protein conformations
   
    for protein in f.keys():
        print(f"--- Currently analyzing: {protein} ---")

        #Extract data for individual conformations of the protein
        protein_data  = f[protein]
        final_velocity = f[protein]["final_velocity"][:]
        mass_data = f[protein]["mass"][:]
        unit_displacement = f[protein]["displacement"][:]
        
        #Create mean positions over all simulations
        mean_displacement = np.mean(unit_displacement, axis=0)

        # The other function is similar but instead of binning the hits onto a plane, 
        # it bins them onto a healpix map, which is a way to represent data on a sphere, 
        # This also captures all outgoing ions, just not the ones in a specific direction
        hp_map = build_healpix_full(mean_displacement, nside=16, nest=False)[0] # shape (npix,) where npix is the number of pixels in the healpix map, which depends on nside
        dot_path = os.path.join(map_path,f"{protein}.png") 
        hp.mollview(hp_map,cmap='viridis',title={protein})
        plt.savefig(dot_path, dpi=150, bbox_inches="tight")
        plt.close()
                
        #Create a smoothed map with a Gauusian curve over all bins
        hp_map_smoothed = hp.smoothing(hp_map, sigma=0.1)
        
        smooth_path = os.path.join(map_path,f"{protein}_smooth.png")
        maps[protein] = hp_map_smoothed  
        smoothed_maps_raw[protein] = hp_map_smoothed                               #Store map of protein "protein"
        hp.mollview(hp_map_smoothed,cmap='viridis',title=f'{protein} smooth')
        plt.savefig(smooth_path, dpi=150, bbox_inches="tight")
        plt.close()
    return (maps, protein_h5)
    

def Pearsson(mapping):
    """
    Was used in an earlier draft of the project but replaced by PCA euclidean distance
    """

    maps, protein_h5 = mapping
    ROOT = os.path.abspath(".")      #This should be the analysis folder
    output_path = os.path.join(ROOT,"output")
    map_path = os.path.join(output_path,f"Maps_{protein_h5}")
    #Create the correlation indeces
    names = list(maps.keys())
    data = [maps[name] for name in names]
    print(f"-- Currently making correlation matrix --")
    # Make data mtx: number of proteins x size of protein
    data_matrix = np.vstack(data)
    corr_matrix = np.corrcoef(data_matrix)
    data_mtx_path = os.path.join(map_path,f"Correlation_{protein_h5}.png")

    #Make correlation image
    sns.set_theme(style="white")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True,              
                fmt=".2f",              
                cmap='viridis', 
                xticklabels=names, 
                yticklabels=names,
                cbar_kws={'label': 'Pearsson Correlation Coefficient'})
                
    plt.title(f"Pearsson Correlation Matrix for {protein_h5}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(data_mtx_path, dpi=300)
    plt.close()
    print(f"All done")
    return corr_matrix


def package_dataset(h5_path, proteins, mass_filter=None, distance_filter=None, smoothing='yes'):
    """
    Function to package explosion map dataset into Bunch object
    """
    f = h5py.File(h5_path, "r")

    maps = []

    for protein in proteins:
        print(f"--- Currently analyzing: {protein} ---")
        #Extract data for individual conformations of the protein
        protein_data  = f[protein]
        final_velocity = f[protein]["final_velocity"][:]
        mass_data = f[protein]["mass"][:]
        unit_displacement = f[protein]["displacement"][:]
        submaps = []
        for displacement in unit_displacement:
            hp_map = build_healpix_full(displacement, nside=16, mass=mass_data, mass_filter=mass_filter, distance_filter=distance_filter, nest=False)[0] # shape (npix,) where npix is the number of pixels in the healpix map, which depends on nside
            if smoothing=='no':
                hp_map_smoothed = hp_map
            else:
                hp_map_smoothed = hp.smoothing(hp_map, sigma=0.1) # Changed from 0.05 to 0.1
            submaps.append(hp_map_smoothed)
        submaps = np.array(submaps)
        maps.append(submaps)

    # Combine arrays and create target labels
    data = np.vstack(maps)
    print(data.shape)
    target = np.concatenate([np.full(len(arr), i) for i, arr in enumerate(maps)])

    dataset = Bunch(
    data=data,
    target=target,
    target_names=np.array(proteins)
    )    

    return dataset

import pickle
from pathlib import Path

def load_data(path: str | None = None) -> Bunch:
    """
    Function to load pickled Bunch data
    """
    with open(path, "rb") as f:
        bunch = pickle.load(f)
    print(f"[load] Loaded data from '{path}'")
    X = np.array(bunch.data, dtype=np.float64)
    y = np.array(bunch.target, dtype=int)
    names = np.array(bunch.target_names)
    return Bunch(data=X, target=y, target_names=names)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import pandas as pd 


def PCA_matrix(dataset, dimension, fig_name, csv_name=None, metric = 'inner_product',scaler = None, mean = 'after', svd_solver = 'full'):
    """
    PCA function that saves the resulting crrelation metric.
    Correlation can be measured using Euclidean distance, Mahalanobis distance and cosine.
    """

    if scaler is not None:
        if scaler == 'standard':
            scaler = StandardScaler()
            data = scaler.fit_transform(dataset.data) # Fråga om detta
        elif scaler == 'sqrt':
            data = np.sqrt(dataset.data)
        else:
            raise Exception('Invalid scaler. Scaler must be standard or sqrt.')
    else:
        data = dataset.data
        
    X_reduced = PCA(n_components=dimension, svd_solver=svd_solver).fit_transform(data)

    n_targets = len(dataset.target_names)

    if mean == 'before':
        # Compute mean position in nD space for each target
        target_means = np.array([
            X_reduced[dataset.target == i].mean(axis=0)
            for i in range(n_targets)
        ])

        if metric == 'inner_product':
            # Compute pairwise inner product between all target means
            dist_matrix = (target_means[:, np.newaxis, :]*target_means[np.newaxis, :, :]).sum(axis=2) / (np.linalg.norm(target_means[:, np.newaxis, :], axis=2)*np.linalg.norm(target_means[np.newaxis, :, :], axis=2))
        elif metric == 'distance':
            # Compute pairwise Euclidean distances between all target means
            dist_matrix = np.sqrt(((target_means[:, np.newaxis, :] - target_means[np.newaxis, :, :]) ** 2).sum(axis=2))
        else:
            raise Exception('Invalid metric. Metric must be distance or inner product.')

    elif mean == 'after':
        dist_matrix = np.zeros((n_targets, n_targets))
        for i in range(n_targets):
            for j in range(n_targets):
                samples_i = X_reduced[dataset.target == i]
                samples_j = X_reduced[dataset.target == j]
                if metric == 'inner_product':
                    # Compute all pairwise inner products between samples in group i and group j
                    pairwise = (samples_i[:, np.newaxis, :]*samples_j[np.newaxis, :, :]).sum(axis=2) / (np.linalg.norm(samples_i[:, np.newaxis, :], axis=2)*np.linalg.norm(samples_j[np.newaxis, :, :], axis=2))
                elif metric == 'distance':
                    # Compute all pairwise distances between samples in group i and group j
                    pairwise = np.sqrt(((samples_i[:, np.newaxis, :] - samples_j[np.newaxis, :, :]) ** 2).sum(axis=2))
                elif metric == 'mahalanobis':
                    # Compute covariance matrices and their inverses for both groups
                    cov_i_inv = np.linalg.inv(np.cov(samples_i, rowvar=False))
                    cov_j_inv = np.linalg.inv(np.cov(samples_j, rowvar=False))
                    # Compute means for both groups
                    mean_i = samples_i.mean(axis=0)
                    mean_j = samples_j.mean(axis=0)
                    # Distance from each point in group i to the mean of group j (shape: n_i)
                    diff_i = samples_i - mean_j
                    mahal_i_to_j = np.sqrt((diff_i @ cov_j_inv * diff_i).sum(axis=1))
                    # Distance from each point in group j to the mean of group i (shape: n_j)
                    diff_j = samples_j - mean_i
                    mahal_j_to_i = np.sqrt((diff_j @ cov_i_inv * diff_j).sum(axis=1))
                    # Symmetric pairwise matrix: average both directions for each pair (k, l)
                    pairwise = 0.5 * (mahal_i_to_j[:, np.newaxis] + mahal_j_to_i[np.newaxis, :])
                else:
                    raise Exception('Invalid metric. Metric must be distance, inner product, or mahalanobis.')
                #Take mean
                dist_matrix[i, j] = pairwise.mean()
    else:
        print('Error. Mean must be before or after.')   

    # Plot the distance matrix as a heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(dist_matrix, cmap='viridis')
    if metric == 'inner_product':
        plt.colorbar(im, ax=ax, label=f'Inner Product ({dimension}D PCA space)')
    elif metric == 'distance':
        plt.colorbar(im, ax=ax, label=f'Euclidean Distance ({dimension}D PCA space)')
    else:
        plt.colorbar(im, ax=ax, label=f'Mahalanobis Distance ({dimension}D PCA space)')
    ax.set_xticks(range(n_targets))
    ax.set_yticks(range(n_targets))
    ax.set_xticklabels(dataset.target_names, rotation=45, ha='right')
    ax.set_yticklabels(dataset.target_names)
    if metric == 'inner_product':
        if mean == 'after':
            ax.set_title(f'Pairwise Mean Inner Products Between Targets ({dimension}D PCA)')
        else:
            ax.set_title(f'Pairwise Inner Products Between Target Means ({dimension}D PCA)')
    elif metric == 'distance':
        if mean == 'after':
            ax.set_title(f'Pairwise Mean Euclidean Distances Between Targets ({dimension}D PCA)')
        else:
            ax.set_title(f'Pairwise Euclidean Distances Between Target Means ({dimension}D PCA)')
    else:
        ax.set_title(f'Pairwise Mean Mahalanobis Distances Between Target Points and Means ({dimension}D PCA)')

    # Annotate each cell with the distance value
    for i in range(n_targets):
        for j in range(n_targets):
            ax.text(j, i, f'{dist_matrix[i, j]:.2f}', ha='center', va='center',
                    color='white' if dist_matrix[i, j] < dist_matrix.max() * 0.6 else 'black',
                    fontsize=8)

    plt.tight_layout()
    #plt.savefig("Mad2_inner_product_matrix_90D.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{fig_name}.png", dpi=150, bbox_inches="tight")
    plt.clf()

    if csv_name is not None:
        # Save the matrix a dataframe
        dist_df = pd.DataFrame(dist_matrix, index=dataset.target_names, columns=dataset.target_names)
        dist_df.to_csv(f"{csv_name}.csv")
    # --- End correlation matrix ---
    return X_reduced



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


def plot_mixed_matrix(pc_df, rmsd_df):
    """
    Function to merge correlation matrices into one matrix split along the diagonal
    """
    assert pc_df.shape == rmsd_df.shape, \
        f"Matrix shapes do not match: {pc_df.shape} vs {rmsd_df.shape}"
    assert (pc_df.index == rmsd_df.index).all(), \
        "Row labels do not match between matrices"

    n = len(pc_df)
    labels = pc_df.index.tolist()
    pc_matrix = pc_df.values
    rmsd_matrix = rmsd_df.values

    upper_vals = pc_matrix[np.triu_indices(n, k=1)]
    lower_vals = rmsd_matrix[np.tril_indices(n, k=-1)]

    norm_upper = Normalize(vmin=upper_vals.min(), vmax=upper_vals.max())
    norm_lower = Normalize(vmin=lower_vals.min(), vmax=lower_vals.max())
    #cmap_upper = cm.Blues_r # Blues
    cmap_upper = cm.viridis
    #cmap_lower = cm.Oranges_r # Oranges
    cmap_lower = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

    for i in range(n):
        for j in range(n):
            if i <= j:
                color = cmap_upper(norm_upper(pc_matrix[i, j]))
                text_val = f"{pc_matrix[i, j]:.2f}"
            elif i > j:
                color = cmap_lower(norm_lower(rmsd_matrix[i, j])) # Multiply by -1 for lddt
                text_val = f"{rmsd_matrix[i, j]:.2f}"
            else:
                color = (0.9, 0.9, 0.9, 1.0)
                text_val = "0"
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color=color))
            ax.text(j + 0.5, n - 0.5 - i, text_val,
                    ha='center', va='center', fontsize=8)

    sm_upper = cm.ScalarMappable(cmap=cmap_upper, norm=norm_upper)
    sm_lower = cm.ScalarMappable(cmap=cmap_lower, norm=norm_lower)
    plt.colorbar(sm_upper, ax=ax, fraction=0.03, pad=0.01).set_label('lDDT (upper)')
    #plt.colorbar(sm_lower, ax=ax, fraction=0.03, pad=0.12).set_label('RMSD in Å (lower)')
    plt.colorbar(sm_lower, ax=ax, fraction=0.03, pad=0.12).set_label('TM-score (lower)')
    # Explicit axes for each colorbar: [left, bottom, width, height] in figure fractions
    #cax1 = fig.add_axes([0.88, 0.55, 0.02, 0.35])
    #cax2 = fig.add_axes([0.88, 0.10, 0.02, 0.35])

    #plt.colorbar(sm_upper, cax=cax1).set_label('PCA Euclidean Distance (upper)')
    ##plt.colorbar(sm_lower, cax=cax2).set_label('RMSD in Å (lower)')

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels[::-1])
    #ax.set_title('Structure Similarity — Upper: Euclidean PC Distance | Lower: RMSD (Å)')
    ax.set_title('Structure Similarity — Upper: lDDT | Lower: TM-score')
    #plt.tight_layout()
    plt.savefig("RfaH_merged_lddt_tm_prevac.png", dpi=150, bbox_inches="tight")
    plt.clf()
