# This script calculates the TM score for all pairs of PDB files found in 
# folder_name and saves the results in a heatmap png image and a csv file.

import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align

# ------- User settings - adjust these! -------
folder_name = "rfah_selection"
desired_order = ["RfaH-AIn", "RfaH-α1", "RfaH-α4", "RfaH-α5", "RfaH-α6", "RfaH-α3", "RfaH-α2", "RfaH-active"]
output_name = "rfah_selection_tm_matrix"
protein_name = "RfaH"
# ---------------------------------------------


# Create a function for calculating tm score using tmtools
def calc_tm_score(model_path, reference_path):
    model_structure = get_structure(model_path)
    ref_structure = get_structure(reference_path)

    model_chain = next(model_structure.get_chains())
    ref_chain = next(ref_structure.get_chains())

    model_coords, model_seq = get_residue_data(model_chain)
    ref_coords, ref_seq = get_residue_data(ref_chain)

    res = tm_align(model_coords, ref_coords, model_seq, ref_seq)

    # Take the maximum of the two normalized TM scores to not disfavor shorter PDB structures
    tm_score = max(res.tm_norm_chain2, res.tm_norm_chain1) 
    return tm_score


def main():
    folder_path = Path(folder_name)

    # find all PDB files in the folder
    pdb_files = sorted([f for f in folder_path.glob("*.pdb")])

    if not pdb_files:
        print(f"No PDB files found in {folder_path}")
        return

    num_files = len(pdb_files)
    file_names = [f.name for f in pdb_files]
    
    # SEmpty matrix for the scores
    matrix = np.zeros((num_files, num_files))

    print(f"Analyzing {num_files} files...")

    for i in range(num_files):
        for j in range(num_files):
            if i == j:
                matrix[i, j] = 1.0 # A structure compared to itself has a TM score of 1
            elif i < j:
                tm_score = calc_tm_score(pdb_files[i], pdb_files[j])
                matrix[i, j] = tm_score
                matrix[j, i] = tm_score # Symmetric matrix

        # Create a dataframe for better handling and visualization
        df = pd.DataFrame(matrix, index=file_names, columns=file_names)

    # Make TM matrix  
    print("\nGenerating PNG file...")
    sns.set_theme(style="white")
    
    short_names = [f.removesuffix(".pdb") for f in file_names]
    df.index = short_names
    df.columns = short_names


    scale = max(8, num_files * 0.4)
    fig, ax = plt.subplots(figsize=(scale, scale))

    df_sorted = df.loc[desired_order, desired_order]

    sns.heatmap(
        df_sorted, 
        annot=num_files < 15, # Only print values if there are fewer than 15 models to avoid clutter
        cmap="magma",         
        vmin=0, vmax=1.0, 
        square=True,
        cbar_kws={"shrink": .8, "label": "TM Score"},
        xticklabels=True, yticklabels=True,)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(f"TM Matrix: {num_files} {protein_name} Models", fontsize=16)

    # output paths
    png_output_path = folder_path / f"{output_name}.png"
    csv_output_path = folder_path / f"{output_name}.csv"
    # Save the figure and the matrix
    plt.savefig(png_output_path, dpi=300, bbox_inches='tight') 
    df_sorted.to_csv(csv_output_path)

    print(f"Done: results saved as {output_name}")

if __name__ == "__main__":
    main()