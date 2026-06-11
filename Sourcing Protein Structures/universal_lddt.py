import numpy as np
from Bio.PDB import PDBParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----- User settings -----
folder_name = "transition_selection"
desired_order = ["4KSOchainA", "KaiB_033", "KaiB_114", "KaiB_112", "KaiB_134",
                    "KaiB_188", "KaiB_186", "KaiB_072", "5N8YchainG"]
output_name = "lddt_transition_matrix"
# -------------------------

def get_coords(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('p', pdb_file)

    coords = {}
    model = structure[0]
    target_chain = list(model.get_chains())[0]
    for residue in target_chain:
        # Check if the residue has a CA atom as expected
        if "CA" in residue:
            res_num = residue.get_id()[1] # fetch residue number
            coords[res_num] = residue["CA"].get_coord()
    return coords


def fast_lddt(coords1_dict, coords2_dict, cutoff=15.0, thresholds=[0.5, 1.0, 2.0, 4.0]):
    # Find the residue numbers that are present in BOTH files
    common_res = sorted(set(coords1_dict.keys()) & set(coords2_dict.keys()))
    
    if not common_res: # completely wacky
        return np.nan

    # Create arrays with coordinates for only the common residues
    coords1 = np.array([coords1_dict[r] for r in common_res])
    coords2 = np.array([coords2_dict[r] for r in common_res])

    # Get pairwise distances in both structures by broadcasting
    diff1 = coords1[:, None, :] - coords1[None, :, :]
    dist1 = np.sqrt(np.sum(diff1**2, axis=-1))
    
    diff2 = coords2[:, None, :] - coords2[None, :, :]
    dist2 = np.sqrt(np.sum(diff2**2, axis=-1))
    
    # Only consider pairs of residues that are within the cutoff distance in the first structure and are not the same residue (dist1 > 0)
    mask = (dist1 < cutoff) & (dist1 > 0)
    dist_diff = np.abs(dist1 - dist2)
    
    results = []
    for t in thresholds:
        n_preserved = np.sum((dist_diff < t) & mask)
        n_total = np.sum(mask)
        results.append(n_preserved / n_total)
    
    return np.mean(results)


def main():
    folder_path = Path(folder_name)
    # Find all PDB files in the folder
    pdb_files = sorted([f for f in folder_path.glob("*.pdb")])

    if not pdb_files:
        print(f"No PDB files found in {folder_path}")
        return

    num_files = len(pdb_files)
    file_names = [f.name for f in pdb_files]
    short_names = [f.removesuffix(".pdb") for f in file_names]
    
    # Create an empty matrix for the scores
    matrix = np.zeros((num_files, num_files))

    print(f"Analyzing {num_files} files...")

    # Cache for coordinates so we don't read the same file 100 times
    coord_cache = {f: get_coords(f) for f in pdb_files}

    for i in range(num_files):
        for j in range(num_files):
            if i == j:
                matrix[i, j] = 1.0  # structure has lddt of 1 with itself
            elif i < j:
                score = fast_lddt(coord_cache[pdb_files[i]], coord_cache[pdb_files[j]])
                matrix[i, j] = score
                matrix[j, i] = score # symmetric matrix

    # create a dataframe for better handling and visualization
    df = pd.DataFrame(matrix, index=short_names, columns=short_names)
    
    print("\n--- lDDT Correlation Matrix ---")
    print(df.round(3).to_string())

    # Generate a matrix in png format   

    print("\nGenerating PNG file...")
    sns.set_theme(style="white")
    
    df.index = short_names
    df.columns = short_names

    scale = max(8, num_files * 0.4)
    fig, ax = plt.subplots(figsize=(scale, scale))
    
    df_sorted = df.loc[desired_order, desired_order]
    sns.heatmap(
        df_sorted, 
        annot=num_files < 15, # print nums only if there are fewer than 15 models to avoid clutter
        cmap="magma",         
        vmin=0, vmax=1.0, 
        square=True,
        cbar_kws={"shrink": .8, "label": "lDDT Similarity"},
        xticklabels=True, yticklabels=True,)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.title(f"lDDT Matrix: {num_files} Mad2 Models", fontsize=16)
    
    # output paths
    png_output_path = folder_path / f"{output_name}.png"
    csv_output_path = folder_path / f"{output_name}.csv"

    # Save the figure and the matrix
    plt.savefig(png_output_path, dpi=300, bbox_inches='tight') 
    df_sorted.to_csv(csv_output_path)
    print(f"Done: results saved as {output_name}")

if __name__ == "__main__":
    main()