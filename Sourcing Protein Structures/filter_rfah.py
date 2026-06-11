from universal_lddt import get_coords, fast_lddt
from tm_matrix import calc_tm_score
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Necessary info ----
csv_path = "RfaH_AFcluster_analysis.csv"
pdb_folder = "RfaH_pdbs"
lddt_output_name = "RfaH_lddt_matrix"
tm_output_name = "RfaH_tm_matrix"
n_models = 50
# -----------------------

def remove_redundant_structures(plddt_limit, tm_threshold=0.95, lddt_threshold=0.95, remove_on_either=True):
    df = pd.read_csv(csv_path)
    filtered_df = df[df['mean_plddt'] > plddt_limit]
    top_ids = filtered_df['id'].tolist()
    plddt_dict = dict(zip(filtered_df['id'], filtered_df['mean_plddt']))
    paths = [os.path.join(pdb_folder, f"{id}.pdb") for id in top_ids]
    print(f"Found {len(top_ids)} structures with mean pLDDT above {plddt_limit} (out of {len(df)}). Checking for redundancy...")
    
    n = len(top_ids)
    tm_matrix = np.full((n,n), np.nan) # init with NaN 
    lddt_matrix = np.full((n,n), np.nan)
    tm_matrix[np.diag_indices(n)] = 1.0 # set diagonal to 1.0
    lddt_matrix[np.diag_indices(n)] = 1.0

    redundant = set()
    for i in range(n):
        print(f"At stage {i+1} of {n}...")
        # Check if structure i is already marked as redundant, if so skip it
        if paths[i] in redundant:
            continue
            
        for j in range(i + 1, n):
            # Check if structure j is already marked as redundant, if so skip it
            if paths[j] in redundant:
                continue
            
            # Start calculating lddt as it is faster
            lddt_score = fast_lddt(get_coords(paths[i]), get_coords(paths[j]))
            lddt_matrix[i, j] = lddt_score
            lddt_matrix[j, i] = lddt_score
            
            # Check if we need to calculate TM-score based on the condition below
            need_tm = True
            
            # if "either" we can skip tm-score if lddt shows similarity
            if remove_on_either and lddt_score > lddt_threshold:
                need_tm = False
                i_plddt = plddt_dict[top_ids[i]]
                j_plddt = plddt_dict[top_ids[j]]
                
                if i_plddt >= j_plddt: # Mark the one with lower pLDDT as redundant
                    redundant.add(paths[j])
                    print(f"{top_ids[j]} marked as redundant because of {top_ids[i]} based on LDDT")
                else:
                    redundant.add(paths[i])
                    print(f"{top_ids[i]} marked as redundant because of {top_ids[j]} based on LDDT")
                    break # No need to compare this structure with others if it's already marked as redundant
            if need_tm:
                tm_score = calc_tm_score(paths[i], paths[j])
                tm_matrix[i, j] = tm_score
                tm_matrix[j, i] = tm_score
                
                i_plddt = plddt_dict[top_ids[i]]
                j_plddt = plddt_dict[top_ids[j]]
                
                if remove_on_either and tm_score > tm_threshold:
                    if i_plddt >= j_plddt:
                        redundant.add(paths[j])
                        print(f"{top_ids[j]} marked as redundant because of {top_ids[i]} based on TM-score")
                    else:
                        redundant.add(paths[i])
                        print(f"{top_ids[i]} marked as redundant because of {top_ids[j]} based on TM-score")
                        break

                elif remove_on_either == False and tm_score > tm_threshold and lddt_score > lddt_threshold:
                    if i_plddt >= j_plddt:
                        redundant.add(paths[j])
                        print(f"{top_ids[j]} marked as redundant because of {top_ids[i]} based on both TM-score and LDDT")
                    else:
                        redundant.add(paths[i])
                        print(f"{top_ids[i]} marked as redundant because of {top_ids[j]} based on both TM-score and LDDT")
                        break

    # filter out the redundant structures and create the final matrices for the important ones
    important_list = []
    important_idx = []
    for i, path in enumerate(paths):
        if path not in redundant:
            important_list.append(top_ids[i])
            important_idx.append(i)
            
    important_tm_matrix = tm_matrix[np.ix_(important_idx, important_idx)]
    important_lddt_matrix = lddt_matrix[np.ix_(important_idx, important_idx)]
    
    print(f"Done, removed {len(redundant)} redundant structures. {len(important_list)} structures remain.")
    return important_list, important_tm_matrix, important_lddt_matrix

if __name__ == "__main__":
    ids, tm_matrix, lddt_matrix = remove_redundant_structures(70, tm_threshold=0.85, lddt_threshold=0.92, remove_on_either=True)

    sns.clustermap(
        pd.DataFrame(tm_matrix, index=ids, columns=ids),
        annot= len(ids) < 15, fmt=".2f",
        cmap="magma",
        vmin=0, vmax=1.0,
        square=True,
        cbar_kws={"shrink": .8, "label": "TM-score Similarity"},
        xticklabels=True, yticklabels=True)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig("non_redundant_tm_matrix.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 10))
    sns.clustermap(
        pd.DataFrame(lddt_matrix, index=ids, columns=ids),
        annot= len(ids) < 15, fmt=".2f",
        cmap="magma",
        vmin=0, vmax=1.0,
        square=True,
        cbar_kws={"shrink": .8, "label": "LDDT similarity"},
        xticklabels=True, yticklabels=True)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig("non_redundant_lddt_matrix.png", dpi=300, bbox_inches='tight')

    print(f"Non-redundant structures: {ids}")