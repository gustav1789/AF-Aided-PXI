# This script takes as input a folder containing colabfold_batch results. 
# It will generate a csv containing the analyses of the protein as well as a folder with renamed .pdbs.
# It assumes you have only used one model in colabfold and your original structure
# name didn't contain three numbers in a row (because indices are naively implemented).
# note that this might break for non-monomer inputs (i.e. >1 chains)

from tmtools.io import get_structure, get_residue_data
from tmtools import tm_align
from Bio.PDB import PDBParser
import numpy as np
import os
import shutil
import pandas as pd
from pymol import cmd
import json
from universal_lddt import get_coords, fast_lddt

# ----- Input stuff here! -----
model_folder = "RfaH_pdbs"  # the folder containing all results from colabfold_batch (or regular colabfold I think...)
ref1_path = "2OUGchainA.pdb"  # a .pdb-file
ref2_path = "6C6SchainD.pdb"  # a .pdb-file 

renamed_folder = "RfaH_pdbs_again" # Will be created if nonexisting
new_naming_scheme = "RfaH_" # Preferably one ending with an underscore, e.g. "Mad2_"
csv_name = "RfaH_AFcluster_analysis.csv" # Will contain the analysis
plddt_from_json = False # Set to True if you have json files containing plddt values, 
                        # otherwise it fethces plddt from the bfactor field in the pdbs (since AF puts it there)
# ------------------------------


# create folder for renamed pdbs:
renamed_folder_path = os.path.join(os.getcwd(), renamed_folder)
if not os.path.exists(renamed_folder_path):
    os.makedirs(renamed_folder_path)

# Grab out the reference names
ref1_name = os.path.basename(ref1_path).strip(".pdb")
ref2_name = os.path.basename(ref2_path).strip(".pdb")


# Create a function for calculating tm score using tmtools
def calc_tm_score(model_path, reference_path):
    model_structure = get_structure(model_path)
    ref_structure = get_structure(reference_path)

    model_chain = next(model_structure.get_chains())
    ref_chain = next(ref_structure.get_chains())

    model_coords, model_seq = get_residue_data(model_chain)
    ref_coords, ref_seq = get_residue_data(ref_chain)

    res = tm_align(model_coords, ref_coords, model_seq, ref_seq)

    tm_score = res.tm_norm_chain2
    return tm_score

# Create a helper function for calculating lddt
def get_lddt(pdb_path, ref_path):
    c1_dict = get_coords(pdb_path)
    c2_dict = get_coords(ref_path)
    return fast_lddt(c1_dict, c2_dict)

# Create a function for calculating rmsd using pymol
# and load the references into pymol, once for all
cmd.load(ref1_path, ref1_name)
cmd.load(ref2_path, ref2_name)
def get_rmsd(model_path, ref_name):
    cmd.load(model_path, "hejhej")
    rmsd = cmd.align("hejhej", ref_name, cycles=0)[0] # Calculate the RMSD, cycles=0 ensures pymol doesn't discount outliers
    # Do note, we do not save the aligned structures afterwards, so this will NOT align cluster_xxx.pdb!
    cmd.delete("hejhej") # This unloads the structure from memory, hopefully increasing efficieny
    return rmsd

# Create a function for fetching plddt from bfactors
# (since AF puts the plddt in the bfactor field of the pdb)
def bfac_plddt(model_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('AF_model', model_path)
    
    plddt_scores = []
    for residue in structure.get_residues():
        # Only grab the CA atoms, since there is only on CA per residue
        if 'CA' in residue:
            plddt_scores.append(residue['CA'].get_bfactor())
            
    return np.mean(plddt_scores)


# List all the models in folder
model_files = [f for f in sorted(os.listdir(model_folder)) if f.endswith(".pdb")]
print(f"Found {len(model_files)} pdb files")

# Copy over the pdbs to a more concise naming scheme, also create a dataframe to later fill with scores!
results = []
idx = []
for pdb in model_files:
    full_path = os.path.join(model_folder, pdb)
    # Find cluster index, then read id which is 3 digits long
    first_num_idx = [x.isdigit() for x in pdb].index(True)
    file_idx = pdb[first_num_idx: first_num_idx + 3]
    new_name = new_naming_scheme+file_idx # Here you can set a new naming convention for the pdbs!
    renamed_path = os.path.join(renamed_folder_path, new_name+".pdb")

    # Copy the pdb 
    shutil.copy(full_path, renamed_path)
    results.append({"id":new_name,
                    "idx":file_idx})
df = pd.DataFrame(results)

# Create helper functions to extend the dataframe!
_tm1 = lambda cluster_id: calc_tm_score(os.path.join(renamed_folder_path, cluster_id+".pdb"), ref1_path)
_tm2 = lambda cluster_id: calc_tm_score(os.path.join(renamed_folder_path, cluster_id+".pdb"), ref2_path)
_rmsd1 = lambda cluster_id: get_rmsd(os.path.join(renamed_folder_path, cluster_id+".pdb"), ref1_name)
_rmsd2 = lambda cluster_id: get_rmsd(os.path.join(renamed_folder_path, cluster_id+".pdb"), ref2_name)
_lddt1 = lambda cluster_id: get_lddt(os.path.join(renamed_folder_path, cluster_id+".pdb"), ref1_path)
_lddt2 = lambda cluster_id: get_lddt(os.path.join(renamed_folder_path, cluster_id+".pdb"), ref2_path)
helper_funs = [_tm1, _tm2, _rmsd1, _rmsd2, _lddt1, _lddt2]

# Define the names of the scores in the dataframe
tm1_name = f"tm_{ref1_name}"
tm2_name = f"tm_{ref2_name}"
rmsd1_name = f"rmsd_{ref1_name}"
rmsd2_name = f"rmsd_{ref2_name}"
lddt1_name = f"lddt_{ref1_name}"
lddt2_name = f"lddt_{ref2_name}"
names = [tm1_name, tm2_name, rmsd1_name, rmsd2_name, lddt1_name, lddt2_name]

for name, fun in zip(names, helper_funs): # Finally, add everything to the dataframe
    print(f"Calculating {name} scores...",end="\t", flush=True)
    df[name]=df["id"].apply(fun)
    print("done!")

if plddt_from_json == True:
    # Fetch the plddt scores from json files
    plddt_list = []
    # Identify the json files containing the plddts
    # THIS ASSUMES ALL JSONS IN FOLDER BELONG TO A CLUSTER_XXX (except for config.json)
    json_files = [f for f in sorted(os.listdir(model_folder)) if f.endswith(".json") and "config" not in f]
    print(f"Found {len(json_files)} json files (exluding config.json)")
    for json_file in json_files:
        # Note, json is basically "blablabla_XXX_blabla.json"
        full_path = os.path.join(model_folder, json_file)

        # Find cluster index in file name, then read id which is 3 digits long
        first_num_idx = [x.isdigit() for x in json_file].index(True)
        json_idx = json_file[first_num_idx: first_num_idx + 3] # This is now "xxx"

        # Sanity check that the file exists
        if not os.path.exists(full_path):
            raise(FileNotFoundError(f"No json found at {full_path} bruh"))
        with open(full_path, 'r') as f:
            data = json.load(f)
            # Get list of plddt-values for this structure
            plddt_values = data.get("plddt", [])
            
            if plddt_values:
                mean_plddt = np.mean(plddt_values)
                plddt_list.append({
                    "id": new_naming_scheme+json_idx,
                    "mean_plddt": round(mean_plddt, 2)
                })
    plddt_df = pd.DataFrame(plddt_list)

    # Now merge the big dataframe with plddt and save everything to a csv
    df = pd.merge(df, plddt_df, on="id")
else:
    # If you don't have json files, you can also fetch the plddt from the bfactor field of the pdbs (since AF puts it there)
    print("Calculating mean plddt from bfactor field in pdbs...", end="\t", flush=True)
    df["mean_plddt"] = df["id"].apply(lambda cluster_id: bfac_plddt(os.path.join(renamed_folder_path, cluster_id+".pdb")))
    print("done!")

df.to_csv(csv_name, index=False)
print(f"Analysis done, see: {csv_name}")