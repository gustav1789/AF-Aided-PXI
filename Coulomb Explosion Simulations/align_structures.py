# This script aligns structures in a folder to a reference structure using PyMOL's align function.
# Usage in bash terminal: python3 align_structures.py "reference.pdb" "input_folder" "output_folder" "resi_start-resi_end"
# It will same the aligned structures as well as the reference structure in the output folder. 
# If resi_start and resi_end are provided, it will only align based on that residue range,
# otherwise it will align based on all atoms.

import os
import glob
import sys
from pymol import cmd


def align_all_in_folder(ref_file, input_folder, output_folder, resi_span=None):
    # make sure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # find all PDB files in the input folder
    pdb_files = sorted(glob.glob(os.path.join(input_folder, "*.pdb")))

    if not pdb_files:
        print("No .pdb files found!")
        return

    # load reference structure into PyMOL
    ref_name = "reference"
    cmd.load(ref_file, ref_name)
    
    # Save the reference pdb to the output folder
    cmd.save(os.path.join(output_folder, os.path.basename(ref_file)), ref_name)
    print(f"Using {ref_file} as reference.")

    # If resi_span is provided, create selection for the specified residue range
    if resi_span:
        print(f"Aligning only on residues: {resi_span}")
        ref_sel = ref_name + f" and resi {resi_span[0]}-{resi_span[1]}"
    
    # Loop through all target files and align them to the reference
    for target_file in pdb_files:
        target_name = "target"
        cmd.load(target_file, target_name)
        
        # If resi_span is provided, align only on the specified residue range, else on all atoms
        if resi_span:
            target_sel = target_name + f" and resi {resi_span[0]}-{resi_span[1]}"
            rms = cmd.align(target_sel, ref_sel)
            print(f"Aligning {os.path.basename(target_file)} by residues {resi_span} -> RMSD: {rms[0]:.3f} Å")
        else:
            rms = cmd.align(target_name, ref_name)
            print(f"Aligning {os.path.basename(target_file)} -> RMSD: {rms[0]:.3f} Å")
            
        # Save aligned structure to the output folder and delete it from PyMOL memory
        out_path = os.path.join(output_folder, os.path.basename(target_file))
        cmd.save(out_path, target_name)
        cmd.delete(target_name)
    print("\nDone, result in: ", output_folder)

if __name__ == "__main__":
    reference_path = os.path.join(".", sys.argv[1])
    unaligned_folder_path = os.path.join(".", sys.argv[2])
    aligned_folder_path = os.path.join(".", sys.argv[3])
    resi_span = sys.argv[4].split('-') if len(sys.argv) > 4 else None

    align_all_in_folder(reference_path, unaligned_folder_path, aligned_folder_path, resi_span)