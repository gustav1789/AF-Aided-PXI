import os
import multiprocessing
import shutil
from GROMACS import _run_sim
from GROMACS import _run_visualization


# Here we specify in what folder to look for proteins (pre-aligned) to run simulations of!!
# Remember to align
# python3 align.py "input_path" "output_path"
Active_protein = "proteins/RfaH"            # <---- Right here

#number of simulations per protein and amount of cores used
N_sims = 100
num_cores = 15

ROOT = os.path.abspath(".")

input_data = os.path.join(ROOT, "input_data")
output_data = os.path.join(ROOT, "output_data")
atomic_data = os.path.join(input_data, "Atomic_data")

protein_structures = os.path.join(input_data, Active_protein) 
exp_MDP = os.path.join(input_data,"exp.mdp") # This is our input MDP file for explosion simulation
em_MDP = os.path.join(input_data, "em.mdp")  # This is our input MDP file for energy minimization
vac_MDP = os.path.join(input_data,"vac.mdp") # This is our input MDP file for vacuum simulation

ff = "charmm36-mar2019-Fe-S" # standard force field that we will use
#ff = "charmm27"
#gmx = "/home/spidocstester/MolDStruct/bin"  # Path to Gromacs installation
#gmx = "/home/andre/software/install_path/gromacs-mc/bin"

# Identify proteins in folder
_removesuffix = lambda x: x.removesuffix(".pdb")
proteins = list(map(_removesuffix, os.listdir(protein_structures)))
print(f"Proteins in folder: {proteins}")
 
# Remove some if you're testing things out
proteins = proteins
print(f"Using proteins: {proteins}")


def clear_directory(path):
    for name in os.listdir(path):
        full = os.path.join(path, name)

        if os.path.isfile(full) or os.path.islink(full):
            os.remove(full)
        elif os.path.isdir(full):
            shutil.rmtree(full)


if __name__=="__main__":
       for protein in proteins:
              print(f"Simulating {protein}...", end="\t")

              # Create the output folder for this protein
              output_protein_folder = os.path.join(output_data, protein)
              '''if os.path.exists(output_protein_folder):
                     shutil.rmtree(output_protein_folder)
                     os.makedirs(output_protein_folder)
              '''
              if not os.path.exists(output_protein_folder):
                     os.makedirs(output_protein_folder)
              clear_directory(output_protein_folder)

              # Locate the structure file
              PDB = os.path.join(protein_structures, protein+".pdb")

              # Set up a folder and define params for each run             
              outputs_paths = [os.path.join(output_protein_folder,f"sim_{k:02d}") for k in range(N_sims)]

              # Add atomic data to output protein folder (required to run at all)
              for simxx in outputs_paths:
                     Atomic_data = os.path.join(simxx,"Atomic_data")
                     shutil.copytree(atomic_data, Atomic_data)

              #Define parameters for simulations (and images)        
              params = [[PDB, em_MDP, vac_MDP, exp_MDP, outputs_paths[k]] for k in range(N_sims)]
              im_params = [N_sims, protein, output_protein_folder]

              # Run it
              with multiprocessing.Pool(num_cores) as pool:
                     res = pool.map(_run_sim,params)

              #Create images
              #im = _run_visualization(im_params)                    
              print("done!")
              # You should now see the outputs in the defined directories!
            
       print("All done!")