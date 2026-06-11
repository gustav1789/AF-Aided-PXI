import h5py
import os
import MDAnalysis as mda
import numpy as np

def h5_converter(data_path, output_path, file_name, N_sims):

    output_h5_path = os.path.join(output_path, file_name + ".h5")

    file_names = os.listdir(data_path)
    print(file_names)

    with h5py.File(output_h5_path, "w") as f:
        
        for file in file_names:
            positions = None
            velocities = None
            protein_path = os.path.join(data_path, file)

            for k in range(N_sims):
                sim = os.path.join(protein_path, f"sim_{k:02d}")
                TRR = os.path.join(sim, "md.trr")
                GRO = os.path.join(sim, "md.gro")
                U = mda.Universe(GRO, TRR)

                all_atoms = U.select_atoms("all")

                # Pre-allocate properly shaped arrays once we know N_atoms
                if positions is None:
                    N_atoms = len(all_atoms)
                    positions = np.zeros((N_sims, N_atoms, 3), dtype=np.float32)
                    velocities = np.zeros((N_sims, N_atoms, 3), dtype=np.float32)
                    mass_data = np.array(all_atoms.masses, dtype=np.float32)
                    atom_indices = np.array(all_atoms.indices, dtype=np.int32)

                # Select the last frame
                U.trajectory[-1]
                positions[k] = all_atoms.positions.copy()
                velocities[k] = all_atoms.velocities.copy()


            group = f.create_group(file)
            group.create_dataset("displacement", data=positions)
            group.create_dataset("final_velocity", data=velocities)
            group.create_dataset("mass", data=mass_data)
            group.create_dataset("atom_indices", data=atom_indices)