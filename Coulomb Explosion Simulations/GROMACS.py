import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import healpy as hp

from detector_functions import Detector
from detector_functions import bin_plane_xaxis
from detector_functions import build_healpix_full


# Lets define our GROMACS calls as functions so that we can easily call them later, this will also make it easier to run multiple simulations with different parameters without having to repeat the same code multiple times.


def run_pdb2gmx(pdb_path, top_path, gro_path, itp_path, water_model="tip3p", force_field="charmm36-mar2019-Fe-S",gmx_path="/home/spidocstester/MolDStruct/bin"):
    pdb2gmx = os.path.join(gmx_path, "pdb2gmx")
    cmd = [pdb2gmx, "-f", pdb_path,
                   "-p", top_path,
                   "-o", gro_path,
                    "-i", itp_path,
                    "-water", water_model,
                    "-ff", force_field,
                    "-ignh",
                    "-missing"]

    result = subprocess.run(cmd, capture_output=True, text=True,errors="replace")
    return result 
    

def run_grompp(mdp_path, gro_path, top_path, tpr_path, mdout_path, maxwarn=5,gmx_path="/home/spidocstester/MolDStruct/bin"):
    grompp = os.path.join(gmx_path, "grompp")
    cmd = [grompp, '-f', mdp_path, '-c',gro_path,
        '-p', top_path, '-o', tpr_path, 
        "-po",mdout_path,'-maxwarn', str(maxwarn)]
    result = subprocess.run(cmd, capture_output=True, text=True,errors="replace")
    return result

def increase_box(gro_small_path, gro_big_path, gmx_path="/home/spidocstester/MolDStruct/bin"):
    editconf = os.path.join(gmx_path, "editconf")
    cmd = [editconf, '-f', gro_small_path, '-o', gro_big_path, '-box', '10', '10', '10',  '-c']
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return result


def run_mdrun(tpr_path, output_path, nt=1, ionize=False, deffnm = "md",
              gmx_path="/home/spidocstester/MolDStruct/bin"):

    mdrun = os.path.join(gmx_path, "mdrun")

    cmd = [
        mdrun,
        "-s", tpr_path,
        "-deffnm", deffnm,
        "-nt", str(nt),
        "-v"
    ]

    if ionize:
        cmd.append("-ionize")

    result = subprocess.run(
        cmd,
        cwd=output_path,
        capture_output=True,
        text=True,
        errors="replace"
    )

    return result


def set_parameters(mdp_path,
                       nsteps=None,
                       time_step=None,
                       pulse_peak=None,
                       num_photons=None,
                       sigma=None,
                       FWHM=None,
                       focus=None,
                       energy=None,
                       charge_transfer=None,
                       autostop=None,
                       autostop_limit=None,
                       logging=None,
                       ionize=None,
                       gen_vel=None,
                       gen_temp=None,
                       log_frequency=None,
                       set_charges=None,
                       rc=None,
                       ):
        
        if FWHM is not None:
            sigma = FWHM/(2*np.sqrt(2*np.log(2)))

        not_none_names = [
            k for k, v in locals().items()
            if v is not None and k not in {"FWHM", "self", "mdp_path"}
        ]

        not_none_values = [
            v for k, v in locals().items()
            if v is not None and k not in {"FWHM", "self", "mdp_path", "not_none_names"}
        ]

        var_mdp = {"nsteps":"nsteps",
                "time_step":"dt",
                "pulse_peak":"userreal1",
                "num_photons":"userreal2",
                "sigma":"userreal3",
                "focus":"userreal4",
                "energy":"userreal5",
                "charge_transfer":"userint2",
                "autostop":"userint3",
                "autostop_limit":"userreal6",
                "logging":"userint5",
                "ionize":"userint1",
                "gen_vel":"gen_vel",
                "gen_temp":"gen_temp",
                "set_charges":"userint9",
                
                }

        def change_line(split_line,line,parameter,value):
            if parameter in split_line:
                return f"{parameter}                   = {value}; \n"
            else:
                return line

        with open(mdp_path,"r") as f:
            lines = f.readlines()

        with open(mdp_path,"w") as f:
            for line in lines: 
                # skip lines that are comments
                if line.strip().startswith(";") or line.strip() == "":
                    f.write(line)
                    continue
                
                split_line = line.split()
                
                for name,value in zip(not_none_names,not_none_values):
                    if name == "log_frequency":
                        # log_frequency is a special case, it correspond to multiple parameters in the MDP file
                        if "nstxout" in split_line:
                            line = change_line(split_line,line,"nstxout",value)
                        if "nstfout" in split_line:
                            line = change_line(split_line,line,"nstfout",value)
                        if "nstvout" in split_line:
                            line = change_line(split_line,line,"nstvout",value)
                        if "nstenergy" in split_line:
                            line = change_line(split_line,line,"nstenergy",value)
                        if "nstlog" in split_line:
                            line = change_line(split_line,line,"nstlog",value)
                        if "xtc_precision" in split_line:
                            line = change_line(split_line,line,"xtc_precision",value)
                    elif name == "rc":
                        # rc is a special case, it correspond to multiple parameters in the MDP file
                        if "rlist" in split_line:
                            line = change_line(split_line,line,"rlist",value)
                        if "rcoulomb" in split_line:
                            line = change_line(split_line,line,"rcoulomb",value)
                        if "rvdw" in split_line:
                            line = change_line(split_line,line,"rvdw",value)
                    
                    else:
                        par = var_mdp[name]
                        line = change_line(split_line,line,par,value)

                f.write(line)





# Putting it all together with the gromacs stuff we can do it like this 

# First we define a function that runs one simulation (I will do everything in one step here, 
# might be easier to split such that we have one function that does the pdb2gmx, one for grompp, and so on)
# For many cases like this one where we use the same input (PDB) for all cases we would not need to run pdb2gmx each time, but it is not a big performance loss

# We can also simplify the function if we assume that we want to put all intermediate files in the same directory "output_path"
def run_visualization(N_sims, protein, p_output_root):
    positions = np.zeros(N_sims, dtype=object)
    velocities = np.zeros(N_sims, dtype=object)
    p_output =  p_output_root            #os.path.join(p_output_root, protein)

    for k in range(N_sims):
        sim_dir = os.path.join(p_output, f"sim_{k:02d}")
        trr_file = os.path.join(sim_dir, "md.trr")
        gro_file = os.path.join(sim_dir, "md.gro")
        
        if not os.path.exists(trr_file):
            print(f"Warning: {trr_file} not found. Skipping.")
            continue

        u = mda.Universe(gro_file, trr_file)
        all_atoms = u.atoms.select_atoms("all")

        # Select the last frame 
        u.trajectory[-1] 
        velocities[k] = all_atoms.velocities.copy()
        positions[k] = all_atoms.positions.copy() 

    # Calculate mean and visualize
    mean_displacement = positions.mean(axis=0)
    hp_map_mean = build_healpix_full(mean_displacement, nside=64, nest=False)[0]
    # Plotting
    hp.mollview(hp_map_mean, cmap='inferno', title=f"Mean: {protein}")
    save_path_raw = os.path.join(p_output, f"{protein}.png")
    plt.savefig(save_path_raw, dpi=150, bbox_inches="tight")
    plt.close()

    hp_map_smoothed = hp.smoothing(hp_map_mean, sigma=0.05)
    hp.mollview(hp_map_smoothed, cmap='inferno', title=f"Mean Smoothed: {protein}")
    save_path_smooth = os.path.join(p_output, f"{protein}_smooth.png")
    plt.savefig(save_path_smooth, dpi=150, bbox_inches="tight")
    plt.close()

def run_sim(pdb_path,
            em_mdp_path, 
            vac_mdp_path,
            exp_mdp_path,
            output_path,
            water_model="tip3p",  # These
            force_field="charmm36-mar2019-Fe-S", # are 
            #force_field = "charmm27", # are
            gmx_path="/home/spidocstester/MolDStruct/bin", # default parameters, we do not need to specify them when calling the function
            ):
        
    
    # Simplifying input into grompp (files created by pdb2gmx) for energy minimization
    em_path = os.path.join(output_path,"energy_sim")
    em_gro_path = os.path.join(em_path,"gro.gro")
    em_tpr_path = os.path.join(em_path,"tpr.tpr")
    em_mdout_path = os.path.join(em_path,"mdout.mdp")

    # Create topology and itp paths (used for all steps)
    top_path = os.path.join(output_path,"top.top")
    itp_path = os.path.join(output_path,"itp.itp")

    # Create vaccum simulation path
    vac_path = os.path.join(output_path,"vacuum_sim")

    # Make sure our output directories exist
    os.makedirs(output_path,exist_ok=True)
    os.makedirs(em_path, exist_ok=True)
    os.makedirs(vac_path,exist_ok=True)
    sim_output = os.path.join(output_path, "simulation_output")
    if not os.path.exists(sim_output):
        os.makedirs(sim_output)

    # and lets make copies of our "base" mdps
    em_sim_mdp = os.path.join(em_path, "em_sim.mdp")
    exp_sim_mdp = os.path.join(output_path,"exp_sim.mdp")
    vac_sim_mdp = os.path.join(vac_path,"vac_sim.mdp")
    shutil.copy(exp_mdp_path,exp_sim_mdp)
    shutil.copy(vac_mdp_path,vac_sim_mdp)
    shutil.copy(em_mdp_path, em_sim_mdp)
    
    pdb2gmx_res = run_pdb2gmx(pdb_path, top_path, em_gro_path, itp_path, water_model=water_model, force_field=force_field,gmx_path=gmx_path)
        
    # Energy minimization simulations
    grompp_res = run_grompp(em_sim_mdp, em_gro_path, top_path, em_tpr_path, em_mdout_path, gmx_path=gmx_path)
    em_output_name = "em"
    mdrun_res = run_mdrun(em_tpr_path, em_path, nt=1, ionize=False,gmx_path=gmx_path, deffnm=em_output_name)
    
    # Define path to the energy-minimzation simulated gro file
    em_gro_path = os.path.join(em_path, em_output_name+".gro")
    vac_gro_path = os.path.join(vac_path,"gro.gro")
    # Update gro path
    shutil.copy(em_gro_path, vac_gro_path)    
    vac_tpr_path = os.path.join(vac_path,"tpr.tpr")
    vac_mdout_path = os.path.join(vac_path,"mdout.mdp")

    # Vacuum simulations    
    grompp_res = run_grompp(vac_sim_mdp, vac_gro_path, top_path, vac_tpr_path, vac_mdout_path,gmx_path=gmx_path)
    
    vac_output_name = "vac"
    mdrun_res = run_mdrun(vac_tpr_path, vac_path, nt=1, ionize=False,gmx_path=gmx_path, deffnm=vac_output_name)

    # Define path to the vacuum simulated gro file:
    vac_gro_path = os.path.join(vac_path, vac_output_name+".gro")
    exp_gro_path = os.path.join(output_path,"gro.gro" )
    # Update gro path
    shutil.copy(vac_gro_path, exp_gro_path)    
    exp_tpr_path = os.path.join(output_path,"tpr.tpr")
    exp_mdout_path = os.path.join(output_path,"mdout.mdp")


    #Explosion simulations

    # Say we want to change parameter of the simulation, in this case the intensity we can use our previously defined function
    #set_parameters(exp_sim_mdp,rc=cutoff)
    # This changes sim_mdp, which was previously just a copy of our original mdp

    grompp_res = run_grompp(exp_sim_mdp, exp_gro_path, top_path, exp_tpr_path, exp_mdout_path,gmx_path=gmx_path)
    
    mdrun_res = run_mdrun(exp_tpr_path, output_path, nt=1, ionize=True,gmx_path=gmx_path)


# And now we simply define a helper function to use with pool.map 
def _run_sim(params):
        pdb_path, em_mdp_path, vac_mdp_path, exp_mdp_path, output_path = params
        run_sim(pdb_path, em_mdp_path, vac_mdp_path,exp_mdp_path, output_path)

def _run_visualization(params):
    N_sims, protein, p_output_root = params
    run_visualization(N_sims, protein, p_output_root)            


   
