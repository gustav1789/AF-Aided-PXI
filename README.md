# AlphaFold-Aided Protein Explosion Imaging

This repository contains the code needed to replicate the PXI simulations
of AlphaFold-generated protein structures from the Bachelor's thesis
project "AlphaFold-aided Protein Explosion Imaging".

The code is organised into three folders for each step in the simulations
and data analysis. The first step is the sourcing of protein structures.
This encompasses the comparison and selection of AF-generated structures.
Simulations are run in GROMACS, but for ease of use the GROMACS commands
are written into Python functions that can easily be called upon.
The simulation files are package into h5 files, the re-packaged into
pickle objects of the Bunch format for easy implementation of scikit-
learn functions for data anlysis and machine learning steps.

## Sourcing Protein Structures

In the folder `Sourcing Protein Structures/` all scripts used to analyze direct outputs from AlphaFold are collected. The scripts `universalt_lddt.py` and `tm_matrix.py` contains definitions for lDDT and TM-score, respectively. `AF_batch_analysis.py` compares a folder of multiple AF predictions against two reference structures and generates a csv file. The AF-cluster predictions for *Mad2* and *KaiB* and generated .csv files are also saved here. To plot these results, i.e. create Fig. 10, the `plot_all_models.py` may be used. Finally, the script `filter_rfah.py` contains the implementation for the "blind" selection process used for *RfaH*.

Note, to run AF-Cluster we refer to [their repository](https://github.com/HWaymentSteele/AF_Cluster). Also, in the [CF-random repository](https://github.com/ncbi/CF-random) one can find all the AlphaFold predictions we used for RfaH. To patch a pdb file that is missing residues — as we did for 2OUG — check out the [MODELLER](https://salilab.org/modeller/resources/missing_residues.html) software (by overriding the `select_atoms` function in a `Model` class the rest of the structure is kept intact).

## Coulomb Explosion Simulations

The folder `Coulomb Explosion Simulations/` contains the Python scripts we used to run GROMACS automatically and in parallel. Also the script `align_structures.py` is stored here, which uses the PyMOL API to align structures on the range of residues specified.  `GROMACS.py` contains function definitions necessary to run GROMACS commands, while `runner.py` mostly manages files and parallelisation. `input_data/Atomic_data` contains parameters needed for MoLDSTRUCT to model electron transitions, and the `.mdp` files includes all parameters used in our simulations. In `proteins/` one can find all pdbs that were simulated in the project.

## Analysis

Data analysis and machine learning scripts are found in the folder `Analysis/`.
h5 files stored in the `sims/` subfolder are first re-packaged
in the Bunch format by the script `data_packaging.py`. Protein names are
changed to follow the convention established in the report by the script 
`dataset_function.py`. This step is optional, though keep in mind that
other scripts expect these names by default when loading datasets.

PCA and calculation of PC Euclidean distances between explosion maps
are performed by the script `data_analysis_plotting.py`. From there comparrissons 
can be done to pre-explosions by first running `multi_score.py` directly on the output.
