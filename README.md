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

Coming soon
## Coulomb Explosion Simulations

Coming soon

## Analysis

Data analysis and machine learning scripts are found in the folder "Analysis".
h5 files stored in the "sims" subfolder are first re-packaged
in the Bunch format by the script "data_packaging.py". Protein names are
changed to follow the convention established in the report by the script 
"dataset_function.py". This step is optional, though keep in mind that
other scripts expect these names by default when loading datasets.

PCA and calculation of PC Euclidean distances between explosion maps
are performed by the script "data_analysis_plotting.py". From there comparrissons 
can be done to pre-explosions by first running "multi_score.py" directely on the output.
