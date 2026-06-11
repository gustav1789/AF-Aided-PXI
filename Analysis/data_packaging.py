# This script is used to package explosion maps into pickled Bunch objects
# It uses the function package_dataset in the script analysis_functions.py
# Inputs are explosion maps together with associated protein names as h5 files
# Bunch objects contain 3 keys: data (explosion map vectors), target (protein index), target_names (name of protein)

import h5py
import os

from Analysis.analysis_functions import package_dataset

# Greek letter alpha: α

ROOT = os.path.abspath(".") # Make sure you are working in the Analysis directory

sim_path = os.path.join(ROOT, "sims")
h5_path = os.path.join(sim_path, "KaiB.h5")

f = h5py.File(h5_path, "r")

print(f"All conformations: {f.keys()}")

import pickle

mad2_af = ['cluster_055', 'cluster_081', 'cluster_019', 'cluster_027']
mad2_og = ['1S2H', '1DUJ']
mad2_extra_af = ['cluster_087']
mad2_extended_tutti = ['1S2H', 'cluster_055', 'cluster_081', '1DUJ', 'cluster_019', 'cluster_027', 'cluster_087']

kaib_af = ['KaiB_033', 'KaiB_102', 'KaiB_048', 'KaiB_072']
kaib_og = ['4KSOchainA', '5N8YchainG']
kaib_extra_af = ['KaiB_094', 'KaiB_114']
kaib_extended_tutti = ['4KSOchainA', 'KaiB_033', 'KaiB_102', 'KaiB_114', '5N8YchainG', 'KaiB_048', 'KaiB_072', 'KaiB_094']

rfah_tutti = ['RfaH-AIn', 'RfaH-α1', 'RfaH-active', 'RfaH-α2', 'RfaH-α3', 'RfaH-α4', 'RfaH-α5', 'RfaH-α6']
rfah_healthy = ['RfaH-AIn', 'RfaH-α1', 'RfaH-α2', 'RfaH-α3', 'RfaH-α4', 'RfaH-α5', 'RfaH-α6']
rfah_af = ['RfaH-α1', 'RfaH-α2', 'RfaH-α3', 'RfaH-α4', 'RfaH-α5', 'RfaH-α6']
rfah_og = ['RfaH-AIn', 'RfaH-active']

#Example:
dataset = package_dataset(h5_path, kaib_af)
with open('kaib_af.pkl', 'wb') as f:
    pickle.dump(kaib_af, f)