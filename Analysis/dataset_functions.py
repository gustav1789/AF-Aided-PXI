# This script contains functions for various operations on the Bunch datasets
# Importantly, it was used to rename KaiB and Mad2 proteins
# This was done in order to follow the naming convention used in the report

import pickle
import numpy as np
from sklearn.utils import Bunch

def merge_datasets(ds1, ds2):
    # Load the two datasets
    with open('KaiB_af.pkl', 'rb') as f:
        ds1 = pickle.load(f)

    with open('KaiB_extra.pkl', 'rb') as f:
        ds2 = pickle.load(f)

    # Remap ds2 target labels to avoid collision with ds1
    offset = len(ds1.target_names)
    remapped_targets = ds2.target + offset

    # Combine
    combined = Bunch(
        data=np.concatenate([ds1.data, ds2.data], axis=0),
        target=np.concatenate([ds1.target, remapped_targets], axis=0),
        target_names=np.concatenate([ds1.target_names, ds2.target_names])
    )

    # Optionally save
    with open('KaiB_af.pkl', 'wb') as f:
        pickle.dump(combined, f)

def reverse_dataset(old_dataset, new_dataset):
    # Reverse dataset

    # Load the pickled Bunch object
    with open(old_dataset, 'rb') as f:
        bunch = pickle.load(f)

    # Reverse the data and target arrays
    reversed_bunch = Bunch(
        data=bunch.data[::-1],
        target=bunch.target[::-1],
        target_names=bunch.target_names  # these are class labels, so don't reverse
    )

    # Save the reversed Bunch object
    with open(new_dataset, 'wb') as f:
        pickle.dump(reversed_bunch, f)


with open('kaib_extended_af.pkl', 'rb') as f:
    bunch= pickle.load(f)

# kaib_extended_tutti
#old_names = ['4KSOchainA', 'KaiB_033', 'KaiB_102', 'KaiB_114', '5N8YchainG', 'KaiB_048', 'KaiB_072', 'KaiB_094']
#new_names = ['KaiB-ground', 'KaiB-α1', 'KaiB-α2', 'KaiB-α3', 'KaiB-FS', 'KaiB-α4', 'KaiB-α5', 'KaiB-α6']

#kaib_af
#old_names = ['KaiB_033','KaiB_102', 'KaiB_048', 'KaiB_072']
#new_names = ['KaiB-α1', 'KaiB-α2', 'KaiB-α4', 'KaiB-α5']

#kaib_og
#old_names = ['4KSOchainA', '5N8YchainG']
#new_names = ['KaiB-ground', 'KaiB-FS']

#kaib_extra_af
#old_names = ['KaiB_094', 'KaiB_114']
#new_names = ['KaiB-α6', 'KaiB-α3']

#mad2_extended_tutti
old_names = ['1S2H', 'cluster_055', 'cluster_081', '1DUJ', 'cluster_019', 'cluster_027', 'cluster_087']
new_names = ['Mad2-closed', 'Mad2-α1', 'Mad2-α2', 'Mad2-open', 'Mad2-α3', 'Mad2-α4', 'Mad2-α5']

#mad2_af
#old_names = ['cluster_055', 'cluster_081', 'cluster_019', 'cluster_027']
#new_names = ['Mad2-α1', 'Mad2-α2', 'Mad2-α3', 'Mad2-α4']

#mad2_og
#old_names = ['1S2H', '1DUJ']
#new_names = ['Mad2-closed', 'Mad2-open']

#mad2_extra_af
#old_names = ['cluster_087']
#new_names = ['Mad2-α5']

#mad2_extended_af
#old_names = ['cluster_055', 'cluster_081', 'cluster_019', 'cluster_027', 'cluster_087']
#new_names = ['Mad2-α1', 'Mad2-α2', 'Mad2-α3', 'Mad2-α4', 'Mad2-α5']

# Build a mapping from old → new
name_map = dict(zip(old_names, new_names))

# Replace target_names in the Bunch
bunch.target_names = np.array([name_map.get(name, name) for name in bunch.target_names])

with open('mad2_extended_tutti.pkl', 'wb') as f:
    pickle.dump(bunch, f)