import mcrpy

# Define settings with both descriptor types
settings = mcrpy.CharacterizationSettings(
    descriptor_types=['Tortuosity', 'FFTCorrelations'],
    limit_to=8
)

# Load microstructure
ms = mcrpy.load('example_microstructures/BlockingLayer_X_2D_32x32.npy')

# Characterize with both descriptors
characterization = mcrpy.characterize(ms, settings)


import os
os.system("\
./mcrpy/characterize.py \
    --microstructure_filenames example_microstructures/BlockingLayer_X_32x32x32.npy \
    --descriptor_types Tortuosity \
    --slice_mode no_slicing \
    --information tort3d \
    --data_folder results")

import os
os.system("\
./mcrpy/characterize.py \
    --microstructure_filenames example_microstructures/BlockingLayer_X_32x32x32.npy \
    --descriptor_types FFTCorrelations \
    --limit_to 8 \
    --slice_mode average \
    --information fft2d \
    --data_folder results")