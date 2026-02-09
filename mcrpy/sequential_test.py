import mcrpy
import numpy as np
from mcrpy.view import view
import logging
from mcrpy.src import loader
from mcrpy.src.descriptor_factory import get_class

''' This code is for the purpose of performing a microstructure reconstruction 
using differiable and non-differentiable descriptors. For performance reasons, 
this is split into 2 steps. 
Step 1: Reconstruction using only the differentiable descriptors.
Step 2: Further Reconstruction using all desired descriptors with the reconstructed 
MS from step 1 as start point for the iteration.'''

desired_descriptor_list = ['Tortuosity3D',
                    'VolumeFractions3D',
                    'TPB3D',
                    'DPB3D',
                    'Percolation',
                    'FFTCorrelations3D'] #TODO: Percolation3D


# Load descriptor modules to check whether they are differentiable
loader.load_plugins([f'mcrpy.descriptors.{descriptor}' for descriptor in desired_descriptor_list])

# Descriptor types that are differentiable
desired_differentiable_descriptor_list = [descriptor for descriptor in desired_descriptor_list 
                                   if get_class(descriptor).is_differentiable]

use_multigrid = False
use_multiphase = True
do_paraview_plot = False

limit_to = 2    # maximale Laenge des Vektors in x oder y-Richtung for example for FFTCorrelations3D.
                # if used on singlegrid --> only short-range descriptor, else also long-range.

# Characterization of the MS using all descriptors
characterization_settings = mcrpy.CharacterizationSettings(descriptor_types=desired_descriptor_list,
                                                           full_3d=True,
                                                           limit_to=limit_to,
                                                           use_multigrid_descriptor=use_multigrid,
                                                           use_multiphase=use_multiphase,
                                                           target_folder='results',
                                                           logging_level=logging.WARNING)

print("Load similar 3D microstructure ...")
#ms2D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_2D_20x20.npy")
#ms = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy",use_multiphase=use_multiphase)
# ms = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Diag_4x4x4.npy",use_multiphase=use_multiphase)
# ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_3D_2x2x2.npy",use_multiphase=False)
#ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_8x8x8.npy",use_multiphase=False)
ms = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_32x32x32.npy",use_multiphase=use_multiphase)
#ms = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Holzer2020_Segmented_Fine_Pristine_Zoom0.33_size600 copy.npy",use_multiphase=use_multiphase)


if do_paraview_plot:
    view(ms,save_as='MyMicrostructure')

print("Characterize similar 3D image ...")
description3D = mcrpy.characterize(ms, characterization_settings)
print("="*60)
print(f"characterization: {description3D}")

reconstruction_settings3D = mcrpy.ReconstructionSettings(descriptor_types=desired_descriptor_list,
                                    use_multiphase=use_multiphase, 
                                    max_iter=200,
                                    full_3d=True,
                                    limit_to=limit_to,
                                    convergence_data_steps=1, outfile_data_steps=1,
                                    #optimizer_type="GeneticAlgorithm",
                                    optimizer_type="SimulatedAnnealing",
                                    use_multigrid_descriptor=use_multigrid,
                                    use_multigrid_reconstruction=use_multigrid,
                                    target_folder='results',
                                    population_size=5,
                                    tolerance=1e-5,
                                    logging_level=logging.INFO)


print("="*60)
print("Reconstruct microstructure...")
convergence_data3D, ms_reconstruct3D = mcrpy.reconstruct(description3D, (32, 32, 32), 
                                          settings=reconstruction_settings3D,
                                          )

# view(convergence_data3D)

# # view(ms_reconstruct3D, )
# print("characterization of reconstructed ms:")
# ms_result = convergence_data3D['raw_data'][-1]
# description3D = mcrpy.characterize(ms_result, characterization_settings)
# print("="*60)
# print(f"characterization: {description3D}")

# logging.info("="*60)
# logging.shutdown() 