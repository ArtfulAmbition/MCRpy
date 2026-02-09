import mcrpy
import numpy as np
from mcrpy.view import view
import logging
from mcrpy.src import loader
from mcrpy.src.descriptor_factory import get_class

''' This code is for the purpose of performing a microstructure reconstruction 
using differiable and non-differentiable descriptors. For performance reasons, 
this is split into 3 steps.
Step 1: Loading and Characterization of a MS to resemble
Step 2: Loading a MS as starting point for the further optimization 
Step 3: Reconstruction using only the differentiable descriptors. Save the results.
Step 2: Further Reconstruction using all desired descriptors with the reconstructed 
MS from step 1 as start point for the iteration.'''


##### Define the list of desired descriptors and define neccesary parameters:
desired_descriptor_list = ['Tortuosity3D',
                    'VolumeFractions3D',
                    'TPB3D',
                    'DPB3D',
                    'Percolation',
                    'FFTCorrelations3D']


# Load descriptor modules to check whether they are differentiable
loader.load_plugins([f'mcrpy.descriptors.{descriptor}' for descriptor in desired_descriptor_list])

# Descriptor types that are differentiable
desired_differentiable_descriptor_list = [descriptor for descriptor in desired_descriptor_list 
                                   if get_class(descriptor).is_differentiable]

desired_differentiable_descriptor_list2D = desired_descriptor_list = [
    'FFTCorrelations', 
    'VolumeFractions', 
    'DPB',
    'TPB', 
    'LinealPathApproximation']

use_multigrid = False
use_multiphase = True


##### Load all required microstructures:

#ms2D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_2D_20x20.npy")
ms_to_reconstruct = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy",use_multiphase=use_multiphase)
#initial_microstructure = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Random_20x20x20.npy",use_multiphase=use_multiphase)
initial_microstructure = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Random2_20x20x20.npy",use_multiphase=use_multiphase)
# ms = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Diag_4x4x4.npy",use_multiphase=use_multiphase)
# ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_3D_2x2x2.npy",use_multiphase=Faluse_multiphasese)
#ms_to_reconstruct = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_8x8x8.npy",use_multiphase=use_multiphase)
#ms = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_32x32x32.npy",use_multiphase=use_multiphase)
#ms = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Holzer2020_Segmented_Fine_Pristine_Zoom0.33_size600 copy.npy",use_multiphase=use_multiphase)
ms_to_reconstruct_shape = ms_to_reconstruct.spatial_shape


do_paraview_plot = False
if do_paraview_plot:
    view(ms_to_reconstruct,save_as='MyMicrostructure')

limit_to = 8    # maximale Laenge des Vektors in x oder y-Richtung for example for FFTCorrelations3D.
                # if used on singlegrid --> only short-range descriptor, else also long-range.

##### Characterization of the MS which should be resembled using all descriptors
characterization_settings = mcrpy.CharacterizationSettings(descriptor_types=desired_descriptor_list,
                                                           full_3d=True,
                                                           limit_to=limit_to,
                                                           use_multigrid_descriptor=use_multigrid,
                                                           use_multiphase=use_multiphase,
                                                           target_folder='results',
                                                           logging_level=logging.WARNING)


print(f"Characterize 3D image of shape {ms_to_reconstruct_shape}...")
description3D = mcrpy.characterize(ms_to_reconstruct, characterization_settings)
print("="*60)
print(f"characterization: {description3D}")


##### Reconstruction using differential descriptors and using and initial MS as starting point:
reconstruction_settings3D_differentiable = mcrpy.ReconstructionSettings(
            descriptor_types=desired_differentiable_descriptor_list2D,
            use_multiphase=use_multiphase, 
            max_iter=100,
            full_3d=False,
            limit_to=limit_to,
            convergence_data_steps=1, outfile_data_steps=1,
            optimizer_type="LBFGSB",
            #optimizer_type="SimulatedAnnealing",
            use_multigrid_descriptor=use_multigrid,
            use_multigrid_reconstruction=use_multigrid,
            target_folder='results',
            population_size=100,
            tolerance=1e-4,
            logging_level=logging.INFO)

if initial_microstructure:
    print(f"Reconstruct microstructure with shape {initial_microstructure.spatial_shape}...")
else:
    print(f"Reconstruct microstructure with shape {ms_to_reconstruct.spatial_shape}...")

convergence_data3D, ms_reconstruct3D = mcrpy.reconstruct(description3D, ms_to_reconstruct_shape, 
                                          settings=reconstruction_settings3D_differentiable,
                                          initial_microstructure=initial_microstructure
                                          )



# reconstruction_settings3D = mcrpy.ReconstructionSettings(descriptor_types=desired_descriptor_list,
#                                     use_multiphase=use_multiphase, 
#                                     max_iter=100,
#                                     full_3d=True,
#                                     limit_to=limit_to,
#                                     convergence_data_steps=1, outfile_data_steps=1,
#                                     #optimizer_type="GeneticAlgorithm",
#                                     optimizer_type="SimulatedAnnealing",
#                                     use_multigrid_descriptor=use_multigrid,
#                                     use_multigrid_reconstruction=use_multigrid,
#                                     target_folder='results',
#                                     population_size=100,
#                                     tolerance=1e-4,
#                                     logging_level=logging.INFO)



# print("="*60)
# if initial_microstructure:
#     print(f"Reconstruct microstructure with shape {initial_microstructure.spatial_shape}...")
# convergence_data3D, ms_reconstruct3D = mcrpy.reconstruct(description3D, ms_to_reconstruct_shape, 
#                                           settings=reconstruction_settings3D,
#                                           initial_microstructure=initial_microstructure
#                                           )

# view(convergence_data3D)

# # view(ms_reconstruct3D, )
# print("characterization of reconstructed ms:")
# ms_result = convergence_data3D['raw_data'][-1]
# description3D = mcrpy.characterize(ms_result, characterization_settings)
# print("="*60)
# print(f"characterization: {description3D}")

# logging.info("="*60)
# logging.shutdown() 