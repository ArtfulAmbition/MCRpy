import mcrpy
import numpy as np
import os
import pickle
from mcrpy.view import view
import logging

limit_to = 8 # maximale Laenge des Vektors in x oder y-Richtung
descriptor_types = ['Tortuosity3D','VolumeFractions3D']
characterization_settings = mcrpy.CharacterizationSettings(descriptor_types=descriptor_types,
                                                           full_3d=True,
                                                           use_multigrid_descriptor=False,
                                                           use_multiphase=True,
                                                           target_folder='results',
                                                           logging_level=logging.WARNING)

print("Load similar 3D microstructure ...")
ms2D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_2D_20x20.npy")
#ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy")
# ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Diag_4x4x4.npy")
# ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_3D_2x2x2.npy",use_multiphase=False)
ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_8x8x8.npy",use_multiphase=True)

ms = ms3D

print("Characterize similar 3D image ...")
description3D = mcrpy.characterize(ms, characterization_settings)
print("="*60)
print(f"characterization: {description3D}")

# reconstruction_settings3D = mcrpy.ReconstructionSettings(descriptor_types=descriptor_types,
#                                     use_multiphase=False, max_iter=4002,
#                                     full_3d=True,
#                                     convergence_data_steps=5000, outfile_data_steps=10000,
#                                     optimizer_type="SimulatedAnnealing",
#                                     mutation_rule="relaxed_neighbor",
#                                     use_multigrid_descriptor=False)

reconstruction_settings3D = mcrpy.ReconstructionSettings(descriptor_types=descriptor_types,
                                    use_multiphase=True, max_iter=30,
                                    full_3d=True,
                                    convergence_data_steps=1, outfile_data_steps=1,
                                    optimizer_type="GeneticAlgorithm",
                                    use_multigrid_descriptor=False,
                                    use_multigrid_reconstruction=False,
                                    target_folder='results',
                                    population_size=500,
                                    tolerance=1e-5,
                                    logging_level=logging.INFO)


print("="*60)
print("Reconstruct microstructure...")
convergence_data3D, ms_reconstruct3D = mcrpy.reconstruct(description3D, (8, 8, 8), 
                                          settings=reconstruction_settings3D,
                                          )

view(convergence_data3D)

# view(ms_reconstruct3D, )
print("characterization of reconstructed ms:")
ms_result = convergence_data3D['raw_data'][-1]
description3D = mcrpy.characterize(ms_result, characterization_settings)
print("="*60)
print(f"characterization: {description3D}")

logging.info("="*60)
logging.shutdown() 