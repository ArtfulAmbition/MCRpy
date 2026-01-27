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
ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/Diag_3x3x3.npy")

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
                                    use_multiphase=True, max_iter=2,
                                    full_3d=True,
                                    convergence_data_steps=1, outfile_data_steps=1,
                                    optimizer_type="GeneticAlgorithm",
                                    #mutation_rule="relaxed_neighbor",
                                    use_multigrid_descriptor=False,
                                    use_multigrid_reconstruction=False,
                                    target_folder='results',
                                    logging_level=logging.WARNING)

# init_ms = np.full_like(ms)
print("="*60)
print("Reconstruct microstructure...")
convergence_data3D, ms_reconstruct3D = mcrpy.reconstruct(description3D, (20, 20, 20), 
                                          settings=reconstruction_settings3D,
                                          )

# view(convergence_data3D)

# view(ms_reconstruct3D, )