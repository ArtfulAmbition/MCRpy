import mcrpy
import numpy as np


limit_to = 8 # maximale Laenge des Vektors in x oder y-Richtung
descriptor_types = ['Tortuosity3D','VolumeFractions3D']
characterization_settings = mcrpy.CharacterizationSettings(descriptor_types=descriptor_types,
                                                           full_3d=True,
                                                           use_multigrid_descriptor=False)

print("Load similar 3D microstructure ...")
ms2D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_2D_20x20.npy")
ms3D = mcrpy.load("/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy")

ms = ms3D

print("Characterize similar 3D image ...")
description3D = mcrpy.characterize(ms, characterization_settings)
print("="*60)
print("characterization:")

reconstruction_settings3D = mcrpy.ReconstructionSettings(descriptor_types=descriptor_types,
                                    use_multiphase=False, max_iter=40002,
                                    convergence_data_steps=5000, outfile_data_steps=10000,
                                    optimizer_type="YTPost",
                                    mutation_rule="reduce_any_islands",
                                    use_multigrid_descriptor=False)

# init_ms = np.full_like(ms)
# print("Reconstruct microstructure based on 3 orthogonal slices...", flush=True)
# convergence_data3D, ms_reconstruct3D = mcrpy.reconstruct(description3D, (150, 150, 150), 
#                                          settings=reconstruction_settings3D,
#                                          initial_microstructure=init_ms)