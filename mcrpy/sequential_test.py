import mcrpy
import numpy as np
from mcrpy.view import view
import logging
from mcrpy.src import loader
from mcrpy.src.descriptor_factory import get_class
from typing import Union
from mcrpy.src.Microstructure import Microstructure
from mcrpy.src.Settings import CharacterizationSettings, ReconstructionSettings
import datetime

''' This code is for the purpose of performing a microstructure reconstruction 
using differiable and non-differentiable descriptors. For performance reasons, 
this is split into 3 steps.
Step 1: Loading and Characterization of a MS to resemble
Step 2: Loading a MS as starting point for the further optimization 
Step 3: Reconstruction using only the differentiable descriptors. Save the results.
Step 2: Further Reconstruction using all desired descriptors with the reconstructed 
MS from step 1 as start point for the iteration.'''

class MultiStepOptimizer:
    # Wrapper class to obtain a desired MS from a multistep sequence of characterizations and reconstructions
    def __init__(self,
                 goal_ms:Union[str,Microstructure]=None, # path or Microstructure of the ms to resemble after optimization
                 characterization_goal:dict = None,
                 full_3d:bool=True,
                 max_iter:int = 100,
                 population_size = 100,
                 is_differentiable:bool=False,
                 characterization_settings:CharacterizationSettings=None,
                 reconstruction_settings:ReconstructionSettings=None,
                 descriptor_list:list[str]=None,
                 use_multigrid=False,
                 use_multiphase=True,
                 verbose:bool=False,
                 save_files:bool=False,
                 tolerance=1e-4,
                 initial_ms:Union[str,Microstructure] = None, # path or Microstructure of the ms used as starting point for optimization
                 info:str='',
                 result_ms_shape=None,
                 plot_intial_ms:bool=False,
                 plot_result_ms:bool=False,
                 plot_convergence_data:bool=False,
                 **kwargs):
        self.save_files = save_files
        self.tolerance = tolerance,
        self.population_size = population_size
        self.max_iter = max_iter
        self.datetime_string = ('{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        self.is_differentiable=is_differentiable
        self.full_3d = full_3d
        self.info=info
        self.verbose = verbose
        self.result_ms = None
        self.use_multigrid = use_multigrid
        self.use_multiphase = use_multiphase
        self.characterization_goal=characterization_goal
        assert isinstance(descriptor_list,list)
        self.descriptor_list = descriptor_list
        self.map_descriptor_list()
        

        self.initial_ms = self.get_microstructure(initial_ms)
        self.goal_ms = self.get_microstructure(goal_ms)

        if not result_ms_shape:
            if self.initial_ms:
                self.goal_ms_shape = self.initial_ms.spatial_shape
            else:
                raise AssertionError('Either a result_ms_shape must be given or an initial_ms to derive the output size from.')
            
        if self.verbose:
            if self.initial_ms:
                print(f'Using initial ms with shape {self.initial_ms.spatial_shape}')
            else:
                print(f'Using no initial ms.')

            if not self.goal_ms and not self.characterization_goal:
                raise AssertionError('Goal MS or goal_characteriation must be specified.')
        
        if not characterization_settings:
            self.setup_default_characterization_settings()
        else:
            self.characterization_settings = characterization_settings

        if not reconstruction_settings:
            self.setup_default_reconstruction_settings()
        else:
            self.reconstruction_settings = reconstruction_settings
        

    def get_microstructure(self, ms:Union[str,Microstructure]):
        if isinstance(ms,str):
            ms_path = ms
            return mcrpy.load(ms_path,use_multiphase=self.use_multiphase)
        elif isinstance(ms,Microstructure) or not ms:
            return ms            
        else:
            raise TypeError

    def load_microstructures(self):
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

    def view_initial_ms(self):
        view(self.initial_ms,save_as='initial_MS'+self.info)

    def view_result_ms(self):
        view(self.result_ms,save_as='result_MS'+self.info)

    def setup_default_characterization_settings(self):

        self.characterization_settings = mcrpy.CharacterizationSettings(descriptor_types=self.descriptor_list,
                                                           full_3d=self.full_3d,
                                                           limit_to=8,
                                                           use_multigrid_descriptor=self.use_multigrid,
                                                           use_multiphase=self.use_multiphase,
                                                           target_folder='results',
                                                           logging_level=logging.WARNING)

    def setup_default_reconstruction_settings(self):

        self.reconstruction_settings = mcrpy.ReconstructionSettings(
            descriptor_types=self.descriptor_list,
            use_multiphase=self.use_multiphase, 
            max_iter=self.max_iter,
            full_3d=False,
            limit_to=8,
            convergence_data_steps=1, outfile_data_steps=1,
            optimizer_type="LBFGSB",
            #optimizer_type="SimulatedAnnealing",
            use_multigrid_descriptor=self.use_multigrid,
            use_multigrid_reconstruction=self.use_multigrid,
            target_folder='results',
            population_size=self.population_size,
            tolerance=self.tolerance,
            logging_level=logging.INFO)

    def get_characterization_goal(self, redo=True, verbose=None):
        if verbose==None:
            verbose = self.verbose

        if redo:
            if verbose:
                print(f"Characterize 3D image of shape {self.goal_ms.spatial_shape}...")
            self.characterization_goal = mcrpy.characterize(self.goal_ms, self.characterization_settings)
        else:
            pass

        if verbose:
            print("="*60)
            print(self.characterization_goal)
            print("="*60)

        return self.characterization_goal

    def reconstruct(self):
        if self.verbose:
            print(f"Reconstruct microstructure with shape {self.goal_ms_shape}...")

        self.convergence_data, self.result_ms = mcrpy.reconstruct(
                                        self.characterization_goal,
                                        self.goal_ms_shape, 
                                        settings=self.reconstruction_settings,
                                        initial_microstructure=self.initial_ms)
        
    def view_convergence_data(self):
        view(self.convergence_data)

    def map_descriptor_list(self):
        # Define the mapping between 2D and 3D descriptors
        dim_key = "3D" if self.full_3d else "2D"

        descriptor_mapping = {
            "Tortuosity": {"2D": None, "3D": "Tortuosity3D"},
            "VolumeFractions": {"2D": "VolumeFractions", "3D": "VolumeFractions3D"},
            "TPB": {"2D": "TPB", "3D": "TPB3D"},
            "DPB": {"2D": "DPB", "3D": "DPB3D"},
            "Percolation": {"2D": None, "3D": "Percolation3D"},
            "FFTCorrelations": {"2D": "FFTCorrelations", "3D": "FFTCorrelations3D"}
            }
        
        dim_mapped_descriptor_list = []
        for descriptor_string in self.descriptor_list:
            try:
                mapped_descriptor = descriptor_mapping[descriptor_string][dim_key]
            except:
                try:
                    corrected_descriptor_string = descriptor_string.removesuffix("3D")
                    mapped_descriptor = descriptor_mapping[corrected_descriptor_string][dim_key]
                except:
                    raise KeyError("The descriptor {descriptor} is not in descriptor_mapping.")
            if mapped_descriptor:
                dim_mapped_descriptor_list.append(mapped_descriptor)
        self.descriptor_list = dim_mapped_descriptor_list
        
        if self.is_differentiable:
            loader.load_plugins([f'mcrpy.descriptors.{descriptor}' for descriptor in self.descriptor_list])
            # Descriptor types that are differentiable
            differentiable_descriptor_list = [descriptor for descriptor in self.descriptor_list 
                                    if get_class(descriptor).is_differentiable]
            self.descriptor_list = differentiable_descriptor_list
        
        if self.verbose:
            print("="*60)
            print(f'setting descriptor list to: {self.descriptor_list}.')
            print("="*60)
    
            

        ##### Define the list of desired descriptors and define neccesary parameters:
        desired_descriptor_list = ['Tortuosity3D',
                            'VolumeFractions3D',
                            'TPB3D',
                            'DPB3D',
                            'Percolation',
                            'FFTCorrelations3D']
    
##### Define the list of desired descriptors and define neccesary parameters:
desired_descriptor_list = ['Tortuosity3D',
                    'VolumeFractions3D',
                    'TPB3D',
                    'DPB3D',
                    'Percolation',
                    'FFTCorrelations3D']

diff_2D_opimizer = MultiStepOptimizer(full_3d=False,
                                      goal_ms="/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy",
                                      is_differentiable=True,
                                      descriptor_list=desired_descriptor_list,
                                      verbose=True)

diff_2D_opimizer.get_characterization_goal(verbose=True)








# do_paraview_plot = False
# if do_paraview_plot:
#     view(ms_to_reconstruct,save_as='MyMicrostructure')



# ##### Characterization of the MS which should be resembled using all descriptors
# characterization_settings3D = mcrpy.CharacterizationSettings(descriptor_types=desired_descriptor_list,
#                                                            full_3d=True,
#                                                            limit_to=limit_to,
#                                                            use_multigrid_descriptor=use_multigrid,
#                                                            use_multiphase=use_multiphase,
#                                                            target_folder='results',
#                                                            logging_level=logging.WARNING)


# ##### Characterization of the MS which should be resembled using all descriptors
# characterization_settings2D = mcrpy.CharacterizationSettings(descriptor_types=desired_differentiable_descriptor_list2D,
#                                                            full_3d=False,
#                                                            limit_to=limit_to,
#                                                            use_multigrid_descriptor=use_multigrid,
#                                                            use_multiphase=use_multiphase,
#                                                            target_folder='results',
#                                                            logging_level=logging.WARNING)

# print(f"Characterize 3D image of shape {ms_to_reconstruct_shape}...")
# description2D = mcrpy.characterize(ms_to_reconstruct, characterization_settings2D)
# print("="*60)
# print(f"characterization: {description2D}")


# print(f"Characterize 3D image of shape {ms_to_reconstruct_shape}...")
# description3D = mcrpy.characterize(ms_to_reconstruct, characterization_settings)
# print("="*60)
# print(f"characterization: {description3D}")


##### Reconstruction using differential descriptors in 2D and using and initial MS as starting point:
# reconstruction_settings2D_differentiable = mcrpy.ReconstructionSettings(
#             descriptor_types=desired_differentiable_descriptor_list2D,
#             use_multiphase=use_multiphase, 
#             max_iter=100,
#             full_3d=False,
#             limit_to=limit_to,
#             convergence_data_steps=1, outfile_data_steps=1,
#             optimizer_type="LBFGSB",
#             #optimizer_type="SimulatedAnnealing",
#             use_multigrid_descriptor=use_multigrid,
#             use_multigrid_reconstruction=use_multigrid,
#             target_folder='results',
#             population_size=100,
#             tolerance=1e-4,
#             logging_level=logging.INFO)

if initial_microstructure:
    print(f"Reconstruct microstructure with shape {initial_microstructure.spatial_shape}...")
else:
    print(f"Reconstruct microstructure with shape {ms_to_reconstruct.spatial_shape}...")

convergence_data2D, ms_reconstruct2D = mcrpy.reconstruct(description2D, ms_to_reconstruct_shape, 
                                          settings=reconstruction_settings2D_differentiable,
                                          initial_microstructure=initial_microstructure
                                          )
view(convergence_data2D)




# convergence_data3D, ms_reconstruct3D = mcrpy.reconstruct(description3D, ms_to_reconstruct_shape, 
#                                           settings=reconstruction_settings3D_differentiable,
#                                           initial_microstructure=initial_microstructure
#                                           )



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