"""
   Copyright 2025 TU Dresden (Martin Sobczyk as Scientific Employee)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
# from __future__ import annotations

import tensorflow as tf
from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor
from numpy.typing import NDArray
from typing import Union, Any
import numpy as np
from scipy.ndimage import label

class Percolation(PhaseDescriptor):
    is_differentiable = False
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    @staticmethod
    def make_singlephase_descriptor(
        connectivity : Union[int,str] = 'sides', # implemented connectivities: only via sides, only via sides and edges, and via sides, edges and corners. 
        # for connectivity only via sides --> possible arguments: ['sides' (for 2D and 3D), 6 (for 3D), 4 (for 2D)], 
        # for connectivity only via sides and edges --> possible arguments: ['edges' (for 2D and 3D), 18 (for 3D), 4 (for 2D)] 
        # for connectivity via sides, edges and corners --> possible arguments ['corners' (for 2D and 3D), 26 (for 3D), 8 (for 2D)]  
        directions : Union[int,list[int]] = 1, #0:x, 1:y, 2:z
        phase_of_interest : Union[int,list[int]] = 1, #for which phase number the tortuosity shall be calculated
        **kwargs) -> callable:

        def percolation3D(ms_phase_of_interest: NDArray[np.bool_]):
            
            dimensionality = len(ms_phase_of_interest.shape)
            assert dimensionality in [2,3] 

            # labeling of connected voxels with equal integers using scipy. 
            # The scheme how connection is defined is governed by the structure argument.
            # (default is connection via sides, so if structure == None)
            # definition of possible connections:
            assert connectivity in ['sides', 'edges', 'corners', 6, 18, 28, 4, 8], "Valid inputs for connectivity are ['sides', 'edges, 'corners' 4, 6, 8, 18, 26]"
            #--------- connect nodes for connectivity=6 implemented for 2D and 3D: -----------------
            if ((connectivity in ['sides', 6] and dimensionality == 3) or 
                (connectivity in ['sides', 4] and dimensionality == 2) or
                (connectivity in ['edges', 4] and dimensionality == 2)):
                def increment_direction(direction_tuple, index, val):
                    new_direction = np.copy(direction_tuple)
                    new_direction[index] += val
                    return tuple(new_direction)
                connectivity_directions=np.zeros(dimensionality)
                connectivity_directions = np.array([increment_direction(connectivity_directions, index, val) for index in range(dimensionality) for val in [1, -1]], dtype=int)  # This creates both +1 and -1 increments
               
            elif connectivity in ['edges', 18] and dimensionality == 3:
                connectivity_directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                            (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)], dtype=int)
                                
            elif connectivity in ['corners', 26] and dimensionality == 3:
                connectivity_directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                      (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                      (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                      (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
                      (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)], dtype=int)  
            
            elif connectivity in ['corners', 8] and dimensionality == 2:
                connectivity_directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, 1), (1, -1), (-1, -1)], dtype=int)  
            else:
                raise Exception(f'Connectivity argument (connectivity={connectivity}) and dimensionality (dimensionality={dimensionality}D) mismatch!  \nValid arguments for 3D are [sides, edges, corners, 6, 18, 28] \nand for 2D [sides, edges, corners, 4, 8].')

            # convert connectivity_directions to a connectivity_structure as required in label function
            connectivity_structure = np.zeros((3,) * dimensionality, dtype=int)
            connectivity_structure[(1,)*dimensionality] = 1 #center point also needs to be 1 in 2D and 3D
            for row in connectivity_directions:
                ind = row + np.ones(dimensionality,dtype=int)
                connectivity_structure[tuple(ind)] = 1 


            print(f'shape ms_phase_of_interest: {ms_phase_of_interest.shape}')
            print(f'connectivity_structure: {connectivity_structure}')
            print(f'shape of connectivity_structure: {connectivity_structure.shape}')
            
            print(f'ms_phase_of_interest: {ms_phase_of_interest}')
            array_labeled, n_labels = label(ms_phase_of_interest, structure=connectivity_structure)

            print(f'array_labeled: {array_labeled}')
            print(f'n_labels: {n_labels}')

            fraction_connected_voxels = 0 
            fraction_isolated_voxels = 0
            fraction_unknown_voxels = 0
            return fraction_connected_voxels, fraction_isolated_voxels, fraction_unknown_voxels



        #@tf.function
        def model(ms: Union[tf.Tensor, NDArray[Any]]) -> tf.Tensor:
            
            # ms_phase_of_interest is an np.ndarray with bool values representing the 
            # microstructure ms where the searched for phase is represented as True, else False.
            # For further calculations, use ms_phase_of_interest:
            
            if (len(ms.shape) > 3): # if called from mcrpy (would be a 4D tensor). If an microstructure is already 2 or 3D, don't change it.
                if ms.shape[0]==1: #if the microstructure is 2D
                    desired_shape =tuple(ms.shape[1:-1])
                else: #if the microstructure is 3D
                    desired_shape =tuple(ms.shape[0:-1])
                ms = tf.reshape(ms, desired_shape).numpy()
            
            ms_phase_of_interest = ms == phase_of_interest
            
            fraction_connected_voxels, fraction_isolated_voxels, fraction_unknown_voxels = percolation3D(ms_phase_of_interest)

            return tf.cast(tf.constant(fraction_connected_voxels), tf.float64)#, tf.cast(tf.constant(mean_tortuosity), tf.float64)
        return model

    @staticmethod
    def make_multiphase_descriptor():
        return 0

def register() -> None:
    descriptor_factory.register("Percolation", Percolation)

       

if __name__=="__main__":

    import os
    folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    #minimal_example_ms = os.path.join(folder,'BlockingLayer_X_2D_20x20.npy')
    minimal_example_ms = os.path.join(folder,'BlockingLayer_X_32x32x32.npy')




    #minimal_example_ms = os.path.join(folder,'composite_resized_s.npy')

    #minimal_example_ms = os.path.join(result_folder,'BlockingLayer_X_2D_20x20.npy')

    # for filename in os.listdir(folder):
    #     if filename.endswith('.npy'):  # Check if the file has a .npy extension
    #         file_path = os.path.join(folder, filename)  # Full path to the file
    #         print(f'filename: {filename}')
    #         ms = np.load(file_path)  # Load the .npy file
    #         #print(f'ms: {ms}')
    #         print(f'type: {type(ms[0])}')
    #         print(f'ms type: {type(ms)}, size: {ms.size}')
    #         print(f'shape: {ms.shape}')
    #         print(f'unique: {np.unique(ms)}')
    #         print('\n\n')


    #minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    # minimal_example_ms = os.path.join(folder,'alloy_resized_s.npy')
    
    ms = np.load(minimal_example_ms)


    ms = np.zeros((3, 3, 3))
    ms[1,:, :] = 1

    # print(f'ms: {ms}')
    # print(f'type: {type(ms[0])}')
    # print(f'ms type: {type(ms)}, size: {ms.size}')
    # print(f'shape: {ms.shape}')
    # print(np.unique(ms))


#     # ms = np.zeros((3, 3, 3))
#     # ms[1,:,:] = 1
#     # print(f'ms: {ms}')
#     # print(f'shape: {(ms.shape)}')
#     # print(f'ms type: {type(ms)}, size: {ms.size}')

#     ms = np.zeros((3, 3))
#     ms[1,:] = 1
#     print(f'ms: {ms}')
#     print(f'shape: {(ms.shape)}')
#     print(f'ms type: {type(ms)}, size: {ms.size}')

#     # ms = np.random.randint(2, size=(3, 3, 3))
#     # print(f'ms: {ms}, size: {ms.size}')




    # Step 2: Open the pickle file
    # result_folder = '/home/sobczyk/Dokumente/MCRpy/mcrpy/results' 
    # pickle_filename = os.path.join(result_folder,'BlockingLayer_X_2D_32x32_characterization.pickle')
    # with open(pickle_filename, 'rb') as file:  # Replace 'filename.pkl' with your filepath
    #     # Step 3: Load the data
    #     data = pickle.load(file)
    # print(f"data: {data}")
 

##------------------------------------------------------------------
   
    percolation_descriptor = Percolation()
    singlephase_descriptor = percolation_descriptor.make_singlephase_descriptor()

    fraction_connected_voxels = singlephase_descriptor(ms)
    print('\n -----------------------------')
    print(f'Fraction of connected voxels: {fraction_connected_voxels}')

##------------------------------------------------------------------
