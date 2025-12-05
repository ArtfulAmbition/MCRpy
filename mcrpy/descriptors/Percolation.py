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
from typing import Union, Any, Tuple
import numpy as np
from scipy.ndimage import label
from mcrpy.descriptors.descriptor_utils.descriptor_utils import get_connectivity_directions


def get_connected_phases_of_interest(labeled_ms: np.ndarray[int], direction:int=0) -> np.ndarray[bool]:
            
            dimensionality = len(labeled_ms.shape)
            assert dimensionality in [2,3] # only 2 and 3D microstructures
            assert direction in [0,1,2]


            # Generate the appropriate slices based on dimensionality and direction
            slices_source = [slice(None)] * dimensionality  # Initialize a list of slices
            slices_target = [slice(None)] * dimensionality  # Initialize a list of slices
            slices_source[direction] =  0        # First layer
            slices_target[direction] = -1        # Last layer

            # Extract the source and target node labels using the dynamically created slices
            source_node_labels = np.unique(labeled_ms[tuple(slices_source)])
            target_node_labels = np.unique(labeled_ms[tuple(slices_target)])

            # Exclude the zero from source- and target_node_labels, since zero represents phases which are not of interest
            source_node_labels = source_node_labels[source_node_labels != 0]
            target_node_labels = target_node_labels[target_node_labels != 0]
            
            #Check which labels are present at the source and the target surface / edge
            labels_at_both_surface_and_target = np.intersect1d(source_node_labels, target_node_labels)

            boolean_mask_connected = np.isin(labeled_ms, labels_at_both_surface_and_target)
            return boolean_mask_connected, labels_at_both_surface_and_target

def get_labeled_ms(ms_phase_of_interest: NDArray[np.bool_], connectivity='sides') -> np.ndarray[bool]:
            
            dimensionality = len(ms_phase_of_interest.shape)
            assert dimensionality in [2,3] 

            connectivity_directions = get_connectivity_directions(dimensionality=dimensionality, connectivity=connectivity, mode="full")
            # convert connectivity_directions to a connectivity_structure as required in label function
            connectivity_structure = np.zeros((3,) * dimensionality, dtype=int)
            connectivity_structure[(1,)*dimensionality] = 1 #center point is always 1 in 2D and 3D
            for row in connectivity_directions:
                ind = row + np.ones(dimensionality,dtype=int)
                connectivity_structure[tuple(ind)] = 1 

            labeled_ms, n_labels = label(ms_phase_of_interest, structure=connectivity_structure)

            return labeled_ms, n_labels

class Percolation(PhaseDescriptor):
    is_differentiable = False
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    @staticmethod
    def make_singlephase_descriptor(
        connectivity : Union[int,str] = 'sides', # implemented connectivities: only via sides, only via sides and edges, and via sides, edges and corners. 
        # for connectivity only via sides --> possible arguments: ['sides' (for 2D and 3D), 6 (for 3D), 4 (for 2D)], 
        # for connectivity only via sides and edges --> possible arguments: ['edges' (for 2D and 3D), 18 (for 3D), 4 (for 2D)] 
        # for connectivity via sides, edges and corners --> possible arguments ['corners' (for 2D and 3D), 26 (for 3D), 8 (for 2D)]  
        direction : int = 0, #0:x, 1:y, 2:z
        phase_of_interest : Union[int,list[int]] = 0, #for which phase number(s) the tortuosity shall be calculated
        **kwargs) -> callable:

        def calculate_percolation(ms_phase_of_interest: NDArray[np.bool_]):

            dimensionality = len(ms_phase_of_interest.shape)
            assert dimensionality in [2,3] # only 2 and 3D microstructures

            labeled_ms, n_labels = get_labeled_ms(ms_phase_of_interest,connectivity=connectivity)
            # labeled_ms is an np.ndarray of the size of ms. 
            # It contains equal integer values for connected voxels of interest (cluster).
            # The function checks, if there is a connection between the sides of the ms in the prescribed direction.
            # This is the case, if there are the same cluster labels in labeled_ms at both relevant sides

            # find a boolean microstructre for clusters which are connected with the relevant sides of the specified direction 
            ms_connected_phase_of_interest, labels_at_both_surface_and_target = get_connected_phases_of_interest(ms_phase_of_interest, direction)

            n_connected_voxels:int = np.count_nonzero(ms_connected_phase_of_interest)

            ## Check for cluster labels which are present on all sides:
            #Accessing outer borders
            if dimensionality == 2:
                labels_at_borders = np.unique(
                    np.concatenate((
                    labeled_ms[0,:].flatten(),
                    labeled_ms[-1,:].flatten(),
                    labeled_ms[:,0].flatten(),
                    labeled_ms[:,-1].flatten(),
                )))
            else: # dimensionality == 3
                labels_at_borders = np.unique(
                    np.concatenate((
                    labeled_ms[0, :, :].flatten(),       
                    labeled_ms[-1, :, :].flatten(),     
                    labeled_ms[:, 0, :].flatten(),       
                    labeled_ms[:, -1, :].flatten(),      
                    labeled_ms[:, :, 0].flatten(),       
                    labeled_ms[:, :, -1].flatten()       
                )))

            # Extract labels, which are only present on the sides,
            # and not at the sides belonging to the specified direction
            # then, count the respective voxels
            labels_only_at_other_surfaces = list(set(labels_at_borders) - set(labels_at_both_surface_and_target))
            labels_only_at_other_surfaces = [label for label in labels_only_at_other_surfaces if label != 0]  # Exclude the zero (zero represents phases which are not of interest)
            n_unknown_voxels = np.count_nonzero(np.isin(labeled_ms, labels_only_at_other_surfaces))

            # Find the voxels, which are not connected to the borders (isolated clusters):
            isolated_labels = [label for label in range(n_labels) if label not in labels_at_borders]
            #Count the number of respective voxels
            boolean_mask_isolated = np.isin(labeled_ms, isolated_labels)
            n_isolated_voxels = np.count_nonzero(boolean_mask_isolated)

            # Find and count the voxels of phases which are not of interest
            n_voxels_not_of_interest = np.count_nonzero(labeled_ms==0)

            total_number_voxels = np.size(ms_phase_of_interest)

            fraction_connected_voxels = n_connected_voxels / total_number_voxels
            fraction_isolated_voxels = n_isolated_voxels / total_number_voxels
            fraction_unknown_voxels = n_unknown_voxels / total_number_voxels
            fraction_voxels_without_phase_of_interest = n_voxels_not_of_interest / total_number_voxels

            print(f'{fraction_connected_voxels}, {fraction_isolated_voxels}, {fraction_unknown_voxels}, {fraction_voxels_without_phase_of_interest}')

            percolation_info_dict = {'connected': fraction_connected_voxels, 
                                'isolated': fraction_isolated_voxels,
                                'unknown': fraction_unknown_voxels,
                                'not_phase_of_interest': fraction_voxels_without_phase_of_interest}

            is_percolating:np.bool_ = (fraction_connected_voxels > 0)

            return fraction_connected_voxels, is_percolating, percolation_info_dict
            
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
            
            assert isinstance(phase_of_interest, (int, list)), "type error: phase_of_interest must be an integer or a list of integers"

            if isinstance(phase_of_interest, int): # Ensure that phase_of_interest is a list
                phase_of_interest_list = [phase_of_interest]
            else:
                assert all(isinstance(item, int) for item in phase_of_interest), "type error: phase_of_interest must be an integer or a list of integers"
                phase_of_interest_list = phase_of_interest

            ms_phase_of_interest = np.isin(ms, phase_of_interest_list)
            
            fraction_connected_voxels, is_percolating, percolation_info_dict = calculate_percolation(ms_phase_of_interest)

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


    ms = np.zeros((200, 200, 200))
    ms[1,:, :] = 1

    ms = np.random.randint(low=0, high=2, size=(4, 4, 4)) 



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
