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
import networkx as nx
from itertools import combinations

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor
from numpy.typing import NDArray
from typing import Any, Union
import numpy as np
from skimage.morphology import skeletonize
from mcrpy.descriptors.Percolation import get_connected_phases_of_interest, get_labeled_ms
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

class Tortuosity(PhaseDescriptor):
    is_differentiable = False
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    @staticmethod
    def make_singlephase_descriptor(
        connectivity : Union[int,str] = 'sides', # implemented connectivities: only via sides, only via sides and edges, and via sides, edges and corners. 
        # for connectivity only via sides --> possible arguments: ['sides' (for 2D and 3D), 6 (for 3D), 4 (for 2D)], 
        # for connectivity only via sides and edges --> possible arguments: ['edges' (for 2D and 3D), 18 (for 3D), 4 (for 2D)] 
        # for connectivity via sides, edges and corners --> possible arguments ['corners' (for 2D and 3D), 26 (for 3D), 8 (for 2D)]  
        method : str = 'DSPSM', # implemented methods: 'DSPSM' and 'SSPSM'
        direction : int = 0, #0:x, 1:y, 2:z
        phase_of_interest : Union[int,list[int]] = [0], #for which phase number the tortuosity shall be calculated
        voxel_dimension:tuple[float] =(-1,1,1),
        **kwargs) -> callable:

        assert connectivity.lower() in ['sides', 'edges', 'corners', 6, 18, 28, 4, 8], "Valid inputs for connectivity are ['sides', 'edges, 'corners' 4, 6, 8, 18, 26]"
        assert method.upper() in ['DSPSM', 'SSPSM'], "method must be 'DSPSM' or 'SSPSM'."
        assert direction in [0,1,2], "Input error. Valid inputs for direction are 0, 1, 2 (x,y,z)"
        assert isinstance(phase_of_interest, (int, list)), "type error: phase_of_interest must be an integer or a list of integers"
        assert isinstance(voxel_dimension,tuple)
        assert all([val>0 for val in voxel_dimension]), "Only positive values for the voxel dimensions are permitted."

        def get_connectivity_directions(dimensionality: int):
            ''' connectivity_directions describes the connectivity for both directions (to and from a node pair) in the different directions, 
                that means for example for (1,0) but also for (-1,0). 
                This can lead to inefficencies in the code. 
                In the current code, the negative direction is implicitly included by making the calculation unidirective.
                for this, only one direction is needed (one_way_connectivity_directions)
            '''
            if ((connectivity in ['sides', 6] and dimensionality == 3) or 
                (connectivity in ['sides', 4] and dimensionality == 2) or
                (connectivity in ['edges', 4] and dimensionality == 2)):
                def increment_direction(direction_tuple, index, val):
                    new_direction = np.copy(direction_tuple)
                    new_direction[index] += val
                    return tuple(new_direction)
                connectivity_directions = np.array([increment_direction(np.zeros(dimensionality), index, val) 
                                                    for index in range(dimensionality) for val in [1, -1]])  # This creates both +1 and -1 increments
            elif connectivity in ['edges', 18] and dimensionality == 3:
                connectivity_directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                            (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)])
            elif connectivity in ['corners', 26] and dimensionality == 3:
                connectivity_directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                      (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                      (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                      (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
                      (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)])
            elif connectivity in ['corners', 8] and dimensionality == 2:
                connectivity_directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, 1), (1, -1), (-1, -1)])
            else:
                raise Exception(f'Connectivity argument (connectivity={connectivity}) and dimensionality (dimensionality={dimensionality}D) mismatch!  \nValid arguments for 3D are [sides, edges, corners, 6, 18, 28] \nand for 2D [sides, edges, corners, 4, 8].')

            # Create a mask where all values are non-negative and filter the array using the mask
            mask = np.all(connectivity_directions >= 0, axis=1)
            one_way_connectivity_directions = connectivity_directions[mask]
            
            return one_way_connectivity_directions


        #@tf.function
        def DSPSM(ms_phase_of_interest: NDArray[np.bool_]):
            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"

            # Quick exit if ms_phase_of_interest contains only False
            if not ms_phase_of_interest.any():
                return np.float64(0)

            # Build sparse adjacency between voxels of the phase using vectorized shifts
            shape = ms_phase_of_interest.shape
            ndim = len(shape)
            # determine connectivity offsets for this dimensionality
            connectivity_directions = get_connectivity_directions(ndim)
            idx_grid = np.arange(ms_phase_of_interest.size).reshape(shape) #each position gets an scalar idx

            # list of flat indices of nodes present
            node_coords = np.argwhere(ms_phase_of_interest)
            if node_coords.size == 0:
                return np.float64(0) # Quick exit if no nodes are present
            node_flat = np.ravel_multi_index(node_coords.T, shape) #putting all coordinates into an array.
            # mapping flat index -> compact index
            mapping = {int(flat): i for i, flat in enumerate(node_flat)}

            rows = []
            cols = []
            data = []

            # iterate connectivity offsets
            for off in connectivity_directions:
                off = tuple(int(x) for x in off)
                # compute slices for source and target to avoid wrap-around
                src_slices = []
                tgt_slices = []
                for dim_idx, o in enumerate(off):
                    if o > 0:
                        src_slices.append(slice(0, shape[dim_idx] - o))
                        tgt_slices.append(slice(o, shape[dim_idx]))
                    elif o < 0:
                        src_slices.append(slice(-o, shape[dim_idx]))
                        tgt_slices.append(slice(0, shape[dim_idx] + o))
                    else:
                        src_slices.append(slice(0, shape[dim_idx]))
                        tgt_slices.append(slice(0, shape[dim_idx]))
                src_slices = tuple(src_slices)
                tgt_slices = tuple(tgt_slices)

                src_mask = ms_phase_of_interest[src_slices]
                tgt_mask = ms_phase_of_interest[tgt_slices]
                valid_mask = src_mask & tgt_mask
                if not np.any(valid_mask):
                    continue

                src_idx = idx_grid[src_slices][valid_mask]
                tgt_idx = idx_grid[tgt_slices][valid_mask]

                weight = float(np.linalg.norm(np.array(off) * np.array(voxel_dimension[:ndim])))

                rows.extend(src_idx.tolist())
                cols.extend(tgt_idx.tolist())
                data.extend([weight] * len(src_idx))

                # also add reverse direction for undirected graph 
                # (this should only be used if only one-way connections between voxels are considered)
                rows.extend(tgt_idx.tolist())
                cols.extend(src_idx.tolist())
                data.extend([weight] * len(src_idx))

            if len(rows) == 0:
                return np.float64(0)

            # map flat rows/cols to compact indices
            rows_m = [mapping[int(r)] for r in rows]
            cols_m = [mapping[int(c)] for c in cols]


            # sparse matrix with data_val, rows, cols and shape args
            A = coo_matrix((np.array(data, dtype=np.float64), (np.array(rows_m), np.array(cols_m))), shape=(len(node_flat), len(node_flat))).tocsr()

            # identify source and target compact indices
            idx_max_position_in_direction = shape[direction] - 1
            source_mask_coords = node_coords[:, direction] == 0
            target_mask_coords = node_coords[:, direction] == idx_max_position_in_direction
            if not np.any(source_mask_coords) or not np.any(target_mask_coords):
                return np.float64(0)

            source_compact = [mapping[int(f)] for f in node_flat[source_mask_coords]]
            target_compact = [mapping[int(f)] for f in node_flat[target_mask_coords]]

            # run multi-source dijkstra (compute distances from all sources)
            try:
                dist_matrix = sp_dijkstra(csgraph=A, directed=False, indices=source_compact)
            except Exception:
                return np.float64(0)

            # dist_matrix shape (n_sources, n_nodes)
            # For each target, find the minimal distance from any source
            path_length_list = []
            len_first_voxel_in_direction = voxel_dimension[direction]
            for t_idx in target_compact:
                # extract column for target across sources
                if dist_matrix.ndim == 1:
                    dists = np.array([dist_matrix[t_idx]])
                else:
                    dists = dist_matrix[:, t_idx]
                finite = dists[np.isfinite(dists)]
                if finite.size > 0:
                    path_length_list.append(float(np.min(finite)) + len_first_voxel_in_direction)

            if not path_length_list:
                return np.float64(0)

            mean_path_length = np.mean(path_length_list)
            length_of_ms_in_specified_direction = shape[direction] * voxel_dimension[direction]
            return mean_path_length / length_of_ms_in_specified_direction
        
        def SSPSM(ms_phase_of_interest: NDArray[np.bool_]):
            '''
            Skeleton Shortest Path Searching Method
            '''     
            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"
            
            print(f'ms_phase_of_interest: {ms_phase_of_interest}')
            skeleton_ms = skeletonize(ms_phase_of_interest)
            print(f'skeleton_ms: {skeleton_ms}')
            return DSPSM(skeleton_ms) # calculate the tortuosity based on the skeleton of the ms 

        #@tf.function
        def model(ms: Union[tf.Tensor, NDArray[Any]]) -> tf.Tensor:
            
            
            
            if (len(ms.shape) > 3): # if called from mcrpy (would be a 4D tensor). If an microstructure is already 2 or 3D, don't change it.
                if ms.shape[0]==1: #if the microstructure is 2D
                    desired_shape =tuple(ms.shape[1:-1])
                else: #if the microstructure is 3D
                    desired_shape =tuple(ms.shape[0:-1])
                ms = tf.reshape(ms, desired_shape).numpy()
            
            if isinstance(phase_of_interest, int):
                phase_of_interest_list = [phase_of_interest]
            else:
                assert all(isinstance(item, int) for item in phase_of_interest), "type error: phase_of_interest must be an integer or a list of integers"
                phase_of_interest_list = phase_of_interest
            
            ms_phase_of_interest:np.ndarray[bool] = np.isin(ms, phase_of_interest_list) 
                # ms_phase_of_interest is an np.ndarray with bool values representing the 
                # microstructure ms where the searched for phase is represented as True, else False.
                # For further calculations, use ms_phase_of_interest:
            if not ms_phase_of_interest.any():
                return tf.cast(tf.constant(0), tf.float64)
            
            # the following is optional: reducing the number of voxels to check by only considering cluster which 
            # go from one side to another: 
            labeled_ms, _ = get_labeled_ms(ms_phase_of_interest, connectivity=connectivity)
            ms_connected_phase_of_interest, _ = get_connected_phases_of_interest(labeled_ms, direction)
            ms_phase_of_interest = ms_connected_phase_of_interest

            if method == 'DSPSM':  
                mean_tortuosity = DSPSM(ms_phase_of_interest)
            elif method == 'SSPSM':  
                mean_tortuosity = SSPSM(ms_phase_of_interest)

            return tf.cast(tf.constant(mean_tortuosity), tf.float64)#, tf.cast(tf.constant(mean_tortuosity), tf.float64)
        return model

    @staticmethod
    def make_multiphase_descriptor():
        return 0

def register() -> None:
    descriptor_factory.register("Tortuosity", Tortuosity)

       

if __name__=="__main__":

    import os
    folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    #minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
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
    minimal_example_ms = os.path.join(folder,'alloy_resized_s.npy')
    
    ms = np.load(minimal_example_ms)
    # print(f'ms: {ms}')
    # print(f'type: {type(ms[0])}')
    # print(f'ms type: {type(ms)}, size: {ms.size}')
    
    # print(np.unique(ms))


    # ms = np.zeros((6, 6))
    # ms[1,:] = 1
    # ms[1,5] = 0
    # ms[2,:,:] = 2
    print(f'ms: {ms}')
    print(f'shape: {(ms.shape)}')
    print(f'ms type: {type(ms)}, size: {ms.size}')

#     ms = np.zeros((3, 3))
#     ms[1,:] = 1
#     print(f'ms: {ms}')
#     print(f'shape: {(ms.shape)}')
#     print(f'ms type: {type(ms)}, size: {ms.size}')

    np.random.seed(10)
    ms = np.random.randint(2, size=(70,70,70))
    #print(f'ms: {ms}, size: {ms.size}')




    # Step 2: Open the pickle file
    # result_folder = '/home/sobczyk/Dokumente/MCRpy/mcrpy/results' 
    # pickle_filename = os.path.join(result_folder,'BlockingLayer_X_2D_32x32_characterization.pickle')
    # with open(pickle_filename, 'rb') as file:  # Replace 'filename.pkl' with your filepath
    #     # Step 3: Load the data
    #     data = pickle.load(file)
    # print(f"data: {data}")
    print(f'ms.shape: {ms.shape}')

##------------------------------------------------------------------
   
    tortuosity_descriptor = Tortuosity()
    singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()

    mean_tort = singlephase_descriptor(ms)
    print('\n -----------------------------')
    print(f'Mean tortuosity value: {mean_tort}')

##------------------------------------------------------------------
