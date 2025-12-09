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
from skimage.morphology import medial_axis
from mcrpy.descriptors.Percolation import get_connected_phases_of_interest, get_labeled_ms
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from mcrpy.descriptors.descriptor_utils.descriptor_utils import get_connectivity_directions, slice_ndarray, plot_slices
import logging

class Tortuosity(PhaseDescriptor):
    is_differentiable = False
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    @staticmethod
    def make_singlephase_descriptor(
        connectivity : Union[int,str] = 'corners', # implemented connectivities: only via sides, only via sides and edges, and via sides, edges and corners. 
        # for connectivity only via sides --> possible arguments: ['sides' (for 2D and 3D), 6 (for 3D), 4 (for 2D)], 
        # for connectivity only via sides and edges --> possible arguments: ['edges' (for 2D and 3D), 18 (for 3D), 4 (for 2D)] 
        # for connectivity via sides, edges and corners --> possible arguments ['corners' (for 2D and 3D), 26 (for 3D), 8 (for 2D)]  
        method : str = 'DSPSM', # implemented methods: 'DSPSM' and 'SSPSM'
        direction : int = 0, #0:x, 1:y, 2:z.
        phase_of_interest : Union[int,list[int]] = [2], #for which phase number the tortuosity shall be calculated
        voxel_dimension:tuple[float] =(1,1,1),
        **kwargs) -> callable:

        logging.info(f'input: connectivity: {connectivity}')
        logging.info(f'input: phase_of_interest: {phase_of_interest}')
        logging.info(f'input: method: {method}')
        logging.info(f'input: direction: {direction}')
        logging.info(f'input: voxel_dimension: {voxel_dimension}')

        assert connectivity.lower() in ['sides', 'edges', 'corners', 6, 18, 28, 4, 8], "Valid inputs for connectivity are ['sides', 'edges, 'corners' 4, 6, 8, 18, 26]"
        assert method.upper() in ['DSPSM', 'SSPSM'], "method must be 'DSPSM' or 'SSPSM'."
        assert direction in [0,1,2], "Input error. Valid inputs for direction are 0, 1, 2 (x,y,z)"
        assert isinstance(phase_of_interest, (int, list)), "type error: phase_of_interest must be an integer or a list of integers"
        assert isinstance(voxel_dimension,tuple)
        assert all([val>0 for val in voxel_dimension]), "Only positive values for the voxel dimensions are permitted."

        
        #@tf.function
        def DSPSM(ms_phase_of_interest: NDArray[np.bool_]):
            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"
            logging.info('Entering DSPSM function.')
            
            # Quick exit if ms_phase_of_interest contains only False
            if not ms_phase_of_interest.any():
                logging.debug("DSPSM: No phase voxels found, returning 0")
                return np.float64(0)

            # Build sparse adjacency between voxels of the phase using vectorized shifts
            shape = ms_phase_of_interest.shape
            ndim = len(shape)

            # determine connectivity (in which direction are voxels considered as connected?)
            connectivity_directions = get_connectivity_directions(ndim, connectivity=connectivity)
            
            #each position gets an scalar idx (=flat indices)
            idx_grid = np.arange(ms_phase_of_interest.size).reshape(shape) 

            # coordinates where ms_phase_of_interest is True
            node_coords = np.argwhere(ms_phase_of_interest)
            
            # Quick exit if no nodes are present
            if node_coords.size == 0:
                return np.float64(0) 
            
            # transforming node_coords into an 1D array using flat indices
            node_flat = np.ravel_multi_index(node_coords.T, shape) 
            
            # mapping flat index -> compact index 
            # The compact index is basically the graph nodal number, so from 0 to n-1 nodes
            mapping = {int(flat): i for i, flat in enumerate(node_flat)}

            # Initializing the lists needed to create the sparse matrix
            rows = []
            cols = []
            data = []

            # iterate over all connectivity offsets
            for off in connectivity_directions:
                off = tuple(int(x) for x in off) #making sure that all direction (= offset) tuples are of type int
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

                # creating a boolean mask for all start nodes (src) and end nodes (tgt) 
                # for the respective offset direction
                # only if src and tgt are True at the same position in their own respective mask, 
                # a connection is valid (-> valid mask)
                src_mask = ms_phase_of_interest[src_slices]
                tgt_mask = ms_phase_of_interest[tgt_slices]
                valid_mask = src_mask & tgt_mask
                if not np.any(valid_mask):
                    continue

                # finding the flat indices for src and tgt is now simple by applying the respective masks:
                # from all flat indices, take the src nodes for the respective offset direction. From these only
                # take the flat node numbers which have a valid connection in this direction. 
                # Analogous for tgt. 
                src_idx = idx_grid[src_slices][valid_mask]
                tgt_idx = idx_grid[tgt_slices][valid_mask]

                # calculate the weight for the graph nodes.
                # Here, the weigth is the distance between voxels in the current offset direction
                weight = float(np.linalg.norm(np.array(off) * np.array(voxel_dimension[:ndim])))

                # put all found flat node indices and weights into the respective lists.
                rows.extend(src_idx.tolist())
                cols.extend(tgt_idx.tolist())
                data.extend([weight] * len(src_idx))

                # also add reverse direction for undirected graph 
                # (this should only be used if only one-way connections between voxels are considered,
                #  but it seems that performancewise this doesn't make a big difference.)
                # rows.extend(tgt_idx.tolist())
                # cols.extend(src_idx.tolist())
                # data.extend([weight] * len(src_idx))

            if len(rows) == 0:
                return np.float64(0)

            # map flat rows/cols to compact indices
            # This is to reduce the size of the resulting matrix. 
            # This is a NxN matrix where N is not the number of True voxels.
            # Without the mapping, the size would be MxM with M beeing the total number of 
            # all voxels, which might be much larger than N.
            rows_m = [mapping[int(r)] for r in rows]
            cols_m = [mapping[int(c)] for c in cols]

            logging.info('Finished creating inputs for sparse adjacency matrix.')
            # sparse matrix with data_val, rows, cols and shape args
            A = coo_matrix((np.array(data, dtype=np.float64), (np.array(rows_m), np.array(cols_m))), shape=(len(node_flat), len(node_flat))).tocsr()
            logging.info('Finished creating sparse adjacency matrix.')
            
            # identify source and target compact indices
            idx_max_position_in_direction = shape[direction] - 1
            source_mask_coords = node_coords[:, direction] == 0
            target_mask_coords = node_coords[:, direction] == idx_max_position_in_direction
            if not np.any(source_mask_coords) or not np.any(target_mask_coords):
                return np.float64(0)

            source_compact = [mapping[int(f)] for f in node_flat[source_mask_coords]]
            target_compact = [mapping[int(f)] for f in node_flat[target_mask_coords]]

            # run multi-source dijkstra (compute distances from all sources)
            # [[dist_from_src_node1_to_node1, dist_from_src_node1_to_node2, dist_from_src_node1_to_node3, ...]]
            try:
                logging.debug(f"DSPSM: Running Dijkstra with {len(source_compact)} source(s) and {len(target_compact)} target(s)")
                dist_matrix = sp_dijkstra(csgraph=A, directed=False, indices=source_compact)
                logging.debug(f"DSPSM: Dijkstra completed")
            except Exception as e:
                logging.error(f"DSPSM: Dijkstra computation failed: {e}")
                print("DSPSM: Dijkstra computation failed. Returning zero.")
                return np.float64(0)
            logging.info('Finished multi-source dijkstra computation.')


            # dist_matrix shape (n_sources, n_nodes)
            # For each target, find the minimal distance from any source
            path_length_list = []
            for t_idx in target_compact:
                # extract column for target across sources
                if dist_matrix.ndim == 1: # (if there is only one source node)
                    dists = np.array([dist_matrix[t_idx]])
                else: # (if there are more than one source node)
                    dists = dist_matrix[:, t_idx] # get all distances from all source nodes to all targets
                finite = dists[np.isfinite(dists)]
                if finite.size > 0:
                    path_length_list.append(float(np.min(finite))) #only add the shortest paths from source to target

            if not path_length_list:
                logging.warning("DSPSM: No path lengths computed")
                return np.float64(0)

            mean_path_length = np.mean(path_length_list)
            length_of_ms_in_specified_direction = (shape[direction]-1) * voxel_dimension[direction]
            tortuosity = mean_path_length / length_of_ms_in_specified_direction
            logging.info(f"DSPSM: Tortuosity = {tortuosity:.4f} (mean path length: {mean_path_length:.2f})")
            return tortuosity
        
        def SSPSM(ms_phase_of_interest: NDArray[np.bool_]):
            '''
            Skeleton Shortest Path Searching Method
            '''     
            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"
            


            total_number_slices = ms_phase_of_interest.shape[direction]
            skeleton_slice_list = []
            for slice_number in range(total_number_slices):
                # Get the slice from the original ndarray
                ms_slice = slice_ndarray(data=ms_phase_of_interest,axis=direction,index=slice_number)

                # Apply medial_axis to the sliced data
                skeleton_slice = medial_axis(ms_slice)

                # Append the resulting skeleton slice to the list
                skeleton_slice_list.append(skeleton_slice)


            # Stack the list of skeleton slices back into an ndarray, maintaining the original shape
            skeleton_ms = np.stack(skeleton_slice_list, axis=direction)


            plotting=True
            if plotting:
                plot_slices(data=ms_phase_of_interest,direction=direction,block=False)
                plot_slices(data=skeleton_ms,direction=direction)


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

    # Configure logging for standalone execution
    import os
    log_dir = './tortuosity_logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'tortuosity.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any previous basicConfig
    )
    logging.info("="*60)
    logging.info("TORTUOSITY DESCRIPTOR - STANDALONE EXECUTION")
    logging.info("="*60)
    # Suppress Matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    import os
    folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    #minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    #minimal_example_ms = os.path.join(folder,'BlockingLayer_X_32x32x32.npy')

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


    # minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    minimal_example_ms = os.path.join(folder,'Holzer2020_Segmented_Fine_Pristine_Zoom0.33_size600.npy')

    # minimal_example_ms = os.path.join(folder,'alloy_resized_s.npy')
    
    ms = np.load(minimal_example_ms)

    #ms = ms[:,:,-2]

    #ms = np.fliplr(ms)

    # ms_len:int = 200
    # ms = np.ones((ms_len,ms_len,ms_len))
    # ms[:,round(ms_len/4):round(3/4*ms_len),round(ms_len/4):round(3/4*ms_len)] = 0

    # l = 100
    # ms = np.zeros((l,l,l))


    #print(ms)

    # print(f'ms: {ms}')
    # print(f'type: {type(ms[0])}')
    # print(f'ms type: {type(ms)}, size: {ms.size}')
    
    # print(np.unique(ms))


    # ms = np.ones((3,3,3))
    # ms[0,0,0] = 0
    # ms[1,1,1] = 0
    # ms[2,2,2] = 0
    # ms = ms.astype(int)

    # ms = np.zeros((5,5,1))
    # ms[1,2,0] = 1
    # ms[2,1,0] = 1
    # ms[2,2,0] = 1
    # ms[2,3,0] = 1
    # ms[3,2,0] = 1
    # ms = ms.astype(int)

    # ms = np.zeros((3, 3))
    # ms[0,0] = 1
    # ms[0,1] = 1
    # ms[1,1] = 1
    # ms[1,2] = 1

    # ms = np.zeros((3, 3))
    # ms[0,1] = 1
    # ms[1,1] = 1
    # ms[2,1] = 1
    # ms[2,2] = 1

    # print(f'ms:\n {ms}')
    # print(f'shape: {(ms.shape)}')
    # print(f'ms type: {type(ms)}, size: {ms.size}')

#     ms = np.zeros((3, 3))
#     ms[1,:] = 1
#     print(f'ms: {ms}')
#     print(f'shape: {(ms.shape)}')
#     print(f'ms type: {type(ms)}, size: {ms.size}')

    # np.random.seed(10)
    # ms = np.random.randin=(70,70,70))
    #print(f'ms: {ms}, size: {ms.size}')




    # Step 2: Open the pickle file
    # result_folder = '/home/sobczyk/Dokumente/MCRpy/mcrpy/results' 
    # pickle_filename = os.path.join(result_folder,'BlockingLayer_X_2D_32x32_characterization.pickle')
    # with open(pickle_filename, 'rb') as file:  # Replace 'filename.pkl' with your filepath
    #     # Step 3: Load the data
    #     data = pickle.load(file)
    # print(f"data: {data}")
    #print(f'ms.shape: {ms.shape}')

    
    plotting=True
    if plotting:
        ms_to_plot = ms
        if len(ms_to_plot.shape)==2 or (len(ms_to_plot.shape)==3 and ms_to_plot.shape[-1] == 1):
            import matplotlib.pyplot as plt
            plt.matshow(ms_to_plot)
            plt.show()

##------------------------------------------------------------------
   
    tortuosity_descriptor = Tortuosity()
    singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()

    logging.info(f'Starting tortuosity calculation with microstructure of shape: {ms.shape}')
    mean_tort = singlephase_descriptor(ms)
    print('\n -----------------------------')
    print(f'Mean tortuosity value: {mean_tort}')
    logging.info(f"Standalone execution completed successfully. Result for Mean Tortuosity: {mean_tort}")
    logging.info("="*60)

##------------------------------------------------------------------
