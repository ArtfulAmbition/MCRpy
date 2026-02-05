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
disable_tf_warnings = True
if disable_tf_warnings:
    # Configure TensorFlow/C++ logging and oneDNN before any TensorFlow import.
    # - TF_CPP_MIN_LOG_LEVEL: 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
    #   Setting to '3' hides INFO and WARNING, keeping only ERROR messages.
    # - TF_ENABLE_ONEDNN_OPTS=0 disables oneDNN custom-op informational messages.
    import os
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf
from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor
from numpy.typing import NDArray
from typing import Any, Union
import numpy as np
from skimage.morphology import medial_axis, skeletonize
from mcrpy.descriptors.Percolation import get_connected_phases_of_interest, get_labeled_ms
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra
from mcrpy.descriptors.descriptor_utils.descriptor_utils import get_connectivity_directions, slice_ndarray, plot_slices
import logging
from mcrpy.descriptors.PhaseDescriptor3D import PhaseDescriptor3D
from mcrpy.view import view
import SimpleITK as sitk

tf.config.run_functions_eagerly(True)

class Pathfinder():
    def __init__(self,
                 ms_phase_of_interest: NDArray[np.bool_],
                 connectivity : Union[int,str] = 'sides',
                 direction_list : Union[int,list[int]] = 0,
                 direction_mode:str = 'positive',
                 voxel_dimension:tuple[float] =(1,1,1)):
        self.ms_phase_of_interest = ms_phase_of_interest
        self.shape = ms_phase_of_interest.shape
        self.dimensionality = len(self.shape)
        self.connectivity = connectivity
        self.direction_list = direction_list
        self.direction_mode = direction_mode
        self.voxel_dimension = voxel_dimension,
        self.tortuosity_list = []
        self.adjacency_matrix = None
        self.distance_matrix = None
        self.compact_source_list = []
        self.compact_target_list = []
        self.unique_target_indices = []
        self.unique_sources_indices = []
        self.node_flat = np.array([])
        self.path_length_list = []
        self.compact_direction_list = [] # list that holds the direction corresponding to each source/target pair

        # basic checks
        assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"
        assert isinstance(self.direction_list, list), "Error: direction must be a list of ints." 
        assert all(isinstance(dir, int) and 0 <= dir < self.dimensionality 
                   for dir in self.direction_list), f"All elements of direction_list must be positve integers smaller than the dimensionality {self.dimensionality}.)"
        # fix voxel_dimension accidentally being a 1-tuple if trailing comma existed
        if isinstance(voxel_dimension, tuple) and len(voxel_dimension) == 1 and isinstance(voxel_dimension[0], tuple):
            self.voxel_dimension = voxel_dimension[0]
        else:
            self.voxel_dimension = voxel_dimension       

        self.compute()

    def abort_calculation(self,mode='this_direction'):
        if mode == 'this_direction':
            """No Paths can be found in this direction. Stop calculation and set tortuosity to zero."""
            self.tortuosity_list.append(np.float64(0))
        if mode == 'global':
            """No Paths can be found globally (in all directions). 
            Stop calculation and set tortuosity in all directions to zero."""
            number_of_direction = len(self.direction_list)
            if self.direction_mode == 'both':
                number_of_tortuosities = number_of_direction * 2
            else:
                number_of_tortuosities = number_of_direction
            self.tortuosity_list = [np.float64(0) for x in range(number_of_tortuosities)]

    def construct_adjacency_matrix(self):
        """Build sparse adjacency matrix using the configured connectivity."""
        ms = self.ms_phase_of_interest
        # each position gets a scalar idx (=flat indices)
        idx_grid = np.arange(ms.size).reshape(self.shape)

        # coordinates where ms_phase_of_interest is True
        self.node_coords = np.argwhere(ms)

        # Quick exit if no nodes are present
        if self.node_coords.size == 0:
            self.abort_calculation(mode='global')
            return False

        # transforming node_coords into an 1D array using flat indices
        self.node_flat = np.ravel_multi_index(self.node_coords.T, self.shape)

        # mapping flat index -> compact index
        self.mapping = {int(flat): i for i, flat in enumerate(self.node_flat)}

        rows = []
        cols = []
        data = []

        connectivity_directions = get_connectivity_directions(self.dimensionality, connectivity=self.connectivity)

        for off in connectivity_directions:
            off = tuple(int(x) for x in off)
            src_slices = []
            tgt_slices = []
            for dim_idx, o in enumerate(off):
                if o > 0:
                    src_slices.append(slice(0, self.shape[dim_idx] - o))
                    tgt_slices.append(slice(o, self.shape[dim_idx]))
                elif o < 0:
                    src_slices.append(slice(-o, self.shape[dim_idx]))
                    tgt_slices.append(slice(0, self.shape[dim_idx] + o))
                else:
                    src_slices.append(slice(0, self.shape[dim_idx]))
                    tgt_slices.append(slice(0, self.shape[dim_idx]))
            src_slices = tuple(src_slices)
            tgt_slices = tuple(tgt_slices)

            src_mask = ms[src_slices]
            tgt_mask = ms[tgt_slices]
            valid_mask = src_mask & tgt_mask
            if not np.any(valid_mask):
                continue

            src_idx = idx_grid[src_slices][valid_mask]
            tgt_idx = idx_grid[tgt_slices][valid_mask]

            weight = float(np.linalg.norm(np.array(off) * np.array(self.voxel_dimension[:self.dimensionality])))

            rows.extend(src_idx.tolist())
            cols.extend(tgt_idx.tolist())
            data.extend([weight] * len(src_idx))

        if len(rows) == 0:
            self.abort_calculation(mode='global')
            return False

        rows_m = [self.mapping[int(r)] for r in rows]
        cols_m = [self.mapping[int(c)] for c in cols]

        logging.info('Finished creating inputs for sparse adjacency matrix.')
        self.adjacency_matrix = coo_matrix((np.array(data, dtype=np.float64), (np.array(rows_m), np.array(cols_m))),
                                          shape=(len(self.node_flat), len(self.node_flat))).tocsr()
        logging.info('Finished creating sparse adjacency matrix.')
        return True

    def find_compact_border_node_coords(self, direction: int, border_type: str):
        ''' return a list of the compact node coordinates of the voxels at the searched for border. 
        The border is the border in the specified direction (so in x,y or z direction). In each direction 
        there are two borders (at for example min x-value or max x-value). To specify which border is searched for, 
        the border type must be specified.'''
        if border_type.lower() == 'min':
            idx_in_direction = 0
        elif border_type.lower() == 'max':
            idx_in_direction = self.shape[direction] - 1
        else:
            raise ValueError("extremum needs to be 'min' or 'max'.")

        mask_coordinates = self.node_coords[:, direction] == idx_in_direction
        border_nodes_compact = []
        if self.node_flat.size != 0:
            border_nodes_compact = [self.mapping[int(f)] for f in self.node_flat[mask_coordinates]]
        return border_nodes_compact

    def find_compact_source_and_target_nodes(self):

        for dir in self.direction_list:
            if self.direction_mode == 'positive':
                # this means, that the tortuosity is calculated in positve direction of the specified direction
                # for example in positive x-direction
                self.compact_source_list.append(self.find_compact_border_node_coords(direction=dir, border_type='min'))
                self.compact_target_list.append(self.find_compact_border_node_coords(direction=dir, border_type='max'))
                self.compact_direction_list.append(dir)
            elif self.direction_mode == 'negative':
                # this means, that the tortuosity is calculated in the opposite direction of the specified direction
                # for example in negative x-direction
                self.compact_source_list.append(self.find_compact_border_node_coords(direction=dir, border_type='max'))
                self.compact_target_list.append(self.find_compact_border_node_coords(direction=dir, border_type='min'))
                self.compact_direction_list.append(-dir)
            elif self.direction_mode == 'both':
                # this means, that the tortuosity is calculated in both the positive and opposite direction of the specified direction
                # for example in positive and negative x-direction
                self.compact_source_list.append(self.find_compact_border_node_coords(direction=dir, border_type='min'))
                self.compact_target_list.append(self.find_compact_border_node_coords(direction=dir, border_type='max'))
                self.compact_direction_list.append(dir)

                self.compact_source_list.append(self.find_compact_border_node_coords(direction=dir, border_type='max'))
                self.compact_target_list.append(self.find_compact_border_node_coords(direction=dir, border_type='min'))
                self.compact_direction_list.append(-dir)

        if not self.compact_source_list or not self.compact_target_list:
            self.abort_calculation(mode='global')
            return False
        return True

    def calculate_distance_matrix(self):

        all_sources = np.concatenate(self.compact_source_list) if len(self.compact_source_list) > 0 else np.array([], dtype=int)
        all_targets = np.concatenate(self.compact_target_list) if len(self.compact_target_list) > 0 else np.array([], dtype=int)

        if all_sources.size == 0 or all_targets.size == 0:
            logging.warning('No sources or targets found for Dijkstra.')
            self.abort_calculation(mode='global')
            return False

        try:
            logging.debug(f"DSPSM: Running Dijkstra with a total of {len(all_sources)} unique source(s) and {len(all_targets)} unique target(s)")
            self.distance_matrix = sp_dijkstra(csgraph=self.adjacency_matrix, directed=False, indices=all_sources)
            logging.debug(f"DSPSM: Dijkstra completed")
        except Exception as e:
            logging.error(f"DSPSM: Dijkstra computation failed: {e}", exc_info=True)
            # Abort calculation and set tortuosities to zero when Dijkstra fails
            self.abort_calculation(mode='global')
            print(f'Error: Dijkstra computation failed: {e}')
            return False
        logging.info('Finished multi-source dijkstra computation.')
        return True

    def get_shortest_paths_from_distance_matrix(self):
        """Extract shortest path lengths from the distance matrix and compute tortuosity.
        For each source/target group we compute the minimum distance from any group source to every group target,
        then compute the group's mean path length and normalize by the physical length in that direction.
        The final tortuosity is the mean across group tortuosities.
        """

        dist_matrix = self.distance_matrix
        if dist_matrix is None:
            logging.error('DSPSM: Distance matrix is None. Dijkstra did not complete successfully.')
            # Ensure we abort calculation and set tortuosities to zero
            self.abort_calculation(mode='global')
            return False
        # normalize dist_matrix to 2D where rows correspond to sources
        if dist_matrix.ndim == 1:
            dist_matrix = dist_matrix[np.newaxis, :]

        group_tortuosities = [] #
        number_of_source_target_groups = len(self.compact_source_list)

        for ind in range(number_of_source_target_groups):
            group_sources = np.array(self.compact_source_list[ind], dtype=int)
            group_targets = np.array(self.compact_target_list[ind], dtype=int)
            group_dir = self.compact_direction_list[ind]

            # find rows in dist_matrix that correspond to the current group sources
            all_srcs = np.concatenate(self.compact_source_list)
            source_rows = np.where(np.isin(all_srcs, group_sources))[0]

            path_length_list = []
            for t_idx in np.unique(group_targets):
                # extract distances from all source rows to target node (column t_idx)
                dists = dist_matrix[source_rows, t_idx]
                finite = dists[np.isfinite(dists)]
                if finite.size > 0:
                    path_length_list.append(float(np.min(finite)))          

            if path_length_list:
                mean_path_length = float(np.mean(path_length_list))
            else:
                mean_path_length = 0
            length_of_ms = (self.shape[group_dir] - 1) * float(self.voxel_dimension[group_dir])
           
            group_tortuosity = mean_path_length / length_of_ms


            logging.info(f"DSPSM: group {ind} (dir {group_dir}) tortuosity = {group_tortuosity:.4f} (mean path length: {mean_path_length:.2f})")
            group_tortuosities.append(group_tortuosity)

        self.path_length_list = path_length_list
        self.tortuosity_list = group_tortuosities
        return True

    def compute(self):
        """Run full pipeline: adjacency -> sources/targets -> dijkstra -> extract tortuosity."""
        if not self.construct_adjacency_matrix():
            return False
        if not self.find_compact_source_and_target_nodes():
            return False
        if not self.calculate_distance_matrix():
            return False
        if not self.get_shortest_paths_from_distance_matrix():
            return False
        return True

class Tortuosity(PhaseDescriptor3D):
    is_differentiable = False
    tf.experimental.numpy.experimental_enable_numpy_behavior()

    @staticmethod
    def make_singlephase_descriptor(
        
        connectivity : Union[int,str] = 'corners', # implemented connectivities: only via sides, only via sides and edges, and via sides, edges and corners. 
        # for connectivity only via sides --> possible arguments: ['sides' (for 2D and 3D), 6 (for 3D), 4 (for 2D)], 
        # for connectivity only via sides and edges --> possible arguments: ['edges' (for 2D and 3D), 18 (for 3D), 4 (for 2D)] 
        # for connectivity via sides, edges and corners --> possible arguments ['corners' (for 2D and 3D), 26 (for 3D), 8 (for 2D)]  
        method : str = 'SSPSM', # implemented methods: 'DSPSM' and 'SSPSM'
        directions_list : Union[list[int],None] = None, #0:x, 1:y, 2:z if None, calculate in all available direction
        direction_mode:str = 'positive', # specifies in which direction the tortuosity is calculated. +#
                                         # 'positive': in direction of the direction coordinate
                                         # 'negative': in oppositve direction of the direction coordinate
                                         # 'both': calculates the tortuosity based on paths in coordinate direction and opposite
        phase_of_interest : Union[int,list[int]] = [1], #for which phase number the tortuosity shall be calculated
        voxel_dimension:tuple[float] =(1,1,1),
        **kwargs) -> callable:

        logging.info(f'input: connectivity: {connectivity}')
        logging.info(f'input: phase_of_interest: {phase_of_interest}')
        logging.info(f'input: method: {method}')
        logging.info(f'input: directions_list: {directions_list}')
        logging.info(f'input: voxel_dimension: {voxel_dimension}')

        assert connectivity.lower() in ['sides', 'edges', 'corners', 6, 18, 28, 4, 8], "Valid inputs for connectivity are ['sides', 'edges, 'corners' 4, 6, 8, 18, 26]"
        assert method.upper() in ['DSPSM', 'SSPSM'], "method must be 'DSPSM' or 'SSPSM'."
        assert isinstance(directions_list, Union[list,None])

        assert isinstance(phase_of_interest, (int, list)), "type error: phase_of_interest must be an integer or a list of integers"
        assert isinstance(voxel_dimension,tuple)
        assert all([val>0 for val in voxel_dimension]), "Only positive values for the voxel dimensions are permitted."
        assert direction_mode in ['positive', 'negative', 'both'], "Valid inputs for direction_mode are 'positive', 'negative' or 'both'."
        
        #@tf.function
        def DSPSM(ms_phase_of_interest: NDArray[np.bool_], directions: list):
            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"
            logging.info('Entering DSPSM function.')
            
            pathfinder = Pathfinder(ms_phase_of_interest=ms_phase_of_interest,
                                    direction_list=directions, 
                                    direction_mode=direction_mode) 

            return pathfinder.tortuosity_list
        
        def SSPSM(ms_phase_of_interest: NDArray[np.bool_], directions: list, method='medial_axis', do_paraview_plot:bool=True ):
            '''
            Skeleton Shortest Path Searching Method
            '''     
            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"

            if method == 'medial_axis':

                # using SimpleITK:
                sitk_image = sitk.GetImageFromArray(ms_phase_of_interest.astype(np.uint8))
                distance_transform = sitk.SignedMaurerDistanceMap(sitk_image, 
                                                                  insideIsPositive=True, 
                                                                  squaredDistance=False, 
                                                                  useImageSpacing=False)
                medial_axis = sitk.BinaryThinning(distance_transform > 0)
                medial_axis_array = sitk.GetArrayFromImage(medial_axis)
                skeleton_ms = medial_axis_array.astype(bool)
                #raise NotImplementedError('Error: Method "medial_axis" not yet implemented in SSPSM.')
                
                # dimensionality = len(ms_phase_of_interest.shape)
                # if dimensionality == 3 and not any(dim == 1 for dim in ms_phase_of_interest.shape):
                #     total_number_slices = ms_phase_of_interest.shape[direction]
                #     skeleton_slice_list = []
                #     for slice_number in range(total_number_slices):
                #         # Get the slice from the original ndarray
                #         ms_slice = slice_ndarray(data=ms_phase_of_interest,axis=direction,index=slice_number)

                #         # Apply medial_axis to the sliced data
                #         skeleton_slice = medial_axis(ms_slice)

                #         # Append the resulting skeleton slice to the list
                #         skeleton_slice_list.append(skeleton_slice)


                #     # Stack the list of skeleton slices back into an ndarray, maintaining the original shape
                #     skeleton_ms = np.stack(skeleton_slice_list, axis=direction)

                #     if plotting:
                #         plot_slices(data=ms_phase_of_interest,direction=direction,block=False)
                #         plot_slices(data=skeleton_ms,direction=direction)
                # else:
                #     skeleton_ms = medial_axis(ms_phase_of_interest)

                #     if plotting:
                #         plt.matshow(skeleton_ms)
                #         plt.show()
            
            elif method == 'skeletonize':
                skeleton_ms = skeletonize(ms_phase_of_interest)
            else:
                raise NotImplementedError('Error: Method {method} not implemented in SSPSM.')

            if do_paraview_plot:
                from mcrpy.src.Microstructure import Microstructure
                MS_phase_of_interest = Microstructure(ms_phase_of_interest.astype(int))
                skeleton_MS = Microstructure(skeleton_ms.astype(int))
                view(MS_phase_of_interest,save_as='PhaseofInterest')
                view(skeleton_MS,save_as='SkeletonPhaseofInterest')

            return DSPSM(skeleton_ms, directions) # calculate the tortuosity based on the skeleton of the ms 

        #@tf.function
        def model(ms: Union[tf.Tensor, NDArray[Any]]) -> tf.Tensor:
            if (len(ms.shape) > 3): # if called from mcrpy (would be a 4D tensor). If an microstructure is already 2 or 3D, don't change it.
                # make sure ms is a numpy array
                ms = np.asarray(ms)
                # if the last axis encodes multi-phase or orientation (axis > 1), decode it to a scalar phase id per voxel
                if ms.shape[-1] > 1:
                    # collapse one-hot/multiphase encoding -> integer phase ids
                    ms = np.argmax(ms, axis=-1)
                    # if there's a leading batch dimension of 1, remove it
                    if ms.shape[0] == 1:
                        ms = ms[0]
                else:
                    # usual path: remove batch and channel dimensions
                    if ms.shape[0] == 1:  # if called from mcrpy with batch dim
                        desired_shape = tuple(ms.shape[1:-1])
                    else:  # already a plain 3D array with channels at the end
                        desired_shape = tuple(ms.shape[0:-1])
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
            
            # the following is optional: reducing the number of voxels to check by only considering cluster which 
            # go from one side to another: 
            labeled_ms, _ = get_labeled_ms(ms_phase_of_interest, connectivity=connectivity)
            ms_connected_phase_of_interest = np.zeros_like(labeled_ms, dtype=bool)
           
            dimensionality = len(ms.shape)      

            if directions_list is None:
                directions = [dir for dir in range(dimensionality)] 
            else:
                directions = directions_list
            assert all(isinstance(dir, int) and 0 <= dir <= dimensionality-1
                   for dir in directions), f"All elements of direction_list must be positve integers <= {dimensionality-1} (0=x,1=y,2=z).)"
           
            for direction in directions:
                ms_connected_phase_of_interest_dir, _ = get_connected_phases_of_interest(labeled_ms, direction)
                ms_connected_phase_of_interest = ms_connected_phase_of_interest | ms_connected_phase_of_interest_dir

            ms_phase_of_interest = ms_connected_phase_of_interest

            if method == 'DSPSM':  
                mean_tortuosity = DSPSM(ms_phase_of_interest, directions=directions)
            elif method == 'SSPSM':  
                mean_tortuosity = SSPSM(ms_phase_of_interest, directions=directions)

            mean_tortuosity = np.array(mean_tortuosity)
            return tf.cast(tf.constant(mean_tortuosity), tf.float64)#, tf.cast(tf.constant(mean_tortuosity), tf.float64)
        return model

    # @staticmethod
    # def make_multiphase_descriptor():
    #     return 0

def register() -> None:
    descriptor_factory.register("Tortuosity3D", Tortuosity)

       

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
    # folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    #minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    #minimal_example_ms = os.path.join(folder,'BlockingLayer_X_32x32x32.npy')

    #minimal_example_ms = os.path.join(folder,'composite_resized_s.npy')

    # minimal_example_ms = os.path.join(folder,'BlockingLayer_X_2D_20x20.npy')

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
    #minimal_example_ms = os.path.join(folder,'Holzer2020_Segmented_Fine_Pristine_Zoom0.33_size600.npy')
    # minimal_example_ms = os.path.join(folder,'alloy_resized_s.npy')
    # minimal_example_ms = os.path.join(folder,'BlockingLayer_X_32x32x32.npy')

    # ms = np.load(minimal_example_ms)

    # ms = ms[:,:,-2:-1]

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

    np.random.seed(11)  
    ms = np.random.randint(low=0,high=2,size=(3,3,3))
    #ms=np.ones(shape=(2,2)).astype(bool)
    # ms = np.ones(shape=(2,2,2))
    # ms[0,0] = 1
    # ms[0,1] = 0
    # ms[1,1] = 0
    # ms[1,0] = 1
    # ms = ms.astype(bool)
    # # ms[0,0,0] = 0
    # # ms[1,1,1] = 0
    # # ms[2,2,2] = 0
    # #ms = ms.astype(int)
    print(f'ms:\n {ms}')

    # pt = Pathfinder(ms_phase_of_interest=ms,
    #                 direction_list=[0,1,2], direction_mode='both')
    # pt.construct_adjacency_matrix()
    # pt.find_compact_source_and_target_nodes()
    # pt.calculate_distance_matrix()
    # pt.get_shortest_paths_from_distance_matrix()
    # pt.compute()

    # print(pt.get_shortest_paths_from_distance_matrix())
    # print(f'tort: {pt.tortuosity_list}')
    # print(f'compact direction list: {pt.compact_direction_list}')
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

    # 
    # ms = np.random.randin=(70,70,70))
    # print(f'ms: {ms}, size: {ms.size}')




    
    plotting=False
    if plotting:
        ms_to_plot = ms
        if len(ms_to_plot.shape)==2 or (len(ms_to_plot.shape)==3 and ms_to_plot.shape[-1] == 1):
            import matplotlib.pyplot as plt
            plt.matshow(ms_to_plot)
            plt.show()

##------------------------------------------------------------------
   
    tortuosity_descriptor = Tortuosity()
    singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor(phase_of_interest=[0], 
                                                                               direction_mode='positive', 
                                                                               connectivity='corners')

    logging.info(f'Starting tortuosity calculation with microstructure of shape: {ms.shape}')
    mean_tort = singlephase_descriptor(ms)
    print('\n -----------------------------')
    print(f'tortuosity: {mean_tort}')
    logging.info(f"Standalone execution completed successfully. Result for Mean Tortuosity: {mean_tort}")
    logging.info("="*60)

##------------------------------------------------------------------

    # import pickle
    # # Step 2: Open the pickle file
    # result_folder = '/home/sobczyk/Dokumente/MCRpy/results' 
    # pickle_filename = os.path.join(result_folder,'BlockingLayer_X_32x32x32_characterization.pickle')
    # with open(pickle_filename, 'rb') as file:  # Replace 'filename.pkl' with your filepath
    #     # Step 3: Load the data
    #     data = pickle.load(file)
    # print(f"data: {data}")
    # print(f'ms.shape: {ms.shape}')

    # print(f'ms: {ms}, size: {ms.size}')