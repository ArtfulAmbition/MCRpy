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
import logging

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor
from numpy.typing import NDArray
from typing import Any, Union
import numpy as np
from skimage.morphology import skeletonize

class Tortuosity(PhaseDescriptor):
    is_differentiable = False

    @staticmethod
    def make_singlephase_descriptor(
        connectivity : int = 6,
        method : str = 'SSPSM', # implemented methods: 'DSPSM' and 'SSPSM'
        directions : Union[int,list[int]] = 2, #0:x, 1:y, 2:z
        phase_of_interest : Union[int,list[int]] = 0, #for which phase number the tortuosity shall be calculated
        voxel_dimension:tuple[float] =(1,1,1),
        **kwargs) -> callable:

        #@tf.function
        def ms_to_graph(ms_phase_of_interest: NDArray[np.bool_]) -> nx.Graph:
            # connectivity: number of adjacent neighbors for which a connectivity should be allowed. Implemented possibilites are 6, 18 and 24
            # voxel_dimension: the dimension (lx,ly,lz) of each considered voxel. Is needed to calculate the distance between neighbors
            _voxel_dimension = voxel_dimension

            #--------- create nodes: ------------------
            coordinates = tf.where(ms_phase_of_interest) # getting the coordinates of voxels with phase id equal to phase_to_investigate
            coordinates_np = coordinates.numpy()

            nodes = map(tuple,coordinates_np) # creating a vector of node tuples for insertion to nx.Graph()
            
            graph = nx.Graph()
            graph.add_nodes_from(nodes) 
            #print(f'nodes: {graph.nodes}')
            
            
            dimensionality = len(ms_phase_of_interest.shape)
            _voxel_dimension = _voxel_dimension[:dimensionality] # omit voxel dimensions, which don't fit to the given microstructure dimensionality

            def increment_direction(direction_tuple, index, val):
                new_direction = np.copy(direction_tuple)
                new_direction[index] += val
                return tuple(new_direction)
            directions=np.zeros(dimensionality)
            
            #--------- connect nodes for connectivity=6 implemented for 2D and 3D: -----------------
            if connectivity==6:
                directions = np.array([increment_direction(directions, index, val) for index in range(dimensionality) for val in [1, -1]])  # This creates both +1 and -1 increments

            #--------- connect nodes 3D: -----------------
            # if connectivity == 6:
            #     directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])

            elif connectivity == 18:
                directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                            (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)])
                
            elif connectivity == 26:
                directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                      (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                      (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                      (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
                      (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)])  
            else:
                raise Exception("wrong input for connectivity value. Valid inputs are 6, 18 or 26.")

            graph_nodes = np.array(list(graph.nodes))  # Convert to np.array for vectorized operations
            edges_to_add = []  # List to store edges before adding
            
            i = 0
            for node in graph.nodes:
                #x, y, z = node
                neighbors = node + directions #find all neighbors by coordinates
                #valid_neighbors = neighbors[np.isin(neighbors, graph_nodes).all(axis=1)] #extract the  neighbors which are of the same phase
                # Use broadcasting to create a mask of valid neighbors
                
                # Use broadcasting to check for matches
                valid_neighbors_mask = np.any(np.all(neighbors[:, np.newaxis] == graph_nodes, axis=2), axis=1)
                # Select valid neighbors based on the mask
                valid_neighbors = neighbors[valid_neighbors_mask]

                # print(f'node: {node}, valid_neighbors: {valid_neighbors}')
                # print(f'node: {node}, neighbors: {neighbors}')
                # print(f'node: {node}, graph_nodes: {graph_nodes}')

                # calculating the distance to the neighbor:
                normed_distance_vectors = valid_neighbors - node # unit vector differences between the current node to its valid neighbors 
                weighted_distance_vectors = normed_distance_vectors * [*_voxel_dimension] # weighted vector differences between the current node to its valid neighbors 
                distances = np.linalg.norm(weighted_distance_vectors, axis=1)
                                
                edges_to_add.extend([(node, tuple(valid_neighbor), {'weight': distance}) for valid_neighbor, distance in zip(valid_neighbors, distances)])

                # if i==0:
                #     print(f'valid_neighbors= {valid_neighbors}')
                #     print(f'node= {node}')
                #     print(f'distance= {distances}')
                #     print(f'edges_to_add= {edges_to_add}')
                #     i+=1

            graph.add_edges_from(edges_to_add)
            #print(f'edges: {graph.edges}')
            return graph


        # @tf.function
        def DSPSM(ms_phase_of_interest: NDArray[np.bool_]):

            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"

            print(f'ms_phase_of_interest:\n {ms_phase_of_interest}')

            if type(directions)==int:
                direction = directions
            #create the graph for phase_of_interest
            graph:nx.Graph = ms_to_graph(ms_phase_of_interest)
            node_array:np.ndarray = np.array(graph.nodes)
            #print(f'node_array: {node_array}')

            #identify the source nodes (from where the paths through the microstructure shall start, at minimum of coordinate in specified direction)
            # and the target nodes (to which the shortest path is searched for)
            idx_max_position_in_direction = ms_phase_of_interest.shape[direction] -1 #index of the voxels of the target surface in the specified direction
            source_nodes_array = node_array[node_array[:, direction] == 0]            
            target_nodes_array = node_array[node_array[:, direction] == idx_max_position_in_direction] 
            source_nodes = [tuple(row) for row in source_nodes_array.tolist()] # Convert arrays back to former to list of tuples representation
            target_nodes = [tuple(row) for row in target_nodes_array.tolist()] # Convert arrays back to former to list of tuples representation

            #calculation of the shortest path from the source points to a single specified target node (compare nx.multi_source_dijkstra manual)
            if (not target_nodes) or (not source_nodes):
                print(f'No valid paths were found for the specified microstructure for phase {phase_of_interest} in direction {direction}.')
                return None

            def calculate_path_length(graph, source_nodes, target_node):
                try:
                    return nx.multi_source_dijkstra(G=graph, sources=source_nodes, target=target_node)[0] 
                except:
                    return None

            len_first_voxel_in_direction = voxel_dimension[direction] # needs to be added to each path length, because the length of the length of the source voxels are omitted when calculation the path lenght via a graph. 
            path_length_list = [length + len_first_voxel_in_direction for target_node in target_nodes if (length := calculate_path_length(graph, source_nodes, target_node)) is not None]

            if not path_length_list:
                print(f'No valid paths were found for the specified microstructure for phase {phase_of_interest} in direction {direction}.')
                return None

            mean_path_length = np.mean(path_length_list)
            length_of_ms_in_specified_direction = ms_phase_of_interest.shape[direction]*voxel_dimension[direction]
            print(f'length_of_ms_in_specified_direction: {length_of_ms_in_specified_direction}')
            print(f'mean_path_length: {mean_path_length}')
            print(f'path_length_list: {path_length_list}')

            return mean_path_length/length_of_ms_in_specified_direction
        
        def SSPSM(ms_phase_of_interest: NDArray[np.bool_]):
            '''
            Skeleton Shortest Path Searching Method
            '''     
            assert ms_phase_of_interest.dtype == bool, "Error: ms_phase_of_interest must only contain bool values!"
            
            print(f'ms_phase_of_interest:\n {ms_phase_of_interest}')
            skeleton_ms = skeletonize(ms_phase_of_interest)
            print(f'skeleton:\n {skeleton_ms}')
            print(f'skeleton_type:\n {type(skeleton_ms)}, {type(skeleton_ms[0,0,0])}')
            print(f'ms_phase_of_interest:\n {type(ms_phase_of_interest)}, {type(ms_phase_of_interest[0,0,0])}')
            return DSPSM(skeleton_ms) # calculate the tortuosity based on the skeleton of the ms 

        # @tf.function
        def model(ms: Union[tf.Tensor, NDArray[Any]]) -> tf.Tensor:
            
            # ms_phase_of_interest is an np.ndarray with bool values representing the 
            # microstructure ms where the searched for phase is represented as True, else False.
            # For further calculations, use ms_phase_of_interest:
            ms_phase_of_interest = ms == phase_of_interest

            if method == 'DSPSM':  
                mean_tortuosity = DSPSM(ms_phase_of_interest)
            elif method == 'SSPSM':  
                mean_tortuosity = SSPSM(ms_phase_of_interest)
                
            return mean_tortuosity
        return model


def register() -> None:
    descriptor_factory.register("Tortuosity", Tortuosity)

       

if __name__=="__main__":

    import os
    folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    #minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    #minimal_example_ms = os.path.join(folder,'alloy_resized_s.npy')
    #ms = np.load(minimal_example_ms)
    # ms = ms.astype(np.float64)
    # print(f'ms: {ms}')
    # print(f'type: {type(ms[0])}')
    # print(f'ms type: {type(ms)}, size: {ms.size, ms.shape}')

    ms = np.zeros((3, 3, 3))
    ms[1,:,:] = 1
    print(f'ms: {ms}')
    print(f'shape: {(ms.shape)}')
    print(f'ms type: {type(ms)}, size: {ms.size}')

    # ms = np.random.randint(2, size=(3, 3, 3))
    # print(f'ms: {ms}, size: {ms.size}')

    tortuosity_descriptor = Tortuosity()
    singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()

    mean_tort = singlephase_descriptor(ms)
    print('\n -----------------------------')
    print(f'Mean tortuosity value: {mean_tort}')


