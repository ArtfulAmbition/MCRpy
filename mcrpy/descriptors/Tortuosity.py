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

import logging

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor
from numpy.typing import NDArray
from typing import Any, Union
import numpy as np

class Tortuosity(PhaseDescriptor):
    is_differentiable = False

    @staticmethod
    def make_singlephase_descriptor(
        connectivity : int = 6,
        method : str = 'DSPSM',
        directions : Union[int,list[int]] = 0, #0:x, 1:y, 2:z
        phase_of_interest : Union[int,list[int]] = 0, #for which phase number the tortuosity shall be calculated
        voxel_dimension:tuple[float] =(1,1,1),
        **kwargs) -> callable:

        #@tf.function
        def ms_to_graph(ms:Union[tf.Tensor,NDArray],) -> nx.Graph:
            # ms mcirostructure
            # phase_to_investigate integer representation of the phase for which the connectivity graph should be investigated
            # connectivity: number of adjacent neighbors for which a connectivity should be allowed. Implemented possibilites are 6, 18 and 24
            # voxel_dimension: the dimension (lx,ly,lz) of each considered voxel. Is needed to calculate the distance between neighbors

            lx, ly, lz = voxel_dimension

            #--------- create nodes: ------------------
            #TODO: make more efficient by not switching between tf and np.ndarray
            if (type(ms) is tf.Tensor):
                ms = ms.numpy()
            coordinates = tf.where(ms == phase_of_interest) # getting the coordinates of voxels with phase id equal to phase_to_investigate
            coordinates_np = coordinates.numpy()

            nodes = map(tuple,coordinates_np) # creating a vector of node tuples for insertion to nx.Graph()
            
            graph = nx.Graph()
            graph.add_nodes_from(nodes) 

            #--------- connect nodes: -----------------
            if connectivity == 6:
                directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])

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
                valid_neighbors = neighbors[np.isin(neighbors, graph_nodes).all(axis=1)] #extract the  neighbors which are of the same phase

                # calculating the distance to the neighbor:
                normed_distance_vectors = valid_neighbors - node # unit vector differences between the current node to its valid neighbors 
                weighted_distance_vectors = normed_distance_vectors * [lx,ly,lz] # weighted vector differences between the current node to its valid neighbors 
                distances = np.linalg.norm(weighted_distance_vectors, axis=1)
                                
                edges_to_add.extend([(node, tuple(valid_neighbor), {'weight': distance}) for valid_neighbor, distance in zip(valid_neighbors, distances)])

                # if i==0:
                #     print(f'valid_neighbors= {valid_neighbors}')
                #     print(f'node= {node}')
                #     print(f'distance= {distances}')
                #     print(f'edges_to_add= {edges_to_add}')
                #     i+=1

            graph.add_edges_from(edges_to_add)
            return graph


        # @tf.function
        def DSPSM(ms:tf.Tensor, direction=1):

            #create the graph for phase_of_interest
            graph:nx.Graph = ms_to_graph(ms)
            node_array:np.ndarray = np.array(graph.nodes)

            #identify the source nodes (from where the paths through the microstructure shall start, at minimum of coordinate in specified direction)
            idx_max_position_in_direction = ms.shape[direction] -1 #index of the voxels of the target surface in the specified direction
            source_nodes = node_array[node_array[:, direction] == 0]            
            target_nodes = node_array[node_array[:, direction] == idx_max_position_in_direction] 

            #calculation of the shortest path from source to target nodes
            #shortest_path = calculate_shortest_path(graph,source_nodes,target_nodes)
            #print(f'source.nodes: {source_nodes}, {type(source_nodes)}')

            return 1

        # @tf.function
        def model(ms: Union[tf.Tensor, NDArray[Any]]) -> tf.Tensor:
            tortuosity_val = DSPSM(ms)
            return tortuosity_val
        return model


def register() -> None:
    descriptor_factory.register("Tortuosity", Tortuosity)

       

if __name__=="__main__":

    import os
    folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    ms = np.load(minimal_example_ms)
    print(f'ms type: {type(ms)}, size: {ms.size}')
    tortuosity_descriptor = Tortuosity()
    singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()

    tort = singlephase_descriptor(ms)
    print('\n -----------------------------')
    print(tort)


