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
        phases_to_investigate : Union[int,list[int]] = 0, #for which phase number the tortuosity shall be calculated
        **kwargs) -> callable:

        #tf.function
        def array_to_graph(ms, phase_to_investigate:int) -> nx.Graph:
            bool_array = ms == phase_to_investigate
            print(bool_array)
            print('Hallo Word')
            print(type(ms))
            graph = nx.Graph()
            return nx.Graph()

        #@tf.function
        def create_graph_global(ms):
            k = array_to_graph(ms, phase_to_investigate=0)
            graph = 11 + connectivity
            return graph

        #@tf.function
        def DSPSM(ms,connectivity):
            return create_graph_global(ms) + connectivity

        #@tf.function
        def model(ms: Union[tf.Tensor, NDArray[Any]]) -> tf.Tensor:
            tortuosity_val = DSPSM(ms,connectivity)
            return tortuosity_val
        return model


def register() -> None:
    descriptor_factory.register("Tortuosity", Tortuosity)

       

if __name__=="__main__":

    import os
    folder = '../../example_microstructures/' 
    minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    ms = np.load(minimal_example_ms)
    print(f'ms type: {type(ms)}')
    tortuosity_descriptor = Tortuosity()
    singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()

    tort = singlephase_descriptor(ms)
    print(tort)

