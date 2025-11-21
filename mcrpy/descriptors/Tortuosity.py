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
        a=1,
        b=2,
        **kwargs) -> callable:

        @tf.function
        def compute_descriptor(microstructure: Union[tf.Tensor, NDArray[Any]]) -> tf.Tensor:
            return a, b, 11
        return compute_descriptor


def register() -> None:
    descriptor_factory.register("Tortuosity", Tortuosity)

       

if __name__=="__main__":

    import os
    folder = '../../example_microstructures/' 
    minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')
    ms = np.load(minimal_example_ms)
    tortuosity_descriptor = Tortuosity()
    singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()

    a, b, c = singlephase_descriptor(ms)
    print(c)

