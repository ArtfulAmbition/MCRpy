"""
   Copyright 10/2020 - 04/2021 Paul Seibert for Diploma Thesis at TU Dresden
   Copyright 05/2021 - 12/2021 TU Dresden (Paul Seibert as Scientific Assistant)
   Copyright 2022 TU Dresden (Paul Seibert as Scientific Employee)

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
from __future__ import annotations

import tensorflow as tf

from mcrpy.src import descriptor_factory
from mcrpy.descriptors.PhaseDescriptor import PhaseDescriptor
from typing import Union


class VolumeFractions(PhaseDescriptor):
    is_differentiable = True

    @staticmethod
    def make_singlephase_descriptor(
        phase_of_interest : Union[int,list[int]] = 0,
        **kwargs) -> callable:

        #@tf.function
        def compute_descriptor(microstructure: tf.Tensor) -> tf.Tensor:
            ms_phase_of_interest = ms == phase_of_interest
            return np.mean(ms_phase_of_interest)
        return compute_descriptor


def register() -> None:
    descriptor_factory.register("VolumeFractions", VolumeFractions)

if __name__=="__main__":

    import os
    import numpy as np
    folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
    minimal_example_ms = os.path.join(folder,'BlockingLayer_X_32x32x32.npy')
    ms = np.load(minimal_example_ms)

##------------------------------------------------------------------
   
    volume_fraction_descriptor = VolumeFractions()
    singlephase_descriptor = volume_fraction_descriptor.make_singlephase_descriptor()

    volume_fraction = singlephase_descriptor(ms)
    print('\n -----------------------------')
    print(f'Volume fraction value: {float(volume_fraction)}')

##------------------------------------------------------------------