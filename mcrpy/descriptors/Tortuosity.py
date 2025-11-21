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
from typing import Any

class Tortuosity(PhaseDescriptor):
    is_differentiable = False

    @staticmethod
    def make_singlephase_descriptor(**kwargs) -> callable:

        @tf.function
        def compute_descriptor(microstructure: tf.Tensor) -> NDArray[Any]:
            return mean_tortuosity(microstructure)
        return compute_descriptor


def register() -> None:
    descriptor_factory.register("Tortuosity", Tortuosity)


def mean_tortuosity(ms: tf.Tensor) -> NDArray[Any]:

    # ---------------------Funktion: Den kuerzesten Pfad von unten nach oben mit 'Dijkstra Algorithmus' suchen--------------
    def calculate_shortest_path(graph, target_face_list, source_nodes):
        '''
        Calculation of the shortest path with the Dijkstra algorithm
        Function written by Shihai Liu (Forschungspraktikum)
        '''
        try:
            shortest_length, shortest_path = nx.multi_source_dijkstra(graph, target_face_list, source_nodes)
            return shortest_length, shortest_path, source_nodes
        # Situation, dass der Pfad vom Startpunkt zum Endpunkt nicht erreichbar ist, None zurückgeben.
        except:
            return None, None, None