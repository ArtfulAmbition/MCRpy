import config
import unittest
import numpy as np
import mcrpy
from example_data import example_ms, minimal_example_ms

class TestTortuosity(unittest.TestCase):
    def setUp(self):
        self.ms = mcrpy.Microstructure.from_npy(minimal_example_ms)
    
    # def test_descriptor_consistency(self):
    #     descriptors = mcrpy.characterize(
    #         self.ms,
    #         settings=mcrpy.CharacterizationSettings(
    #             descriptor_types=['Tortuosity'],
    #             use_multigrid_descriptor=False,
    #             use_multiphase=False,
    #             periodic=True
    #         )
    #     )

    

