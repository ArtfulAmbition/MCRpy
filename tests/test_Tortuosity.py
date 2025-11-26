import unittest
import numpy as np
import mcrpy
from example_data import example_ms, minimal_example_ms
from mcrpy.descriptors.Tortuosity import Tortuosity
from Langner_functions import DSPSM

class TestTortuosity(unittest.TestCase):
    def setUp(self):
        self.microstructures = {}
        
        ms = np.zeros((3, 3))
        ms[1,:] = 1
        self.microstructures['BlockingLayer_X_2D'] = ms

        ms = np.ones((3, 3))
        ms[:,0] = 0
        self.microstructures['PassingLayer_X_2D'] = ms

        ms = np.zeros((3, 3, 3))
        ms[1,:, :] = 1
        self.microstructures['BlockingLayer_X_3D'] = ms

        ms = np.zeros((3, 3, 3))
        ms[:,0,0] = 1
        self.microstructures['PassingLayer_X_3D'] = ms

        ms = np.ones((3, 3, 3))
        ms[0,0, 0] = 1
        ms[1,0, 0] = 1
        ms[1,1, 0] = 1
        ms[1,2, 0] = 1
        self.microstructures['Edges_X_3D'] = ms

        ms = np.ones((3, 3, 3))
        ms[0,0, 0] = 0
        ms[1,1, 1] = 0
        ms[2,2, 2] = 0
        self.microstructures['Corners_X_3D'] = ms
    
    def test_tortuosity_vals(self):
        self.setUp()
        args = {
            'connectivity': 'sides',
            'method': 'DSPSM',
            'directions': 0,
            'phase_of_interest': 0,
            'voxel_dimension':(2,2,2)       
            }


        tortuosity_descriptor = Tortuosity()

        for method in ['DSPSM','SSPSM']:
            args['method'] = method
            singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor(**args)

            assert(singlephase_descriptor(self.microstructures['BlockingLayer_X_2D']) is None)
            assert(singlephase_descriptor(self.microstructures['PassingLayer_X_2D']) == 1)

            assert(singlephase_descriptor(self.microstructures['BlockingLayer_X_3D']) is None)
            assert(singlephase_descriptor(self.microstructures['PassingLayer_X_3D']) == 1)
        
if __name__ == '__main__':
    tort = TestTortuosity()
    tort.setUp()
    tort.test_tortuosity_vals()
    

