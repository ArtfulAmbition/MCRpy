import unittest
import numpy as np
import mcrpy
from example_data import example_ms, minimal_example_ms
from mcrpy.descriptors.Tortuosity import Tortuosity
from Langner_functions import DSPSM

class TestTortuosity(unittest.TestCase):
    def setUp(self):
        self.microstructures = {}
        
        ms = np.zeros((20, 20))
        ms[1:5,:] = 1
        ms = ms.astype(int)
        self.microstructures['BlockingLayer_X_2D_20x20'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_2D_20x20.npy', self.microstructures['BlockingLayer_X_2D_20x20'])

        # can also be used by other descriptors in mcrpy
        ms = np.zeros((32, 32))
        ms[1:5,:] = 1
        ms = ms.astype(int)
        self.microstructures['BlockingLayer_X_2D_32x32'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_2D_32x32.npy', self.microstructures['BlockingLayer_X_2D_32x32'])


        ms = np.zeros((64, 64))
        ms[1:5,:] = 1
        ms = ms.astype(int)
        self.microstructures['BlockingLayer_X_2D_64x64'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_2D_64x64.npy', self.microstructures['BlockingLayer_X_2D_64x64'])

        ms = np.zeros((32, 32, 32))
        ms[1:5,:] = 1
        ms = ms.astype(int)
        self.microstructures['BlockingLayer_X_32x32x32'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_32x32x32.npy', self.microstructures['BlockingLayer_X_32x32x32'])

        ms = np.zeros((20, 20, 20))
        ms[1:5,:] = 1
        ms = ms.astype(int)
        self.microstructures['BlockingLayer_X_20x20x20'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy', self.microstructures['BlockingLayer_X_20x20x20'])

        ms = np.zeros((20, 20, 20))
        ms[1:5,:] = 1
        ms = ms.astype(int)
        self.microstructures['BlockingLayer_X_20x20x20'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy', self.microstructures['BlockingLayer_X_20x20x20'])

        ms = np.ones((3,3,3))
        ms[0,0,0] = 0
        ms[1,1,1] = 0
        ms[2,2,2] = 0
        ms = ms.astype(int)
        self.microstructures['Diag_3x3x3'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/Diag_3x3x3.npy', self.microstructures['Diag_3x3x3'])

        ms = np.zeros((5,5,1))
        ms[1,2,0] = 1
        ms[2,1,0] = 1
        ms[2,2,0] = 1
        ms[2,3,0] = 1
        ms[3,2,0] = 1
        ms = ms.astype(int)
        self.microstructures['Cross_5x5x1'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/Cross_5x5x1.npy', self.microstructures['Cross_5x5x1'])


        ms = np.ones((3,3))
        ms[0,0] = 0
        ms[1,1] = 0
        ms[2,2] = 0
        ms = ms.astype(int)
        self.microstructures['Diag_3x3'] = ms
        np.save('/home/sobczyk/Dokumente/MCRpy/example_microstructures/Diag_3x3.npy', self.microstructures['Diag_3x3'])


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
            'direction': 0,
            'phase_of_interest': 0,
            'voxel_dimension':(2,2,2)       
            }


        tortuosity_descriptor = Tortuosity()
        singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor(**args)

        #assert(singlephase_descriptor(self.microstructures['BlockingLayer_X_2D']) is None)
        assert(singlephase_descriptor(self.microstructures['PassingLayer_X_2D']) == 1)

        assert(singlephase_descriptor(self.microstructures['BlockingLayer_X_3D']) == 0)
        assert(singlephase_descriptor(self.microstructures['PassingLayer_X_3D']) == 1)

        args2 = args.copy()
        args2['connectivity'] = 'corners'
        args2['voxel_dimension'] = (2,2,2)
        singlephase_descriptor2 = tortuosity_descriptor.make_singlephase_descriptor(**args2)
        assert(singlephase_descriptor2(self.microstructures['Diag_3x3'])==(2+2*np.sqrt(2**2+2**2))/(3*2))
        assert(singlephase_descriptor2(self.microstructures['Diag_3x3x3'])==(2+2*np.sqrt(2**2+2**2+2**2))/(3*2))

        
if __name__ == '__main__':
    tort = TestTortuosity()
    tort.setUp()
    tort.test_tortuosity_vals()

    

