import unittest
import numpy as np
import mcrpy
from example_data import example_ms, minimal_example_ms
from mcrpy.descriptors.Tortuosity import Tortuosity
from Langner_functions import DSPSM

# class TestTortuosity(unittest.TestCase):
#     def setUp(self):

# folder = '/home/sobczyk/Dokumente/MCRpy/example_microstructures' 
# minimal_example_ms = os.path.join(folder,'Holzer2020_Fine_Zoom0.33_Size60.npy')

print("Testing old function: \n\n")

ms = np.load(minimal_example_ms)

print(f'ms type: {type(ms)}, size: {ms.size}')
tort = DSPSM(_array=ms, _Size_of_Voxel_x=1, _Size_of_Voxel_y =1, _Size_of_Voxel_z=1, _connectivity=6, dir=1, _celltags=[0,1,2])
print(f'tort= {tort}')
#singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()

#tort = singlephase_descriptor(ms)

print('Testing new function: \n\n')

tortuosity_descriptor = Tortuosity()
singlephase_descriptor = tortuosity_descriptor.make_singlephase_descriptor()
tort = singlephase_descriptor(ms)


    

