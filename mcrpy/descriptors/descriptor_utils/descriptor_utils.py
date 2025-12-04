import numpy as np
from typing import Union


def get_connectivity_directions(dimensionality: int, connectivity: Union[str,int], mode:str='half'):
            ''' connectivity_directions describes the connectivity for both directions (to and from a node pair) in the different directions, 
                that means for example for (1,0) but also for (-1,0). 
                This can lead to inefficencies in the code. 
                In the current code, the negative direction is implicitly included by making the calculation unidirective.
                for this, only one direction is needed (one_way_connectivity_directions)
            '''
            assert (isinstance(dimensionality,int) and dimensionality in [1,2,3])
            assert connectivity.lower() in ['sides', 'edges', 'corners', 6, 18, 28, 4, 8], "Valid inputs for connectivity are ['sides', 'edges, 'corners' 4, 6, 8, 18, 26]"
            assert mode in ['half', 'full']

            if ((connectivity in ['sides', 6] and dimensionality == 3) or 
                (connectivity in ['sides', 4] and dimensionality == 2) or
                (connectivity in ['edges', 4] and dimensionality == 2)):
                def increment_direction(direction_tuple, index, val):
                    new_direction = np.copy(direction_tuple)
                    new_direction[index] += val
                    return tuple(new_direction)
                full_connectivity_directions = np.array([increment_direction(np.zeros(dimensionality), index, val) 
                                                    for index in range(dimensionality) for val in [1, -1]],dtype=int)  # This creates both +1 and -1 increments
            elif connectivity in ['edges', 18] and dimensionality == 3:
                full_connectivity_directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                            (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                            (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)],dtype=int)
            elif connectivity in ['corners', 26] and dimensionality == 3:
                full_connectivity_directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
                      (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                      (1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1),
                      (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                      (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
                      (1, 1, -1), (1, -1, -1), (-1, 1, -1), (-1, -1, -1)],dtype=int)
            elif connectivity in ['corners', 8] and dimensionality == 2:
                full_connectivity_directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, 1), (1, -1), (-1, -1)],dtype=int)
            else:
                raise Exception(f'Connectivity argument (connectivity={connectivity}) and dimensionality (dimensionality={dimensionality}D) mismatch!  \nValid arguments for 3D are [sides, edges, corners, 6, 18, 28] \nand for 2D [sides, edges, corners, 4, 8].')

            # Create a mask where all values are non-negative and filter the array using the mask
            mask = np.all(full_connectivity_directions >= 0, axis=1)
            one_way_connectivity_directions = full_connectivity_directions[mask]
            
            if mode.lower()=='full':
                return full_connectivity_directions
            else:
                 return one_way_connectivity_directions