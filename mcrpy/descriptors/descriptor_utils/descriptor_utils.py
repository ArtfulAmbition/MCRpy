from typing import Union
import itertools

def get_connectivity_directions(dimensionality: int, connectivity: Union[str,int]):
    ''' returns a list of tuples which represent the directions of connectivity for a voxel at origin (0,0) or (0,0,0).
        The directions of connectivity are governed by connectivity.
        Valid inputs for connectivity are 'sides', 'edges, 'corners' 4, 6, 8, 18 or 26.
    '''
    assert (isinstance(dimensionality,int) and dimensionality in [2,3]), 'Only 2D and 3D implemented.'
    assert connectivity.lower() in ['sides', 'edges', 'corners', 6, 18, 28, 4, 8], "Valid inputs for connectivity are ['sides', 'edges, 'corners' 4, 6, 8, 18, 26]"

    def positive(original_tuple:tuple[int]):
         ''' returns a tuple with the absolute vals of the input tuple '''
         return tuple(abs(x) for x in original_tuple)

    # create a list of all possible direction tuples from origin (0,0,0)
    # For 2D this list looks like this: [(0,1), (0,-1), (1,0) ... ]
    values_to_combine = [int(0), int(1), int(-1)]
    all_combinations_with_origin = list(itertools.product(values_to_combine, repeat=dimensionality))
    all_combinations_without_origin = [combination for combination in all_combinations_with_origin 
                        if any(x !=0 for x in combination)]

    # remove all direction tuples which refer to corners, that are for example (1,1) or (1,1,1)
    no_corner_combination= [combination for combination in all_combinations_without_origin 
                            if (sum(positive(combination)) <= 2)]
    
    # remove all direction tuples which refer to corners or edges(for example (1,1) or (1,1,1) or (1,0,1) or (-1,1,0))
    combinations_only_sides = [combination for combination in all_combinations_without_origin 
                                       if (sum(positive(combination)) == 1)]

    if ((connectivity in ['sides', 6] and dimensionality == 3) or 
        (connectivity in ['sides', 4] and dimensionality == 2) or
        (connectivity in ['edges', 4] and dimensionality == 2)):

        valid_directions = combinations_only_sides       

    elif (connectivity in ['edges', 18] and dimensionality == 3):
         
         valid_directions = no_corner_combination
    
    elif ((connectivity in ['corners', 26] and dimensionality == 3) or
          (connectivity in ['corners', 8] and dimensionality == 2)):
         
        valid_directions = all_combinations_without_origin 

    else:
        raise Exception(f'Connectivity argument (connectivity={connectivity}) and dimensionality (dimensionality={dimensionality}D) mismatch!  \nValid arguments for 3D are [sides, edges, corners, 6, 18, 28] \nand for 2D [sides, edges, corners, 4, 8].')

    return valid_directions


if __name__=="__main__":
    directions = get_connectivity_directions(dimensionality=2, connectivity="corners")
    print(f'directions: {directions}')
    print(f'len(directions): {len(directions)}')
