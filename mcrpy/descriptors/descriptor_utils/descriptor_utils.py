from typing import Union
import numpy as np
import itertools
import matplotlib.pyplot as plt

def get_connectivity_directions(dimensionality: int, connectivity: Union[str,int]):
    ''' returns a list of tuples [(0,1), (0,-1), (1,0), (-1,0), (1,1) ... ]
        which represent the directions of connectivity for a voxel at origin (0,0) or (0,0,0).
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

def slice_ndarray(data:np.ndarray, axis:int=0, index:int=0):
    """
    Slices an ndarray of any dimensionality based on the specified axis and index.

    Parameters:
    - data (ndarray): The ndarray to be sliced.
    - axis (int): The axis to slice (0-based indexing).
    - index (int): The index at which to slice.

    Returns:
    - ndarray: The sliced ndarray.
    """
    assert (isinstance(data,np.ndarray) and 
            isinstance(axis,int) and 
            isinstance(index,int)), "Input type error in slice_ndarray."

    # Check if the axis is valid
    if axis < 0 or axis >= data.ndim:
        raise ValueError(f"Axis must be between 0 and {data.ndim - 1}.")

    # Create a slice object
    slices = [slice(None)] * data.ndim  # Start with full slices for all axes
    slices[axis] = index  # Set the specified axis to the index

    return data[tuple(slices)]  # Return the sliced ndarray

def plot_slices(data: np.ndarray, 
                direction: int, 
                rows: int=2, 
                max_number_slices:int=10, 
                block:bool=True):
    """
    Plots each slice of the given ndarray in subplots using matshow,
    arranged over a specified number of rows.

    Parameters:
    - data (ndarray): The input n-dimensional array.
    - direction (int): The direction to slice the array (0, 1, or 2 for 3D arrays).
    - rows (int): The number of rows in which to arrange the subplots.
    """
    # Check if direction is valid
    if direction < 0 or direction >= data.ndim:
        raise ValueError(f"Direction must be between 0 and {data.ndim - 1}.")

    # Number of slices along the specified direction
    total_slices = data.shape[direction] if data.shape[direction] < max_number_slices else max_number_slices
    
    # Calculate number of columns needed
    cols = (total_slices + rows - 1) // rows  # Ceiling division
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # Flatten axes for easier indexing
    
    # Plot each slice
    for slice_number in range(total_slices):
        # Extract the current slice using the provided slice_ndarray function
        slice_data = slice_ndarray(data, axis=direction, index=slice_number)
        
        # Plot using matshow
        axes[slice_number].matshow(slice_data)
        axes[slice_number].set_title(f'Slice {slice_number}')
        axes[slice_number].axis('off')  # Turn off axis lines and labels

    # Turn off axes for any unused subplots
    for i in range(total_slices, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.show(block=block)


if __name__=="__main__":
    directions = get_connectivity_directions(dimensionality=2, connectivity="corners")
    print(f'directions: {directions}')
    print(f'len(directions): {len(directions)}')
