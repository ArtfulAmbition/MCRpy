#!/usr/bin/env python3
"""
Standalone script to read MCRpy characterization pickle files.
Run this from outside the MCRpy directory to avoid import conflicts.
"""
import sys
import os
import pickle

# Try to import numpy, but continue without it if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: numpy not available, some features limited")

def read_characterization(filename):
    """Read a characterization pickle file."""
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return None
    
    print(f"Reading: {filename}\n")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data

def print_characterization_info(data):
    """Print information about characterization data."""
    if data is None:
        return
    
    print('='*70)
    print('CHARACTERIZATION FILE CONTENTS')
    print('='*70)
    
    print('\nKeys:', list(data.keys()))
    
    if 'settings' in data:
        settings = data['settings']
        print('\nSettings:')
        print(f'  Descriptor types: {settings.descriptor_types}')
        print(f'  Limit to: {settings.limit_to}')
        print(f'  Use multiphase: {settings.use_multiphase}')
    
    print('\nDescriptor Data:')
    for key in data.keys():
        if key != 'settings':
            value = data[key]
            print(f'\n{key}:')
            if hasattr(value, 'shape'):
                print(f'  Shape: {value.shape}')
                print(f'  Type: {type(value).__name__}')
                if hasattr(value, 'dtype'):
                    print(f'  Dtype: {value.dtype}')
                if HAS_NUMPY and isinstance(value, np.ndarray):
                    print(f'  Min: {value.min():.6f}, Max: {value.max():.6f}')
            else:
                print(f'  Type: {type(value)}')
                print(f'  Value: {value}')
    
    print('\n' + '='*70)

if __name__ == '__main__':
    # Default file
    filename = '/home/sobczyk/Dokumente/MCRpy/results/BlockingLayer_X_32x32x32_characterization.pickle'
    
    # Allow command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    data = read_characterization(filename)
    print(data)
    # print_characterization_info(data)
