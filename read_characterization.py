#!/usr/bin/env python3
"""
Standalone script to read MCRpy characterization pickle files.
Run this from outside the MCRpy directory to avoid import conflicts.
"""
import sys
import os
import pickle

def read_characterization(filename):
    """Read a characterization pickle file."""
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        return None
    
    print(f"Reading: {filename}\n")
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data

if __name__ == '__main__':
    # Default file
    filename = '/home/sobczyk/Dokumente/MCRpy/results/BlockingLayer_X_32x32x32_characterization.pickle'
    
    # Allow command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    data = read_characterization(filename)
    print(data)
