#!/usr/bin/env python3
"""
Programmatic test runner: call characterization and reconstruction
via direct function calls (no subprocess). This ensures consistent
settings and avoids CLI parsing issues.
"""
from __future__ import annotations

from typing import Tuple
import traceback

from mcrpy.characterize import characterize
from mcrpy.reconstruct import reconstruct
from mcrpy.src.Settings import CharacterizationSettings, ReconstructionSettings
from mcrpy.src.Microstructure import Microstructure

# Configuration ------------------------------------------------------------
MS_FILE = "/home/sobczyk/Dokumente/MCRpy/example_microstructures/BlockingLayer_X_20x20x20.npy"
CHAR_DESCRIPTOR_TYPES = ["Tortuosity", "VolumeFractions"]
CHAR_SLICE_MODE = "no_slicing"
CHAR_DIRECTIONS = [0, 1, 2]
CHAR_INFORMATION = "tort_vf"
CHAR_DATA_FOLDER = "results"
CHAR_FULL_3D = True

RECON_DESCRIPTOR_TYPES = ["Tortuosity"]
RECON_OPTIMIZER = "SimulatedAnnealing"
RECON_MAX_ITER = 500
RECON_EXTENT = (32, 32, 32)  # (x, y, z)

# Helper functions --------------------------------------------------------

def run_characterization(ms_file: str) -> dict:
    """Load microstructure and run characterization, return descriptor dict."""
    print(f"Loading microstructure from {ms_file}")
    ms = Microstructure.from_npy(ms_file, use_multiphase=False, trainable=False)

    settings = CharacterizationSettings(
        descriptor_types=CHAR_DESCRIPTOR_TYPES,
        slice_mode=CHAR_SLICE_MODE,
        information=CHAR_INFORMATION,
        data_folder=CHAR_DATA_FOLDER,
        directions_list=CHAR_DIRECTIONS,
        full_3d=CHAR_FULL_3D,
    )

    print("Running characterization with settings:", settings)
    descriptors = characterize(ms, settings)
    print("Characterization completed. Keys:", list(descriptors.keys()))
    return descriptors


def run_reconstruction(descriptor_dict: dict, desired_shape: Tuple[int, int, int]):
    """Run reconstruction with a given descriptor dict and desired shape."""
    print("Preparing reconstruction settings")
    recon_settings = ReconstructionSettings(
        descriptor_types=RECON_DESCRIPTOR_TYPES,
        optimizer_type=RECON_OPTIMIZER,
        max_iter=RECON_MAX_ITER,
        full_3d=True,
    )
    print("Starting reconstruction with desired shape:", desired_shape)
    convergence_data, last_frame = reconstruct(descriptor_dict, desired_shape, settings=recon_settings)
    print("Reconstruction finished. Returned last frame and convergence data keys:", list(convergence_data.keys()))
    return convergence_data, last_frame


if __name__ == "__main__":
    try:
        descriptors = run_characterization(MS_FILE)
        conv, last = run_reconstruction(descriptors, RECON_EXTENT)
    except Exception as e:
        print("An error occurred during programmatic test run:")
        traceback.print_exc()
        raise



