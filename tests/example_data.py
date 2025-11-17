import os

examples_folder = os.path.join(os.path.dirname(__file__), "..", "example_microstructures")
example_ms = os.path.join(
    examples_folder,
    'pymks_ms_64x64_2.npy'
)

minimal_example_ms = os.path.join(
    examples_folder,
    'Holzer2020_Fine_Zoom0.33_Size60.npy'
)