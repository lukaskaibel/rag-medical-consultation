import os
from tqdm import tqdm
import pandas as pd

def for_each_pickle_file(dir_path: str, callback):
    """
    Iterates over all .pkl files in the given directory (in alphabetical order)
    and passes their decoded df to the callback function.

    Args:
        dir_path (str): Path to the directory containing .pkl files.
        callback (function): A function that accepts two arguments: filename and df.
    """
    # Collect and sort all .pkl files alphabetically
    pkl_files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(".pkl")]
    )

    # Optional: if you want case-insensitive sorting, use:
    # pkl_files = sorted(pkl_files, key=str.lower)

    # Process in order
    for filename in tqdm(pkl_files, desc="Processing Pickle files"):
        file_path = os.path.join(dir_path, filename)
        df = pd.read_pickle(file_path)
        callback(filename, df)