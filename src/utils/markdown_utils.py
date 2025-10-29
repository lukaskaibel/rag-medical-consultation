import os
from tqdm import tqdm

def for_each_markdown_file(dir_path: str, callback):
    """
    Iterates over all .md files in the given directory and passes their bytes to the callback function.

    Args:
        dir_path (str): Path to the directory containing .md files.
        callback (function): A function that accepts two arguments: filename and bytes.
    """
    # Collect all .md files first
    md_files = [f for f in os.listdir(dir_path) if f.endswith(".md")]

    # Wrap with tqdm for progress bar
    for filename in tqdm(md_files, desc="Processing Markdown files"):
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "rb") as f:
            bytes = f.read()
            callback(filename, bytes)
