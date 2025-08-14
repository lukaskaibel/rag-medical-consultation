import os
from tqdm import tqdm
import pandas as pd
from haystack.document_stores.in_memory import InMemoryDocumentStore

def for_each_document_store(dir_path: str, callback):
    # Collect and sort all .pkl files alphabetically
    pkl_files = sorted(
        [f for f in os.listdir(dir_path) if f.endswith(".json")]
    )

    # Process in order
    for filename in tqdm(pkl_files, desc="Processing document store files"):
        file_path = os.path.join(dir_path, filename)
        store = InMemoryDocumentStore.load_from_disk(file_path)
        callback(filename, store)