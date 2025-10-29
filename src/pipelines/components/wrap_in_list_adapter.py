from haystack import component
from haystack.dataclasses import Document
from typing import List

@component
class WrapInListAdapter:
    @component.output_types(output=List[List[Document]])
    def run(self, documents: List[Document]):
        return { "output": [documents] }