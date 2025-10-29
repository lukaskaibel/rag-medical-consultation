import re
from typing import List, Dict
from haystack import component
from haystack.dataclasses import Document

@component
class Filter:
    def __init__(self, condition):
        self.condition = condition

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        filtered = [
            doc
            for doc in documents
            if self.condition(doc)
        ]
        return {"documents": filtered}
