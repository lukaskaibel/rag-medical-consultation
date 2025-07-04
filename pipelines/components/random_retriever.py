from typing import List
from haystack import component
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
import random

@component
class RandomRetriever():
    def __init__(self, document_store: InMemoryDocumentStore):
        self.document_store = document_store
    
    @component.output_types(documents=List[Document])
    def run(self, top_k: int):
        documents = self.document_store.filter_documents()
        top_k = min(top_k, len(documents))
        
        sampled_docs = random.sample(documents, k=top_k) if documents else []
        return { "documents": sampled_docs }