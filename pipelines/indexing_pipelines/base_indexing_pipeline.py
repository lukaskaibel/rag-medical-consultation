import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, OpenAIDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice, Device
from models import EmbeddingModelConfig, EmbeddingModelProvider
import torch

def get_base_indexing_pipeline(document_store: InMemoryDocumentStore, embedding_model_config: EmbeddingModelConfig):
    pipeline = Pipeline()
    
    if embedding_model_config.provider == EmbeddingModelProvider.SENTENCE_TRANSFORMER:
        document_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model_config.name,
            prefix="",
            device=ComponentDevice.from_single(Device.gpu(id=0)),
            model_kwargs={"torch_dtype": "float16"},
            batch_size=8,
        )
        document_embedder.warm_up()
    elif embedding_model_config.provider == EmbeddingModelProvider.OPENAI:
        document_embedder = OpenAIDocumentEmbedder(
            model=embedding_model_config.name
        )

    document_writer = DocumentWriter(document_store=document_store)

    pipeline.add_component(instance=document_embedder, name="embedder")
    pipeline.add_component(instance=document_writer, name="writer")

    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline