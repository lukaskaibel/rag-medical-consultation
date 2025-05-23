import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice, Device

def add_base_indexing(pipeline: Pipeline, document_store: InMemoryDocumentStore):
    
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=os.environ["EMBEDDING_MODEL_NAME"],
        prefix="",
        device=ComponentDevice.from_single(Device.gpu(id=0)),
        model_kwargs={"torch_dtype": "float16"}
    )
    document_embedder.warm_up()

    document_writer = DocumentWriter(document_store=document_store)

    pipeline.add_component(instance=document_embedder, name="embedder")
    pipeline.add_component(instance=document_writer, name="writer")

    pipeline.connect("embedder.documents", "writer.documents")