import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice, Device
from pipelines.components.contextualiser import Contextualiser
from prompt import CONTEXTUALISER_PROMPT

def get_context_indexing_pipeline(document_store: InMemoryDocumentStore):
    pipeline = Pipeline()

    contextualiser = Contextualiser(
        prompt=CONTEXTUALISER_PROMPT,
        model=os.environ["LLM_NAME"],
        generation_kwargs={ "temperature": 0.0, "num_ctx": int(os.environ["LLM_CONTEXT_SIZE"]) },
        keep_alive=-1
    )
    
    document_embedder = SentenceTransformersDocumentEmbedder(
        model=os.environ["EMBEDDING_MODEL_NAME"],
        prefix="",
        device=ComponentDevice.from_single(Device.gpu(id=0)),
        model_kwargs={"torch_dtype": "float16"}
    )
    document_embedder.warm_up()

    document_writer = DocumentWriter(document_store=document_store)

    pipeline.add_component(instance=contextualiser, name="contextualiser")
    pipeline.add_component(instance=document_embedder, name="embedder")
    pipeline.add_component(instance=document_writer, name="writer")

    pipeline.connect("contextualiser.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline