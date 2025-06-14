from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils import ComponentDevice, Device
import os

def get_base_retrieval_pipeline(document_store: InMemoryDocumentStore):
    pipeline = Pipeline()

    query_embedder = SentenceTransformersTextEmbedder(
        model=os.environ["EMBEDDING_MODEL_NAME"],
        prefix="",
        device=ComponentDevice.from_single(Device.gpu(id=2)),
        model_kwargs={"torch_dtype": "float16"}
    )
    query_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=int(os.environ["EMBEDDING_MODEL_TOP_K"]))

    pipeline.add_component(instance=query_embedder, name="query_embedder")
    pipeline.add_component(instance=retriever, name="retriever")

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

    return pipeline
