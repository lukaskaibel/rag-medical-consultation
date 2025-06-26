from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils import ComponentDevice, Device
from haystack.components.rankers import SentenceTransformersSimilarityRanker
import os

def get_base_retrieval_pipeline(document_store: InMemoryDocumentStore):
    pipeline = Pipeline()

    query_embedder = SentenceTransformersTextEmbedder(
        model=os.environ["EMBEDDING_MODEL_NAME"],
        prefix="",
        device=ComponentDevice.from_single(Device.mps()),
        model_kwargs={"torch_dtype": "float16"}
    )
    query_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=int(os.environ["EMBEDDING_MODEL_TOP_K"]))
    reranker = SentenceTransformersSimilarityRanker(
        model="Qwen/Qwen3-Reranker-0.6B", 
        top_k=int(os.environ["EMBEDDING_MODEL_TOP_K"]),
        device=ComponentDevice.from_single(Device.mps()),
        model_kwargs={"torch_dtype": "float16"},
        batch_size=1,
    )
    reranker.warm_up()

    pipeline.add_component(instance=query_embedder, name="query_embedder")
    pipeline.add_component(instance=retriever, name="retriever")
    pipeline.add_component(instance=reranker, name="reranker")

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "reranker")

    return pipeline
