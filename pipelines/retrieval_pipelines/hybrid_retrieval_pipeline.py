from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.utils import ComponentDevice, Device
from haystack.components.joiners.document_joiner import DocumentJoiner
from models import EmbeddingModelConfig, RerankingModelConfig, EmbeddingModelProvider, RerankingModelProvider, RewriterModelConfig
from pipelines.components.qwen_yes_no_reranker import QwenYesNoReranker
from typing import Optional
import os

def get_hybrid_retrieval_pipeline(
    document_store: InMemoryDocumentStore,
    embedding_model_config: EmbeddingModelConfig,
    reranking_model_config: Optional[RerankingModelConfig] = None,
    rewriter_model_config: Optional[RewriterModelConfig] = None,
) -> Pipeline:
    pipeline = Pipeline()

    query_embedder = SentenceTransformersTextEmbedder(
        model=embedding_model_config.name,
        prefix="Instruct: Given a question, retrieve relevant passages that answer the question\nQuestion:",
        device=ComponentDevice.from_single(Device.gpu(id=2)),
        model_kwargs={"torch_dtype": "float16"}
    )
    query_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    bm25_retriever = InMemoryBM25Retriever(document_store=document_store)

    document_joiner = DocumentJoiner()

    reranker = QwenYesNoReranker(
        model=reranking_model_config.name,
        device=ComponentDevice.from_single(Device.gpu(id=3)),
        batch_size=1,
        instruction="Given a question, retrieve all the relevant passages that answer that query",
    )

    pipeline.add_component(instance=query_embedder, name="query_embedder")
    pipeline.add_component(instance=bm25_retriever, name="bm25_retriever")
    pipeline.add_component(instance=retriever, name="retriever")
    pipeline.add_component(instance=document_joiner, name="joiner")
    pipeline.add_component(instance=reranker, name="reranker")

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("bm25_retriever", "joiner")
    pipeline.connect("retriever", "joiner")
    pipeline.connect("joiner", "reranker")

    return pipeline
