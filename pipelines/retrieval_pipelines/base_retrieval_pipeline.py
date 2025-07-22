import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, OpenAITextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils import ComponentDevice, Device
from haystack.components.joiners import BranchJoiner
from haystack.dataclasses import Document
from models import EmbeddingModelConfig, RerankingModelConfig, EmbeddingModelProvider, RerankingModelProvider, RewriterModelConfig
from pipelines.components.qwen_yes_no_reranker import QwenYesNoReranker
from pipelines.components.query_rewriter import QueryRewriter
from typing import Optional, List

def get_base_retrieval_pipeline(
    document_store: InMemoryDocumentStore, 
    embedding_model_config: EmbeddingModelConfig,
    reranking_model_config: Optional[RerankingModelConfig] = None,
    rewriter_model_config: Optional[RewriterModelConfig] = None,
) -> Pipeline:
    pipeline = Pipeline()

    if embedding_model_config.provider == EmbeddingModelProvider.SENTENCE_TRANSFORMER:
        query_embedder = SentenceTransformersTextEmbedder(
            model=embedding_model_config.name,
            prefix="Instruct: Given a question, retrieve relevant passages that answer the question\nQuestion:",
            device=ComponentDevice.from_single(Device.gpu(id=2)),
            model_kwargs={"torch_dtype": "float16"}
        )
        query_embedder.warm_up()
    elif embedding_model_config.provider == EmbeddingModelProvider.OPENAI:
        query_embedder = OpenAITextEmbedder(
            model=embedding_model_config.name
        )
    else:
        return ValueError("Embedding provider not implemented")

    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    
    pipeline.add_component(instance=query_embedder, name="query_embedder")
    pipeline.add_component(instance=retriever, name="retriever")
    pipeline.add_component(instance=BranchJoiner(List[Document]), name="joiner")

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

    if reranking_model_config != None:
        if reranking_model_config.provider == RerankingModelProvider.HUGGING_FACE:
            reranker = QwenYesNoReranker(
                model=reranking_model_config.name,
                device=ComponentDevice.from_single(Device.gpu(id=3)),
                batch_size=1,
                instruction="Given a question, retrieve all the relevant passages that answer that query",
            )
            reranker.warm_up()
            pipeline.add_component(instance=reranker, name="reranker")
            pipeline.connect("retriever", "reranker")
            pipeline.connect("reranker", "joiner")
        else:
            return ValueError("Reranking provider not implemented")
    else:
        pipeline.connect("retriever", "joiner")

    
    if rewriter_model_config != None:
        rewriter = QueryRewriter(
            rewriter_model_config.prompt,
            rewriter_model_config.llm_config,
            generation_kwargs={ "temperature": 0.0, "num_ctx": int(os.environ["LLM_CONTEXT_SIZE"]) },
            keep_alive=-1
        )
        pipeline.add_component("rewriter", rewriter)
        pipeline.connect("rewriter.query", "query_embedder.text")

    return pipeline
