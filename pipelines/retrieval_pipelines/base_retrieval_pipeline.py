from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.utils import ComponentDevice, Device
from models import EmbeddingModelConfig, RerankingModelConfig, EmbeddingModelProvider, RerankingModelProvider
from pipelines.components.qwen_yes_no_reranker import QwenYesNoReranker

def get_base_retrieval_pipeline(
    document_store: InMemoryDocumentStore, 
    embedding_model_config: EmbeddingModelConfig,
    reranking_model_config: RerankingModelConfig,
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
    else:
        return ValueError("Embedding provider not implemented")

    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    if reranking_model_config.provider == RerankingModelProvider.HUGGING_FACE:
        reranker = QwenYesNoReranker(
            model=reranking_model_config.name,
            device=ComponentDevice.from_single(Device.gpu(id=3)),
            batch_size=1,
            instruction="Given a question, retrieve all the relevant passages that answer that query",
        )
        reranker.warm_up()
    else:
        return ValueError("Reranking provider not implemented")

    pipeline.add_component(instance=query_embedder, name="query_embedder")
    pipeline.add_component(instance=retriever, name="retriever")
    pipeline.add_component(instance=reranker, name="reranker")

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "reranker")

    return pipeline
