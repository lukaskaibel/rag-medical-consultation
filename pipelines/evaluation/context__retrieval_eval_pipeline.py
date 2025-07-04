from pipelines.retrieval_pipelines.base_retrieval_pipeline import get_base_retrieval_pipeline
from pipelines.components.wrap_in_list_adapter import WrapInListAdapter
from pipelines.components.evaluators import DocumentMAPEvaluator, DocumentMRREvaluator, DocumentRecallEvaluator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from models import EmbeddingModelConfig, RerankingModelConfig

def get_context_retrieval_eval_pipeline(context_indexing_store: InMemoryDocumentStore, embedding_model_config: EmbeddingModelConfig, reranking_model_config: RerankingModelConfig) -> Pipeline:
    context_retrieval_pipeline = get_base_retrieval_pipeline(context_indexing_store, embedding_model_config, reranking_model_config)

    map_evaluator = DocumentMAPEvaluator()
    mrr_evaluator = DocumentMRREvaluator()
    recall_evaluator = DocumentRecallEvaluator()

    retriever_to_evaluator_adapter = WrapInListAdapter()

    context_retrieval_pipeline.add_component(instance=map_evaluator, name="map_evaluator")
    context_retrieval_pipeline.add_component(instance=mrr_evaluator, name="mrr_evaluator")
    context_retrieval_pipeline.add_component(instance=recall_evaluator, name="recall_evaluator")
    context_retrieval_pipeline.add_component(instance=retriever_to_evaluator_adapter, name="retriever_to_evaluator_adapter")
    context_retrieval_pipeline.connect("reranker", "retriever_to_evaluator_adapter")
    context_retrieval_pipeline.connect("retriever_to_evaluator_adapter.output", "map_evaluator.retrieved_documents")
    context_retrieval_pipeline.connect("retriever_to_evaluator_adapter.output", "mrr_evaluator.retrieved_documents")
    context_retrieval_pipeline.connect("retriever_to_evaluator_adapter.output", "recall_evaluator.retrieved_documents")

    return context_retrieval_pipeline

