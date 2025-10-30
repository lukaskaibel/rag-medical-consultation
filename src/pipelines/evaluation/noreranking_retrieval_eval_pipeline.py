from src.pipelines.components.wrap_in_list_adapter import WrapInListAdapter
from src.pipelines.retrieval_pipelines.base_retrieval_pipeline import get_base_retrieval_pipeline
from src.pipelines.components.evaluators import DocumentMAPEvaluator, DocumentMRREvaluator, DocumentRecallEvaluator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from src.models import EmbeddingModelConfig

def get_base_retrieval_eval_pipeline(
    base_indexing_store: InMemoryDocumentStore,
    embedding_model_config: EmbeddingModelConfig,
) -> Pipeline:
    base_retrieval_pipeline = get_base_retrieval_pipeline(base_indexing_store, embedding_model_config)

    map_evaluator = DocumentMAPEvaluator()
    mrr_evaluator = DocumentMRREvaluator()
    recall_evaluator = DocumentRecallEvaluator()

    retriever_to_evaluator_adapter = WrapInListAdapter()

    base_retrieval_pipeline.add_component(instance=map_evaluator, name="map_evaluator")
    base_retrieval_pipeline.add_component(instance=mrr_evaluator, name="mrr_evaluator")
    base_retrieval_pipeline.add_component(instance=recall_evaluator, name="recall_evaluator")
    base_retrieval_pipeline.add_component(instance=retriever_to_evaluator_adapter, name="retriever_to_evaluator_adapter")
    base_retrieval_pipeline.connect("retriever", "retriever_to_evaluator_adapter")
    base_retrieval_pipeline.connect("retriever_to_evaluator_adapter.output", "map_evaluator.retrieved_documents")
    base_retrieval_pipeline.connect("retriever_to_evaluator_adapter.output", "mrr_evaluator.retrieved_documents")
    base_retrieval_pipeline.connect("retriever_to_evaluator_adapter.output", "recall_evaluator.retrieved_documents")

    return base_retrieval_pipeline