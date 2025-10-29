from pipelines.retrieval_pipelines.base_retrieval_pipeline import get_base_retrieval_pipeline
from pipelines.components.wrap_in_list_adapter import WrapInListAdapter
from pipelines.components.evaluators import DocumentMAPEvaluator, DocumentMRREvaluator, DocumentRecallEvaluator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pipelines.components.random_retriever import RandomRetriever
from haystack import Pipeline

def get_random_retrieval_eval_pipeline(document_store: InMemoryDocumentStore) -> Pipeline:
    pipeline = Pipeline()

    pipeline.add_component(instance=RandomRetriever(document_store), name="retriever")

    map_evaluator = DocumentMAPEvaluator()
    mrr_evaluator = DocumentMRREvaluator()
    recall_evaluator = DocumentRecallEvaluator()

    retriever_to_evaluator_adapter = WrapInListAdapter()

    pipeline.add_component(instance=map_evaluator, name="map_evaluator")
    pipeline.add_component(instance=mrr_evaluator, name="mrr_evaluator")
    pipeline.add_component(instance=recall_evaluator, name="recall_evaluator")
    pipeline.add_component(instance=retriever_to_evaluator_adapter, name="retriever_to_evaluator_adapter")
    pipeline.connect("retriever", "retriever_to_evaluator_adapter")
    pipeline.connect("retriever_to_evaluator_adapter.output", "map_evaluator.retrieved_documents")
    pipeline.connect("retriever_to_evaluator_adapter.output", "mrr_evaluator.retrieved_documents")
    pipeline.connect("retriever_to_evaluator_adapter.output", "recall_evaluator.retrieved_documents")

    return pipeline

