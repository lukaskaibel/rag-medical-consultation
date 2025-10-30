from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack_integrations.components.evaluators.ragas import RagasEvaluator
from ragas.metrics import Faithfulness, ResponseRelevancy, AnswerAccuracy
from src.models import EmbeddingModelConfig, RerankingModelConfig, RewriterModelConfig, LLMConfig
from src.pipelines.rag_pipelines.rag_pipeline import get_rag_pipeline

def get_rag_evaluation_pipeline(
    base_indexing_store: InMemoryDocumentStore,
    embedding_model_config: EmbeddingModelConfig,
    reranking_model_config: RerankingModelConfig,
    rewriting_model_config: RewriterModelConfig,
    llm_config: LLMConfig,
) -> Pipeline:
    pipeline = get_rag_pipeline(llm_config, base_indexing_store, embedding_model_config, reranking_model_config, rewriting_model_config)

    metrics = [
        AnswerAccuracy(),
        ResponseRelevancy(),
        Faithfulness(),
    ]

    ragas_evaluator = RagasEvaluator(ragas_metrics=metrics)

    pipeline.add_component("evaluator", ragas_evaluator)
    if reranking_model_config != None:
        pipeline.connect("reranker", "evaluator.documents")
    elif embedding_model_config != None:
        pipeline.connect("retriever", "evaluator.documents")
    pipeline.connect("generator.replies", "evaluator.response")

    return pipeline