from typing import Optional
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from models import EmbeddingModelConfig, RerankingModelConfig, RewriterModelConfig, LLMConfig, LLMProvider
from pipelines.retrieval_pipelines.base_retrieval_pipeline import get_base_retrieval_pipeline

def get_rag_pipeline(
    llm_config: LLMConfig,
    document_store: Optional[InMemoryDocumentStore] = None,
    embedding_model_config: Optional[EmbeddingModelConfig] = None,
    reranking_model_config: Optional[RerankingModelConfig] = None,
    rewriter_model_config: Optional[RewriterModelConfig] = None,
) -> Pipeline:
    if embedding_model_config != None:
        pipeline = get_base_retrieval_pipeline(document_store, embedding_model_config, reranking_model_config, rewriter_model_config)
    else:
        pipeline = Pipeline()

    if llm_config.provider == LLMProvider.OLLAMA:
        generation_kwargs = {}
        if llm_config.temperature != None:
            generation_kwargs["temperature"] = llm_config.temperature
        if llm_config.context_length != None:
            generation_kwargs["num_ctx"] = llm_config.context_length

        generator = OllamaChatGenerator(
            model=llm_config.name,
            generation_kwargs=generation_kwargs, 
            keep_alive=-1
        )
    elif llm_config.provider == LLMProvider.OPEN_AI:
        generation_kwargs = {}
        if llm_config.temperature != None:
            generation_kwargs["temperature"] = llm_config.temperature
        if llm_config.context_length != None:
            generation_kwargs["max_tokens"] = llm_config.context_length

        generator = OpenAIChatGenerator(
            model=llm_config.name,
            generation_kwargs=generation_kwargs,
        )
    else:
        raise ValueError("LLM Provider not supported")
    
    prompt_builder = ChatPromptBuilder(required_variables="*", variables=["documents"])
    
    pipeline.add_component("generator", generator)
    pipeline.add_component("prompt_builder", prompt_builder)

    if reranking_model_config == None and embedding_model_config != None:
        pipeline.connect("retriever.documents", "prompt_builder.documents")
    elif reranking_model_config != None:
        pipeline.connect("reranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.messages")

    return pipeline