import os
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, OpenAIDocumentEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.utils import ComponentDevice, Device
from pipelines.components.contextualiser import Contextualiser
from config.prompt import CONTEXTUALISER_PROMPT
from models import EmbeddingModelConfig, EmbeddingModelProvider, LLMConfig

def get_context_indexing_pipeline(document_store: InMemoryDocumentStore, embedding_model_config: EmbeddingModelConfig, contextualizer_model_config: LLMConfig):
    pipeline = Pipeline()

    contextualiser = Contextualiser(
        prompt=CONTEXTUALISER_PROMPT,
        llm_model_config=contextualizer_model_config,
        generation_kwargs={ "temperature": 0.0, "num_ctx": int(os.environ["LLM_CONTEXT_SIZE"]) },
        keep_alive=-1
    )
    
    if embedding_model_config.provider == EmbeddingModelProvider.SENTENCE_TRANSFORMER:
        document_embedder = SentenceTransformersDocumentEmbedder(
            model=embedding_model_config.name,
            prefix="",
            device=ComponentDevice.from_single(Device.gpu(id=1)),
            model_kwargs={"torch_dtype": "float16"}
        )
        document_embedder.warm_up()
    elif embedding_model_config.provider == EmbeddingModelProvider.OPENAI:
        document_embedder = OpenAIDocumentEmbedder(
            model=embedding_model_config.name
        )

    document_writer = DocumentWriter(document_store=document_store)

    pipeline.add_component(instance=contextualiser, name="contextualiser")
    pipeline.add_component(instance=document_embedder, name="embedder")
    pipeline.add_component(instance=document_writer, name="writer")

    pipeline.connect("contextualiser.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline