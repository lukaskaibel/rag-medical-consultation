from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.utils import ComponentDevice, Device
import os

def get_base_rag_pipeline(document_store: InMemoryDocumentStore):
    pipeline = Pipeline()

    query_embedder = SentenceTransformersTextEmbedder(
        model=os.environ["EMBEDDING_MODEL_NAME"],
        prefix="",
        device=ComponentDevice.from_single(Device.gpu(id=0)),
        model_kwargs={"torch_dtype": "float16"}
    )
    query_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    prompt_builder = ChatPromptBuilder(required_variables="*", variables=["documents"])

    ollama_generator = OllamaChatGenerator(
        model=os.environ["LLM_NAME"],
        generation_kwargs={ "temperature": 0.0, "num_ctx": int(os.environ["LLM_CONTEXT_SIZE"]) }, 
        keep_alive=-1
    )

    pipeline.add_component(instance=query_embedder, name="query_embedder")
    pipeline.add_component(instance=retriever, name="retriever")
    pipeline.add_component(instance=prompt_builder, name="prompt_builder")
    pipeline.add_component(instance=ollama_generator, name="llm")

    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")

    return pipeline
