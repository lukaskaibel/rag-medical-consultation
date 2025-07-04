from enum import Enum

class EmbeddingModelProvider(Enum):
    OLLAMA = 0
    SENTENCE_TRANSFORMER = 1

class EmbeddingModelConfig:
    def __init__(self, name: str, provider: EmbeddingModelProvider):
        self.name = name
        self.provider = provider

    name: str
    provider: EmbeddingModelProvider