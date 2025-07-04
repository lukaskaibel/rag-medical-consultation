from enum import Enum

class RerankingModelProvider(Enum):
    HUGGING_FACE = 0

class RerankingModelConfig:
    def __init__(self, name: str, provider: RerankingModelProvider):
        self.name = name
        self.provider = provider

    name: str
    provider: RerankingModelProvider