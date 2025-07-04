from enum import Enum

class LLMProvider(Enum):
    OLLAMA = 0
    OPEN_AI = 1

class LLMConfig:
    def __init__(self, name: str, provider: LLMProvider):
        self.name = name
        self.provider = provider

    name: str
    provider: LLMProvider