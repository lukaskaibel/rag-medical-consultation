from enum import Enum
from typing import Optional

class LLMProvider(Enum):
    OLLAMA = 0
    OPEN_AI = 1

class LLMConfig:
    def __init__(
        self, 
        name: str, 
        provider: LLMProvider,
        temperature: Optional[float] = None,
        context_length: Optional[int] = None,
    ):
        self.name = name
        self.provider = provider
        self.temperature = temperature
        self.context_length = context_length

    name: str
    provider: LLMProvider
    temperature: Optional[float]
    context_length: Optional[int]
