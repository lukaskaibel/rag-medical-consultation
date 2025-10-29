from models import LLMConfig

class RewriterModelConfig:
    def __init__(self, llm_config: LLMConfig, prompt: str):
        self.llm_config = llm_config
        self.prompt = prompt

    llm_config: LLMConfig
    prompt: str