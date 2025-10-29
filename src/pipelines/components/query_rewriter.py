from typing import Any, Dict, Optional, Union, List
from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
import os
from models import LLMConfig, LLMProvider

@component
class QueryRewriter():
    def __init__(
        self,
        prompt: str,
        llm_model_config: LLMConfig,
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[float, str]] = None,
    ):
        self.prompt = prompt
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.llm_model_config = llm_model_config
        self.keep_alive = keep_alive
    
    @component.output_types(query=str)
    def run(self, query: str, previous_messages: List[ChatMessage]):
        if len(query) == 0:
            msg = "No query provided"
            raise ValueError(msg)
        
        prompt_builder = ChatPromptBuilder(
            template=[ChatMessage.from_system(text=self.prompt)] + previous_messages + [ChatMessage.from_user(text=query)]
        )

        if self.llm_model_config.provider == LLMProvider.OLLAMA:
            generator = OllamaChatGenerator(
                model=self.llm_model_config.name,
                url=self.url,
                generation_kwargs=self.generation_kwargs,
                keep_alive=self.keep_alive,
            )
        elif self.llm_model_config.provider == LLMProvider.OPEN_AI:
            generator = OpenAIChatGenerator(
                model=self.llm_model_config.name,
            )

        prompt = prompt_builder.run()["prompt"]
        result = generator.run(prompt)
        rewritten_query = result["replies"][0].text
        return { "query": rewritten_query }