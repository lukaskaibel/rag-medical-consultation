from typing import Any, Dict, List, Optional, Union
from haystack import component
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.dataclasses import Document
from haystack.components.builders import PromptBuilder
import os
from models import LLMConfig, LLMProvider

@component
class Contextualiser():
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
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], context: str):
        if len(context) == 0:
            msg = "No context provided"
            raise ValueError(msg)
        
        context_prompt_builder = PromptBuilder(
            template=self.prompt, 
            required_variables="*", 
            variables=["context", "document"]
        )

        if self.llm_model_config.provider == LLMProvider.OLLAMA:
            generator = OllamaGenerator(
                model=self.llm_model_config.name,
                url=self.url,
                generation_kwargs=self.generation_kwargs,
                keep_alive=self.keep_alive,
            )
        elif self.llm_model_config.provider == LLMProvider.OPEN_AI:
            generator = OpenAIGenerator(
                model=self.llm_model_config.name,
            )

        for document in documents:
            context_prompt = context_prompt_builder.run(template_variables={"context": context, "document": document})["prompt"]
            result = generator.run(context_prompt)
            document.content = result["replies"][0] + "\n\n" + document.content

        return { "documents": documents }