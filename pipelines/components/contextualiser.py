from typing import Any, Dict, List, Optional, Union
from haystack import component
from haystack.components.generators import OpenAIGenerator
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.dataclasses import Document
from haystack.components.builders import PromptBuilder
import os

@component
class Contextualiser():
    def __init__(
        self,
        prompt: str,
        model: str = "orca-mini",
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[Union[float, str]] = None,
    ):
        self.prompt = prompt
        self.generation_kwargs = generation_kwargs or {}
        self.url = url
        self.model = model
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

        if os.environ["LLM_PROVIDER"] == "ollama":
            generator = OllamaGenerator(
                model=self.model,
                url=self.url,
                generation_kwargs=self.generation_kwargs,
                keep_alive=self.keep_alive,
            )
        elif os.environ["LLM_PROVIDER"] == "openai":
            generator = OpenAIGenerator(
                model=self.model,
                generation_kwargs={
                    "temperature": self.generation_kwargs["temperature"],
                    "max_tokens": self.generation_kwargs["num_ctx"]
                }
            )

        for document in documents:
            context_prompt = context_prompt_builder.run(template_variables={"context": context, "document": document})["prompt"]
            result = generator.run(context_prompt)
            document.content = result["replies"][0] # + "\n\n" + document.content

        return { "documents": documents }