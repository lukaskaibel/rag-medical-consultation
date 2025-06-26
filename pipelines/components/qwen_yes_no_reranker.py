from copy import copy
from typing import Dict, List, Optional

import torch
from haystack import Document, component
from haystack.utils import ComponentDevice
from transformers import AutoModelForCausalLM, AutoTokenizer


@component
class QwenYesNoReranker:
    """
    Ranks documents by using a causal LLM to predict 'yes' or 'no' next-token probabilities.
    It formats each (query, document) pair into a prompt, then computes relevance as the probability of 'yes'.
    """
    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-4B",
        device: Optional[ComponentDevice] = None,
        top_k: int = 10,
        instruction: Optional[str] = None,
        padding_side: str = "left",
        batch_size: int = 8,
    ):
        self.model_name = model
        self.device = ComponentDevice.resolve_device(device)
        self.top_k = top_k
        self.instruction = instruction or (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        self.padding_side = padding_side
        self.batch_size = batch_size
        # will be set in warm_up
        self.tokenizer = None
        self.model = None
        # tokens
        self.token_yes_id = None
        self.token_no_id = None
        # prompt tokens
        self.prefix_tokens = None
        self.suffix_tokens = None
        self.max_length = None

    def warm_up(self):
        """Load tokenizer and model."""
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side=self.padding_side
        )
        # ensure pad_token is defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = self.padding_side
        # model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16 if self.device.to_torch_str().startswith("cuda") else None
        ).to(self.device.to_torch_str())
        self.model.eval()
        # yes/no token ids
        self.token_yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.tokenizer.convert_tokens_to_ids("no")
        # build prompt prefix and suffix
        prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
        )
        suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        # tokenize prompt tokens
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        # compute max length
        self.max_length = self.model.config.max_position_embeddings

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> Dict[str, List[Document]]:
        """
        Ranks the given documents by relevance to the query.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("QwenYesNoReranker not warmed up. Call warm_up() first.")
        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        # build prompts
        pairs = []
        for doc in documents:
            prompt = (
                f"<Instruct>: {self.instruction}\n"
                f"<Query>: {query}\n"
                f"<Document>: {doc.content}"
            )
            pairs.append(prompt)
        # tokenize and batch
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            # tokenize without padding
            inputs = self.tokenizer(
                batch,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
                max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )
            # add prefix and suffix
            input_ids = []
            for seq in inputs["input_ids"]:
                seq = self.prefix_tokens + seq.tolist() + self.suffix_tokens
                input_ids.append(seq)
            # pad to batch
            padded = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            padded = {k: v.to(self.device.to_torch_str()) for k, v in padded.items()}
            # forward
            with torch.no_grad():
                logits = self.model(**padded).logits[:, -1, :]
            # extract yes/no logits
            yes_logits = logits[:, self.token_yes_id]
            no_logits = logits[:, self.token_no_id]
            two_logits = torch.stack([no_logits, yes_logits], dim=1)
            probs = torch.nn.functional.softmax(two_logits, dim=1)[:, 1]
            all_scores.extend(probs.cpu().tolist())
        # attach scores
        scored = []
        for doc, score in zip(documents, all_scores):
            d = copy(doc)
            d.score = score
            scored.append(d)
        # sort and top_k
        scored = sorted(scored, key=lambda d: d.score, reverse=True)
        return {"documents": scored[:top_k]}