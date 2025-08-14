#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os

# # Setting temp dir to /srv/data directory, as it otherwise fills up the home directory too much
# # Just comment out on machines that are not "Goober"
# os.environ["TMPDIR"] = "/srv/data/tmp"
# os.makedirs("/srv/data/tmp", exist_ok=True)


# In[2]:


# %pip install haystack-ai
# %pip install ragas-haystack
# %pip install nltk
# %pip install markdown-it-py
# %pip install mdit_plain
# %pip install openai
# %pip install pandas
# %pip install ragas-haystack
# %pip install sentence-transformers
# %pip install hf_xet
# %pip install ollama-haystack==2.4.2
# %pip install tqdm # For Progress Bar


# In[3]:


import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from pipelines.evaluation.rag_eval_pipeline import get_rag_evaluation_pipeline
from models import EmbeddingModelConfig, EmbeddingModelProvider, LLMConfig, LLMProvider, RerankingModelConfig, RerankingModelProvider, RewriterModelConfig
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage, Document
from config.prompt import PROMPT_TEMPLATE


# In[4]:


from config.secret import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

os.environ["EMBEDDING_MODEL_NAME"] = "Linq-AI-Research/Linq-Embed-Mistral"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model-assets/sentence-transformers"


# In[5]:


test_configs = [
    {
        "name": "Open Source RAG (Top-k=5)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that rewrites a user's question for a RAG system. 
            Keep the original meaning and language. Strip out filler words and irrelevant context, preserve all named entities and technical terms, and enrich phrasing with clearer structure or synonyms. 
            If prior messages are provided, include only the essential details from them to ensure the question is fully self-contained. 
            Output only the rewritten question—no additional text.

            Example 1
            Original: “Um, like, what medication should I take for my morning headaches? I've been getting them almost every day.”
            Rewritten: Which medication is most effective for treating daily morning headaches?

            Example 2
            Original: “Hey, I'm confused—what's the normal blood pressure range for adults? I've seen different numbers online.”
            Rewritten: What is the normal adult blood pressure range?
            """,
        ),
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "retrieval-top-k": 5,
    },
    {
        "name": "Open Source RAG (Top-k=10)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that rewrites a user's question for a RAG system. 
            Keep the original meaning and language. Strip out filler words and irrelevant context, preserve all named entities and technical terms, and enrich phrasing with clearer structure or synonyms. 
            If prior messages are provided, include only the essential details from them to ensure the question is fully self-contained. 
            Output only the rewritten question—no additional text.

            Example 1
            Original: “Um, like, what medication should I take for my morning headaches? I've been getting them almost every day.”
            Rewritten: Which medication is most effective for treating daily morning headaches?

            Example 2
            Original: “Hey, I'm confused—what's the normal blood pressure range for adults? I've seen different numbers online.”
            Rewritten: What is the normal adult blood pressure range?
            """,
        ),
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "retrieval-top-k": 10,
    },
    {
        "name": "Open Source RAG (Top-k=20)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that rewrites a user's question for a RAG system. 
            Keep the original meaning and language. Strip out filler words and irrelevant context, preserve all named entities and technical terms, and enrich phrasing with clearer structure or synonyms. 
            If prior messages are provided, include only the essential details from them to ensure the question is fully self-contained. 
            Output only the rewritten question—no additional text.

            Example 1
            Original: “Um, like, what medication should I take for my morning headaches? I've been getting them almost every day.”
            Rewritten: Which medication is most effective for treating daily morning headaches?

            Example 2
            Original: “Hey, I'm confused—what's the normal blood pressure range for adults? I've seen different numbers online.”
            Rewritten: What is the normal adult blood pressure range?
            """,
        ),
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "retrieval-top-k": 20,
    },
    {
        "name": "Open Source Long Context",
        "embedding_model": None,
        "reranking_model": None,
        "contextualizer_model": None,
        "rewriter_model": None,
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
    },
    {
        "name": "Closed Source RAG (Top-k=5)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that rewrites a user's question for a RAG system. 
            Keep the original meaning and language. Strip out filler words and irrelevant context, preserve all named entities and technical terms, and enrich phrasing with clearer structure or synonyms. 
            If prior messages are provided, include only the essential details from them to ensure the question is fully self-contained. 
            Output only the rewritten question—no additional text.

            Example 1
            Original: “Um, like, what medication should I take for my morning headaches? I've been getting them almost every day.”
            Rewritten: Which medication is most effective for treating daily morning headaches?

            Example 2
            Original: “Hey, I'm confused—what's the normal blood pressure range for adults? I've seen different numbers online.”
            Rewritten: What is the normal adult blood pressure range?
            """,
        ),
        "llm": LLMConfig(name="gpt-5-mini-2025-08-07", provider=LLMProvider.OPEN_AI),
        "retrieval-top-k": 5,
    },
    {
        "name": "Closed Source RAG (Top-k=10)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that rewrites a user's question for a RAG system. 
            Keep the original meaning and language. Strip out filler words and irrelevant context, preserve all named entities and technical terms, and enrich phrasing with clearer structure or synonyms. 
            If prior messages are provided, include only the essential details from them to ensure the question is fully self-contained. 
            Output only the rewritten question—no additional text.

            Example 1
            Original: “Um, like, what medication should I take for my morning headaches? I've been getting them almost every day.”
            Rewritten: Which medication is most effective for treating daily morning headaches?

            Example 2
            Original: “Hey, I'm confused—what's the normal blood pressure range for adults? I've seen different numbers online.”
            Rewritten: What is the normal adult blood pressure range?
            """,
        ),
        "llm": LLMConfig(name="gpt-5-mini-2025-08-07", provider=LLMProvider.OPEN_AI),
        "retrieval-top-k": 10,
    },
    {
        "name": "Closed Source RAG (Top-k=20)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that rewrites a user's question for a RAG system. 
            Keep the original meaning and language. Strip out filler words and irrelevant context, preserve all named entities and technical terms, and enrich phrasing with clearer structure or synonyms. 
            If prior messages are provided, include only the essential details from them to ensure the question is fully self-contained. 
            Output only the rewritten question—no additional text.

            Example 1
            Original: “Um, like, what medication should I take for my morning headaches? I've been getting them almost every day.”
            Rewritten: Which medication is most effective for treating daily morning headaches?

            Example 2
            Original: “Hey, I'm confused—what's the normal blood pressure range for adults? I've seen different numbers online.”
            Rewritten: What is the normal adult blood pressure range?
            """,
        ),
        "llm": LLMConfig(name="gpt-5-mini-2025-08-07", provider=LLMProvider.OPEN_AI),
        "retrieval-top-k": 20,
    },
    {
        "name": "Closed Source Long Context",
        "embedding_model": None,
        "reranking_model": None,
        "contextualizer_model": None,
        "rewriter_model": None,
        "llm": LLMConfig(name="gpt-5-mini-2025-08-07", provider=LLMProvider.OPEN_AI),
    },
]


# In[6]:


import uuid
from typing import List
from haystack.dataclasses import Document
from utils.markdown_utils import for_each_markdown_file

def get_full_documents() -> List[str]:
    full_documents = []
    def add_to_docs_list(filename: str, bytes):
        file_content = bytes.decode("utf-8")
        document = Document(id=str(uuid.uuid4()), content=file_content)
        full_documents.append(document)
    for_each_markdown_file("data/md_files", add_to_docs_list)
    return full_documents


# ## Generate RAG Responses

# In[ ]:


from pipelines.rag_pipelines.rag_pipeline import get_rag_pipeline
import json

now = datetime.now()
NUMBER_OF_QUESTIONS_IN_EVAL = 10

def run_retrieval_eval(filename, df):
    import re

    match = re.search(r"answers_(.*?)_dataset", filename)
    if match:
        splitting_strategy = match.group(1)
    else:
        splitting_strategy = None

    # 1) Filter out the null‐question rows
    df_nonnull = df[df["question"].notnull()]

    df_shuffled = df_nonnull.sample(n=NUMBER_OF_QUESTIONS_IN_EVAL, random_state=42)
    full_documents = get_full_documents()

    for index, row in tqdm(
        df_shuffled.iterrows(),
        total=len(df_shuffled),
        desc="Processing rows",
        unit="row"
    ):
        question = row["question"]
        reference = row["groundTruth"]
        previous_messages = [
            ChatMessage.from_user(msg) if idx == 0
            else ChatMessage.from_assistant(msg)
            for idx, msg in enumerate(row["prev_messages"])
        ]

        for test_config in test_configs:
            if test_config["embedding_model"] == None:
                pipeline = get_rag_evaluation_pipeline(
                    llm_config=test_config["llm"],
                    base_indexing_store=None,
                    embedding_model_config=None,
                    reranking_model_config=None,
                    rewriting_model_config=None,
                )
                request_payload = {
                    "prompt_builder": {
                        "template": [ChatMessage.from_system(PROMPT_TEMPLATE)] + previous_messages + [ChatMessage.from_user(question)],
                        "documents": full_documents,
                    },
                    "answer_builder": {
                        "query": question,
                    },
                    "evaluator": {
                        "query": question,
                        "reference": reference,
                        "documents": [document.content for document in full_documents],
                        "reference_contexts": [document.content for document in full_documents],
                    }
                }
            else:
                base_indexing_store = InMemoryDocumentStore.load_from_disk(f"data/document_stores/{test_config['embedding_model'].name}/context/{test_config['contextualizer_model'].name}/{splitting_strategy}_indexing_store.json")
                pipeline = get_rag_evaluation_pipeline(
                    llm_config=test_config["llm"],
                    base_indexing_store=base_indexing_store,
                    embedding_model_config=test_config["embedding_model"],
                    reranking_model_config=test_config["reranking_model"],
                    rewriting_model_config=test_config["rewriter_model"],
                )
                request_payload = {
                    "retriever": {
                        "top_k": test_config["retrieval-top-k"],
                    },
                    "prompt_builder": {
                        "template": [ChatMessage.from_system(PROMPT_TEMPLATE)] + previous_messages + [ChatMessage.from_user(question)]
                    },
                    "answer_builder": {
                        "query": question,
                    },
                    "evaluator": {
                        "query": question,
                        "reference": reference,
                        "reference_contexts": [document.content for document in row["documents"]]
                    }
                }
                if "reranker" in pipeline.graph.nodes:
                    request_payload["reranker"] = {
                        "query": question,
                    }

                if "rewriter" in pipeline.graph.nodes:
                    request_payload["rewriter"] = {
                        "query": question,
                        "previous_messages": previous_messages,
                    }
                else:
                    request_payload["query_embedder"] = {
                        "text": question,
                    }
            result = pipeline.run(request_payload)

            answer_accuracy = result.get("evaluator", {}).get("result", {})["nv_accuracy"]
            response_relevancy = result.get("evaluator", {}).get("result", {})["answer_relevancy"]
            faithfulness = result.get("evaluator", {}).get("result", {})["faithfulness"]
            df.at[index, f"{test_config['name']}_answer_accuracy"] = answer_accuracy
            df.at[index, f"{test_config['name']}_response_relevancy"] = response_relevancy
            df.at[index, f"{test_config['name']}_faithfulness"] = faithfulness

        save_path = f"results/generation/{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_pickle(save_path)

df = pd.read_pickle("data/qa_with_docs_flat/question_answers_docs_word_100_20_dataset_flat.pkl")

run_retrieval_eval("question_answers_docs_word_100_20_dataset_flat.pkl", df)


# In[13]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle("results/generation/2025-08-14_10-26-20.pkl")

# --- 1) Settings ---
METRICS = [
    "answer_accuracy",
    "response_relevancy",
    "faithfulness",
]
METRIC_LABELS = {
    "answer_accuracy": "Answer Accuracy",
    "response_relevancy": "Response Relevancy",
    "faithfulness": "Faithfulness",
}
CONFIGS = ["Top-k=5", "Top-k=10", "Top-k=20", "Long Context"]
SOURCES = ["Open Source", "Closed Source"]

# --- 2) Flexible regex ---
pat = re.compile(
    r"^(Open|Closed) Source (?:RAG \(Top-k=(\d+)\)|Long Context)_(answer_accuracy|response_relevancy|faithfulness)$",
    re.IGNORECASE
)

parsed = []
for col in df.columns:
    m = pat.match(col.strip())
    if not m:
        continue
    source = f"{m.group(1).title()} Source"
    k = m.group(2)
    config = f"Top-k={k}" if k is not None else "Long Context"
    metric = m.group(3).lower()
    if config in CONFIGS and metric in METRICS and source in SOURCES:
        parsed.append((col, source, config, metric))

parsed_df = pd.DataFrame(parsed, columns=["column", "source", "config", "metric"])

# --- 3) Prepare matrices for each config ---
heatmaps = {}
for config in CONFIGS:
    sub = parsed_df[parsed_df["config"] == config]
    mat = np.full((len(METRICS), len(SOURCES)), np.nan, dtype=float)
    def unwrap_single_element(val):
        """If val is a list/tuple with one element, return that element, else val."""
        if isinstance(val, (list, tuple)) and len(val) == 1:
            return val[0]
        return val

    # --- Inside the loop where we compute mean_val ---
    for _, row in sub.iterrows():
        # Unwrap list values first
        series = df[row["column"]].apply(unwrap_single_element)
        # Convert to numeric
        vals = pd.to_numeric(series, errors="coerce")
        mean_val = vals.mean(skipna=True)
        i = METRICS.index(row["metric"])
        j = SOURCES.index(row["source"])
        mat[i, j] = mean_val
    heatmaps[config] = pd.DataFrame(
        mat,
        index=[METRIC_LABELS[m] for m in METRICS],
        columns=SOURCES
    )

# --- 4) Plot horizontally with Seaborn style ---
fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5*len(CONFIGS), 5), sharey=True)

if len(CONFIGS) == 1:
    axes = [axes]

for ax, config in zip(axes, CONFIGS):
    summary_df = heatmaps[config]
    sns.heatmap(
        summary_df,
        annot=True,
        cmap="Blues",
        fmt=".3f",
        cbar=False,
        linewidths=0.5,
        vmax=1.0,
        ax=ax
    )
    ax.set_title(config)
    if ax != axes[0]:
        ax.set_ylabel("")  # remove repeated ylabel
    else:
        ax.set_ylabel("Metric")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
plt.show()


# In[ ]:


df[pd.notna(df["Open Source RAG (Top-k=5)_answer_correctness"])]

