#!/usr/bin/env python
# coding: utf-8

# # Retrieval Evaluation

# In[1]:


# %pip install "haystack-ai>=2.15.0rc1"
# %pip install ragas-haystack
# %pip install nltk
# %pip install openai
# %pip install pandas
# %pip install ragas-haystack
# %pip install "sentence-transformers>=3.0.0"
# %pip install hf_xet
# %pip install "ollama-haystack==2.4.2"
# %pip install tqdm # For Progress Bar
# %pip install einops


# In[ ]:


import os
from haystack.document_stores.in_memory import InMemoryDocumentStore
import importlib
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()
import logging
from utils.pickle_utils import for_each_pickle_file
from models import EmbeddingModelConfig, EmbeddingModelProvider, RerankingModelConfig, RerankingModelProvider, LLMConfig, LLMProvider
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("haystack").setLevel(logging.WARNING)

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model-assets/sentence-transformers"
os.environ["HF_HUB_CACHE"] = "./model-assets/hugging-face"


# In[ ]:


from config.secret import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LLM_CONTEXT_SIZE"] = "8192"

open_embedding_model = "Qwen/Qwen3-Embedding-4B"
closed_embedding_model = "text-embedding-3-large"

open_contextualizer_model = "gemma3:12b"
closed_contextualizer_model = "gpt-4.1-mini"

NUMBER_OF_QUESTIONS_IN_EVAL = 600


# In[ ]:


test_configs = [
    {
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
        "retrieval-top-k": 10,
    },
    {
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gpt-4.1-mini", provider=LLMProvider.OLLAMA),
        "retrieval-top-k": 10,
    },
    {
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
        "retrieval-top-k": 20,
    },
    {
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gpt-4.1-mini", provider=LLMProvider.OLLAMA),
        "retrieval-top-k": 40,
    },
]


# In[ ]:


from pipelines.evaluation.base_retrieval_eval_pipeline import get_base_retrieval_eval_pipeline
from pipelines.evaluation.hybrid_retrieval_eval_pipeline import get_hybrid_retrieval_eval_pipeline
from pipelines.evaluation.context__retrieval_eval_pipeline import get_context_retrieval_eval_pipeline
from pipelines.evaluation.random_retrieval_eval_pipeline import get_random_retrieval_eval_pipeline
from pipelines.evaluation.noreranking_retrieval_eval_pipeline import get_base_retrieval_eval_pipeline
from models import EmbeddingModelConfig, EmbeddingModelProvider, RerankingModelConfig, RerankingModelProvider

def get_test_cases(splitting_strategy: str):
    both_open_indexing_store = InMemoryDocumentStore.load_from_disk(f"data/document_stores/{open_embedding_model}/context/{open_contextualizer_model}/{splitting_strategy}_indexing_store.json")
    open_embedding_indexing_store = InMemoryDocumentStore.load_from_disk(f"data/document_stores/{open_embedding_model}/context/{closed_contextualizer_model}/{splitting_strategy}_indexing_store.json")
    open_contextualizer_indexing_store = InMemoryDocumentStore.load_from_disk(f"data/document_stores/{closed_embedding_model}/context/{open_contextualizer_model}/{splitting_strategy}_indexing_store.json")
    both_closed_indexing_store = InMemoryDocumentStore.load_from_disk(f"data/document_stores/{closed_embedding_model}/context/{closed_contextualizer_model}/{splitting_strategy}_indexing_store.json")

    # TODO: Need to check if reranker boosts performance first, because if so i also need to test open vs. closed reranker
    test_cases = [
        {
            "name": "Both Open Source",
            "pipeline": get_base_retrieval_eval_pipeline(both_open_indexing_store, EmbeddingModelConfig(name=open_embedding_model, provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER)),
        },
        {
            "name": "Basic RAG",
            "pipeline": get_base_retrieval_eval_pipeline(open_embedding_indexing_store, EmbeddingModelConfig(name=open_embedding_model, provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER)),
        },
        {
            "name": "Hybrid RAG",
            "pipeline": get_hybrid_retrieval_eval_pipeline(base_indexing_store),
        },
        {
            "name": "Contextual RAG",
            "pipeline": get_context_retrieval_eval_pipeline(context_indexing_store, EmbeddingModelConfig(name=embedder, provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER), RerankingModelConfig(name=reranker, provider=RerankingModelProvider.HUGGING_FACE)),
        },
    ]

    return test_cases


# In[16]:


now = datetime.now()

def run_retrieval_eval(filename, df):
    import config.prompt
    importlib.reload(config.prompt)

    import re

    match = re.search(r"answers_(.*?)_dataset", filename)
    if match:
        splitting_strategy = match.group(1)
    else:
        splitting_strategy = None

    # 1) Filter out the null‐question rows
    df_nonnull = df[df["question"].notnull()]

    df_shuffled = df_nonnull.sample(n=NUMBER_OF_QUESTIONS_IN_EVAL, random_state=42).reset_index(drop=True)

    for top_k in TOP_K_VALUES:
        for index, row in tqdm(
            df_shuffled.iterrows(),
            total=len(df_shuffled),
            desc="Processing rows",
            unit="row"
        ):
            relevant_documents = row["documents"]
            question = row["question"]

            test_cases = get_test_cases(splitting_strategy)

            for test_case in test_cases:
                pipeline = test_case["pipeline"]
                request_payload = {
                    "retriever": {
                        "top_k": top_k,
                    },
                    "map_evaluator": {
                        "ground_truth_documents": [relevant_documents],
                    },
                    "mrr_evaluator": {
                        "ground_truth_documents": [relevant_documents],
                    },
                    "recall_evaluator": {
                        "ground_truth_documents": [relevant_documents],
                    }
                }
                if "query_embedder" in pipeline.graph.nodes:
                    request_payload["query_embedder"] = {
                        "text": question,
                    }
                if "reranker" in pipeline.graph.nodes:
                    request_payload["reranker"] = {
                        "query": question,
                        "top_k": top_k,
                    }
                if "bm25_retriever" in pipeline.graph.nodes:
                    request_payload["bm25_retriever"] = {
                        "query": question,
                        "top_k": top_k,
                    }
                result = pipeline.run(request_payload)

                map_score = result.get("map_evaluator", {}).get("score", {})
                mrr_score = result.get("mrr_evaluator", {}).get("score", {})
                recall_score = result.get("recall_evaluator", {}).get("score", {})

                df.at[index, f"{test_case['name']}_map"] = map_score
                df.at[index, f"{test_case['name']}_mrr"] = mrr_score
                df.at[index, f"{test_case['name']}_recall"] = recall_score

        embedding_model_name = os.environ['EMBEDDING_MODEL_NAME'].replace('/', '')
        save_path = f"results/retrieval/{now.strftime('%Y-%m-%d_%H-%M-%S')}/{embedding_model_name}/{splitting_strategy}/topk_{top_k}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_pickle(save_path)

for_each_pickle_file("data/qa_with_docs_flat", run_retrieval_eval)


# In[ ]:




