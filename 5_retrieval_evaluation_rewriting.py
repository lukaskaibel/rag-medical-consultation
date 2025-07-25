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


# In[2]:


import os
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage
import importlib
from datetime import datetime
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
import logging
from pipelines.evaluation.base_retrieval_eval_pipeline import get_base_retrieval_eval_pipeline
from models import EmbeddingModelConfig, EmbeddingModelProvider, LLMConfig, LLMProvider, RewriterModelConfig
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("haystack").setLevel(logging.WARNING)

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model-assets/sentence-transformers"
os.environ["HF_HUB_CACHE"] = "./model-assets/hugging-face"
os.environ["LLM_CONTEXT_SIZE"] = "8192"


# In[3]:


FINAL_TOP_K = 10 # Number of documents returned at the end of pipeline
NUMBER_OF_QUESTIONS_IN_EVAL = 600


# In[4]:


test_configs = [
    {
        "name": "No Rewriting",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
        "rewriter_model": None,
        "retrieval-top-k": 10,
    },
    {
        "name": "Rewriting Zero Shot",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that rewrites a user's question for a RAG system. 
            Keep the original meaning and language. Strip out filler words and irrelevant context, preserve all named entities and technical terms, and enrich phrasing with clearer structure or synonyms. 
            If prior messages are provided, include only the essential details from them to ensure the question is fully self-contained. 
            Output only the rewritten question—no additional text.
            """     
        ),
        "retrieval-top-k": 10,
    },
    {
        "name": "Rewriting Few Shot",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
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
            """     
        ),
        "retrieval-top-k": 10,
    },
    {
        "name": "HyDE Zero Shot",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that, given a user’s medical question and any prior conversation context, produces a single concise paragraph addressing that question. 
            Keep the original meaning and language; strip out filler words and irrelevant context; preserve all named entities and technical terms; enrich phrasing with clearer structure or synonyms; and incorporate necessary context from previous messages only when essential. 
            Output only the paragraph—no additional text.
            """     
        ),
        "retrieval-top-k": 10,
    },
    {
        "name": "HyDE Few Shot",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-4B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": None,
        "contextualizer_model": LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:12b", provider=LLMProvider.OLLAMA),
            """You are a helpful assistant that, given a user’s medical question and any prior conversation context, produces a single concise paragraph addressing that question. 
            Keep the original meaning and language; strip out filler words and irrelevant context; preserve all named entities and technical terms; enrich phrasing with clearer structure or synonyms; and incorporate necessary context from previous messages only when essential. 
            Output only the paragraph—no additional text.

            Example 1
            Original: “Um, like, what medication should I take for my morning headaches? I’ve been getting them almost every day.”
            Paragraph: Daily morning headaches warrant evaluation of underlying etiologies such as tension-type or migraine; first-line management typically includes NSAIDs (e.g., ibuprofen 400 mg with breakfast) or acetaminophen if NSAIDs are contraindicated, with migraine-specific options like sumatriptan 50 mg at headache onset and preventive therapy (e.g., topiramate 25 mg daily) for frequent episodes, alongside nonpharmacologic measures such as optimizing sleep hygiene and stress reduction.

            Example 2
            Original: “Hey, I'm confused—what’s the normal blood pressure range for adults? I’ve seen different numbers online.”
            Paragraph: Normal adult blood pressure is defined as systolic < 120 mm Hg and diastolic < 80 mm Hg, while elevated levels (120–129/< 80) and stage 1 hypertension (130–139/80–89) reflect updated American Heart Association criteria, contrasted with European guidelines that consider values < 130/85 mm Hg as normal, informing clinical decisions on lifestyle modification and pharmacotherapy thresholds.
            """     
        ),
        "retrieval-top-k": 10,
    },

]


# In[6]:


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

    for index, row in tqdm(
        df_shuffled.iterrows(),
        total=len(df_shuffled),
        desc="Processing rows",
        unit="row"
    ):
        relevant_documents = row["documents"]
        question = row["question"]

        for test_config in test_configs:
            base_indexing_store = InMemoryDocumentStore.load_from_disk(f"data/document_stores/{test_config['embedding_model'].name}/context/{test_config['contextualizer_model'].name}/{splitting_strategy}_indexing_store.json")
            pipeline = get_base_retrieval_eval_pipeline(
                base_indexing_store=base_indexing_store,
                embedding_model_config=test_config["embedding_model"],
                reranking_model_config=test_config["reranking_model"],
                rewriting_model_config=test_config["rewriter_model"],
                is_hyde="HyDE" in test_config["name"],
            )
            request_payload = {
                "retriever": {
                    "top_k": test_config["retrieval-top-k"],
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

            previous_messages = [
                ChatMessage.from_user(msg) if idx == 0
                else ChatMessage.from_assistant(msg)
                for idx, msg in enumerate(row["prev_messages"])
            ]
            if "rewriter" in pipeline.graph.nodes:
                request_payload["rewriter"] = {
                    "query": question,
                    "previous_messages": previous_messages,
                }
            else:
                request_payload["query_embedder"] = {
                    "text": question,
                }
            if "reranker" in pipeline.graph.nodes:
                request_payload["reranker"] = {
                    "query": question,
                    "top_k": FINAL_TOP_K,
                }
            result = pipeline.run(request_payload)

            map_score = result.get("map_evaluator", {}).get("score", {})
            mrr_score = result.get("mrr_evaluator", {}).get("score", {})
            recall_score = result.get("recall_evaluator", {}).get("score", {})

            df.at[index, f"{test_config['name']}_map"] = map_score
            df.at[index, f"{test_config['name']}_mrr"] = mrr_score
            df.at[index, f"{test_config['name']}_recall"] = recall_score

    save_path = f"results/retrieval/rewriter/{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_pickle(save_path)

df = pd.read_pickle("data/qa_with_docs_flat/question_answers_docs_word_50_10_dataset_flat.pkl")
run_retrieval_eval("question_answers_docs_word_50_10_dataset_flat.pkl", df)


# In[ ]:


df


# In[ ]:




