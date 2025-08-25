#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import os

# # Setting temp dir to /srv/data directory, as it otherwise fills up the home directory too much
# # Just comment out on machines that are not "Goober"
# os.environ["TMPDIR"] = "/srv/data/tmp"
# os.makedirs("/srv/data/tmp", exist_ok=True)


# In[5]:


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


# In[6]:


import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from pipelines.evaluation.rag_eval_pipeline import get_rag_evaluation_pipeline
from models import EmbeddingModelConfig, EmbeddingModelProvider, LLMConfig, LLMProvider, RerankingModelConfig, RerankingModelProvider, RewriterModelConfig
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage
from config.prompt import PROMPT_TEMPLATE


# In[7]:


from config.secret import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model-assets/sentence-transformers"


# In[8]:


test_configs = [
    {
        "name": "Open Source RAG (Top-k=5)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
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
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "retrieval-top-k": 5,
    },
    {
        "name": "Open Source RAG (Top-k=10)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
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
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "retrieval-top-k": 10,
    },
    {
        "name": "Open Source RAG (Top-k=20)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
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
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "retrieval-top-k": 20,
    },
    {
        "name": "Open Source Long Context",
        "embedding_model": None,
        "reranking_model": None,
        "contextualizer_model": None,
        "rewriter_model": None,
        "llm": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
    },
    {
        "name": "Closed Source RAG (Top-k=5)",
        "embedding_model": EmbeddingModelConfig(name="Qwen/Qwen3-Embedding-8B", provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
        "reranking_model": RerankingModelConfig(name="Qwen/Qwen3-Reranker-8B", provider=RerankingModelProvider.HUGGING_FACE),
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
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
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
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
        "contextualizer_model": LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gemma3:27b", provider=LLMProvider.OLLAMA, context_length=40000),
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


# In[9]:


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


now = datetime.now()
start_index = 272
NUMBER_OF_QUESTIONS_IN_EVAL = 600

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
        df_shuffled.iloc[start_index:].iterrows(),
        total=len(df_shuffled) - start_index,
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
                        "top_k": 20,
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
                        "top_k": test_config["retrieval-top-k"],
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
            answer = result["answer_builder"]["answers"][0].data
            print(answer)
            df.at[index, f"{test_config['name']}_answer_accuracy"] = answer_accuracy
            df.at[index, f"{test_config['name']}_response_relevancy"] = response_relevancy
            df.at[index, f"{test_config['name']}_faithfulness"] = faithfulness
            df.at[index, f"{test_config['name']}_answer"] = answer

        # save_path = f"results/generation/{now.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        save_path = f"results/generation/2025-08-15_13-43-52.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_pickle(save_path)

# df = pd.read_pickle("data/qa_with_docs_flat/question_answers_docs_word_100_20_dataset_flat.pkl")
df = pd.read_pickle("results/generation/2025-08-15_13-43-52.pkl")

run_retrieval_eval("question_answers_docs_word_100_20_dataset_flat.pkl", df)


# In[1]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_pickle("results/generation/2025-08-15_13-43-52.pkl")

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
        vmin=0,
        ax=ax
    )
    ax.set_title(config)
    if ax != axes[0]:
        ax.set_ylabel("")  # remove repeated ylabel
    else:
        ax.set_ylabel("Metric")

plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
plt.show()
fig.savefig('figures/generation_eval_broad.pgf')


# In[2]:


len(df[pd.notna(df["Open Source RAG (Top-k=5)_answer_accuracy"])])


# In[ ]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- 1) Load ---
df = pd.read_pickle("results/generation/2025-08-15_13-43-52.pkl")

# --- 2) Settings ---
METRICS = ["answer_accuracy", "response_relevancy", "faithfulness"]
METRIC_LABELS = {
    "answer_accuracy": "Answer Accuracy",
    "response_relevancy": "Response Relevancy",
    "faithfulness": "Faithfulness",
}
CONFIGS = ["Top-k=5", "Top-k=10", "Top-k=20", "Long Context"]
SOURCES = ["Open Source", "Closed Source"]

# --- 3) Regex parse ---
pat = re.compile(
    r"^(Open|Closed) Source (?:RAG \(Top-k=(\d+)\)|Long Context)_(answer_accuracy|response_relevancy|faithfulness)$",
    re.IGNORECASE,
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

# --- 4) Compute mean scores ---
def unwrap_single_element(val):
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return val[0]
    return val

rows = []
for _, row in parsed_df.iterrows():
    series = df[row["column"]].apply(unwrap_single_element)
    vals = pd.to_numeric(series, errors="coerce")
    mean_val = vals.mean(skipna=True)
    rows.append({
        "metric": row["metric"],
        "config": row["config"],
        "source": row["source"],
        "score": float(mean_val) if pd.notna(mean_val) else np.nan,
    })

tidy = pd.DataFrame(rows).dropna(subset=["score"])
tidy["metric"] = pd.Categorical(tidy["metric"], categories=METRICS, ordered=True)
tidy["config"] = pd.Categorical(tidy["config"], categories=CONFIGS, ordered=True)
tidy["source"] = pd.Categorical(tidy["source"], categories=SOURCES, ordered=True)

# --- 5) Plot: metrics side by side horizontally ---
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, len(METRICS), figsize=(5*len(METRICS), 4), sharey=True)

palette = sns.color_palette(n_colors=len(METRICS))
metric_colors = {m: palette[i] for i, m in enumerate(METRICS)}
source_styles = {"Open Source": "-", "Closed Source": "--"}

for ax, metric in zip(axes, METRICS):
    sub = tidy[tidy["metric"] == metric].copy()
    if sub.empty:
        ax.axis("off")
        continue

    for src in SOURCES:
        sub_src = sub[sub["source"] == src].sort_values("config")
        if sub_src.empty:
            continue
        ax.plot(
            sub_src["config"].astype(str),
            sub_src["score"],
            marker="o",
            linewidth=2,
            color=metric_colors[metric],
            linestyle=source_styles[src],
        )

    ax.set_ylim(0.0, 1.0)
    ax.set_title(METRIC_LABELS.get(metric, metric))

# Build legends (shared)
source_handles = [
    Line2D([0],[0], color="black", lw=2, linestyle=ls, label=src)
    for src, ls in source_styles.items()
]

axes[-1].legend(handles=source_handles, title="Source (line style)",
                bbox_to_anchor=(1.02, 0.45), loc="upper left", borderaxespad=0.)

plt.tight_layout()
plt.show()
fig.savefig("figures/generation_eval_metrics_horizontal.pgf")


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# assumes `df` is already loaded (your big pickle with a "documents" column)

# --- unwrap helper ---
def unwrap_single_element(val):
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return val[0]
    return val

# --- 1) Build tidy dataset again (like before) ---
rows = []
for col in df.columns:
    if not isinstance(col, str):
        continue
    if "_answer_accuracy" in col or "_response_relevancy" in col or "_faithfulness" in col:
        # detect meta info from col name
        if "Top-k=10" not in col:
            continue  # only keep top-k=10
        source = "Open Source" if col.lower().startswith("open") else "Closed Source"
        if "answer_accuracy" in col:
            metric = "answer_accuracy"
        elif "response_relevancy" in col:
            metric = "response_relevancy"
        elif "faithfulness" in col:
            metric = "faithfulness"
        else:
            continue

        # unwrap values
        series = df[col].apply(unwrap_single_element)
        vals = pd.to_numeric(series, errors="coerce")

        # get number of documents for each row (same index as df)
        num_docs = df["documents"].apply(lambda x: len(x) if isinstance(x, (list, tuple)) else np.nan)

        tmp = pd.DataFrame({
            "score": vals,
            "metric": metric,
            "source": source,
            "num_docs": num_docs
        })
        rows.append(tmp)

tidy_docs = pd.concat(rows, ignore_index=True).dropna(subset=["score", "num_docs"])

# --- 2) Aggregate: mean score per (metric, source, num_docs) ---
agg = (
    tidy_docs
    .groupby(["metric", "source", "num_docs"], as_index=False)["score"]
    .mean()
)

# --- 3) Plot ---
METRIC_LABELS = {
    "answer_accuracy": "Answer Accuracy",
    "response_relevancy": "Response Relevancy",
    "faithfulness": "Faithfulness",
}
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))

palette = sns.color_palette(n_colors=agg["metric"].nunique())
metric_colors = {m: palette[i] for i, m in enumerate(agg["metric"].unique())}
source_styles = {"Open Source": "-", "Closed Source": "--"}

for metric in agg["metric"].unique():
    sub_m = agg[agg["metric"] == metric]
    for src in agg["source"].unique():
        sub_src = sub_m[sub_m["source"] == src].sort_values("num_docs")
        if sub_src.empty:
            continue
        plt.plot(
            sub_src["num_docs"],
            sub_src["score"],
            marker="o",
            linewidth=2,
            color=metric_colors[metric],
            linestyle=source_styles[src],
        )

plt.xlabel("Number of Documents")
plt.ylabel("Score")
plt.ylim(0.0, 1.0)
plt.title("Top-k=10: Metrics vs Number of Documents")

# legends
metric_handles = [
    Line2D([0], [0], color=metric_colors[m], lw=3, label=METRIC_LABELS.get(m, m))
    for m in agg["metric"].unique()
]
source_handles = [
    Line2D([0], [0], color="black", lw=2, linestyle=ls, label=src)
    for src, ls in source_styles.items()
]

leg1 = plt.legend(handles=metric_handles, title="Metric (color)",
                  bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.)
plt.gca().add_artist(leg1)
plt.legend(handles=source_handles, title="Source (line style)",
           bbox_to_anchor=(1.02, 0.5), loc="upper left", borderaxespad=0.)

plt.tight_layout()
plt.show()


# In[ ]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Load ---
df = pd.read_pickle("results/generation/2025-08-15_13-43-52.pkl")

# --- Settings ---
METRICS = ["answer_accuracy", "response_relevancy", "faithfulness"]
METRIC_LABELS = {
    "answer_accuracy": "Answer Accuracy",
    "response_relevancy": "Response Relevancy",
    "faithfulness": "Faithfulness",
}
CONFIGS = ["Top-k=5", "Top-k=10", "Top-k=20", "Long Context"]
SOURCES = ["Open Source", "Closed Source"]

# How much to smooth (as a fraction of the x-range).
# Increase for smoother lines, decrease for more detail.
SMOOTH_SPAN_FRAC = 0.35

# --- Regex parse (same as before) ---
pat = re.compile(
    r"^(Open|Closed) Source (?:RAG \(Top-k=(\d+)\)|Long Context)_(answer_accuracy|response_relevancy|faithfulness)$",
    re.IGNORECASE,
)

parsed = []
for col in df.columns:
    if not isinstance(col, str):
        continue
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

def unwrap_single_element(val):
    if isinstance(val, (list, tuple)) and len(val) == 1:
        return val[0]
    return val

# number of docs per row
num_docs_series = df["documents"].apply(
    lambda x: len(x) if isinstance(x, (list, tuple)) else (int(x) if pd.notna(x) and np.isscalar(x) else np.nan)
)

# Build tidy row-level data
rows = []
for _, r in parsed_df.iterrows():
    series = df[r["column"]].apply(unwrap_single_element)
    vals = pd.to_numeric(series, errors="coerce")
    rows.append(pd.DataFrame({
        "score": vals,
        "metric": r["metric"],
        "source": r["source"],
        "config": r["config"],
        "num_docs": num_docs_series,
    }))

tidy = pd.concat(rows, ignore_index=True).dropna(subset=["score", "num_docs"])
tidy["metric"] = pd.Categorical(tidy["metric"], categories=METRICS, ordered=True)
tidy["config"] = pd.Categorical(tidy["config"], categories=CONFIGS, ordered=True)
tidy["source"] = pd.Categorical(tidy["source"], categories=SOURCES, ordered=True)

# Aggregate to (metric, source, config, num_docs) with mean score and COUNT = weight
agg = (
    tidy
    .groupby(["metric", "source", "config", "num_docs"], as_index=False)
    .agg(score_mean=("score", "mean"), n=("score", "size"))
)

# --- Weighted kernel smoother (Gaussian) ---
def weighted_gaussian_smooth(x, y, w, x_grid=None, span_frac=0.35):
    """
    Returns (x_grid, y_smoothed) where y_smoothed[i] is a weighted average
    of y using Gaussian weights around x_grid[i]. Weights are multiplied by w.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    if x_grid is None:
        x_grid = np.unique(x)
    else:
        x_grid = np.asarray(x_grid, dtype=float)

    rng = x.max() - x.min() if len(x) > 1 else 1.0
    h = max(1.0, span_frac * rng)  # bandwidth in "num_docs" units

    y_hat = np.empty_like(x_grid, dtype=float)
    for i, x0 in enumerate(x_grid):
        # Gaussian kernel centered at x0
        kern = np.exp(-0.5 * ((x - x0) / h) ** 2)
        ww = w * kern
        denom = ww.sum()
        y_hat[i] = (ww @ y) / denom if denom > 0 else np.nan
    return x_grid, y_hat

# --- Plot: one subplot per config, horizontally; color=metric, linestyle=source ---
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, len(CONFIGS), figsize=(5 * len(CONFIGS), 4), sharey=True)

palette = sns.color_palette(n_colors=len(METRICS))
metric_colors = {m: palette[i] for i, m in enumerate(METRICS)}
source_styles = {"Open Source": "-", "Closed Source": "--"}

for ax, config in zip(axes, CONFIGS):
    sub_c = agg[agg["config"] == config]
    if sub_c.empty:
        ax.axis("off")
        ax.set_title(config)
        continue

    # grid of x values (sorted unique num_docs) for smooth curves
    x_grid = np.sort(sub_c["num_docs"].unique())

    for metric in METRICS:
        for src in SOURCES:
            sub = sub_c[(sub_c["metric"] == metric) & (sub_c["source"] == src)].sort_values("num_docs")
            if sub.empty:
                continue

            # smoothed, weighted by counts n
            x_s, y_s = weighted_gaussian_smooth(
                sub["num_docs"].values,
                sub["score_mean"].values,
                sub["n"].values,
                x_grid=x_grid,
                span_frac=SMOOTH_SPAN_FRAC
            )

            # plot smoothed line
            ax.plot(
                x_s, y_s,
                linewidth=2,
                color=metric_colors[metric],
                linestyle=source_styles[src],
            )

            # (optional) show original points, size ~ weight
            ax.scatter(
                sub["num_docs"], sub["score_mean"],
                s=10 + 2.0 * sub["n"],           # emphasize high-count points
                color=metric_colors[metric],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
            )

    ax.set_title(config)
    ax.set_xlabel("Number of Documents")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="x", visible=False)  # remove vertical grid lines
    if ax is axes[0]:
        ax.set_ylabel("Score")
    else:
        ax.set_ylabel("")

# Shared legends
metric_handles = [Line2D([0],[0], color=metric_colors[m], lw=3, label=METRIC_LABELS[m]) for m in METRICS]
source_handles = [Line2D([0],[0], color="black", lw=2, linestyle=ls, label=src) for src, ls in source_styles.items()]

leg1 = axes[-1].legend(handles=metric_handles, title="Metric (color)",
                       bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.)
axes[-1].add_artist(leg1)
axes[-1].legend(handles=source_handles, title="Source (line style)",
                bbox_to_anchor=(1.02, 0.45), loc="upper left", borderaxespad=0.)

plt.tight_layout()
plt.show()
# fig.savefig("figures/generation_docs_vs_score_weighted_smooth.pgf")


# In[ ]:




