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
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import logging
from models import EmbeddingModelConfig, EmbeddingModelProvider, LLMConfig, LLMProvider, RerankingModelConfig, RerankingModelProvider, RewriterModelConfig
from pipelines.evaluation.base_retrieval_eval_pipeline import get_base_retrieval_eval_pipeline
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger("haystack").setLevel(logging.WARNING)

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model-assets/sentence-transformers"
os.environ["HF_HUB_CACHE"] = "./model-assets/hugging-face"


# In[3]:


from config.secret import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LLM_CONTEXT_SIZE"] = "8192"

open_embedding_model = "Qwen/Qwen3-Embedding-4B"
closed_embedding_model = "text-embedding-3-large"

open_contextualizer_model = "gemma3:12b"
closed_contextualizer_model = "gpt-4.1-mini"

NUMBER_OF_QUESTIONS_IN_EVAL = 600
TOP_K = 10


# In[4]:


test_configs = [
    {
        "name": "Open Source",
        "embedding_model": EmbeddingModelConfig(name=open_embedding_model, provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
    },
    {
        "name": "Open Source with Rewriting",
        "embedding_model": EmbeddingModelConfig(name=open_embedding_model, provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER),
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
    },
    {
        "name": "Closed Source",
        "embedding_model": EmbeddingModelConfig(name=closed_embedding_model, provider=EmbeddingModelProvider.OPENAI),
    },
    {
        "name": "Closed Source with Rewriting",
        "embedding_model": EmbeddingModelConfig(name=closed_embedding_model, provider=EmbeddingModelProvider.OPENAI),
        "rewriter_model": RewriterModelConfig(
            LLMConfig(name="gpt-4.1-mini", provider=LLMProvider.OPEN_AI),
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
    },
]


# In[ ]:


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
        for test_config in test_configs:
            relevant_documents = row["documents"]
            question = row["question"]

            index_store = InMemoryDocumentStore.load_from_disk(f"data/document_stores/{test_config['embedding_model'].name}/base/{splitting_strategy}_indexing_store.json")
            pipeline = get_base_retrieval_eval_pipeline(
                index_store, 
                test_config["embedding_model"], 
                None, 
                test_config.get("rewriter_model") if test_config.get("rewriter_model") is not None else None
            )
            request_payload = {
                "retriever": {
                    "top_k": TOP_K,
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
            result = pipeline.run(request_payload)

            map_score = result.get("map_evaluator", {}).get("score", {})
            mrr_score = result.get("mrr_evaluator", {}).get("score", {})
            recall_score = result.get("recall_evaluator", {}).get("score", {})

            df.at[index, f"{test_config['name']}_map"] = map_score
            df.at[index, f"{test_config['name']}_mrr"] = mrr_score
            df.at[index, f"{test_config['name']}_recall"] = recall_score

        save_path = f"results/retrieval/open_closed_base/{now.strftime('%Y-%m-%d_%H-%M-%S')}/open_closed_result.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_pickle(save_path)

run_retrieval_eval("data/qa_with_docs_flat/question_answers_docs_word_100_20_dataset_flat.pkl", pd.read_pickle("data/qa_with_docs_flat/question_answers_docs_word_100_20_dataset_flat.pkl"))


# ## Analysis

# In[ ]:


import pandas as pd

df = pd.read_pickle("results/retrieval/open_closed/2025-07-21_14-03-03/open_closed_result_topk-10.pkl")
df


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Your DataFrame is assumed to be `df`
metric_prefixes = ['Qwen/Qwen3-Embedding-4B_gemma3:12b',
                   'Qwen/Qwen3-Embedding-4B_gpt-4.1-mini',
                   'text-embedding-3-large_gemma3:12b',
                   'text-embedding-3-large_gpt-4.1-mini']
metrics = ['map', 'mrr', 'recall']

# Build data dictionary
data = {}
for prefix in metric_prefixes:
    values = []
    for metric in metrics:
        col_name = f"{prefix}_{metric}"
        avg_value = df[col_name].mean()
        values.append(avg_value)
    data[prefix] = values

# Create DataFrame for plotting
metrics_df = pd.DataFrame(data, index=metrics)

# Find the minimum and maximum to scale y-axis
y_min = metrics_df.min().min()

# Plotting
ax = metrics_df.T.plot(kind='bar', figsize=(12, 6))
plt.title("Average MAP, MRR, and Recall for Each Configuration")
plt.ylabel("Score")
plt.xlabel("Embedding Configuration")
plt.xticks(rotation=45, ha='right')
plt.ylim(y_min * 0.9, 1.0)  # add some padding
plt.legend(title="Metric")
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming `df` is your DataFrame
metric_prefixes = ['Qwen/Qwen3-Embedding-4B_gemma3:12b',
                   'Qwen/Qwen3-Embedding-4B_gpt-4.1-mini',
                   'text-embedding-3-large_gemma3:12b',
                   'text-embedding-3-large_gpt-4.1-mini']
metrics = ['map', 'mrr', 'recall']

# Compute the average value for each metric/config combo
data = {}
for prefix in metric_prefixes:
    values = []
    for metric in metrics:
        col_name = f"{prefix}_{metric}"
        avg_value = df[col_name].mean()
        values.append(avg_value)
    data[prefix] = values

# Convert to DataFrame: rows = metrics, columns = configurations
heatmap_df = pd.DataFrame(data, index=metrics)

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Heatmap of Average MAP, MRR, and Recall")
plt.xlabel("Embedding Configuration")
plt.ylabel("Metric")
plt.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp

# ─── 0) assume `df` is loaded with columns:
#      'variant',
#      'Qwen/Qwen3-Embedding-4B_gemma3:12b_map',  …_mrr, …_recall,
#      'Qwen/Qwen3-Embedding-4B_gpt-4.1-mini_map', …,
#      'text-embedding-3-large_gemma3:12b_map', …,
#      'text-embedding-3-large_gpt-4.1-mini_map', …
# ────────────────────────────────────────────────────────────────────────────────

BASE    = 'Qwen/Qwen3-Embedding-4B_gemma3:12b'
CONFIGS = [
    'Qwen/Qwen3-Embedding-4B_gpt-4.1-mini',
    'text-embedding-3-large_gemma3:12b',
    'text-embedding-3-large_gpt-4.1-mini',
]
METRICS = ['map', 'mrr', 'recall']

# 1) compute per-row diffs for each config & metric
for cfg in CONFIGS:
    for m in METRICS:
        df[f'diff_{cfg}_{m}'] = (
            df[f'{cfg}_{m}'].astype(float)
          - df[f'{BASE}_{m}'].astype(float)
        )

# 2) aggregate by variant, metric & config, run t-test
records = []
for variant, grp in df.groupby('variant'):
    for cfg in CONFIGS:
        for m in METRICS:
            col = f'diff_{cfg}_{m}'
            vals = grp[col].dropna().astype(float)
            mean_diff = vals.mean()
            _, p = ttest_1samp(vals, 0)
            records.append({
                'variant':     variant,
                'config':      cfg,
                'metric':      m.upper(),
                'mean_diff':   mean_diff,
                'significant': p < 0.05
            })

summary = pd.DataFrame(records)

# 3) prepare for plotting: combine config+metric into one hue key
summary['key'] = summary['config'] + ' ' + summary['metric']

# 4) build a 3×3 palette: one sequential palette per config
pal = {}
# Blues for CONFIGS[0], Greens for CONFIGS[1], Reds for CONFIGS[2]
seqs = [sns.color_palette("Blues",3)[::-1],
        sns.color_palette("Greens",3)[::-1],
        sns.color_palette("Reds",3)[::-1]]
for cfg, seq in zip(CONFIGS, seqs):
    for col_idx, m in enumerate(['MAP','MRR','RECALL']):
        pal[f"{cfg} {m}"] = seq[col_idx]

hue_order = [f"{cfg} {m}" for cfg in CONFIGS for m in ['MAP','MRR','RECALL']]
variants  = sorted(summary['variant'].unique())

# 5) draw the barplot
plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=summary,
    y='variant',
    x='mean_diff',
    hue='key',
    hue_order=hue_order,
    palette=pal,
    order=variants,
    ci=None
)

# zero line
ax.axvline(0, color='gray', linewidth=1)

# annotate significance stars
sig_map = {
    (r.variant, r['config'], r['metric']): r.significant
    for r in summary.itertuples()
}
for hue_idx, container in enumerate(ax.containers):
    key = hue_order[hue_idx]            # e.g. "text-embedding-3-large_gemma3:12b RECALL"
    cfg, metr = key.rsplit(' ',1)
    for bar_idx, bar in enumerate(container.patches):
        variant = variants[bar_idx]
        if not sig_map.get((variant, cfg, metr), False):
            continue
        x = bar.get_width()
        y = bar.get_y() + bar.get_height()/2
        dx = 3 if x >= 0 else -3
        ax.annotate(
            "✱",
            xy=(x,y),
            xytext=(dx,0),
            textcoords="offset points",
            ha="left" if x>=0 else "right",
            va="center",
            fontsize=6
        )

ax.set_title("Mean MAP/MRR/Recall Difference vs. Qwen/Qwen3-Embedding-4B_gemma3:12b\n(by Variant)")
ax.set_xlabel("Mean Score Difference")
ax.set_ylabel("Variant")
ax.legend(title="", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()


# In[ ]:




