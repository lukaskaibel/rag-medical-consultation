#!/usr/bin/env python
# coding: utf-8

# # Indexing
# 
# This part indexes documents (creates embedding and stores them in a Haystack DocumentStore). It uses different indexing variant s.a. normal indexing and contextualized indexing (used later for contextual RAG evaluation).

# In[ ]:


# import os
# # Setting temp dir to /srv/data directory, as it otherwise fills up the home directory too much
# # Just comment out on machines that are not "Goober"
# os.environ["TMPDIR"] = "/srv/data/tmp"
# os.makedirs("/srv/data/tmp", exist_ok=True)

# %pip install haystack-ai==2.16.1
# %pip install nltk
# %pip install openai==1.99.7
# %pip install pandas
# %pip install sentence-transformers
# %pip install hf_xet
# %pip install ollama-haystack==2.4.2
# %pip install tqdm # For Progress Bar


# In[2]:


import os
from utils.markdown_utils import for_each_markdown_file
import pandas as pd
from haystack.document_stores.in_memory import InMemoryDocumentStore
from tqdm import tqdm
tqdm.pandas()

from pipelines.indexing_pipelines.base_indexing_pipeline import get_base_indexing_pipeline
from pipelines.indexing_pipelines.context_indexing_pipeline import get_context_indexing_pipeline
from models import EmbeddingModelConfig, EmbeddingModelProvider, LLMProvider, LLMConfig

import logging
from utils.pickle_utils import for_each_pickle_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S'
)

# silence haystack’s pipeline logs
logging.getLogger("haystack").setLevel(logging.WARNING)
logging.getLogger("haystack.core.pipeline").setLevel(logging.WARNING)

# if you see similar spam from transformers, ragas, etc.
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("ragas").setLevel(logging.WARNING)


# In[ ]:


from config.secret import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./model-assets/sentence-transformers"
os.environ["LLM_CONTEXT_SIZE"] = "40000"


embedding_model_name = "Qwen/Qwen3-Embedding-8B"
embedding_model_provider = EmbeddingModelProvider.SENTENCE_TRANSFORMER

contexualization_model_name = "gpt-5-mini-2025-08-07"
contexualization_model_provider = LLMProvider.OPEN_AI


# Check maximum passage length so make sure every chunk fits in the context.

# In[ ]:


df = pd.read_pickle("data/preprocessed_documents/docs_passage_1_0.pkl")

doc_contents = df["document"].map(lambda doc: doc.content)
doc_contents.apply(len).max()


# ## Base Indexing

# ## Context Indexing

# In[ ]:


def context_indexing(filename, df):
    documents = df["document"].tolist()

    context_indexing_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    context_indexing_pipeline = get_context_indexing_pipeline(
        context_indexing_store, 
        embedding_model_config=EmbeddingModelConfig(name=embedding_model_name, provider=EmbeddingModelProvider.SENTENCE_TRANSFORMER), 
        contextualizer_model_config=LLMConfig(name=contexualization_model_name, provider=contexualization_model_provider)
    )

    def index_with_context(filename, bytes):
        documents_from_file = [document for document in documents if document.meta["title"] == filename]
        file_content = bytes.decode("utf-8")
        context_indexing_pipeline.run({
            "contextualiser": {
                "context": file_content,
                "documents": documents_from_file
            }
        })

    for_each_markdown_file("data/md_files", index_with_context)

    filepath = f"data/document_stores/{embedding_model_name}/context/{contexualization_model_name}"
    os.makedirs(filepath, exist_ok=True)
    clean_name = os.path.splitext(os.path.basename(filename))[0]
    context_indexing_store.save_to_disk(f"{filepath}/{clean_name}_indexing_store.json")

for_each_pickle_file("data/preprocessed_documents", context_indexing)


# #### Index already contextualized documents
# 
# Use this if only the embedding model changes, but you don't want to re-contextualize all the documents

# In[ ]:


# from utils.json_utils import for_each_document_store
# from pipelines.indexing_pipelines.base_indexing_pipeline import get_base_indexing_pipeline
# from models import EmbeddingModelProvider, EmbeddingModelConfig
# from haystack.document_stores.in_memory import InMemoryDocumentStore
# from config.secret import OPENAI_API_KEY
# import os

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# contextualizer_model_name = "gemma3:27b"

# old_embedding_model_name = "Qwen/Qwen3-Embedding-8B"

# new_embedding_model_name = "text-embedding-3-large"
# new_embedding_model_provider = EmbeddingModelProvider.OPENAI

# def embeddings_for_contextualized_chunks(filename, old_store):
#     contextualized_documents = old_store.filter_documents()

#     context_indexing_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
#     base_indexing_pipeline = get_base_indexing_pipeline(
#         context_indexing_store, 
#         EmbeddingModelConfig(new_embedding_model_name, new_embedding_model_provider)
#     )
#     base_indexing_pipeline.run({
#         "embedder": { 
#             "documents": contextualized_documents
#         },
#     })

#     filepath = f"data/document_stores/{new_embedding_model_name}/context/{contextualizer_model_name}"
#     os.makedirs(filepath, exist_ok=True)
#     context_indexing_store.save_to_disk(f"{filepath}/{filename}")


# for_each_document_store(f"data/document_stores/{old_embedding_model_name}/context/{contextualizer_model_name}", embeddings_for_contextualized_chunks)


# In[ ]:






# In[ ]:





# ## Validation

# In[ ]:


# base_documents = base_indexing_store.filter_documents()
base_documents = InMemoryDocumentStore.load_from_disk("data/document_stores/Qwen/Qwen3-Embedding-4B/base/docs_passage_1_0_indexing_store.json").filter_documents()
contextualized_documents = InMemoryDocumentStore.load_from_disk("data/document_stores/Qwen/Qwen3-Embedding-4B/context/gpt-4.1-mini/docs_passage_1_0_indexing_store.json").filter_documents()


# In[ ]:


import numpy as np

# Get content lengths
base_lengths = [len(doc.content) for doc in base_documents]
contextualized_lengths = [len(doc.content) for doc in contextualized_documents]

# Compute stats
base_mean = np.mean(base_lengths)
base_std = np.std(base_lengths)

contextualized_mean = np.mean(contextualized_lengths)
contextualized_std = np.std(contextualized_lengths)

print(f"Base documents - Mean: {base_mean:.2f} chars, Std Dev: {base_std:.2f}")
print(f"Contextualized documents - Mean: {contextualized_mean:.2f} chars, Std Dev: {contextualized_std:.2f}")


# In[ ]:


[document.content for document in contextualized_documents]


# In[ ]:




