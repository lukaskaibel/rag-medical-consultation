# Retrieval-Augmented Generation (RAG) for Medical Consultation

> Official research repository accompanying the master’s thesis *“Retrieval-Augmented Generation for Medical Consultation”* (Freie Universität Berlin, 2025).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![Framework: Haystack](https://img.shields.io/badge/framework-Haystack-lightgrey.svg)]()
[![LLM: OpenAI & Qwen](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Qwen-green.svg)]()

**Topics:** _rag_ · _retrieval-augmented-generation_ · _llm_ · _medical-ai_ · _haystack_ · _research_ · _thesis_ · _german-language_

This repository provides a **reproducible research framework** for evaluating Retrieval-Augmented Generation (RAG) systems in medical consultation settings.  
It was developed as part of the master’s thesis *“Retrieval-Augmented Generation for Medical Consultation”* (Freie Universität Berlin, 2025) and contributes to the **PIA (Patient Information Assistant)** project, which investigates how conversational AI can support patients before and after pre-treatment consultations.

## Why this project

- End-to-end RAG evaluation, from document preprocessing and QA dataset generation to retrieval metrics and generation quality analysis.
- Side-by-side comparisons of open vs. closed components, different RAG variants (Contextual and Hybrid RAG), and ablations for reranking and query rewriting.
- Reproducible runs: centralized dependencies, environment variables via .env, and deterministic sampling where applicable.

## Key capabilities

- Document preprocessing from markdown sources to a Haystack-compatible store
- Synthetic QA dataset generation and question-document matching
- Retrieval evaluation at scale with MAP, MRR, and Recall
- Optional components: rerankers, query contextualization, and rewriting
- Generation evaluation and analysis (via notebooks), compatible with Haystack and RAG evaluation tooling

## Repository layout

- `data/`
  - `md_files/` – domain corpus (German obstetrics, markdown)
  - `document_stores/` – persisted Haystack stores for experiments
  - `preprocessed_documents/`, `qa_with_docs/`, `qa_with_docs_flat/` – cached artifacts
- `model-assets/`
  - `sentence-transformers/`, `hugging-face/` – local cache roots for embeddings and other models
- `notebooks/` (archival / optional) – original notebook locations; current notebooks live at repo root
- Project root notebooks
  - `1_document_preprocessing.ipynb`
  - `2_qa_dataset_generation.ipynb`
  - `3_question_document_matching.ipynb`
  - `4_indexing.ipynb`
  - Retrieval evaluation series:
    - `5_retrieval_evaluation.ipynb` (baseline) — consider keeping as 5 or renaming to `5.0_...` for lexicographic order
    - `5.1_retrieval_evaluation_line_passage.ipynb`
    - `5.2_retrieval_evaluation_reranker.ipynb`
    - `5.3_retrieval_evaluation_rewriting.ipynb`
    - `5.4_retrieval_evaluation_closed_vs_open.ipynb`
  - `6_generation_evaluation.ipynb`
  - `6_generation_analysis.ipynb`
  - `7_analysis.ipynb`
- `src/`
  - `pipelines/` – Haystack pipelines for retrieval evaluation, etc.
  - `models/` – model configuration enums and helpers
  - `config/` – prompts and experiment configuration
  - `utils/` – utilities
- `helper/` – scripts (e.g., running notebooks in tmux)
- `results/`
  - `retrieval/` and `generation/` – timestamped experiment outputs
- `requirements.txt` – single source of dependencies

## Setup

Prerequisites
- Python 3.10+ recommended
- Linux (tested) with bash
- Optionally: GPU for large local models; or use remote providers as configured

Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Environment variables via `.env`
- Copy `.env.example` to `.env` and fill in values.
- Notebooks load environment variables automatically in their first cell using `python-dotenv`.

Common variables
- `OPENAI_API_KEY` – for OpenAI components when used
- `SENTENCE_TRANSFORMERS_HOME=./model-assets/sentence-transformers` – local cache root
- `HF_HUB_CACHE=./model-assets/hugging-face` – local Hugging Face cache
- `TMPDIR` – optional temp dir override
- `LLM_CONTEXT_SIZE` – context window configuration for supported providers

## Data and domain

- Source corpus: German-language medical content focused on obstetrics, stored in `data/md_files/`.
- Preprocessing: Use `1_document_preprocessing.ipynb` to clean, segment, and normalize content.
- Indexing: `4_indexing.ipynb` builds Haystack document stores for different configurations.

## Models and providers

- Embeddings: open-source (e.g., `Qwen/Qwen3-Embedding-8B` via Sentence Transformers) and closed-source (e.g., `text-embedding-3-large`).
- Optional components:
  - Contextualization LLMs (e.g., local via Ollama such as `gemma3:27b`, and closed via OpenAI as configured in notebooks)
  - Rewriting LLMs for query rewriting
  - Rerankers (Qwen reranker families are cached under `model-assets/`)

Caches and assets
- By default, model caches are directed to `./model-assets/` via environment variables to keep runs reproducible and offline-friendly.

## Reproducible Workflows (Notebooks)

All notebooks automatically load environment variables from `.env` and share dependencies via `requirements.txt` — no inline `%pip install` required.

### 🧩 Suggested Execution Order

| Step | Notebook | Description |
|------|-----------|-------------|
| 1 | **`1_document_preprocessing.ipynb`** | Cleans and chunks source markdown documents into retrievable text segments. |
| 2 | **`2_qa_dataset_generation.ipynb`** | Generates diverse question variations with reference sentences and ground-truth answers. |
| 3 | **`3_question_document_matching.ipynb`** | Maps questions to document chunks for retrieval evaluation. |
| 4 | **`4_indexing.ipynb`** | Builds vector stores with chosen embedding models; adds contextualized chunks for Document Context RAG. |
| 5 | **`5_retrieval_evaluation.ipynb`** | Baseline retrieval evaluation across Top-k and chunk-size settings for Basic, Hybrid, and Document Context RAG. |
| 5.1 | **`5.1_retrieval_evaluation_line_passage.ipynb`** | Compares syntactic chunking (by line/paragraph) with fixed-length word chunking. |
| 5.2 | **`5.2_retrieval_evaluation_reranker.ipynb`** | Ablation study evaluating the impact of reranking. |
| 5.3 | **`5.3_retrieval_evaluation_rewriting.ipynb`** | Tests query rewriting and HyDE techniques for retrieval improvement. |
| 5.4 | **`5.4_retrieval_evaluation_closed_vs_open.ipynb`** | Compares open-source vs. closed-source models for embeddings, contextualization, and rewriting. |
| 6 | **`6_generation_evaluation.ipynb`** | Evaluates generation quality under different context lengths using the fixed retrieval setup. |
| 7 | **`7_analysis.ipynb`** | Produces plots and comparative analyses from retrieval and generation results. |

---

💡 **Tip:**  
You can execute notebooks interactively in Jupyter or run them in batch mode via  
`helper/run_notebook_in_tmux.sh` for automated, reproducible experiments.

## Retrieval evaluation

- Pipeline: built via `src/pipelines/evaluation/base_retrieval_eval_pipeline.py` (imported in 5.x notebooks).
- Metrics: Mean Average Precision (MAP), Mean Reciprocal Rank (MRR), and Recall.
- Configurations (example from closed vs. open comparisons):
  - Embedding models: open-source `Qwen/Qwen3-Embedding-8B` vs. closed `text-embedding-3-large`
  - Optional contextualization (LLMs) and rewriting components (e.g., `gemma3:27b` via Ollama and an OpenAI model as configured in the notebook)
- Outputs: Pickled DataFrames under `results/retrieval/<TIMESTAMP>/...` and plots in analysis sections.

## Generation evaluation

- See `6_generation_evaluation.ipynb` and `6_generation_analysis.ipynb`.
- Compatible with Haystack and RAG evaluation tooling present in `requirements.txt` (e.g., `ragas-haystack`). Consult the notebooks for the exact metric set used in this project.

## Reproducing headline experiments

- Ensure the corresponding document stores are built in `data/document_stores/` (run `4_indexing.ipynb`).
- Run `5.4_retrieval_evaluation_closed_vs_open.ipynb` to generate side-by-side results for open vs. closed configurations. Results are saved under `results/retrieval/open_closed/<TIMESTAMP>/`.
- Use `7_analysis.ipynb` (and the analysis cells at the end of 5.x) to visualize MAP/MRR/Recall comparisons.

## Testing

- A minimal test scaffold exists under `tests/` to test the custom component for the Qwen3 reranker. If you use pytest, you can run tests with:

```bash
python -m pytest -q
```

## How this ties into the thesis

This repository implements the experimental framework described in the accompanying master’s thesis, focusing on RAG for a specialized German medical domain. It operationalizes the methodology (data preparation, retrieval/generation pipelines, and evaluation) and provides all artifacts needed to reproduce the core findings. If you are the thesis reader or examiner, start with the notebooks in numerical order and refer to the `results/` directory for the generated outputs.

If you need a citation for this codebase, please use the thesis citation and optionally add:

```
Lukas Kaibel, “Retrieval-Augmented Generation for Medical Consultation” Master’s Thesis, Freie Universität Berlin, 2025. Code: https://github.com/lukaskaibel/rag-medical-consultation
```

---

## License

This project is licensed under the **MIT License**.

---

**Maintainer:** Lukas Kaibel  
For questions or collaboration inquiries, feel free to open an issue or contact via GitHub.