# PIA RAG Evaluation

A research-grade, reproducible pipeline to evaluate Retrieval-Augmented Generation (RAG) systems across retrieval and generation, developed in the context of a master’s thesis. The project compares open-source and closed-source model components (embedders, contextualizers/rewriters, rerankers) on a domain-specific corpus of German medical documents related to obstetrics (e.g., Kaiserschnitt, Einleitung der Geburt, Narkose).

## Why this project

- End-to-end RAG evaluation, from document preprocessing and QA dataset generation to retrieval metrics and generation quality analysis.
- Side-by-side comparisons of open vs. closed components, and ablations for reranking and query rewriting.
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
    - `5.4_retrieval_evaluation_closed_vs_open_base.ipynb`
    - `5.5_retrieval_evaluation_closed_vs_open.ipynb`
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

## Reproducible workflows (notebooks)

All notebooks start by loading `.env`. Dependencies are centralized in `requirements.txt` (no in-notebook `%pip install`).

Suggested execution order
1. `1_document_preprocessing.ipynb`
2. `2_qa_dataset_generation.ipynb`
3. `3_question_document_matching.ipynb`
4. `4_indexing.ipynb`
5. Retrieval evaluation (5.x series)
6. `6_generation_evaluation.ipynb`
7. `6_generation_analysis.ipynb` and `7_analysis.ipynb`

Numbered retrieval notebooks (5.x)
- 5 (baseline): `5_retrieval_evaluation.ipynb`
- 5.1 line vs. passage: `5.1_retrieval_evaluation_line_passage.ipynb`
- 5.2 reranker: `5.2_retrieval_evaluation_reranker.ipynb`
- 5.3 rewriting: `5.3_retrieval_evaluation_rewriting.ipynb`
- 5.4 closed vs. open (base): `5.4_retrieval_evaluation_closed_vs_open_base.ipynb`
- 5.5 closed vs. open (full): `5.5_retrieval_evaluation_closed_vs_open.ipynb`

Running notebooks non-interactively
- See `helper/run_notebook_in_tmux.sh` or `notebooks/run_notebook_in_tmux.sh` for batch execution patterns that activate `.venv` and run a given notebook.

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
- Run `5.5_retrieval_evaluation_closed_vs_open.ipynb` to generate side-by-side results for open vs. closed configurations. Results are saved under `results/retrieval/open_closed/<TIMESTAMP>/`.
- Use `7_analysis.ipynb` (and the analysis cells at the end of 5.x) to visualize MAP/MRR/Recall comparisons.

## Testing

- A minimal test scaffold exists under `tests/`. If you use pytest, you can run tests with:

```bash
python -m pytest -q
```

## Notes and limits

- Large local models (e.g., 8B embeddings or rerankers) may require substantial RAM/VRAM. Alternatively, use remote providers as configured by environment variables.
- Provider access (e.g., OpenAI) requires valid API keys in `.env`.
- Some notebooks sample subsets for evaluation (e.g., a fixed number of questions); set random seeds where reproducibility matters.

## How this ties into the thesis

This repository implements the experimental framework described in the accompanying master’s thesis, focusing on RAG for a specialized German medical domain. It operationalizes the methodology (data preparation, retrieval/generation pipelines, and evaluation) and provides all artifacts needed to reproduce the core findings. If you are the thesis reader or examiner, start with the notebooks in numerical order and refer to the `results/` directory for the generated outputs.

If you need a citation for this codebase, please use the thesis citation and optionally add:

```
Author Name, “Thesis Title,” Master’s Thesis, Institution, Year. Code: https://github.com/<repo or internal location>/pia-rag-eval
```

Replace the placeholders with the actual thesis details.

## License

Specify your license here (e.g., MIT, Apache-2.0) or mark as “All rights reserved” if private for thesis submission. Ensure compatibility with included models and datasets.

## Acknowledgements

- Haystack community for the RAG pipeline components
- Providers and model authors (OpenAI, Qwen, Sentence Transformers, etc.)
- Advisors and reviewers supporting the thesis work
