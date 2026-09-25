# Turkish Medical Agentic RAG

This project loads the Turkish medical Markdown corpus in `data/`, creates heading-aware parent and child chunks, indexes the children in local Qdrant, stores the parents in local SQLite, and runs a LangGraph workflow:

```text
Responder -> Retrieval Tool -> Pruner -> Responder -> Compactor
```

The intended use is Turkish educational and TUS-style medical study. Answers are grounded in retrieved source chunks; this is not a patient-specific diagnosis system.

## Setup

Python 3.13 is supported by the pinned environment.

```bash
cd agentic_rag
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Set `TOGETHER_API_KEY` in `.env`. The default chat model is `zai-org/GLM-5.3-Flash`; override it with `TOGETHER_MODEL` when needed. Never commit `.env` or paste credentials into the notebook.

The first embedding run downloads `intfloat/multilingual-e5-base` from Hugging Face. It is used locally on CPU by default.

## Build and validate the index

Create the collection when it does not exist:

```bash
python3 -m scripts.indexing.indexer
```

The command records a manifest containing source hashes and all index-shaping settings. Later runs reuse an exact match. If the data, embedding model, or chunk settings changed, indexing stops rather than silently deleting the collection. The previous single-level index must be rebuilt once for parent–child retrieval:

```bash
python3 -m scripts.indexing.indexer --force
```

Inspect corpus-wide chunk metrics without writing to Qdrant:

```bash
python3 -m scripts.indexing.audit
```

Child vectors are written to `qdrant_storage/` and parent documents to `qdrant_storage/parent_documents.sqlite`; both are ignored by Git. Embedded Qdrant is intended for one notebook/process at a time. Close the notebook kernel before rebuilding from the CLI.

## Run the agent

Open `main.ipynb` from the `agentic_rag/` directory and use **Run All**. The notebook:

1. loads settings and previews parent and child chunk statistics;
2. creates or validates the Qdrant index and parent store;
3. performs a retrieval sanity check that returns parent documents;
4. creates Together AI clients and compiles the graph;
5. displays the graph and defines a stateful `chat()` helper;
6. runs one Turkish example question.

Set `FORCE_REINDEX = True` in the index cell once to rebuild an older index, then restore it to `False`.

## Package layout

```text
scripts/
  indexing/   Markdown loading, chunking, audit, index manifest and build CLI
  retrieval/  E5 prefix adapter, local Qdrant and SQLite parent store, retriever
  tools/      source-preserving retrieval tool
  llm/        Together AI model construction
  agents/     responder, pruner and deterministic compactor nodes
  workflow/   LangGraph assembly
  prompts/    grounded-response and evidence-pruning instructions
```

Parent chunks have a 2000-token limit and no overlap. Child chunks have a 250-token limit and 60-token overlap by default. Each child stores a `parent_id` pointing to its full parent document in SQLite. Both levels keep source, heading, and deterministic chunk metadata. Qdrant searches the child embeddings; retrieval returns each matching parent once, in child relevance order.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q
RUN_CORPUS_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -m corpus
```

Tests cover loading, literal Markdown hierarchy, token limits, E5 prefixes, Qdrant lifecycle, tool formatting, retrieval failures, graph routing, pruning, and compaction.

## Security note

An older deleted notebook committed credentials in repository history. Those credentials should be rotated. This refactor does not rewrite Git history; history cleanup must be performed separately and deliberately if required.
