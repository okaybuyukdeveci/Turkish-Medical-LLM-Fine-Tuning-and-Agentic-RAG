# TUS evaluation

This directory contains a reproducible 100-question evaluation of the existing
LangGraph RAG agent. The source is
[`batuhanaktas/tus_eval`](https://huggingface.co/datasets/batuhanaktas/tus_eval),
revision `da3dfff0db1e14bad2a75ba54917fcfb9306cc7e`. Hugging Face exposes
115 rows in one `train` split; it does not publish a separate evaluation split.
`run.py` selects 100 rows with Python's `random.Random(20260925)` and stores
their original zero-based row indices in `questions.jsonl`. The selection is
deterministic and includes no model output.

## Run

From `agentic_rag/`, with the project's `.venv` and `.env` configured:

```bash
.venv/bin/python -m evaluation.run --prepare-only
.venv/bin/python -m evaluation.run --answers-only
.venv/bin/python -m evaluation.run --rescore-only
.venv/bin/python -m evaluation.run --assess-only --workers 3
```

The full command sends each question and retrieved medical text to the
configured Together AI model. It reuses the current index settings but copies
the embedded Qdrant files to a temporary directory, avoiding the lock held by
an open notebook. It never rebuilds or modifies the original index. Each
question starts a fresh conversation and uses the project's actual graph,
including retrieval and pruning. The last command only assesses saved
answers and does not resend questions through the graph. The output is appended to
`cache/results.jsonl` after every question; rerunning skips complete IDs and
retries errors. `--limit 1` is useful for a smoke run.
`--rescore-only` reapplies the deterministic A–E answer parser without an API
call; run it only after the answer process has stopped.
The optional `--workers` setting runs independent assessment calls concurrently;
cache writes remain serial in the main process.

## Review

Open `review.ipynb` from `agentic_rag/` to inspect summary counts, per-question
answers, queries, full retrieved documents, evidence shown to the responder,
and candidate corpus locations. The cache preserves the raw model answer,
the extracted A–E choice, and `correct` (`null` when the choice is ambiguous).

The separate assessment is the evaluated model's self-report, not independent
proof. It compares the pruned evidence shown to the responder and one scored
excerpt per retrieved document with the model's final answer. Full retrieved
documents and eight lexical candidate passages for the gold answer remain in
the cache for review. `corpus_location` locates a verified quotation supporting
the **model answer**; `gold_corpus_location` is populated when that answer also
matches the dataset key. Quotes are verified against source text exactly or
after Unicode and whitespace normalization. Line numbers refer to the original
Markdown file. An absent gold location means it was **not verified**, not that
the complete corpus lacks the fact. OCR noise, paraphrases, and facts spread
across passages may require manual review. The notebook lists these cases.

Do not interpret the 100-question accuracy as a clinical safety result.
