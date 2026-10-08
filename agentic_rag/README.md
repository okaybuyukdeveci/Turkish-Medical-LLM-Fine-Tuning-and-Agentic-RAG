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

## Local Qwen benchmark

The notebook uses Together AI. A separate CLI runs four local Qwen conditions:
`base`, `sft`, `base_agentic`, and `sft_agentic`. The agentic conditions reuse
`scripts/workflow/graph.py`, the parent–child retriever, and the existing prompts.
Both the responder and pruner use the selected model condition. No v2 router or
hybrid retrieval is added. Retrieval remains model-driven; skipped or malformed
tool calls are recorded rather than replaced with forced retrieval.

From `agentic_rag/`, install the dependencies, including local Qwen support:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Configure `benchmark.yaml`. Paths are relative to the YAML file. The local LoRA
adapter is at `models/sft`; base weights are downloaded separately from Hugging
Face. Loading uses Transformers and PEFT directly, without an inference server.
The default is float16 with automatic device placement; GPU memory and CPU
offloading affect performance. All generations use `do_sample=False` and disable
thinking through the base model's chat template.

On the tool-bound Qwen turn, the adapter removes the benchmark's final-answer
format instruction and appends a search-first instruction. The model still
generates its own query and tool call; retrieval is not synthesized when it
skips the tool. Check retrieval-call counts in each result before interpreting
agentic scores. Unicode-equivalent corpus paths with identical bytes are counted
once during loading and manifest validation; conflicting copies raise an error.

The YAML deliberately points to `benchmark/2026/nihai/`, not the preliminary OCR
output in `hazir/`. Complete the PDF review before creating those final files.
Questions require `id`, `question`, and five `options` objects with `label` and
`text`. The separate answer key requires `id` and `correct_option`. The CLI checks
counts, matching unique IDs, options, and basic answer leakage before loading the
model; these checks do not establish OCR or medical correctness.

```bash
python3 -m scripts.evaluation.run_benchmark --validate-only
python3 -m scripts.evaluation.run_benchmark --limit 5
```

The four modes use these model and workflow combinations:

| Mode | Model | Workflow |
| --- | --- | --- |
| `base` | Base Qwen | Direct generation |
| `sft` | Qwen + SFT v1 | Direct generation |
| `base_agentic` | Base Qwen | Existing agentic graph |
| `sft_agentic` | Qwen + SFT v1 | Same agentic graph |

Run each mode separately from `agentic_rag/`:

```bash
python3 -m scripts.evaluation.run_benchmark --modes base
python3 -m scripts.evaluation.run_benchmark --modes sft
python3 -m scripts.evaluation.run_benchmark --modes base_agentic
python3 -m scripts.evaluation.run_benchmark --modes sft_agentic
```

Run all four sequentially in one process, sharing the loaded model weights:

```bash
python3 -m scripts.evaluation.run_benchmark --modes base sft base_agentic sft_agentic
```

Omitting `--modes` uses the list in `benchmark.yaml`, which defaults to all four.
To try a subset on five questions:

```bash
python3 -m scripts.evaluation.run_benchmark --modes base sft --limit 5
```

The commands are sequential evaluations. Do not launch separate agentic modes
concurrently against the same embedded Qdrant storage.

Inspect the small run for actual retrieval calls, pruning, final answers, and
errors before starting a full benchmark. Close the notebook kernel before using
the embedded Qdrant index in another process. For a background run:

```bash
mkdir -p logs
nohup python3 -u -m scripts.evaluation.run_benchmark > logs/qwen-benchmark.log 2>&1 &
```

Each run gets its own results directory with the resolved configuration, one
JSONL per mode, and `summary.json`. Per-question records contain the response,
prediction, correctness, elapsed seconds, retrieval-call count, retrieval errors,
and generation/graph errors. Unparseable or failed answers count as incorrect.
The tool parser supports Qwen's XML-style retrieval calls and rejects malformed
calls. Local Qwen generation and tool following must still be verified on the
target hardware; adding this CLI does not establish benchmark scores.

Results are saved under the YAML setting `benchmark.output`, currently
`../benchmark/2026/results`, relative to `agentic_rag/benchmark.yaml`:

```text
benchmark/2026/results/<YYYYMMDD-HHMMSS>/
├── config.json
├── base.jsonl
├── sft.jsonl
├── base_agentic.jsonl
├── sft_agentic.jsonl
└── summary.json
```

Only selected modes produce JSONL files. Each invocation creates a new directory;
previous runs are not overwritten or resumed. The root `.gitignore` excludes
`benchmark/`, including these results.

`config.json` stores the loaded YAML settings with resolved adapter path. Each
mode's JSONL contains one record per completed question:

| Field | Meaning |
| --- | --- |
| `id`, `mode` | Question identifier and evaluated condition |
| `expected` | Correct option from the separate answer-key file |
| `response` | Final model response |
| `prediction` | A–E parsed from the initial `Doğru cevap: X` line; null if parsing fails |
| `correct` | Whether the prediction matches the answer key |
| `seconds` | Elapsed generation/graph time for the question; excludes initial model/index loading |
| `retrieval_calls` | Number of tool calls retained in the returned messages |
| `retrieval_errors` | Number of retained tool results whose artifact status is `error` |
| `error` | Exception type and message, present when execution fails |

On an execution error, the record stores `prediction: null`, `correct: false`,
and `error`; response and retrieval fields may be absent. Each line is flushed
immediately, so completed records remain available if the process stops.

After all selected modes finish, `summary.json` stores `questions`, `accuracy`,
and `errors` for each mode. Accuracy is a fraction from 0 to 1, with failed and
unparseable answers counted as incorrect. `errors` counts execution exceptions,
not every incorrect answer or retrieval failure. An interrupted run may have
partial JSONL files without a summary.

The background process log is separate from benchmark results:
`agentic_rag/logs/qwen-benchmark.log` for the example command above. The current
2026 launch uses `agentic_rag/logs/qwen-benchmark-2026.log`.

The current compactor contains an early return and is inactive. Single-question
runs use a fresh graph state each time. This benchmark measures the current
graph, including its one-retrieval limit, rather than a multi-turn conversation.

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
