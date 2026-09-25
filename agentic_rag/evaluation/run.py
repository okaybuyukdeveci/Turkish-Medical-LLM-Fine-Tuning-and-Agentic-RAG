"""Reproducible, resumable 100-question TUS RAG evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from scripts.llm.together import create_together_llms
from scripts.retrieval.retriever import load_retriever
from scripts.settings import RagSettings
from scripts.workflow.graph import build_graph


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source" / "TusQA - Sheet1.csv"
QUESTIONS = HERE / "questions.jsonl"
RESULTS = HERE / "cache" / "results.jsonl"
DATASET_REVISION = "da3dfff0db1e14bad2a75ba54917fcfb9306cc7e"
SOURCE_SHA256 = "699dafa2be18bd7d2c42a06d852d5246bcf3b11d5054bb1ccc13dfe391d20bb0"
SEED = 20260925
STOPWORDS = {
    "asagidaki", "asagidakilerden", "hangisi", "hangisidir", "hangisinde",
    "hangileri", "ilgili", "olan", "olarak", "gore", "ile", "icin", "bir",
    "ve", "veya", "en", "bu", "da", "de", "ile", "olasi", "dogrudur",
    "yanlistir", "hastada", "hasta", "sonra", "durumda", "gosterir",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_questions() -> list[dict]:
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise ValueError("The dataset CSV differs from the pinned Hugging Face revision")
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 100 or set(rows[0]) != {"question", "choices", "answer"}:
        raise ValueError("Unexpected Hugging Face dataset shape")
    indices = sorted(random.Random(SEED).sample(range(len(rows)), 100))
    selected = [
        {
            "id": f"tus_eval_{index:03d}",
            "source_row": index,
            "question": rows[index]["question"].strip(),
            "choices": rows[index]["choices"].strip(),
            "gold_answer": rows[index]["answer"].strip().upper(),
        }
        for index in indices
    ]
    if any(row["gold_answer"] not in "ABCDE" for row in selected):
        raise ValueError("A gold answer is not a single A-E choice")
    write_jsonl(QUESTIONS, selected)
    (HERE / "selection.json").write_text(
        json.dumps(
            {
                "dataset": "batuhanaktas/tus_eval",
                "split": "train (the only published split)",
                "revision": DATASET_REVISION,
                "source_sha256": source_hash,
                "total_rows": len(rows),
                "selected_rows": len(selected),
                "seed": SEED,
                "row_index_base": 0,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return selected


def words(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return {
        word for word in re.findall(r"[a-z0-9]{4,}", normalized)
        if word not in STOPWORDS
    }


def gold_choice_text(choices: str, letter: str) -> str:
    positions = list(re.finditer(r"(?<!\w)([A-E])\)", choices))
    for index, match in enumerate(positions):
        if match.group(1) == letter:
            end = positions[index + 1].start() if index + 1 < len(positions) else len(choices)
            return choices[match.end():end].strip()
    return ""


def corpus_passages(data_dir: Path) -> list[dict]:
    passages = []
    for path in sorted(data_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text):
            passage = match.group().strip()
            if len(passage) < 30:
                continue
            first_line = text.count("\n", 0, match.start()) + 1
            # Very long OCR paragraphs are divided into line windows.
            lines = passage.splitlines()
            for offset in range(0, len(lines), 8):
                fragment = "\n".join(lines[offset:offset + 10]).strip()
                if len(fragment) >= 30:
                    passages.append({
                        "source": path.relative_to(data_dir).as_posix(),
                        "line": first_line + offset,
                        "text": fragment[:1600],
                        "tokens": words(fragment[:1600]),
                    })
    return passages


def corpus_candidates(passages: list[dict], question: dict, count: int = 8) -> list[dict]:
    query = words(question["question"] + " " + gold_choice_text(
        question["choices"], question["gold_answer"]
    ))
    document_frequency = Counter(token for passage in passages for token in passage["tokens"])
    scores = []
    for index, passage in enumerate(passages):
        common = query & passage["tokens"]
        if common:
            score = sum(1 / (1 + document_frequency[token]) ** 0.5 for token in common)
            scores.append((score, index))
    top = sorted(scores, reverse=True)[:count]
    return [
        {"id": f"P{rank}", "source": passages[index]["source"],
         "line": passages[index]["line"], "text": passages[index]["text"]}
        for rank, (_, index) in enumerate(top, 1)
    ]


def folded(text: str, with_offsets: bool = False):
    result = []
    offsets = []
    for position, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKD", character.casefold()).replace("ı", "i"):
            if unicodedata.combining(normalized):
                continue
            value = normalized if normalized.isalnum() else " "
            if value == " " and (not result or result[-1] == " "):
                continue
            result.append(value)
            if with_offsets:
                offsets.append(position)
    if result and result[-1] == " ":
        result.pop()
        if with_offsets:
            offsets.pop()
    return ("".join(result), offsets) if with_offsets else "".join(result)


class CorpusLocator:
    def __init__(self, data_dir: Path):
        self.files = {}
        for path in data_dir.rglob("*.md"):
            source = path.relative_to(data_dir).as_posix()
            content = path.read_text(encoding="utf-8")
            self.files[unicodedata.normalize("NFC", source)] = (source, content, folded(content))

    def find(self, quote: str, preferred_sources: list[str]) -> dict | None:
        needle = folded(quote)
        if len(needle) < 35:
            return None
        keys = list(dict.fromkeys(
            [unicodedata.normalize("NFC", source) for source in preferred_sources]
            + list(self.files)
        ))
        for key in keys:
            if key not in self.files:
                continue
            source, content, haystack = self.files[key]
            position = haystack.find(needle)
            if position < 0:
                continue
            verified, offsets = folded(content, with_offsets=True)
            if verified != haystack:
                raise AssertionError("Corpus normalization changed while locating evidence")
            start = offsets[position]
            end = offsets[position + len(needle) - 1]
            return {
                "source": source,
                "line_start": content.count("\n", 0, start) + 1,
                "line_end": content.count("\n", 0, end) + 1,
                "quote": content[start:end + 1],
                "match_method": "normalized_quote",
            }
        return None


def extract_choice(answer: str) -> str | None:
    patterns = [
        r"(?:seçtiğim\s*şık|cevabım|yanıtım|tercihim|doğru\s*şık)\s*[:：\-]?\s*\*{0,2}([A-E])\b",
        r"(?:doğru\s*(?:cevap|yanıt|seçenek|şıkk?ı)|cevap|yanıt)\s*[:：\-]?\s*\*{0,2}([A-E])\b",
        r"\b([A-E])\s*\)\s*(?:şıkk?ı|seçeneği|doğru)",
        r"^\s*\*{0,2}([A-E])\s*[).:]",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
        unique = {match.upper() for match in matches}
        if len(unique) == 1:
            return unique.pop()
        if len(unique) > 1:
            return None
    return None


class RecordingRetriever:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.calls = []

    def invoke(self, query: str):
        documents = self.wrapped.invoke(query)
        self.calls.append({
            "query": query,
            "documents": [
                {"content": document.page_content, "metadata": document.metadata}
                for document in documents
            ],
        })
        return documents


def run_question(graph, recording_retriever: RecordingRetriever, question: dict) -> dict:
    recording_retriever.calls.clear()
    prompt = (
        question["question"] + "\n\n" + question["choices"]
        + "\n\nYanıtında seçtiğin şıkkı açıkça belirt."
    )
    final_answer = ""
    queries = []
    raw_tool_messages = []
    pruned_evidence = []
    for update in graph.stream({"messages": [HumanMessage(content=prompt)]}, stream_mode="updates"):
        for node, payload in update.items():
            if not isinstance(payload, dict):
                continue
            for message in payload.get("messages", []):
                if node == "responder":
                    for call in getattr(message, "tool_calls", []) or []:
                        if call.get("name") == "retrieve_documents":
                            queries.append(call.get("args", {}).get("query", ""))
                    if not getattr(message, "tool_calls", None):
                        final_answer = str(message.content)
                elif node == "retrieval_tools":
                    raw_tool_messages.append(str(message.content))
                elif node == "prune":
                    pruned_evidence.append(str(message.content))
    return {
        "prompt": prompt,
        "model_answer": final_answer,
        "predicted_answer": extract_choice(final_answer),
        "correct": extract_choice(final_answer) == question["gold_answer"]
        if extract_choice(final_answer) else None,
        "retrieval_queries": queries,
        "retrieval_calls": recording_retriever.calls.copy(),
        "raw_tool_messages": raw_tool_messages,
        "pruned_evidence": pruned_evidence,
    }


def retrieval_excerpts(question: dict, run: dict) -> str:
    chosen = run.get("predicted_answer") or ""
    query = words(question["question"] + " " + gold_choice_text(question["choices"], chosen))
    blocks = []
    for call in run["retrieval_calls"]:
        for number, document in enumerate(call["documents"], 1):
            paragraphs = [part.strip() for part in document["content"].split("\n\n") if part.strip()]
            ranked = sorted(
                paragraphs,
                key=lambda part: len(words(part) & query),
                reverse=True,
            )
            excerpt = (ranked[0] if ranked else "")[:1000]
            blocks.append(
                f"[DOCUMENT {number}] {document['metadata'].get('source')} "
                f"chunk {document['metadata'].get('chunk_index')}\n{excerpt}"
            )
    return "\n\n".join(blocks)


def assess(llm, question: dict, run: dict, locator: CorpusLocator) -> dict:
    # The complete raw documents remain in the cache for human inspection.
    evidence = "\n\n".join(run["pruned_evidence"])[:6000]
    raw_excerpts = retrieval_excerpts(question, run)
    system = SystemMessage(content=(
        "Review your prior medical answer against the supplied retrieved evidence. "
        "Return only short JSON with keys answer_uses_evidence "
        "(yes/partial/no/uncertain), present_in_retrieval (array of short facts), "
        "missing_in_retrieval (array of facts used in your answer but absent from "
        "the shown evidence), support_quote (one short exact excerpt supporting "
        "your chosen answer, or null). Use no outside medical knowledge. "
        "The raw excerpts are selected snippets, so uncertainty about the full "
        "retrieved documents must stay uncertain."
    ))
    user = HumanMessage(content=(
        f"QUESTION: {question['question']}\nCHOICES: {question['choices']}\n"
        f"MODEL ANSWER: {run['model_answer'][:2500]}\n"
        f"EVIDENCE SHOWN TO RESPONDER:\n{evidence or '[none]'}\n"
        f"RAW RETRIEVED EXCERPTS:\n{raw_excerpts or '[none]'}"
    ))
    raw = str(llm.invoke([system, user]).content)
    try:
        parsed = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return {"raw": raw, "parse_error": True}
    sources = [
        document["metadata"].get("source", "")
        for call in run["retrieval_calls"] for document in call["documents"]
    ]
    quote = parsed.get("support_quote")
    parsed["corpus_location"] = locator.find(quote, sources) if isinstance(quote, str) else None
    parsed["corpus_assessment"] = "supported" if parsed["corpus_location"] else "not_located"
    parsed["gold_corpus_location"] = parsed["corpus_location"] if run.get("correct") is True else None
    parsed["retrieved_support"] = parsed.get("answer_uses_evidence", "uncertain")
    parsed["gold_review_flag"] = run.get("correct") is False and bool(parsed["corpus_location"])
    return {"raw": raw, "parsed": parsed, "scope": "pruned evidence and one scored excerpt per retrieved document"}


def cached_results() -> dict[str, dict]:
    if not RESULTS.exists():
        return {}
    latest = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        latest[row["id"]] = row
    return latest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rescore-only", action="store_true")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--answers-only", action="store_true")
    phase.add_argument("--assess-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent assessment calls; assess-only mode")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.workers < 1 or (args.workers != 1 and not args.assess_only):
        parser.error("--workers above 1 is only valid with --assess-only")
    questions = prepare_questions()
    print(f"Prepared {len(questions)} questions from {SOURCE.name}", flush=True)
    if args.prepare_only:
        return
    if args.rescore_only:
        rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            if row.get("model_answer"):
                predicted = extract_choice(row["model_answer"])
                row["predicted_answer"] = predicted
                row["correct"] = predicted == row["gold_answer"] if predicted else None
        temporary = RESULTS.with_suffix(".tmp")
        write_jsonl(temporary, rows)
        os.replace(temporary, RESULTS)
        print(f"Rescored {len(rows)} cached records", flush=True)
        return

    settings = RagSettings.from_env()
    llms = create_together_llms(settings) if not args.assess_only else None
    from langchain_together import ChatTogether
    judge_llm = ChatTogether(
        model=settings.together_model,
        api_key=settings.require_together_api_key(),
        temperature=0,
        timeout=45,
        max_retries=0,
    )
    passages = corpus_passages(settings.data_dir) if not args.answers_only else []
    locator = CorpusLocator(settings.data_dir) if not args.answers_only else None
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    # A notebook currently holds the embedded Qdrant lock. Copy the read-only
    # index into a temporary location so the evaluation has its own client.
    index_context = nullcontext(None) if args.assess_only else tempfile.TemporaryDirectory(prefix="tus-eval-index-")
    with index_context as temporary:
        graph = retriever = None
        if not args.assess_only:
            snapshot = Path(temporary) / "qdrant_storage"
            shutil.copytree(settings.qdrant_path, snapshot, ignore=shutil.ignore_patterns(".lock"))
            retriever = RecordingRetriever(load_retriever(replace(settings, qdrant_path=snapshot)))
            graph = build_graph(llms.responder, llms.pruning, retriever)
        previous = cached_results()
        pending = []
        for number, question in enumerate(questions[:args.limit], 1):
            cached = previous.get(question["id"], {})
            if args.answers_only and cached.get("model_answer"):
                continue
            if args.assess_only and not cached.get("model_answer"):
                continue
            if cached.get("status") == "complete" and not cached.get("assessment", {}).get("parse_error"):
                continue
            pending.append((number, question, cached))

        def process(item):
            number, question, cached = item
            result = cached.copy() if cached.get("model_answer") else {
                "id": question["id"], "source_row": question["source_row"],
                "question": question["question"], "choices": question["choices"],
                "gold_answer": question["gold_answer"], "model": settings.together_model,
            }
            started = time.monotonic()
            try:
                if not result.get("model_answer"):
                    result.update(run_question(graph, retriever, question))
                    result["answer_seconds"] = round(time.monotonic() - started, 1)
                if args.answers_only:
                    result["status"] = "answered"
                else:
                    candidates = corpus_candidates(passages, question)
                    result["corpus_candidates"] = candidates
                    result["assessment"] = assess(judge_llm, question, result, locator)
                    result["status"] = "complete"
            except Exception as exc:
                result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            result["total_seconds"] = round(time.monotonic() - started, 1)
            return number, question, result

        if args.workers > 1:
            executor = ThreadPoolExecutor(max_workers=args.workers)
            processed = executor.map(process, pending)
        else:
            executor = None
            processed = map(process, pending)
        for number, question, result in processed:
            with RESULTS.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"{number}/{min(args.limit, 100)} {question['id']}: {result['status']}", flush=True)
        if executor is not None:
            executor.shutdown()


if __name__ == "__main__":
    main()
