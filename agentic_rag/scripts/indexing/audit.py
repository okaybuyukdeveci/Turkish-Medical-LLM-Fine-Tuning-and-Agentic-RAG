"""Report chunk quality metrics for the complete local corpus."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter

from scripts.indexing.chunker import (
    ChunkingConfig,
    _has_substantive_body,
    chunk_documents,
)
from scripts.indexing.loader import load_markdown_documents
from scripts.settings import RagSettings


_EMPTY_ALT_IMAGE = re.compile(r"!\[\s*\]\([^)]*\)")


def audit_corpus(settings: RagSettings) -> dict:
    documents = load_markdown_documents(settings.data_dir)
    config = ChunkingConfig(
        tokenizer_name=settings.embedding_model,
        max_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    chunks = chunk_documents(documents, config, tokenizer=tokenizer)
    token_counts = [
        len(
            tokenizer.encode(
                chunk.page_content, add_special_tokens=False, verbose=False
            )
        )
        for chunk in chunks
    ]
    ordered = sorted(token_counts)
    return {
        "document_count": len(documents),
        "word_count": sum(len(document.page_content.split()) for document in documents),
        "chunk_count": len(chunks),
        "removed_empty_image_references": sum(
            len(_EMPTY_ALT_IMAGE.findall(document.page_content)) for document in documents
        ),
        "non_substantive_chunks": sum(
            not _has_substantive_body(chunk.page_content.split("\n\n", 1)[-1])
            for chunk in chunks
        ),
        "remaining_empty_image_references": sum(
            len(_EMPTY_ALT_IMAGE.findall(chunk.page_content)) for chunk in chunks
        ),
        "token_counts": {
            "minimum": min(token_counts),
            "median": statistics.median(token_counts),
            "p95": ordered[int(0.95 * (len(ordered) - 1))],
            "maximum": max(token_counts),
        },
        "chunks_per_category": dict(
            sorted(Counter(chunk.metadata["category"] for chunk in chunks).items())
        ),
    }


def main() -> None:
    print(json.dumps(audit_corpus(RagSettings.from_env()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
