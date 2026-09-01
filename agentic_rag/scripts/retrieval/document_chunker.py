"""Hierarchical chunker for Turkish medical Markdown documents.

The raw TUSDATA / MinerU exports use flat heading levels (headers are often
`##` regardless of their logical depth, and sub-items like "1.1" or "A)" can
receive the same number of hashes as a top-level section). Splitting purely on
character count therefore tears medical context apart and drops the section
provenance that grounded citation needs.

This module fixes that in three stages:

1. ``normalize_markdown_headers`` normalizes MinerU typography so sub-items drop
   to a lower heading level instead of hijacking the main section level.
2. A two-pass split first cuts on heading boundaries (``MarkdownHeaderTextSplitter``)
   and then re-sizes any oversized / table-heavy sections with a
   ``RecursiveCharacterTextSplitter``.
3. Every chunk carries a ``breadcrumb`` metadata string (branch > section >
   heading) so downstream agents can cite their sources and filter by topic.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Match a sub-item line that MinerU promoted to full heading level:
# a hierarchical enumerator ("1.1", "1.1.2"), a lettered item ("A)") or a
# numbered item ("2)"). Only # / ## levels are demoted; ### items already sit
# at the right depth, and single enumerators ("1. KONU") are real sections.
_PROMOTED_SUBHEADING = re.compile(
    r"^(#{1,2})[ \t]+(?:\d+\.\d+(?:\.\d+)*|[A-Za-z]\)|\d+\))[ \t]+"
)


def normalize_markdown_headers(text: str) -> str:
    """Normalize MinerU heading decorations so sub-items nest under their parent.

    When a sub-item was written with the same number of hashes as the section it
    belongs to, demote it by one level so the hierarchy reads correctly.

    Args:
        text: Raw Markdown content from MinerU.

    Returns:
        The normalized Markdown text.
    """
    lines = []
    for line in text.splitlines():
        match = _PROMOTED_SUBHEADING.match(line)
        if match:
            # Add one hash (demote) while keeping the enumerator intact.
            lines.append("#" + line)
        else:
            lines.append(line)
    return "\n".join(lines)


def _split_on_headers(documents: list[Document]) -> list[Document]:
    """Stage one: split each document into logical sections by heading level."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "section"),
            ("##", "heading"),
            ("###", "subheading"),
        ],
        strip_headers=False,
    )
    header_chunks = []
    for document in documents:
        normalized = normalize_markdown_headers(document.page_content)
        for chunk in splitter.split_text(normalized):
            # Fold any recorded header keys back into the breadcrumb string.
            breadcrumb = _breadcrumb_from_metadata(chunk.metadata, document.metadata)
            header_chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata={
                        **document.metadata,
                        **chunk.metadata,
                        "breadcrumb": breadcrumb,
                    },
                )
            )
    return header_chunks


def _breadcrumb_from_metadata(
    section_metadata: dict, source_metadata: dict
) -> str:
    """Join branch, section and heading names into a single topic path."""
    parts = []
    category = source_metadata.get("category")
    if category and category != ".":
        parts.append(str(category))
    for key in ("section", "heading", "subheading"):
        value = section_metadata.get(key)
        if value:
            parts.append(str(value).strip())
    return " > ".join(parts)


def _resize_oversized(sections: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Stage two: re-split sections that are still too large for the retriever."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for section in sections:
        if len(section.page_content) <= chunk_size:
            chunks.append(section)
            continue
        for piece in splitter.split_text(section.page_content):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={**section.metadata},
                )
            )
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into heading-aware chunks with a breadcrumb trail.

    Args:
        documents: Documents to chunk (as produced by ``data_loader``).
        chunk_size: Maximum size (in characters) per final chunk.
        chunk_overlap: Characters of overlap between resized chunks.

    Returns:
        A new list of ``Document`` objects. Each keeps ``source``, ``category``,
        ``title`` and ``breadcrumb`` metadata; oversized sections are re-split
        under an incremental ``chunk`` index.
    """
    sections = _split_on_headers(documents)
    sized = _resize_oversized(sections, chunk_size, chunk_overlap)

    # Number chunks per source so ordering is recoverable downstream.
    counters: dict[str, int] = {}
    final_chunks = []
    for chunk in sized:
        key = chunk.metadata.get("source", "")
        index = counters.get(key, 0)
        counters[key] = index + 1
        chunk.metadata["chunk"] = index
        final_chunks.append(chunk)
    return final_chunks
