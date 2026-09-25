"""Literal Markdown heading parser and tokenizer-aware medical document chunker."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Protocol, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_EMPTY_ALT_IMAGE = re.compile(r"!\[\s*\]\([^)]*\)")
_DEEP_HEADING_LINE = re.compile(r"^#{3,6}(?:[ \t]+.*)?$", re.MULTILINE)
_MARKDOWN_LINK = re.compile(r"\[([^]]*)\]\([^)]*\)")
_CHUNK_NAMESPACE = uuid.UUID("3f5039ea-35c3-4a7e-953c-97fa52ddbfbc")


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...


@dataclass(frozen=True)
class ChunkingConfig:
    tokenizer_name: str = "intfloat/multilingual-e5-base"
    max_tokens: int = 768
    overlap_tokens: int = 64

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")


@dataclass(frozen=True)
class _Section:
    title: str
    subtitle: str | None
    body: str


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _EMPTY_ALT_IMAGE.sub("", text)
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _has_substantive_body(body: str) -> bool:
    candidate = _DEEP_HEADING_LINE.sub("", body)
    candidate = _MARKDOWN_LINK.sub(r"\1", candidate)
    return any(character.isalnum() for character in candidate)


def _parse_sections(document: Document) -> list[_Section]:
    document_title = str(document.metadata["document_title"])
    title = document_title
    subtitle: str | None = None
    body_lines: list[str] = []
    sections: list[_Section] = []

    def flush() -> None:
        body = _clean_text("\n".join(body_lines))
        if body and _has_substantive_body(body):
            sections.append(_Section(title=title, subtitle=subtitle, body=body))
        body_lines.clear()

    for raw_line in document.page_content.splitlines():
        match = _HEADING.match(raw_line)
        if not match or len(match.group(1)) > 2:
            body_lines.append(raw_line)
            continue

        flush()
        heading_text = _clean_text(match.group(2))
        if len(match.group(1)) == 1:
            title = heading_text or document_title
            subtitle = None
        else:
            subtitle = heading_text or None

    flush()
    return sections


def _token_count(tokenizer: Tokenizer, text: str) -> int:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
    except TypeError:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def _context_prefix(title: str, subtitle: str | None) -> str:
    lines = [f"# {title}"]
    if subtitle:
        lines.append(f"## {subtitle}")
    return "\n".join(lines) + "\n\n"


def _split_body(
    body: str,
    prefix: str,
    tokenizer: Tokenizer,
    config: ChunkingConfig,
) -> list[str]:
    available_tokens = config.max_tokens - _token_count(tokenizer, prefix)
    if available_tokens <= 0:
        raise ValueError(f"Heading context exceeds the chunk token limit: {prefix.strip()}")
    if _token_count(tokenizer, body) <= available_tokens:
        return [body]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_tokens,
        chunk_overlap=min(config.overlap_tokens, available_tokens - 1),
        length_function=lambda value: _token_count(tokenizer, value),
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_text(body)


def _load_tokenizer(name: str) -> Tokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for tokenizer-aware chunking. "
            "Install the packages in requirements.txt."
        ) from exc
    return AutoTokenizer.from_pretrained(name)


def chunk_documents(
    documents: list[Document],
    config: ChunkingConfig | None = None,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[Document]:
    """Create deterministic, heading-contextualized chunks without mutating inputs."""

    selected_config = config or ChunkingConfig()
    selected_config.validate()
    selected_tokenizer = tokenizer or _load_tokenizer(selected_config.tokenizer_name)
    chunks: list[Document] = []
    source_counters: dict[str, int] = {}

    for document in documents:
        for section_number, section in enumerate(_parse_sections(document)):
            prefix = _context_prefix(section.title, section.subtitle)
            pieces = _split_body(
                section.body, prefix, selected_tokenizer, selected_config
            )
            for piece_number, piece in enumerate(pieces):
                cleaned_piece = piece.strip()
                if not _has_substantive_body(cleaned_piece):
                    continue
                page_content = prefix + cleaned_piece
                if _token_count(selected_tokenizer, page_content) > selected_config.max_tokens:
                    raise ValueError("Chunk exceeds configured token limit")

                source = str(document.metadata["source"])
                chunk_index = source_counters.get(source, 0)
                source_counters[source] = chunk_index + 1
                content_hash = hashlib.sha256(page_content.encode("utf-8")).hexdigest()
                identity = f"{source}\n{section_number}\n{piece_number}\n{content_hash}"
                if parent_id := document.metadata.get("parent_id"):
                    identity = f"{parent_id}\n{identity}"
                chunk_id = str(
                    uuid.uuid5(
                        _CHUNK_NAMESPACE,
                        identity,
                    )
                )
                breadcrumb_parts = [
                    str(document.metadata.get("category", "")),
                    section.title,
                    section.subtitle or "",
                ]
                metadata = {
                    **document.metadata,
                    "title": section.title,
                    "breadcrumb": " > ".join(part for part in breadcrumb_parts if part),
                    "chunk_index": chunk_index,
                    "content_hash": content_hash,
                    "chunk_id": chunk_id,
                }
                if section.subtitle:
                    metadata["subtitle"] = section.subtitle
                chunks.append(Document(page_content=page_content, metadata=metadata))

    if not chunks:
        raise ValueError("The documents did not contain any indexable text")
    return chunks


def chunk_parent_child_documents(
    documents: list[Document],
    parent_config: ChunkingConfig,
    child_config: ChunkingConfig,
    *,
    tokenizer: Tokenizer | None = None,
) -> tuple[list[Document], list[Document]]:
    """Split source documents into parents and then into linked child chunks."""

    selected_tokenizer = tokenizer or _load_tokenizer(parent_config.tokenizer_name)
    parents = chunk_documents(documents, parent_config, tokenizer=selected_tokenizer)
    children = []
    for parent in parents:
        parent_id = parent.metadata["chunk_id"]
        linked_parent = Document(
            page_content=parent.page_content,
            metadata={**parent.metadata, "parent_id": parent_id},
        )
        children.extend(
            chunk_documents([linked_parent], child_config, tokenizer=selected_tokenizer)
        )
    return parents, children
