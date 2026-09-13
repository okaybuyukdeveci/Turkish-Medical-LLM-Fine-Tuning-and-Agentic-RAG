"""Load the Markdown corpus into LangChain documents."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from langchain_core.documents import Document


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _document_title(file_name: str) -> str:
    if file_name.endswith(".pdf.md"):
        return file_name[: -len(".pdf.md")]
    if file_name.endswith(".md"):
        return file_name[: -len(".md")]
    return file_name


def load_markdown_documents(data_dir: str | Path) -> list[Document]:
    """Read every Markdown file recursively in a deterministic order."""

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    documents: list[Document] = []
    for file_path in sorted(data_path.rglob("*.md"), key=lambda path: path.as_posix()):
        relative_path = file_path.relative_to(data_path)
        category = relative_path.parent.as_posix()
        documents.append(
            Document(
                page_content=_normalize(file_path.read_text(encoding="utf-8")),
                metadata={
                    "source": _normalize(relative_path.as_posix()),
                    "category": "" if category == "." else _normalize(category),
                    "document_title": _normalize(_document_title(file_path.name)),
                },
            )
        )

    if not documents:
        raise ValueError(f"No Markdown files found in: {data_path}")
    return documents
