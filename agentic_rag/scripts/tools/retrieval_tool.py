"""Source-preserving retrieval tool used by the Responder."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.tools import tool


def format_evidence(documents: list[Document]) -> str:
    """Format retrieved chunks as explicit, citable evidence blocks."""

    if not documents:
        return "[NO_EVIDENCE]\nNo relevant evidence was returned by the knowledge base."

    blocks = []
    for number, document in enumerate(documents, start=1):
        metadata = document.metadata
        labels = [
            f"source: {metadata.get('source', 'unknown')}",
            f"title: {metadata.get('title', metadata.get('document_title', 'unknown'))}",
        ]
        if metadata.get("subtitle"):
            labels.append(f"subtitle: {metadata['subtitle']}")
        labels.append(f"chunk: {metadata.get('chunk_index', 'unknown')}")
        blocks.append(
            f"[EVIDENCE {number}]\n"
            + "\n".join(labels)
            + f"\ncontent:\n{document.page_content}"
        )
    return "\n\n".join(blocks)


def _artifact(documents: list[Document], status: str = "ok") -> dict:
    return {
        "status": status,
        "sources": [
            {
                "source": document.metadata.get("source", "unknown"),
                "title": document.metadata.get(
                    "title", document.metadata.get("document_title", "unknown")
                ),
                "subtitle": document.metadata.get("subtitle"),
                "chunk_index": document.metadata.get("chunk_index"),
            }
            for document in documents
        ],
    }


def create_retrieve_documents_tool(retriever):
    """Create one retrieval tool bound to the supplied retriever."""

    @tool(response_format="content_and_artifact")
    def retrieve_documents(query: str) -> tuple[str, dict]:
        """Search the Turkish medical knowledge base for a concise medical query."""

        if not query.strip():
            return format_evidence([]), {"status": "empty_query", "sources": []}
        try:
            documents = list(retriever.invoke(query.strip()))
        except Exception as exc:
            return (
                "[RETRIEVAL_ERROR]\nThe knowledge base could not be searched. "
                "Do not answer from memory.",
                {"status": "error", "sources": [], "error_type": type(exc).__name__},
            )
        return format_evidence(documents), _artifact(documents)

    return retrieve_documents
