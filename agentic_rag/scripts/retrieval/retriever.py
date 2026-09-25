"""Retrieve local parent documents using their Qdrant child embeddings."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from scripts.indexing.indexer import validate_existing_index
from scripts.retrieval.embeddings import E5Embeddings
from scripts.retrieval.parent_store import ParentDocumentStore
from scripts.retrieval.qdrant_store import LocalQdrantStore
from scripts.settings import RagSettings


class ParentDocumentRetriever(BaseRetriever):
    child_retriever: VectorStoreRetriever
    parent_store_path: Path

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        children = self.child_retriever.invoke(query)
        parent_ids = list(
            dict.fromkeys(child.metadata["parent_id"] for child in children)
        )
        return ParentDocumentStore(self.parent_store_path).get_many(parent_ids)


def load_retriever(settings: RagSettings) -> ParentDocumentRetriever:
    """Load the existing index without creating or modifying it."""

    validate_existing_index(settings)
    embeddings = E5Embeddings(settings.embedding_model, settings.embedding_device)
    store = LocalQdrantStore(
        embedding=embeddings,
        storage_path=settings.qdrant_path,
        collection_name=settings.collection_name,
        vector_dimension=settings.embedding_dimension,
    )
    return ParentDocumentRetriever(
        child_retriever=store.as_retriever(k=settings.retrieval_k),
        parent_store_path=settings.parent_store_path,
    )
