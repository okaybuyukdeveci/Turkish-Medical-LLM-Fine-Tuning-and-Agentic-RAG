"""Connect to the validated Qdrant index and expose a LangChain retriever."""

from __future__ import annotations

from langchain_core.vectorstores import VectorStoreRetriever

from scripts.indexing.indexer import validate_existing_index
from scripts.retrieval.embeddings import E5Embeddings
from scripts.retrieval.qdrant_store import LocalQdrantStore
from scripts.settings import RagSettings


def load_retriever(settings: RagSettings) -> VectorStoreRetriever:
    """Load the existing index without creating or modifying it."""

    validate_existing_index(settings)
    embeddings = E5Embeddings(settings.embedding_model, settings.embedding_device)
    store = LocalQdrantStore(
        embedding=embeddings,
        storage_path=settings.qdrant_path,
        collection_name=settings.collection_name,
        vector_dimension=settings.embedding_dimension,
    )
    return store.as_retriever(k=settings.retrieval_k)
