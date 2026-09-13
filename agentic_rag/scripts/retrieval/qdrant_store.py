"""Small local-Qdrant lifecycle wrapper used by indexing and retrieval."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


class LocalQdrantStore:
    def __init__(
        self,
        embedding: Embeddings | None,
        storage_path: str | Path,
        collection_name: str,
        vector_dimension: int,
    ) -> None:
        self.embedding = embedding
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self._client = None
        self._vector_store = None

    def _new_client(self):
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RuntimeError(
                "qdrant-client is required. Install the packages in requirements.txt."
            ) from exc
        self.storage_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(self.storage_path))

    def collection_exists(self) -> bool:
        client = self._new_client()
        try:
            return client.collection_exists(self.collection_name)
        finally:
            client.close()

    def count(self) -> int:
        client = self._new_client()
        try:
            if not client.collection_exists(self.collection_name):
                return 0
            return int(client.count(self.collection_name, exact=True).count)
        finally:
            client.close()

    def build(self, documents: list[Document]) -> int:
        if not documents:
            raise ValueError("At least one document is required to build the index")
        if self.embedding is None:
            raise RuntimeError("An embedding implementation is required to build the index")

        try:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import models
        except ImportError as exc:
            raise RuntimeError(
                "langchain-qdrant and qdrant-client are required for indexing"
            ) from exc

        client = self._new_client()
        try:
            if client.collection_exists(self.collection_name):
                client.delete_collection(self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
                embedding=self.embedding,
            )
            vector_store.add_documents(
                documents,
                ids=[str(document.metadata["chunk_id"]) for document in documents],
            )
            return int(client.count(self.collection_name, exact=True).count)
        except Exception:
            if client.collection_exists(self.collection_name):
                client.delete_collection(self.collection_name)
            raise
        finally:
            client.close()

    def connect(self):
        if self.embedding is None:
            raise RuntimeError("An embedding implementation is required to query the index")
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as exc:
            raise RuntimeError(
                "langchain-qdrant is required. Install the packages in requirements.txt."
            ) from exc

        self._client = self._new_client()
        if not self._client.collection_exists(self.collection_name):
            self._client.close()
            self._client = None
            raise RuntimeError(
                f"Qdrant collection does not exist: {self.collection_name}. Build the index first."
            )
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self.collection_name,
            embedding=self.embedding,
        )
        return self._vector_store

    def as_retriever(self, k: int) -> VectorStoreRetriever:
        vector_store = self._vector_store or self.connect()
        return vector_store.as_retriever(search_kwargs={"k": k})

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._vector_store = None
