from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore


class LocalQdrantVectorStore:
    """Build and access a LangChain Qdrant store persisted on local disk."""

    def __init__(
        self,
        embedding: Embeddings,
        storage_path: str | Path,
        collection_name: str = "turkish_medical_documents",
    ):
        self.embedding = embedding
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.vector_store: QdrantVectorStore | None = None

    def build(self, documents: list[Document]) -> QdrantVectorStore:
        if not documents:
            raise ValueError("At least one document is required to build the vector store.")

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.vector_store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embedding,
            path=str(self.storage_path),
            collection_name=self.collection_name,
            force_recreate=True,
        )
        return self.vector_store

    def connect(self) -> QdrantVectorStore:
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding,
            path=str(self.storage_path),
            collection_name=self.collection_name,
        )
        return self.vector_store

    def as_retriever(self, k: int = 6) -> VectorStoreRetriever:
        if self.vector_store is None:
            self.connect()
        return self.vector_store.as_retriever(search_kwargs={"k": k})
