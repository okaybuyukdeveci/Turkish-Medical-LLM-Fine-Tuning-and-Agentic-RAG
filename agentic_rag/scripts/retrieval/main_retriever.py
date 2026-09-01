from pathlib import Path

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from scripts.retrieval.data_loader import load_documents
from scripts.retrieval.document_chunker import chunk_documents
from scripts.retrieval.qdrant_vector_store import LocalQdrantVectorStore


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
QDRANT_STORAGE_DIR = PROJECT_DIR / "qdrant_storage"
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


def build_retriever() -> VectorStoreRetriever:
    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    local_store = LocalQdrantVectorStore(
        embedding=embedding,
        storage_path=QDRANT_STORAGE_DIR,
    )
    local_store.build(chunks)

    print(
        f"Indexed {len(chunks)} chunks from {len(documents)} documents "
        f"in {QDRANT_STORAGE_DIR}"
    )
    return local_store.as_retriever()


if __name__ == "__main__":
    build_retriever()
