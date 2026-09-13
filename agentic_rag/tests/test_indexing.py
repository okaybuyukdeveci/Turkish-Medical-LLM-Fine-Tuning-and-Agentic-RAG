import hashlib
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from scripts.indexing import indexer
from scripts.indexing.indexer import StaleIndexError, ensure_index
from scripts.retrieval.qdrant_store import LocalQdrantStore
from scripts.settings import RagSettings


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


def qdrant_document(content, number):
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    return Document(
        page_content=content,
        metadata={
            "source": "source.md",
            "category": "test",
            "document_title": "source",
            "title": "title",
            "breadcrumb": "test > title",
            "chunk_index": number,
            "content_hash": content_hash,
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{number}:{content_hash}")),
        },
    )


def test_local_qdrant_build_count_and_metadata_round_trip(tmp_path):
    store = LocalQdrantStore(FakeEmbeddings(), tmp_path / "qdrant", "test", 2)

    assert store.build([qdrant_document("alpha evidence", 0), qdrant_document("beta", 1)]) == 2
    assert store.collection_exists()
    assert store.count() == 2

    result = store.as_retriever(k=1).invoke("alpha")[0]
    assert result.page_content == "alpha evidence"
    assert result.metadata["chunk_index"] == 0
    store.close()


def test_index_manifest_reuse_staleness_and_force_rebuild(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = data_dir / "source.md"
    source.write_text("# title\nbody", encoding="utf-8")
    settings = RagSettings(
        project_dir=tmp_path,
        data_dir=data_dir,
        qdrant_path=tmp_path / "qdrant",
        collection_name="test",
        embedding_model="fake",
        embedding_dimension=2,
        chunk_size_tokens=20,
        chunk_overlap_tokens=2,
        retrieval_k=1,
    )
    monkeypatch.setattr(indexer, "E5Embeddings", lambda *args, **kwargs: FakeEmbeddings())
    monkeypatch.setattr(
        indexer,
        "chunk_documents",
        lambda documents, config: [qdrant_document(documents[0].page_content, 0)],
    )

    created = ensure_index(settings)
    reused = ensure_index(settings)

    assert created.status == "created"
    assert reused.status == "reused"
    assert reused.chunk_count == 1

    source.write_text("# title\nchanged body", encoding="utf-8")
    with pytest.raises(StaleIndexError, match="stale"):
        ensure_index(settings)

    rebuilt = ensure_index(settings, force_recreate=True)
    assert rebuilt.status == "rebuilt"
    assert rebuilt.corpus_hash != created.corpus_hash
