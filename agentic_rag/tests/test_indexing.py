import hashlib
import uuid
from dataclasses import replace

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from scripts.indexing import indexer
from scripts.indexing.indexer import StaleIndexError, ensure_index
from scripts.retrieval import retriever as retriever_module
from scripts.retrieval.parent_store import ParentDocumentStore
from scripts.retrieval.qdrant_store import LocalQdrantStore
from scripts.retrieval.retriever import ParentDocumentRetriever
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


def test_parent_retriever_deduplicates_children(tmp_path):
    parent = qdrant_document("# title\n\nparent evidence", 0)
    children = [qdrant_document("alpha first", 1), qdrant_document("alpha second", 2)]
    for child in children:
        child.metadata["parent_id"] = parent.metadata["chunk_id"]
    parent_path = tmp_path / "qdrant" / "parent_documents.sqlite"
    ParentDocumentStore(parent_path).write([parent])
    store = LocalQdrantStore(FakeEmbeddings(), tmp_path / "qdrant", "test", 2)
    store.build(children)
    retriever = ParentDocumentRetriever(
        child_retriever=store.as_retriever(k=2), parent_store_path=parent_path
    )

    assert [document.page_content for document in retriever.invoke("alpha")] == [
        parent.page_content
    ]
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
        retriever_module, "E5Embeddings", lambda *args, **kwargs: FakeEmbeddings()
    )
    def fake_chunking(documents, parent_config, child_config):
        parent = qdrant_document(documents[0].page_content, 0)
        child = qdrant_document(documents[0].page_content, 1)
        child.metadata["parent_id"] = parent.metadata["chunk_id"]
        return [parent], [child]

    monkeypatch.setattr(indexer, "chunk_parent_child_documents", fake_chunking)

    created = ensure_index(settings)
    reused = ensure_index(settings)

    assert created.status == "created"
    assert reused.status == "reused"
    assert reused.parent_count == 1
    assert reused.chunk_count == 1
    assert ParentDocumentStore(settings.parent_store_path).count() == 1

    parent_retriever = retriever_module.load_retriever(settings)
    assert parent_retriever.invoke("alpha")[0].page_content == "# title\nbody"
    parent_retriever.child_retriever.vectorstore.client.close()

    source.write_text("# title\nchanged body", encoding="utf-8")
    with pytest.raises(StaleIndexError, match="stale"):
        ensure_index(settings)

    rebuilt = ensure_index(settings, force_recreate=True)
    assert rebuilt.status == "rebuilt"
    assert rebuilt.corpus_hash != created.corpus_hash

    changed_parent_settings = replace(settings, parent_chunk_size_tokens=3000)
    with pytest.raises(StaleIndexError, match="stale"):
        ensure_index(changed_parent_settings)

    settings.parent_store_path.unlink()
    with pytest.raises(StaleIndexError, match="parent document store"):
        ensure_index(settings)
