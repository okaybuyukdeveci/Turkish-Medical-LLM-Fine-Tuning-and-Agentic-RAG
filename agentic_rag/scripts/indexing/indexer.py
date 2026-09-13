"""Build, validate, and fingerprint the persisted Qdrant index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.indexing.chunker import ChunkingConfig, chunk_documents
from scripts.indexing.loader import load_markdown_documents
from scripts.retrieval.embeddings import E5Embeddings
from scripts.retrieval.qdrant_store import LocalQdrantStore
from scripts.settings import RagSettings


INDEX_SCHEMA_VERSION = 1


class StaleIndexError(RuntimeError):
    """Raised when an existing index no longer describes the current corpus."""


@dataclass(frozen=True)
class IndexReport:
    status: str
    document_count: int
    chunk_count: int
    collection_name: str
    storage_path: str
    corpus_hash: str


def _file_records(data_dir: Path) -> list[dict[str, str]]:
    records = []
    for file_path in sorted(data_dir.rglob("*.md"), key=lambda path: path.as_posix()):
        records.append(
            {
                "path": unicodedata.normalize(
                    "NFC", file_path.relative_to(data_dir).as_posix()
                ),
                "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
            }
        )
    if not records:
        raise ValueError(f"No Markdown files found in: {data_dir}")
    return records


def expected_manifest(settings: RagSettings) -> dict:
    files = _file_records(settings.data_dir)
    serialized_files = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "corpus_hash": hashlib.sha256(serialized_files.encode("utf-8")).hexdigest(),
        "files": files,
        "document_count": len(files),
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "chunk_size_tokens": settings.chunk_size_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "collection_name": settings.collection_name,
    }


def _read_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleIndexError(
            f"Index manifest is unreadable: {path}. Rebuild with --force."
        ) from exc


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_path, path)


def _store(
    settings: RagSettings, embeddings: E5Embeddings | None = None
) -> LocalQdrantStore:
    return LocalQdrantStore(
        embedding=embeddings,
        storage_path=settings.qdrant_path,
        collection_name=settings.collection_name,
        vector_dimension=settings.embedding_dimension,
    )


def _manifest_without_count(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != "chunk_count"}


def validate_existing_index(settings: RagSettings) -> IndexReport:
    settings.validate()
    expected = expected_manifest(settings)
    manifest = _read_manifest(settings.manifest_path)
    if manifest is None:
        raise StaleIndexError(
            f"No index manifest exists at {settings.manifest_path}. "
            "Build the index with `python -m scripts.indexing.indexer`."
        )
    if _manifest_without_count(manifest) != expected:
        raise StaleIndexError(
            "The Qdrant index is stale because the corpus or indexing settings changed. "
            "Review the change, then rebuild with "
            "`python -m scripts.indexing.indexer --force`."
        )

    store = _store(settings)
    if not store.collection_exists():
        raise StaleIndexError(
            "The index manifest exists but the Qdrant collection is missing. "
            "Rebuild with `python -m scripts.indexing.indexer --force`."
        )
    point_count = store.count()
    if point_count != manifest.get("chunk_count"):
        raise StaleIndexError(
            "The Qdrant point count does not match the manifest. "
            "Rebuild with `python -m scripts.indexing.indexer --force`."
        )
    return IndexReport(
        status="reused",
        document_count=expected["document_count"],
        chunk_count=point_count,
        collection_name=settings.collection_name,
        storage_path=str(settings.qdrant_path),
        corpus_hash=expected["corpus_hash"],
    )


def ensure_index(settings: RagSettings, force_recreate: bool = False) -> IndexReport:
    """Create a missing index, reuse a valid one, or refuse a stale one."""

    settings.validate()
    expected = expected_manifest(settings)
    manifest = _read_manifest(settings.manifest_path)
    store = _store(settings)
    collection_exists = store.collection_exists()

    if collection_exists and not force_recreate:
        return validate_existing_index(settings)
    if manifest is not None and not collection_exists and not force_recreate:
        raise StaleIndexError(
            "An index manifest exists but its Qdrant collection is missing. "
            "Rebuild with `python -m scripts.indexing.indexer --force`."
        )

    documents = load_markdown_documents(settings.data_dir)
    chunks = chunk_documents(
        documents,
        ChunkingConfig(
            tokenizer_name=settings.embedding_model,
            max_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        ),
    )
    embeddings = E5Embeddings(settings.embedding_model, settings.embedding_device)
    point_count = _store(settings, embeddings).build(chunks)
    if point_count != len(chunks):
        raise RuntimeError(
            f"Qdrant stored {point_count} points but {len(chunks)} chunks were produced"
        )

    completed_manifest = {**expected, "chunk_count": point_count}
    _write_manifest(settings.manifest_path, completed_manifest)
    return IndexReport(
        status="rebuilt" if collection_exists or manifest is not None else "created",
        document_count=len(documents),
        chunk_count=point_count,
        collection_name=settings.collection_name,
        storage_path=str(settings.qdrant_path),
        corpus_hash=expected["corpus_hash"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="recreate an existing or stale index"
    )
    args = parser.parse_args()
    report = ensure_index(RagSettings.from_env(), force_recreate=args.force)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
