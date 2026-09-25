"""Persistent local store for the full parent documents returned by retrieval."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

from langchain_core.documents import Document


class ParentDocumentStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, documents: list[Document]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix="parent_documents_", suffix=".sqlite", dir=self.path.parent
        )
        os.close(handle)
        temporary_path = Path(temporary_name)
        try:
            with sqlite3.connect(temporary_path) as connection:
                connection.execute(
                    "CREATE TABLE parents (id TEXT PRIMARY KEY, content TEXT NOT NULL, metadata TEXT NOT NULL)"
                )
                connection.executemany(
                    "INSERT INTO parents VALUES (?, ?, ?)",
                    (
                        (
                            document.metadata["chunk_id"],
                            document.page_content,
                            json.dumps(document.metadata, ensure_ascii=False),
                        )
                        for document in documents
                    ),
                )
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def count(self) -> int:
        if not self.path.is_file():
            return 0
        with sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM parents").fetchone()[0])

    def get_many(self, ids: list[str]) -> list[Document]:
        if not ids:
            return []
        with sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute(
                f"SELECT id, content, metadata FROM parents WHERE id IN ({','.join('?' for _ in ids)})",
                ids,
            ).fetchall()
        found = {
            identifier: Document(page_content=content, metadata=json.loads(metadata))
            for identifier, content, metadata in rows
        }
        if missing := set(ids) - found.keys():
            raise RuntimeError(f"Parent documents missing from local store: {sorted(missing)}")
        return [found[identifier] for identifier in ids]
