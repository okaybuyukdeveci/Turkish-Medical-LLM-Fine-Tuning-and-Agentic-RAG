"""Embedding adapter for asymmetric multilingual E5 retrieval."""

from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class E5Embeddings(Embeddings):
    """Apply the prefixes required by E5 before delegating to an embedder."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
        *,
        backend: EmbeddingBackend | None = None,
    ) -> None:
        if backend is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-huggingface is required for local E5 embeddings"
                ) from exc
            backend = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        self._backend = backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_documents([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._backend.embed_query(f"query: {text}")
