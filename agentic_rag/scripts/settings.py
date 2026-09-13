"""Environment-backed configuration for the medical RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _environment_path(name: str, default: Path, relative_to: Path) -> Path:
    path = Path(os.getenv(name, str(default))).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


@dataclass(frozen=True)
class RagSettings:
    """Validated runtime settings shared by indexing and agent execution."""

    project_dir: Path = PROJECT_DIR
    data_dir: Path = PROJECT_DIR / "data"
    qdrant_path: Path = PROJECT_DIR / "qdrant_storage"
    collection_name: str = "turkish_medical_documents"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dimension: int = 768
    embedding_device: str = "cpu"
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 48
    retrieval_k: int = 6
    together_model: str = "zai-org/GLM-5.3-Flash"
    together_api_key: str | None = field(default=None, repr=False)

    @property
    def manifest_path(self) -> Path:
        return self.qdrant_path / "index_manifest.json"

    def validate(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if not 0 <= self.chunk_overlap_tokens < self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        if self.retrieval_k <= 0:
            raise ValueError("retrieval_k must be positive")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

    def require_together_api_key(self) -> str:
        if not self.together_api_key:
            raise RuntimeError(
                "TOGETHER_API_KEY is required to run the agent. "
                "Copy .env.example to .env and add your key."
            )
        return self.together_api_key

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> "RagSettings":
        """Load `.env` without overriding values already exported by the caller."""

        dotenv_path = Path(env_file) if env_file else PROJECT_DIR / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=False)

        project_dir = _environment_path("RAG_PROJECT_DIR", PROJECT_DIR, PROJECT_DIR)
        settings = cls(
            project_dir=project_dir,
            data_dir=_environment_path(
                "RAG_DATA_DIR", project_dir / "data", project_dir
            ),
            qdrant_path=_environment_path(
                "QDRANT_PATH", project_dir / "qdrant_storage", project_dir
            ),
            collection_name=os.getenv("QDRANT_COLLECTION", "turkish_medical_documents"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
            ),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "384")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "48")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "6")),
            together_model=os.getenv("TOGETHER_MODEL", "zai-org/GLM-5.3-Flash"),
            together_api_key=os.getenv("TOGETHER_API_KEY"),
        )
        settings.validate()
        return settings
