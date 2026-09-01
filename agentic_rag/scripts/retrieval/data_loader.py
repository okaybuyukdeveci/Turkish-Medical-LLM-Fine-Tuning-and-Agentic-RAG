from pathlib import Path

from langchain_core.documents import Document


def load_documents(data_dir: str | Path) -> list[Document]:
    """Read every Markdown file under data_dir into a LangChain document."""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")

    documents = []
    for file_path in sorted(data_path.rglob("*.md")):
        relative_path = file_path.relative_to(data_path)
        documents.append(
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={
                    "source": relative_path.as_posix(),
                    "category": relative_path.parent.as_posix(),
                    "title": file_path.name.removesuffix(".md"),
                },
            )
        )

    if not documents:
        raise ValueError(f"No Markdown files found in: {data_path}")

    return documents
