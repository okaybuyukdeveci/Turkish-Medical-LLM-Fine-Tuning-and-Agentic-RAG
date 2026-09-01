from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Split documents while retaining their source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for document in documents:
        document_chunks = splitter.split_documents([document])
        for chunk_number, chunk in enumerate(document_chunks):
            chunk.metadata["chunk"] = chunk_number
            chunks.append(chunk)

    return chunks
