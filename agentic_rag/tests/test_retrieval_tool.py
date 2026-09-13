from langchain_core.documents import Document

from scripts.tools.retrieval_tool import create_retrieve_documents_tool, format_evidence


class Retriever:
    def __init__(self, documents=None, error=None):
        self.documents = documents or []
        self.error = error

    def invoke(self, query):
        if self.error:
            raise self.error
        return self.documents


def test_evidence_contains_source_labels():
    document = Document(
        page_content="# Kalp\n## Tedavi\n\nKanıt",
        metadata={
            "source": "Dahiliye/a.md",
            "title": "Kalp",
            "subtitle": "Tedavi",
            "chunk_index": 3,
        },
    )

    content = format_evidence([document])

    assert "[EVIDENCE 1]" in content
    assert "source: Dahiliye/a.md" in content
    assert "subtitle: Tedavi" in content
    assert "chunk: 3" in content


def test_tool_handles_empty_results_and_failures():
    empty_content, empty_artifact = create_retrieve_documents_tool(Retriever()).func("x")
    error_content, error_artifact = create_retrieve_documents_tool(
        Retriever(error=RuntimeError("private detail"))
    ).func("x")

    assert empty_content.startswith("[NO_EVIDENCE]")
    assert empty_artifact == {"status": "ok", "sources": []}
    assert error_content.startswith("[RETRIEVAL_ERROR]")
    assert error_artifact["status"] == "error"
    assert "private detail" not in error_content
