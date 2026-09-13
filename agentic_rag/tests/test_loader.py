import unicodedata

import pytest

from scripts.indexing.loader import load_markdown_documents


def test_loader_is_recursive_sorted_and_normalizes_metadata(tmp_path):
    nested = tmp_path / "DAHİLİYE"
    nested.mkdir()
    (nested / "B.pdf.md").write_text("ikinci", encoding="utf-8")
    (nested / "A.md").write_text("I\u0307lk", encoding="utf-8")

    documents = load_markdown_documents(tmp_path)

    assert [document.metadata["document_title"] for document in documents] == ["A", "B"]
    assert all(unicodedata.is_normalized("NFC", document.page_content) for document in documents)
    assert documents[0].metadata == {
        "source": "DAHİLİYE/A.md",
        "category": "DAHİLİYE",
        "document_title": "A",
    }


def test_loader_rejects_missing_or_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_markdown_documents(tmp_path / "missing")
    with pytest.raises(ValueError, match="No Markdown"):
        load_markdown_documents(tmp_path)
