import copy

from langchain_core.documents import Document

from scripts.indexing.chunker import (
    ChunkingConfig,
    chunk_documents,
    chunk_parent_child_documents,
)


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(text)


def make_document(text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "Dahiliye/source.md",
            "category": "Dahiliye",
            "document_title": "Source",
        },
    )


def test_literal_titles_context_and_artifact_filtering():
    document = make_document(
        "Önsöz metni.\n"
        "# Kardiyoloji\n"
        "## Boş Başlık\n"
        "![](images/missing.jpg)\n"
        "## Kalp Yetmezliği\n"
        "Tedavi ve bulgular.\n"
        "### Ayrıntı\n"
        "Ek açıklama."
    )

    chunks = chunk_documents(
        [document],
        ChunkingConfig(max_tokens=120, overlap_tokens=10),
        tokenizer=CharacterTokenizer(),
    )

    assert len(chunks) == 2
    assert chunks[0].page_content == "# Source\n\nÖnsöz metni."
    assert chunks[1].page_content.startswith(
        "# Kardiyoloji\n## Kalp Yetmezliği\n\nTedavi ve bulgular."
    )
    assert "### Ayrıntı" in chunks[1].page_content
    assert all("Boş Başlık" not in chunk.page_content for chunk in chunks)
    assert all("images/missing.jpg" not in chunk.page_content for chunk in chunks)
    assert chunks[1].metadata["document_title"] == "Source"
    assert chunks[1].metadata["title"] == "Kardiyoloji"
    assert chunks[1].metadata["subtitle"] == "Kalp Yetmezliği"
    assert chunks[1].metadata["breadcrumb"] == "Dahiliye > Kardiyoloji > Kalp Yetmezliği"


def test_long_sections_repeat_context_and_respect_token_limit():
    document = make_document(
        "# Başlık\n## Alt Başlık\n" + ("kelime " * 80) + "\n\n##"
    )
    config = ChunkingConfig(max_tokens=80, overlap_tokens=12)

    chunks = chunk_documents([document], config, tokenizer=CharacterTokenizer())

    assert len(chunks) > 1
    assert all(chunk.page_content.startswith("# Başlık\n## Alt Başlık\n\n") for chunk in chunks)
    assert all(len(chunk.page_content) <= config.max_tokens for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk.page_content.split("\n\n", 1)[-1] != "##" for chunk in chunks)


def test_chunking_is_deterministic_and_does_not_mutate_input():
    document = make_document("# Başlık\n## Alt\n- Bir\n- İki\n\n| A | B |\n|---|---|\n| 1 | 2 |")
    original = copy.deepcopy(document)
    config = ChunkingConfig(max_tokens=100, overlap_tokens=10)

    first = chunk_documents([document], config, tokenizer=CharacterTokenizer())
    second = chunk_documents([document], config, tokenizer=CharacterTokenizer())

    assert document == original
    assert [(chunk.page_content, chunk.metadata) for chunk in first] == [
        (chunk.page_content, chunk.metadata) for chunk in second
    ]
    assert "| 1 | 2 |" in first[0].page_content


def test_parent_child_chunks_link_and_respect_both_limits():
    document = make_document("# Başlık\n## Alt\n" + "kelime " * 100)
    tokenizer = CharacterTokenizer()
    parents, children = chunk_parent_child_documents(
        [document],
        ChunkingConfig(max_tokens=120, overlap_tokens=0),
        ChunkingConfig(max_tokens=60, overlap_tokens=8),
        tokenizer=tokenizer,
    )

    parent_ids = {parent.metadata["chunk_id"] for parent in parents}
    assert len(parents) > 1
    assert len(children) > len(parents)
    assert {child.metadata["parent_id"] for child in children} == parent_ids
    assert len({child.metadata["chunk_id"] for child in children}) == len(children)
    assert all(len(parent.page_content) <= 120 for parent in parents)
    assert all(len(child.page_content) <= 60 for child in children)
