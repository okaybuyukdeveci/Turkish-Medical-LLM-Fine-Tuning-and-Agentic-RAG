import os

import pytest

from scripts.indexing.audit import audit_corpus
from scripts.settings import RagSettings


@pytest.mark.corpus
@pytest.mark.skipif(
    os.getenv("RUN_CORPUS_TESTS") != "1",
    reason="set RUN_CORPUS_TESTS=1 to scan the complete corpus",
)
def test_complete_corpus_chunk_contract():
    report = audit_corpus(RagSettings.from_env())

    assert report["document_count"] == 24
    assert report["removed_empty_image_references"] == 3023
    assert report["non_substantive_chunks"] == 0
    assert report["remaining_empty_image_references"] == 0
    assert report["chunk_count"] > 0
    assert report["token_counts"]["maximum"] <= 384
    assert len(report["chunks_per_category"]) == 11
