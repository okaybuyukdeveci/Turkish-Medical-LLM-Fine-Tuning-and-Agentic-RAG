from scripts.retrieval.embeddings import E5Embeddings


class RecordingBackend:
    def __init__(self):
        self.documents = None
        self.query = None

    def embed_documents(self, texts):
        self.documents = texts
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.query = text
        return [1.0, 0.0]


def test_e5_prefixes_documents_and_queries():
    backend = RecordingBackend()
    embeddings = E5Embeddings(backend=backend)

    embeddings.embed_documents(["belge"])
    embeddings.embed_query("sorgu")

    assert backend.documents == ["passage: belge"]
    assert backend.query == "query: sorgu"
