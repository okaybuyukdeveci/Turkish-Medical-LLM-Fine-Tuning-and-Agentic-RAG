from scripts.retrieval.main_retriever import build_retriever

retriever = build_retriever()

def retrieve_documents(query, retriever):
    retriever.invoke(query)
