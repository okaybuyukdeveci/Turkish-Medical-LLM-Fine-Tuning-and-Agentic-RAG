"""Tools bound to the two agents.

- transfer_to_retriever: bound to the Responder (Agent 2). Acts as a handoff —
  calling it is the signal that this conversation needs retrieval. Routing
  itself happens via graph edges, not inside the tool.
- retrieve_documents: bound to the Retriever (Agent 1). Currently backed by
  Wikipedia as a test stand-in for a real vector store; swap the body for the
  real embedding + search call later.
"""

from langchain_core.tools import tool


@tool
def transfer_to_retriever(request: str) -> str:
    """Hand off to the Retriever agent when answering the user requires fetching documents from the knowledge base.

    Args:
        request: Short description of what information is needed, for the Retriever to work from.
    """

    return f"Transferred to retriever: {request}"

# Instead of a static @tool decorator, define a factory function
def create_retrieve_documents_tool(compressed_retriever) -> tool:
    """Creates a retrieve_documents tool bound to the provided retriever instance."""
    @tool
    def retrieve_documents(query: str) -> str:
        """Retrieve relevant passages from the knowledge base for the given search query.
    
        Args:
            query: The search query to run. Works best as a concise topic/title
                (e.g. "Python programming language") rather than a full question.
        """

        # bu yöntem doğru değil daha sonradan değişmesi lazım
        docs = compressed_retriever.invoke(query)
    
        # Format the documents into a single string for the LLM
        formatted_docs = "\n\n".join(
            [f"Document:\n{doc.page_content}" for doc in docs]
        )
        
        return formatted_docs
    return retrieve_documents
