"""Retrieval tools bound to the Responder."""

from langchain_core.tools import tool

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
