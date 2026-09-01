"""
Document retrieval system with reranking capabilities
"""

from libs.langchain_practicus_reranker import PracticusDocumentCompressor
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from typing import List
from langchain.docstore.document import Document
from loguru import logger
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import json
from langfuse import observe, get_client


class RetrievalSystem:
    """Document retrieval and reranking system"""

    def __init__(self, parent_child_retriever=None, table_list=None):
        """Initialize retrieval system"""
        self.langfuse = get_client()
        self.model_kwargs = {"device": 0}
        self.model_name = "BAAI/bge-reranker-v2-m3"
        self.k = 6
        self.context_size = 3

        self.chat_history = []
        self.weights = (0.25, 0.25, 0, 0)
        self.table_list = table_list

        self.parent_child_retriever = parent_child_retriever
        self.reranker = None
        self.retriever = None

    def _init_online_reranker(self, config_path):
        with open(config_path, "r") as file:
            data = json.load(file)["reranker"]
        endpoint_url = data["endpoint_url"]
        api_token = data["api_token"]
        model_name = data["model_name"]

        self.model = PracticusDocumentCompressor(endpoint_url=endpoint_url, api_token=api_token, model=model_name)

        self.reranker = CrossEncoderReranker(model=self.model, top_n=self.context_size)
        # logger.info("Online reranker connected successfully.")

    def _init_reranker(self):
        """Initialize cross-encoder reranker"""
        self.model = HuggingFaceCrossEncoder(model_name=self.model_name, model_kwargs=self.model_kwargs)
        self.reranker = CrossEncoderReranker(model=self.model, top_n=self.context_size)
        # logger.info("Reranker initialized successfully.")

    def _build_retriever(self, doc_chunks, reranker, k=6):
        """Build retrieval pipeline with reranking"""
        try:
            bm25_retriever = BM25Retriever.from_documents(doc_chunks)
            bm25_retriever.k = k
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, self.parent_child_retriever], weights=[0.5, 0.5]
            )

            self.vector_retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=self.ensemble_retriever
            )
            # logger.info("Retriever with reranking has been initialized.")
            return self.vector_retriever

        except Exception as e:
            logger.error(
                f"Retriever initialization has been failed. You may have forgotten to initialize the reranker. \n {e}"
            )
            return

    def compress(self, retriever_1, retriever_2):
        try:
            compressor = CrossEncoderReranker(model=self.model, top_n=2)

            ensemble = EnsembleRetriever(retrievers=[retriever_1, retriever_2], weights=[0.5, 0.5])
            self.compressed_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble
            )
            # logger.info("Retrievers are successfully compressed.")
            return self.compressed_retriever

        except Exception as e:
            logger.error(f"Retreivers have failed to be compressed. \n {e}")
            return

    def flatten_context(self, docs):
        result = []
        for doc in docs:
            try:
                page_title = doc.metadata["page_title"]
            except:
                page_title = doc.metadata["title"]

            formatted_string = f"{page_title} Piece {doc.metadata['chunk']}\n{doc.page_content}"
            result.append(formatted_string)
            logger.debug(f"Retrieved {page_title} document")
        final_output = "\n\n".join(result)
        return final_output

    def table_extract(self, context_doc, table_list):
        page_id = context_doc[0].metadata["page_id"]

        for table in table_list:
            if table["page_id"] == page_id:
                return table["content"]

    @observe(name="retrieve", as_type="retriever")
    def retrieve(self, question):
        context_doc = self.vector_retriever.invoke(question)
        try:
            table_html = self.table_extract(context_doc, self.table_list)
            context = self.flatten_context(context_doc)
        except:
            print(context_doc)

        return context
