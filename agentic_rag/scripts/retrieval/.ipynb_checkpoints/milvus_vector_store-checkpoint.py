from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langchain_community.storage import RedisStore

from langfuse import observe, get_client
import practicuscore as prt

import redis
import json
from loguru import logger
import gc
import traceback

class MilvusVectorStore:
    """Vector database storage and retrieval"""

    def __init__(self, redis_store_name, milvus_db, username, secret_name):
        # self.store = InMemoryStore()
        self.embedding_model = None
        self.milvus = None
        self.child_chunk_size = 512
        self.child_chunk_overlap = 50
        self.search_type = "similarity_score_threshold"
        self.retrieval_k = 6
        self.score_threshold = 0.7
        self.langfuse = get_client()
        self.milvus_db = milvus_db
        self.username = username
        self.secret_name = secret_name

        self.r_client = redis.from_url(
            url="rediss://default:redis_practicus@practicus-redis-master.prt-ns-redis.svc.cluster.local:6379/0",
            decode_responses=False,
            ssl_cert_reqs=None,
        )

        fs = RedisStore(client=self.r_client, namespace=redis_store_name)
        self.store = create_kv_docstore(fs)
        self._init_test_token()
        

    def _init_token(self):
        self.milvus_token, age = prt.vault.get_secret(name="milvus_mladm_token", shared=True)

    def _init_test_token(self):
        #self.token, age = prt.vault.get_secret(self.secret_name, shared=True)
        self.token = "QQf4Fy8tS0i7iIWU"
        self.milvus_token = f"{self.username}:{self.token}"

    @observe(name="init_embedding", as_type="span")
    def _init_embedding(self, config_path):
        with open(config_path, "r") as file:
            data = json.load(file)["embedding"]
        endpoint_url = data["endpoint_url"]
        model_name = data["model_name"]
        api_token = data["api_token"]

        self.embedding_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_token,
            openai_api_base=endpoint_url,
            check_embedding_ctx_length=False
        )
        
    def _init_offline_embedding(self):
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="/home/ubuntu/my/version_control/genai-apphost-vbu_chatbot/indexing/=/home/ubuntu/.cache/models--jinaai--jina-embeddings-v3/snapshots/ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
            model_kwargs={
                "device": "cuda",           # ya da "cpu"
                "trust_remote_code": "True",  # bool
            },
            encode_kwargs={"normalize_embeddings": True},
        )

    def build_db(self, collection, embedding):
        try:
            self.milvus = Milvus(
                embedding_function=embedding,
                connection_args={
                    "uri": "http://practicus-milvus.prt-ns-milvus.svc.cluster.local",
                    "token": self.milvus_token,
                    "db_name": self.milvus_db,
                },
                collection_name=collection,
                consistency_level="Strong",
                index_params={
                    "index_type": "HNSW",
                    "params": {"M": 64, "efConstruction": 300},
                    "metric_type": "COSINE",
                },
                auto_id=True,
                drop_old=True,
                enable_dynamic_field=True,
            )
            # logger.info(f"Connection for {collection} DB has setted!")
            return self.milvus
            
        except Exception as e:
            logger.error(f"DB connection failed! {e}")
            print(traceback.format_exc())

    @observe(name="connect_db", as_type="span")
    def connect_db(self, collection, embedding):
        try:
            self.milvus = Milvus(
                embedding_function=embedding,
                connection_args={
                    "uri": "http://practicus-milvus.prt-ns-milvus.svc.cluster.local",
                    "token": self.milvus_token,
                    "db_name": self.milvus_db,
                },
                collection_name=collection,
                consistency_level="Strong",
                index_params={
                    "index_type": "HNSW",
                    "params": {"M": 64, "efConstruction": 300},
                    "metric_type": "COSINE",
                },
                auto_id=True,
                drop_old=False,
                enable_dynamic_field=True,
            )
            # logger.info(f"Connection for {collection} DB has setted!")
            return self.milvus
        except Exception as e:
            logger.error(f"DB connection failed! {e}")
            print(traceback.format_exc())

    def add_documents(self, documents, batch_size=100):
        if not hasattr(self, "milvus"):
            logger.error("Database not initialized. Call init_db first.")
            return

        # Add documents in batches
        total_batches = (len(documents) - 1) // batch_size + 1
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            current_batch = i // batch_size + 1
            # logger.info(f"Processing batch {current_batch}/{total_batches}, size {len(batch)}")

            try:
                self.parent_child_retriever.add_documents(batch)
                # logger.info(f"Batch {current_batch} processed successfully")
            except Exception as e:
                logger.error(f"Error processing batch {current_batch}: {str(e)}")

            gc.collect()

        # logger.info(f"Added {len(documents)} documents to the database in {total_batches} batches")

    @observe(name="initilize_parent_child", as_type="span")
    def initilize_parent_child(self, k=6):
        """Get parent child vector retriever"""
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            length_function=len,
            separators=["\n\n", "----", "\n\n", "\n", ". ", " ", ""],
        )

        self.parent_child_retriever = ParentDocumentRetriever(
            vectorstore=self.milvus,
            docstore=self.store,
            child_splitter=child_splitter,
            search_type=self.search_type,
            # search_type="score",
            search_kwargs={"k": k, "score_threshold": self.score_threshold, "param": {"ef": 50}},
        )
        # logger.info("Parent child system has been initialized.")
        return self.parent_child_retriever
