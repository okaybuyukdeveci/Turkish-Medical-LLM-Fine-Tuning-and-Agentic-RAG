from scripts.retrieval.milvus_vector_store import MilvusVectorStore
from scripts.retrieval.retrieval_system import RetrievalSystem
import redis, pickle


class BuildRetriever:

    def __init__(self, config_path):
        self.r_client = redis.from_url(
            url="rediss://default:redis_practicus@practicus-redis-master.prt-ns-redis.svc.cluster.local:6379/0",
            decode_responses=False,
            ssl_cert_reqs=None,
        )
        self.config_path = config_path

        self._init_redis()

    def redis_init_pickle(self, redis_ns):
        blob = self.r_client.get(redis_ns)
        loaded_docs = pickle.loads(blob)
        return loaded_docs

    def _init_redis(self):


        self.kurumsal = self.redis_init_pickle("prod_vodex_techbot_docs_public_pickle")
        self.vbu_chunks = self.redis_init_pickle("prod_vbts_techbot_docs_private_pickle")
        
        raw = self.r_client.get("prod_vbts_techbot_table_list_private_pickle")
        self.table_list = pickle.loads(pickle.loads(raw))

    def _init_retriever(self):
        vbu_store = MilvusVectorStore("prod_vbts_techbot_private_store", "prod_vbu", "prod_vbu_ro_user", "milvus_prod_vbu_ro_secret")
        vbu_store._init_embedding(self.config_path)

        vbu_db = vbu_store.connect_db("prod_vbts_techbot_private", vbu_store.embedding_model)
        vbts_parent_child_retriever = vbu_store.initilize_parent_child()

        vodex_store = MilvusVectorStore("prod_vodex_techbot_public_store", "prod_vbu", "prod_vbu_ro_user", "milvus_prod_vbu_ro_secret")
        vodex_db = vodex_store.connect_db("prod_vodex_techbot_public", vbu_store.embedding_model)
        vodex_parent_child_retriever = vodex_store.initilize_parent_child()

        vbts_retrieval_system = RetrievalSystem(vbts_parent_child_retriever, [])
        vodex_retrieval_system = RetrievalSystem(vodex_parent_child_retriever)
        vbts_retrieval_system._init_online_reranker(self.config_path)

        vbts_retriever = vbts_retrieval_system._build_retriever(self.vbu_chunks, vbts_retrieval_system.reranker)
        vodex_retriever = vodex_retrieval_system._build_retriever(self.kurumsal, vbts_retrieval_system.reranker)
        compressed_retriever = vbts_retrieval_system.compress(vbts_retriever, vodex_retriever)

        return compressed_retriever



