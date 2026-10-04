import re, unicodedata, torch, gc
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker
from scripts.indexing.loader import load_markdown_documents
from scripts.indexing.chunker import ChunkingConfig, chunk_parent_child_documents

LIMIT, ADAY_N, TUT_N = 6000, 30, 6

print("parent chunk'lar uretiliyor (Qdrant'a dokunulmuyor)")
belgeler = load_markdown_documents(DATA)
parents, _ = chunk_parent_child_documents(
    belgeler,
    parent_config=ChunkingConfig(max_tokens=settings.parent_chunk_size_tokens, overlap_tokens=0),
    child_config=ChunkingConfig(max_tokens=settings.chunk_size_tokens,
                                overlap_tokens=settings.chunk_overlap_tokens),
)
print(f"parent: {len(parents)}")

def jeton(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2]

print("BM25 kuruluyor")
bm25 = BM25Okapi([jeton(p.page_content) for p in parents])
icerik_idx = {p.page_content: i for i, p in enumerate(parents)}

def rrf(siralar, k=60):
    puan = {}
    for sira in siralar:
        for yer, kimlik in enumerate(sira):
            puan[kimlik] = puan.get(kimlik, 0.0) + 1.0 / (k + yer + 1)
    return [kimlik for kimlik, _ in sorted(puan.items(), key=lambda x: -x[1])]

print("reranker yukleniyor")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda:0"])

print(f"\nhibrit aday havuzu ({ADAY_N}) + yeniden siralama")
eski_k = retriever.child_retriever.search_kwargs.get("k", settings.retrieval_k)
retriever.child_retriever.search_kwargs["k"] = ADAY_N
HR_docs = []
for i, r in enumerate(test_rows, 1):
    yogun = retriever.invoke(r["question"])
    yogun_idx = [x for x in (icerik_idx.get(d.page_content) for d in yogun) if x is not None]
    seyrek_idx = list(bm25.get_top_n(jeton(r["question"]), range(len(parents)), n=ADAY_N))
    aday = [parents[j] for j in rrf([yogun_idx, seyrek_idx])[:ADAY_N]]
    if len(aday) <= TUT_N:
        HR_docs.append(aday)
    else:
        skor = reranker.compute_score([[r["question"], d.page_content[:2000]] for d in aday],
                                      normalize=True)
        sirali = sorted(zip(skor, range(len(aday))), reverse=True)
        HR_docs.append([aday[j] for _, j in sirali[:TUT_N]])
    print(f"  {i}/{len(test_rows)}", end="\r")
retriever.child_retriever.search_kwargs["k"] = eski_k
print()

del reranker; gc.collect(); torch.cuda.empty_cache()

HR_pr = [prompt(SYSTEM_RAG, f"KAYNAK METİN:\n{baglam_kur(d, LIMIT)}\n\n---\n\nSORU:\n{r['question']}")
         for r, d in zip(test_rows, HR_docs)]

adapter_sec("v2")
HR_outs = uret(HR_pr, 64, bs=4, etiket="[hib+rr v2] ")
HR_acc, HR_pred = dogruluk(HR_outs, test_rows, "[hib+rr v2] ")

HR_v1_acc, HR_v1_pred = None, None
if V1_VAR:
    adapter_sec("v1")
    HR_v1_outs = uret(HR_pr, 64, bs=4, etiket="[hib+rr v1] ")
    HR_v1_acc, HR_v1_pred = dogruluk(HR_v1_outs, test_rows, "[hib+rr v1] ")
    adapter_sec("v2")
    print(f"\nayni belgeler, ayni prompt — v1 {HR_v1_acc:.1%} vs v2 {HR_acc:.1%}")
else:
    print(f"\nv1 olculemedi; gecen seferki v1 sayisi 88,9% (ayri oturum, esli test yapilamaz)")
