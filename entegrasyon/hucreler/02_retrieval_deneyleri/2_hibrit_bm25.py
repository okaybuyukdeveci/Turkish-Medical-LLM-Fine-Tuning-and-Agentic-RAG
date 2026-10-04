!pip install -q rank_bm25 2>/dev/null || None

import re, unicodedata
from rank_bm25 import BM25Okapi
from scripts.indexing.loader import load_markdown_documents
from scripts.indexing.chunker import ChunkingConfig, chunk_parent_child_documents

LIMIT, TUT_N = 6000, 6

print("parent chunk'lar yeniden uretiliyor (Qdrant'a dokunulmuyor)")
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
korpus = [jeton(p.page_content) for p in parents]
bm25 = BM25Okapi(korpus)

def rrf(siralar, k=60):
    """Reciprocal Rank Fusion — birden cok siralamayi birlestirir."""
    puan = {}
    for sira in siralar:
        for yer, kimlik in enumerate(sira):
            puan[kimlik] = puan.get(kimlik, 0.0) + 1.0 / (k + yer + 1)
    return [kimlik for kimlik, _ in sorted(puan.items(), key=lambda x: -x[1])]

icerik_idx = {p.page_content: i for i, p in enumerate(parents)}

print(f"\nhibrit getirme ({len(test_rows)} soru)")
eski_k = retriever.child_retriever.search_kwargs.get("k", settings.retrieval_k)
retriever.child_retriever.search_kwargs["k"] = 20
H_docs = []
for i, r in enumerate(test_rows, 1):
    yogun = retriever.invoke(r["question"])
    yogun_idx = [icerik_idx.get(d.page_content) for d in yogun]
    yogun_idx = [x for x in yogun_idx if x is not None]
    seyrek_idx = list(bm25.get_top_n(jeton(r["question"]), range(len(parents)), n=20))
    birlesik = rrf([yogun_idx, seyrek_idx])[:TUT_N]
    H_docs.append([parents[j] for j in birlesik])
    print(f"  {i}/{len(test_rows)}", end="\r")
retriever.child_retriever.search_kwargs["k"] = eski_k
print()

ortak = sum(len(set(icerik_idx.get(d.page_content) for d in H_docs[i])) for i in range(len(H_docs)))
print(f"ortalama belge: {ortak/len(H_docs):.1f}")

H_pr = [prompt(SYSTEM_RAG, f"KAYNAK METİN:\n{baglam_kur(d, LIMIT)}\n\n---\n\nSORU:\n{r['question']}")
        for r, d in zip(test_rows, H_docs)]
H_outs = uret(H_pr, 64, bs=4, etiket="[hibrit] ")
H_acc, H_pred = dogruluk(H_outs, test_rows, "[hibrit] ")
print(f"\nRAG duz {B_acc:.1%}  ->  hibrit {H_acc:.1%}  ({H_acc-B_acc:+.1%})")
