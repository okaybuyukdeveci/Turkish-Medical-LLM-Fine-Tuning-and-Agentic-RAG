!pip install -q FlagEmbedding 2>/dev/null || None

import torch, gc
from FlagEmbedding import FlagReranker

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
GETIR_K, TUT_N, LIMIT = 30, 6, 6000

print("reranker yukleniyor")
reranker = FlagReranker(RERANK_MODEL, use_fp16=True, devices=["cuda:0"])

print(f"\nk={GETIR_K} ile genis getirme ({len(test_rows)} soru)")
eski_k = retriever.child_retriever.search_kwargs.get("k", settings.retrieval_k)
retriever.child_retriever.search_kwargs["k"] = GETIR_K
genis_docs = []
for i, r in enumerate(test_rows, 1):
    genis_docs.append(retriever.invoke(r["question"]))
    print(f"  {i}/{len(test_rows)}", end="\r")
retriever.child_retriever.search_kwargs["k"] = eski_k
print(f"\nortalama aday belge: {sum(len(d) for d in genis_docs)/len(genis_docs):.1f}")

print("\nyeniden siralama")
R_docs = []
for i, (r, docs) in enumerate(zip(test_rows, genis_docs), 1):
    if len(docs) <= TUT_N:
        R_docs.append(docs)
    else:
        ciftler = [[r["question"], d.page_content[:2000]] for d in docs]
        skor = reranker.compute_score(ciftler, normalize=True)
        sirali = sorted(zip(skor, range(len(docs))), reverse=True)
        R_docs.append([docs[j] for _, j in sirali[:TUT_N]])
    print(f"  {i}/{len(test_rows)}", end="\r")
print()

R_pr = [prompt(SYSTEM_RAG, f"KAYNAK METİN:\n{baglam_kur(d, LIMIT)}\n\n---\n\nSORU:\n{r['question']}")
        for r, d in zip(test_rows, R_docs)]
R_outs = uret(R_pr, 64, bs=4, etiket="[rerank] ")
R_acc, R_pred = dogruluk(R_outs, test_rows, "[rerank] ")
print(f"\nRAG duz {B_acc:.1%}  ->  rerank {R_acc:.1%}  ({R_acc-B_acc:+.1%})")

del reranker; gc.collect(); torch.cuda.empty_cache()
