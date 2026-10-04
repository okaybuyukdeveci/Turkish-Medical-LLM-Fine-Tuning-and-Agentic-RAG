import torch, gc
from FlagEmbedding import FlagReranker

LIMIT, ADAY_N, TUT_N = 6000, 30, 6
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda:0"])

print(f"hibrit aday havuzu ({ADAY_N}) + yeniden siralama")
eski_k = retriever.child_retriever.search_kwargs.get("k", settings.retrieval_k)
retriever.child_retriever.search_kwargs["k"] = ADAY_N
HR_docs = []
for i, r in enumerate(test_rows, 1):
    yogun = retriever.invoke(r["question"])
    yogun_idx = [icerik_idx.get(d.page_content) for d in yogun]
    yogun_idx = [x for x in yogun_idx if x is not None]
    seyrek_idx = bm25.get_top_n(jeton(r["question"]), range(len(parents)), n=ADAY_N)
    aday = [parents[j] for j in rrf([yogun_idx, list(seyrek_idx)])[:ADAY_N]]
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

HR_pr = [prompt(SYSTEM_RAG, f"KAYNAK METİN:\n{baglam_kur(d, LIMIT)}\n\n---\n\nSORU:\n{r['question']}")
         for r, d in zip(test_rows, HR_docs)]
HR_outs = uret(HR_pr, 64, bs=4, etiket="[hibrit+rerank] ")
HR_acc, HR_pred = dogruluk(HR_outs, test_rows, "[hibrit+rerank] ")
print(f"\nRAG duz {B_acc:.1%}  ->  hibrit+rerank {HR_acc:.1%}  ({HR_acc-B_acc:+.1%})")

del reranker; gc.collect(); torch.cuda.empty_cache()
