import re, unicodedata, torch, gc, os, json
from pathlib import Path
from math import comb
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker
from scripts.indexing.loader import load_markdown_documents
from scripts.indexing.chunker import ChunkingConfig, chunk_parent_child_documents

LIMIT, ADAY_N, TUT_N = 6000, 30, 6

print("parent chunk'lar + BM25 yeniden kuruluyor")
belgeler = load_markdown_documents(DATA)
parents, _ = chunk_parent_child_documents(
    belgeler,
    parent_config=ChunkingConfig(max_tokens=settings.parent_chunk_size_tokens, overlap_tokens=0),
    child_config=ChunkingConfig(max_tokens=settings.chunk_size_tokens,
                                overlap_tokens=settings.chunk_overlap_tokens))

def jeton(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2]

bm25 = BM25Okapi([jeton(p.page_content) for p in parents])
icerik_idx = {p.page_content: i for i, p in enumerate(parents)}
print(f"parent: {len(parents)}  (temizlik oncesi 17610)")

def rrf(siralar, k=60):
    puan = {}
    for sira in siralar:
        for yer, kim in enumerate(sira):
            puan[kim] = puan.get(kim, 0.0) + 1.0 / (k + yer + 1)
    return [k_ for k_, _ in sorted(puan.items(), key=lambda x: -x[1])]

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda:0"])
eski_k = retriever.child_retriever.search_kwargs.get("k", settings.retrieval_k)
retriever.child_retriever.search_kwargs["k"] = ADAY_N

print(f"\nhibrit + rerank ({len(test_rows)} soru)")
T_docs = []
for i, r in enumerate(test_rows, 1):
    yogun = [x for x in (icerik_idx.get(d.page_content) for d in retriever.invoke(r["question"])) if x is not None]
    seyrek = list(bm25.get_top_n(jeton(r["question"]), range(len(parents)), n=ADAY_N))
    aday = [parents[j] for j in rrf([yogun, seyrek])[:ADAY_N]]
    if len(aday) <= TUT_N:
        T_docs.append(aday)
    else:
        skor = reranker.compute_score([[r["question"], d.page_content[:2000]] for d in aday], normalize=True)
        T_docs.append([aday[j] for _, j in sorted(zip(skor, range(len(aday))), reverse=True)[:TUT_N]])
    print(f"  {i}/{len(test_rows)}", end="\r")
retriever.child_retriever.search_kwargs["k"] = eski_k
print()

del reranker; gc.collect(); torch.cuda.empty_cache()

T_pr = [prompt(SYSTEM_RAG, f"KAYNAK METİN:\n{baglam_kur(d, LIMIT)}\n\n---\n\nSORU:\n{r['question']}")
        for r, d in zip(test_rows, T_docs)]

adapter_sec("v1")
T_v1_outs = uret(T_pr, 64, bs=4, etiket="[temiz v1] ")
T_v1_acc, T_v1_pred = dogruluk(T_v1_outs, test_rows, "[temiz korpus v1] ")

adapter_sec("v2")
T_v2_outs = uret(T_pr, 64, bs=4, etiket="[temiz v2] ")
T_v2_acc, T_v2_pred = dogruluk(T_v2_outs, test_rows, "[temiz korpus v2] ")

def mcnemar(p1, p2):
    a = sum(x != r["letter"] and y == r["letter"] for x, y, r in zip(p1, p2, test_rows))
    b = sum(x == r["letter"] and y != r["letter"] for x, y, r in zip(p1, p2, test_rows))
    n = a + b
    return a, b, min((sum(comb(n, k) for k in range(max(a, b), n+1)) / 2**n * 2) if n else 1.0, 1.0)

print("\n" + "="*62)
print(f"{'hibrit + rerank':<28}{'ham korpus':>14}{'temiz korpus':>16}")
print("-"*62)
print(f"{'v1 adapter (cevaplayici)':<28}{HR_v1_acc:>14.1%}{T_v1_acc:>16.1%}")
print(f"{'v2 adapter':<28}{HR_acc:>14.1%}{T_v2_acc:>16.1%}")
print("="*62)

print("\nkorpus etkisi (esli, ayni model, ayni ayar):")
for ad, eski, yeni, yeni_acc, eski_acc in [
        ("v1", HR_v1_pred, T_v1_pred, T_v1_acc, HR_v1_acc),
        ("v2", HR_pred,    T_v2_pred, T_v2_acc, HR_acc)]:
    a, b, p = mcnemar(eski, yeni)
    print(f"  {ad}: {yeni_acc-eski_acc:+.1%}  lehte {a} aleyhte {b}  p = {p:.4f}"
          + ("  ANLAMLI" if p < 0.05 else "  anlamli degil"))

OUT = "/content/drive/MyDrive/tus_sft/rag_sonuc"
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/v2_korpus_sonrasi.json", "w", encoding="utf-8") as f:
    json.dump({"korpus": "temizlenmis", "parent": len(parents),
               "v1_acc": T_v1_acc, "v2_acc": T_v2_acc,
               "tahminler": {"v1": T_v1_pred, "v2": T_v2_pred},
               "dogru": [r["letter"] for r in test_rows]}, f, ensure_ascii=False, indent=2)
print("\nkaydedildi:", f"{OUT}/v2_korpus_sonrasi.json")
