import time, statistics, torch
from FlagEmbedding import FlagReranker

ADAY_N, TUT_N, LIMIT = 30, 6, 6000

print("reranker yukleniyor (urunde surekli bellekte durur, olcume dahil degil)")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=["cuda:0"])
adapter_sec("v1")

def tam_akis(soru, max_new_tokens=64):
    z = {}
    t0 = time.perf_counter()
    retriever.child_retriever.search_kwargs["k"] = ADAY_N
    yogun = [x for x in (icerik_idx.get(d.page_content) for d in retriever.invoke(soru)) if x is not None]
    t1 = time.perf_counter(); z["yogun"] = t1 - t0

    seyrek = list(bm25.get_top_n(jeton(soru), range(len(parents)), n=ADAY_N))
    t2 = time.perf_counter(); z["bm25"] = t2 - t1

    aday = [parents[j] for j in rrf([yogun, seyrek])[:ADAY_N]]
    skor = reranker.compute_score([[soru, d.page_content[:2000]] for d in aday], normalize=True)
    secilen = [aday[j] for _, j in sorted(zip(skor, range(len(aday))), reverse=True)[:TUT_N]]
    t3 = time.perf_counter(); z["rerank"] = t3 - t2

    p = prompt(SYSTEM_RAG, f"KAYNAK METİN:\n{baglam_kur(secilen, LIMIT)}\n\n---\n\nSORU:\n{soru}")
    enc = tok(p, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    t4 = time.perf_counter(); z["uretim"] = t4 - t3
    z["toplam"] = t4 - t0
    z["token"] = int(g[0].shape[0] - enc["input_ids"].shape[1])
    z["metin"] = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return z

sorular = [r["question"] for r in test_rows[:12]]
tam_akis(sorular[0])

for etiket, mnt in [("sadece cevap (64 token)", 64), ("tam aciklama (300 token)", 300)]:
    olcum = [tam_akis(s, mnt) for s in sorular]
    print(f"\n{etiket} — {len(sorular)} soru, tek tek (toplu islem yok)")
    for ad in ("yogun", "bm25", "rerank", "uretim"):
        d = [o[ad] for o in olcum]
        print(f"  {ad:<9} medyan {statistics.median(d):>6.2f} sn   (%{100*statistics.median(d)/statistics.median([o['toplam'] for o in olcum]):.0f})")
    top = [o["toplam"] for o in olcum]
    tk  = [o["token"] for o in olcum]
    print(f"  {'TOPLAM':<9} medyan {statistics.median(top):>6.2f} sn | p95 {sorted(top)[int(len(top)*.95)]:.2f} | en yavas {max(top):.2f}")
    print(f"  uretim hizi: {statistics.median(tk)/statistics.median([o['uretim'] for o in olcum]):.0f} token/sn")

print("\nornek cikti (300 token):")
print(tam_akis(sorular[1], 300)["metin"][:500])
print("\nNot: A100-40GB, model ve reranker bellekte. Soguk baslatma (model yukleme ~2 dk) dahil degil.")
