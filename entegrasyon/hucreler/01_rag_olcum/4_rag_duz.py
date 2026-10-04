print("[duz RAG] belgeler getiriliyor (k=6, baglam 6000)")
B_docs = []
for i, r in enumerate(test_rows, 1):
    B_docs.append(retriever.invoke(r["question"]))
    print(f"  getirme {i}/{len(test_rows)}", end="\r")
print(f"\nortalama belge: {sum(len(d) for d in B_docs)/len(B_docs):.1f}")

B_pr = [prompt(SYSTEM_RAG, f"KAYNAK METİN:\n{baglam_kur(d)}\n\n---\n\nSORU:\n{r['question']}")
        for r, d in zip(test_rows, B_docs)]
uz = sorted(len(tok(p)["input_ids"]) for p in B_pr)
print(f"prompt token p50 {uz[len(uz)//2]} | p95 {uz[int(len(uz)*.95)]} | max {uz[-1]}")

B_outs = uret(B_pr, 64, bs=4, etiket="[duz RAG] ")
B_acc, B_pred = dogruluk(B_outs, test_rows, "[duz RAG v2] ")
print(f"v1 ayni ayarda: 83,9%  |  v2: {B_acc:.1%}")
