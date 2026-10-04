import os, json
from math import comb

def mcnemar(p1, p2):
    a = sum(x != r["letter"] and y == r["letter"] for x, y, r in zip(p1, p2, test_rows))
    b = sum(x == r["letter"] and y != r["letter"] for x, y, r in zip(p1, p2, test_rows))
    n = a + b
    p = (sum(comb(n, k) for k in range(max(a, b), n + 1)) / 2**n * 2) if n else 1.0
    return a, b, min(p, 1.0)

print("="*66)
print(f"{'kurulum':<34}{'v1':>10}{'v2':>10}")
print("-"*66)
print(f"{'RAG yok':<34}{'70,9%':>10}{sade_acc:>10.1%}")
print(f"{'duz RAG (k=6, 6000)':<34}{'83,9%':>10}{B_acc:>10.1%}")
v1_hr = f"{HR_v1_acc:.1%}" if HR_v1_acc is not None else "88,9%*"
print(f"{'hibrit + rerank':<34}{v1_hr:>10}{HR_acc:>10.1%}")
print("="*66)
if HR_v1_acc is None:
    print("* ayri oturumdan, esli test yapilamaz")

print("\nv2 icinde (esli, McNemar):")
for ad, acc, pred in [("duz RAG", B_acc, B_pred), ("hibrit + rerank", HR_acc, HR_pred)]:
    a, b, p = mcnemar(sade_pred, pred)
    print(f"  RAG yok -> {ad:<18} {acc-sade_acc:+.1%}  lehte {a} aleyhte {b}  p={p:.4f}"
          + ("  ANLAMLI" if p < 0.05 else ""))
a, b, p = mcnemar(B_pred, HR_pred)
print(f"  duz RAG -> hibrit+rerank    {HR_acc-B_acc:+.1%}  lehte {a} aleyhte {b}  p={p:.4f}"
      + ("  ANLAMLI" if p < 0.05 else ""))

if HR_v1_pred is not None:
    a, b, p = mcnemar(HR_v1_pred, HR_pred)
    print(f"\nadapter karsilastirmasi (ayni belgeler, hibrit+rerank):")
    print(f"  v1 -> v2  {HR_acc-HR_v1_acc:+.1%}  lehte {a} aleyhte {b}  p={p:.4f}"
          + ("  ANLAMLI" if p < 0.05 else "  anlamli degil"))

OUT = "/content/drive/MyDrive/tus_sft/rag_sonuc"
os.makedirs(OUT, exist_ok=True)
kayit = {
    "adapter": "qwen35_4b_v2",
    "korpus": "temizlik oncesi (repo hali)",
    "sade": sade_acc, "duz_rag": B_acc, "hibrit_rerank": HR_acc,
    "hibrit_rerank_v1": HR_v1_acc,
    "tahminler": {"sade": sade_pred, "duz_rag": B_pred,
                  "hibrit_rerank": HR_pred, "hibrit_rerank_v1": HR_v1_pred},
    "dogru": [r["letter"] for r in test_rows],
}
with open(f"{OUT}/v2_korpus_oncesi.json", "w", encoding="utf-8") as f:
    json.dump(kayit, f, ensure_ascii=False, indent=2)
print("\nkaydedildi:", f"{OUT}/v2_korpus_oncesi.json")
print("tahminler de kaydedildi — korpus temizligi sonrasi esli test bununla yapilacak")
