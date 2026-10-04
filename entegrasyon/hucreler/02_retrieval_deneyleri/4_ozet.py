import json
from math import comb

def mcnemar(p1, p2):
    a = sum(x != r["letter"] and y == r["letter"] for x, y, r in zip(p1, p2, test_rows))
    b = sum(x == r["letter"] and y != r["letter"] for x, y, r in zip(p1, p2, test_rows))
    n = a + b
    p = (sum(comb(n, k) for k in range(max(a, b), n + 1)) / 2**n * 2) if n else 1.0
    return a, b, min(p, 1.0)

adaylar = [("RAG yok", sade_acc, None), ("RAG duz (k=6, 6000)", B_acc, B_pred)]
for ad, deg in [("rerank", "R"), ("hibrit", "H"), ("hibrit + rerank", "HR")]:
    if f"{deg}_acc" in globals():
        adaylar.append((ad, globals()[f"{deg}_acc"], globals()[f"{deg}_pred"]))

print("="*64)
for ad, acc, _ in adaylar:
    print(f"{ad:<26}{acc:>7.1%}")
print("="*64)
print("\nRAG duz'e gore (McNemar):")
for ad, acc, pred in adaylar[2:]:
    a, b, p = mcnemar(B_pred, pred)
    print(f"  {ad:<22} {acc-B_acc:+.1%}  |  lehte {a} aleyhte {b}  |  p = {p:.3f}"
          + ("  ANLAMLI" if p < 0.05 else ""))

OUT = "/content/drive/MyDrive/tus_sft/rag_sonuc"
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/agir_deneyler.json", "w", encoding="utf-8") as f:
    json.dump({ad: acc for ad, acc, _ in adaylar}, f, ensure_ascii=False, indent=2)
print("\nkaydedildi:", OUT)
