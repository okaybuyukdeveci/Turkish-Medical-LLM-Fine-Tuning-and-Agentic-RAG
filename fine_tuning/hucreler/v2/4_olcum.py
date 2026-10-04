import json, re
from math import comb

BASLIK = re.compile(r"^\s*Doğru cevap\s*:?\s*\(?([A-E])\)?")
HER = re.compile(r"(?:Doğru\s*)?(?:cevap|yanıt)\s*[:\-]?\s*\(?([A-E])\)?", re.I)
def parse_cevap(t):
    c = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()
    m = BASLIK.match(c) or HER.search(c) or re.search(r"\b([A-E])\)", c)
    return m.group(1).upper() if m else None

FastLanguageModel.for_inference(model)

print("[1/2] TUS dogrulugu")
t_out = uret([prompt(SIS_TUS, r["question"]) for r in test_rows], 64, bs=16, etiket="[tus] ")
t_pred = [parse_cevap(o) for o in t_out]
t_acc = sum(p == r["letter"] for p, r in zip(t_pred, test_rows)) / len(test_rows)
print(f"TUS dogruluk: {t_acc:.1%}   (v1: 70.9%)")

print("\n[2/2] talimat takibi — arama sorgusu uretme")
q_out = uret([prompt(SIS_SORGU, r["question"]) for r in test_rows], 48, bs=16, etiket="[sorgu] ")
def kullanilabilir(t, soru):
    s = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip().split("\n")[0].strip(' "\'`')
    s = re.sub(r"^(arama|sorgu|anahtar kelimeler)\s*[:\-]\s*", "", s, flags=re.I).strip()
    if not (3 <= len(s) <= 220): return False, s
    if re.match(r"^[A-E][\)\.]", s): return False, s
    if re.search(r"Doğru cevap", s, re.I): return False, s
    if s[:40].lower() in soru[:70].lower(): return False, s
    return True, s
sonuc = [kullanilabilir(o, r["question"]) for o, r in zip(q_out, test_rows)]
iyi = sum(ok for ok, _ in sonuc)
print(f"kullanilabilir sorgu: {iyi}/{len(test_rows)} (%{100*iyi/len(test_rows):.0f})")
print("  v1 SFT: bozuk (hep cevapliyordu) | ham model: 311/316 (%98)")
for ok, s in sonuc[:6]: print(f"  [{'+' if ok else '-'}] {s[:85]}")

print(f"""
{'='*58}
                     SFT v1      SFT v2
TUS dogrulugu         70,9%      {t_acc:>6.1%}
talimat takibi        bozuk      {iyi/len(test_rows):>6.0%}
{'='*58}
Hedef: TUS dusmemis VE talimat takibi %80 ustu""")

OUT = "/content/drive/MyDrive/tus_sft/qwen35_4b_v2"
with open(f"{OUT}/sonuc_v2.json", "w", encoding="utf-8") as f:
    json.dump({"tus_acc": t_acc, "talimat_takibi": iyi/len(test_rows),
               "train_size": len(train_ds), "v1_tus_acc": 0.7089}, f, ensure_ascii=False, indent=2)
print("kaydedildi:", OUT)
