import re, unicodedata

def sadelestir(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s)

def sik_metni(soru, harf):
    m = re.search(rf"^\s*{harf}\s*[\)\.]\s*(.+?)\s*$", soru, re.M)
    return m.group(1).strip() if m else None

def ortusme(metin, baglam):
    a, b = sadelestir(metin).split(), set(sadelestir(baglam).split())
    a = [w for w in a if len(w) > 3]
    return sum(w in b for w in a) / len(a) if a else 0.0

dogru   = [r["letter"] for r in test_rows]
gerile  = [i for i in range(len(test_rows)) if HR_v1_pred[i] == dogru[i] and HR_pred[i] != dogru[i]]
ilerle  = [i for i in range(len(test_rows)) if HR_v1_pred[i] != dogru[i] and HR_pred[i] == dogru[i]]
print(f"v2'nin kaybettigi: {len(gerile)} | kazandigi: {len(ilerle)}")

print("\n[1] v2 baglami mi yok sayiyor? (RAG'li cevap == RAG'siz cevabi mi)")
for ad, idx in [("gerileyen", gerile), ("tum sorular", range(len(test_rows)))]:
    idx = list(idx)
    ayni_v2 = sum(HR_pred[i] == sade_pred[i] for i in idx) / len(idx)
    print(f"  {ad:<14} v2 RAG cevabi = v2 RAG'siz cevabi: %{100*ayni_v2:.0f}  (n={len(idx)})")

print("\n[2] dogru cevabin kaniti baglamda var miydi?")
for ad, idx in [("gerileyen", gerile), ("tum sorular", range(len(test_rows)))]:
    idx = list(idx); skor = []
    for i in idx:
        mt = sik_metni(test_rows[i]["question"], dogru[i])
        if mt: skor.append(ortusme(mt, baglam_kur(HR_docs[i], 6000)))
    if skor:
        print(f"  {ad:<14} dogru sik metninin baglamla ortusmesi: ort %{100*sum(skor)/len(skor):.0f}  (n={len(skor)})")

print("\n[3] cikti uzunlugu (karakter)")
for ad, outs in [("v1", HR_v1_outs), ("v2", HR_outs)]:
    uz = sorted(len(o) for o in outs)
    print(f"  {ad}: p50 {uz[len(uz)//2]} | p95 {uz[int(len(uz)*.95)]} | max {uz[-1]}")
    kesik = sum(not re.search(r"[.!?]\s*$", o.strip()) for o in outs)
    print(f"      cumle ortasinda kesilen: {kesik}/{len(outs)}")

print("\n[4] gerileyen sorulardan ilk 5 — yan yana")
for i in gerile[:5]:
    print("\n" + "="*70)
    print(f"soru {i} | dogru {dogru[i]} | v1 {HR_v1_pred[i]} | v2 {HR_pred[i]} | v2 RAG'siz {sade_pred[i]}")
    print("-"*70)
    print("SORU :", test_rows[i]["question"][:230].replace("\n", " "))
    print("v1   :", HR_v1_outs[i][:220].replace("\n", " "))
    print("v2   :", HR_outs[i][:220].replace("\n", " "))
