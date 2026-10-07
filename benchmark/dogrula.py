"""Benchmark veri setlerini doğrular. GPU gerekmez, CI'da koşar."""
import argparse, json, re, sys, unicodedata, hashlib
from pathlib import Path
from collections import Counter

KOK = Path(__file__).resolve().parent
SIK = re.compile(r"^\s*([A-E])\s*[\)\.]", re.M)

def jeton(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2]

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def shingle(ws, n=8):
    return {hashlib.blake2b(" ".join(ws[i:i+n]).encode(), digest_size=8).hexdigest()
            for i in range(len(ws) - n + 1)}

def oku(yol):
    satirlar = []
    ham = Path(yol).read_text(encoding="utf-8")
    for no, l in enumerate(ham.split("\n"), 1):
        if not l.strip(): continue
        try: satirlar.append((no, json.loads(l)))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{yol}:{no} — bozuk JSON: {e}")
    return satirlar

def dogrula(yol, parmak_izi=None):
    hata, uyari = [], []
    kayitlar = oku(yol)
    if not kayitlar: hata.append("dosya boş")

    gorulen_id = set()
    for no, r in kayitlar:
        yer = f"satır {no}"
        for alan in ("id", "soru", "dogru_cevap"):
            if alan not in r or not str(r.get(alan, "")).strip():
                hata.append(f"{yer}: '{alan}' eksik")
        if "id" in r:
            if r["id"] in gorulen_id: hata.append(f"{yer}: id tekrar ediyor — {r['id']}")
            gorulen_id.add(r["id"])
        c = str(r.get("dogru_cevap", "")).strip().upper()
        if c not in set("ABCDE"): hata.append(f"{yer}: dogru_cevap '{c}' A–E dışında")
        siklar = set(SIK.findall(r.get("soru", "")))
        if len(siklar) < 4: hata.append(f"{yer}: {len(siklar)} şık bulundu, en az 4 gerekir")
        elif c and c not in siklar: hata.append(f"{yer}: doğru cevap {c} şıklar arasında yok")

    # set ici kopya
    # Tam hesap, aday daraltma YOK. Daraltma denendi ve benzerligi sistematik
    # olarak dusuk gosterdi: sik gecen kelimeleri eleyince ortak kelime sayisi
    # eksik cikiyor. Bir set ici n^2 karsilastirma birkac yuz soruda sorun degil.
    kumeler = [(r.get("id", no), set(jeton(r.get("soru", "")))) for no, r in kayitlar]
    for i, (kid, s) in enumerate(kumeler):
        for j in range(i + 1, len(kumeler)):
            jc = jaccard(s, kumeler[j][1])
            if jc >= 0.9:
                hata.append(f"set içi kopya: {kid} ↔ {kumeler[j][0]} (Jaccard {jc:.2f})")

    # egitim setiyle leakage
    if parmak_izi and Path(parmak_izi).exists():
        pi = set(Path(parmak_izi).read_text().split())
        sizan = []
        for (no, r), (kid, _) in zip(kayitlar, kumeler):
            sh = shingle(jeton(r.get("soru", "")))
            if sh and len(sh & pi) / len(sh) >= 0.5:
                sizan.append(kid)
        if sizan:
            hata.append(f"eğitim setiyle leakage: {len(sizan)} soru — {sizan[:5]}")
    else:
        uyari.append("parmak izi dosyası yok, leakage kontrolü ATLANDI")

    # cevap dagilimi
    dag = Counter(str(r.get("dogru_cevap", "")).upper() for _, r in kayitlar)
    n = sum(dag.values())
    if n:
        for h, k in dag.most_common(1):
            if k / n > 0.40:
                uyari.append(f"cevap dağılımı dengesiz: {h} şıkkı %{100*k/n:.0f}")

    return kayitlar, hata, uyari

def main():
    a = argparse.ArgumentParser()
    a.add_argument("yollar", nargs="*", help="boş bırakılırsa veri/*.jsonl")
    a.add_argument("--parmak-izi", default=str(KOK / "parmak_izi" / "egitim_shingle.txt"))
    A = a.parse_args()
    yollar = [Path(y) for y in A.yollar] or sorted((KOK / "veri").glob("*.jsonl"))
    if not yollar:
        print("doğrulanacak dosya yok"); return 0

    toplam_hata = 0
    for y in yollar:
        kayitlar, hata, uyari = dogrula(y, A.parmak_izi)
        durum = "HATA" if hata else ("uyarı" if uyari else "TAMAM")
        print(f"\n{y.name}  —  {len(kayitlar)} soru  [{durum}]")
        for u in uyari: print(f"  ! {u}")
        for h in hata[:20]: print(f"  X {h}")
        if len(hata) > 20: print(f"  X ... {len(hata)-20} hata daha")
        toplam_hata += len(hata)
    print(f"\n{'='*48}\ntoplam hata: {toplam_hata}")
    return 1 if toplam_hata else 0

if __name__ == "__main__":
    sys.exit(main())
