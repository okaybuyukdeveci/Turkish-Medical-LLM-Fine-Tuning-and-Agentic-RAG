"""Yeniden yazilan aciklamalari kaynakla karsilastirir; kaynakta olmayan
sayi / oran / ozel isim iceren kayitlari isaretler (uydurma suphesi).

  python dataset_build/yazim_denetim.py [--n 15]

Kaynak = v2 kaydinin sorusu + orijinal aciklamasi (KESILEN blok dahil, ayni metinde).
Turkce ekler icin kelime eslesmesi ON EK bazli yapilir (ilk 6 harf).
"""
import argparse
import glob
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v3_uret import GIRDI, V3  # noqa: E402

SAYI = re.compile(r"\d+(?:[.,]\d+)?\s*%|%\s*\d+(?:[.,]\d+)?|\b\d{2,}\b")
OZEL = re.compile(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{3,}|\b[A-ZÇĞİÖŞÜ]{3,}\b")
# tibbi metinde sik gecen, uydurma sinyali olmayan bas-harf kaliplari
YOKSAY = {"ancak", "bunun", "dolayi", "boyle", "ayrica", "burada", "genel", "ayni",
          "daha", "cunku", "buna", "yani", "iste", "bunlar", "sonuc", "dogru", "yanlis"}


def nrm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = s.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    return re.sub(r"[^a-z0-9%., ]", " ", s)


def kaynakta(parca, kaynak_n):
    p = nrm(parca).strip()
    if not p:
        return True
    if p in kaynak_n:
        return True
    return len(p) >= 6 and p[:6] in kaynak_n      # ek toleransi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="listelenecek en supheli kayit sayisi")
    a = ap.parse_args()

    yaz = {}
    for f in sorted(glob.glob(str(V3 / "inceleme/kararlar*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                k = json.loads(l)
                if "aciklama" in k:
                    yaz[k["id"]] = k["aciklama"]
                elif k["id"] in yaz:
                    del yaz[k["id"]]          # sonraki karar aciklamayi geri almis

    supheli, temiz, toplam_isaret = [], 0, 0
    for l in open(GIRDI, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] not in yaz:
            continue
        kaynak_n = nrm(r["messages"][0]["content"] + " " + r["messages"][1]["content"])
        yeni = yaz[r["id"]]
        isaret = []
        for m in SAYI.finditer(yeni):
            if not kaynakta(m.group(0).replace(" ", ""), kaynak_n.replace(" ", "")):
                isaret.append(("sayi", m.group(0)))
        for m in OZEL.finditer(yeni):
            w = m.group(0)
            if nrm(w).strip() in YOKSAY:
                continue
            if not kaynakta(w, kaynak_n):
                isaret.append(("ozel", w))
        if isaret:
            supheli.append((len(isaret), r["id"], isaret[:6]))
            toplam_isaret += len(isaret)
        else:
            temiz += 1

    n = temiz + len(supheli)
    print(f"yeniden yazilan aciklama : {n}")
    print(f"  kaynakla tam uyumlu    : {temiz} (%{100*temiz/n:.1f})")
    print(f"  isaretli               : {len(supheli)} (%{100*len(supheli)/n:.1f}), toplam {toplam_isaret} isaret")
    supheli.sort(reverse=True)
    print(f"\nen supheli {min(a.n, len(supheli))} kayit:")
    for c, i, ör in supheli[:a.n]:
        print(f"  {i}  {c} isaret  {ör}")


if __name__ == "__main__":
    main()
