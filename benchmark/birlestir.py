"""Ek olcumu ana sonuc dosyasina birlestirir.

    python benchmark/birlestir.py --ana 2025_sonbahar --ek 2025_sonbahar_ek

Ayristiricidaki duzeltmelerden sonra bir sete soru eklendiginde, 200 sorunun
tamamini yeniden olcmek yerine yalnizca yeni sorular olculup birlestirilir.
Kosul: once olculmus sorularin METNI degismemis olmali — aksi halde o sorular
da yeniden olculmeli. Birlesmis dosya `birlestirme` alaniyla isaretlenir.
"""
import argparse, json
from pathlib import Path
KOK = Path(__file__).resolve().parent

a = argparse.ArgumentParser()
a.add_argument("--ana", required=True)
a.add_argument("--ek", required=True)
A = a.parse_args()

for ana in sorted((KOK/"sonuclar").glob(f"{A.ana}__*.json")):
    konfig = ana.stem.split("__")[1]
    ekler = sorted((KOK/"sonuclar").glob(f"{A.ek}__{konfig}__*.json"))
    if not ekler:
        print(f"  {konfig:<20} ek olcum YOK, atlandi"); continue
    d, e = json.loads(ana.read_text(encoding="utf-8")), json.loads(ekler[-1].read_text(encoding="utf-8"))
    cakisma = set(d["id"]) & set(e["id"])
    if cakisma: raise SystemExit(f"{konfig}: {len(cakisma)} soru iki dosyada da var")
    for alan in ("id", "tahminler", "dogru", "ciktilar"):
        d[alan] = d[alan] + e[alan]
    # soru numarasina gore sirala
    sira = sorted(range(len(d["id"])), key=lambda i: d["id"][i])
    for alan in ("id", "tahminler", "dogru", "ciktilar"):
        d[alan] = [d[alan][i] for i in sira]
    d["soru_sayisi"] = len(d["id"])
    d["dogru_sayisi"] = sum(p == g for p, g in zip(d["tahminler"], d["dogru"]))
    d["dogruluk"] = d["dogru_sayisi"] / d["soru_sayisi"]
    d["cevapsiz"] = sum(p is None for p in d["tahminler"])
    d["sure_sn"] = d["sure_sn"] + e["sure_sn"]
    d["birlestirme"] = {"ana": ana.name, "ek": ekler[-1].name, "ek_soru": len(e["id"]),
                        "not": "ek sorular ayri batchte olculdu; bf16 batch dolgusu "
                               "nedeniyle tek kosuyla birebir ayni olmayabilir"}
    ana.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {konfig:<20} {len(e['id'])} soru eklendi -> {d['soru_sayisi']} soru, {d['dogruluk']:.1%}")
