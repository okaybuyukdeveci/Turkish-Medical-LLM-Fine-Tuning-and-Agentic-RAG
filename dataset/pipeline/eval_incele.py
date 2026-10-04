"""Test (eval) setini kayit kayit inceleme araci.

  python dataset_build/eval_incele.py goster <ac|cv|liste_dosyasi> [n]
  python dataset_build/eval_incele.py ok <id> ...
  python dataset_build/eval_incele.py karar '<json>'
  python dataset_build/eval_incele.py durum

Egitim setindeki incele.py'nin test seti karsiligi. FARK: burada amac olcum
aracini saglam tutmak, veri uretmek degil. Bu yuzden:
  - OCR onarimi (govde, siklar) YAPILIR — sorunun aslini geri getirir.
  - Cevap anahtari DEGISTIRILMEZ. Anahtar supheliyse kayit `at` edilir.
  - Aciklama yeniden yazilmaz (test setinde aciklama referanstir).

Kararlar: dataset_v3/eval/kararlar_eval.jsonl  (append-only)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v3_uret import V3  # noqa: E402

EV = V3 / "eval"
YAZ = EV / os.environ.get("KARAR_DOSYA", "kararlar_eval.jsonl")
DOSYA = {"ac": EV / "test_aciklamali.jsonl", "cv": EV / "test_cevap.jsonl"}
ALANLAR = {"id", "karar", "neden", "not", "govde", "siklar", "cevap", "zaman"}
NEDENLER = {"soru_bozuk", "cevap_supheli", "sik_kayip", "gorsel_gerekli",
            "baglam_gerekli", "tekrar", "eksik_bilgi", "diger"}
SIK = re.compile(r"^([A-E])\) (.+)$", re.M)
TEPE = re.compile(r"^<think>\n(.*)\n</think>\n\nDoğru cevap: ([A-E])$", re.S)


def kayitlar():
    d = {}
    for tur, f in DOSYA.items():
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if tur == "ac":
                m = TEPE.match(r["messages"][1]["content"])
                d[r["id"]] = dict(tur=tur, soru=r["messages"][0]["content"],
                                  cevap=m.group(2), acik=m.group(1), meta=r.get("meta") or {})
            else:
                d[r["id"]] = dict(tur=tur, soru=r["soru"], cevap=r["dogru_cevap"],
                                  acik=None, meta=r.get("meta") or {})
    return d


def kararlar():
    d = {}
    for f in sorted(EV.glob("kararlar_eval*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                k = json.loads(l); d[k["id"]] = k
    return d


def dogrula(k):
    fazla = set(k) - ALANLAR
    if fazla: raise SystemExit(f"bilinmeyen alan {fazla} ({k.get('id')})")
    if k["karar"] not in ("ok", "at", "duzelt"): raise SystemExit(f"karar ok/at/duzelt olmali: {k}")
    if k["karar"] == "at" and k.get("neden") not in NEDENLER:
        raise SystemExit(f"neden {sorted(NEDENLER)} icinden olmali: {k.get('id')}")
    if "cevap" in k:
        raise SystemExit(f"TEST SETINDE CEVAP ANAHTARI DEGISTIRILMEZ. Supheliyse 'at' (cevap_supheli): {k['id']}")
    if "siklar" in k:
        sk = k["siklar"]; h = list(sk) if isinstance(sk, dict) else [x[0] for x in sk]
        if h not in (list("ABCDE"), list("ABCD")):
            raise SystemExit(f"sik harfleri A'dan sirayla olmali: {k['id']} -> {h}")


def yaz(ks):
    for k in ks: dogrula(k)
    YAZ.parent.mkdir(parents=True, exist_ok=True)
    with open(YAZ, "a", encoding="utf-8") as f:
        for k in ks:
            k["zaman"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(json.dumps(k, ensure_ascii=False) + "\n")


def goster(tur, n):
    """tur: 'ac' | 'cv' | id listesi iceren bir dosya yolu"""
    R, K = kayitlar(), kararlar()
    yol = Path(tur)
    if yol.exists():
        hepsi = [l.strip() for l in open(yol, encoding="utf-8") if l.strip()]
        ids = [i for i in hepsi if i not in K]
        print(f"# {yol.name}: {len(hepsi)} kayit | {len(ids)} kaldi\n")
    else:
        hepsi = [i for i, r in R.items() if r["tur"] == tur]
        ids = [i for i in hepsi if i not in K]
        print(f"# {tur}: {len(hepsi)} kayit | {len(ids)} kaldi\n")
    for i in ids[:n]:
        r = R[i]
        h = [x for x, _ in SIK.findall(r["soru"])]
        bay = []
        if h not in (list("ABCDE"), list("ABCD")): bay.append(f"SIK_DIZISI={''.join(h)}")
        if r["cevap"] not in h: bay.append("CEVAP_SIKTA_YOK")
        print(f"=== {i} | {r['meta'].get('source_pdf','?')} {' '.join(bay)}")
        print(r["soru"])
        print(f"CEVAP: {r['cevap']}")
        if r["acik"]:
            a = r["acik"]
            print("ACIKLAMA:", (a[:1200] + " …") if len(a) > 1200 else a)
        print()


def durum():
    R, K = kayitlar(), kararlar()
    for tur in ("ac", "cv"):
        t = [i for i, r in R.items() if r["tur"] == tur]
        print(f"{tur}: {sum(1 for i in t if i in K)}/{len(t)}")
    from collections import Counter
    print("kararlar:", dict(Counter(k["karar"] for k in K.values())))
    print("at nedenleri:", dict(Counter(k.get("neden") for k in K.values() if k["karar"] == "at")))


def main():
    a = sys.argv[1:]
    if a[0] == "goster":
        goster(a[1], next((int(x) for x in a[2:] if x.isdigit()), 20))
    elif a[0] == "ok":
        yaz([{"id": i, "karar": "ok"} for i in a[1:]]); print(f"ok: {len(a)-1}")
    elif a[0] == "karar":
        ks = []
        for s in a[1:]:
            k = json.loads(s); ks += k if isinstance(k, list) else [k]
        yaz(ks); print(f"yazildi: {len(ks)}")
    elif a[0] == "durum":
        if len(a) > 1 and Path(a[1]).exists():
            K = kararlar()
            ids = [l.strip() for l in open(a[1], encoding="utf-8") if l.strip()]
            print(f"{a[1]}: {sum(1 for i in ids if i in K)}/{len(ids)}")
        else:
            durum()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
