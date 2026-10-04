"""Test setine verilen kararlari uygular ve son halini uretir.

  python dataset_build/eval_uygula.py

Girdi : dataset_v3/eval/test_*.jsonl  +  kararlar_eval*.jsonl
Cikti : dataset_v3/eval/son/test_*.jsonl  +  son/elenen.jsonl

`duzelt` yalnizca govde/siklar uygular (cevap anahtarina dokunulmaz — arac zaten
reddediyor). `at` kayitlari elenenlere tasinir, gerekcesiyle.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v3_uret import V3  # noqa: E402
from v3_temizlik import soru_yaz  # noqa: E402

EV = V3 / "eval"
SON = EV / "son"
SIK = re.compile(r"^([A-E])\) (.+)$", re.M)
TEPE = re.compile(r"^<think>\n(.*)\n</think>\n\nDoğru cevap: ([A-E])$", re.S)


def kararlar():
    d = {}
    for f in sorted(EV.glob("kararlar_eval*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if l.strip():
                k = json.loads(l)
                if k["id"] not in d or k.get("zaman", "") >= d[k["id"]].get("zaman", ""):
                    d[k["id"]] = k
    return d


def yeni_soru(eski, k):
    """duzelt karari uygulanmis soru metni."""
    m = SIK.findall(eski)
    govde = eski.split("\n" + (m[0][0] + ") " + m[0][1]))[0].rstrip() if m else eski
    govde = k.get("govde", govde)
    if "siklar" in k:
        sk = k["siklar"]
        ciftler = sorted(sk.items()) if isinstance(sk, dict) else [tuple(x) for x in sk]
    else:
        ciftler = m
    return soru_yaz(govde, ciftler) if ciftler else govde


def main():
    SON.mkdir(parents=True, exist_ok=True)
    K = kararlar()
    say, elenen = Counter(), []

    tut = []
    for l in open(EV / "test_aciklamali.jsonl", encoding="utf-8"):
        r = json.loads(l); k = K.get(r["id"])
        if k and k["karar"] == "at":
            elenen.append({"id": r["id"], "liste": "ac", "neden": k.get("neden"), "not": k.get("not")})
            say["ac_at:" + str(k.get("neden"))] += 1; continue
        if k and k["karar"] == "duzelt":
            r["messages"][0]["content"] = yeni_soru(r["messages"][0]["content"], k)
            say["ac_duzelt"] += 1
        tut.append(r); say["ac_tut"] += 1
    with open(SON / "test_aciklamali.jsonl", "w", encoding="utf-8") as f:
        for r in tut: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tut2 = []
    for l in open(EV / "test_cevap.jsonl", encoding="utf-8"):
        r = json.loads(l); k = K.get(r["id"])
        if k and k["karar"] == "at":
            elenen.append({"id": r["id"], "liste": "cv", "neden": k.get("neden"), "not": k.get("not")})
            say["cv_at:" + str(k.get("neden"))] += 1; continue
        if k and k["karar"] == "duzelt":
            r["soru"] = yeni_soru(r["soru"], k); say["cv_duzelt"] += 1
        tut2.append(r); say["cv_tut"] += 1
    with open(SON / "test_cevap.jsonl", "w", encoding="utf-8") as f:
        for r in tut2: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(SON / "elenen.jsonl", "w", encoding="utf-8") as f:
        for r in elenen: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for k, v in sorted(say.items()): print(f"{k:<26} {v}")
    print(f"\nSON: aciklamali {len(tut)} + cevap {len(tut2)} = {len(tut)+len(tut2)} | elenen {len(elenen)}")


if __name__ == "__main__":
    main()
