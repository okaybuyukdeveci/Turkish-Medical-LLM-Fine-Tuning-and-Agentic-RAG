"""Eğitim setinden shingle parmak izi üretir.

Ham metin DEĞİL, 8 kelimelik pencerelerin hash'leri. Telifli içerik repoya
girmeden leakage kontrolü yapılabilsin diye. Eğitim verisi değişince yeniden
çalıştır.
"""
import argparse, json, re, unicodedata, hashlib
from pathlib import Path

def jeton(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2]

def soru_metni(r):
    if "soru" in r: return r["soru"]
    for m in r.get("messages", []):
        if m.get("role") == "user": return m["content"]
    return ""

a = argparse.ArgumentParser()
a.add_argument("egitim_jsonl", help="eğitim seti (repoda tutulmaz)")
a.add_argument("--cikti", default=str(Path(__file__).parent / "parmak_izi" / "egitim_shingle.txt"))
a.add_argument("--n", type=int, default=8)
A = a.parse_args()

sh = set()
kayit = 0
# splitlines() kullanma: U+2028/U+0085 gibi karakterleri de satir sonu sayar
for l in Path(A.egitim_jsonl).read_text(encoding="utf-8").split("\n"):
    if not l.strip(): continue
    ws = jeton(soru_metni(json.loads(l)))
    kayit += 1
    for i in range(len(ws) - A.n + 1):
        sh.add(hashlib.blake2b(" ".join(ws[i:i+A.n]).encode(), digest_size=8).hexdigest())

Path(A.cikti).parent.mkdir(parents=True, exist_ok=True)
Path(A.cikti).write_text("\n".join(sorted(sh)), encoding="utf-8")
print(f"{kayit} kayıt → {len(sh):,} shingle → {A.cikti}")
print(f"boyut: {Path(A.cikti).stat().st_size/2**20:.1f} MB")
