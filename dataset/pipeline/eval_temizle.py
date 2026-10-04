"""Test (eval) setlerini egitim setiyle AYNI temizlik hattindan gecirir.

  python dataset_build/eval_temizle.py

Egitim seti v3 hattindan geciyordu, eval gecmiyordu: sik harfleri eksik,
LaTeX kalintilari ve soru numaralari duruyordu. Bu, dogruluk olcumunu
oldugundan dusuk gosteriyor.

Cikti: dataset_v3/eval/  (temiz) + ayrica kullanilamaz kayitlarin listesi.
Cevap harfi yeniden harflendirmede kayarsa ESLEME ile tasinir; dogru sikkin
metni tamamen kaybolmussa kayit KULLANILMAZ sayilir ve ayri dosyaya yazilir.
"""
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v3_uret import V3, aciklama_temizle, soru_temizle, turkce_yap, yapisik_ayir  # noqa: E402

KOK = Path(__file__).resolve().parent.parent
GIRDI = KOK / "dataset_v2/eval"
CIKTI = V3 / "eval"
SIK = re.compile(r"^([A-E])\) ", re.M)
TEPE = re.compile(r"^<think>\n(.*)\n</think>\n\nDoğru cevap: ([A-E])$", re.S)


def _nrm(s):
    s = unicodedata.normalize("NFKD", s.lower()).translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))
    return frozenset(w for w in re.findall(r"[a-z]{4,}", s))


def _sizinti_dizini():
    """Egitim setindeki soru govdelerinin kelime kumeleri + ters indeks."""
    kume, idx = [], defaultdict(list)
    for l in open(V3 / "sft/tus_mcq_egitim.jsonl", encoding="utf-8"):
        t = _nrm(json.loads(l)["messages"][0]["content"].split("\n")[0])
        for w in t: idx[w].append(len(kume))
        kume.append(t)
    return kume, idx


def _sizdi(soru, kume, idx, esik=0.8):
    """Egitim setinde Jaccard >= esik bir soru var mi? (yakin kopya yakalar)"""
    t = _nrm(soru.split("\n")[0])
    if len(t) < 4: return False
    say = Counter()
    for w in t:
        for i in idx.get(w, ()): say[i] += 1
    for i, c in say.most_common(3):
        if c / len(t | kume[i]) >= esik: return True
    return False


def temizle_soru(ham, harf):
    """-> (temiz_soru, yeni_harf | None).  None: dogru sikkin metni kayip."""
    soru, esleme, _onarim, _sorun, _sinav, _govde, siklar = soru_temizle(ham)
    if not siklar:
        return soru, None
    yeni = esleme.get(harf)
    if yeni is None or yeni not in {h for h, _ in siklar}:
        return soru, None
    return soru, yeni


def main():
    CIKTI.mkdir(parents=True, exist_ok=True)
    say = Counter()
    kume, idx = _sizinti_dizini()

    # --- 1) aciklamali test (messages semasi) ---
    tut, at = [], []
    for l in open(GIRDI / "tus_mcq_eval_aciklamali.jsonl", encoding="utf-8"):
        r = json.loads(l)
        m = TEPE.match(r["messages"][1]["content"])
        if not m:
            at.append({"id": r["id"], "neden": "bicim_bozuk"}); say["ac_bicim"] += 1; continue
        soru, harf = temizle_soru(r["messages"][0]["content"], m.group(2))
        if harf is None:
            at.append({"id": r["id"], "neden": "dogru_sik_kayip"}); say["ac_sik_kayip"] += 1; continue
        if SIK.findall(soru) not in (list("ABCDE"), list("ABCD")):
            at.append({"id": r["id"], "neden": "sik_dizisi_bozuk"}); say["ac_dizi_bozuk"] += 1; continue
        if _sizdi(soru, kume, idx):
            at.append({"id": r["id"], "neden": "egitimle_cakisiyor"}); say["ac_sizinti"] += 1; continue
        if harf != m.group(2): say["ac_harf_kaydi"] += 1
        acik = yapisik_ayir(turkce_yap(aciklama_temizle(m.group(1), ayir=False), temkinli=True))
        r["messages"][0]["content"] = soru
        r["messages"][1]["content"] = f"<think>\n{acik}\n</think>\n\nDoğru cevap: {harf}"
        tut.append(r); say["ac_tutuldu"] += 1
    with open(CIKTI / "test_aciklamali.jsonl", "w", encoding="utf-8") as f:
        for r in tut: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- 2) cevap testi (soru/dogru_cevap semasi) ---
    tut2 = []
    for l in open(GIRDI / "tus_mcq_eval.jsonl", encoding="utf-8"):
        r = json.loads(l)
        soru, harf = temizle_soru(r["soru"], r["dogru_cevap"])
        if harf is None:
            at.append({"id": r["id"], "neden": "dogru_sik_kayip"}); say["cv_sik_kayip"] += 1; continue
        if SIK.findall(soru) not in (list("ABCDE"), list("ABCD")):
            at.append({"id": r["id"], "neden": "sik_dizisi_bozuk"}); say["cv_dizi_bozuk"] += 1; continue
        if _sizdi(soru, kume, idx):
            at.append({"id": r["id"], "neden": "egitimle_cakisiyor"}); say["cv_sizinti"] += 1; continue
        if harf != r["dogru_cevap"]: say["cv_harf_kaydi"] += 1
        r["soru"], r["dogru_cevap"] = soru, harf
        tut2.append(r); say["cv_tutuldu"] += 1
    with open(CIKTI / "test_cevap.jsonl", "w", encoding="utf-8") as f:
        for r in tut2: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(CIKTI / "kullanilamaz.jsonl", "w", encoding="utf-8") as f:
        for r in at: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for k, v in sorted(say.items()): print(f"{k:<16} {v}")
    print(f"\nkullanilamaz toplam: {len(at)}  -> {CIKTI}/kullanilamaz.jsonl")


if __name__ == "__main__":
    main()
