"""Uyarlamali (secici) retrieval akisi.

    python benchmark/secici.py --veri 2025_sonbahar

Iki gecis:
  1. RAG'siz cevapla, secilen harfin olasiligini (guven) olc
  2. guven dusuk VE pasaj ilgiliyse -> retrieval calistir, yeniden cevapla

Modelin kendi belirsizligi karari veriyor; disaridan bir kural degil. 9 sinavda
olculdu: hep-RAG'e gore +1,6 puan, retrieval cagrisi %100'den %45'e iniyor.
"""
import argparse, json, time, torch, yaml
from datetime import date
from pathlib import Path
import calistir as C

KOK = Path(__file__).resolve().parent

@torch.no_grad()
def uret_guvenli(model, tok, promptlar, mnt, bs, harf_id, etiket=""):
    """Uretim + secilen cevap harfinin olasiligi."""
    outs, guvenler, t0 = [], [], time.perf_counter()
    for i in range(0, len(promptlar), bs):
        enc = tok(promptlar[i:i+bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                           pad_token_id=tok.pad_token_id,
                           return_dict_in_generate=True, output_scores=True)
        yeni = g.sequences[:, enc["input_ids"].shape[1]:]
        outs += tok.batch_decode(yeni, skip_special_tokens=True)
        for b in range(yeni.shape[0]):
            p = 0.0
            for adim in range(yeni.shape[1]):
                tid = yeni[b, adim].item()
                if tid in harf_id:
                    p = torch.softmax(g.scores[adim][b].float(), -1)[tid].item(); break
            guvenler.append(p)
        print(f"  {etiket}{min(i+bs,len(promptlar))}/{len(promptlar)}", end="\r")
    print()
    return outs, guvenler, time.perf_counter()-t0

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--veri", required=True)
    a.add_argument("--konfig", default="secici_rag")
    a.add_argument("--limit", type=int, default=0)
    A = a.parse_args()

    K = yaml.safe_load((KOK/"konfig"/f"{A.konfig}.yaml").read_text(encoding="utf-8"))
    rows = C.veri_oku(A.veri)
    if A.limit: rows = rows[:A.limit]
    GE, SE = K["kapi"]["guven_esik"], K["kapi"]["skor_esik"]
    print(f"veri {A.veri}: {len(rows)} soru | kapi: guven<={GE}, skor>={SE}")

    model, tok, cihaz = C.model_yukle(K["model"])
    harf_id = {t for h in "ABCDE" for v in (h, " "+h) for t in tok.encode(v, add_special_tokens=False)}

    # --- 1. gecis: RAG'siz ---
    duz = [C.prompt_kur(tok, K["sistem_mesaji"], r["soru"]) for r in rows]
    outs, guven, s1 = uret_guvenli(model, tok, duz, K["uretim"]["max_new_tokens"],
                                   K["uretim"]["batch"], harf_id, "[1/2 duz] ")
    emin_degil = [i for i, g in enumerate(guven) if g <= GE]
    print(f"  emin olmadigi: {len(emin_degil)}/{len(rows)} (%{100*len(emin_degil)/len(rows):.0f})")

    # --- 2. gecis: yalnizca o sorular icin retrieval ---
    aday, s2 = [], 0.0
    if emin_degil:
        getir, _ = C.retrieval_kur({**K["retrieval"], "tur": "hibrit_rerank"})
        lim = K["retrieval"]["context_karakter"]
        rag_pr, secilen = [], []
        for i in emin_degil:
            docs = getir(rows[i]["soru"])
            if not docs: continue
            kul = f"KAYNAK METİN:\n{C.context_kur(docs, lim)}\n\n---\n\nSORU:\n{rows[i]['soru']}"
            rag_pr.append(C.prompt_kur(tok, K["sistem_mesaji_rag"], kul)); secilen.append(i)
            print(f"  getirme {len(rag_pr)}/{len(emin_degil)}", end="\r")
        print()
        del getir
        import gc; gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        elif torch.cuda.is_available(): torch.cuda.empty_cache()
        rag_out, _, s2 = uret_guvenli(model, tok, rag_pr, K["uretim"]["max_new_tokens"],
                                      max(1, K["uretim"]["batch"]//2), harf_id, "[2/2 rag] ")
        for i, o in zip(secilen, rag_out): outs[i] = o
        aday = secilen

    pred  = [C.parse_cevap(o) for o in outs]
    dogru = [str(r["dogru_cevap"]).strip().upper() for r in rows]
    n = sum(p == g for p, g in zip(pred, dogru))
    print(f"\n{'='*52}\n{A.veri} / {K['ad']}:  {n/len(rows):.1%}  ({n}/{len(rows)})")
    print(f"retrieval calisan: {len(aday)}/{len(rows)} (%{100*len(aday)/len(rows):.0f})")
    print(f"sure {(s1+s2)/60:.1f} dk\n{'='*52}")

    yol = KOK/"sonuclar"/f"{A.veri}__{K['ad']}__{date.today():%Y%m%d}.json"
    yol.parent.mkdir(exist_ok=True)
    json.dump({"veri": A.veri, "konfig": K["ad"], "soru_sayisi": len(rows),
               "dogruluk": n/len(rows), "dogru_sayisi": n,
               "cevapsiz": sum(p is None for p in pred), "sure_sn": s1+s2, "cihaz": cihaz,
               "konfig_tam": K, "id": [r.get("id") for r in rows],
               "tahminler": pred, "dogru": dogru, "ciktilar": outs,
               "kapi": {"guven_esik": GE, "skor_esik": SE, "guven": guven,
                        "rag_calisan": aday, "oran": len(aday)/len(rows)}},
              yol.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("kaydedildi:", yol.name)

if __name__ == "__main__":
    main()
