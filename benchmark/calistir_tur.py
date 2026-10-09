"""Bir sınavda dört konfigürasyonu tek süreçte ölçer.

    python benchmark/calistir_tur.py --veri 2024_sonbahar

calistir.py her konfig icin modeli bastan yukluyor ve retrievali iki kez
hesapliyordu. Burada:
  - retrieval BIR kez kosar; iki RAG konfigu ayni belgeleri kullanir
    (retrieval modelden bagimsiz, query = sorunun kendisi), bu ayni zamanda
    aradaki paired testi de gecerli kilar
  - taban model BIR kez yuklenir; adapter uzerine takilir
Sonuc dosyalari calistir.py ile ayni bicimde, ozet.py okuyabilir.
"""
import argparse, gc, json, resource, time
from datetime import date
from pathlib import Path

import torch, yaml
import calistir as C

KOK = Path(__file__).resolve().parent

def bellek():
    """macOS'ta ru_maxrss Metal tahsislerini saymiyor; gercek sayi surucuden."""
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**30
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30

def uret(model, tok, promptlar, mnt, bs, etiket):
    outs, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(promptlar), bs):
            enc = tok(promptlar[i:i+bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=mnt, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            n = min(i+bs, len(promptlar)); gec = time.perf_counter()-t0
            print(f"  {etiket} {n}/{len(promptlar)} | {gec:.0f} sn | "
                  f"kalan ~{gec/n*(len(promptlar)-n)/60:.0f} dk", end="\r")
    print()
    return outs, time.perf_counter()-t0

def kaydet(veri, konfig, K, rows, outs, sure, cihaz, parent_n):
    pred  = [C.parse_cevap(o) for o in outs]
    dogru = [str(r["dogru_cevap"]).strip().upper() for r in rows]
    n_dg  = sum(p == g for p, g in zip(pred, dogru))
    acc   = n_dg / len(rows)
    yol = KOK / "sonuclar" / f"{veri}__{konfig}__{date.today():%Y%m%d}.json"
    yol.parent.mkdir(exist_ok=True)
    json.dump({"veri": veri, "konfig": konfig, "soru_sayisi": len(rows),
               "dogruluk": acc, "dogru_sayisi": n_dg,
               "cevapsiz": sum(p is None for p in pred), "sure_sn": sure,
               "cihaz": cihaz, "parent_chunk": parent_n, "konfig_tam": K,
               "id": [r.get("id") for r in rows],
               "tahminler": pred, "dogru": dogru, "ciktilar": outs},
              yol.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  -> {konfig:<20} {acc:>6.1%}  ({n_dg}/{len(rows)})  cevapsiz "
          f"{sum(p is None for p in pred)}  {sure/60:.1f} dk  bellek {bellek():.1f} GB")
    return acc

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--veri", required=True)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--bs-duz", type=int, default=16)
    a.add_argument("--bs-rag", type=int, default=8)
    A = a.parse_args()

    kon = {k: yaml.safe_load((KOK/"konfig"/f"{k}.yaml").read_text(encoding="utf-8"))
           for k in ("baseline", "sft", "rag_baseline", "rag_hibrit_rerank")}
    rows = C.veri_oku(A.veri)
    if A.limit: rows = rows[:A.limit]
    print(f"veri {A.veri}: {len(rows)} soru | 4 konfig tek surecte\n")

    tok = C.AutoTokenizer.from_pretrained(kon["baseline"]["model"]["taban"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # --- retrieval BIR kez ---
    R = kon["rag_hibrit_rerank"]["retrieval"]
    getir, parent_n = C.retrieval_kur(R)
    lim = R["context_karakter"]
    rag_pr = []
    for i, r in enumerate(rows, 1):
        kul = f"KAYNAK METİN:\n{C.context_kur(getir(r['soru']), lim)}\n\n---\n\nSORU:\n{r['soru']}"
        rag_pr.append(C.prompt_kur(tok, kon["rag_hibrit_rerank"]["sistem_mesaji"], kul))
        print(f"  getirme {i}/{len(rows)}", end="\r")
    print()
    del getir; gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    elif torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"retrieval yigini birakildi | bellek {bellek():.1f} GB\n")

    duz_pr = [C.prompt_kur(tok, kon["baseline"]["sistem_mesaji"], r["soru"]) for r in rows]

    # --- taban model BIR kez ---
    M = kon["baseline"]["model"]
    cfg = C.AutoConfig.from_pretrained(M["taban"])
    Sinif = (C.AutoModelForImageTextToText
             if (hasattr(cfg, "vision_config") or hasattr(cfg, "text_config"))
             else C.AutoModelForCausalLM)
    cihaz = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = Sinif.from_pretrained(M["taban"], dtype=getattr(torch, M.get("dtype","bfloat16"))).to(cihaz).eval()
    print(f"taban model yuklendi ({cihaz}) | bellek {bellek():.1f} GB\n")

    mnt = kon["baseline"]["uretim"]["max_new_tokens"]
    o, s = uret(model, tok, duz_pr, mnt, A.bs_duz, "[baseline]")
    kaydet(A.veri, "baseline", kon["baseline"], rows, o, s, cihaz, None)
    o, s = uret(model, tok, rag_pr, mnt, A.bs_rag, "[rag_baseline]")
    kaydet(A.veri, "rag_baseline", kon["rag_baseline"], rows, o, s, cihaz, parent_n)

    # --- adapter tak ---
    from peft import PeftModel
    S = kon["sft"]["model"]
    model = PeftModel.from_pretrained(model, S["adapter"], subfolder=S.get("alt_klasor"),
                                      revision=S.get("revizyon")).eval()
    if not any("lora" in n for n, _ in model.named_modules()):
        raise SystemExit("adapter yuklenemedi: LoRA katmani yok")
    print("\nadapter takildi\n")

    o, s = uret(model, tok, duz_pr, mnt, A.bs_duz, "[sft]")
    kaydet(A.veri, "sft", kon["sft"], rows, o, s, cihaz, None)
    o, s = uret(model, tok, rag_pr, mnt, A.bs_rag, "[sft+rag]")
    kaydet(A.veri, "rag_hibrit_rerank", kon["rag_hibrit_rerank"], rows, o, s, cihaz, parent_n)

    print(f"\ntepe bellek (surucu): {bellek():.1f} GB")

if __name__ == "__main__":
    main()
