"""Benchmark koşucusu.

    python benchmark/calistir.py --veri 316_mevcut --konfig rag_hibrit_rerank

Sonuç `sonuclar/<veri>__<konfig>__<tarih>.json` olarak yazılır; soru bazında
tahminleri de içerir — iki koşuyu paired (McNemar) karşılaştırabilmek için.
Sadece toplam doğruluk saklamak yetmiyor, bunu bir kez zor yoldan öğrendik.
"""
import argparse, json, os, re, sys, time, unicodedata
from datetime import date
from pathlib import Path

import torch, yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

KOK = Path(__file__).resolve().parent
REPO = KOK.parent

BASLIK = re.compile(r"^\s*\**\s*Doğru cevap\s*\**\s*:?\s*\**\s*\(?([A-E])\)?")
HER    = re.compile(r"(?:Doğru\s*)?(?:cevap|yanıt)\s*[:\-]?\s*\(?([A-E])\)?", re.I)

def parse_cevap(t):
    c = re.sub(r"<think>.*?</think>", "", t or "", flags=re.S).strip()
    m = BASLIK.match(c) or HER.search(c) or re.search(r"\b([A-E])\)", c)
    return m.group(1).upper() if m else None

def veri_oku(ad):
    y = KOK / "veri" / f"{ad}.jsonl"
    if not y.exists(): raise SystemExit(f"veri seti yok: {y}")
    return [json.loads(l) for l in y.read_text(encoding="utf-8").split("\n") if l.strip()]

def model_yukle(k):
    cfg = AutoConfig.from_pretrained(k["taban"])
    cok_modlu = hasattr(cfg, "vision_config") or hasattr(cfg, "text_config")
    # Qwen3.5 goruntu+metin modeli: CausalLM ile yuklenirse adapter agirliklari
    # eslesmiyor, peft sessizce sifirdan baslatiyor ve adapter ETKISIZ kaliyor.
    Sinif = AutoModelForImageTextToText if cok_modlu else AutoModelForCausalLM
    print(f"mimari {cfg.architectures} -> {Sinif.__name__}")

    tok = AutoTokenizer.from_pretrained(k["taban"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    cihaz = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = Sinif.from_pretrained(k["taban"], dtype=getattr(torch, k.get("dtype", "bfloat16"))).to(cihaz)

    if k.get("adapter"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, k["adapter"],
                                          subfolder=k.get("alt_klasor"),
                                          revision=k.get("revizyon"))
        # adapter gercekten takildi mi — sessiz basarisizliga karsi
        if not any("lora" in n for n, _ in model.named_modules()):
            raise SystemExit("adapter yuklenemedi: LoRA katmani yok")
    model.eval()
    return model, tok, cihaz

def prompt_kur(tok, sistem, kullanici):
    msgs = [{"role": "system", "content": sistem}, {"role": "user", "content": kullanici}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def retrieval_kur(r):
    """agentic_rag modullerini kullanir. Donen: sorular -> belge listesi fonksiyonu."""
    proj = (REPO / "agentic_rag").resolve()
    sys.path.insert(0, str(proj)); os.chdir(proj)
    from scripts.settings import RagSettings
    from scripts.indexing.indexer import ensure_index
    from scripts.retrieval.retriever import load_retriever

    ayar = RagSettings(embedding_device="cuda" if torch.cuda.is_available() else "cpu")
    ayar.validate()
    print("indeks kontrol ediliyor (ilk kosuda 10-15 dk)")
    print("indeks:", ensure_index(ayar))
    ret = load_retriever(ayar)

    if r["tur"] == "dense":
        ret.child_retriever.search_kwargs["k"] = r.get("k", 6)
        return lambda s: ret.invoke(s), None

    # hibrit + rerank
    from rank_bm25 import BM25Okapi
    from FlagEmbedding import FlagReranker
    from scripts.indexing.loader import load_markdown_documents
    from scripts.indexing.chunker import ChunkingConfig, chunk_parent_child_documents

    belgeler = load_markdown_documents(str((REPO / r["korpus"]).resolve())
                                       if not Path(r["korpus"]).is_absolute() else r["korpus"])
    parents, _ = chunk_parent_child_documents(
        belgeler,
        parent_config=ChunkingConfig(max_tokens=ayar.parent_chunk_size_tokens, overlap_tokens=0),
        child_config=ChunkingConfig(max_tokens=ayar.chunk_size_tokens,
                                    overlap_tokens=ayar.chunk_overlap_tokens))
    def jeton(s):
        s = unicodedata.normalize("NFD", s.lower())
        s = "".join(c for c in s if not unicodedata.combining(c))
        return [w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2]
    bm25 = BM25Okapi([jeton(p.page_content) for p in parents])
    idx = {p.page_content: i for i, p in enumerate(parents)}
    rr = FlagReranker(r["reranker"], use_fp16=True)
    N, T, K = r.get("aday_n", 30), r.get("tut_n", 6), r.get("rrf_k", 60)
    ret.child_retriever.search_kwargs["k"] = N

    def rrf(siralar):
        puan = {}
        for s in siralar:
            for yer, kim in enumerate(s): puan[kim] = puan.get(kim, 0.0) + 1.0/(K + yer + 1)
        return [k for k, _ in sorted(puan.items(), key=lambda x: -x[1])]

    def getir(soru):
        yogun = [x for x in (idx.get(d.page_content) for d in ret.invoke(soru)) if x is not None]
        seyrek = list(bm25.get_top_n(jeton(soru), range(len(parents)), n=N))
        aday = [parents[j] for j in rrf([yogun, seyrek])[:N]]
        if len(aday) <= T: return aday
        sk = rr.compute_score([[soru, d.page_content[:2000]] for d in aday], normalize=True)
        return [aday[j] for _, j in sorted(zip(sk, range(len(aday))), reverse=True)[:T]]
    return getir, len(parents)

def context_kur(docs, limit):
    parca, top = [], 0
    for i, d in enumerate(docs, 1):
        blok = f"[Kaynak {i} — {Path((d.metadata or {}).get('source','?')).stem}]\n{d.page_content.strip()}"
        if top + len(blok) > limit:
            kalan = limit - top
            if kalan > 400: parca.append(blok[:kalan])
            break
        parca.append(blok); top += len(blok)
    return "\n\n".join(parca)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--veri", required=True)
    a.add_argument("--konfig", required=True)
    a.add_argument("--limit", type=int, default=0, help="duman testi icin ilk N soru")
    A = a.parse_args()

    K = yaml.safe_load((KOK / "konfig" / f"{A.konfig}.yaml").read_text(encoding="utf-8"))
    rows = veri_oku(A.veri)
    if A.limit: rows = rows[:A.limit]
    print(f"veri {A.veri}: {len(rows)} soru | konfig {K['ad']}")

    getir, parent_n = (None, None)
    if K.get("retrieval"):
        getir, parent_n = retrieval_kur(K["retrieval"])
    model, tok, cihaz = model_yukle(K["model"])
    print(f"cihaz: {cihaz}")

    promptlar = []
    if getir:
        lim = K["retrieval"]["context_karakter"]
        for i, r in enumerate(rows, 1):
            kul = f"KAYNAK METİN:\n{context_kur(getir(r['soru']), lim)}\n\n---\n\nSORU:\n{r['soru']}"
            promptlar.append(prompt_kur(tok, K["sistem_mesaji"], kul))
            print(f"  getirme {i}/{len(rows)}", end="\r")
        print()
    else:
        promptlar = [prompt_kur(tok, K["sistem_mesaji"], r["soru"]) for r in rows]

    U, bs = K["uretim"], K["uretim"].get("batch", 8)
    outs, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(promptlar), bs):
            enc = tok(promptlar[i:i+bs], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            g = model.generate(**enc, max_new_tokens=U["max_new_tokens"],
                               do_sample=U.get("do_sample", False), pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            n = min(i+bs, len(promptlar)); gec = time.perf_counter()-t0
            print(f"  uretim {n}/{len(promptlar)} | {gec:.0f} sn | kalan ~{gec/n*(len(promptlar)-n)/60:.0f} dk", end="\r")
    sure = time.perf_counter() - t0
    print()

    pred  = [parse_cevap(o) for o in outs]
    dogru = [str(r["dogru_cevap"]).strip().upper() for r in rows]
    n_dg  = sum(p == g for p, g in zip(pred, dogru))
    acc   = n_dg / len(rows)
    print(f"\n{'='*52}\n{A.veri} / {K['ad']}:  {acc:.1%}  ({n_dg}/{len(rows)})")
    print(f"cevapsiz {sum(p is None for p in pred)} | sure {sure/60:.1f} dk\n{'='*52}")

    cikti = KOK / "sonuclar" / f"{A.veri}__{K['ad']}__{date.today():%Y%m%d}.json"
    cikti.parent.mkdir(exist_ok=True)
    json.dump({"veri": A.veri, "konfig": K["ad"], "soru_sayisi": len(rows),
               "dogruluk": acc, "dogru_sayisi": n_dg,
               "cevapsiz": sum(p is None for p in pred), "sure_sn": sure,
               "cihaz": cihaz, "parent_chunk": parent_n, "konfig_tam": K,
               "id": [r.get("id") for r in rows],
               "tahminler": pred, "dogru": dogru, "ciktilar": outs},
              cikti.open("w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("kaydedildi:", cikti.relative_to(REPO))

if __name__ == "__main__":
    main()
