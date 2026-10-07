# Benchmark pipeline

Bir test setini seç, bir yapılandırma seç, koş. Sonuç `sonuclar/` altına
soru bazında tahminleriyle birlikte yazılır.

## Kurulum

```bash
git clone <repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r benchmark/gereksinimler.txt
hf auth login          # adapter private repoda
```

GPU: NVIDIA (CUDA) ya da Apple Silicon (MPS). CPU'da da koşar ama çok yavaş.
Bellek: SFT yapılandırması ~11 GB, RAG yapılandırmaları ~16 GB.

## Çalıştırma

```bash
python benchmark/calistir.py --veri 316_mevcut --konfig sft
python benchmark/calistir.py --veri 316_mevcut --konfig rag_hibrit_rerank
python benchmark/ozet.py        # sonuclar/OZET.md tablosunu yeniler
```

`--limit 8` ile duman testi yapabilirsin.

İlk RAG koşusu Qdrant indeksini kurar (10–15 dk) ve `agentic_rag/qdrant_storage/`
altına yazar; sonraki koşular onu kullanır. Klasör `.gitignore`'da.

## Yapılandırmalar

| Konfig | Ne yapar |
|---|---|
| `sft` | Yalnızca ince ayarlı model. Modelin kendi bilgisi. |
| `rag_duz` | Dense retrieval, query = sorunun kendisi |
| `rag_hibrit_rerank` | BM25 + dense → RRF → cross-encoder rerank. **Üretim yapılandırması.** |

## Yeni bir test seti eklemek

1. `benchmark/veri/<yil>_<kaynak>.jsonl` olarak ekle — şema: `veri/SEMA.md`
2. `python benchmark/dogrula.py` çalıştır, temiz olmalı
3. PR aç — CI aynı doğrulamayı koşar, kırmızıysa birleştirme
4. Birleşince `calistir.py` ile ölç

Doğrulama GPU gerektirmez. Şema, set içi kopya ve **eğitim setiyle leakage**
kontrolü yapar. Sonuncusu önemli: örtüşen bir test seti üzerinde ölçülen
doğruluk anlamsızdır.

## Neden parmak izi dosyası

`parmak_izi/egitim_shingle.txt` eğitim setindeki 8 kelimelik pencerelerin
hash'leri — ham metin değil. Eğitim verisi telifli olduğu için repoda
tutulmuyor, ama leakage kontrolü için karşılaştırılacak bir şey lazım.
Hash'ler bunu çözüyor: metin dışarıda kalıyor, kontrol CI'da koşabiliyor.

Eğitim verisi değişirse yeniden üret:

```bash
python benchmark/parmak_izi_uret.py /yol/egitim.jsonl
```

## Tekrarlanabilirlik

- Üretim deterministik (`do_sample=False`), `repetition_penalty` **kullanılmıyor**
- Adapter sürümü konfigte `revizyon` ile pinlenebilir — yayına çıkmış bir sayıyı
  savunmak gerekirse commit sha yaz, `main` bırakma
- Sonuçlar soru bazında tahminleri tutar; aynı set üzerindeki iki koşu paired
  McNemar ile karşılaştırılabilir
- Aynı yapılandırma iki farklı cihazda ±3 soru oynayabilir (bf16 batch dolgusu).
  Birebir aynı sayıyı bekleme.
