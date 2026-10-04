# Entegrasyon — ince ayarlı model + agentic RAG

`agentic_rag/` getirme katmanını kendi ince ayarlı Qwen modelimizle birleştirip
316 soruluk test setinde ölçen defterler.

## Önce şunu bil

`agentic_rag/` varsayılan olarak **Together API / GLM-5.3-Flash** kullanır.
Buradaki defterler onu kullanmaz; yerine yerel **Qwen3.5-4B + LoRA adapter**
koyar. Together anahtarına gerek yok, GPU'ya gerek var.

Kullanılan parçalar `agentic_rag`'den: `scripts.indexing` (chunk'lama ve Qdrant
indeksi), `scripts.retrieval` (parent–child getirme), `data/` (24 markdown).
LangGraph akışı kullanılmıyor — tek adımlı soru-cevap ölçülüyor.

## Üretim yapılandırması

```
cevaplama      : Qwen3.5-4B + v1 adapter
sorgu üretme   : Qwen3.5-4B + v2 adapter   (set_adapter ile aynı oturumda)
getirme        : BM25 + yoğun arama → RRF (k=60) → bge-reranker-v2-m3 → en iyi 6
bağlam         : 6000 karakter
korpus         : ham (temizlenmemiş) — nedeni aşağıda
```

Doğruluk **88,6%**. Yanıt süresi A100'de 6,2 sn (kısa cevap) / 11,8 sn (tam açıklama).

## Defterler

| Defter | Ne yapar | Süre |
|---|---|---|
| `notebooks/01_rag_olcum.ipynb` | Kurulum, indeks, model, düz RAG, hibrit+rerank, özet, teşhis, gecikme | ~90 dk |
| `notebooks/02_retrieval_deneyleri.ipynb` | Rerank / hibrit / ikisi birlikte karşılaştırması | ~2 sa |
| `notebooks/03_korpus_temizlik.ipynb` | Korpus temizliği — **denendi, işe yaramadı** | ~80 dk |

01 numaralı defter kendi kendine yeter. 02 ve 03 onun değişkenlerini kullanır,
**aynı oturumda** çalıştırılmalı.

## Neden ham korpus

Korpus temizliği (Türkçe karakter onarımı, künye satırlarının atılması, boş
satırların sıkıştırılması) denendi ve **doğruluğu düşürdü**:

```
hibrit + rerank    ham 88,6%  →  temiz 87,0%
5 soru bozuldu, 0 soru düzeldi, p = 0,06
```

Temizlik içeriğe değil biçime dokundu (%2,4 küçülme, çoğu resim yer tutucu) ama
chunk sınırlarını kaydırdı (17.610 → 17.497 parent). Kazanç yok, gürültü var.
03 numaralı defter kayıt için duruyor ki aynı fikir tekrar denenmesin.

## Ölçümler

Tam tablo, anlamlılık testleri ve denenip elenen yöntemler:
[`sonuclar/OLCUMLER.md`](sonuclar/OLCUMLER.md)

Soru bazında tahminler Drive'da (`MyDrive/tus_sft/rag_sonuc/`) — sonraki
ölçümlerin eşli (McNemar) karşılaştırma yapabilmesi için saklanıyor.

## Ön koşullar

- GPU'lu Colab (A100 ya da L4; T4 dar gelir)
- `MyDrive/tus_sft/qwen35_4b/final_adapter` ve `.../qwen35_4b_v2/final_adapter`
- `MyDrive/tus_data/` altında test verisi zip'i
- 03 numaralı defter için ayrıca `MyDrive/tus_data/sozlukler.zip`
