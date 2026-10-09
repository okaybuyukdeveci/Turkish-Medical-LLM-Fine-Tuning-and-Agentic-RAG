# Benchmark sonuçları

Gerçek ÖSYM TUS sınavlarında ölçüm. Soru verisi repoda yok (telifli, `.gitignore`);
`pdf_ayristir.py` + `havuz_kur.py` ile PDF'lerden yeniden üretilir.

## Son sistem

```
Qwen/Qwen3.5-4B (bf16, kuantizasyon yok)
  + v1 adapter          tus-llm/tus-qwen35-4b-lora, alt klasör: answering
  + kapılı RAG          hibrit arama → RRF(k=60) → bge-reranker-v2-m3 → 6 chunk / 6000 karakter
```

**Kapı:** cevap harfinin softmax olasılığı ≤ 0,84 **ve** reranker en iyi skoru ≥ 0,2
olduğunda retrieval çalışır. Soruların ~%45'inde tetikleniyor. Ek model ya da ek
çağrı değil — cevap üretilirken zaten ortaya çıkan olasılığı okuyor.

**Sorgu:** sorunun kendi metni. Model sorgu üretmiyor (aşağıya bakın).

## Toplu sonuç — 7 temiz sınav, 1.471 soru

| Yapılandırma | Doğruluk | Not |
|---|---|---|
| Qwen3.5-4B taban | 63,0% | |
| + v1 (SFT) | 73,8% | +10,8, eşleştirilmiş McNemar p = 4,6e-21 |
| + her soruda RAG | 78,8% | +5,0, p = 1,9e-05 |
| **+ kapılı RAG** | **80,3%** | +1,5, RAG soruların %45'inde |
| oracle (kusursuz kapı) | 86,4% | üst sınır, ulaşılmadı |

## Sınav bazında

`*` işaretli iki sınav eğitim setiyle örtüşme nedeniyle ortalamaya katılmaz.

| Sınav | SFT | her soruda RAG | kapılı RAG | v3 kapısı |
|---|---|---|---|---|
| 2021 ilkbahar | 71,3% | 80,6% | 81,4% | 80,2% |
| 2021 sonbahar | 75,7% | 80,3% | 82,0% | 80,8% |
| 2022 sonbahar * | 72,1% | 76,2% | 77,5% | 76,7% |
| 2023 ilkbahar * | 68,8% | 75,0% | 73,8% | 75,8% |
| 2023 sonbahar | 71,4% | 76,5% | 79,6% | 77,0% |
| 2024 ilkbahar | 70,0% | 73,5% | 75,0% | 73,0% |
| 2024 sonbahar | 76,4% | 81,4% | 82,4% | 82,9% |
| 2025 ilkbahar | 79,5% | 81,0% | 83,5% | 83,0% |
| 2025 sonbahar | 72,5% | 77,5% | 78,5% | 77,0% |

## Kullanılmayan yollar ve nedenleri

**v2 adapter (`query`) — kullanılmıyor.** v1 yalnızca TUS verisiyle eğitildiği için
başka talimatları izlemiyor (catastrophic forgetting). v2 eğitim setine %6 sorgu
üretme görevi karıştırılarak bunu düzeltti, talimat izleme %98'e çıktı — ama RAG
altında cevaplama %88,6'dan %85,1'e düştü (13 soru bozuldu, 2 düzeldi, p = 0,007).
Sistem yalnızca soru çözdüğü için v1 tercih edildi.

**Modelin ürettiği sorgu — kullanılmıyor.** Model 316 sorunun 309'unda geçerli bir
sorgu üretiyor, yani yapabiliyor. Ama o sorguyla doğruluk %73,4; sorunun kendi
metniyle %83,9. TUS sorusu zaten terim yoğun; model özetlerken ayırt edici terimi
düşürüyor.

**v3 adapter (`karar`) — kullanılmıyor.** Retrieval kararını modelin kendi çıktısı
olarak vermesi için eğitildi (2006-2020 arası 26 sınav, 5.217 soru; etiketler
ölçümden, SFT'nin o soruyu doğru cevaplayıp cevaplamadığından). Çalışıyor: karar
rastgeleden 2-3 puan iyi ve prompt ile elde edilen karardan belirgin iyi. Ama eşit
RAG bütçesinde logit kapısının altında kalıyor (%30 bütçede −1,1, %40'ta −0,6) ve
logit kapısıyla birleştirildiğinde ortak arama v3'ü tamamen eliyor — taşıdığı bilgi
logit güveninin kendisi. Adapter duruyor; kararın denetlenebilir bir çıktı olması
gerekirse kullanılabilir, bedeli eşit doğruluk için ~%20 fazla retrieval.

## Ölçüm disiplini

- Yedi sınav gerçek ÖSYM sınavı; kendi hazırladığımız test seti değil.
- Her soru eğitim setinin shingle parmak iziyle karşılaştırıldı; fazla örtüşen iki
  sınav ortalamadan çıkarıldı.
- Kendi 316 soruluk test setimizdeki %88,6 **raporlanmıyor**: o set RAG korpusuyla
  aynı yayınevi ekosisteminden geliyor ve gerçek sınav performansını abartıyor.
- Doğruluk ölçümünde `repetition_penalty` kullanılmıyor; HF bunu prompt tokenlarına
  da uyguluyor ve skoru düşürüyor (%70,9 → %68,4).

## RAG sürümlerinin gelişimi

Kendi 316 soruluk test setimizde, sürümleri birbiriyle kıyaslamak için:

| Sürüm | Doğruluk |
|---|---|
| RAG yok | 70,9% |
| düz RAG (6 chunk, 6000 karakter) | 83,9% |
| + hibrit arama | 85,4% |
| + yeniden sıralama | 86,4% |
| hibrit + yeniden sıralama | 88,9% |
