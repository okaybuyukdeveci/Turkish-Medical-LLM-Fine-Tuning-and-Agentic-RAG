# SFT v2 eğitim kaydı

2026-10-03/04, Colab A100-SXM4-40GB.

## Veri

```
eğitim 22.433 | val 457
  TUS sorusu      21.511
  arama sorgusu    1.379  (karma %6)
token p50 354 | p95 777
```

Üçüncü bir görev (terim tanımı üretme) hazırlanmış ama **eğitime alınmamıştı**:
tanımları temel model üretiyordu ve doğrulanmamışlardı. Örneklerden biri tıbben
yanlıştı ("elementer cisimcik" bir organel olarak tarif edilmişti; oysa
Chlamydia'nın enfeksiyöz formudur). Veri setinden tıbben yanlış açıklamaları
ayıklamak için harcanan emekle çelişeceği için görev çıkarıldı.

## Eğitim

2.806 adım, 2 epoch, 0,55-0,57 it/s, ~1 sa 20 dk.

| adım | train loss | val loss |
|---|---|---|
| 200 | 2,363 | 1,587 |
| 400 | 2,304 | 1,535 |
| 600 | 2,209 | 1,492 |
| 800 | 2,280 | 1,462 |
| 1000 | 2,261 | 1,436 |
| 1200 | 2,148 | 1,422 |
| 1400 | 2,124 | 1,406 |
| 1600 | 1,967 | 1,404 |
| 1800 | 2,020 | 1,390 |
| 2000 | 1,921 | 1,393 |
| 2200 | 2,065 | **1,376** |
| 2400 | 2,083 | 1,383 |

Dip 2.200 civarı; `load_best_model_at_end=True` olduğu için kaydedilen adapter o
checkpoint. Train loss'un val loss'tan yüksek görünmesi normal — train loss son
200 adımın ortalaması, val anlık değer.

## Sonuç

```
                  SFT v1    SFT v2
TUS doğruluğu      70,9%     71,8%   (227/316)
talimat takibi     bozuk      98%    (309/316)
```

Ham modelin talimat takibi 311/316 (%98) — v2 taban seviyeyi korudu.

TUS doğruluğundaki +0,9 puan 316 soruda ~3 soru; tek başına anlamlı değil.
Önemli olan düşmemiş olması.

**Devamı için:** v2'nin RAG altındaki gerilemesi ve iki adapter'ın rol ayrımı
`entegrasyon/sonuclar/OLCUMLER.md` içinde.
