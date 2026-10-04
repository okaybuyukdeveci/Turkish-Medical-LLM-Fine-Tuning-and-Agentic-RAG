# İnce ayar

Qwen3.5-4B üzerinde LoRA ile iki tur ince ayar. Unsloth + TRL, Colab A100.

## Defterler

| Defter | Ne yapar | Süre |
|---|---|---|
| `notebooks/01_sft_v1.ipynb` | İlk tur: yalnızca TUS verisi | ~1,5 sa |
| `notebooks/02_sft_v2.ipynb` | Zayıf açıklama ayıklama + %6 karma talimat verisi | ~1,5 sa |

Defterlerin hücreleri `hucreler/v1/` ve `hucreler/v2/` altında ayrı dosyalar
olarak da duruyor; defter bunlardan üretildi.

## Hiperparametreler

İki turda da aynı:

```
LoRA        r=16, alpha=32, dropout=0.05, tüm lineer katmanlar
             (q, k, v, o, gate, up, down) — 21,2 M eğitilebilir param (%0,47)
optimizer   adamw_8bit, lr 1e-4, cosine, warmup %3, weight decay 0
batch       8 x 2 gradient accumulation = efektif 16
epoch       2, seed 42, bf16
uzunluk     max_seq_len 2048
loss        yalnızca cevap tokenlarında (prompt maskeli)
çıktı       "Doğru cevap: <harf>) <şık metni>" ilk satırda, sonra açıklama
```

Şık karıştırma (augmentation) **yapılmadı**: açıklamalar harfe atıf yapıyor
("B yanlıştır"), karıştırmak açıklamayı bozar. Cevap harfi dağılımı zaten dengeli
(A–E arası %19–21).

## Sonuçlar

316 soruluk test setinde:

| | TUS doğruluğu | Talimat takibi |
|---|---|---|
| Qwen3.5-4B ham | 62,3% | 311/316 (%98) |
| SFT v1 | 70,9% | bozuk |
| SFT v2 | 71,8% | 309/316 (%98) |

v1 → ham fark: +8,5 puan, McNemar p = 0,003.

## v1'in sorunu: catastrophic forgetting

v1 yalnızca "TUS sorusu çöz" biçiminde veri gördü. Sonuç: başka bir talimat
verildiğinde (örneğin "bu soru için bir arama sorgusu yaz") modelin yine soruyu
cevaplaması. Agentic akışta bu kullanılamaz hale getiriyordu.

v2 eğitim setine %6 oranında arama sorgusu üretme görevi karıştırarak bunu
düzeltti — talimat takibi ham modelin seviyesine döndü.

## v2'nin yan etkisi — okumadan kullanma

**v2, RAG altında v1'den kötü çalışıyor.** Aynı belgeler, aynı prompt, eşli test:

```
hibrit + rerank    v1 88,6%    v2 85,1%
13 soru v1 doğru v2 yanlış, 2 soru tersi, p = 0,007
```

Teşhis (`entegrasyon/sonuclar/OLCUMLER.md` içinde ayrıntılı): v2 bağlamı yok
saymıyor, bağlam tarafından **yanıltılıyor**. Gerilediği sorularda cevabını
RAG'siz halinden değiştirmiş ve yanlışa değiştirmiş.

Bu yüzden üretimde iki adapter rol ayrımıyla kullanılıyor: **cevaplamada v1,
serbest metinden arama sorgusu üretmede v2**. İkisi tek oturumda `set_adapter`
ile değiştirilebiliyor.

## v2 hücrelerindeki terim görevi hakkında

`hucreler/v2/` altında `2b_terim_temizle.py` ve `2c_terim_yeniden.py` var ama
son eğitimde **kullanılmadılar**. Üçüncü bir görev (tıbbi terim tanımı üretme)
denenmiş, sonra `2d_terim_cikar.py` ile çıkarılmıştı.

Neden: tanımları temel modelin kendisi üretiyordu ve hiçbiri doğrulanmamıştı.
Örneklerden biri tıbben yanlıştı — "elementer cisimcik" bir organel olarak tarif
edilmişti; oysa Chlamydia'nın enfeksiyöz formudur. Veri setinden tıbben yanlış
açıklamaları ayıklamak için harcanan emekle doğrudan çelişiyordu.

Hücreler kararın kaydı olarak duruyor. Yeniden eğitimde sırayla çalıştırmak
sorun çıkarmaz; 2d zaten görevi çıkarıyor.

## Ön koşul

Eğitim verisi `MyDrive/tus_data/` altında olmalı — `dataset/README.md`.
